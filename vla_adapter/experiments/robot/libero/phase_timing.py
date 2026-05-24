"""Phase timing for LIBERO-10 task 6.

Phases (from the task description "put the white mug on the plate and put
the chocolate pudding to the right of the plate"):
  P0  approach mug                  : start                -> t_mug_lift
  P1  carry mug to plate, place it  : t_mug_lift           -> t_mug_drop
  P2  return to pudding (transit)   : t_mug_drop           -> t_pudding_lift_or_approach
  P3  grasp + place pudding         : t_pudding_lift       -> t_pudding_drop / end

Detection heuristics (z-position based, robust):
  t_mug_lift     : first step where porcelain_mug_1 z > z0_mug + LIFT_EPS
  t_mug_drop     : first step after t_mug_lift where mug returns to z < z0_mug + LIFT_EPS
                   AND has stayed there for HOLD_STEPS consecutive steps
  t_pudding_lift : first step where chocolate_pudding_1 z > z0_pud + LIFT_EPS
  t_pudding_drop : first step after t_pudding_lift where pudding back to z < z0_pud + LIFT_EPS
                   (held for HOLD_STEPS); if never drops, use episode end
"""
import sys
import json
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union, List

import draccus
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    get_libero_wrist_image, quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head, get_proprio_projector, get_processor, resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action, get_image_resize_size, get_model, set_seed_everywhere,
)


LIFT_EPS = 0.03      # metres above resting z to count as "lifted"
HOLD_STEPS = 3       # consecutive low-z steps to count as "placed"


@dataclass
class Cfg:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, list] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_10"
    task_id: int = 6
    num_steps_wait: int = 10
    env_img_res: int = 256
    save_version: str = "vla-adapter"
    use_pro_version: bool = False
    phase: str = "Inference"
    seed: int = 7
    num_episodes: int = 10
    max_steps: int = 520


def _prep(obs, resize_size):
    img = resize_image_for_policy(get_libero_image(obs), resize_size)
    wrist = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)
    return {
        "full_image": img, "wrist_image": wrist,
        "state": np.concatenate((obs["robot0_eef_pos"],
                                 quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"])),
    }


def detect_phases(mug_z, pud_z):
    z0_mug = mug_z[0]
    z0_pud = pud_z[0]
    n = len(mug_z)

    def first_lift(z, z0):
        for i, v in enumerate(z):
            if v > z0 + LIFT_EPS:
                return i
        return None

    def first_drop_after(z, z0, start):
        if start is None:
            return None
        low_streak = 0
        for i in range(start, len(z)):
            if z[i] < z0 + LIFT_EPS:
                low_streak += 1
                if low_streak >= HOLD_STEPS:
                    return i - HOLD_STEPS + 1
            else:
                low_streak = 0
        return None

    t_mug_lift = first_lift(mug_z, z0_mug)
    t_mug_drop = first_drop_after(mug_z, z0_mug, t_mug_lift)
    t_pud_lift = first_lift(pud_z, z0_pud)
    t_pud_drop = first_drop_after(pud_z, z0_pud, t_pud_lift)
    return t_mug_lift, t_mug_drop, t_pud_lift, t_pud_drop


@draccus.wrap()
def main(cfg: Cfg):
    set_seed_everywhere(cfg.seed)
    print(f"Loading model ...")
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    processor = get_processor(cfg)
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    cfg.unnorm_key = unnorm_key
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    init_states = suite.get_task_init_states(cfg.task_id)
    env, desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    print(f"Task {cfg.task_id}: {desc}\n")

    rows = []
    for ep in range(cfg.num_episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        mug_z, pud_z = [], []
        q = deque(maxlen=cfg.num_open_loop_steps)
        success = False
        for t in range(cfg.max_steps):
            mug_z.append(float(obs["porcelain_mug_1_pos"][2]))
            pud_z.append(float(obs["chocolate_pudding_1_pos"][2]))
            observation = _prep(obs, resize_size)
            if not q:
                actions = get_action(cfg, model, observation, desc,
                                     processor=processor, action_head=action_head,
                                     proprio_projector=proprio_projector,
                                     noisy_action_projector=None,
                                     use_film=cfg.use_film, use_minivlm=cfg.use_minivlm)
                q.extend(actions)
            from experiments.robot.libero.run_libero_eval import process_action
            action = process_action(q.popleft(), cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break

        tml, tmd, tpl, tpd = detect_phases(mug_z, pud_z)
        end = len(mug_z)
        row = dict(ep=ep, success=success, end=end,
                   t_mug_lift=tml, t_mug_drop=tmd,
                   t_pud_lift=tpl, t_pud_drop=tpd)
        rows.append(row)
        print(f"ep={ep:2d} success={success} end={end:4d}  "
              f"mug_lift={tml}  mug_drop={tmd}  pud_lift={tpl}  pud_drop={tpd}")

    env.close()

    # Summarize
    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (int(np.median(vals)), len(vals)) if vals else (None, 0)

    tml_m, n1 = med("t_mug_lift")
    tmd_m, n2 = med("t_mug_drop")
    tpl_m, n3 = med("t_pud_lift")
    tpd_m, n4 = med("t_pud_drop")
    end_m, _ = med("end")

    print("\n===== Median phase boundaries (control steps, 50 ms each) =====")
    print(f"P0 (approach mug)         : 0    -> {tml_m}     [n={n1}/{len(rows)}]")
    print(f"P1 (carry/place mug)      : {tml_m}  -> {tmd_m}     [n={n2}]")
    print(f"P2 (return to pudding)    : {tmd_m}  -> {tpl_m}     [n={n3}]")
    print(f"P3 (carry/place pudding)  : {tpl_m}  -> {tpd_m or end_m}  [n={n4}]")
    print(f"\nEpisode length (median)   : {end_m} steps  (~{end_m*0.05:.1f}s)")
    print("\nIn seconds (×0.05 s):")
    for name, v in [("P0->P1", tml_m), ("P1->P2", tmd_m),
                    ("P2->P3", tpl_m), ("P3 end", tpd_m or end_m)]:
        print(f"  {name}: {v*0.05:.2f}s" if v else f"  {name}: n/a")

    with open("phase_timing_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n(per-episode rows saved to phase_timing_rows.json)")


if __name__ == "__main__":
    main()
