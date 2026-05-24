"""Measure forward-pass latency of VLA-Adapter on a single LIBERO task.

Usage:
    cd VLA-Adapter
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 PYTHONPATH=../LIBERO:. \
    python experiments/robot/libero/benchmark_latency.py \
        --pretrained_checkpoint pretrained_models/LIBERO-Long \
        --task_suite_name libero_10 --task_id 6 \
        --num_warmup 5 --num_iters 50
"""
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_proprio_projector,
    get_processor,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)


@dataclass
class BenchConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # Env
    task_suite_name: str = "libero_10"
    task_id: int = 6
    num_steps_wait: int = 10
    env_img_res: int = 256
    # Misc
    save_version: str = "vla-adapter"
    use_pro_version: bool = False
    phase: str = "Inference"
    seed: int = 7
    # Bench
    num_warmup: int = 5
    num_iters: int = 50

    # noise flags read by libero_utils/robot_utils (unused here)
    obs_noise: float = 0.0
    obs_noise_type: str = "gaussian"
    obs_salt_pepper_probability: float = 0.1
    obs_blur_kernel_size: int = 5
    obs_blur_sigma: float = 1.0
    obs_image_shift_ratio: float = 0.1
    obs_image_rotation_angle: float = 30.0
    obs_enhanced_color_jitter_factor: float = 3.0


def _prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_r = resize_image_for_policy(img, resize_size)
    wrist_r = resize_image_for_policy(wrist_img, resize_size)
    return {
        "full_image": img_r,
        "wrist_image": wrist_r,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


@draccus.wrap()
def main(cfg: BenchConfig) -> None:
    set_seed_everywhere(cfg.seed)

    print(f"Loading model from {cfg.pretrained_checkpoint} ...")
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    # set unnorm_key
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    cfg.unnorm_key = unnorm_key

    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)

    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    print(f"Task {cfg.task_id}: {task_description}")

    env.reset()
    obs = env.set_init_state(initial_states[0])
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    observation = _prepare_observation(obs, resize_size)

    def _call():
        return get_action(
            cfg, model, observation, task_description,
            processor=processor, action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=None,
            use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
        )

    print(f"Warmup: {cfg.num_warmup} iters ...")
    for _ in range(cfg.num_warmup):
        _ = _call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Timing: {cfg.num_iters} iters ...")
    lat_ms = []
    for _ in range(cfg.num_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = _call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(lat_ms)
    n_chunk = max(1, cfg.num_open_loop_steps)
    print("\n===== Forward-pass latency =====")
    print(f"task_suite     : {cfg.task_suite_name}")
    print(f"task_id        : {cfg.task_id}  ({task_description})")
    print(f"checkpoint     : {cfg.pretrained_checkpoint}")
    print(f"num_open_loop  : {cfg.num_open_loop_steps}  (actions per forward pass)")
    print(f"iters          : {cfg.num_iters}  (warmup {cfg.num_warmup})")
    print(f"device         : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"mean   : {lat.mean():8.2f} ms   ({lat.mean()/n_chunk:7.2f} ms/action)")
    print(f"median : {np.median(lat):8.2f} ms")
    print(f"std    : {lat.std():8.2f} ms")
    print(f"min    : {lat.min():8.2f} ms")
    print(f"max    : {lat.max():8.2f} ms")
    print(f"p90    : {np.percentile(lat, 90):8.2f} ms")
    print(f"p99    : {np.percentile(lat, 99):8.2f} ms")
    print(f"throughput: {1000.0/lat.mean():.2f} fwd/s  |  {1000.0*n_chunk/lat.mean():.2f} actions/s")

    env.close()


if __name__ == "__main__":
    main()
