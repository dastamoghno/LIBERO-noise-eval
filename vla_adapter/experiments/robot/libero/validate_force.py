"""Validate force->displacement model on a LIBERO object.

Reads mass + friction from MuJoCo, predicts displacement for a given F,
then actually applies F horizontally and measures the resulting displacement.
"""
import sys
import numpy as np
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
from experiments.robot.libero.env_perturbations import apply_force_to_object, clear_force_on_object


G = 9.81


def object_properties(env, body_name):
    sim = env.sim
    bid = sim.model.body_name2id(body_name)
    mass = float(sim.model.body_mass[bid])
    geom_ids = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    if geom_ids:
        mu = float(np.mean([sim.model.geom_friction[g][0] for g in geom_ids]))
    else:
        mu = 0.5
    return mass, mu, bid, geom_ids


def force_for_displacement(mass, mu, dx, duration_steps, dt):
    """Exact force for target displacement given push duration.

    Solves Δx = (μg t²/2)·k(1+k) with k = (F/m - μg)/(μg) for k, then F.
    Includes both the slide-during-push and the coast-to-stop phases.
    """
    t = duration_steps * dt
    if t <= 0 or dx <= 0:
        return 0.0
    C = 2.0 * dx / (mu * G * t * t)
    return mass * mu * G * (1.0 + np.sqrt(1.0 + 4.0 * C)) / 2.0


def force_for_displacement_impulse(mass, mu, dx, duration_steps, dt):
    """Short-push approximation: F = (m/t)·sqrt(2μg·Δx)."""
    t = duration_steps * dt
    return (mass / t) * np.sqrt(2.0 * mu * G * dx)


def predicted_displacement(F, mass, mu, duration_steps, dt):
    """Full model: slide-during-push + coast-to-stop."""
    t = duration_steps * dt
    a = F / mass - mu * G
    if a <= 0:
        return 0.0  # static friction not overcome
    v_end = a * t
    dx_push = 0.5 * a * t * t
    dx_coast = v_end * v_end / (2.0 * mu * G)
    return dx_push + dx_coast


def measure_one(env, body_name, F, duration_steps, settle_steps=30, post_steps=60):
    sim = env.sim
    bid = sim.model.body_name2id(body_name)
    # let scene settle
    for _ in range(settle_steps):
        env.step(get_libero_dummy_action("openvla"))
    pos0 = sim.data.body_xpos[bid].copy()
    # apply force in +X horizontally
    force_vec = np.array([F, 0.0, 0.0])
    for _ in range(duration_steps):
        apply_force_to_object(env, body_name, force_vec)
        env.step(get_libero_dummy_action("openvla"))
    clear_force_on_object(env, body_name)
    # coast
    for _ in range(post_steps):
        env.step(get_libero_dummy_action("openvla"))
    pos1 = sim.data.body_xpos[bid].copy()
    return pos0, pos1, np.linalg.norm(pos1[:2] - pos0[:2])


def main():
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(6)
    body_name = "chocolate_pudding_1_main"

    env, desc = get_libero_env(task, "openvla", resolution=128)
    print(f"Task: {desc}")
    env.reset()
    init_states = suite.get_task_init_states(6)
    env.set_init_state(init_states[0])

    mass, mu, bid, geom_ids = object_properties(env, body_name)
    dt = float(env.sim.model.opt.timestep)
    # LIBERO uses control_freq; effective control dt = sim_dt * n_substeps
    try:
        n_sub = int(env.env.control_timestep / env.sim.model.opt.timestep)
    except Exception:
        n_sub = 1
    ctrl_dt = dt * n_sub

    print("\n=== Object properties ===")
    print(f"body            : {body_name}")
    print(f"body id         : {bid}, #geoms={len(geom_ids)}")
    print(f"mass            : {mass:.4f} kg")
    print(f"sliding mu      : {mu:.3f}")
    print(f"sim dt          : {dt*1000:.2f} ms")
    print(f"control dt      : {ctrl_dt*1000:.2f} ms (n_substeps={n_sub})")

    print("\n=== Required F as a joint function of (Δx, duration) ===")
    durations = [1, 2, 5, 10]
    print(f"{'Δx (cm)':>9}  " + "  ".join(f"d={d:>2}({d*ctrl_dt*1000:>5.0f}ms)" for d in durations))
    for dx in (0.005, 0.01, 0.02, 0.05):
        row = f"{dx*100:9.2f}  " + "  ".join(
            f"{force_for_displacement(mass, mu, dx, d, ctrl_dt):14.4f}" for d in durations
        )
        print(row)
    print("(values in N — exact, includes duration)")

    print("\n=== Validation: predict vs measure (varying both F and duration) ===")
    print(f"{'F (N)':>6}  {'dur':>4}  {'t_push(ms)':>10}  "
          f"{'pred dx (cm)':>13}  {'meas dx (cm)':>13}  {'ratio':>6}")
    cases = [
        (0.10, 1), (0.10, 2), (0.10, 5), (0.10, 10),
        (0.15, 2), (0.15, 5),
        (0.20, 1), (0.20, 2), (0.20, 5),
        (0.05, 5), (0.05, 10),
    ]
    for F, dur in cases:
        env.reset()
        env.set_init_state(init_states[0])
        pos0, pos1, dx_meas = measure_one(env, body_name, F, dur)
        dx_pred = predicted_displacement(F, mass, mu, dur, ctrl_dt)
        ratio = (dx_meas / dx_pred) if dx_pred > 0 else float("inf")
        print(f"{F:6.3f}  {dur:4d}  {dur*ctrl_dt*1000:10.1f}  "
              f"{dx_pred*100:13.3f}  {dx_meas*100:13.3f}  {ratio:6.2f}")

    print("\n=== Round-trip: pick Δx + duration, compute F, measure ===")
    print(f"{'tgt Δx(cm)':>11}  {'dur':>4}  {'F(N)':>7}  {'meas Δx(cm)':>12}  {'err(%)':>7}")
    for dx_tgt, dur in [(0.5, 5), (1.0, 5), (2.0, 5), (1.0, 10), (2.0, 10)]:
        F = force_for_displacement(mass, mu, dx_tgt/100.0, dur, ctrl_dt)
        env.reset()
        env.set_init_state(init_states[0])
        _, _, dx_meas = measure_one(env, body_name, F, dur)
        err = 100.0 * (dx_meas - dx_tgt/100.0) / (dx_tgt/100.0)
        print(f"{dx_tgt:11.2f}  {dur:4d}  {F:7.4f}  {dx_meas*100:12.3f}  {err:7.1f}")

    env.close()


if __name__ == "__main__":
    main()
