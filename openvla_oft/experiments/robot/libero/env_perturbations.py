import numpy as np


def apply_force_to_object(env, object_name: str, force: np.ndarray, torque: np.ndarray = None):
    """Apply an external force (and optional torque) to a named MuJoCo body."""
    body_id = env.sim.model.body_name2id(object_name)
    env.sim.data.xfrc_applied[body_id, :3] = force
    env.sim.data.xfrc_applied[body_id, 3:] = torque if torque is not None else [0.0, 0.0, 0.0]


def clear_force_on_object(env, object_name: str):
    """Zero out any applied force/torque on a named MuJoCo body."""
    body_id = env.sim.model.body_name2id(object_name)
    env.sim.data.xfrc_applied[body_id, :] = 0.0


def sample_random_force(magnitude: float, horizontal_only: bool = False, tilt_up: float = 0.0) -> np.ndarray:
    """Sample a random unit-direction force vector scaled by magnitude.

    Args:
        magnitude: Force magnitude in Newtons.
        horizontal_only: If True, restrict lateral force to the XY plane.
        tilt_up: Fixed upward Z bias added before normalising. Reduces the
                 object's normal force against the table, lowering friction so
                 the lateral component produces visible displacement. A value
                 of ~0.3 gives a noticeable upward tilt while keeping the
                 force mostly lateral.
    """
    if horizontal_only:
        direction = np.array([np.random.randn(), np.random.randn(), tilt_up])
    else:
        direction = np.random.randn(3)
        direction[2] += tilt_up
    direction /= np.linalg.norm(direction)
    return direction * magnitude


def list_body_names(env):
    """Return all body names in the current scene (useful for finding object names)."""
    return [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]


_G = 9.81


def get_body_mass_and_friction(env, body_name: str):
    """Read mass (kg) and average sliding-friction coefficient for a body."""
    sim = env.sim
    bid = sim.model.body_name2id(body_name)
    mass = float(sim.model.body_mass[bid])
    geom_ids = [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]
    mu = float(np.mean([sim.model.geom_friction[g][0] for g in geom_ids])) if geom_ids else 0.5
    return mass, mu


def get_control_dt(env) -> float:
    """Return the policy-level timestep (sim_dt × n_substeps)."""
    sim_dt = float(env.sim.model.opt.timestep)
    try:
        n_sub = max(1, int(round(env.env.control_timestep / sim_dt)))
    except Exception:
        n_sub = 1
    return sim_dt * n_sub


def force_for_displacement(mass: float, mu: float, dx: float, duration_steps: int, dt: float) -> float:
    """Closed-form force needed to displace a body by `dx` metres over `duration_steps` steps.

    Models slide-during-push + coast-to-stop under kinetic friction.
    Returns 0 if dx<=0 or duration<=0.
    """
    t = duration_steps * dt
    if t <= 0 or dx <= 0:
        return 0.0
    C = 2.0 * dx / (mu * _G * t * t)
    return mass * mu * _G * (1.0 + np.sqrt(1.0 + 4.0 * C)) / 2.0
