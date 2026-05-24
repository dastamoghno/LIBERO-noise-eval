# VLA-Adapter rollout files (modified)

Drop-in replacements for the corresponding files in
[OpenHelix-Team/VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter).

Apply with:

```bash
# from the LIBERO-noise-eval repo root, with VLA-Adapter cloned at ../VLA-Adapter
cp -r vla_adapter/experiments/ /path/to/VLA-Adapter/
```

## Contents

| File | Status | Purpose |
|---|---|---|
| `experiments/robot/libero/run_libero_eval.py` | modified | Adds `--obj_force_*`, `--obs_noise*`, `--action_noise_std`, `--noise_type`, `--task_id`, `--log_grasp_steps` flags. Fixes a missing `import time`. |
| `experiments/robot/libero/libero_utils.py` | modified | 1-line edit. |
| `experiments/robot/openvla_utils.py` | modified | Auto-config-map handling cleanup. |
| `experiments/robot/libero/env_perturbations.py` | new | MuJoCo external-force perturbation + closed-form `F(target_displacement, mass, mu, duration)` model. |
| `experiments/robot/libero/action_noise.py` | new | 7-dim action-vector noise (gaussian, uniform, salt-pepper, impulse, constant). |
| `experiments/robot/libero/visual_noise.py` | new | Camera-image noise (gaussian, salt-pepper, blur, shift, rotation, colour jitter). Sourced from RobustVLA. |
| `experiments/robot/libero/benchmark_latency.py` | new | Forward-pass latency benchmark. |
| `experiments/robot/libero/phase_timing.py` | new | Z-height-based detector of grasp/place phase boundaries. |
| `experiments/robot/libero/validate_force.py` | new | Validates the closed-form force model against measured MuJoCo displacement. |

## License

Original VLA-Adapter code is MIT-licensed by Yihao Wang et al. (2025); see
`LICENSE` in this directory. New files added in this repo are also MIT-licensed.
