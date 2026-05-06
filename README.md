# Perturbation Evaluation for VLA-Adapter on LIBERO

This document describes how to add noise and physical perturbations to rollout evaluation in the VLA-Adapter + LIBERO pipeline. Three independent perturbation mechanisms are implemented, each controllable via CLI flags.

---

## Overview

| Perturbation type | Module | What it corrupts |
|---|---|---|
| Action noise | `action_noise.py` | Policy output actions before execution |
| Observation noise | `visual_noise.py` | Camera images before feeding to policy |
| Object force | `env_perturbations.py` | MuJoCo physics — pushes objects mid-rollout |

All three can be combined in a single run.

---

## Setup

### Prerequisites

Clone and install VLA-Adapter and LIBERO side-by-side:

```bash
git clone https://github.com/your-org/VLA-Adapter
cd VLA-Adapter
pip install -e .
cd ..

# Clone this repository side-by-side with VLA-Adapter
git clone https://github.com/dastamoghno/LIBERO-noise-eval LIBERO
cd LIBERO && pip install -e . && cd ..
```

Install OpenCV (required for observation noise):

```bash
pip install opencv-python
```

Download the LIBERO-Long checkpoint from HuggingFace:

```bash
mkdir -p pretrained_models
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='VLA-Adapter/LIBERO-Long', local_dir='pretrained_models/LIBERO-Long')
"
```

### Base evaluation command

All examples below extend this base command:

```bash
cd /path/to/VLA-Adapter

MUJOCO_GL=egl EGL_DEVICE_ID=0 \
PYTHONPATH=../LIBERO:. \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint pretrained_models/LIBERO-Long \
    --task_suite_name libero_10 \
    --num_trials_per_task 20 \
    --num_open_loop_steps 8 \
    --use_l1_regression True \
    --use_proprio True \
    --use_pro_version False \
    --num_images_in_input 2 \
    --use_wandb False \
    [PERTURBATION FLAGS]
```

To evaluate a single task (0-indexed), add `--task_id N`. Without it, all tasks in the suite are evaluated.

---

## 1. Action Noise

**File:** `experiments/robot/libero/action_noise.py`

Noise is added to the 7-dimensional action vector **after** the policy produces it and **before** it is sent to the environment. The action is clipped to `[-1, 1]` after noise is applied.

### Noise types

| `--noise_type` | Description | Magnitude meaning |
|---|---|---|
| `gaussian` | N(0, σ) per action dimension | Standard deviation σ |
| `uniform` | Uniform in [-m, +m] | Half-width m |
| `constant` | Fixed offset added to all dims | Offset value |
| `salt_pepper` | Random dims snapped to ±magnitude | Snap value; `--salt_pepper_probability` controls fraction |
| `impulse` | Rare high-amplitude spikes | Spike amplitude; `--impulse_probability` controls rate |

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--action_noise_std` | `0.0` | Noise magnitude (0 = disabled) |
| `--noise_type` | `gaussian` | Noise distribution |

### Examples

```bash
# Gaussian action noise, σ=0.1
--action_noise_std 0.1 --noise_type gaussian

# Gaussian action noise, σ=0.2
--action_noise_std 0.2 --noise_type gaussian

# Uniform action noise, half-width 0.15
--action_noise_std 0.15 --noise_type uniform

# Salt & pepper — 10% of action dims flipped to ±1.0
--action_noise_std 1.0 --noise_type salt_pepper

# Impulse — 5% chance per step of a large spike
--action_noise_std 2.0 --noise_type impulse
```

### Measured results on LIBERO-10 (Gaussian, all 10 tasks, 20 episodes each)

| σ | Overall success rate |
|---|---|
| 0 (baseline) | 91.5% |
| 0.1 | 79.0% |
| 0.2 | 36.0% |
| 0.3 | 7.0% |

Per-task breakdown:

| Task | Description | σ=0 | σ=0.1 | σ=0.2 | σ=0.3 |
|------|-------------|-----|-------|-------|-------|
| 0 | put alphabet soup + tomato sauce in basket | 95% | 75% | 35% | 0% |
| 1 | put cream cheese + butter in basket | 100% | 90% | 40% | 5% |
| 2 | turn on stove + put moka pot on it | 100% | 95% | 45% | 5% |
| 3 | put black bowl in bottom drawer + close | 100% | 95% | 55% | 25% |
| 4 | put white mug on left plate + yellow/white mug on right | 95% | 75% | 10% | 10% |
| 5 | pick up book + place in back compartment of caddy | 100% | 90% | 60% | 10% |
| 6 | put white mug on plate + chocolate pudding to right | 70% | 60% | 15% | 0% |
| 7 | put alphabet soup + cream cheese in basket | 95% | 90% | 40% | 10% |
| 8 | put both moka pots on stove | 70% | 50% | 20% | 0% |
| 9 | put yellow/white mug in microwave + close | 90% | 70% | 40% | 5% |

---

## 2. Observation Noise

**File:** `experiments/robot/libero/visual_noise.py`

Sourced from [RobustVLA](https://github.com/gakakulicc/RobustVLA). Noise is applied to both the front-facing and wrist camera images **after** resizing, **before** the images are passed to the policy. Operates on `uint8` images in `[0, 255]` pixel space using NumPy and OpenCV.

For perturbations with spatial structure (shift, rotation, color jitter, blur), parameters are **sampled once per episode** and held fixed for all timesteps in that episode — so the perturbation is consistent within an episode rather than flickering every step.

### Noise types

| `--obs_noise_type` | Description | Key parameter |
|---|---|---|
| `gaussian` | Additive pixel noise N(0, σ) | `--obs_noise` = σ in pixel units [0–255] |
| `salt_pepper` | Random pixels set to 0 or 255 | `--obs_salt_pepper_probability` = fraction corrupted |
| `blur` | Gaussian blur | `--obs_blur_kernel_size`, `--obs_blur_sigma` |
| `image_shift` | Translate image toward upper-left | `--obs_image_shift_ratio` = max shift as fraction of image size |
| `image_rotation` | Rotate image counterclockwise | `--obs_image_rotation_angle` = max angle in degrees |
| `enhanced_color_jitter` | Brightness, contrast, saturation, sharpness | `--obs_enhanced_color_jitter_factor` = max factor |

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--obs_noise` | `0.0` | Noise magnitude (0 = disabled) |
| `--obs_noise_type` | `gaussian` | Noise type |
| `--obs_salt_pepper_probability` | `0.1` | Fraction of pixels corrupted (salt & pepper only) |
| `--obs_blur_kernel_size` | `5` | Blur kernel size, must be odd |
| `--obs_blur_sigma` | `1.0` | Gaussian blur sigma |
| `--obs_image_shift_ratio` | `0.1` | Max shift as fraction of image width/height |
| `--obs_image_rotation_angle` | `30.0` | Max rotation in degrees |
| `--obs_enhanced_color_jitter_factor` | `3.0` | Max jitter factor |

### Examples

```bash
# Gaussian pixel noise, σ=25 (moderate)
--obs_noise 25 --obs_noise_type gaussian

# Gaussian pixel noise, σ=70 (aggressive)
--obs_noise 70 --obs_noise_type gaussian

# Salt & pepper — 10% of pixels corrupted
--obs_noise 1 --obs_noise_type salt_pepper --obs_salt_pepper_probability 0.1

# Gaussian blur
--obs_noise 1 --obs_noise_type blur --obs_blur_kernel_size 7 --obs_blur_sigma 2.0

# Image shift — up to 10% of image size, fixed per episode
--obs_noise 1 --obs_noise_type image_shift --obs_image_shift_ratio 0.1

# Image rotation — up to 30°, fixed per episode
--obs_noise 1 --obs_noise_type image_rotation --obs_image_rotation_angle 30.0

# Color jitter
--obs_noise 1 --obs_noise_type enhanced_color_jitter --obs_enhanced_color_jitter_factor 3.0
```

### Measured results on LIBERO-10 (Gaussian, all 10 tasks, 20 episodes each)

| obs_noise (σ) | Overall success rate |
|---|---|
| 0 (baseline) | 91.5% |
| 70 | 4.0% |

Per-task breakdown at obs_noise=70:

| Task | Description | σ=0 | obs σ=70 |
|---|---|---|---|
| 0 | put alphabet soup + tomato sauce in basket | 95% | 0% |
| 1 | put cream cheese + butter in basket | 100% | 0% |
| 2 | turn on stove + put moka pot on it | 100% | 0% |
| 3 | put black bowl in bottom drawer + close | 100% | 40% |
| 4 | put white mug on left plate + yellow/white mug on right | 95% | 0% |
| 5 | pick up book + place in back compartment of caddy | 100% | 0% |
| 6 | put white mug on plate + chocolate pudding to right | 70% | 0% |
| 7 | put alphabet soup + cream cheese in basket | 95% | 0% |
| 8 | put both moka pots on stove | 70% | 0% |
| 9 | put yellow/white mug in microwave + close | 90% | 0% |

---

## 3. Object Force Perturbation

**File:** `experiments/robot/libero/env_perturbations.py`

Applies a random-direction external force directly to a named MuJoCo body in the scene (e.g. the object being manipulated). Force is applied via `sim.data.xfrc_applied` and takes effect at the next `env.step()`. Units are **Newtons** (MuJoCo standard SI).

The force is triggered stochastically — each step has a configurable probability of starting a new force impulse, which is then sustained for a fixed number of steps. The direction is sampled uniformly on the unit sphere each time a new impulse fires. Forces are always cleared at episode end.

### Finding body names

Before running, identify the MuJoCo body names for objects in your task:

```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 \
PYTHONPATH=../LIBERO:. \
python -c "
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.env_perturbations import list_body_names

suite = benchmark.get_benchmark_dict()['libero_10']()
task = suite.get_task(6)  # change task index as needed
env, _ = get_libero_env(task, 'openvla', resolution=256)
env.reset()
print([b for b in list_body_names(env) if b])
env.close()
"
```

Example output for LIBERO-10 task 6:
```
['world', 'floor', 'living_room_table', 'robot0_base', ...,
 'porcelain_mug_1_main', 'red_coffee_mug_1_main', 'plate_1_main', 'chocolate_pudding_1_main']
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--obj_force_magnitude` | `0.0` | Force in Newtons (0 = disabled) |
| `--obj_force_body` | `""` | MuJoCo body name to push (empty = disabled) |
| `--obj_force_probability` | `0.05` | Per-step probability of triggering a new impulse |
| `--obj_force_duration` | `5` | Steps to sustain each impulse |

### Guidance on force magnitude

LIBERO tabletop objects are small (≈0.1–0.2 kg). Recommended values:

| Force (N) | Effect |
|---|---|
| 0.05–0.1 | Subtle nudge, realistic for a light tap |
| 0.5 | Strong push, object noticeably displaced |
| 5.0+ | Unrealistically aggressive, immediate task failure |

### Examples

```bash
# Light push on mug (task 6), 1% chance per step, sustained 5 steps
--task_id 6 \
--obj_force_magnitude 0.1 \
--obj_force_body porcelain_mug_1_main \
--obj_force_probability 0.01 \
--obj_force_duration 5

# Push on chocolate pudding (second sub-goal)
--task_id 6 \
--obj_force_magnitude 0.1 \
--obj_force_body chocolate_pudding_1_main \
--obj_force_probability 0.01 \
--obj_force_duration 5
```

### Measured results on LIBERO-10 task 6 (baseline: 70%)

| Target object | Force (N) | Probability | Success rate |
|---|---|---|---|
| — | — | — | 70% |
| white mug | 0.1 | 1% | 65% |
| white mug | 0.5 | 1% | 30% |
| white mug | 0.5 | 5% | 0% |
| white mug | 5.0 | 5% | 0% |
| chocolate pudding | 0.1 | 1% | 35% |

Pushing the second-goal object (chocolate pudding) is more disruptive than pushing the primary manipulation target (mug), because the pudding's final resting position determines task success and the policy cannot recover from a displaced target.

---

## Combining perturbations

All three perturbation types can be used simultaneously:

```bash
MUJOCO_GL=egl EGL_DEVICE_ID=0 \
PYTHONPATH=../LIBERO:. \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint pretrained_models/LIBERO-Long \
    --task_suite_name libero_10 \
    --num_trials_per_task 20 \
    --num_open_loop_steps 8 \
    --use_l1_regression True \
    --use_proprio True \
    --use_pro_version False \
    --num_images_in_input 2 \
    --use_wandb False \
    --task_id 6 \
    --action_noise_std 0.1 --noise_type gaussian \
    --obs_noise 25 --obs_noise_type gaussian \
    --obj_force_magnitude 0.1 --obj_force_body chocolate_pudding_1_main \
    --obj_force_probability 0.01 --obj_force_duration 5
```

---

## Running in a persistent session (tmux)

For long all-task evaluations that should survive SSH disconnects:

```bash
tmux new-session -d -s eval

tmux send-keys -t eval "
MUJOCO_GL=egl EGL_DEVICE_ID=0 \
PYTHONPATH=../LIBERO:. \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint pretrained_models/LIBERO-Long \
    --task_suite_name libero_10 \
    --num_trials_per_task 20 \
    --num_open_loop_steps 8 \
    --use_l1_regression True \
    --use_proprio True \
    --use_pro_version False \
    --num_images_in_input 2 \
    --use_wandb False \
    --action_noise_std 0.2 --noise_type gaussian \
    2>&1 | tee libero10_action_noise_0.2.log
" Enter

# Reattach later
tmux attach -t eval

# Check progress without attaching
tail -f libero10_action_noise_0.2.log
```

---

## File reference

```
VLA-Adapter/experiments/robot/libero/
├── action_noise.py          # Action perturbation module
├── visual_noise.py          # Observation image perturbation module (from RobustVLA)
├── env_perturbations.py     # MuJoCo object force perturbation module
└── run_libero_eval.py       # Main eval script (all perturbations wired in)
```
