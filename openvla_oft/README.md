# OpenVLA-OFT rollout files (modified)

Drop-in replacements for the corresponding files in
[moojink/openvla-oft](https://github.com/moojink/openvla-oft).

Apply with:

```bash
cp -r openvla_oft/experiments/ /path/to/openvla-oft/
```

## Contents

| File | Status | Purpose |
|---|---|---|
| `experiments/robot/libero/run_libero_eval.py` | modified | Ports the `--obj_force_*` and `--task_id` flags from VLA-Adapter so the same perturbation grid can be run. |
| `experiments/robot/libero/env_perturbations.py` | new | Identical copy of the VLA-Adapter module — closed-form force model + MuJoCo body push/clear helpers. |

## License

Original OpenVLA-OFT code is MIT-licensed by Moo Jin Kim, Chelsea Finn, Percy Liang
(2025); see `LICENSE` in this directory. New files added in this repo are also
MIT-licensed.
