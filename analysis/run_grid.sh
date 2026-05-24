#!/usr/bin/env bash
# Factorial perturbation grid for OFT vs VLA-Adapter robustness analysis.
# 20 new runs (10 per model): baseline, phaseA x{1,3,5,8}, phaseB x{8}, phaseC x{1,3,5,8}.
# Phase B x{1,3,5} already exist and are reused by the parser.
#
# Usage:  bash run_grid.sh adapter   |   bash run_grid.sh oft   |   bash run_grid.sh all
set -u

ADAPTER_DIR=/home/tamoghno/VLA-Adapter
OFT_DIR=/home/tamoghno/openvla-oft
ADAPTER_PY=/home/tamoghno/miniconda3/envs/vla-adapter/bin/python
OFT_PY=/home/tamoghno/miniconda3/envs/openvla-oft/bin/python

# phase -> "MIN MAX"
declare -A PHASE_WIN=( [A]="0 60" [B]="60 200" [C]="200 240" )

run_adapter() {
  cd "$ADAPTER_DIR" || exit 1
  local common="--pretrained_checkpoint pretrained_models/LIBERO-Long \
    --task_suite_name libero_10 --task_id 6 --num_trials_per_task 50 \
    --num_open_loop_steps 8 --use_l1_regression True --use_proprio True \
    --use_pro_version False --num_images_in_input 2 --use_wandb False"
  # baseline
  echo "=== adapter baseline ==="
  PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
    PYTHONPATH=$ADAPTER_DIR:/home/tamoghno/LIBERO $ADAPTER_PY \
    experiments/robot/libero/run_libero_eval.py $common \
    --run_id_note pert_adapter_baseline 2>&1 | tail -3
  # perturbed cells
  for cell in "A 1" "A 3" "A 5" "A 8" "B 8" "C 1" "C 3" "C 5" "C 8"; do
    set -- $cell; ph=$1; dx=$2
    read lo hi <<< "${PHASE_WIN[$ph]}"
    echo "=== adapter phase$ph dx${dx}cm  win[$lo,$hi) ==="
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
      PYTHONPATH=$ADAPTER_DIR:/home/tamoghno/LIBERO $ADAPTER_PY \
      experiments/robot/libero/run_libero_eval.py $common \
      --obj_force_body chocolate_pudding_1_main --obj_force_duration 5 \
      --obj_force_horizontal True --obj_force_tilt_up 0.1 --obj_force_num_triggers 1 \
      --obj_force_target_displacement_m 0.0${dx} \
      --obj_force_min_trigger_step $lo --obj_force_max_trigger_step $hi \
      --run_id_note pert_adapter_phase${ph}_dx${dx}cm 2>&1 | tail -3
  done
}

run_oft() {
  cd "$OFT_DIR" || exit 1
  local common="--pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 --task_id 6 --num_trials_per_task 50 \
    --num_open_loop_steps 8 --use_wandb False"
  echo "=== oft baseline ==="
  PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
    PYTHONPATH=$OFT_DIR:$OFT_DIR/LIBERO $OFT_PY \
    experiments/robot/libero/run_libero_eval.py $common \
    --run_id_note pert_oft_baseline 2>&1 | tail -3
  for cell in "A 1" "A 3" "A 5" "A 8" "B 8" "C 1" "C 3" "C 5" "C 8"; do
    set -- $cell; ph=$1; dx=$2
    read lo hi <<< "${PHASE_WIN[$ph]}"
    echo "=== oft phase$ph dx${dx}cm  win[$lo,$hi) ==="
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
      PYTHONPATH=$OFT_DIR:$OFT_DIR/LIBERO $OFT_PY \
      experiments/robot/libero/run_libero_eval.py $common \
      --obj_force_body chocolate_pudding_1_main --obj_force_duration 5 \
      --obj_force_horizontal True --obj_force_tilt_up 0.1 --obj_force_num_triggers 1 \
      --obj_force_target_displacement_m 0.0${dx} \
      --obj_force_min_trigger_step $lo --obj_force_max_trigger_step $hi \
      --run_id_note pert_oft_phase${ph}_dx${dx}cm 2>&1 | tail -3
  done
}

case "${1:-all}" in
  adapter) run_adapter ;;
  oft)     run_oft ;;
  all)     run_adapter; run_oft ;;
  *) echo "usage: run_grid.sh adapter|oft|all"; exit 1 ;;
esac
echo "GRID DONE: ${1:-all}"
