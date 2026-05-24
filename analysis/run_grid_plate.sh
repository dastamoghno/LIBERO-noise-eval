#!/usr/bin/env bash
# Plate-perturbation sweep: push plate_1_main, phase B [60,200), dx {1,3,5,8} cm,
# both models, n=50. 8 runs total.
set -u

ADAPTER_DIR=/home/tamoghno/VLA-Adapter
OFT_DIR=/home/tamoghno/openvla-oft
ADAPTER_PY=/home/tamoghno/miniconda3/envs/vla-adapter/bin/python
OFT_PY=/home/tamoghno/miniconda3/envs/openvla-oft/bin/python
LO=60; HI=200; BODY=plate_1_main

run_adapter() {
  cd "$ADAPTER_DIR" || exit 1
  for dx in 1 3 5 8; do
    echo "=== adapter plate phaseB dx${dx}cm ==="
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
      PYTHONPATH=$ADAPTER_DIR:/home/tamoghno/LIBERO $ADAPTER_PY \
      experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint pretrained_models/LIBERO-Long \
      --task_suite_name libero_10 --task_id 6 --num_trials_per_task 50 \
      --num_open_loop_steps 8 --use_l1_regression True --use_proprio True \
      --use_pro_version False --num_images_in_input 2 --use_wandb False \
      --obj_force_body $BODY --obj_force_duration 5 \
      --obj_force_horizontal True --obj_force_tilt_up 0.1 --obj_force_num_triggers 1 \
      --obj_force_target_displacement_m 0.0${dx} \
      --obj_force_min_trigger_step $LO --obj_force_max_trigger_step $HI \
      --run_id_note pert_adapter_plate_phaseB_dx${dx}cm 2>&1 | tail -3
  done
}

run_oft() {
  cd "$OFT_DIR" || exit 1
  for dx in 1 3 5 8; do
    echo "=== oft plate phaseB dx${dx}cm ==="
    PYTHONNOUSERSITE=1 MUJOCO_GL=egl EGL_DEVICE_ID=0 \
      PYTHONPATH=$OFT_DIR:$OFT_DIR/LIBERO $OFT_PY \
      experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
      --task_suite_name libero_10 --task_id 6 --num_trials_per_task 50 \
      --num_open_loop_steps 8 --use_wandb False \
      --obj_force_body $BODY --obj_force_duration 5 \
      --obj_force_horizontal True --obj_force_tilt_up 0.1 --obj_force_num_triggers 1 \
      --obj_force_target_displacement_m 0.0${dx} \
      --obj_force_min_trigger_step $LO --obj_force_max_trigger_step $HI \
      --run_id_note pert_oft_plate_phaseB_dx${dx}cm 2>&1 | tail -3
  done
}

case "${1:-all}" in
  adapter) run_adapter ;;
  oft) run_oft ;;
  all) run_adapter; run_oft ;;
  *) echo "usage: run_grid_plate.sh adapter|oft|all"; exit 1 ;;
esac
echo "PLATE GRID DONE: ${1:-all}"
