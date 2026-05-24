conda activate openvlaoft
conda activate openvlaoft

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

pip install packaging ninja
pip install --no-cache-dir "flash-attn==2.5.5" --no-build-isolation

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt

python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  "mujoco==3.3.2"

pip install fastapi "uvicorn[standard]" msgpack msgpack-numpy requests hf_transfer

# experiments/robot/libero/libero_utils.py
def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    render_gpu_device_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": False,
        "render_gpu_device_id": render_gpu_device_id,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description

export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"