# Fine-Tuning π₀.₅ for Your 6-DOF Robot Arm with Dexterous Hand

This guide provides a complete walkthrough for fine-tuning the π₀.₅ model on your custom robot arm (6-DOF arm + 6-DOF dexterous hand = 12 total DOF).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Understanding Your Robot Configuration](#understanding-your-robot-configuration)
3. [Data Requirements](#data-requirements)
4. [Converting HDF5 Data to LeRobot Format](#converting-hdf5-data-to-lerobot-format)
5. [Creating Custom Policy Classes](#creating-custom-policy-classes)
6. [Creating Training Configuration](#creating-training-configuration)
7. [Computing Normalization Statistics](#computing-normalization-statistics)
8. [Running Training](#running-training)
9. [Running Inference](#running-inference)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 22.5 GB VRAM for LoRA fine-tuning (e.g., RTX 4090)
- **GPU** (Full fine-tuning): 70+ GB VRAM (e.g., A100 80GB or H100)
- **Storage**: Sufficient space for your dataset (typically 10-100 GB depending on episodes)

### Software Requirements
- Ubuntu 22.04 (tested)
- Python 3.11
- NVIDIA drivers (no need for system CUDA - installed via uv)

### Installation
```bash
# Clone the repository with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# Install dependencies using uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Install additional dependencies for data conversion
uv pip install h5py opencv-python
```

---

## Understanding Your Robot Configuration

### Your Robot Setup
- **Arm**: 6 DOF (typically: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
- **Hand**: 6 active DOF (dexterous manipulation)
- **Total Action Dimension**: 12
- **Cameras**: You'll need at least 1-3 camera views:
  - Third-person view (recommended)
  - Wrist camera (highly recommended for manipulation)
  - Optional: Additional exterior views

### Action Space Considerations
You need to decide on your action representation:

1. **Joint Position (Absolute)**: Target joint angles
   - Pros: Direct control, easier to interpret
   - Cons: Requires delta conversion for π₀.₅
   
2. **Joint Velocity (Delta)**: Change in joint angles per timestep
   - Pros: Already in delta form, works directly with π₀.₅
   - Cons: Requires integration for absolute positions

3. **End-Effector Pose**: Cartesian coordinates + orientation
   - Pros: More intuitive for some tasks
   - Cons: Requires inverse kinematics, more complex

**Recommendation**: Start with joint velocities or convert absolute positions to deltas.

### State Space
- **Proprio (State)**: Current robot configuration
  - Arm joint positions (6D)
  - Hand joint positions (6D)
  - Optional: joint velocities, torques, gripper state
  - Total: 12D minimum (can be more with additional sensors)

---

## Data Requirements

### Minimum Data Requirements
- **Minimum episodes**: 50-100 episodes for basic tasks
- **Recommended**: 500-1000+ episodes for robust generalization
- **Episode length**: Varies by task (typically 50-500 timesteps)
- **Frame rate**: 10-15 Hz recommended (matches π₀.₅ pre-training)

### Data Collection Best Practices
1. **Diverse conditions**: Vary object positions, lighting, backgrounds
2. **Success and partial success**: Include both successful and partially successful demonstrations
3. **Multiple viewpoints**: Ensure cameras capture relevant workspace
4. **Language annotations**: Provide clear task descriptions
5. **Reset states**: Start from diverse initial configurations

### HDF5 Data Format
Your HDF5 files should contain:

```
episode_0/
  ├── observations/
  │   ├── images/
  │   │   ├── camera_0: (T, H, W, 3) uint8
  │   │   ├── camera_1: (T, H, W, 3) uint8  [optional]
  │   │   └── wrist_camera: (T, H, W, 3) uint8  [optional]
  │   └── state: (T, 12) float32  # Joint positions
  ├── actions: (T, 12) float32  # Your chosen action representation
  └── language_instruction: str  # Task description

episode_1/
  └── ...
```

### Required Data Fields
- **Images**: At least one RGB camera view (224x224 or higher recommended)
- **State**: Robot proprioceptive state (joint positions at minimum)
- **Actions**: Robot actions matching your action space
- **Language**: Task description for each episode

---

## Converting HDF5 Data to LeRobot Format

LeRobot is the data format used by OpenPI for training. See the conversion script created for you:
- `examples/custom_robot/convert_hdf5_to_lerobot.py`

### Conversion Process

1. **Organize your HDF5 files**:
```bash
your_data_dir/
  ├── episode_0.h5
  ├── episode_1.h5
  └── ...
```

2. **Modify the conversion script** (see `examples/custom_robot/convert_hdf5_to_lerobot.py`):
   - Update image keys to match your HDF5 structure
   - Update state/action dimensions
   - Adjust image resizing if needed

3. **Run conversion**:
```bash
uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
    --data_dir /path/to/your/hdf5/files \
    --output_name "your_username/robot_dataset"
```

This will create a LeRobot dataset in `~/.cache/huggingface/lerobot/` (or `$HF_LEROBOT_HOME`).

4. **Verify conversion**:
```bash
# The script will print the output path
ls ~/.cache/huggingface/lerobot/your_username/robot_dataset/
# Should contain: meta/info.json, data/chunk-000.parquet, videos/chunk-000-*.mp4
```

---

## Creating Custom Policy Classes

Policy classes define how data flows between your robot and the model. See the example created for you:
- `src/openpi/policies/custom_robot_policy.py`

### Key Components

1. **Input Transform** (`CustomRobotInputs`):
   - Converts your robot's data format to model input format
   - Handles image preprocessing
   - Manages state/action padding

2. **Output Transform** (`CustomRobotOutputs`):
   - Converts model predictions back to robot commands
   - Removes padding added during input processing

### Customization Steps

1. **Update image keys** in `CustomRobotInputs.__call__()`:
```python
# Match these to your camera names
base_image = _parse_image(data["observation/camera_0"])
wrist_image = _parse_image(data["observation/wrist_camera"])
```

2. **Update action dimension** in `CustomRobotOutputs.__call__()`:
```python
# Change 12 to your actual action dimension
return {"actions": np.asarray(data["actions"][:, :12])}
```

3. **Add additional sensors** if needed:
```python
# In CustomRobotInputs
inputs["state"] = np.concatenate([
    data["observation/state"],
    data["observation/velocities"],  # Add if available
])
```

---

## Creating Training Configuration

Training configurations specify model architecture, data processing, and training hyperparameters. See:
- `src/openpi/training/custom_robot_config.py`

### Configuration Components

1. **Data Config** (`CustomRobotDataConfig`):
   - Specifies LeRobot dataset location
   - Defines data transforms
   - Sets up normalization

2. **Training Config** (`CUSTOM_ROBOT_PI05_CONFIG`):
   - Model architecture (π₀.₅)
   - Learning rate and optimizer settings
   - Batch size and training duration

### Customization Steps

1. **Update `repo_id`**:
```python
@dataclasses.dataclass(frozen=True)
class CustomRobotDataConfig(DataConfigFactory):
    repo_id: str = "your_username/robot_dataset"  # Match your dataset name
```

2. **Adjust action space** if using absolute positions:
```python
# In CustomRobotDataConfig.create()
if self.use_absolute_actions:
    # Convert absolute joint positions to deltas
    # Apply to first N dimensions (arm joints), leave gripper absolute
    delta_action_mask = _transforms.make_bool_mask(6, -6)  # First 6 are deltas, last 6 absolute
    data_transforms = data_transforms.push(
        inputs=[_transforms.DeltaActions(delta_action_mask)],
        outputs=[_transforms.AbsoluteActions(delta_action_mask)],
    )
```

3. **Modify training hyperparameters** (optional):
```python
TrainConfig(
    name="custom_robot_pi05",
    batch_size=32,  # Reduce if OOM errors
    max_steps=20_000,  # Increase for larger datasets
    eval_freq=1_000,
    save_freq=5_000,
)
```

### Register Your Config

Add your config to `src/openpi/training/config.py`:
```python
from openpi.training.custom_robot_config import CUSTOM_ROBOT_PI05_CONFIG

_CONFIGS = {
    # ... existing configs ...
    "custom_robot_pi05": CUSTOM_ROBOT_PI05_CONFIG,
}
```

---

## Computing Normalization Statistics

Normalization statistics are essential for stable training. They normalize state and action values to a standard range.

### Why Normalization Matters
- Prevents numerical instabilities during training
- Ensures all dimensions contribute equally to learning
- Required for π₀.₅ model training

### Compute Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05
```

This will:
1. Load your LeRobot dataset
2. Compute statistics (mean, std, quantiles) for state and actions
3. Save to `./assets/custom_robot_pi05/norm_stats.json`

### Verify Statistics

```bash
cat ./assets/your_username/robot_dataset/norm_stats.json
```

Check that:
- No dimensions have very small std values (< 1e-6)
- q01 and q99 values are reasonable for your action space
- No NaN or Inf values

**Common Issue**: If gripper or certain joints rarely move, they may have very small std values. You can manually adjust these in the JSON file.

---

## Running Training

### Environment Setup

```bash
# Allow JAX to use more GPU memory (90%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Optional: Set data cache location
export OPENPI_DATA_HOME=~/.cache/openpi
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot
```

### Training Modes

#### 1. LoRA Fine-Tuning (Recommended)
Trains only a small set of adapter parameters. Much faster and memory-efficient.

```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite
```

**Memory**: ~22.5 GB GPU RAM

#### 2. Full Fine-Tuning
Trains all model parameters. Better performance but requires more memory.

```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_full \
    --use-lora=false \
    --overwrite
```

**Memory**: ~70 GB GPU RAM

#### 3. Multi-GPU Training
Distribute training across multiple GPUs using FSDP.

```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_multigpu \
    --fsdp-devices=4 \
    --overwrite
```

### Monitor Training

Training progress is logged to:
1. **Console**: Loss, learning rate, steps per second
2. **Weights & Biases**: Create account at wandb.ai
   - Run `wandb login` first
   - View training curves, hyperparameters online

### Checkpoints

Checkpoints are saved to:
```
checkpoints/custom_robot_pi05/my_robot_lora/
  ├── 5000/    # Checkpoint at step 5000
  ├── 10000/
  ├── 15000/
  └── 20000/
```

### Training Duration

Expected training times (single A100 80GB):
- **50 episodes**: 1-2 hours (10k steps)
- **500 episodes**: 5-10 hours (20k steps)
- **1000+ episodes**: 10-20 hours (50k steps)

---

## Running Inference

### Start Policy Server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_robot_lora/20000
```

The server will listen on `http://localhost:8000`.

### Query the Server

See `examples/custom_robot/inference_example.py` for a complete example.

```python
import requests
import numpy as np

# Prepare observation
observation = {
    "observation/camera_0": camera_image,  # (H, W, 3) uint8
    "observation/wrist_camera": wrist_image,  # (H, W, 3) uint8
    "observation/state": joint_positions,  # (12,) float32
    "prompt": "pick up the red cube"
}

# Query policy server
response = requests.post(
    "http://localhost:8000/predict",
    json={"observation": observation}
)

actions = np.array(response.json()["actions"])  # (action_horizon, 12)
```

### Deploy on Robot

For deployment on your robot:
1. **Remote inference**: Run model on powerful GPU, stream to robot
   - See `docs/remote_inference.md`
2. **On-robot inference**: Run model on robot's compute (if sufficient)

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors
**Solutions**:
- Use LoRA fine-tuning instead of full fine-tuning
- Reduce batch size: `--batch-size=16`
- Enable FSDP: `--fsdp-devices=<num_gpus>`
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
- Disable EMA: Set `ema_decay=None` in config

#### 2. Diverging Loss
**Symptoms**: Loss increases or becomes NaN

**Solutions**:
- Check norm stats for very small std values
- Reduce learning rate: `lr_schedule.peak_lr=1e-5`
- Check action space matches (absolute vs delta)
- Verify data quality (no corrupted episodes)

#### 3. Data Loading Errors
**Symptoms**: "Key not found" or shape mismatch errors

**Solutions**:
- Verify LeRobot dataset structure
- Check key names in policy input transforms
- Ensure image shapes match (H, W, C) format
- Verify action dimensions match config

#### 4. Poor Generalization
**Symptoms**: Works on training scenes but fails on new ones

**Solutions**:
- Collect more diverse data (lighting, positions, objects)
- Increase training duration
- Use data augmentation (built-in to π₀.₅)
- Fine-tune from π₀.₅-base (not π₀.₅-droid)

#### 5. Actions Don't Match Robot
**Symptoms**: Robot moves erratically or not at all

**Solutions**:
- Verify action space (absolute vs velocity)
- Check action dimension slicing in output transform
- Ensure action normalization is correct
- Test with known good actions first

### Debug Mode

Test your setup with fake data:

```bash
# This will use randomly generated data
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=debug_test \
    --config.data.repo_id=fake \
    --max-steps=10
```

### Getting Help

1. Check existing GitHub issues
2. Review the [CONTRIBUTING.md](CONTRIBUTING.md) guide
3. Post in GitHub Discussions
4. Include:
   - Error message and full traceback
   - Your config file
   - GPU specs and memory usage
   - OpenPI version: `git rev-parse HEAD`

---

## Next Steps

1. **Start small**: Test with 10-20 episodes first
2. **Verify training**: Check that loss decreases
3. **Test inference**: Run on held-out test episodes
4. **Scale up**: Collect more data and retrain
5. **Deploy**: Integrate with your robot control system

## Additional Resources

- [π₀.₅ Paper](https://www.physicalintelligence.company/blog/pi05)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [LIBERO Example](examples/libero/README.md) - Similar to your use case
- [DROID Example](examples/droid/README_train.md) - Full training pipeline
- [Remote Inference Guide](docs/remote_inference.md)

---

## Checklist

Before starting training, ensure you have:

- [ ] Collected robot data in HDF5 format
- [ ] At least 1-3 camera views (including wrist camera)
- [ ] Language instructions for each episode
- [ ] Installed OpenPI and dependencies
- [ ] Created LeRobot dataset from HDF5
- [ ] Created custom policy classes
- [ ] Created training configuration
- [ ] Registered config in `config.py`
- [ ] Computed normalization statistics
- [ ] Verified dataset structure
- [ ] Set up GPU with sufficient memory
- [ ] Configured Weights & Biases (optional)

You're ready to train! 🚀
