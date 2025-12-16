# Custom Robot Fine-Tuning Example

This directory contains example code for fine-tuning π₀.₅ on a custom robot arm with dexterous hand (6-DOF arm + 6-DOF hand = 12 total DOF).

## Overview

This example demonstrates:
1. Converting HDF5 robot data to LeRobot format
2. Creating custom policy input/output transforms
3. Configuring and running training
4. Running inference with the trained model

## Files

- `convert_hdf5_to_lerobot.py` - Convert your HDF5 data to LeRobot format
- `inference_example.py` - Example inference script for querying trained policy
- `README.md` - This file

## Quick Start

### 1. Prepare Your Data

Organize your HDF5 files:
```bash
your_data_dir/
  ├── episode_0.h5
  ├── episode_1.h5
  └── ...
```

Each HDF5 file should contain:
- `observations/images/camera_0`: (T, H, W, 3) uint8
- `observations/images/wrist_camera`: (T, H, W, 3) uint8 [optional]
- `observations/state`: (T, 12) float32
- `actions`: (T, 12) float32
- `language_instruction`: str

### 2. Convert to LeRobot Format

```bash
# Install additional dependencies
uv pip install h5py opencv-python

# Run conversion
uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
    --data_dir /path/to/your/hdf5/files \
    --output_name "your_username/robot_dataset" \
    --fps 10 \
    --state_dim 12 \
    --action_dim 12
```

This creates a LeRobot dataset at `~/.cache/huggingface/lerobot/your_username/robot_dataset/`

### 3. Set Up Training Configuration

The training configuration is already created for you in:
- `src/openpi/policies/custom_robot_policy.py` - Input/output transforms
- `src/openpi/training/custom_robot_config.py` - Training config

**Important**: Update the `repo_id` in `custom_robot_config.py` to match your dataset name:
```python
repo_id: str = "your_username/robot_dataset"  # Change this!
```

### 4. Register Your Configuration

Add to `src/openpi/training/config.py`:
```python
from openpi.training.custom_robot_config import CUSTOM_ROBOT_PI05_CONFIG

_CONFIGS = {
    # ... existing configs ...
    "custom_robot_pi05": CUSTOM_ROBOT_PI05_CONFIG,
}
```

### 5. Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05
```

This creates `./assets/your_username/robot_dataset/norm_stats.json`

### 6. Start Training

```bash
# Set environment variable for GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Start training with LoRA (recommended, ~22.5 GB GPU)
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_experiment \
    --overwrite
```

Monitor training:
- Console: loss, learning rate, steps/sec
- Weights & Biases: https://wandb.ai (after running `wandb login`)

Checkpoints saved to: `checkpoints/custom_robot_pi05/my_robot_experiment/`

### 7. Run Inference

Start policy server:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_robot_experiment/20000
```

In another terminal, test inference:
```bash
python examples/custom_robot/inference_example.py
```

## Customization Guide

### Adjusting for Different Robot Configurations

#### Different DOF Configuration

If your robot has different DOF than 12:

1. **Update `convert_hdf5_to_lerobot.py`**:
```python
--state_dim 18  # Example: 6 arm + 12 hand
--action_dim 18
```

2. **Update `custom_robot_policy.py`**:
```python
# In CustomRobotOutputs.__call__()
return {"actions": np.asarray(data["actions"][:, :18])}  # Change from :12
```

3. **Update `custom_robot_config.py`**:
```python
action_dim=18,  # Change from 12
```

#### Different Camera Setup

**Only 1 camera (third-person)**:

In `custom_robot_policy.py`:
```python
def __call__(self, data: dict) -> dict:
    camera_0 = _parse_image(data["observation/camera_0"])
    
    inputs = {
        "state": data["observation/state"],
        "image": {
            "base_0_rgb": camera_0,
            "left_wrist_0_rgb": np.zeros_like(camera_0),  # Padding
            "right_wrist_0_rgb": np.zeros_like(camera_0),  # Padding
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.False_,  # Masked
            "right_wrist_0_rgb": np.False_,  # Masked
        },
    }
```

**3 cameras (2 exterior + 1 wrist)**:

All cameras available - no masking needed. Set all `image_mask` values to `np.True_`.

#### Absolute vs Delta Actions

Your actions can be:
- **Absolute**: Target joint positions
- **Delta**: Joint velocities or position changes

In `custom_robot_config.py`:
```python
# If your actions are absolute positions
use_absolute_actions: bool = True

# If your actions are already velocities/deltas
use_absolute_actions: bool = False
```

When `use_absolute_actions=True`, specify which dimensions to convert:
```python
# Convert first 11 to deltas, keep last 1 (gripper) absolute
delta_action_mask = _transforms.make_bool_mask(11, -1)

# Convert all to deltas
delta_action_mask = _transforms.make_bool_mask(12)

# Convert first 6 to deltas, keep last 6 absolute
delta_action_mask = _transforms.make_bool_mask(6, -6)
```

## Training Tips

### Hyperparameter Tuning

**Batch Size**:
- Start with `batch_size=32`
- Reduce to 16 or 8 if OOM errors
- Increase to 64 if you have memory

**Learning Rate**:
- LoRA: `peak_lr=3e-5` (default)
- Full fine-tuning: `peak_lr=1e-5`
- If loss diverges, reduce by 2-3x

**Training Duration**:
- Small dataset (50-100 episodes): 10k steps
- Medium dataset (500 episodes): 20k steps
- Large dataset (1000+ episodes): 50k steps

### Multi-GPU Training

Distribute across GPUs with FSDP:
```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=multigpu \
    --fsdp-devices=4  # Use 4 GPUs
```

### Memory Optimization

If running out of memory:
1. Use LoRA (not full fine-tuning)
2. Reduce batch size: `--batch-size=16`
3. Enable FSDP: `--fsdp-devices=<num_gpus>`
4. Disable EMA: Set `ema_decay=None` in config
5. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`

## Troubleshooting

### "Key not found" errors during training

Check that:
- Camera names in LeRobot dataset match those in `CustomRobotInputs`
- All required keys are present: `state`, `actions`, `prompt`
- Repack transform maps keys correctly

### Poor performance / robot moves erratically

1. **Check action space**: Verify absolute vs delta actions
2. **Check normalization**: Inspect `norm_stats.json` for anomalies
3. **Collect more data**: 500+ episodes recommended
4. **Verify data quality**: Remove corrupted/failed episodes

### Diverging loss during training

1. **Reduce learning rate**: Try `peak_lr=1e-5`
2. **Check norm stats**: Look for very small std values
3. **Verify action space**: Ensure actions are in correct format
4. **Longer warmup**: Increase `warmup_steps=1000`

### Inference too slow

1. **Use JAX backend** (default): Faster than PyTorch for inference
2. **Remote inference**: Run model on powerful GPU, stream actions
3. **Batch inference**: Process multiple observations together
4. **Model compilation**: First inference is slow (compilation), subsequent calls are fast

## Next Steps

1. **Collect diverse data**: Vary object positions, lighting, backgrounds
2. **Test generalization**: Evaluate on held-out scenes/objects
3. **Deploy on robot**: Integrate with your robot control system
4. **Iterate**: Collect more data where model fails, retrain

## Additional Resources

- [Main Fine-Tuning Guide](../../ROBOT_FINETUNING_GUIDE.md) - Comprehensive guide
- [LIBERO Example](../libero/README.md) - Similar tabletop manipulation
- [DROID Example](../droid/README_train.md) - Large-scale training
- [Remote Inference](../../docs/remote_inference.md) - Deploy model remotely

## Support

For issues or questions:
1. Check the [main troubleshooting guide](../../ROBOT_FINETUNING_GUIDE.md#troubleshooting)
2. Search [existing issues](https://github.com/Physical-Intelligence/openpi/issues)
3. Post in [GitHub Discussions](https://github.com/Physical-Intelligence/openpi/discussions)
