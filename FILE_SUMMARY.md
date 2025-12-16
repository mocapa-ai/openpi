# Custom Robot Fine-Tuning - File Summary

This document lists all the files created for fine-tuning π₀.₅ on your custom 6-DOF arm + 6-DOF hand robot.

## 📚 Documentation Files

### 1. **ROBOT_FINETUNING_GUIDE.md** (Main Guide)
**Location:** `/home/ethan/openpi/ROBOT_FINETUNING_GUIDE.md`

Comprehensive guide covering:
- Prerequisites and installation
- Understanding robot configuration
- Data requirements and collection best practices
- HDF5 to LeRobot conversion
- Creating custom policy classes
- Training configuration
- Computing normalization statistics
- Running training (LoRA, full, multi-GPU)
- Running inference
- Troubleshooting common issues

**Start here!** This is your complete reference.

### 2. **QUICK_START_CHECKLIST.md** (Checklist)
**Location:** `/home/ethan/openpi/QUICK_START_CHECKLIST.md`

Quick reference checklist with:
- Pre-training checklist (hardware, data, configs)
- Training commands
- Monitoring guidelines
- Post-training steps
- Common issues and solutions
- Key configuration points

**Use this** to ensure you don't miss any steps.

### 3. **examples/custom_robot/README.md** (Quick Start)
**Location:** `/home/ethan/openpi/examples/custom_robot/README.md`

Example-specific documentation:
- Quick start guide
- File descriptions
- Customization guide for different robot configs
- Training tips and hyperparameters
- Multi-GPU training
- Troubleshooting

## 🐍 Python Code Files

### 4. **examples/custom_robot/convert_hdf5_to_lerobot.py** (Data Conversion)
**Location:** `/home/ethan/openpi/examples/custom_robot/convert_hdf5_to_lerobot.py`

Converts HDF5 robot data to LeRobot format.

**Features:**
- Automatic camera detection
- Flexible HDF5 structure support
- Image resizing
- Configurable dimensions
- Progress tracking
- Optional Hugging Face Hub upload

**Usage:**
```bash
uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
    --data_dir /path/to/hdf5/files \
    --output_name "username/dataset_name" \
    --fps 10 \
    --state_dim 12 \
    --action_dim 12
```

**Modify for your setup:**
- Camera key mapping (lines 165-185)
- State/action dimensions (command line args)
- Feature definitions (lines 193-219)

### 5. **src/openpi/policies/custom_robot_policy.py** (Policy Transforms)
**Location:** `/home/ethan/openpi/src/openpi/policies/custom_robot_policy.py`

Defines input/output transforms for your robot.

**Key Classes:**
- `CustomRobotInputs`: Converts robot data → model format
- `CustomRobotOutputs`: Converts model predictions → robot actions
- `make_custom_robot_example()`: Creates dummy data for testing

**Modify for your setup:**
- Camera keys (lines 85-95)
- Action dimension (line 176)
- Additional sensors (add to state, lines 115-120)

### 6. **src/openpi/training/custom_robot_config.py** (Training Config)
**Location:** `/home/ethan/openpi/src/openpi/training/custom_robot_config.py`

Complete training configuration.

**Key Components:**
- `CustomRobotDataConfig`: Data loading and processing
- `CUSTOM_ROBOT_PI05_CONFIG`: Main training config
- `CUSTOM_ROBOT_PI05_FULL_CONFIG`: Full fine-tuning variant
- `CUSTOM_ROBOT_PI05_DEBUG_CONFIG`: Debug config

**Modify for your setup:**
- `repo_id`: Your LeRobot dataset name (line 49)
- `use_absolute_actions`: True if absolute positions (line 54)
- `action_dim`: Your robot's DOF (line 126)
- Delta action mask: Which dims to convert (line 102)
- Hyperparameters: batch_size, learning rate, etc. (lines 130-170)

### 7. **examples/custom_robot/inference_example.py** (Inference)
**Location:** `/home/ethan/openpi/examples/custom_robot/inference_example.py`

Example inference script.

**Features:**
- Policy server communication
- Dummy observation creation
- Action execution placeholder
- Integration examples for ROS, PyBullet

**Usage:**
```bash
# Terminal 1: Start policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_exp/20000

# Terminal 2: Run inference
python examples/custom_robot/inference_example.py
```

**Modify for your setup:**
- Replace `create_dummy_observation()` with real sensors
- Replace `execute_actions_on_robot()` with real control

## 🔧 Configuration Integration

### 8. **Modified: src/openpi/training/config.py** (Config Registration)
**Location:** `/home/ethan/openpi/src/openpi/training/config.py`

**Changes made:**
1. Added import (line 21):
   ```python
   import openpi.policies.custom_robot_policy as custom_robot_policy
   ```

2. Added import (line 27):
   ```python
   import openpi.training.custom_robot_config as custom_robot_config
   ```

3. Registered config (line 967):
   ```python
   custom_robot_config.CUSTOM_ROBOT_PI05_CONFIG,
   ```

This makes your config available via:
```bash
uv run scripts/train.py custom_robot_pi05 --exp-name=my_experiment
```

## 📋 Complete Workflow

### Step-by-Step Process

```bash
# 1. Prepare environment
cd /home/ethan/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install h5py opencv-python

# 2. Convert data
uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
    --data_dir /path/to/your/hdf5/data \
    --output_name "your_username/robot_dataset"

# 3. Update configuration
# Edit src/openpi/training/custom_robot_config.py:
#   - Set repo_id = "your_username/robot_dataset"
#   - Set use_absolute_actions (True/False)
#   - Verify action_dim = 12 (or your value)

# 4. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05

# 5. Start training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite

# 6. Run inference
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_robot_lora/20000

# In another terminal:
python examples/custom_robot/inference_example.py
```

## 🎯 Quick Customization Guide

### For Different DOF Count

**Example: 18 DOF (6 arm + 12 hand)**

1. Data conversion:
   ```bash
   --state_dim 18 --action_dim 18
   ```

2. Policy output (`custom_robot_policy.py` line 176):
   ```python
   return {"actions": np.asarray(data["actions"][:, :18])}
   ```

3. Training config (`custom_robot_config.py` line 126):
   ```python
   action_dim=18,
   ```

### For Different Camera Setup

**Example: Only wrist camera**

Policy input (`custom_robot_policy.py` lines 85-100):
```python
wrist_camera = _parse_image(data["observation/wrist_camera"])

inputs = {
    "image": {
        "base_0_rgb": wrist_camera,  # Use wrist as primary
        "left_wrist_0_rgb": np.zeros_like(wrist_camera),  # Pad
        "right_wrist_0_rgb": np.zeros_like(wrist_camera),  # Pad
    },
    "image_mask": {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.False_,
    },
}
```

### For Absolute Actions

Training config (`custom_robot_config.py` line 54):
```python
use_absolute_actions: bool = True
```

And specify which dimensions (line 102):
```python
# Example: Convert first 11 to delta, keep gripper absolute
delta_action_mask = _transforms.make_bool_mask(11, -1)
```

## 📊 File Structure

```
openpi/
├── ROBOT_FINETUNING_GUIDE.md          # Main comprehensive guide
├── QUICK_START_CHECKLIST.md           # Quick reference checklist
├── examples/
│   └── custom_robot/
│       ├── README.md                   # Example-specific docs
│       ├── convert_hdf5_to_lerobot.py  # Data conversion script
│       └── inference_example.py        # Inference example
├── src/
│   └── openpi/
│       ├── policies/
│       │   └── custom_robot_policy.py  # Input/output transforms
│       └── training/
│           ├── config.py               # Modified: registered config
│           └── custom_robot_config.py  # Training configuration
└── [Generated during training]
    ├── assets/
    │   └── your_username/
    │       └── robot_dataset/
    │           └── norm_stats.json     # Normalization statistics
    └── checkpoints/
        └── custom_robot_pi05/
            └── my_experiment/
                ├── 5000/               # Checkpoint at step 5000
                ├── 10000/
                └── 20000/
```

## 🚀 Next Steps

1. **Read** `ROBOT_FINETUNING_GUIDE.md` - Understand the complete process
2. **Check** `QUICK_START_CHECKLIST.md` - Ensure you have everything
3. **Customize** the three Python files for your robot
4. **Run** the data conversion script
5. **Compute** normalization statistics
6. **Train** your model
7. **Test** with inference example
8. **Deploy** on your robot

## 📞 Getting Help

- **Documentation**: Start with `ROBOT_FINETUNING_GUIDE.md`
- **Checklist**: Use `QUICK_START_CHECKLIST.md`
- **Issues**: Check GitHub Issues
- **Discussions**: Post on GitHub Discussions
- **Examples**: See LIBERO and DROID examples for reference

## ✅ Summary

You now have:
- ✅ Complete documentation (3 guides)
- ✅ Data conversion script (HDF5 → LeRobot)
- ✅ Policy transforms (robot ↔ model)
- ✅ Training configuration (π₀.₅)
- ✅ Inference example
- ✅ Config registered in OpenPI

**Everything is ready to start training!** 🎉

Follow the workflow above and refer to the guides as needed.
