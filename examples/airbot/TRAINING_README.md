# Fine-Tuning Pi0.5 on AirBot Robot Data

This guide covers the complete pipeline for converting aligned robot demonstrations to LeRobot format and training Pi0.5 models on your custom AirBot data.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Workflow](#pipeline-workflow)
4. [Step 1: Convert to LeRobot Format](#step-1-convert-to-lerobot-format)
5. [Step 2: Verify Dataset](#step-2-verify-dataset)
6. [Step 3: Understanding OpenPI Structure](#step-3-understanding-openpi-structure)
7. [Step 4: Training Configuration](#step-4-training-configuration)
8. [Step 5: Compute Normalization Stats](#step-5-compute-normalization-stats)
9. [Step 6: Train the Model](#step-6-train-the-model)
10. [Helper Scripts](#helper-scripts)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This directory contains scripts and configuration for fine-tuning Pi0.5 on AirBot robot data. The complete workflow:

```
Aligned HDF5 Files (*_pi05_*.hdf5)
    ↓ convert_aligned_to_lerobot.py
LeRobot Dataset (stored in ~/.cache/huggingface/lerobot/)
    ↓ verify_dataset_for_pi05.py (optional validation)
Validated Dataset
    ↓ compute_stats (via lerobot)
Dataset with Normalization Stats
    ↓ train.py (with airbot_pi05 config)
Trained Pi0.5 LoRA Checkpoint
```

---

## Prerequisites

### 1. Completed Data Collection

You should have already:
- [x] Collected demonstrations using `data_logger_worker.py`
- [x] Aligned data using `batch_align_for_pi05.sh`  
- [x] Have multiple `*_pi05_*.hdf5` files ready

See the motion retargeting repo `motion_retargeting/motion_retargeting/data_collection/DATA_COLLECTION_README.md` for the data collection workflow.



### 2. Robot Specifications

**AirBot Setup**:
- **Arm**: 6-DOF 
- **Hand**: 6-DOF 
- **Total Dimensions**: 12 (6 arm + 6 hand)
- **Camera**: Single egocentric RGB camera
- **Action Type**: Absolute joint positions (converted to deltas during training)

---

## Pipeline Workflow

### Quick Start (TL;DR)

```bash
cd ~/openpi

# 1. Convert aligned HDF5 → LeRobot dataset
uv run examples/airbot/convert_aligned_to_lerobot.py \
    --data_dir ~/data_factory/motion_retargeting/aligned_demos \
    --output_name "myusername/airbot_picking" \
    --fps 30

# 2. Verify dataset (optional but recommended)
uv run examples/airbot/verify_dataset_for_pi05.py "myusername/airbot_picking"

# 3. Compute normalization stats (REQUIRED - one time only)
uv run python -m lerobot.scripts.compute_stats \
    --repo-id "myusername/airbot_picking"

# 4. Train Pi0.5 with LoRA
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05 \
    --exp-name=my_first_run \
    --overwrite
```

---

## Step 1: Convert to LeRobot Format

### 1.1 Understanding the Conversion

The `convert_aligned_to_lerobot.py` script transforms aligned HDF5 files into LeRobot's standardized dataset format.

**Input** (Aligned HDF5):
```
/data/demo_0/
  obs/qpos              # (T, 12) - Arm+hand positions
  obs/images/top        # (T, H, W, 3) - RGB camera
  action                # (T, 12) - Arm+hand actions
  timestamps            # (T,) - Frame timestamps
  task                  # Language instruction
```

**Output** (LeRobot Dataset):
```
~/.cache/huggingface/lerobot/myusername/airbot_picking/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── episodes.jsonl      # Episode information
│   ├── tasks.jsonl         # Language instructions
│   └── stats.json          # Normalization stats (after compute_stats)
├── data/
│   └── chunk-*.parquet     # Episode data in Parquet format
└── videos/
    └── chunk-*-episode_*.mp4  # Visualization videos
```

### 1.2 Running the Conversion

```bash
cd ~/openpi

uv run examples/airbot/convert_aligned_to_lerobot.py \
    --data_dir /path/to/aligned/hdf5/files \
    --output_name "myusername/dataset_name" \
    --fps 10 \
    --image_width 320 \
    --image_height 180 \
    --push_to_hub  # Optional: upload to HuggingFace
```

#### Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Directory with `*_pi05_*.hdf5` files | **Required** |
| `--output_name` | Dataset name (format: username/name) | `your_username/airbot_robot_dataset` |
| `--fps` | Framerate of robot data | `10` |
| `--image_width` | Resize images to this width | `320` |
| `--image_height` | Resize images to this height | `180` |
| `--push_to_hub` | Upload to HuggingFace Hub | `False` |
| `--no_overwrite` | Don't overwrite existing dataset | Overwrites by default |

### 1.3 Example Usage

```bash
# Convert all aligned demos from data_factory
uv run examples/airbot/convert_aligned_to_lerobot.py \
    --data_dir ~/data_factory/motion_retargeting/aligned_pi \
    --output_name "ethan/cube_picking_v1" \
    --fps 10

# Expected output:
# Found 48 HDF5 files to convert
# Detected dimensions:
#   State dim: 12
#   Action dim: 12
# Creating LeRobot dataset with features: ['image', 'state', 'actions']
# Converting episodes: 100%|████████| 48/48 [01:23<00:00,  1.74s/it]
# 
# Dataset created successfully at: ~/.cache/huggingface/lerobot/ethan/cube_picking_v1
# Total episodes: 48
# Total frames: 14,520
```
---

## Step 2: Verify Dataset

### 2.1 Quick Verification

Use the verification script for a quick check:

```bash
cd ~/openpi

uv run examples/airbot/verify_lerobot.py "myusername/dataset_name"

# Output shows:
# - Dataset location
# - File structure
# - Episode count and frame count
# - Feature dimensions
# - Sample frame data
```

### 2.2 Comprehensive Verification

For detailed validation before training:

```bash
cd ~/openpi

uv run examples/airbot/verify_dataset_for_pi05.py \
    "myusername/dataset_name" \
    --visualize  # Optional: save sample images

# This performs 6 checks:
# 1. Dataset exists at correct location
# 2. Dataset loads successfully
# 3. Has sufficient episodes and frames
# 4. Has required features (state, actions, image)
# 5. Data validation (no NaN, reasonable ranges)
# 6. Cross-episode consistency
```

### 2.3 What to Look For

The verification script will report:

**Critical Requirements** (must pass):
- [x] Dataset exists and loads
- [x] Has `state` feature (12D for arm+hand positions)
- [x] Has `actions` feature (12D for arm+hand actions)
- [x] Has at least one camera feature (`image`)
- [x] Has language instructions (`task` field)

**Recommended** (warnings if not met):
- [x] At least 1000 frames total
- [x] At least 10 episodes
- [x] No NaN or Inf values in data
- [x] Consistent dimensions across episodes

---

## Step 3: Understanding OpenPI Structure

OpenPI organizes robot configurations in a modular way. For AirBot, the relevant files are:

### 3.1 Policy Transforms

**File**: `src/openpi/policies/airbot_policy.py`

Defines how to convert between model format and robot format:

```python
class AirBotInputs:
    """Converts AirBot observations to Pi0.5 input format"""
    def __call__(self, observations):
        # Maps single 'image' camera to model's multi-camera format
        # Pi0.5 expects: {top, wrist, wrist2} cameras
        # AirBot has: {image} camera
        # Solution: replicate single camera to all views
        
class AirBotOutputs:
    """Extracts AirBot actions from Pi0.5 predictions"""
    def __call__(self, predictions):
        # Extracts 12D actions from model output
        # Returns deltas (relative changes) in joint positions
```

### 3.2 Data Configuration

**File**: `src/openpi/training/config.py`

Contains `LeRobotAirBotDataConfig` class that specifies:

```python
@dataclass
class LeRobotAirBotDataConfig(BaseDataConfig):
    repo_id: str = "myusername/airbot_dataset"
    
    # Feature mapping
    state_keys: list = field(default_factory=lambda: ["state"])
    action_keys: list = field(default_factory=lambda: ["actions"])
    camera_keys: list = field(default_factory=lambda: ["image"])
    language_key: str = "task"
    
    # Dimensions
    state_dim: int = 12  # 6 arm + 6 hand
    action_dim: int = 12
    
    # Action conversion
    absolute_actions: bool = True  # AirBot uses absolute positions
    # During training: converts to deltas
    # During inference: integrates deltas back to absolute
```

### 3.3 Training Configurations

Three configs are available in `config.py`:

1. **`airbot_pi05`** (Recommended)
   - LoRA fine-tuning
   - Batch size: 32
   - Learning rate: 3e-5
   - Steps: 20,000
   - GPU memory: ~22GB

2. **`airbot_pi05_full`** (Not implemented)
   - Full model fine-tuning
   - Higher GPU requirements (~70GB)
   - Better for large datasets (10,000+ frames)

---

## Step 4: Training Configuration

### 4.1 Modifying Data Config

If you created a dataset with a different name, update the config:

```bash
# When training, override repo_id:
uv run scripts/train.py airbot_pi05 \
    --exp-name=my_run \
    --data.repo_id="myusername/my_dataset"
```

### 4.2 Hyperparameter Tuning

Key parameters you might want to adjust:

```bash
# Adjust learning rate
--learning_rate=1e-5

# Change batch size (if GPU memory allows)
--batch_size=64

# More training steps
--num_steps=30000

# Evaluation frequency
--eval_freq=1000
```

See all options: `uv run scripts/train.py airbot_pi05 --help`

---

## Step 5: Compute Normalization Stats

**CRITICAL STEP**: Must be done once before training.

### 5.1 Why Normalization Stats?

Pi0.5 requires normalized inputs for stable training:
- **State normalization**: Centers and scales joint positions
- **Action normalization**: Standardizes action magnitudes
- **Image normalization**: Already handled (pixel values 0-255 → 0-1)

Stats are computed from your dataset and saved in `meta/stats.json`.

### 5.2 Computing Stats

```bash
cd ~/openpi

uv run scripts/compute_norm_stats.py --config-name airbot_pi05


# Output:
# Processing episodes: 100%
# Stats saved to: ~/.cache/huggingface/lerobot/myusername/dataset_name/meta/stats.json
```

**Note**: Only need to run once per dataset. Re-run if you add more demonstrations.

---

## Step 6: Train the Model

### 6.1 Basic Training

```bash
cd ~/openpi

# Set GPU memory allocation (prevents OOM errors)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Start training
uv run scripts/train.py airbot_pi05 \
    --exp-name=picking_v1 \
    --overwrite
```

### 6.2 Training with Custom Settings

```bash
# Override dataset name
uv run scripts/train.py airbot_pi05 \
    --exp-name=picking_v1 \
    --data.repo_id="myusername/my_dataset" \
    --overwrite

# Adjust training hyperparameters
uv run scripts/train.py airbot_pi05 \
    --exp-name=picking_v1 \
    --learning_rate=1e-5 \
    --batch_size=64 \
    --num_steps=30000 \
    --overwrite
```

### 6.3 Training Output

**Checkpoints**: Saved to `checkpoints/airbot_pi05/{exp-name}/`

```
checkpoints/airbot_pi05/picking_v1/
├── 5000/          # Checkpoint at step 5000
├── 10000/         # Checkpoint at step 10000
├── 15000/         # Checkpoint at step 15000
└── 20000/         # Final checkpoint
```

**Logs**: Monitor training progress

```bash
# Option 1: Console output
# Shows loss, learning rate, steps/sec

# Option 2: TensorBoard (if enabled)
tensorboard --logdir checkpoints/airbot_pi05/picking_v1/

# Option 3: Weights & Biases (if configured)
# Check wandb.ai dashboard
```

---

## Helper Scripts

This directory provides several helper scripts for development and debugging:

### `inspect_hdf5.py`

Quick inspection of aligned HDF5 structure:

```bash
uv run python examples/airbot/inspect_hdf5.py /path/to/demo_pi05.hdf5

# Shows:
# - File attributes (task, arm side, dimensions)
# - Data shapes and types
# - Whether it's Pi0.5 compatible
```

### `verify_dataset_for_pi05.py`

Comprehensive dataset validation (described in Step 2):

```bash
uv run examples/airbot/verify_dataset_for_pi05.py \
    "myusername/dataset" \
    --visualize
```

### `verify_lerobot.py`

Lightweight dataset verification:

```bash
uv run python examples/airbot/verify_lerobot.py "myusername/dataset"
```

### `test_conversion.sh`

End-to-end test with a single demonstration:

```bash
cd ~/openpi/examples/airbot

# Place one aligned HDF5 file (demo0.hdf5) in this directory
./test_conversion.sh

# This script:
# 1. Inspects the HDF5 structure
# 2. Converts to LeRobot (creates test/single_trajectory)
# 3. Verifies the conversion
# 4. Shows next steps
```

---

## Troubleshooting

### Dataset Issues

**Problem**: "Dataset not found at..."

**Solution**:
- Check dataset name matches exactly (case-sensitive)
- Verify location: `ls ~/.cache/huggingface/lerobot/myusername/`
- Re-run conversion if needed

---

**Problem**: "Missing required feature: state"

**Solution**:
- Verify aligned HDF5 files have `/data/demo_0/obs/qpos`
- Re-run alignment with correct `--arm-side` parameter
- Check with `inspect_hdf5.py` before conversion

---

**Problem**: "Inconsistent dimensions across episodes"

**Solution**:
- All demos must use same arm side (left vs right)
- Re-align with consistent `--arm-side` parameter
- Separate left/right demos into different datasets

---

### Training Issues

**Problem**: "Stats not found - please run compute_stats first"

**Solution**:
```bash
uv run python -m lerobot.scripts.compute_stats \
    --repo-id "myusername/dataset"
```

---

**Problem**: CUDA Out of Memory

**Solutions**:
1. Reduce batch size: `--batch_size=16`
2. Use LoRA instead of full: Use `airbot_pi05` not `airbot_pi05_full`
3. Set memory fraction: `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`
4. Use gradient accumulation (if available)

---

**Problem**: Training loss not decreasing

**Possible causes**:
- Learning rate too high/low → Try `--learning_rate=1e-5` or `--learning_rate=1e-4`
- Insufficient data → Collect more demos (aim for 50+)
- Data quality issues → Review demos, ensure successful task completions
- Normalization stats not computed → Re-run `compute_stats`

---

**Problem**: Large gap between train and eval loss

**Cause**: Overfitting to training data

**Solutions**:
- Collect more diverse demonstrations
- Reduce model capacity (use LoRA with lower rank)
- Add regularization
- Early stopping based on eval loss

---

### Conversion Issues

**Problem**: "No *_pi05_*.hdf5 files found"

**Solution**:
- Check `--data_dir` path is correct
- Verify files were aligned (see data collection guide)
- Look for `*_pi05_left.hdf5` or `*_pi05_right.hdf5` files

---

**Problem**: Images look wrong / all black

**Solution**:
- Check camera was working during data collection
- Verify RGB images in raw HDF5: `h5ls -r demo_raw.hdf5`
- BGR→RGB conversion is automatic, but verify with `--visualize` flag

---

## File Structure Reference

```
openpi/examples/airbot/
├── TRAINING_README.md                          # This file
├── convert_aligned_to_lerobot.py     # HDF5 → LeRobot conversion
├── verify_dataset_for_pi05.py        # Comprehensive validation
├── verify_lerobot.py                 # Quick validation
├── inspect_hdf5.py                   # HDF5 structure inspection
└── test_conversion.sh                # End-to-end test script

Related files in openpi/src/:
├── openpi/policies/airbot_policy.py   # Input/output transforms
└── openpi/training/config.py          # Training configurations
```

---

## Related Documentation

- **Data Collection**: `motion_retargeting/motion_retargeting/data_collection/DATA_COLLECTION_README.md`
- **OpenPI Main README**: `~/openpi/README.md`
- **LeRobot Documentation**: https://github.com/huggingface/lerobot
- **Pi0.5 Paper**: https://physicalintelligence.company/blog/pi05

---

## Summary Checklist

Before training, ensure:

- [ ] Aligned HDF5 files created (`*_pi05_*.hdf5`)
- [ ] Converted to LeRobot dataset
- [ ] Dataset verified with `verify_dataset_for_pi05.py`
- [ ] Normalization stats computed
- [ ] Training config points to correct dataset
- [ ] Sufficient GPU memory available
