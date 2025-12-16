# Data Preparation Guide: FastUMI Dataset → Training-Ready Format

This guide clarifies the complete data preparation workflow step-by-step.

## Understanding the Data Flow

```
FastUMI HuggingFace     Extract        FastUMI HDF5       Your Script       LeRobot Dataset
(tar.gz archives)   →   tar.gz    →   (TCP poses)    →   (IK to joints) →  (Training format)
     ↓                     ↓               ↓                    ↓                   ↓
 pick_place.tar.gz    episode_0.hdf5   [x,y,z,qx...]    [j1,j2..j6,grip]     Ready to train!
```

## Detailed Step-by-Step

### Step 1: Download FastUMI Dataset (Compressed Archives)

The FastUMI dataset on HuggingFace is stored as **tar.gz** archives:

```bash
# List available tasks in the dataset
huggingface-cli download IPEC-COMMUNITY/FastUMI-Data --repo-type dataset --local-dir data/fastumi_raw

# You'll see files like:
# - pick_place.tar.gz.part-001
# - pick_place.tar.gz.part-002  
# - fold_towel.tar.gz
# - clean_table.tar.gz
# etc.
```

**For pick-and-place specifically**, download the relevant archives. Some tasks are split into parts (part-001, part-002, etc.).

### Step 2: Extract the Archives to Get HDF5 Files

**If single file:**
```bash
# Single archive file
tar -xzf data/fastumi_raw/pick_place.tar.gz -C data/fastumi_extracted/
```

**If split into multiple parts** (e.g., part-001, part-002):
```bash
# Combine parts first, then extract
cat data/fastumi_raw/pick_place.tar.gz.part-* > data/fastumi_raw/pick_place.tar.gz
tar -xzf data/fastumi_raw/pick_place.tar.gz -C data/fastumi_extracted/
```

**After extraction**, you should have:
```
data/fastumi_extracted/
└── pick_place/
    ├── episode_0.hdf5
    ├── episode_1.hdf5
    ├── episode_2.hdf5
    └── ...
```

Each HDF5 file contains:
```python
{
  'observations/images/front': (T, 1920, 1080, 3),  # RGB images
  'observations/qpos': (T, 7),  # TCP poses [x, y, z, qx, qy, qz, qw]
  'action': (T, 7)  # Same as qpos
}
```

### Step 3: Two Conversion Approaches

Now you have **two options**:

#### **Option A: Use Our Integrated Script (RECOMMENDED)**

Our `convert_fastumi_to_lerobot.py` script does **both conversions in one step**:
- TCP poses → Joint angles (via IK)
- HDF5 → LeRobot format

```bash
# Single command does everything
uv run examples/fastumi/convert_fastumi_to_lerobot.py \
    --data_dir data/fastumi_extracted/pick_place \
    --config_path examples/fastumi/config.json \
    --output_repo USERNAME/fastumi_xarm6_pickplace
```

**What happens inside:**
1. Loads each `episode_X.hdf5`
2. For each frame:
   - Transforms TCP pose to base frame
   - Runs IK with your xArm6 URDF → joint angles
   - Detects gripper from ArUco markers
   - Resizes images to 224x224
3. Saves to LeRobot format at `~/.cache/lerobot/USERNAME/fastumi_xarm6_pickplace/`

✅ **Use this option** - it's designed for the openpi training pipeline.

#### **Option B: Use FastUMI's Original Scripts First**

If you want to use the FastUMI repo's scripts to convert TCP→joints first:

```bash
# 1. Clone FastUMI_Data repo
git clone https://github.com/zxzm-zak/FastUMI_Data.git /tmp/FastUMI_Data

# 2. Edit their config.json with your xArm6 URDF
nano /tmp/FastUMI_Data/config/config.json
# Set:
#   - urdf_path: path to your xArm6 URDF
#   - start_qpos: initial joint config
#   - gripper settings

# 3. Run their conversion script
cd /tmp/FastUMI_Data
python data_processing_to_joint.py

# This creates new HDF5 files with joint angles instead of TCP poses
```

**Then**, you'd need to modify our `convert_fastumi_to_lerobot.py` to:
- Skip the IK step (data is already in joint space)
- Just convert HDF5 → LeRobot format

⚠️ **NOT recommended** unless you have specific needs. Our script already does this.

### Step 4: Verify Converted Data

Check the LeRobot dataset:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("USERNAME/fastumi_xarm6_pickplace")
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Features: {dataset.features}")

# Check a sample
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Image shape: {sample['observation.image'].shape}")
print(f"Joint position shape: {sample['observation.state'].shape}")
```

### Step 5: Compute Normalization Stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_fastumi_xarm6
```

Verify the stats look reasonable:
```bash
cat checkpoints/pi05_fastumi_xarm6/norm_stats.json
```

### Step 6: Start Training

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_fastumi_xarm6_lora \
    --exp-name=fastumi_v1 --overwrite
```

## Complete Commands Summary

```bash
# 1. Download dataset
huggingface-cli download IPEC-COMMUNITY/FastUMI-Data \
    --repo-type dataset \
    --local-dir data/fastumi_raw

# 2. Extract archives
mkdir -p data/fastumi_extracted
tar -xzf data/fastumi_raw/TASK_NAME.tar.gz -C data/fastumi_extracted/

# If split into parts:
cat data/fastumi_raw/TASK_NAME.tar.gz.part-* > data/fastumi_raw/TASK_NAME.tar.gz
tar -xzf data/fastumi_raw/TASK_NAME.tar.gz -C data/fastumi_extracted/

# 3. Convert to LeRobot format (includes IK)
uv run examples/fastumi/convert_fastumi_to_lerobot.py \
    --data_dir data/fastumi_extracted/TASK_NAME \
    --config_path examples/fastumi/config.json \
    --output_repo USERNAME/fastumi_xarm6_TASK

# 4. Compute norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_fastumi_xarm6

# 5. Train
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_fastumi_xarm6_lora \
    --exp-name=my_experiment --overwrite
```

## Key Clarifications

### Q: Do I need to use FastUMI's conversion scripts?

**A:** No! Our `convert_fastumi_to_lerobot.py` does everything:
- ✅ Reads HDF5 files (TCP poses)
- ✅ Converts TCP → joints via IK
- ✅ Converts to LeRobot format
- ✅ Ready for openpi training

The FastUMI repo's scripts are for their own use cases. We've integrated the logic you need.

### Q: What about the tar.gz files?

**A:** Just extract them with `tar -xzf`. That's it.

If they're split into parts (part-001, part-002), combine them first:
```bash
cat file.tar.gz.part-* > file.tar.gz
tar -xzf file.tar.gz
```

### Q: Where does the IK conversion happen?

**A:** Inside `convert_fastumi_to_lerobot.py`, in the `process_episode()` function:
1. Loads TCP poses from HDF5
2. Transforms to base frame
3. Calls `tcp_to_joint_angles()` using your URDF
4. Returns joint angles

### Q: Do I need to modify their config.json?

**A:** No, use **our** `examples/fastumi/config.json`:
- Specify YOUR xArm6 URDF path
- Set base position/orientation (sensor mounting)
- Configure gripper calibration

### Q: What's the difference between the approaches?

| Approach | Steps | Output | Use Case |
|----------|-------|--------|----------|
| **Our Script** | HDF5 → IK → LeRobot | LeRobot dataset | ✅ Training with openpi |
| FastUMI Scripts | HDF5 → IK → HDF5 | HDF5 with joints | Research / other pipelines |

For **training π₀.₅**, use our script.

## Troubleshooting

### Issue: "Cannot find episode_X.hdf5"

**Solution:** Make sure you extracted the tar.gz archive:
```bash
ls data/fastumi_extracted/TASK_NAME/
# Should show: episode_0.hdf5, episode_1.hdf5, ...
```

### Issue: "IK failed for many frames"

**Possible causes:**
1. Wrong URDF path in config.json
2. Incorrect base_position/orientation
3. TCP offset wrong
4. FastUMI data from different robot

**Solution:** 
- Verify URDF is for xArm6
- Check config.json parameters
- You may need to adjust or collect your own data

### Issue: "ModuleNotFoundError: No module named 'ikpy'"

**Solution:**
```bash
uv pip install ikpy scipy
```

### Issue: Split tar.gz files won't combine

**Solution:**
```bash
# Make sure ALL parts are downloaded first
ls data/fastumi_raw/TASK_NAME.tar.gz.*

# Then combine
cat data/fastumi_raw/TASK_NAME.tar.gz.part-* > combined.tar.gz
tar -xzf combined.tar.gz -C output_dir/
```

## Example: Complete Workflow for Pick-Place

```bash
# Assume you have access to FastUMI dataset

# 1. Download pick-place task
huggingface-cli download IPEC-COMMUNITY/FastUMI-Data \
    --include "pick_place*" \
    --repo-type dataset \
    --local-dir data/fastumi_raw

# 2. Extract (if multiple parts, combine first)
mkdir -p data/fastumi_extracted
cd data/fastumi_raw
cat pick_place.tar.gz.part-* > pick_place_combined.tar.gz
tar -xzf pick_place_combined.tar.gz -C ../fastumi_extracted/
cd ../..

# 3. Verify extraction
ls data/fastumi_extracted/pick_place/
# Should show: episode_0.hdf5, episode_1.hdf5, etc.

# 4. Get xArm6 URDF
git clone https://github.com/xArm-Developer/xarm_ros.git /tmp/xarm_ros
cp /tmp/xarm_ros/xarm_description/urdf/xarm6_robot.urdf \
   examples/fastumi/assets/

# 5. Edit config if needed
nano examples/fastumi/config.json
# Verify urdf_path points to the URDF

# 6. Convert to LeRobot format
uv run examples/fastumi/convert_fastumi_to_lerobot.py \
    --data_dir data/fastumi_extracted/pick_place \
    --config_path examples/fastumi/config.json \
    --output_repo myusername/fastumi_xarm6_pickplace

# 7. Update training config
# Edit src/openpi/training/config.py line ~912:
# Change repo_id to "myusername/fastumi_xarm6_pickplace"

# 8. Compute norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_fastumi_xarm6

# 9. Train!
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_fastumi_xarm6_lora \
    --exp-name=pickplace_v1 --overwrite
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FastUMI HuggingFace                       │
│                  (Compressed Archives)                       │
│     pick_place.tar.gz.part-001, part-002, ...               │
└────────────────────┬────────────────────────────────────────┘
                     │ Download & Extract
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Extracted HDF5 Files                       │
│              data/fastumi_extracted/pick_place/             │
│   episode_0.hdf5, episode_1.hdf5, episode_2.hdf5, ...       │
│                                                              │
│   Each HDF5 contains:                                       │
│   - observations/images/front: (T, 1920, 1080, 3)          │
│   - observations/qpos: (T, 7) [x,y,z,qx,qy,qz,qw]          │
│   - action: (T, 7)                                          │
└────────────────────┬────────────────────────────────────────┘
                     │ convert_fastumi_to_lerobot.py
                     │ (Does IK + Format Conversion)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                  LeRobot Dataset Format                      │
│         ~/.cache/lerobot/USERNAME/fastumi_xarm6/            │
│                                                              │
│   Features:                                                  │
│   - front_image: (T, 224, 224, 3)                          │
│   - joint_position: (T, 7) [j1,j2,j3,j4,j5,j6,gripper]    │
│   - actions: (T, 7)                                         │
│   - task: str                                               │
└────────────────────┬────────────────────────────────────────┘
                     │ compute_norm_stats.py
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              Normalized Training Dataset                     │
│        checkpoints/pi05_fastumi_xarm6/norm_stats.json       │
└────────────────────┬────────────────────────────────────────┘
                     │ train.py
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                   Trained π₀.₅ Model                        │
│     checkpoints/pi05_fastumi_xarm6/EXP_NAME/ITER/           │
│                  Ready for deployment!                       │
└─────────────────────────────────────────────────────────────┘
```

---

**Summary:** Extract tar.gz → Run our conversion script → Train. That's it! 🚀
