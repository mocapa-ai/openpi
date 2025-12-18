# Pi0.5 Data Collection and Training Pipeline

This document describes the complete pipeline for collecting robot data and training Pi0.5.

## Overview

Your pipeline has **3 stages**:

1. **Data Collection** → Raw HDF5 with separate timestamps
2. **Alignment** → Synchronized HDF5 ready for LeRobot
3. **Conversion** → LeRobot format for Pi0.5 training

```
data_logger_worker.py → align_proprioception_pi.py → convert_aligned_to_lerobot.py → Train Pi0.5
```

---

## Stage 1: Data Collection

**Script:** `motion_retargeting/motion_retargeting/data_logger_worker.py`

This runs during teleoperation and saves raw sensor data.

**Output Format:**
```
demo_20251218_120530.hdf5
├── /arm/
│   ├── timestamps        # (N,) float64 - Arm sample times
│   ├── qpos             # (N, 14) float32 - ALL joint positions
│   ├── qvel             # (N, 14) float32 - ALL joint velocities  
│   └── action_qpos      # (N, 34) float32 - ALL joints (left+right)
├── /hand/
│   ├── timestamps       # (M,) float64 - Hand sample times
│   ├── qpos             # (M, 12) float32 - Hand positions
│   └── qvel             # (M, 12) float32 - Hand velocities
└── /camera/
    ├── timestamps       # (T,) float64 - Camera frame times
    ├── rgb_images       # (T, H, W, 3) uint8
    └── depth_images     # (T, H, W) uint16
```

**No changes needed** - Your current data_logger_worker.py is fine!

---

## Stage 2: Alignment for Pi0.5

**Script:** `motion_retargeting/scripts/align_proprioception_pi.py` (NEW)

This script:
- Synchronizes all sensors to camera timestamps
- Extracts specific arm (left or right)
- Combines arm + hand joint positions
- **Adds language instruction** (required for Pi0.5)
- **Only saves positions, not velocities** (Pi0.5 doesn't use velocities)

### Usage:

```bash
cd ~/data_factory/motion_retargeting

# Align a single demo for left arm
python scripts/align_proprioception_pi.py \
    data/demo_20251218_120530.hdf5 \
    --task "pick up the red cube" \
    --arm-side left

# Batch process multiple demos
for demo in data/demo_*.hdf5; do
    python scripts/align_proprioception_pi.py "$demo" \
        --task "pick up the red cube" \
        --arm-side left
done
```

**Output Format:**
```
demo_20251218_120530_pi05_left.hdf5
└── /data/demo_0/
    ├── obs/
    │   ├── qpos              # (T, 12) float32 - arm+hand positions ONLY
    │   └── images/
    │       └── top           # (T, H, W, 3) uint8
    ├── action                # (T, 12) float32 - arm+hand actions
    ├── timestamps            # (T,) float64
    └── task (attr)           # "pick up the red cube"
```

### Important Parameters:

- `--task`: **REQUIRED** - Language description of the task
- `--arm-side`: Choose `left` or `right` (NOT `both` for Pi0.5)
- `--method`: Alignment algorithm (default: `nearest`)
  - `nearest`: Fastest, use closest sample
  - `interpolate`: Linear interpolation between samples
  - `weighted`: Gaussian-weighted average
  - `window`: Average within time window

---

## Stage 3: Convert to LeRobot Format

**Script:** `openpi/examples/custom_robot/convert_aligned_to_lerobot.py` (NEW)

This converts your aligned HDF5 files to the LeRobot dataset format that Pi0.5 expects.

### Usage:

```bash
cd ~/openpi

# Convert all aligned demos
uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir ~/data_factory/motion_retargeting/data \
    --output_name "your_username/robot_dataset" \
    --fps 10 \
    --image_width 320 \
    --image_height 180
```

**Output:** LeRobot dataset saved to `~/.cache/huggingface/lerobot/your_username/robot_dataset/`

### Verify Conversion:

```bash
ls ~/.cache/huggingface/lerobot/your_username/robot_dataset/
# Should contain:
#   meta/info.json
#   data/chunk-000.parquet
#   videos/chunk-000-*.mp4
```

---

## Stage 4: Train Pi0.5

Follow the OpenPI training guide.

### Quick Start:

```bash
cd ~/openpi

# 1. Compute normalization statistics
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05

# 2. Start training (LoRA fine-tuning)
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite
```

---

## Answer to Your Question: Does Pi0.5 Use Hand Velocities?

**No, Pi0.5 only uses joint POSITIONS, not velocities.**

From the OpenPI codebase:
- LIBERO policy: 8D state (6 arm joints + 2 gripper fingers) - positions only
- ALOHA policy: 14D state (7 per arm: 6 joints + 1 gripper) - positions only
- DROID policy: State is joint positions

**Why your pipeline is now correct:**

✅ `data_logger_worker.py` saves both positions AND velocities
✅ `align_proprioception_pi.py` extracts ONLY positions (qpos)
✅ Velocities are discarded (Pi0.5 doesn't need them)

**What goes into Pi0.5's "state":**
- Arm joint positions (6D for left arm)
- Hand joint positions (6D for left hand)
- **Total: 12D state vector**

Pi0.5 learns temporal dynamics from the sequence of positions over time, so it doesn't need explicit velocity inputs.

---

## Complete Pipeline Example

```bash
# 1. Collect data (run during teleoperation)
#    → Outputs: demo_20251218_120530.hdf5

# 2. Align for Pi0.5
cd ~/data_factory/motion_retargeting
python scripts/align_proprioception_pi.py \
    data/demo_20251218_120530.hdf5 \
    --task "pick up the red cube" \
    --arm-side left
#    → Outputs: demo_20251218_120530_pi05_left.hdf5

# 3. Convert to LeRobot (batch all aligned demos)
cd ~/openpi
uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir ~/data_factory/motion_retargeting/data \
    --output_name "ethan/picking_dataset" \
    --fps 10
#    → Outputs: ~/.cache/huggingface/lerobot/ethan/picking_dataset/

# 4. Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05

# 5. Train
uv run scripts/train.py custom_robot_pi05 --exp-name=picking_lora --overwrite
```

---

## Key Differences from Original Scripts

### New Scripts (DO NOT modify originals):

1. **`align_proprioception_pi.py`** vs `align_proprioception.py`:
   - ✅ Adds `--task` argument (required for Pi0.5)
   - ✅ Only outputs positions, not velocities
   - ✅ Combines arm+hand into single arrays
   - ✅ Only supports single arm (left or right)
   - ✅ Simplified output structure

2. **`convert_aligned_to_lerobot.py`** vs `convert_hdf5_to_lerobot.py`:
   - ✅ Reads aligned format from align_proprioception_pi.py
   - ✅ Handles combined arm+hand arrays
   - ✅ Extracts language instruction from HDF5 attributes
   - ✅ Maps to Pi0.5 expected keys: "state", "actions", "image", "task"

---

## Troubleshooting

### "No language instruction found"
→ Add `--task` argument to align_proprioception_pi.py

### "Invalid format in HDF5"
→ Make sure you're using align_proprioception_pi.py, not align_proprioception.py

### "No *_pi05_*.h5 files found"
→ Check that align_proprioception_pi.py completed successfully

### "State dimension mismatch"
→ Make sure you're using same `--arm-side` for all demos in one dataset

---

## Data Requirements for Training

- **Minimum:** 50-100 episodes for basic tasks
- **Recommended:** 500-1000+ episodes for robust generalization
- **Per episode:** 50-500 timesteps (depends on task)
- **Frame rate:** 10 Hz (matches Pi0.5 pre-training)
- **Language:** Clear, consistent task descriptions

---

## Next Steps

1. ✅ Collect 10-20 demos with your existing data_logger_worker.py
2. ✅ Run align_proprioception_pi.py on each with `--task` descriptions
3. ✅ Convert all aligned demos to LeRobot format
4. ✅ Compute normalization stats
5. ✅ Start training with LoRA
6. Test on held-out demos
7. Scale up data collection if needed

Good luck! 🚀
