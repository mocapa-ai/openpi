# Testing Single Trajectory Conversion

Quick guide to test your data pipeline with one trajectory before processing all 50.

## 📋 Prerequisites

You should have already:
1. Collected one demo with `data_logger_worker.py` → `demo_raw.hdf5`
2. Aligned it with `align_proprioception_pi.py` → `demo0.hdf5`

## 🚀 Quick Test (One Command)

```bash
cd ~/openpi/examples/custom_robot
./test_conversion.sh
```

This will:
1. ✅ Inspect your aligned HDF5 structure
2. ✅ Convert to LeRobot format (1 episode)
3. ✅ Verify the LeRobot dataset
4. ✅ Show you what to do next

## 📝 Step-by-Step Testing

### Step 1: Inspect Aligned HDF5

```bash
cd ~/openpi/examples/custom_robot
uv run python inspect_hdf5.py demo0.hdf5
```

**Expected output:**
```
✅ Pi0.5 compatible format detected!
   Task: pick up the ball and place it on the right side of the table
   Arm side: left
   State dim: 12
   Action dim: 12
   Frames: XXX

📊 Demo structure:
  state (qpos): (XXX, 12) float32
  actions: (XXX, 12) float32
  image (top): (XXX, H, W, 3) uint8
  task: "pick up the ball..."

✅ This file is ready for conversion to LeRobot!
```

### Step 2: Convert to LeRobot

```bash
cd ~/openpi
uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir examples/custom_robot \
    --output_name "test/single_trajectory" \
    --fps 10
```

**Expected output:**
```
Found 1 HDF5 files to convert
Detected dimensions:
  State dim: 12
  Action dim: 12
Creating LeRobot dataset with features: ['image', 'state', 'actions']
Converting episodes: 100%
Dataset created successfully at: ~/.cache/huggingface/lerobot/test/single_trajectory
Total episodes: 1
Total frames: XXX
```

### Step 3: Verify LeRobot Dataset

```bash
cd ~/openpi
uv run python examples/custom_robot/verify_lerobot.py test/single_trajectory
```

**Expected output:**
```
✅ Dataset found at: ~/.cache/huggingface/lerobot/test/single_trajectory

📁 Dataset files:
  data/chunk-000.parquet (X.XX MB)
  meta/info.json (X.XX MB)
  videos/chunk-000-episode_0.mp4 (X.XX MB)

✅ Dataset loaded successfully!

📈 Dataset statistics:
  Total episodes: 1
  Total frames: XXX
  FPS: 10
  Robot type: custom

🔑 Features:
  image: shape=(180, 320, 3), dtype=image
  state: shape=(12,), dtype=float32
  actions: shape=(12,), dtype=float32

🔍 First frame:
  state: shape=(12,), dtype=float32
    -> min=X.XXXX, max=X.XXXX
  actions: shape=(12,), dtype=float32
    -> min=X.XXXX, max=X.XXXX
  image: shape=(180, 320, 3), dtype=uint8
  task: pick up the ball and place it on the right side of the table

✅ Verification complete!
```

### Step 4: Manual File Check

```bash
ls -lh ~/.cache/huggingface/lerobot/test/single_trajectory/
```

## ✅ Success Criteria

Your conversion is successful if:

- ✅ `inspect_hdf5.py` shows "Pi0.5 compatible format"
- ✅ Conversion completes without errors
- ✅ `verify_lerobot.py` shows 1 episode with correct dimensions
- ✅ State and action dims are both 12 (for left arm + hand)
- ✅ Task string matches your language instruction
- ✅ Files exist: `meta/info.json`, `data/chunk-000.parquet`, `videos/*.mp4`

## 🎯 What Each Script Does

### `inspect_hdf5.py`
- Checks if HDF5 is in correct aligned format
- Shows data shapes and task description
- Validates it's ready for conversion

### `convert_aligned_to_lerobot.py`
- Converts aligned HDF5 → LeRobot format
- Can handle single file or directory with multiple files
- Creates dataset at `~/.cache/huggingface/lerobot/`

### `verify_lerobot.py`
- Loads LeRobot dataset to verify it's valid
- Shows statistics and first frame contents
- Confirms it's ready for training

### `test_conversion.sh`
- Runs all three steps automatically
- Comprehensive end-to-end test

## 🔧 Troubleshooting

### "NOT a Pi0.5 aligned format"
→ Make sure you used `align_proprioception_pi.py`, not `align_proprioception.py`
→ Check that `--task` argument was provided

### "No HDF5 files found"
→ Make sure `demo0.hdf5` is in `examples/custom_robot/`
→ Or point `--data_dir` to the correct location

### "State dimension mismatch"
→ Verify you used same `--arm-side` for all demos
→ Left arm = 12D (6 arm + 6 hand), Right arm = 12D

### Conversion fails with import errors
→ Make sure you're using `uv run` to execute scripts
→ This ensures correct environment with all dependencies

## 📚 Next Steps After Successful Test

1. **Collect more data** (aim for 50+ trajectories)
   ```bash
   # Run data_logger_worker.py 50 times
   # Each run creates one demo_YYYYMMDD_HHMMSS.hdf5
   ```

2. **Batch align all demos**
   ```bash
   cd ~/data_factory/motion_retargeting
   ./scripts/batch_align_for_pi05.sh \
       data \
       "pick up the ball and place it on the right side of the table" \
       left
   ```

3. **Convert all to one dataset**
   ```bash
   cd ~/openpi
   uv run examples/custom_robot/convert_aligned_to_lerobot.py \
       --data_dir ~/data_factory/motion_retargeting/data \
       --output_name "ethan/picking_dataset" \
       --fps 10
   ```

4. **Compute normalization stats**
   ```bash
   cd ~/openpi
   uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05
   ```

5. **Start training**
   ```bash
   cd ~/openpi
   uv run scripts/train.py custom_robot_pi05 \
       --exp-name=picking_lora \
       --overwrite
   ```

## 📖 Related Documentation

- `PI05_QUICK_REFERENCE.md` - Quick reference for the full pipeline
- `PI05_PIPELINE_GUIDE.md` - Detailed pipeline documentation
- `README.md` - Custom robot setup guide

## ❓ Questions?

If your test succeeds, you're ready to collect more data! If you encounter issues:
1. Check the error messages carefully
2. Verify file locations and permissions
3. Make sure you're using `uv run` for all Python scripts
4. Review the Pi0.5 pipeline guide for detailed explanations

Good luck! 🚀
