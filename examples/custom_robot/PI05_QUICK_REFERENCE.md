# Pi0.5 Training Pipeline - Quick Reference

## 📋 Summary

**Question:** Does Pi0.5 use hand joint velocities?
**Answer:** ❌ No, Pi0.5 only uses joint **positions**, not velocities.

Your pipeline is now compatible with Pi0.5! 🎉

---

## 🔧 What I Fixed

### 1. Created `align_proprioception_pi.py` (NEW)
- ✅ Adds `--task` argument for language instructions (REQUIRED for Pi0.5)
- ✅ Only outputs joint positions (no velocities)
- ✅ Combines arm + hand into single state vector
- ✅ Simplified structure ready for LeRobot

### 2. Created `convert_aligned_to_lerobot.py` (NEW)
- ✅ Reads aligned HDF5 from step 1
- ✅ Converts to LeRobot format for Pi0.5 training
- ✅ Handles language instructions properly
- ✅ Maps to correct Pi0.5 keys

### 3. Created `batch_align_for_pi05.sh` (NEW)
- ✅ Process multiple demos at once
- ✅ Consistent task descriptions

---

## 🚀 Quick Start

### Step 1: Align Your Data
```bash
cd ~/data_factory/motion_retargeting

# Single demo
python scripts/align_proprioception_pi.py \
    data/demo_20251218_120530.hdf5 \
    --task "pick up the red cube" \
    --arm-side left

# Batch process all demos
./scripts/batch_align_for_pi05.sh \
    data \
    "pick up the red cube" \
    left
```

### Step 2: Convert to LeRobot
```bash
cd ~/openpi

uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir ~/data_factory/motion_retargeting/data \
    --output_name "your_username/robot_dataset" \
    --fps 10
```

### Step 3: Train Pi0.5
```bash
cd ~/openpi

# Compute stats
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05

# Train with LoRA
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite
```

---

## 📊 Data Format at Each Stage

### Stage 1: Raw Data (data_logger_worker.py)
```
demo_20251218_120530.hdf5
├── /arm/       - timestamps, qpos, qvel, action_qpos
├── /hand/      - timestamps, qpos, qvel
└── /camera/    - timestamps, rgb_images, depth_images
```

### Stage 2: Aligned for Pi0.5 (align_proprioception_pi.py)
```
demo_20251218_120530_pi05_left.hdf5
└── /data/demo_0/
    ├── obs/qpos              # (T, 12) - arm+hand positions
    ├── obs/images/top        # (T, H, W, 3)
    ├── action                # (T, 12)
    └── task (attr)           # "pick up the red cube"
```

### Stage 3: LeRobot Format (convert_aligned_to_lerobot.py)
```
~/.cache/huggingface/lerobot/your_username/robot_dataset/
├── meta/info.json
├── data/chunk-000.parquet
└── videos/chunk-000-*.mp4
```

---

## 🎯 Key Points

### What Pi0.5 Uses:
- ✅ Joint **positions** (qpos)
- ✅ Camera images
- ✅ Language instructions
- ❌ NOT velocities (qvel)

### Why No Velocities?
Pi0.5 learns temporal dynamics from the **sequence of positions** over time. It doesn't need explicit velocity inputs.

### State Vector Composition:
For left arm:
- 6D arm joint positions
- 6D hand joint positions
- **Total: 12D state**

---

## 📝 Files Created

1. `/home/ethan/data_factory/motion_retargeting/scripts/align_proprioception_pi.py`
2. `/home/ethan/openpi/examples/custom_robot/convert_aligned_to_lerobot.py`
3. `/home/ethan/data_factory/motion_retargeting/scripts/batch_align_for_pi05.sh`
4. `/home/ethan/PI05_PIPELINE_GUIDE.md` (detailed guide)

---

## ⚠️ Important Notes

1. **Do NOT modify** existing files:
   - `align_proprioception.py` - Keep for other uses
   - `convert_hdf5_to_lerobot.py` - Keep as reference

2. **Always specify** `--task` when aligning data

3. **Use single arm** for Pi0.5 training:
   - `--arm-side left` or `--arm-side right`
   - NOT `both` (Pi0.5 expects consistent dimensions)

4. **Collect sufficient data**:
   - Minimum: 50-100 episodes
   - Recommended: 500-1000+ episodes

---

## 📚 Documentation

See `PI05_PIPELINE_GUIDE.md` for:
- Detailed explanations
- Troubleshooting guide
- Training tips
- Complete examples

---

## ✅ Checklist

- [ ] Collected raw data with data_logger_worker.py
- [ ] Aligned data with align_proprioception_pi.py (with --task)
- [ ] Converted to LeRobot with convert_aligned_to_lerobot.py
- [ ] Verified LeRobot dataset structure
- [ ] Computed normalization stats
- [ ] Started training!

Good luck! 🚀
