# AirBot Pi0.5 Validation - Single PC Setup

Simple validation for fine-tuned Pi0.5 on AirBot (6-DOF arm + 6-DOF Revo2 hand).

## Quick Start

```bash
cd ~/openpi

# 1. Test it loads your model
uv run python examples/airbot/validate_simple.py \
    --checkpoint_dir=checkpoints/airbot_pi05/20000 \
    --max_timesteps=10

# 2. Implement robot control
vim examples/airbot/validate_simple.py
# Edit the RobotInterface class (3 functions):
#   - get_observation() - Get camera image and joint positions
#   - set_action() - Send commands to motors  
#   - reset() - Return to home position

# 3. Run validation
uv run python examples/airbot/validate_simple.py
```

## Files

**Validation (Single PC):**
- `SINGLE_PC_SETUP.md` - Complete guide for single-PC validation (no ROS, no remote server)
- `validate_simple.py` - Simple validation script (implement RobotInterface class)
- `START_HERE.txt` - Visual quick start

**Training:**
- `TRAINING_README.md` - Training guide (data conversion, training, etc.)
- `convert_aligned_to_lerobot.py` - Convert HDF5 to LeRobot format
- `verify_dataset_for_pi05.py` - Dataset validation
- `verify_lerobot.py` - Quick dataset check
- `inspect_hdf5.py` - HDF5 inspection
- `test_conversion.sh` - Test conversion script

## What You Need to Do

1. **Read the guide**: `cat SINGLE_PC_SETUP.md`
2. **Edit the script**: `vim validate_simple.py`
3. **Implement RobotInterface** class with your robot control code:
   - `get_observation()` - Get camera image (H,W,3) + joint positions (12,)
   - `set_action(action)` - Execute action (12,) on robot
   - `reset()` - Return robot to home position

## Your Setup

- **Robot**: 6-DOF AirBot arm + 6-DOF Revo2 hand (12 DOF total)
- **Camera**: 1x egocentric RGB camera
- **Model**: Pi0.5 (flow-based VLA)
- **Checkpoint**: `checkpoints/airbot_pi05/20000/`
- **Control**: 30 Hz, absolute positions → deltas

## Architecture

**Single PC (Simple):**
```
┌─────────────────────────────────────┐
│   Your PC                           │
│                                     │
│   ┌──────────┐      ┌────────────┐ │
│   │ Pi0.5    │ ──►  │ Robot      │ │
│   │ Policy   │ ◄──  │ Control    │ │
│   └──────────┘      └────────────┘ │
│                                     │
│   Direct Python calls (no network)  │
└─────────────────────────────────────┘
```

**Advantages:**
- ✅ Simple (one Python process)
- ✅ Fast (<1ms latency)
- ✅ Easy to debug
- ✅ No ROS, no networking

## Getting Help

- Read `SINGLE_PC_SETUP.md` for detailed instructions
- See `validate_simple.py` comments for examples
- Check `TRAINING_README.md` if you need to retrain

## Expected Performance

Based on your training data quality:
- **Training scenarios**: 70-90% success rate
- **Novel scenarios**: 30-50% (if fine-tuned from pi05_base)

Iterate by collecting more data → re-training → re-validating.
