# ✅ AirBot OpenPI Setup - COMPLETE

## What Was Done

All AirBot configurations have been integrated directly into `config.py` following the OpenPI convention (same pattern as LIBERO, ALOHA, etc.).

### Files Modified

**1. Policy File** (kept separate as per convention)
```
src/openpi/policies/airbot_policy.py
```
- `AirBotInputs`: Converts single camera to model format
- `AirBotOutputs`: Extracts 12D actions from model output

**2. Training Configuration** (added inline to config.py)
```
src/openpi/training/config.py
```
Added:
- `LeRobotAirBotDataConfig` class (data processing config)
- `airbot_pi05` training config (LoRA fine-tuning)
- `airbot_pi05_full` training config (full fine-tuning)
- `airbot_pi05_debug` training config (quick test)

**3. Documentation**
```
examples/custom_robot/AIRBOT_FINETUNING_GUIDE.md  - Complete training guide
examples/custom_robot/AIRBOT_SETUP_COMPLETE.md    - This file
```

### Files Removed

- ❌ `src/openpi/training/airbot_config.py` (moved inline to config.py)

## Architecture

The configuration follows the OpenPI convention where:
- **Policy transforms** are in separate files (`airbot_policy.py`)
- **Data configs and training configs** are inline in `config.py`

This matches how LIBERO, ALOHA, and DROID are structured.

## Ready to Use

Your setup now matches the README workflow exactly:

```bash
cd ~/openpi

# 1. Compute stats
uv run scripts/compute_norm_stats.py --config-name airbot_pi05

# 2. Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05 \
    --exp-name=my_experiment --overwrite

# 3. Serve
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/my_experiment/20000
```

## Configuration Summary

### Dataset: `test_pi_finetune`
- 19,818 frames across 49 episodes
- 12D state (6 arm + 6 hand)
- 12D actions (absolute → delta conversion)
- Single "image" camera
- Language instructions from "task" field

### Training Config: `airbot_pi05`
- **Model**: π₀.₅ (pi05_base)
- **Method**: LoRA fine-tuning
- **Batch size**: 32
- **Learning rate**: 3e-5
- **Steps**: 20,000
- **GPU**: ~22.5GB required

### Alternative Configs
- `airbot_pi05_full`: Full fine-tuning (~70GB GPU)
- `airbot_pi05_debug`: Quick test (100 steps, fake data)

## Verification

Test that configs are available:

```bash
cd ~/openpi
uv run scripts/train.py --help | grep airbot
```

Should show:
- `airbot_pi05`
- `airbot_pi05_full`  
- `airbot_pi05_debug`

## Pipeline Flow

```
Raw HDF5 Files
    ↓ align_proprioception_pi.py
Aligned HDF5 (data/demo_0/obs/qpos)
    ↓ convert_aligned_to_lerobot.py
LeRobot Dataset (test_pi_finetune)
    ↓ LeRobotAirBotDataConfig (config.py)
    ↓ AirBotInputs (airbot_policy.py)
Model Training
    ↓ AirBotOutputs (airbot_policy.py)
Robot Actions (12D)
```

## Key Design Decisions

### Why inline config?
- ✅ Matches OpenPI convention (see LIBERO, ALOHA, DROID)
- ✅ Easier to find and maintain
- ✅ Follows README examples exactly
- ✅ Single source of truth

### Why separate policy file?
- ✅ Policy transforms used for both training AND inference
- ✅ Can be imported by robot runtime code
- ✅ Clean separation of concerns

### Delta action conversion?
- ✅ π₀.₅ is trained on delta actions
- ✅ AirBot provides absolute positions
- ✅ Conversion happens during training
- ✅ Converted back to absolute during inference

## What's Next?

You're ready to train! Follow the guide:
📖 **[AIRBOT_FINETUNING_GUIDE.md](./AIRBOT_FINETUNING_GUIDE.md)**

Quick start:
```bash
cd ~/openpi

# Compute stats (required, one-time)
uv run scripts/compute_norm_stats.py --config-name airbot_pi05

# Start training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05 \
    --exp-name=airbot_v1 --overwrite
```

---

**✅ Setup complete! Configuration matches OpenPI conventions exactly.**
