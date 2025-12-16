# Quick Start Checklist for Fine-Tuning π₀.₅

Use this checklist to ensure you have everything ready before training.

## ✅ Pre-Training Checklist

### Hardware & Software
- [ ] NVIDIA GPU with sufficient VRAM
  - [ ] LoRA: ≥22.5 GB (RTX 4090)
  - [ ] Full fine-tuning: ≥70 GB (A100 80GB)
- [ ] Ubuntu 22.04 (or compatible Linux)
- [ ] Installed OpenPI and dependencies (`uv sync`)
- [ ] Installed additional packages (`uv pip install h5py opencv-python`)

### Data Collection
- [ ] Collected robot demonstrations in HDF5 format
- [ ] Minimum 50-100 episodes (500+ recommended)
- [ ] At least 1 camera view (third-person or wrist)
- [ ] State observations (joint positions/velocities)
- [ ] Actions (velocities or positions)
- [ ] Language instructions for each episode
- [ ] Verified HDF5 structure matches expected format

### Data Conversion
- [ ] Organized HDF5 files in single directory
- [ ] Modified `convert_hdf5_to_lerobot.py` for your setup
  - [ ] Updated camera key names
  - [ ] Set correct state/action dimensions
- [ ] Ran conversion script successfully
- [ ] Verified LeRobot dataset created in `~/.cache/huggingface/lerobot/`
- [ ] Checked dataset has correct number of episodes and frames

### Configuration Files
- [ ] Created/modified `src/openpi/policies/custom_robot_policy.py`
  - [ ] Updated image keys to match your cameras
  - [ ] Set correct action dimension in output transform
- [ ] Created/modified `src/openpi/training/custom_robot_config.py`
  - [ ] Set `repo_id` to your LeRobot dataset name
  - [ ] Configured `use_absolute_actions` correctly
  - [ ] Set `action_dim` to match your robot
- [ ] Registered config in `src/openpi/training/config.py`
  - [ ] Added import statement
  - [ ] Added to `_CONFIGS` dictionary

### Normalization Statistics
- [ ] Ran `compute_norm_stats.py` script
- [ ] Verified `norm_stats.json` created in assets directory
- [ ] Checked for anomalies:
  - [ ] No NaN or Inf values
  - [ ] No dimensions with very small std (< 1e-6)
  - [ ] q01 and q99 values are reasonable

### Training Setup
- [ ] Set environment variable: `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
- [ ] Configured Weights & Biases (optional): `wandb login`
- [ ] Decided on training mode:
  - [ ] LoRA (recommended)
  - [ ] Full fine-tuning
  - [ ] Multi-GPU with FSDP

## 🚀 Training Command

```bash
# LoRA training (recommended)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite

# Full fine-tuning (more GPU memory required)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_full \
    --use-lora=false \
    --overwrite

# Multi-GPU training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_multigpu \
    --fsdp-devices=4 \
    --overwrite
```

## 📊 Monitoring Training

### During Training
- [ ] Monitor console output for:
  - [ ] Loss decreasing
  - [ ] No NaN/Inf in loss
  - [ ] Steps per second stable
- [ ] Check Weights & Biases dashboard (if configured)
  - [ ] Loss curve trending down
  - [ ] Learning rate schedule correct
  - [ ] No anomalies in gradients

### Checkpoints
- [ ] Verify checkpoints saved to `checkpoints/custom_robot_pi05/<exp_name>/`
- [ ] Check checkpoint frequency matches `save_freq`
- [ ] Note best checkpoint number for inference

## 🎯 Post-Training

### Inference Setup
- [ ] Start policy server:
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_experiment/<step>
```

- [ ] Test with example script:
```bash
python examples/custom_robot/inference_example.py
```

- [ ] Verify:
  - [ ] Server responds to health check
  - [ ] Actions have correct shape
  - [ ] Action values are reasonable

### Robot Deployment
- [ ] Integrate policy server with robot control
- [ ] Test on simple tasks first
- [ ] Monitor for safety
- [ ] Collect data on failure cases
- [ ] Retrain with augmented dataset

## 🐛 Common Issues

### Issue: Out of Memory
**Solutions:**
- Use LoRA instead of full fine-tuning
- Reduce batch size: `--batch-size=16`
- Enable FSDP: `--fsdp-devices=<num_gpus>`
- Disable EMA: Set `ema_decay=None` in config

### Issue: Loss Diverging
**Solutions:**
- Reduce learning rate: `peak_lr=1e-5`
- Check norm stats for anomalies
- Verify action space (absolute vs delta)
- Increase warmup steps: `warmup_steps=1000`

### Issue: Key Not Found
**Solutions:**
- Check camera names in policy transforms
- Verify repack transform mappings
- Ensure all required keys in dataset

### Issue: Poor Generalization
**Solutions:**
- Collect more diverse data
- Increase training duration
- Use stronger base model
- Add data augmentation

## 📚 Key Configuration Points

### Action Space
```python
# In custom_robot_config.py

# If your HDF5 has absolute positions
use_absolute_actions: bool = True

# If your HDF5 has velocities/deltas
use_absolute_actions: bool = False
```

### Camera Configuration
```python
# In custom_robot_policy.py, CustomRobotInputs

# Modify these lines to match your cameras:
camera_0 = _parse_image(data["observation/camera_0"])
wrist_camera = _parse_image(data["observation/wrist_camera"])
```

### Action Dimensions
```python
# In custom_robot_policy.py, CustomRobotOutputs

# Change :12 to your action dimension
return {"actions": np.asarray(data["actions"][:, :12])}
```

### Training Hyperparameters
```python
# In custom_robot_config.py

batch_size=32  # Reduce if OOM
max_steps=20_000  # Increase for more data
peak_lr=3e-5  # LoRA, or 1e-5 for full fine-tuning
```

## 📞 Getting Help

If you encounter issues:
1. ✅ Check this checklist - missed a step?
2. 📖 Read [ROBOT_FINETUNING_GUIDE.md](ROBOT_FINETUNING_GUIDE.md)
3. 🔍 Search [GitHub Issues](https://github.com/Physical-Intelligence/openpi/issues)
4. 💬 Post in [GitHub Discussions](https://github.com/Physical-Intelligence/openpi/discussions)
5. 🐛 File a new issue with:
   - Full error traceback
   - Your config file
   - GPU specs and memory usage
   - Steps to reproduce

## 🎓 Learning Resources

- [π₀.₅ Paper](https://www.physicalintelligence.company/blog/pi05)
- [LeRobot Docs](https://github.com/huggingface/lerobot)
- [LIBERO Example](examples/libero/README.md)
- [DROID Training](examples/droid/README_train.md)
- [Remote Inference](docs/remote_inference.md)

---

**Ready to train?** Make sure all items above are checked! 🚀
