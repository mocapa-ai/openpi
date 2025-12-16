# π₀.₅ Custom Robot Training - Quick Reference

## 📦 What You Have

### Robot Specs
- **Arm**: 6 DOF
- **Hand**: 6 DOF (dexterous)
- **Total Actions**: 12
- **Data Format**: HDF5

### Files Created
- **3 Documentation files** (comprehensive guides)
- **4 Python scripts** (conversion, training, inference)
- **1 Modified config** (registered your setup)

## 🚀 Quick Start (5 Steps)

### 1. Update Configuration
```python
# Edit: src/openpi/training/custom_robot_config.py

# Line 49: Set your dataset name
repo_id: str = "your_username/robot_dataset"

# Line 54: Set action space
use_absolute_actions: bool = False  # True if positions, False if velocities

# Line 126: Verify DOF
action_dim=12  # Change if different
```

### 2. Convert HDF5 → LeRobot
```bash
# Install dependencies
uv pip install h5py opencv-python

# Convert data
uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
    --data_dir /path/to/hdf5/files \
    --output_name "your_username/robot_dataset" \
    --fps 10 \
    --state_dim 12 \
    --action_dim 12
```

### 3. Compute Normalization Stats
```bash
uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05
```

### 4. Train
```bash
# Set GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Start training (LoRA - recommended)
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=my_robot_lora \
    --overwrite
```

### 5. Inference
```bash
# Terminal 1: Start server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=custom_robot_pi05 \
    --policy.dir=checkpoints/custom_robot_pi05/my_robot_lora/20000

# Terminal 2: Test
python examples/custom_robot/inference_example.py
```

## 📝 HDF5 Format Required

```
your_data/
├── episode_0.h5
├── episode_1.h5
└── ...

Each file contains:
├── observations/
│   ├── images/
│   │   ├── camera_0: (T, H, W, 3) uint8
│   │   └── wrist_camera: (T, H, W, 3) uint8 [optional]
│   └── state: (T, 12) float32
├── actions: (T, 12) float32
└── language_instruction: str
```

## ⚙️ Key Customization Points

### Different DOF (e.g., 18 instead of 12)
```python
# 1. convert_hdf5_to_lerobot.py
--state_dim 18 --action_dim 18

# 2. custom_robot_policy.py (line 176)
return {"actions": np.asarray(data["actions"][:, :18])}

# 3. custom_robot_config.py (line 126)
action_dim=18,
```

### Different Cameras
```python
# custom_robot_policy.py (CustomRobotInputs)

# Only 1 camera (third-person):
"image": {
    "base_0_rgb": camera_0,
    "left_wrist_0_rgb": np.zeros_like(camera_0),
    "right_wrist_0_rgb": np.zeros_like(camera_0),
},
"image_mask": {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": np.False_,
    "right_wrist_0_rgb": np.False_,
}

# 3 cameras (no masking):
# Set all image_mask values to np.True_
```

### Absolute vs Delta Actions
```python
# custom_robot_config.py (line 54)

# Your HDF5 has absolute joint positions:
use_absolute_actions: bool = True
delta_action_mask = _transforms.make_bool_mask(11, -1)  # Convert first 11 to deltas

# Your HDF5 has velocities/deltas:
use_absolute_actions: bool = False
# No delta conversion needed
```

## 💾 Hardware Requirements

| Mode | GPU Memory | Example GPU |
|------|-----------|-------------|
| LoRA Training | 22.5 GB | RTX 4090 |
| Full Training | 70+ GB | A100 80GB |
| Inference | 8+ GB | RTX 4090 |

## 📊 Training Options

### LoRA (Recommended)
```bash
uv run scripts/train.py custom_robot_pi05 --exp-name=lora_exp
```
- Memory: ~22.5 GB
- Speed: Fast
- Quality: Very good

### Full Fine-Tuning
```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=full_exp \
    --use-lora=false
```
- Memory: ~70 GB
- Speed: Slower
- Quality: Best

### Multi-GPU
```bash
uv run scripts/train.py custom_robot_pi05 \
    --exp-name=multigpu_exp \
    --fsdp-devices=4
```
- Distributes across 4 GPUs
- Reduces per-GPU memory
- Slightly slower

## 🐛 Common Issues

### Out of Memory
```bash
# Use LoRA (not full fine-tuning)
# Reduce batch size
--batch-size=16

# Enable FSDP
--fsdp-devices=4

# Increase GPU memory allocation
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

### Loss Diverging
```python
# Reduce learning rate in custom_robot_config.py
lr_schedule=_config._optimizer.CosineDecaySchedule(
    peak_lr=1e-5,  # Lower from 3e-5
)
```

### Key Not Found
```python
# Check camera names in custom_robot_policy.py match LeRobot dataset
# Verify repack_transform in custom_robot_config.py
```

### Poor Generalization
- Collect more diverse data (500+ episodes recommended)
- Increase training steps (50k for large datasets)
- Use data augmentation (built-in to π₀.₅)

## 📚 Documentation Files

1. **ROBOT_FINETUNING_GUIDE.md** - Complete guide (15KB)
   - Prerequisites, installation
   - Data requirements
   - Training configuration
   - Troubleshooting

2. **QUICK_START_CHECKLIST.md** - Step-by-step (6.5KB)
   - Pre-training checklist
   - Training commands
   - Post-training steps

3. **FILE_SUMMARY.md** - File overview (9KB)
   - All files explained
   - Customization examples
   - Quick workflow

4. **examples/custom_robot/README.md** - Example docs (8KB)
   - Quick start
   - Customization guide
   - Training tips

## 🎯 Training Duration

| Dataset Size | Training Time | Steps |
|--------------|---------------|-------|
| 50-100 episodes | 1-2 hours | 10k |
| 500 episodes | 5-10 hours | 20k |
| 1000+ episodes | 10-20 hours | 50k |

*Based on single A100 80GB GPU

## 📈 Monitoring Training

### Console Output
- Loss should decrease
- Steps per second stable
- No NaN/Inf values

### Weights & Biases
```bash
# One-time setup
wandb login

# Automatically logs during training
# View at: https://wandb.ai
```

### Checkpoints
```
checkpoints/custom_robot_pi05/my_experiment/
├── 5000/   # Checkpoint at step 5000
├── 10000/
├── 15000/
└── 20000/  # Use this for inference
```

## 🔧 Data Collection Tips

1. **Minimum**: 50-100 episodes
2. **Recommended**: 500-1000 episodes
3. **Diversity**: Vary positions, lighting, objects
4. **Success rate**: Include partial successes
5. **Frame rate**: 10-15 Hz (matches π₀.₅ pre-training)
6. **Language**: Clear task descriptions

## 🎓 Next Steps

1. ✅ Read ROBOT_FINETUNING_GUIDE.md
2. ✅ Follow QUICK_START_CHECKLIST.md
3. ✅ Convert your HDF5 data
4. ✅ Update configuration files
5. ✅ Compute normalization stats
6. ✅ Start training
7. ✅ Test inference
8. ✅ Deploy on robot

## 💬 Getting Help

- **Issues**: [GitHub Issues](https://github.com/Physical-Intelligence/openpi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Physical-Intelligence/openpi/discussions)
- **Examples**: See LIBERO and DROID for reference

## ✨ Key Points

- **Yes, you can convert HDF5 to LeRobot!**
- **Start with LoRA training** (less memory)
- **Collect diverse data** (500+ episodes ideal)
- **Use the guides** (comprehensive documentation)
- **Test incrementally** (start small, scale up)

---

**Ready to train?** Follow the 5-step Quick Start above! 🚀
