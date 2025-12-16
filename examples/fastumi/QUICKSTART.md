# FastUMI xArm6 Finetuning - Quick Reference

This document provides a condensed reference for finetuning π₀.₅ with FastUMI data.

## File Structure Created

```
openpi/
├── examples/fastumi/
│   ├── README.md                          # Full step-by-step guide
│   ├── config.json                        # Conversion parameters (URDF, calibration)
│   ├── convert_fastumi_to_lerobot.py      # HDF5 → LeRobot conversion script
│   ├── deploy_xarm6.py                    # Robot deployment script
│   └── assets/                            # Place xarm6_robot.urdf here
├── src/openpi/policies/
│   └── fastumi_policy.py                  # Input/output transforms
└── src/openpi/training/
    └── config.py                          # Added pi05_fastumi_xarm6 configs
```

## Quick Start Commands

### 1. Setup (one-time)
```bash
# Install dependencies
uv pip install ikpy scipy opencv-python xarm-python-sdk

# Login to HuggingFace
huggingface-cli login
```

### 2. Get FastUMI Data
```bash
# Download from HuggingFace (requires access approval)
huggingface-cli download IPEC-COMMUNITY/FastUMI-Data \
    --repo-type dataset \
    --local-dir data/fastumi_raw
```

### 3. Get xArm6 URDF
```bash
# Clone xArm ROS repo and copy URDF
git clone https://github.com/xArm-Developer/xarm_ros.git /tmp/xarm_ros
cp /tmp/xarm_ros/xarm_description/urdf/xarm6_robot.urdf \
   examples/fastumi/assets/
```

### 4. Configure Conversion
Edit `examples/fastumi/config.json` if needed:
- `urdf_path`: Path to your URDF
- `start_qpos`: Initial joint configuration
- `base_position/orientation`: T265 sensor mounting
- `offset`: T265 to TCP transformation
- `gripper`: ArUco marker calibration

### 5. Convert Data
```bash
uv run examples/fastumi/convert_fastumi_to_lerobot.py \
    --data_dir data/fastumi_raw \
    --config_path examples/fastumi/config.json \
    --output_repo USERNAME/fastumi_xarm6_pickplace
```

Replace `USERNAME` with your HuggingFace username.

### 6. Update Training Config
Edit `src/openpi/training/config.py`, line ~912:
```python
repo_id="USERNAME/fastumi_xarm6_pickplace",  # Your converted dataset
```

### 7. Compute Norm Stats
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_fastumi_xarm6
```

### 8. Train
```bash
# Full finetuning (requires 70GB+ GPU)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py pi05_fastumi_xarm6 \
    --exp-name=fastumi_v1 --overwrite

# OR: LoRA finetuning (requires 22.5GB+ GPU)
uv run scripts/train.py pi05_fastumi_xarm6_lora \
    --exp-name=fastumi_v1_lora --overwrite
```

### 9. Test Inference
```bash
# Start policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_fastumi_xarm6 \
    --policy.dir=checkpoints/pi05_fastumi_xarm6/fastumi_v1/20000
```

Test in Python:
```python
import requests, numpy as np
obs = {
    "observation/front_image": np.random.randint(0, 255, (224, 224, 3)).tolist(),
    "observation/joint_position": [0.0] * 7,
    "prompt": "pick up cube"
}
r = requests.post("http://localhost:8000/infer", json=obs)
print(r.json()["actions"])
```

### 10. Deploy on Robot
```bash
# Terminal 1: Policy server (same as step 9)

# Terminal 2: Robot deployment
python examples/fastumi/deploy_xarm6.py \
    --robot_ip 192.168.1.XXX \
    --camera_id 0 \
    --prompt "pick up the red cube"
```

⚠️ **Safety**: Keep E-stop accessible!

## Key Concepts

### Data Format
- **Input**: FastUMI HDF5 with TCP poses (x,y,z,qx,qy,qz,qw)
- **Converted**: LeRobot format with joint angles (6 joints + 1 gripper)
- **Process**: IK converts TCP → joints using your URDF

### Model Architecture
- **Base**: π₀.₅ (10k+ hours of robot data)
- **Action Space**: 7D (6 joints + 1 binary gripper)
- **State Space**: 7D (same as action)
- **Images**: 1 front camera @ 224x224
- **Action Horizon**: 10 timesteps

### Training Configs
- `pi05_fastumi_xarm6`: Full finetuning (slower, better performance)
- `pi05_fastumi_xarm6_lora`: LoRA finetuning (faster, less memory)

## Common Issues

| Issue | Solution |
|-------|----------|
| IK fails | Check URDF path, base_position/orientation in config.json |
| NaN loss | Check norm_stats.json for extreme values, reduce LR |
| OOM | Use LoRA config, reduce batch size, increase XLA_PYTHON_CLIENT_MEM_FRACTION |
| Gripper wrong | Verify ArUco marker calibration in config.json |
| Jerky motion | Increase action_smoothing in deploy script |

## File Locations

- **Converted dataset**: `~/.cache/lerobot/USERNAME/fastumi_xarm6_pickplace/`
- **Checkpoints**: `checkpoints/pi05_fastumi_xarm6/EXP_NAME/ITER/`
- **Norm stats**: `checkpoints/pi05_fastumi_xarm6/norm_stats.json`
- **Logs**: Console output + W&B (if configured)

## Training Time Estimates

| Dataset Size | Hardware | Full Finetune | LoRA Finetune |
|--------------|----------|---------------|---------------|
| 100 episodes | RTX 4090 | N/A (OOM)     | 3-5 hours     |
| 100 episodes | A100 80GB | 4-6 hours    | 2-3 hours     |
| 500 episodes | A100 80GB | 12-18 hours  | 6-8 hours     |

## Next Steps After Deployment

1. **Evaluate**: Test on multiple task variations
2. **Collect failures**: Note where the model fails
3. **Collect your own data**: Use FastUMI system with your xArm6
4. **Mix datasets**: Combine pre-trained data + your custom data
5. **Retrain**: Iterate to improve performance

## Resources

- **Full Guide**: `examples/fastumi/README.md`
- **FastUMI Project**: https://fastumi.com/
- **xArm SDK Docs**: https://github.com/xArm-Developer/xArm-Python-SDK
- **OpenPI Docs**: README.md in repo root

## Support

- Check `examples/fastumi/README.md` for detailed troubleshooting
- Review similar examples: `examples/droid/`, `examples/aloha_real/`
- File issues on openpi GitHub repository

---

**Created**: 2024-12-16 for xArm6 with FastUMI dataset
**OpenPI Version**: Compatible with latest main branch
