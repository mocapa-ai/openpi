# Finetuning π₀.₅ with FastUMI Dataset for xArm6

This guide walks you through finetuning the π₀.₅ base model using the FastUMI dataset for deployment on an xArm6 robot arm (6DOF + 1 gripper).

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Obtain FastUMI Dataset](#step-1-obtain-fastumi-dataset)
- [Step 2: Convert FastUMI Data to LeRobot Format](#step-2-convert-fastumi-data-to-lerobot-format)
- [Step 3: Compute Normalization Statistics](#step-3-compute-normalization-statistics)
- [Step 4: Run Finetuning](#step-4-run-finetuning)
- [Step 5: Test Inference](#step-5-test-inference)
- [Step 6: Deploy on Real Robot](#step-6-deploy-on-real-robot)
- [Troubleshooting](#troubleshooting)

## Overview

**What you'll accomplish:**
- Convert FastUMI HDF5 data (TCP poses) → joint angles using IK with xArm6 URDF
- Transform data into LeRobot format for openpi training
- Finetune π₀.₅ base model on pick-and-place tasks
- Deploy the trained policy on a real xArm6 robot

**Key Design Decisions:**
- **Action Space**: Joint angles (6 joints + 1 gripper = 7D)
- **State Space**: Joint positions (7D, same as action)
- **Camera Setup**: 1 front camera (resized from 1920x1080 to 224x224)
- **Model**: π₀.₅ with flow matching head

## Prerequisites

### Hardware Requirements
- GPU with >22.5 GB VRAM (for LoRA finetuning) or >70 GB (for full finetuning)
- xArm6 robot arm with URDF file

### Software Requirements
- Ubuntu 22.04 (recommended)
- Python 3.11+
- openpi repository installed (see main README)
- HuggingFace account with access to FastUMI-Data (dataset is gated)

### Install Additional Dependencies

From the openpi root directory:

```bash
# Install IKPy for inverse kinematics
uv pip install ikpy scipy opencv-python

# Install HuggingFace CLI and login (if not already done)
uv pip install huggingface_hub datasets
huggingface-cli login
```

## Step 1: Obtain FastUMI Dataset

### 1.1 Request Access

1. Go to https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI-Data
2. Request access (the dataset is gated)
3. Wait for approval

### 1.2 Download Data

```bash
# Download the dataset
huggingface-cli download IPEC-COMMUNITY/FastUMI-Data \
    --repo-type dataset \
    --local-dir data/fastumi_raw
```

### 1.3 Verify Data Structure

Each HDF5 episode should contain:
- `observations/images/front`: (T, 1920, 1080, 3) uint8
- `observations/qpos`: (T, 7) float32 - TCP pose [x, y, z, qx, qy, qz, qw]
- `action`: (T, 7) float32

## Step 2: Convert FastUMI Data to LeRobot Format

### 2.1 Obtain xArm6 URDF

Get URDF from https://github.com/xArm-Developer/xarm_ros and place at:
`examples/fastumi/assets/xarm6_robot.urdf`

### 2.2 Configure Parameters

Edit `examples/fastumi/config.json`:

```json
{
  "urdf_path": "examples/fastumi/assets/xarm6_robot.urdf",
  "start_qpos": [0, 0, 0, 0, 0, 0, 0],
  "base_position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "base_orientation": {"roll": 0, "pitch": 0, "yaw": 0},
  "offset": {"x": 0.14565, "z": 0.1586},
  "gripper": {
    "marker_max": 566.0,
    "marker_min": 140.0,
    "gripper_max": 850.0
  }
}
```

### 2.3 Run Conversion

```bash
uv run examples/fastumi/convert_fastumi_to_lerobot.py \
    --data_dir data/fastumi_raw \
    --config_path examples/fastumi/config.json \
    --output_repo your_hf_username/fastumi_xarm6_pickplace
```

## Step 3: Compute Normalization Statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_fastumi_xarm6
```

Verify output:
```bash
cat checkpoints/pi05_fastumi_xarm6/norm_stats.json
```

## Step 4: Run Finetuning

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run scripts/train.py pi05_fastumi_xarm6 \
    --exp-name=fastumi_xarm6_v1 \
    --overwrite
```

Monitor training via console output and Weights & Biases.

## Step 5: Test Inference

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_fastumi_xarm6 \
    --policy.dir=checkpoints/pi05_fastumi_xarm6/fastumi_xarm6_v1/20000
```

Test with Python:
```python
import requests
import numpy as np

obs = {
    "observation/front_image": np.random.randint(0, 255, (224, 224, 3)).tolist(),
    "observation/joint_position": [0.0] * 7,
    "prompt": "pick up the red cube"
}

response = requests.post("http://localhost:8000/infer", json=obs)
print(response.json()["actions"])
```

## Step 6: Deploy on Real Robot

```bash
# Terminal 1: Policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_fastumi_xarm6 \
    --policy.dir=checkpoints/pi05_fastumi_xarm6/fastumi_xarm6_v1/20000

# Terminal 2: Robot interface
python examples/fastumi/deploy_xarm6.py \
    --robot_ip 192.168.1.XXX \
    --camera_id 0
```

**⚠️ Safety**: Keep E-stop accessible, start with reduced speeds!

## Troubleshooting

See detailed troubleshooting in the full README sections above.

## Next Steps

1. Collect your own xArm6 data with FastUMI system
2. Mix pre-training with custom data
3. Iterate on failure cases
4. Experiment with multi-task learning

## Resources

- [OpenPI Docs](../../README.md)
- [FastUMI](https://fastumi.com/)
- [xArm SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
