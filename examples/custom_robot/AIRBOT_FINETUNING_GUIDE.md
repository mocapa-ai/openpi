# Fine-Tuning π₀.₅ on AirBot Play

This guide shows how to fine-tune π₀.₅ on your AirBot Play robot data, following the same workflow as the main README.

## AirBot Robot Specifications

- **Arm**: 6-DOF AirBot Play
- **Hand**: 6-DOF Revo2 dexterous hand  
- **Total Action Dimensions**: 12
- **Camera**: Single top-down RGB camera
- **Action Representation**: Absolute joint positions (converted to deltas for training)

## Prerequisites

Your data should already be:
1. ✅ Collected with `data_logger_worker.py`
2. ✅ Aligned with `align_proprioception_pi.py`
3. ✅ Converted to LeRobot format with `convert_aligned_to_lerobot.py`
4. ✅ Verified with `verify_dataset_for_pi05.py`

## Fine-Tuning Workflow

### 1. Convert your data to a LeRobot dataset

If you haven't already, convert your aligned HDF5 files to LeRobot format:

```bash
cd ~/openpi
uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir ~/motion_retargeting/pi_test/aligned_pi \
    --output_name test_pi_finetune \
    --fps 30
```

This creates a LeRobot dataset at `~/.cache/huggingface/lerobot/test_pi_finetune`.

### 2. Defining training configs

The training configs for AirBot are already defined in [`src/openpi/training/config.py`](../../src/openpi/training/config.py):

- **`LeRobotAirBotDataConfig`**: Defines how to process AirBot data from LeRobot dataset
  - Maps single `image` camera to model's multi-camera format
  - Converts absolute joint positions to delta actions
  - Handles 12D state and action dimensions

- **Training Configs**:
  - `airbot_pi05`: Main LoRA fine-tuning config (recommended)
  - `airbot_pi05_full`: Full fine-tuning (requires more GPU memory)
  - `airbot_pi05_debug`: Quick test with fake data

The policy transforms are defined in [`src/openpi/policies/airbot_policy.py`](../../src/openpi/policies/airbot_policy.py):

- **`AirBotInputs`**: Converts robot observations to model format
- **`AirBotOutputs`**: Converts model predictions to robot actions

### 3. Compute normalization statistics

Before training, compute the normalization statistics for your dataset:

```bash
cd ~/openpi
uv run scripts/compute_norm_stats.py --config-name airbot_pi05
```

This analyzes your dataset and saves normalization stats (mean/std) for states and actions.

### 4. Start training

Now kick off training:

```bash
cd ~/openpi

# LoRA fine-tuning (recommended - requires ~22.5GB GPU memory)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05 --exp-name=airbot_experiment_1 --overwrite

# Full fine-tuning (requires ~70GB GPU memory)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05_full --exp-name=airbot_full_1 --overwrite
```

The `--overwrite` flag overwrites existing checkpoints if you rerun with the same experiment name.

**Training progress** is logged to:
- Console output
- Weights & Biases dashboard (project: `openpi`)
- Checkpoints saved to `checkpoints/airbot_pi05/<exp-name>/`

**Training metrics:**
- Loss curves
- Evaluation metrics every 1,000 steps
- Checkpoints saved every 5,000 steps

### 5. Monitor training

You can monitor training in real-time:

```bash
# Watch console output (shows loss, step time, etc.)

# Or visit Weights & Biases:
# https://wandb.ai/<your-username>/openpi
```

**Expected timeline:**
- **20,000 steps** takes ~4-8 hours depending on GPU
- **Evaluation** runs every 1,000 steps
- **Checkpoints** saved at steps 5000, 10000, 15000, 20000

### 6. Run inference

Once training completes, run inference by spinning up a policy server:

```bash
cd ~/openpi

# Use the final checkpoint (step 20000)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/airbot_experiment_1/20000
```

The server listens on port 8000. You can then query it from your robot runtime code.

## Configuration Details

### Dataset Configuration

```python
class LeRobotAirBotDataConfig(DataConfigFactory):
    repo_id: str = "test_pi_finetune"  # Your LeRobot dataset name
    
    # Key mappings:
    "observation/image" -> "image"     # Single camera
    "observation/state" -> "state"     # 12D joint positions
    "actions" -> "actions"             # 12D actions
    "prompt" -> "prompt"               # Language instruction
    
    # Action conversion:
    # Absolute positions -> Delta actions (all 12 dimensions)
```

### Training Hyperparameters

| Config | `airbot_pi05` | `airbot_pi05_full` |
|--------|---------------|-------------------|
| Training Mode | LoRA | Full Fine-tuning |
| Batch Size | 32 | 16 |
| Learning Rate | 3e-5 | 1e-5 |
| Max Steps | 20,000 | 20,000 |
| GPU Memory | ~22.5GB | ~70GB |
| Training Time | ~4-8 hours | ~8-16 hours |

### Model Configuration

```python
model=pi0_config.Pi0Config(
    model_type=ModelType.PI05,    # π₀.₅ architecture
    action_dim=12,                # 6 arm + 6 hand
    action_horizon=50,            # Predict 50 steps ahead
    max_token_len=256,            # Language + vision tokens
    dtype="bfloat16",             # Memory efficient
)
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1**: Use LoRA fine-tuning instead of full fine-tuning
```bash
uv run scripts/train.py airbot_pi05 --exp-name=my_exp  # Uses LoRA
```

**Solution 2**: Reduce batch size
```bash
# Edit config.py or override via CLI:
uv run scripts/train.py airbot_pi05 --exp-name=my_exp --batch-size=16
```

**Solution 3**: Enable gradient checkpointing (if available)

### Dataset Not Found

```bash
# Check dataset exists:
ls ~/.cache/huggingface/lerobot/test_pi_finetune

# If missing, re-run conversion:
cd ~/openpi
uv run examples/custom_robot/convert_aligned_to_lerobot.py \
    --data_dir ~/motion_retargeting/pi_test/aligned_pi \
    --output_name test_pi_finetune
```

### Normalization Stats Missing

```bash
# Recompute stats:
cd ~/openpi
uv run scripts/compute_norm_stats.py --config-name airbot_pi05
```

### Action Dimension Mismatch

**Problem**: Error says action dimension doesn't match

**Solution**: Verify your dataset has 12-dimensional actions:
```bash
cd ~/openpi
uv run python examples/custom_robot/verify_dataset_for_pi05.py test_pi_finetune
# Should show: "Action shape: torch.Size([12])"
```

### Training Loss Not Decreasing

**Possible causes**:
1. Learning rate too high/low → Try adjusting `peak_lr`
2. Not enough data → Aim for 50+ episodes (1000+ frames per task)
3. Data quality issues → Check alignment errors in `align_proprioception_pi.py` output

## Quick Test (Debug Mode)

To quickly test your setup without training on real data:

```bash
cd ~/openpi

# Test with fake data (~1 minute)
uv run scripts/train.py airbot_pi05_debug --exp-name=test_run --overwrite
```

This runs 100 steps with fake data to verify:
- ✅ Configuration loads correctly
- ✅ Model initializes
- ✅ Training loop works
- ✅ Checkpoints save properly

## Dataset Requirements

For good performance, aim for:
- **Minimum**: 10 episodes, ~2000 frames
- **Recommended**: 50+ episodes, ~10,000 frames
- **Ideal**: 100+ episodes, ~20,000 frames

**Episode quality matters more than quantity!**
- Each episode should be a successful demonstration
- Use `clip_hdf5.py` to remove failed attempts
- Verify alignment quality (check error statistics)

## Next Steps

After training:
1. **Evaluate checkpoints**: Test different checkpoints (5k, 10k, 15k, 20k steps)
2. **Fine-tune hyperparameters**: Adjust learning rate, batch size if needed
3. **Collect more data**: If performance is insufficient
4. **Deploy to robot**: Use the policy server for real-time inference

## Related Scripts

**Data Collection & Processing:**
- `~/motion_retargeting/scripts/align_proprioception_pi.py` - Align data to camera timestamps
- `~/openpi/examples/custom_robot/convert_aligned_to_lerobot.py` - Convert to LeRobot format
- `~/openpi/examples/custom_robot/verify_dataset_for_pi05.py` - Verify dataset

**Training:**
- `~/openpi/scripts/compute_norm_stats.py` - Compute normalization statistics
- `~/openpi/scripts/train.py` - Main training script
- `~/openpi/scripts/serve_policy.py` - Policy server for inference

**Configuration:**
- `~/openpi/src/openpi/training/config.py` - Training configurations
- `~/openpi/src/openpi/policies/airbot_policy.py` - Policy transforms

---

## Summary Commands

```bash
cd ~/openpi

# 1. Compute stats (one time)
uv run scripts/compute_norm_stats.py --config-name airbot_pi05

# 2. Train (LoRA fine-tuning)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py airbot_pi05 \
    --exp-name=airbot_v1 --overwrite

# 3. Serve policy
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/airbot_v1/20000
```

**You're ready to train!** 🚀
