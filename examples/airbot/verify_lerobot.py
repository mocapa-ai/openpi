"""
Quick script to verify LeRobot dataset after conversion.

Usage:
    cd ~/openpi
    uv run python examples/custom_robot/verify_lerobot.py test/single_trajectory
"""

import sys
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

if len(sys.argv) < 2:
    print("Usage: python verify_lerobot.py <dataset_name>")
    print("Example: python verify_lerobot.py test/single_trajectory")
    sys.exit(1)

dataset_name = sys.argv[1]
dataset_path = HF_LEROBOT_HOME / dataset_name

print("=" * 70)
print(f"Verifying LeRobot Dataset: {dataset_name}")
print("=" * 70)

# Check if dataset exists
if not dataset_path.exists():
    print(f"\n[ERROR] Dataset not found at: {dataset_path}")
    print(f"\nExpected location: {HF_LEROBOT_HOME}")
    sys.exit(1)

print(f"\n[OK] Dataset found at: {dataset_path}")

# Show file structure
print(f"\n[FILES] Dataset files:")
for item in sorted(dataset_path.rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(dataset_path)
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  {rel_path} ({size_mb:.2f} MB)")

# Load dataset
print(f"\n[DATA] Loading dataset...")
try:
    dataset = LeRobotDataset(dataset_name)
    print(f"[OK] Dataset loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Show dataset info
print(f"\n[STATS] Dataset statistics:")
print(f"  Total episodes: {len(dataset.episodes)}")
print(f"  Total frames: {len(dataset)}")
print(f"  FPS: {dataset.fps}")
print(f"  Robot type: {dataset.robot_type}")

print(f"\n[KEYS] Features:")
for feature_name, feature_info in dataset.features.items():
    if 'shape' in feature_info:
        print(f"  {feature_name}: shape={feature_info['shape']}, dtype={feature_info['dtype']}")
    else:
        print(f"  {feature_name}: {feature_info}")

# Inspect first frame
print(f"\n[CHECK] First frame:")
first_frame = dataset[0]
for key, value in first_frame.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        # Show min/max for numeric data
        if key in ['state', 'actions']:
            print(f"    -> min={value.min():.4f}, max={value.max():.4f}")
    else:
        print(f"  {key}: {value}")

# Show episode info
print(f"\n[PACKAGE] Episode info:")
for ep_idx, episode in enumerate(dataset.episodes):
    print(f"  Episode {ep_idx}:")
    print(f"    Length: {episode['length']} frames")
    print(f"    Task: {episode.get('task', 'N/A')}")

print(f"\n[OK] Verification complete!")
print(f"\n[TIP] This dataset is ready for training!")
print(f"\nNext steps:")
print(f"  1. Collect more trajectories (aim for 50+)")
print(f"  2. Convert all to one larger dataset")
print(f"  3. Run: uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05")
print(f"  4. Run: uv run scripts/train.py custom_robot_pi05 --exp-name=test_lora")
