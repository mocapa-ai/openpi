#!/bin/bash
# Test conversion of a single trajectory to LeRobot format
#
# Usage:
#   cd ~/openpi/examples/custom_robot
#   ./test_conversion.sh

set -e

echo "=============================================================="
echo "Testing Single Trajectory Conversion to LeRobot"
echo "=============================================================="
echo ""

# Step 1: Check if demo0.hdf5 exists
if [ ! -f "demo0.hdf5" ]; then
    echo "❌ demo0.hdf5 not found in current directory"
    echo "   Please run align_proprioception_pi.py first"
    exit 1
fi

echo "✅ Found demo0.hdf5"
echo ""

# Step 2: Inspect the aligned HDF5 structure
echo "Step 1: Inspecting aligned HDF5 structure..."
echo "------------------------------------------------------------"
uv run python << 'EOF'
import h5py

with h5py.File("demo0.hdf5", 'r') as f:
    print("\n📁 File attributes:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  {attr_name}: {attr_value}")
    
    if 'pi05_compatible' in f.attrs and f.attrs['pi05_compatible']:
        print("\n✅ Pi0.5 compatible format detected!")
        
        demo = f['data']['demo_0']
        qpos = demo['obs']['qpos']
        action = demo['action']
        images = demo['obs']['images']['top']
        
        print(f"\n📊 Data shapes:")
        print(f"  State (qpos): {qpos.shape} - {qpos.dtype}")
        print(f"  Actions: {action.shape} - {action.dtype}")
        print(f"  Images: {images.shape} - {images.dtype}")
        print(f"  Task: {demo.attrs.get('task', 'N/A')}")
    else:
        print("\n❌ NOT a Pi0.5 aligned format!")
        print("   Run align_proprioception_pi.py first")
        exit(1)
EOF

echo ""
echo "Step 2: Converting to LeRobot format..."
echo "------------------------------------------------------------"

# Step 3: Run conversion (creates dataset with 1 episode)
uv run python convert_aligned_to_lerobot.py \
    --data_dir . \
    --output_name "test/single_trajectory" \
    --fps 10

echo ""
echo "Step 3: Verifying LeRobot dataset..."
echo "------------------------------------------------------------"

# Step 4: Verify the LeRobot dataset
uv run python << 'EOF'
import os
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

dataset_path = HF_LEROBOT_HOME / "test/single_trajectory"

print(f"\n📦 LeRobot dataset location:")
print(f"  {dataset_path}")

if not dataset_path.exists():
    print("\n❌ Dataset not found!")
    exit(1)

print(f"\n✅ Dataset created!")
print(f"\n📁 Dataset structure:")
for item in sorted(dataset_path.rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(dataset_path)
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  {rel_path} ({size_mb:.2f} MB)")

# Load and inspect the dataset
print(f"\n📊 Loading dataset...")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("test/single_trajectory")

print(f"\n✅ Dataset loaded successfully!")
print(f"  Total episodes: {len(dataset.episodes)}")
print(f"  Total frames: {len(dataset)}")
print(f"  FPS: {dataset.fps}")
print(f"  Features: {list(dataset.features.keys())}")

# Check first frame
print(f"\n🔍 First frame inspection:")
first_frame = dataset[0]
for key, value in first_frame.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {value}")

print(f"\n✅ All checks passed!")
print(f"\n💡 Next steps:")
print(f"  1. Collect more trajectories (aim for 50+)")
print(f"  2. Run batch alignment")
print(f"  3. Convert all to one dataset")
print(f"  4. Compute norm stats")
print(f"  5. Train Pi0.5!")
EOF

echo ""
echo "=============================================================="
echo "Test Complete!"
echo "=============================================================="
