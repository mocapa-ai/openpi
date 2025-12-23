"""
Quick script to inspect aligned HDF5 file structure.

Usage:
    cd ~/openpi/examples/custom_robot
    uv run python inspect_hdf5.py demo0.hdf5
"""

import h5py
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_hdf5.py <hdf5_file>")
    sys.exit(1)

hdf5_path = sys.argv[1]

if not Path(hdf5_path).exists():
    print(f"[ERROR] File not found: {hdf5_path}")
    sys.exit(1)

print("=" * 70)
print(f"Inspecting: {hdf5_path}")
print("=" * 70)

with h5py.File(hdf5_path, 'r') as f:
    print("\n[FILES] Top-level groups:")
    for key in f.keys():
        print(f"  - {key}")
    
    print("\n[CHECK] File attributes:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  {attr_name}: {attr_value}")
    
    # Check if it's the aligned format
    if 'pi05_compatible' in f.attrs:
        print(f"\n[OK] Pi0.5 compatible format detected!")
        print(f"   Task: {f.attrs.get('task', 'N/A')}")
        print(f"   Arm side: {f.attrs.get('arm_side', 'N/A')}")
        print(f"   State dim: {f.attrs.get('state_dim', 'N/A')}")
        print(f"   Action dim: {f.attrs.get('action_dim', 'N/A')}")
        print(f"   Frames: {f.attrs.get('n_frames', 'N/A')}")
        
        if 'data' in f and 'demo_0' in f['data']:
            demo = f['data']['demo_0']
            print("\n[DATA] Demo structure:")
            
            # Show qpos
            if 'obs' in demo and 'qpos' in demo['obs']:
                qpos = demo['obs']['qpos']
                print(f"  state (qpos): {qpos.shape} {qpos.dtype}")
            
            # Show action
            if 'action' in demo:
                action = demo['action']
                print(f"  actions: {action.shape} {action.dtype}")
            
            # Show images
            if 'obs' in demo and 'images' in demo['obs']:
                for img_name in demo['obs']['images'].keys():
                    img = demo['obs']['images'][img_name]
                    print(f"  image ({img_name}): {img.shape} {img.dtype}")
            
            # Show timestamps
            if 'timestamps' in demo:
                ts = demo['timestamps']
                print(f"  timestamps: {ts.shape} {ts.dtype}")
            
            # Show task
            if 'task' in demo.attrs:
                print(f"  task: \"{demo.attrs['task']}\"")
            
            print("\n[OK] This file is ready for conversion to LeRobot!")
    else:
        print("\n[WARNING] This doesn't appear to be an aligned Pi0.5 format")
        print("   Expected 'pi05_compatible' attribute")
        print("   You may need to run align_proprioception_pi.py first")
