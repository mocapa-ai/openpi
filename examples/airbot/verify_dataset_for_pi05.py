#!/usr/bin/env python3
"""
Comprehensive verification script for LeRobot datasets before Pi0.5 training.

This script performs detailed checks to ensure your dataset is correctly formatted
and ready for OpenPI/Pi0.5 training.

Usage:
    cd ~/openpi
    uv run python examples/airbot/verify_dataset_for_pi05.py your_username/robot_dataset
    
    # With visualization
    uv run python examples/airbot/verify_dataset_for_pi05.py your_username/robot_dataset --visualize
"""

import argparse
import sys
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print formatted section."""
    print(f"\n[{text}]")


def verify_dataset(dataset_name: str, visualize: bool = False):
    """
    Comprehensive verification of LeRobot dataset for Pi0.5 training.
    
    Args:
        dataset_name: Name of dataset (e.g., "your_username/robot_dataset")
        visualize: Whether to save sample images for inspection
        
    Returns:
        bool: True if dataset passes all checks
    """
    print_header(f"Pi0.5 Dataset Verification: {dataset_name}")
    
    dataset_path = HF_LEROBOT_HOME / dataset_name
    
    # ===== Check 1: Dataset exists =====
    print_section("CHECK 1: Dataset Location")
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print(f"\nAvailable datasets in {HF_LEROBOT_HOME}:")
        for user_dir in HF_LEROBOT_HOME.glob("*/"):
            if user_dir.is_dir():
                for dataset_dir in user_dir.glob("*/"):
                    if dataset_dir.is_dir():
                        print(f"  - {user_dir.name}/{dataset_dir.name}")
        return False
    
    print(f"[OK] Dataset found at: {dataset_path}")
    
    # Show file structure
    print(f"\n[FILES] Dataset structure:")
    important_files = ['meta/info.json', 'meta/episodes.jsonl', 'meta/stats.json', 
                       'meta/tasks.jsonl', 'data/']
    for pattern in important_files:
        matches = list(dataset_path.glob(pattern))
        if matches:
            for match in matches[:5]:  # Show first 5
                rel_path = match.relative_to(dataset_path)
                if match.is_file():
                    size_mb = match.stat().st_size / (1024 * 1024)
                    print(f"  [OK] {rel_path} ({size_mb:.2f} MB)")
                else:
                    print(f"  [OK] {rel_path}/ (directory)")
        else:
            if pattern == 'meta/stats.json':
                print(f"  [WARNING] {pattern} not found (will be computed during training)")
            else:
                print(f"  [ERROR] {pattern} not found")
    
    # ===== Check 2: Load dataset =====
    print_section("CHECK 2: Dataset Loading")
    try:
        dataset = LeRobotDataset(dataset_name)
        print(f"[OK] Dataset loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== Check 3: Basic dataset info =====
    print_section("CHECK 3: Dataset Statistics")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")
    robot_type = getattr(dataset, 'robot_type', 'N/A')
    print(f"  Robot type: {robot_type}")
    
    if dataset.num_episodes == 0:
        print("[ERROR] No episodes found in dataset!")
        return False
    
    if len(dataset) < 100:
        print(f"[WARNING] WARNING: Only {len(dataset)} frames. Recommend at least 1000+ for training.")
    
    print(f"\n[DATA] Episode lengths:")
    try:
        for ep_idx, episode in enumerate(dataset.episodes):
            if ep_idx >= 10:  # Only show first 10
                break
            ep_length = episode['length']
            print(f"  Episode {ep_idx}: {ep_length} frames")
            if ep_length < 10:
                print(f"    [WARNING] Very short episode (< 10 frames)")
        
        if dataset.num_episodes > 10:
            print(f"  ... and {dataset.num_episodes - 10} more episodes")
    except Exception as e:
        print(f"  [WARNING] Could not read episode info: {e}")
    
    # ===== Check 4: Required features for Pi0.5 =====
    print_section("CHECK 4: Required Features for Pi0.5")
    
    required_features = ['state', 'actions']
    optional_features = ['task', 'language_instruction']
    
    print(f"\n[KEYS] Dataset features:")
    for feature_name, feature_info in dataset.features.items():
        print(f"  - {feature_name}:")
        print(f"      dtype: {feature_info.get('dtype', 'N/A')}")
        print(f"      shape: {feature_info.get('shape', 'N/A')}")
    
    # Check required features
    missing_required = []
    for feature in required_features:
        if feature not in dataset.features:
            missing_required.append(feature)
            print(f"[ERROR] Missing required feature: {feature}")
        else:
            print(f"[OK] Has required feature: {feature}")
    
    if missing_required:
        print(f"\n[ERROR] Missing required features: {missing_required}")
        return False
    
    # Check camera features
    # Detect camera/image features. New converter creates `image_<camera>`
    # keys (e.g., image_top, image_wrist). Accept legacy 'image' or any
    # feature containing 'image' or 'camera'.
    camera_features = [k for k in dataset.features.keys()
                       if k.lower().startswith('image_') or 'image' in k.lower() or 'camera' in k.lower()]

    if not camera_features:
        print(f"[ERROR] No camera/image features found!")
        print(f"   Pi0.5 requires at least one camera view (e.g. image_top or image_wrist)")
        return False

    print(f"\n[OK] Found {len(camera_features)} camera/image feature(s): {camera_features}")
    
    # ===== Check 5: Data validation =====
    print_section("CHECK 5: Data Validation")
    
    try:
        # Sample first frame
        sample = dataset[0]
        print(f"\n[PACKAGE] First frame data:")
        
        # Check language/task (now that we have sample)
        task_in_features = any(f in dataset.features for f in optional_features)
        task_in_sample = 'task' in sample
        has_language = task_in_features or task_in_sample
        
        if task_in_sample or task_in_features:
            print(f"[OK] Has language instructions")
        else:
            print(f"[WARNING] WARNING: No language instructions found. Pi0.5 requires task descriptions.")
        
        # Check state
        if 'state' in sample:
            state = sample['state']
            # Convert torch tensor to numpy if needed
            if hasattr(state, 'numpy'):
                state_np = state.cpu().numpy() if hasattr(state, 'cpu') else state.numpy()
            else:
                state_np = state
            
            print(f"\n  State:")
            print(f"    Shape: {state.shape}")
            print(f"    Dtype: {state.dtype}")
            print(f"    Range: [{float(state_np.min()):.4f}, {float(state_np.max()):.4f}]")
            print(f"    Mean: {float(state_np.mean()):.4f}")
            print(f"    Std: {float(state_np.std()):.4f}")
            
            # Check for NaN or Inf
            if np.any(np.isnan(state_np)):
                print(f"    [ERROR] Contains NaN values!")
            if np.any(np.isinf(state_np)):
                print(f"    [ERROR] Contains Inf values!")
            if np.all(state_np == 0):
                print(f"    [WARNING] WARNING: All zeros!")
        
        # Check actions
        if 'actions' in sample:
            actions = sample['actions']
            # Convert torch tensor to numpy if needed
            if hasattr(actions, 'numpy'):
                actions_np = actions.cpu().numpy() if hasattr(actions, 'cpu') else actions.numpy()
            else:
                actions_np = actions
            
            print(f"\n  Actions:")
            print(f"    Shape: {actions.shape}")
            print(f"    Dtype: {actions.dtype}")
            print(f"    Range: [{float(actions_np.min()):.4f}, {float(actions_np.max()):.4f}]")
            print(f"    Mean: {float(actions_np.mean()):.4f}")
            print(f"    Std: {float(actions_np.std()):.4f}")
            
            if np.any(np.isnan(actions_np)):
                print(f"    [ERROR] Contains NaN values!")
            if np.any(np.isinf(actions_np)):
                print(f"    [ERROR] Contains Inf values!")
            if np.all(actions_np == 0):
                print(f"    [WARNING] WARNING: All zeros!")
        
        # Check dimension consistency
        if 'state' in sample and 'actions' in sample:
            state_shape = sample['state'].shape
            actions_shape = sample['actions'].shape
            if state_shape != actions_shape:
                print(f"\n  [WARNING] WARNING: State and action dimensions differ:")
                print(f"     State: {state_shape}")
                print(f"     Actions: {actions_shape}")
        
        # Check cameras (support multiple views like image_top, image_wrist)
        for cam_name in camera_features:
            if cam_name in sample:
                image = sample[cam_name]
                # Convert torch tensor to numpy if needed
                if hasattr(image, 'numpy'):
                    image_np = image.cpu().numpy() if hasattr(image, 'cpu') else image.numpy()
                else:
                    image_np = image
                
                print(f"\n  Camera '{cam_name}':")
                print(f"    Shape: {image.shape}")
                print(f"    Dtype: {image.dtype}")
                print(f"    Range: [{float(image_np.min()):.3f}, {float(image_np.max()):.3f}]")
                
                # Check and convert image format
                # LeRobot often stores images as (C, H, W) in range [0, 1]
                if len(image_np.shape) == 3:
                    if image_np.shape[0] == 3 or image_np.shape[0] == 1:
                        # Likely (C, H, W) format, convert to (H, W, C)
                        print(f"    Format: (C, H, W) - converting to (H, W, C) for visualization")
                        image_np = np.transpose(image_np, (1, 2, 0))
                    
                    if image_np.shape[-1] != 3 and image_np.shape[-1] != 1:
                        print(f"    [WARNING] WARNING: Unexpected number of channels: {image_np.shape[-1]}")
                else:
                    print(f"    [WARNING] WARNING: Unexpected image dimensions: {image_np.shape}")
                
                # Check if normalized [0, 1] and convert to [0, 255] for visualization
                if image_np.max() <= 1.0:
                    print(f"    Normalized: [0, 1] range (common for LeRobot)")
                    image_np_vis = (image_np * 255).astype(np.uint8)
                else:
                    image_np_vis = image_np.astype(np.uint8)
                
                # Check image validity
                if np.all(image_np == 0):
                    print(f"    [ERROR] All pixels are zero!")
                elif np.all(image_np == image_np.max()):
                    print(f"    [WARNING] All pixels are same value!")
                else:
                    print(f"    [OK] Image data looks valid")
                
                # Save sample image if requested
                if visualize:
                    try:
                        import matplotlib.pyplot as plt
                        # Name file with dataset and camera to avoid collisions
                        safe_cam = cam_name.replace('/', '_')
                        output_path = f"verify_{safe_cam}_sample.png"
                        plt.figure(figsize=(8, 6))
                        if image_np_vis.shape[-1] == 1:
                            # Grayscale
                            plt.imshow(image_np_vis.squeeze(), cmap='gray')
                        else:
                            # RGB
                            plt.imshow(image_np_vis)
                        plt.title(f"{cam_name} - Frame 0")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(output_path)
                        plt.close()
                        print(f"    [SAVED] Saved sample to: {output_path}")
                    except Exception as e:
                        print(f"    [WARNING] Could not save image: {e}")
        
        # Check task
        if 'task' in sample:
            task = sample['task']
            print(f"\n  Task: '{task}'")
            if not task or task == "":
                print(f"    [WARNING] WARNING: Empty task description")
        
    except Exception as e:
        print(f"\n[ERROR] Error validating data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== Check 6: Cross-episode consistency =====
    print_section("CHECK 6: Cross-Episode Consistency")
    
    try:
        state_dims = []
        action_dims = []
        tasks = set()
        
        for ep_idx, episode in enumerate(dataset.episodes):
            if ep_idx >= 5:  # Check first 5 episodes
                break
            ep_start = episode['from']
            sample = dataset[ep_start]
            
            if 'state' in sample:
                state_dims.append(sample['state'].shape[0] if len(sample['state'].shape) > 0 else 1)
            if 'actions' in sample:
                action_dims.append(sample['actions'].shape[0] if len(sample['actions'].shape) > 0 else 1)
            if 'task' in sample:
                tasks.add(sample['task'])
        
        # Check dimensions
        if len(set(state_dims)) > 1:
            print(f"[ERROR] Inconsistent state dimensions: {state_dims}")
            return False
        else:
            print(f"[OK] Consistent state dimension: {state_dims[0]}")
        
        if len(set(action_dims)) > 1:
            print(f"[ERROR] Inconsistent action dimensions: {action_dims}")
            return False
        else:
            print(f"[OK] Consistent action dimension: {action_dims[0]}")
        
        # Check tasks
        if len(tasks) > 0:
            print(f"\n[TASKS] Found {len(tasks)} unique task(s):")
            for task in sorted(tasks):
                print(f"  - '{task}'")
        
    except Exception as e:
        print(f"[WARNING] Could not verify consistency: {e}")
    
    # ===== Final Summary =====
    print_header("Verification Summary")
    
    checks = [
        ("Dataset exists at correct location", True),
        ("Dataset loads successfully", True),
        ("Has episodes with frames", dataset.num_episodes > 0 and len(dataset) > 0),
        ("Has 'state' feature", 'state' in dataset.features),
        ("Has 'actions' feature", 'actions' in dataset.features),
        ("Has camera feature(s)", len(camera_features) > 0),
        ("Has language instructions", has_language),
        ("Sufficient data (1000+ frames)", len(dataset) >= 1000),
        ("Multiple episodes (10+)", dataset.num_episodes >= 10),
    ]
    
    print()
    for check_name, passed in checks:
        status = "[OK]" if passed else ("[WARNING] " if "Sufficient" in check_name or "Multiple" in check_name else "[ERROR]")
        print(f"  {status} {check_name}")
    
    critical_checks = [c for c in checks[:7] if not c[1]]  # First 7 are critical
    all_critical_passed = len(critical_checks) == 0
    
    print("\n" + "=" * 70)
    if all_critical_passed:
        print("[OK] PASSED: Dataset is ready for Pi0.5 training!")
        print("\n[DOCS] Next steps:")
        stats_path = dataset_path / 'meta' / 'stats.json'
        if not stats_path.exists():
            print(f"  1. Compute normalization stats (REQUIRED before training):")
            print(f"     uv run python -m lerobot.scripts.compute_stats \\")
            print(f"       --repo-id {dataset_name}")
        else:
            print(f"  1. Stats already computed [OK]")
        print(f"  2. Start training:")
        print(f"     uv run scripts/train.py airbot_pi05 --exp-name=my_experiment")
        print(f"  3. Monitor training in wandb or tensorboard")
    else:
        print("[ERROR] FAILED: Dataset has critical issues that need to be fixed")
        print("\n[FIX] Please fix the issues marked with [ERROR] and re-run verification")
    
    print("=" * 70)
    
    return all_critical_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify LeRobot dataset for Pi0.5 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("dataset_name", 
                       help="Dataset name (e.g., 'your_username/robot_dataset')")
    parser.add_argument("--visualize", action="store_true",
                       help="Save sample images for visual inspection")
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataset_name, args.visualize)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
