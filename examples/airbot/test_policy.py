#!/usr/bin/env python3
"""
Simple test script to verify your Pi0.5 checkpoint loads and runs correctly.

This tests ONLY the policy - no robot hardware needed.
Uses dummy observations to check the model works.

Usage:
    # Test with your checkpoint
    uv run python examples/airbot/test_policy.py

    # Or specify a different checkpoint
    uv run python examples/airbot/test_policy.py \
        --checkpoint_dir=checkpoints/airbot_pi05/15000
"""

import dataclasses
import numpy as np
import tyro

from openpi.training import config as _config
from openpi.policies import policy_config


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    checkpoint_dir: str = "checkpoints/airbot_pi05/20000"
    config_name: str = "airbot_pi05"


def main(args: Args):
    """Test policy loading and inference with dummy data."""
    
    print("=" * 80)
    print("AirBot Pi0.5 Policy Test (No Robot Required)")
    print("=" * 80)
    
    # =========================================================================
    # Test 1: Load Policy
    # =========================================================================
    print("\n[Test 1] Loading policy...")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Config: {args.config_name}")
    
    try:
        config = _config.get_config(args.config_name)
        policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
        print("  ✅ Policy loaded successfully")
    except FileNotFoundError as e:
        print(f"  ❌ Checkpoint not found: {e}")
        print("\n  Make sure checkpoint exists:")
        print(f"    ls {args.checkpoint_dir}")
        return False
    except Exception as e:
        print(f"  ❌ Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Test 2: Create Dummy Observation
    # =========================================================================
    print("\n[Test 2] Creating dummy observation...")
    
    # Create random observation (matching your robot specs)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_state = np.random.randn(12).astype(np.float32)  # 6 arm + 6 hand
    dummy_prompt = "pick up the red cube"
    
    observation = {
        "observation/image": dummy_image,
        "observation/state": dummy_state,
        "prompt": dummy_prompt,
    }
    
    print(f"  Image shape: {dummy_image.shape} (uint8)")
    print(f"  State shape: {dummy_state.shape} (float32)")
    print(f"  Prompt: '{dummy_prompt}'")
    print("  ✅ Dummy observation created")
    
    # =========================================================================
    # Test 3: Run Inference
    # =========================================================================
    print("\n[Test 3] Running inference...")
    
    try:
        result = policy.infer(observation)
        print("  ✅ Inference completed")
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # Test 4: Validate Output
    # =========================================================================
    print("\n[Test 4] Validating output...")
    
    # Check response has actions
    if "actions" not in result:
        print(f"  ❌ Response missing 'actions' key")
        print(f"  Response keys: {list(result.keys())}")
        return False
    
    actions = result["actions"]
    print(f"  Action shape: {actions.shape}")
    
    # Expected shape: (10, 12) - 10 timesteps, 12 DOF
    expected_shape = (10, 12)
    if actions.shape != expected_shape:
        print(f"  ❌ Unexpected action shape!")
        print(f"    Expected: {expected_shape}")
        print(f"    Got: {actions.shape}")
        return False
    
    print(f"  ✅ Action shape correct: {expected_shape}")
    
    # Check for NaN/Inf
    if np.isnan(actions).any():
        print("  ❌ Actions contain NaN values")
        return False
    
    if np.isinf(actions).any():
        print("  ❌ Actions contain Inf values")
        return False
    
    print("  ✅ No NaN or Inf values")
    
    # Show action statistics
    print(f"\n  Action statistics:")
    print(f"    Min:  {actions.min():.3f}")
    print(f"    Max:  {actions.max():.3f}")
    print(f"    Mean: {actions.mean():.3f}")
    print(f"    Std:  {actions.std():.3f}")
    
    # Show first action
    print(f"\n  First action (timestep 0):")
    print(f"    Arm joints  [0:6]:  {actions[0, :6]}")
    print(f"    Hand joints [6:12]: {actions[0, 6:12]}")
    
    # =========================================================================
    # Test 5: Multiple Queries
    # =========================================================================
    print("\n[Test 5] Testing multiple sequential queries...")
    
    num_queries = 5
    try:
        for i in range(num_queries):
            # Slightly different observation each time
            obs = {
                "observation/image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "observation/state": np.random.randn(12).astype(np.float32),
                "prompt": dummy_prompt,
            }
            result = policy.infer(obs)
            print(f"  Query {i+1}/{num_queries}: ✅")
    except Exception as e:
        print(f"  ❌ Query failed: {e}")
        return False
    
    print(f"  ✅ All {num_queries} queries succeeded")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYour policy is working correctly. Next steps:")
    print("1. Implement robot control in validate_simple.py")
    print("2. Run: uv run python validate_simple.py")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    success = main(args)
    exit(0 if success else 1)
