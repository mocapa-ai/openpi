#!/usr/bin/env python3
"""
Simple single-PC validation for AirBot Pi0.5 (no ROS, no remote server).

This script runs policy inference and robot control in a single process on one PC.
Much simpler than the remote server architecture when you don't need separation.

Usage:
    # Test with your trained checkpoint
    uv run python examples/airbot/validate_simple.py \
        --checkpoint_dir=checkpoints/airbot_pi05/20000

    # Quick test (10 timesteps)
    uv run python examples/airbot/validate_simple.py \
        --checkpoint_dir=checkpoints/airbot_pi05/20000 \
        --max_timesteps=10
"""

import dataclasses
import datetime
import os
import time
from typing import Optional

import numpy as np
from PIL import Image
import pandas as pd
import tqdm
import tyro

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config


# TODO: Import your robot control interface
# Example options:
# from your_robot_package import AirBotController
# from airbot_sdk import AirBot
# import serial  # For direct serial control
# from dynamixel_sdk import *  # For Dynamixel servos


@dataclasses.dataclass
class Args:
    """Command line arguments for single-PC validation."""
    
    # Model/checkpoint
    checkpoint_dir: str = "checkpoints/airbot_pi05/20000"
    config_name: str = "airbot_pi05"
    
    # Rollout parameters
    max_timesteps: int = 600  # ~20 seconds at 30 Hz
    control_freq: int = 30  # Hz - should match your training data
    open_loop_horizon: int = 8  # Execute N actions before re-querying policy
    
    # Results/logging
    results_dir: str = "results"
    save_videos: bool = True
    video_fps: int = 10  # FPS for saved videos


class RobotInterface:
    """
    Robot control interface (template).
    
    TODO: Implement this class with your actual robot control code.
    Replace the NotImplementedError calls with real robot communication.
    """
    
    def __init__(self):
        """Initialize robot connection."""
        # TODO: Initialize your robot hardware
        # Examples:
        # - Serial port connection
        # - SDK initialization
        # - Motor controller setup
        # - Camera initialization
        
        print("⚠️  WARNING: Using dummy robot interface!")
        print("   Implement RobotInterface class with your robot control code.\n")
        
        # Dummy state
        self._current_joints = np.zeros(12, dtype=np.float32)
    
    def get_observation(self):
        """
        Get current observation from robot.
        
        Returns:
            dict with:
                - "image": (H, W, 3) uint8 RGB image from camera
                - "state": (12,) float32 joint positions (6 arm + 6 hand)
        
        TODO: Implement with your actual robot sensors.
        """
        # TODO: Get real camera image
        # Example:
        # import cv2
        # cap = cv2.VideoCapture(0)
        # ret, frame = cap.read()
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # DUMMY: Random image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # TODO: Get real joint positions
        # Example:
        # arm_joints = self.arm_controller.get_positions()  # (6,)
        # hand_joints = self.hand_controller.get_positions()  # (6,)
        # state = np.concatenate([arm_joints, hand_joints])
        
        # DUMMY: Random state
        state = self._current_joints.copy()
        
        return {
            "image": image,
            "state": state,
        }
    
    def set_action(self, action: np.ndarray):
        """
        Execute action on robot.
        
        Args:
            action: (12,) float32 array of joint positions
                    [0:6] = arm joint positions
                    [6:12] = hand joint positions
        
        TODO: Implement with your actual robot control.
        """
        assert action.shape == (12,), f"Expected (12,) action, got {action.shape}"
        
        # TODO: Send commands to robot
        # Example:
        # arm_pos = action[:6]
        # hand_pos = action[6:]
        # self.arm_controller.set_positions(arm_pos)
        # self.hand_controller.set_positions(hand_pos)
        
        # DUMMY: Just update internal state
        self._current_joints = action.copy()
    
    def reset(self):
        """
        Reset robot to home position.
        
        TODO: Implement with your actual reset procedure.
        """
        # TODO: Move robot to home position
        # Example:
        # home_position = np.zeros(12)
        # self.set_action(home_position)
        # time.sleep(2)  # Wait for motion to complete
        
        # DUMMY: Reset to zeros
        self._current_joints = np.zeros(12, dtype=np.float32)
        print("   Robot reset to home position")
    
    def close(self):
        """
        Clean up robot resources.
        
        TODO: Implement cleanup (close serial ports, etc.)
        """
        # TODO: Close connections, release resources
        # Example:
        # self.serial_port.close()
        # self.camera.release()
        pass


def main(args: Args):
    """Main validation loop for single-PC setup."""
    
    print("=" * 80)
    print("AirBot Pi0.5 Validation - Single PC (No ROS)")
    print("=" * 80)
    print("\nThis script runs inference and robot control in a single process.")
    print("No remote server, no ROS, no networking complexity.\n")
    
    # =========================================================================
    # Load policy (runs on same PC as robot control)
    # =========================================================================
    print(f"1. Loading policy from {args.checkpoint_dir}...")
    try:
        config = _config.get_config(args.config_name)
        policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
        print("   ✓ Policy loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load policy: {e}")
        print("\nMake sure:")
        print(f"  - Checkpoint exists at {args.checkpoint_dir}")
        print(f"  - Config '{args.config_name}' is defined in src/openpi/training/config.py")
        return
    
    # =========================================================================
    # Initialize robot
    # =========================================================================
    print("\n2. Initializing robot...")
    robot = RobotInterface()
    print("   ⚠️  Using dummy robot interface (edit RobotInterface class)")
    
    # Test getting observation
    try:
        test_obs = robot.get_observation()
        print(f"   ✓ Got observation: image {test_obs['image'].shape}, "
              f"state {test_obs['state'].shape}")
    except Exception as e:
        print(f"   ✗ Failed to get observation: {e}")
        return
    
    # =========================================================================
    # Setup results tracking
    # =========================================================================
    os.makedirs(args.results_dir, exist_ok=True)
    results_df = pd.DataFrame(
        columns=["episode", "instruction", "success", "duration", "timestamp"]
    )
    
    # =========================================================================
    # Main validation loop
    # =========================================================================
    episode_num = 0
    dt = 1.0 / args.control_freq
    
    print("\n" + "=" * 80)
    print("Ready for validation!")
    print("=" * 80)
    
    while True:
        print("\n" + "-" * 80)
        instruction = input("Enter instruction (or 'quit' to exit): ").strip()
        
        if instruction.lower() in ["quit", "exit", "q"]:
            break
        
        if not instruction:
            print("Empty instruction, skipping...")
            continue
        
        episode_num += 1
        print(f"\n=== Episode {episode_num}: '{instruction}' ===")
        
        # Reset robot to home
        print("Resetting robot to home position...")
        robot.reset()
        input("Press Enter when ready to start...")
        
        # Video frames
        video_frames = []
        
        # Run rollout
        print(f"\nRunning rollout (max {args.max_timesteps} steps)...")
        print("Press Ctrl+C to stop early.\n")
        
        action_chunk = None
        action_chunk_idx = 0
        
        for t in tqdm.tqdm(range(args.max_timesteps), desc="Timesteps"):
            step_start = time.time()
            
            try:
                # Get observation
                obs = robot.get_observation()
                
                # Save frame for video
                if args.save_videos:
                    video_frames.append(obs["image"])
                
                # Query policy if needed (every open_loop_horizon steps)
                if action_chunk is None or action_chunk_idx >= args.open_loop_horizon:
                    # Prepare observation for policy
                    policy_obs = {
                        "observation/image": obs["image"],
                        "observation/state": obs["state"],
                        "prompt": instruction,
                    }
                    
                    # Run inference (in-process, no network!)
                    action_chunk = policy.infer(policy_obs)["actions"]
                    # Shape: (10, 12) - 10 timesteps, 12 DOF
                    
                    action_chunk_idx = 0
                
                # Get current action from chunk
                action = action_chunk[action_chunk_idx]
                action_chunk_idx += 1
                
                # Execute action on robot
                robot.set_action(action)
                
                # Maintain control frequency
                elapsed = time.time() - step_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\n\nRollout stopped by user")
                break
            except Exception as e:
                print(f"\n\nError during rollout: {e}")
                import traceback
                traceback.print_exc()
                break
        
        actual_duration = t + 1
        
        # Save video
        if args.save_videos and len(video_frames) > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"ep{episode_num:03d}_{timestamp}.gif"
            video_path = os.path.join(args.results_dir, video_filename)
            
            print(f"\nSaving video ({len(video_frames)} frames)...")
            try:
                # Subsample frames for smaller file size
                subsample = max(1, len(video_frames) // 100)
                frames = [Image.fromarray(f) for f in video_frames[::subsample]]
                
                frames[0].save(
                    video_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / args.video_fps),  # ms per frame
                    loop=0
                )
                print(f"   ✓ Saved to {video_path}")
            except Exception as e:
                print(f"   ✗ Failed to save video: {e}")
        
        # Record success
        print("\n" + "-" * 80)
        print("How did the robot do?")
        success_input = input("Success rate (0-100)? [y/n/number]: ").strip().lower()
        
        if success_input == "y":
            success = 1.0
        elif success_input == "n":
            success = 0.0
        else:
            try:
                success = float(success_input) / 100
                success = np.clip(success, 0.0, 1.0)
            except ValueError:
                print(f"Invalid input '{success_input}', marking as failure")
                success = 0.0
        
        # Save result
        new_result = pd.DataFrame([{
            "episode": episode_num,
            "instruction": instruction,
            "success": success,
            "duration": actual_duration,
            "timestamp": datetime.datetime.now().isoformat(),
        }])
        results_df = pd.concat([results_df, new_result], ignore_index=True)
        
        # Show current stats
        print("\n" + "-" * 80)
        print("Current Statistics:")
        print(f"  Episodes completed: {len(results_df)}")
        print(f"  Average success: {results_df['success'].mean() * 100:.1f}%")
        print(f"  Average duration: {results_df['duration'].mean():.1f} steps")
        print("-" * 80)
        
        # Continue?
        if input("\nRun another episode? [y/n]: ").strip().lower() != "y":
            break
    
    # =========================================================================
    # Cleanup and save results
    # =========================================================================
    print("\n" + "=" * 80)
    print("Cleaning up...")
    robot.close()
    
    if len(results_df) > 0:
        # Save results to CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.results_dir, f"results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to {csv_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total episodes: {len(results_df)}")
        print(f"Overall success rate: {results_df['success'].mean() * 100:.1f}%")
        print(f"Average duration: {results_df['duration'].mean():.1f} steps")
        
        if len(results_df) > 1:
            print("\nResults by instruction:")
            summary = results_df.groupby("instruction")["success"].agg(["mean", "count"])
            summary["mean"] = summary["mean"] * 100
            summary.columns = ["Success (%)", "Count"]
            print(summary)
        
        print("=" * 80)
    else:
        print("No episodes completed.")
    
    print("\nDone!")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
