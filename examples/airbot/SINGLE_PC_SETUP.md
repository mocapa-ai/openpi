# AirBot Pi0.5 Validation - Single PC Setup (No ROS)

## Updated Setup for Your Use Case

You mentioned:
- ✅ **Single PC**: Both inference and robot control on same machine
- ✅ **No ROS**: Direct robot control without ROS dependency

This simplifies things significantly! The remote server architecture is optional.

---

## Two Validation Approaches

### Option 1: Direct In-Process (Simpler - Recommended for You)

Run everything in a single Python process - no networking, no server.

```python
# examples/airbot/main_simple.py
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np

# Load your trained policy directly
config = _config.get_config("airbot_pi05")
checkpoint_dir = "checkpoints/airbot_pi05/20000"
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# TODO: Initialize your robot
# from your_robot_interface import AirBotController
# robot = AirBotController()

while True:
    instruction = input("Enter instruction: ")
    
    # TODO: Get observation from robot
    # image = robot.get_camera_image()  # (H, W, 3) uint8
    # state = robot.get_joint_positions()  # (12,) float32
    
    # Create dummy observation for now
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    state = np.random.randn(12).astype(np.float32)
    
    # Run inference (no network calls!)
    observation = {
        "observation/image": image,
        "observation/state": state,
        "prompt": instruction,
    }
    
    action_chunk = policy.infer(observation)["actions"]
    print(f"Got action chunk: {action_chunk.shape}")  # (10, 12)
    
    # TODO: Execute actions on robot
    # for action in action_chunk:
    #     robot.set_joint_positions(action)
    #     time.sleep(1/30)  # 30 Hz control
```

**Advantages:**
- ✅ No networking complexity
- ✅ No separate server process
- ✅ Lower latency (no WebSocket overhead)
- ✅ Simpler debugging

### Option 2: Local Server (Optional)

If you still want the server architecture but on one PC:

```bash
# Terminal 1: Server on localhost
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/20000

# Terminal 2: Client connects to localhost
python examples/airbot/main.py --remote_host=localhost
```

Benefits: Can test inference separately from robot control.

---

## Recommended: Simple Single-File Validation Script

Here's a complete standalone script for single-PC validation:

```python
#!/usr/bin/env python3
"""
Simple single-PC validation for AirBot (no ROS, no remote server).

Usage:
    python examples/airbot/validate_simple.py
"""

import dataclasses
import datetime
import os
import time
import numpy as np
from PIL import Image
import pandas as pd
import tqdm
import tyro

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config

# TODO: Import your robot control interface
# from your_robot_package import AirBotController


@dataclasses.dataclass
class Args:
    """Command line arguments."""
    checkpoint_dir: str = "checkpoints/airbot_pi05/20000"
    config_name: str = "airbot_pi05"
    
    # Rollout parameters
    max_timesteps: int = 600
    control_freq: int = 30  # Hz
    
    # Results
    results_dir: str = "results"
    save_videos: bool = True


def main(args: Args):
    """Single-PC validation loop."""
    
    print("=" * 80)
    print("AirBot Pi0.5 Validation - Single PC (No ROS)")
    print("=" * 80)
    
    # =========================================================================
    # Load policy (runs on same PC)
    # =========================================================================
    print(f"\n1. Loading policy from {args.checkpoint_dir}...")
    config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    print("   ✓ Policy loaded")
    
    # =========================================================================
    # Initialize robot (TODO: Replace with your interface)
    # =========================================================================
    print("\n2. Initializing robot...")
    # TODO: Replace with your actual robot initialization
    # robot = AirBotController()
    # print("   ✓ Robot initialized")
    
    print("   ⚠️  WARNING: Robot control not implemented yet!")
    print("   Edit this file and implement the TODO sections.")
    print("\n   For now, running in DEMO MODE with dummy observations...\n")
    
    # =========================================================================
    # Setup results tracking
    # =========================================================================
    os.makedirs(args.results_dir, exist_ok=True)
    results_df = pd.DataFrame(
        columns=["instruction", "success", "duration", "timestamp"]
    )
    
    # =========================================================================
    # Main validation loop
    # =========================================================================
    episode_num = 0
    
    while True:
        print("\n" + "=" * 80)
        instruction = input("Enter instruction (or 'quit' to exit): ").strip()
        
        if instruction.lower() in ["quit", "exit", "q"]:
            break
        
        if not instruction:
            continue
        
        episode_num += 1
        print(f"\n=== Episode {episode_num}: {instruction} ===\n")
        
        # Video frames (if saving)
        video_frames = []
        
        # Run rollout
        print("Running rollout... (Press Ctrl+C to stop)")
        dt = 1.0 / args.control_freq
        
        for t in tqdm.tqdm(range(args.max_timesteps)):
            start_time = time.time()
            
            try:
                # =============================================================
                # Get observation from robot
                # =============================================================
                # TODO: Replace with actual robot observation
                # image = robot.get_camera_image()  # (H, W, 3) uint8
                # state = robot.get_joint_positions()  # (12,) float32
                
                # DEMO: Random observation
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                state = np.random.randn(12).astype(np.float32)
                
                if args.save_videos:
                    video_frames.append(image)
                
                # =============================================================
                # Run inference (in-process, no network!)
                # =============================================================
                observation = {
                    "observation/image": image,
                    "observation/state": state,
                    "prompt": instruction,
                }
                
                # Get action chunk from policy
                action_chunk = policy.infer(observation)["actions"]
                # Shape: (10, 12) - 10 timesteps, 12 DOF
                
                # =============================================================
                # Execute actions
                # =============================================================
                # Execute multiple actions from chunk before re-querying
                # (this is the "open-loop horizon" - reduces inference frequency)
                open_loop_horizon = 8
                
                for i in range(min(open_loop_horizon, len(action_chunk))):
                    action = action_chunk[i]
                    
                    # TODO: Apply action to robot
                    # robot.set_joint_positions(action)
                    
                    # DEMO: Just print action range
                    if t == 0 and i == 0:
                        print(f"\n   First action: min={action.min():.3f}, "
                              f"max={action.max():.3f}")
                    
                    # Sleep to maintain control frequency
                    elapsed = time.time() - start_time
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    # Re-query every open_loop_horizon steps
                    if i < open_loop_horizon - 1:
                        start_time = time.time()
                
            except KeyboardInterrupt:
                print("\n\nRollout stopped by user")
                break
        
        # =============================================================
        # Save video (optional)
        # =============================================================
        if args.save_videos and len(video_frames) > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                args.results_dir, 
                f"episode_{episode_num:03d}_{timestamp}.gif"
            )
            print(f"\nSaving video to {video_path}...")
            
            # Save as GIF (simpler than mp4, no moviepy needed)
            images = [Image.fromarray(frame) for frame in video_frames[::3]]  # Subsample
            images[0].save(
                video_path,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0
            )
            print(f"   ✓ Saved {len(images)} frames")
        
        # =============================================================
        # Record results
        # =============================================================
        print("\n" + "-" * 80)
        success_input = input("Success rate (0-100)? [y/n/number]: ").strip().lower()
        
        if success_input == "y":
            success = 1.0
        elif success_input == "n":
            success = 0.0
        else:
            try:
                success = float(success_input) / 100
            except ValueError:
                success = 0.0
        
        # Save result
        results_df = pd.concat([results_df, pd.DataFrame([{
            "instruction": instruction,
            "success": success,
            "duration": t + 1,
            "timestamp": datetime.datetime.now().isoformat(),
        }])], ignore_index=True)
        
        # Show stats
        print(f"\n   Episodes: {len(results_df)}")
        print(f"   Avg success: {results_df['success'].mean() * 100:.1f}%")
        
        # Continue?
        if input("\nRun another episode? [y/n]: ").strip().lower() != "y":
            break
        
        # TODO: Reset robot
        # robot.reset_to_home()
    
    # =========================================================================
    # Save final results
    # =========================================================================
    if len(results_df) > 0:
        csv_path = os.path.join(
            args.results_dir,
            f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        print(f"\n=== Final Stats ===")
        print(f"Total episodes: {len(results_df)}")
        print(f"Overall success: {results_df['success'].mean() * 100:.1f}%")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
```

---

## Key Differences from Remote Server Setup

| Aspect | Remote Server | Single PC (Your Case) |
|--------|---------------|----------------------|
| **Architecture** | 2 processes, WebSocket | 1 process, direct calls |
| **Latency** | ~10-50ms network | <1ms (in-memory) |
| **Complexity** | Higher (networking) | Lower (simple Python) |
| **Debugging** | Harder (2 processes) | Easier (1 process) |
| **Flexibility** | Can switch checkpoints easily | Need to restart script |
| **ROS Needed?** | No (but they used it for ALOHA) | No ✓ |

---

## Robot Control Interface (No ROS)

Since you don't want ROS, here are common non-ROS robot control approaches:

### Option A: Direct Hardware Interface

```python
# Example with direct serial/USB control
import serial

class AirBotController:
    def __init__(self, port="/dev/ttyUSB0"):
        self.arm_serial = serial.Serial(port, 115200)
        # Initialize hand, camera, etc.
    
    def get_joint_positions(self):
        # Read from encoders
        return np.array([...])  # (12,)
    
    def set_joint_positions(self, positions):
        # Send commands to motors
        pass
    
    def get_camera_image(self):
        # Read from camera (OpenCV, etc.)
        return image  # (H, W, 3) uint8
```

### Option B: SDK/Library

```python
# If AirBot has a Python SDK
from airbot_sdk import AirBot

class AirBotController:
    def __init__(self):
        self.robot = AirBot.connect()
    
    def get_joint_positions(self):
        return self.robot.get_arm_joints() + self.robot.get_hand_joints()
    
    def set_joint_positions(self, positions):
        arm_pos = positions[:6]
        hand_pos = positions[6:]
        self.robot.move_arm(arm_pos)
        self.robot.move_hand(hand_pos)
```

### Option C: Python API (Dynamixel, etc.)

```python
# If using Dynamixel servos
from dynamixel_sdk import *

class AirBotController:
    def __init__(self):
        self.port_handler = PortHandler("/dev/ttyUSB0")
        self.packet_handler = PacketHandler(2.0)
        # Initialize motors
    
    def get_joint_positions(self):
        positions = []
        for motor_id in range(1, 13):  # 12 motors
            pos = self.read_position(motor_id)
            positions.append(pos)
        return np.array(positions)
```

---

## Minimal Working Example

Save this as `examples/airbot/validate_simple.py` and run it:

```bash
cd ~/openpi

# Run validation (single process, no server needed)
uv run python examples/airbot/validate_simple.py \
    --checkpoint_dir=checkpoints/airbot_pi05/20000
```

This will:
1. Load your policy directly in-process
2. Run inference on dummy observations (until you implement robot control)
3. Show you the action outputs
4. Record success metrics

---

## What You Need to Implement

Just 3 functions in the script above:

1. **Initialize robot** (~line 55)
   ```python
   from your_package import AirBotController
   robot = AirBotController()
   ```

2. **Get observations** (~line 90)
   ```python
   image = robot.get_camera_image()  # (H, W, 3) uint8 RGB
   state = robot.get_joint_positions()  # (12,) float32
   ```

3. **Apply actions** (~line 120)
   ```python
   robot.set_joint_positions(action)  # action is (12,) float32
   ```

That's it! Much simpler than the remote server setup.

---

## Quick Start for Single PC

```bash
cd ~/openpi

# 1. Create the simple validation script
# (I'll create this file next)

# 2. Test it loads your policy
uv run python examples/airbot/validate_simple.py \
    --checkpoint_dir=checkpoints/airbot_pi05/20000 \
    --max_timesteps=10  # Just 10 steps for testing

# 3. Implement robot control (edit the TODO sections)

# 4. Run full validation
uv run python examples/airbot/validate_simple.py
```

---

## Summary: Single PC vs Remote Server

**For your use case (single PC, no ROS):**
- ✅ Use the simple direct-inference approach
- ✅ Single Python script, single process
- ✅ No networking, no WebSocket, no server
- ✅ Lower latency, easier debugging
- ✅ Just implement 3 functions for your robot interface

**The remote server docs are still useful for:**
- Understanding the validation workflow
- Seeing how Physical Intelligence did it
- Reference if you ever want to separate GPU from robot PC

Let me create the simple validation script for you!
