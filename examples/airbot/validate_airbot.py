#!/usr/bin/env python3
"""
AirBot + Revo2 Hand Validation Script

This script validates a fine-tuned Pi0.5 policy on the AirBot arm with Revo2 hand.
Uses the client-server architecture (same as ALOHA examples).

Architecture:
    Terminal 1 (uv): Policy server - runs model inference
    Terminal 2 (conda): This script - controls robot hardware

Usage:
    # Terminal 1: Start policy server
    cd ~/openpi
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=airbot_pi05 \
        --policy.dir=checkpoints/airbot_pi05/20000

    # Terminal 2: Run this validation script (in conda env with robot deps)
    conda activate your_robot_env
    pip install ~/openpi/packages/openpi-client  # one-time
    python examples/airbot/validate_airbot.py --host=localhost --port=8000

Requirements (in conda env):
    - openpi-client (pip install ~/openpi/packages/openpi-client)
    - airbot_py (for AirBot arm control)
    - Your revo2 hand library
    - numpy, opencv-python
"""

import dataclasses
import logging
import time
from typing import Optional

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

# =============================================================================
# Robot imports from motion_retargeting
# Adjust these imports based on your actual package structure
# =============================================================================
try:
    from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile
    AIRBOT_AVAILABLE = True
except ImportError:
    AIRBOT_AVAILABLE = False
    print("[WARN] airbot_py not available - using dummy arm")

# TODO: Import your Revo2 hand library
# try:
#     from motion_retargeting.hand.revo2_hand import Revo2HandAdapter
#     REVO2_AVAILABLE = True
# except ImportError:
#     REVO2_AVAILABLE = False
#     print("[WARN] Revo2 hand not available - using dummy hand")
REVO2_AVAILABLE = False  # Placeholder until you wire it up


# =============================================================================
# Configuration
# =============================================================================
@dataclasses.dataclass
class Args:
    """Command line arguments for AirBot validation."""
    
    # Policy server connection
    host: str = "localhost"
    port: int = 8000
    
    # Robot configuration
    arm_port: str = "50000"  # AirBot serial port or IP
    hand_port: str = "/dev/ttyUSB1"  # Revo2 hand port
    
    # Rollout parameters
    num_episodes: int = 5
    max_episode_steps: int = 600  # ~20 seconds at 30 Hz
    control_freq: int = 30  # Hz
    action_horizon: int = 15  # Execute N actions before re-querying policy
    
    # Home position for arm (6 DOF) - adjust to your setup
    arm_home: tuple = (0.0, -1.52, 0.349, -1.54, -1.16, 2.47)
    
    # Prompts
    default_prompt: str = "pick up the object"


# =============================================================================
# Robot Interface Classes
# =============================================================================
class AirBotArm:
    """AirBot arm controller wrapper."""
    
    def __init__(self, port: str):
        self.port = port
        self.robot = None
        
    def connect(self) -> bool:
        """Connect to AirBot arm."""
        if not AIRBOT_AVAILABLE:
            print("[ARM] Using dummy arm (airbot_py not available)")
            return True
            
        try:
            print(f"[ARM] Connecting to AirBot on port {self.port}...")
            self.robot = AIRBOTPlay(port=self.port)
            self.robot.connect()
            self.robot.set_speed_profile(SpeedProfile.FAST)
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
            print("[ARM] Connected successfully")
            return True
        except Exception as e:
            print(f"[ARM] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from arm."""
        if self.robot:
            try:
                self.robot.disconnect()
                print("[ARM] Disconnected")
            except Exception as e:
                print(f"[ARM] Disconnect error: {e}")
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions (6 DOF)."""
        if not self.robot:
            return np.zeros(6, dtype=np.float32)
        try:
            pos = self.robot.get_joint_pos()
            return np.array(pos, dtype=np.float32)
        except Exception as e:
            print(f"[ARM] Error reading positions: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set joint positions (6 DOF)."""
        if not self.robot:
            return
        try:
            pos_list = positions.tolist() if isinstance(positions, np.ndarray) else list(positions)
            self.robot.servo_joint_pos(joint_pos=pos_list)
        except Exception as e:
            print(f"[ARM] Error setting positions: {e}")
    
    def move_to_home(self, home_pos: tuple, blocking: bool = True):
        """Move arm to home position."""
        if not self.robot:
            print("[ARM] Dummy: would move to home")
            return
        try:
            self.robot.switch_mode(RobotMode.PLANNING_POS)
            self.robot.set_speed_profile(SpeedProfile.SLOW)
            self.robot.move_to_joint_pos(joint_pos=list(home_pos), blocking=blocking)
            self.robot.set_speed_profile(SpeedProfile.FAST)
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        except Exception as e:
            print(f"[ARM] Error moving to home: {e}")


class Revo2Hand:
    """Revo2 hand controller wrapper."""
    
    def __init__(self, port: str, side: str = "right"):
        self.port = port
        self.side = side
        self.hand = None
        
    def connect(self) -> bool:
        """Connect to Revo2 hand."""
        if not REVO2_AVAILABLE:
            print("[HAND] Using dummy hand (revo2 not available)")
            return True
        
        # TODO: Implement actual connection
        # try:
        #     from motion_retargeting.hand.revo2_hand import Revo2HandAdapter
        #     self.hand = Revo2HandAdapter(side=self.side, model_path=self.port)
        #     self.hand.connect()
        #     return True
        # except Exception as e:
        #     print(f"[HAND] Connection failed: {e}")
        #     return False
        return True
    
    def disconnect(self):
        """Disconnect from hand."""
        if self.hand:
            try:
                self.hand.disconnect()
                print("[HAND] Disconnected")
            except Exception as e:
                print(f"[HAND] Disconnect error: {e}")
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current finger positions (6 DOF)."""
        if not self.hand:
            return np.zeros(6, dtype=np.float32)
        try:
            pos = self.hand.get_joint_positions()
            return np.array(pos, dtype=np.float32)
        except Exception as e:
            print(f"[HAND] Error reading positions: {e}")
            return np.zeros(6, dtype=np.float32)
    
    def set_joint_positions(self, positions: np.ndarray):
        """Set finger positions (6 DOF)."""
        if not self.hand:
            return
        try:
            self.hand.set_joint_positions(positions.tolist())
        except Exception as e:
            print(f"[HAND] Error setting positions: {e}")
    
    def set_open(self):
        """Open all fingers."""
        self.set_joint_positions(np.zeros(6))


class Camera:
    """Camera interface placeholder."""
    
    def __init__(self):
        self.pipeline = None
        
    def connect(self) -> bool:
        """Connect to camera."""
        # TODO: Implement your camera connection
        # Example: Orbbec, RealSense, or USB camera
        print("[CAMERA] Using placeholder (implement your camera)")
        return True
    
    def disconnect(self):
        """Disconnect from camera."""
        pass
    
    def get_frame(self) -> np.ndarray:
        """
        Get current RGB frame from camera.
        
        Returns:
            np.ndarray: RGB image (H, W, 3) uint8
            
        TODO: Implement with your actual camera.
        """
        # Placeholder: return random image
        # Replace with actual camera capture
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


# =============================================================================
# Validation Logic
# =============================================================================
def run_episode(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    arm: AirBotArm,
    hand: Revo2Hand,
    camera: Camera,
    args: Args,
    prompt: str,
) -> dict:
    """
    Run a single validation episode.
    
    Returns:
        dict with episode results
    """
    dt = 1.0 / args.control_freq
    
    print(f"\n{'='*60}")
    print(f"Starting episode: '{prompt}'")
    print(f"{'='*60}")
    
    # Move to home position
    print("[EPISODE] Moving to home position...")
    arm.move_to_home(args.arm_home, blocking=True)
    hand.set_open()
    time.sleep(1.0)
    
    input("Press Enter when ready to start...")
    
    steps_completed = 0
    start_time = time.time()
    
    try:
        for step in range(args.max_episode_steps):
            step_start = time.time()
            
            # 1. Get observation
            image = camera.get_frame()
            arm_joints = arm.get_joint_positions()
            hand_joints = hand.get_joint_positions()
            state = np.concatenate([arm_joints, hand_joints]).astype(np.float32)
            
            # 2. Build observation dict for policy
            observation = {
                "observation/image": image,
                "observation/state": state,
                "prompt": prompt,
            }
            
            # 3. Query policy
            result = policy.infer(observation)
            actions = result["actions"]  # Shape: (N, 12)
            
            # 4. Execute action chunk (open-loop)
            for i in range(min(args.action_horizon, len(actions))):
                action = actions[i]
                
                # Split action into arm and hand
                arm_action = action[:6]
                hand_action = action[6:12]
                
                # Execute
                arm.set_joint_positions(arm_action)
                hand.set_joint_positions(hand_action)
                
                # Maintain control frequency
                elapsed = time.time() - step_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                step_start = time.time()
                
                steps_completed += 1
                
                if steps_completed >= args.max_episode_steps:
                    break
            
            # Print progress
            if step % 30 == 0:
                print(f"  Step {steps_completed}/{args.max_episode_steps}", end="\r")
                
    except KeyboardInterrupt:
        print("\n[EPISODE] Stopped by user")
    
    duration = time.time() - start_time
    print(f"\n[EPISODE] Completed {steps_completed} steps in {duration:.1f}s")
    
    # Get success rating from user
    print("\nHow did the robot do?")
    success_input = input("Success (0-100, y=100, n=0): ").strip().lower()
    
    if success_input == "y":
        success = 1.0
    elif success_input == "n":
        success = 0.0
    else:
        try:
            success = float(success_input) / 100.0
            success = np.clip(success, 0.0, 1.0)
        except ValueError:
            success = 0.0
    
    return {
        "prompt": prompt,
        "steps": steps_completed,
        "duration": duration,
        "success": success,
    }


def main(args: Args) -> None:
    """Main validation loop."""
    
    print("=" * 70)
    print("AirBot + Revo2 Hand Policy Validation")
    print("=" * 70)
    print(f"\nConnecting to policy server at {args.host}:{args.port}...")
    
    # 1. Connect to policy server
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = policy.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    print(f"[POLICY] Connected! Metadata: {metadata}")
    
    # 2. Initialize robot hardware
    print("\nInitializing robot hardware...")
    
    arm = AirBotArm(args.arm_port)
    if not arm.connect():
        print("[ERROR] Failed to connect to arm")
        return
    
    hand = Revo2Hand(args.hand_port)
    if not hand.connect():
        print("[ERROR] Failed to connect to hand")
        arm.disconnect()
        return
    
    camera = Camera()
    if not camera.connect():
        print("[ERROR] Failed to connect to camera")
        arm.disconnect()
        hand.disconnect()
        return
    
    print("[ROBOT] All hardware initialized")
    
    # 3. Warm up policy (first inference is slow)
    print("\nWarming up policy...")
    dummy_obs = {
        "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/state": np.zeros(12, dtype=np.float32),
        "prompt": args.default_prompt,
    }
    for _ in range(2):
        policy.infer(dummy_obs)
    print("[POLICY] Warmed up")
    
    # 4. Run validation episodes
    results = []
    
    try:
        for ep in range(args.num_episodes):
            print(f"\n{'#'*70}")
            print(f"Episode {ep + 1}/{args.num_episodes}")
            print(f"{'#'*70}")
            
            # Get prompt from user
            prompt = input(f"Enter instruction (default: '{args.default_prompt}'): ").strip()
            if not prompt:
                prompt = args.default_prompt
            
            # Run episode
            result = run_episode(policy, arm, hand, camera, args, prompt)
            results.append(result)
            
            # Print running stats
            avg_success = np.mean([r["success"] for r in results])
            print(f"\n[STATS] Episodes: {len(results)}, Avg Success: {avg_success*100:.1f}%")
            
            if ep < args.num_episodes - 1:
                cont = input("\nContinue to next episode? [y/n]: ").strip().lower()
                if cont != "y":
                    break
                    
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
    
    # 5. Cleanup
    print("\nCleaning up...")
    arm.disconnect()
    hand.disconnect()
    camera.disconnect()
    
    # 6. Print summary
    if results:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total episodes: {len(results)}")
        print(f"Average success: {np.mean([r['success'] for r in results])*100:.1f}%")
        print(f"Average duration: {np.mean([r['duration'] for r in results]):.1f}s")
        print("\nPer-episode results:")
        for i, r in enumerate(results):
            print(f"  {i+1}. '{r['prompt']}' - {r['success']*100:.0f}% success, {r['steps']} steps")
        print("=" * 70)
    
    print("\nDone!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
