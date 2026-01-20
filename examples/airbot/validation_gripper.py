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
        --policy.dir=checkpoints/airbot_pi05/airbot_v1/20000

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
import sys
import time
from pathlib import Path
from typing import Optional
import rerun as rr

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

# Add openpi root to Python path for revo2_library imports
openpi_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(openpi_root))

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
    arm_port: str = "50001"  # AirBot serial port or IP

    # Rollout parameters
    num_episodes: int = 5
    max_episode_steps: int = 3000  # ~20 seconds at 30 Hz
    control_freq: int = 30  # Hz
    action_horizon: int = 15  # Execute N actions before re-querying policy
    
    # Camera configuration (Orbbec)
    # NOTE: Should match aspect ratio used during training for best results
    # Training used 320x180 (16:9), so we capture at same aspect ratio
    camera_width: int = 1280   # Will be resized by policy server
    camera_height: int = 720  # 16:9 aspect ratio to match training
    camera_fps: int = 30
    
    # Home position for arm (6 DOF) - adjust to your setup
    arm_home: tuple = (0, -1.52, 0.123, -1.54, -1.16, -0.632)
    
    # Prompts
    default_prompt: str = "pick up the bottle and place it in the box"


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
            self.robot.set_speed_profile(SpeedProfile.SLOW)
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
    
    def set_gripper_position(self, position):    
        """Set gripper position (single DOF)."""
        if not self.robot:
            return
        try:
            # 1. If it's a numpy type, convert to python type (float or list)
            if hasattr(position, 'tolist'):
                position = position.tolist()
                
            # 2. If it is now a single float/int, wrap it in a list
            if not isinstance(position, list):
                position = [position]
                
            self.robot.servo_eef_pos(position)
        except Exception as e:
            print(f"[ARM] Error setting gripper position: {e}")
    
    def get_gripper_position(self):
        """Get current gripper position (single DOF)."""
        if not self.robot:
            return np.zeros(1, dtype=np.float32)
        try:
            pos = self.robot.get_eef_pos()
            return np.array(pos, dtype=np.float32)
        except Exception as e:
            print(f"[ARM] Error reading gripper position: {e}")
            return np.zeros(1, dtype=np.float32)
        
    def move_to_home(self, home_pos: tuple, blocking: bool = True):
        """Move arm to home position."""
        if not self.robot:
            print("[ARM] Dummy: would move to home")
            return
        try:
            self.robot.switch_mode(RobotMode.PLANNING_POS)
            self.robot.set_speed_profile(SpeedProfile.SLOW)
            self.robot.move_to_joint_pos(joint_pos=list(home_pos), blocking=blocking)
            self.robot.set_speed_profile(SpeedProfile.SLOW)
            self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        except Exception as e:
            print(f"[ARM] Error moving to home: {e}")





class Camera:
    """
    Orbbec camera interface for RGB image capture.
    
    Uses pyorbbecsdk to capture frames from Orbbec cameras (e.g., Gemini 336).
    Only captures RGB - simplified from the full multi_streams.py implementation.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self._connected = False
        
    def connect(self) -> bool:
        """Connect to Orbbec camera and start RGB stream."""
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
            
            print(f"[CAMERA] Connecting to Orbbec camera ({self.width}x{self.height} @ {self.fps}fps)...")
            
            self.pipeline = Pipeline()
            config = Config()
            
            # Configure color stream only (we just need RGB for policy)
            try:
                color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                # Try to get requested resolution, fall back to what's available
                color_profile = color_profiles.get_video_stream_profile(
                    self.width, self.height, OBFormat.RGB, self.fps
                )
                config.enable_stream(color_profile)
                print(f"[CAMERA] Using profile: {self.width}x{self.height} RGB @ {self.fps}fps")
            except Exception as e:
                print(f"[CAMERA] Warning: Could not get exact profile, trying defaults: {e}")
                # Try common fallback resolutions
                for res in [(1280, 720), (640, 480), (320, 240)]:
                    try:
                        color_profile = color_profiles.get_video_stream_profile(
                            res[0], res[1], OBFormat.RGB, 30
                        )
                        config.enable_stream(color_profile)
                        self.width, self.height = res
                        print(f"[CAMERA] Using fallback profile: {res[0]}x{res[1]} RGB @ 30fps")
                        break
                    except:
                        continue
            
            self.pipeline.start(config)
            self._connected = True
            print("[CAMERA] Connected successfully")
            return True
            
        except ImportError:
            print("[CAMERA] pyorbbecsdk not available - using dummy camera")
            self._connected = False
            return True  # Return True to allow validation to continue with dummy data
            
        except Exception as e:
            print(f"[CAMERA] Connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Stop camera pipeline."""
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("[CAMERA] Disconnected")
            except Exception as e:
                print(f"[CAMERA] Disconnect error: {e}")
        self._connected = False
    
    def get_frame(self) -> np.ndarray:
        """
        Get current RGB frame from camera.
        
        Returns:
            np.ndarray: RGB image (H, W, 3) uint8
        """
        if not self._connected or self.pipeline is None:
            # Return dummy frame if not connected
            return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        try:
            # Wait for frames with 100ms timeout
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                print("[CAMERA] No frames received")
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("[CAMERA] No color frame")
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Convert frame to RGB numpy array

            rgb_image = self._frame_to_rgb(color_frame)
            rr.log("camera/rgb", rr.Image(rgb_image), static=True)
            return rgb_image
            
        except Exception as e:
            print(f"[CAMERA] Error getting frame: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def _frame_to_rgb(self, frame) -> np.ndarray:
        """
        Convert Orbbec frame to RGB numpy array.
        
        Simplified version of frame_to_bgr_image from motion_retargeting.
        """
        try:
            from pyorbbecsdk import OBFormat
            import cv2
            
            width = frame.get_width()
            height = frame.get_height()
            color_format = frame.get_format()
            data = np.asanyarray(frame.get_data())
            
            if color_format == OBFormat.RGB:
                # Already RGB, just reshape
                image = np.resize(data, (height, width, 3))
                return image.astype(np.uint8)
                
            elif color_format == OBFormat.BGR:
                # Convert BGR to RGB
                image = np.resize(data, (height, width, 3))
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
                
            elif color_format == OBFormat.MJPG:
                # Decode MJPEG
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
                    
            elif color_format == OBFormat.YUYV:
                image = np.resize(data, (height, width, 2))
                bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
            
            # Fallback: try to interpret as RGB
            print(f"[CAMERA] Unknown format {color_format}, attempting raw conversion")
            image = np.resize(data, (height, width, 3))
            return image.astype(np.uint8)
            
        except Exception as e:
            print(f"[CAMERA] Frame conversion error: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)


# =============================================================================
# Validation Logic
# =============================================================================
def run_episode(
    policy: _websocket_client_policy.WebsocketClientPolicy,
    arm: AirBotArm,
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
    arm.set_gripper_position(0.0)  # Close gripper
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
            gripper_position = arm.get_gripper_position()
            state = np.concatenate([arm_joints, gripper_position]).astype(np.float32)
            

            # 2. Build observation dict for policy
            observation = {
                "observation/image": image,
                "observation/state": state,
                "prompt": prompt,
            }
            
            # 3. Query policy
            result = policy.infer(observation)
            actions = result["actions"]  # Shape: (N, 7)
            
            
            # 4. Execute action chunk (open-loop)
            for i in range(min(args.action_horizon, len(actions))):
                delta_action = actions[i]
                
                # Split action into arm and hand
                arm_action = arm_joints + delta_action[:6]
                gripper_action = gripper_position + delta_action[6:7]
                
                # Execute
                
                arm.set_joint_positions(arm_action)
                arm.set_gripper_position(gripper_action)
                
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
    rr.init("airbot_validation", spawn=True)
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
    
    camera = Camera(width=args.camera_width, height=args.camera_height, fps=args.camera_fps)
    if not camera.connect():
        print("[ERROR] Failed to connect to camera")
        arm.disconnect()
        return
    
    print("[ROBOT] All hardware initialized")
    
    # Quick hardware test
    print("\n" + "="*70)
    print("HARDWARE TEST")
    print("="*70)
    print("Testing hardware connections...")
    
    # Test arm - read position
    try:
        arm_pos = arm.get_joint_positions()
        gripper_pos = arm.get_gripper_position()
        print(f"[ARM] Current position: {arm_pos}")
        print(f"[ARM] Current gripper position: {gripper_pos}")
        print(f"[ARM] ✓ Reading positions successfully")
    except Exception as e:
        print(f"[ARM] ✗ Error reading positions: {e}")
    # Test camera
    try:
        frame = camera.get_frame()
        print(f"[CAMERA] Frame shape: {frame.shape}")
        print(f"[CAMERA] ✓ Capturing frames successfully")
    except Exception as e:
        print(f"[CAMERA] ✗ Error capturing frame: {e}")
    
    print("="*70)
    print("Review the hardware test results above.")
    print("="*70)
    
    # Interactive movement tests
    print("\n" + "="*70)
    print("INTERACTIVE HARDWARE MOVEMENT TEST")
    print("="*70)
    print("This will test actual hardware movements.")
    print("Make sure the robot has clear space to move safely!")
    print()
    
    proceed = input("Proceed with movement test? [y/n]: ").strip().lower()
    if proceed == 'y':
        # Test 1: Move arm to home position
        print("\n[TEST 1/3] Moving arm to home position...")
        print(f"[ARM] Target home position: {args.arm_home}")
        try:
            arm.move_to_home(args.arm_home, blocking=True)
            print("[ARM] ✓ Successfully moved to home position")
            time.sleep(0.5)
        except Exception as e:
            print(f"[ARM] ✗ Error moving to home: {e}")

    
    print("\nIf everything looks good, press Enter to start validation...")
    print("Otherwise, press Ctrl+C to exit and fix issues.")
    input()
    
    # 3. Warm up policy (first inference is slow)
    print("\nWarming up policy...")
    dummy_obs = {
        "observation/image": np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8),
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
            result = run_episode(policy, arm, camera, args, prompt)
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
