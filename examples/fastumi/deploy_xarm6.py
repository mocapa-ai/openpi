"""
Deploy trained π₀.₅ policy on xArm6 robot.

This script connects to the xArm6 robot, queries the policy server for actions,
and executes them on the hardware.

Usage:
    # Terminal 1: Start policy server
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_fastumi_xarm6 \
        --policy.dir=checkpoints/pi05_fastumi_xarm6/fastumi_xarm6_v1/20000
    
    # Terminal 2: Run deployment
    python examples/fastumi/deploy_xarm6.py \
        --robot_ip 192.168.1.XXX \
        --camera_id 0 \
        --prompt "pick up the red cube"

Safety:
    - Keep emergency stop button accessible
    - Start with reduced speed/acceleration limits
    - Monitor robot behavior closely
"""

import argparse
import time
from typing import Optional

import cv2
import numpy as np
import requests
from xarm.wrapper import XArmAPI


class XArm6PolicyRunner:
    """Runs a VLA policy on xArm6 hardware."""
    
    def __init__(
        self,
        robot_ip: str,
        policy_url: str = "http://localhost:8000",
        camera_id: int = 0,
        control_freq: float = 10.0,
        action_smoothing: float = 0.3,
        gripper_threshold: float = 0.5,
    ):
        """
        Initialize xArm6 policy runner.
        
        Args:
            robot_ip: IP address of xArm6 robot
            policy_url: URL of policy server
            camera_id: Camera device ID
            control_freq: Control frequency in Hz
            action_smoothing: EMA smoothing factor (0 = no smoothing, 1 = full smoothing)
            gripper_threshold: Threshold for binary gripper (> threshold = open)
        """
        self.robot_ip = robot_ip
        self.policy_url = policy_url
        self.camera_id = camera_id
        self.control_freq = control_freq
        self.action_smoothing = action_smoothing
        self.gripper_threshold = gripper_threshold
        
        # Initialize robot
        print(f"Connecting to xArm6 at {robot_ip}...")
        self.arm = XArmAPI(robot_ip)
        self._setup_robot()
        
        # Initialize camera
        print(f"Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera resolution (will be resized to 224x224 for model)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # State tracking
        self.prev_action = None
        self.running = False
        
        print("Initialization complete!")
    
    def _setup_robot(self):
        """Configure robot for operation."""
        # Enable motion
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)  # Position mode
        self.arm.set_state(0)  # Ready state
        
        # Set conservative speed/acceleration limits for safety
        self.arm.set_joint_maxacc(30)  # degrees/s^2
        self.arm.set_joint_maxvel(30)  # degrees/s
        
        # Get initial state
        code, angles = self.arm.get_servo_angle()
        if code != 0:
            raise RuntimeError(f"Failed to get servo angles: code {code}")
        
        print(f"Current joint angles: {angles}")
    
    def get_observation(self, prompt: str) -> dict:
        """
        Get current observation from robot and camera.
        
        Args:
            prompt: Task prompt/instruction
        
        Returns:
            Observation dict for policy
        """
        # Capture camera frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture camera frame")
        
        # Resize to 224x224 (model input size)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Get current joint angles
        code, joint_angles = self.arm.get_servo_angle()
        if code != 0:
            print(f"Warning: Failed to get servo angles: code {code}")
            joint_angles = [0.0] * 6
        
        # Get gripper state (xArm6 uses position, convert to normalized [0,1])
        code, gripper_pos = self.arm.get_gripper_position()
        if code != 0:
            print(f"Warning: Failed to get gripper position: code {code}")
            gripper_state = 0.0
        else:
            # Normalize gripper (0-850 range typically)
            gripper_state = gripper_pos / 850.0
        
        # Combine into state vector (6 joints + 1 gripper)
        state = joint_angles + [gripper_state]
        
        return {
            "observation/front_image": frame_rgb.tolist(),
            "observation/joint_position": state,
            "prompt": prompt
        }
    
    def query_policy(self, observation: dict) -> np.ndarray:
        """
        Query policy server for actions.
        
        Args:
            observation: Observation dict
        
        Returns:
            Action array (action_horizon, 7) - joint angles + gripper
        """
        try:
            response = requests.post(
                f"{self.policy_url}/infer",
                json=observation,
                timeout=5.0
            )
            response.raise_for_status()
            actions = np.array(response.json()["actions"])
            return actions
        except Exception as e:
            print(f"Policy query failed: {e}")
            # Return zero action on failure
            return np.zeros((1, 7))
    
    def execute_action(self, action: np.ndarray):
        """
        Execute action on robot with smoothing.
        
        Args:
            action: (7,) array of [j1, j2, j3, j4, j5, j6, gripper]
        """
        # Apply exponential moving average for smoothing
        if self.prev_action is not None:
            action = (1 - self.action_smoothing) * action + self.action_smoothing * self.prev_action
        self.prev_action = action
        
        # Split into joint angles and gripper
        joint_target = action[:6].tolist()
        gripper_action = action[6]
        
        # Execute joint motion (non-blocking)
        code = self.arm.set_servo_angle(
            angle=joint_target,
            speed=None,  # Use default speed
            wait=False,
            radius=None
        )
        if code != 0:
            print(f"Warning: Joint command failed with code {code}")
        
        # Execute gripper command (binary: open/close)
        if gripper_action > self.gripper_threshold:
            # Open gripper
            gripper_target = 850
        else:
            # Close gripper
            gripper_target = 0
        
        code = self.arm.set_gripper_position(gripper_target, wait=False)
        if code != 0:
            print(f"Warning: Gripper command failed with code {code}")
    
    def run(self, prompt: str, max_steps: Optional[int] = None):
        """
        Run policy execution loop.
        
        Args:
            prompt: Task instruction
            max_steps: Maximum number of steps (None = infinite)
        """
        print("\n" + "="*80)
        print("Starting policy execution")
        print(f"Prompt: {prompt}")
        print(f"Control frequency: {control_freq} Hz")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        self.running = True
        step = 0
        
        try:
            while self.running:
                start_time = time.time()
                
                # Get observation
                obs = self.get_observation(prompt)
                
                # Query policy
                action_chunk = self.query_policy(obs)
                
                # Execute first action from chunk
                self.execute_action(action_chunk[0])
                
                # Timing
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0 / self.control_freq - elapsed)
                time.sleep(sleep_time)
                
                step += 1
                if step % 10 == 0:
                    print(f"Step {step}, Control latency: {elapsed*1000:.1f}ms")
                
                if max_steps is not None and step >= max_steps:
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        
        # Stop robot motion
        self.arm.set_mode(0)
        self.arm.set_state(4)  # Stop state
        
        # Release camera
        self.cap.release()
        
        # Disconnect robot
        self.arm.disconnect()
        
        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Deploy π₀.₅ policy on xArm6")
    parser.add_argument("--robot_ip", type=str, required=True,
                        help="IP address of xArm6 robot")
    parser.add_argument("--policy_url", type=str, default="http://localhost:8000",
                        help="URL of policy server")
    parser.add_argument("--camera_id", type=int, default=0,
                        help="Camera device ID")
    parser.add_argument("--prompt", type=str, default="pick up the red cube",
                        help="Task instruction prompt")
    parser.add_argument("--control_freq", type=float, default=10.0,
                        help="Control frequency in Hz")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of steps (None = infinite)")
    parser.add_argument("--action_smoothing", type=float, default=0.3,
                        help="Action smoothing factor (0-1)")
    parser.add_argument("--gripper_threshold", type=float, default=0.5,
                        help="Gripper open/close threshold")
    
    args = parser.parse_args()
    
    # Create runner
    runner = XArm6PolicyRunner(
        robot_ip=args.robot_ip,
        policy_url=args.policy_url,
        camera_id=args.camera_id,
        control_freq=args.control_freq,
        action_smoothing=args.action_smoothing,
        gripper_threshold=args.gripper_threshold,
    )
    
    # Run policy
    runner.run(prompt=args.prompt, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
