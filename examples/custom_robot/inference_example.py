"""
Example inference script for custom robot policy.

This script demonstrates how to:
1. Load a trained policy from checkpoint
2. Query the policy with observations
3. Execute actions on your robot

Usage:
    # Start policy server in one terminal
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=custom_robot_pi05 \
        --policy.dir=checkpoints/custom_robot_pi05/my_experiment/20000
    
    # Run this script in another terminal
    python examples/custom_robot/inference_example.py

Author: OpenPI
Date: 2025
"""

import numpy as np
import requests
import time
from typing import Dict, Any


# Policy server configuration
POLICY_SERVER_URL = "http://localhost:8000"


def create_dummy_observation() -> Dict[str, Any]:
    """Create a dummy observation for testing.
    
    In a real deployment, you would replace this with actual robot sensor data.
    
    Returns:
        Dictionary containing robot observations
    """
    observation = {
        # State: joint positions (6 arm + 6 hand = 12 DOF)
        "observation/state": np.random.rand(12).tolist(),
        
        # Images: RGB images from cameras (H, W, 3) uint8
        # Note: Convert numpy arrays to lists for JSON serialization
        "observation/camera_0": np.random.randint(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ).tolist(),
        
        "observation/wrist_camera": np.random.randint(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        ).tolist(),
        
        # Language instruction
        "prompt": "pick up the red cube",
    }
    
    return observation


def query_policy(observation: Dict[str, Any]) -> np.ndarray:
    """Query the policy server for actions.
    
    Args:
        observation: Robot observation dictionary
    
    Returns:
        Action array of shape (action_horizon, action_dim)
        Default: (50, 12) for 50 timesteps and 12 DOF
    """
    try:
        response = requests.post(
            f"{POLICY_SERVER_URL}/predict",
            json={"observation": observation},
            timeout=5.0,
        )
        response.raise_for_status()
        
        # Extract actions from response
        actions = np.array(response.json()["actions"])
        return actions
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying policy server: {e}")
        print(f"Make sure the policy server is running at {POLICY_SERVER_URL}")
        raise


def execute_actions_on_robot(actions: np.ndarray):
    """Execute actions on your robot.
    
    This is a placeholder function. Replace with your robot control code.
    
    Args:
        actions: Action array of shape (action_horizon, action_dim)
    
    Example integration:
        # For ROS robots
        for action in actions:
            joint_cmd = JointTrajectoryPoint()
            joint_cmd.positions = action[:6]  # Arm joints
            joint_cmd.velocities = action[6:12]  # Hand joints
            self.pub.publish(joint_cmd)
            rospy.sleep(1.0 / fps)
        
        # For PyBullet simulation
        for action in actions:
            for i, val in enumerate(action):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=val,
                )
            p.stepSimulation()
    """
    action_horizon, action_dim = actions.shape
    print(f"Executing {action_horizon} actions with dimension {action_dim}")
    
    # Placeholder: just print the actions
    for t, action in enumerate(actions):
        print(f"Step {t}: {action}")
        time.sleep(0.1)  # Simulate execution delay


def main():
    """Main inference loop."""
    print("Custom Robot Inference Example")
    print("=" * 50)
    
    # Check if policy server is running
    try:
        response = requests.get(f"{POLICY_SERVER_URL}/health", timeout=2.0)
        print(f"✓ Policy server is running")
    except:
        print(f"✗ Policy server not found at {POLICY_SERVER_URL}")
        print("Start the server first:")
        print("  uv run scripts/serve_policy.py policy:checkpoint \\")
        print("    --policy.config=custom_robot_pi05 \\")
        print("    --policy.dir=checkpoints/custom_robot_pi05/my_experiment/20000")
        return
    
    # Run inference loop
    print("\nRunning inference...")
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Get robot observation
        # In real deployment, replace with actual sensor readings
        observation = create_dummy_observation()
        print(f"Observation state shape: {np.array(observation['observation/state']).shape}")
        print(f"Language instruction: {observation['prompt']}")
        
        # Query policy for actions
        start_time = time.time()
        actions = query_policy(observation)
        inference_time = time.time() - start_time
        
        print(f"✓ Received actions: shape={actions.shape}")
        print(f"  Inference time: {inference_time*1000:.1f}ms")
        print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
        
        # Execute actions on robot
        # In real deployment, this would control your robot
        execute_actions_on_robot(actions)
    
    print("\n" + "=" * 50)
    print("Inference complete!")


if __name__ == "__main__":
    main()
