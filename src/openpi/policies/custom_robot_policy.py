"""
Custom robot policy classes for OpenPI.

This module defines the input/output transforms for a custom robot with:
- 6-DOF arm
- 6-DOF dexterous hand
- Total: 12 action dimensions

These classes are used during both training and inference to convert between
robot-specific data formats and the OpenPI model's expected format.

Author: OpenPI
Date: 2025
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_custom_robot_example() -> dict:
    """Creates a random input example for testing the custom robot policy."""
    return {
        "observation/state": np.random.rand(12),  # 6 arm + 6 hand joints
        "observation/camera_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H, W, C) format.
    
    LeRobot stores images as float32 (C, H, W), but OpenPI expects uint8 (H, W, C).
    This function handles the conversion.
    
    Args:
        image: Input image in various formats
        
    Returns:
        Image as uint8 (H, W, C) array
    """
    image = np.asarray(image)
    
    # Convert float to uint8
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # Convert (C, H, W) to (H, W, C)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    
    return image


@dataclasses.dataclass(frozen=True)
class CustomRobotInputs(transforms.DataTransformFn):
    """
    Converts robot observations to model input format.
    
    Used during both training and inference. Modify this class to match your
    robot's sensor configuration and data format.
    
    Key responsibilities:
    - Parse images to correct format
    - Map camera views to model inputs
    - Prepare state/action data
    - Handle image masking for missing views
    
    Args:
        model_type: The type of model being used (PI0, PI05, or PI0_FAST)
    """
    
    model_type: _model.ModelType
    
    def __call__(self, data: dict) -> dict:
        """Convert robot data to model input format.
        
        Args:
            data: Dictionary containing:
                - "observation/state": (12,) float32 - joint positions
                - "observation/camera_0": (H, W, 3) uint8 - third-person view
                - "observation/wrist_camera": (H, W, 3) uint8 - wrist view [optional]
                - "observation/camera_1": (H, W, 3) uint8 - additional view [optional]
                - "actions": (12,) float32 - actions [training only]
                - "prompt": str - language instruction
        
        Returns:
            Dictionary in model format with keys:
                - "state": proprio state
                - "image": dict of camera views
                - "image_mask": dict of boolean masks
                - "actions": action targets [training only]
                - "prompt": language instruction
        """
        # Parse images to uint8 (H, W, C) format
        # Modify these keys to match your camera names in the LeRobot dataset
        camera_0 = _parse_image(data["observation/camera_0"])
        
        # Handle optional wrist camera
        if "observation/wrist_camera" in data:
            wrist_camera = _parse_image(data["observation/wrist_camera"])
            has_wrist = True
        else:
            wrist_camera = np.zeros_like(camera_0)
            has_wrist = False
        
        # Handle optional second exterior camera
        if "observation/camera_1" in data:
            camera_1 = _parse_image(data["observation/camera_1"])
            has_camera_1 = True
        else:
            camera_1 = np.zeros_like(camera_0)
            has_camera_1 = False
        
        # Create model inputs dict
        # DO NOT change these keys - they are expected by the model
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": camera_0,  # Primary third-person view
                "left_wrist_0_rgb": wrist_camera,  # Wrist camera
                "right_wrist_0_rgb": camera_1,  # Additional view or padding
            },
            "image_mask": {
                "base_0_rgb": np.True_,  # Always present
                "left_wrist_0_rgb": np.True_ if has_wrist else np.False_,
                # PI0_FAST requires all images, PI0/PI05 can mask missing ones
                "right_wrist_0_rgb": (
                    np.True_ if has_camera_1 or self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }
        
        # Add actions (only present during training)
        if "actions" in data:
            inputs["actions"] = data["actions"]
        
        # Add language prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class CustomRobotOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to robot action format.
    
    Used during inference only. The model outputs actions padded to its action
    dimension (e.g., 32), but your robot only needs the first N dimensions.
    
    Key responsibilities:
    - Remove action padding
    - Format actions for robot control
    """
    
    def __call__(self, data: dict) -> dict:
        """Convert model predictions to robot action format.
        
        Args:
            data: Dictionary containing:
                - "actions": (action_horizon, 32) float32 - padded model predictions
        
        Returns:
            Dictionary with:
                - "actions": (action_horizon, 12) float32 - actual robot actions
        """
        # Extract only the first 12 actions (remove padding)
        # The model pads actions to a fixed dimension (e.g., 32), but your
        # robot only uses the first 12 dimensions
        #
        # MODIFY THIS: Change 12 to your actual action dimension
        # - If you have 6-DOF arm + 6-DOF hand = 12
        # - If you have different configuration, adjust accordingly
        #
        # Action layout (example):
        # [0:6]   - arm joint velocities/positions
        # [6:12]  - hand joint velocities/positions
        robot_actions = np.asarray(data["actions"][:, :12])
        
        return {"actions": robot_actions}
