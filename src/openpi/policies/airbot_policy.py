"""
AirBot robot policy classes for OpenPI.

This module defines the input/output transforms for AirBot Play with:
- 6-DOF arm
- 6-DOF dexterous hand (Revo2)
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


def make_airbot_example() -> dict:
    """Creates a random input example for testing the AirBot policy."""
    return {
        "observation/state": np.random.rand(12),  # 6 arm + 6 hand joints
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class AirBotInputs(transforms.DataTransformFn):
    """
    Converts AirBot observations to model input format.
    
    Used during both training and inference. Converts AirBot's single camera
    view to the multi-camera format expected by the model.
    
    Key responsibilities:
    - Parse images to correct format
    - Map single camera to model's expected multiple views
    - Prepare state/action data
    - Handle image masking for missing views
    
    Args:
        model_type: The type of model being used (PI0, PI05, or PI0_FAST)
    """
    
    model_type: _model.ModelType
    
    def __call__(self, data: dict) -> dict:
        """Convert AirBot data to model input format.
        
        Args:
            data: Dictionary containing:
                - "observation/state": (12,) float32 - joint positions
                - "observation/image": (H, W, 3) uint8 - single third-person camera
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
        # Parse image to uint8 (H, W, C) format
        # AirBot has a single top-down camera view
        image = _parse_image(data["observation/image"])
        
        # Create model inputs dict
        # Model expects 3 camera views, so we use the same image for all
        # and mask out the ones we don't have
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": image,  # Primary third-person view
                "left_wrist_0_rgb": np.zeros_like(image),  # No wrist camera
                "right_wrist_0_rgb": np.zeros_like(image),  # No additional camera
            },
            "image_mask": {
                "base_0_rgb": np.True_,  # Always present
                "left_wrist_0_rgb": np.False_,  # Not available
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),  # PI0_FAST requires all images
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
class AirBotOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to AirBot action format.
    
    Used during inference only. The model outputs actions padded to its action
    dimension (e.g., 32), but AirBot only needs the first 12 dimensions.
    
    Key responsibilities:
    - Remove action padding
    - Format actions for robot control
    """
    
    def __call__(self, data: dict) -> dict:
        """Convert model predictions to AirBot action format.
        
        Args:
            data: Dictionary containing:
                - "actions": (action_horizon, 32) float32 - padded model predictions
        
        Returns:
            Dictionary with:
                - "actions": (action_horizon, 12) float32 - AirBot actions
        """
        # Extract only the first 12 actions (remove padding)
        # Action layout:
        # [0:6]   - arm joint positions/velocities
        # [6:12]  - hand (Revo2) joint positions/velocities
        airbot_actions = np.asarray(data["actions"][:, :7])
        
        return {"actions": airbot_actions}
