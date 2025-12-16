"""
FastUMI policy input/output transforms for xArm6.

Defines how to map between LeRobot dataset format and the model's expected format.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_fastumi_xarm6_example() -> dict:
    """Creates a random input example for the FastUMI xArm6 policy."""
    return {
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),  # 6 joints + 1 gripper
        "prompt": "pick up the red cube",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FastUMIXArm6Inputs(transforms.DataTransformFn):
    """Transform FastUMI xArm6 data to model input format."""
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Get state (joint positions + gripper)
        state = np.asarray(data["observation/joint_position"])  # (7,)
        
        # Parse image
        front_image = _parse_image(data["observation/front_image"])
        
        # Create padding images (model expects 3 images for pi0/pi05)
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                # Only front camera is real, others are padding
                images = (front_image, np.zeros_like(front_image), np.zeros_like(front_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (front_image, np.zeros_like(front_image), np.zeros_like(front_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        # Add actions if present (for training)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        
        # Add prompt if present
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class FastUMIXArm6Outputs(transforms.DataTransformFn):
    """Transform model output to FastUMI xArm6 action format."""
    
    def __call__(self, data: dict) -> dict:
        # Return 7D actions (6 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}
