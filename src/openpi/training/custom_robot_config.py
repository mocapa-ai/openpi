"""
Training configuration for custom robot fine-tuning.

This module defines the data processing pipeline and training configuration
for fine-tuning π₀.₅ on your custom robot dataset.

Key components:
1. CustomRobotDataConfig: Defines how to load and process your robot data
2. CUSTOM_ROBOT_PI05_CONFIG: Complete training configuration

To use this config:
1. Update repo_id to match your LeRobot dataset name
2. Register in src/openpi/training/config.py
3. Compute norm stats: uv run scripts/compute_norm_stats.py --config-name custom_robot_pi05
4. Start training: uv run scripts/train.py custom_robot_pi05 --exp-name=my_experiment

Author: OpenPI
Date: 2025
"""

import dataclasses
import pathlib

from openpi.models import pi0_config
from openpi.policies import custom_robot_policy
from openpi.shared import download as _download
from openpi.training import config as _config
from openpi.training import weight_loaders
from openpi import transforms as _transforms


@dataclasses.dataclass(frozen=True)
class CustomRobotDataConfig(_config.DataConfigFactory):
    """
    Data configuration for custom robot dataset.
    
    This class defines how your robot data is loaded and processed for training.
    It handles:
    - Loading data from LeRobot dataset
    - Applying robot-specific transforms
    - Normalizing states and actions
    - Converting between absolute/delta actions
    
    Args:
        repo_id: LeRobot dataset name (format: "username/dataset_name")
        use_absolute_actions: If True, assumes actions are absolute positions
                              and converts to deltas. If False, assumes actions
                              are already in delta/velocity form.
        assets: Configuration for loading normalization stats
    """
    
    # MODIFY THIS: Set to your LeRobot dataset name
    # This should match the output_name you used in convert_hdf5_to_lerobot.py
    repo_id: str = "your_username/custom_robot_dataset"
    
    # MODIFY THIS: Set based on your action representation
    # - True: Your actions are absolute joint positions → will convert to deltas
    # - False: Your actions are already deltas/velocities → no conversion needed
    use_absolute_actions: bool = False
    
    def create(self, assets_dirs: pathlib.Path, model_config: _config._model.BaseModelConfig) -> _config.DataConfig:
        """Create the data configuration for training.
        
        This method is called by the training script to set up the data pipeline.
        You typically don't need to call this directly.
        
        Args:
            assets_dirs: Path to assets directory (for norm stats)
            model_config: Model configuration
            
        Returns:
            Complete data configuration
        """
        
        # Step 1: Define repack transforms
        # These map your LeRobot dataset keys to the keys expected by your policy
        # MODIFY THIS: Update the keys to match your LeRobot dataset
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # Map LeRobot keys -> policy keys
                        # Left side: keys in your LeRobot dataset
                        # Right side: keys expected by CustomRobotInputs
                        "observation/camera_0": "camera_0",
                        "observation/wrist_camera": "wrist_camera",  # Optional
                        # "observation/camera_1": "camera_1",  # Uncomment if you have this
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",  # Language instruction
                    }
                )
            ]
        )
        
        # Step 2: Define data transforms
        # These convert robot-specific data to model format
        data_transforms = _transforms.Group(
            inputs=[custom_robot_policy.CustomRobotInputs(model_type=model_config.model_type)],
            outputs=[custom_robot_policy.CustomRobotOutputs()],
        )
        
        # Step 3: Handle absolute vs delta actions
        # π₀.₅ is trained on delta actions (changes relative to current state)
        # If your data has absolute positions, we need to convert them
        if self.use_absolute_actions:
            # MODIFY THIS: Define which dimensions should be converted to deltas
            # 
            # Example 1: Convert all 12 dimensions to deltas
            # delta_action_mask = _transforms.make_bool_mask(12)
            #
            # Example 2: Convert first 6 (arm) to deltas, keep last 6 (hand) absolute
            # delta_action_mask = _transforms.make_bool_mask(6, -6)
            #
            # Example 3: Convert first 11 to deltas, keep last 1 (gripper) absolute
            # delta_action_mask = _transforms.make_bool_mask(11, -1)
            #
            # The mask is a boolean array where True = convert to delta, False = keep absolute
            
            # Default: Convert all but last dimension (assuming last is gripper)
            delta_action_mask = _transforms.make_bool_mask(11, -1)
            
            # Add delta conversion transforms
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        # Step 4: Define model transforms
        # These are standard transforms for π₀.₅ (tokenization, resizing, etc.)
        # You don't need to modify this
        model_transforms = _config.ModelTransformFactory()(model_config)
        
        # Step 5: Return complete data config
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# Training configuration for π₀.₅ on custom robot
CUSTOM_ROBOT_PI05_CONFIG = _config.TrainConfig(
    name="custom_robot_pi05",
    
    # Project and experiment naming
    project_name="openpi",  # Weights & Biases project name
    
    # Model configuration
    model=pi0_config.Pi0Config(
        # Model architecture
        model_type=_config.ModelType.PI05,
        
        # Action parameters
        # MODIFY THIS: Set to your robot's action dimension
        action_dim=12,  # 6-DOF arm + 6-DOF hand
        action_horizon=50,  # Predict 50 timesteps into future
        
        # Vision parameters
        max_token_len=256,  # Max tokens for language + vision
        
        # Training precision
        dtype="bfloat16",  # Use bfloat16 for memory efficiency
    ),
    
    # Weight initialization
    # This loads pre-trained π₀.₅ weights as starting point
    weight_loader=weight_loaders.PretrainedWeightLoader(
        # MODIFY THIS: Choose your base model
        # - "pi05_base": General base model (recommended for most cases)
        # - "pi05_droid": Pre-trained on DROID data (if your robot is similar to Franka)
        # - "pi05_libero": Pre-trained on LIBERO (if doing tabletop tasks)
        pretrained_dir=_download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"),
    ),
    
    # Training hyperparameters
    lr_schedule=_config._optimizer.CosineDecaySchedule(
        warmup_steps=500,  # Linear warmup for first 500 steps
        peak_lr=3e-5,  # Peak learning rate (LoRA) or 1e-5 (full fine-tuning)
        total_steps=20_000,  # Total training steps
    ),
    
    optimizer=_config._optimizer.AdamW(
        weight_decay=0.01,  # L2 regularization
    ),
    
    # Exponential moving average of weights (improves stability)
    ema_decay=0.99,
    
    # Data configuration
    data=CustomRobotDataConfig(),
    
    # Training settings
    batch_size=32,  # Global batch size (reduce if OOM errors)
    num_workers=4,  # Data loading workers (increase for faster loading)
    seed=42,  # Random seed for reproducibility
    
    # Checkpointing
    save_freq=5_000,  # Save checkpoint every N steps
    max_steps=20_000,  # Total training steps
    
    # Evaluation
    eval_freq=1_000,  # Evaluate every N steps
    
    # LoRA fine-tuning (recommended)
    # To disable LoRA and do full fine-tuning, set use_lora=False via CLI
    # Example: uv run scripts/train.py custom_robot_pi05 --use-lora=false
)


# Optional: Configuration for full fine-tuning (requires more GPU memory)
CUSTOM_ROBOT_PI05_FULL_CONFIG = dataclasses.replace(
    CUSTOM_ROBOT_PI05_CONFIG,
    name="custom_robot_pi05_full",
    
    # Lower learning rate for full fine-tuning
    lr_schedule=_config._optimizer.CosineDecaySchedule(
        warmup_steps=500,
        peak_lr=1e-5,  # Lower LR for full fine-tuning
        total_steps=20_000,
    ),
    
    # Might need smaller batch size for memory
    batch_size=16,
)


# Quick test configuration for debugging
CUSTOM_ROBOT_PI05_DEBUG_CONFIG = dataclasses.replace(
    CUSTOM_ROBOT_PI05_CONFIG,
    name="custom_robot_pi05_debug",
    
    # Smaller settings for faster debugging
    batch_size=4,
    max_steps=100,
    save_freq=50,
    eval_freq=25,
    
    # Use fake data for testing
    data=_config.FakeDataConfig(),
)
