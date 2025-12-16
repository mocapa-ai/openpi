"""
Convert HDF5 robot data to LeRobot format for OpenPI training.

This script converts robot demonstration data stored in HDF5 format to the LeRobot
dataset format used by OpenPI for training vision-language-action models.

Usage:
    uv run examples/custom_robot/convert_hdf5_to_lerobot.py \
        --data_dir /path/to/hdf5/files \
        --output_name "your_username/robot_dataset" \
        --fps 10

Requirements:
    - HDF5 files with robot demonstrations
    - At least one camera view (third-person or wrist)
    - State observations (joint positions)
    - Actions (joint velocities or positions)
    - Language instructions

HDF5 Structure Expected:
    episode_0/
      ├── observations/
      │   ├── images/
      │   │   ├── camera_0: (T, H, W, 3) uint8
      │   │   ├── camera_1: (T, H, W, 3) uint8 [optional]
      │   │   └── wrist_camera: (T, H, W, 3) uint8 [optional]
      │   └── state: (T, N) float32  # Joint positions/velocities
      ├── actions: (T, N) float32
      └── language_instruction: str

Author: OpenPI
Date: 2025
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image using PIL for high quality.
    
    Args:
        image: Input image array (H, W, 3)
        size: Target size (width, height)
    
    Returns:
        Resized image array
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(size, resample=Image.BICUBIC)
    return np.array(resized)


def load_hdf5_episode(hdf5_path: Path) -> dict:
    """Load a single episode from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Dictionary containing episode data with keys:
            - images: dict of camera name -> (T, H, W, 3) array
            - state: (T, N) array
            - actions: (T, N) array
            - language: str
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load images from all available cameras
        images = {}
        if 'observations/images' in f:
            obs_images = f['observations/images']
            for camera_name in obs_images.keys():
                images[camera_name] = np.array(obs_images[camera_name])
        
        # Load state
        if 'observations/state' in f:
            state = np.array(f['observations/state'])
        else:
            raise ValueError(f"No state found in {hdf5_path}")
        
        # Load actions
        if 'actions' in f:
            actions = np.array(f['actions'])
        else:
            raise ValueError(f"No actions found in {hdf5_path}")
        
        # Load language instruction
        if 'language_instruction' in f:
            language = f['language_instruction'][()]
            if isinstance(language, bytes):
                language = language.decode('utf-8')
        elif 'task' in f:
            language = f['task'][()]
            if isinstance(language, bytes):
                language = language.decode('utf-8')
        else:
            language = "Perform task"  # Default if no language provided
            print(f"Warning: No language instruction found in {hdf5_path}, using default")
    
    return {
        'images': images,
        'state': state,
        'actions': actions,
        'language': language
    }


def main(
    data_dir: str,
    output_name: str = "your_username/custom_robot_dataset",
    fps: int = 10,
    image_size: tuple[int, int] = (320, 180),
    state_dim: int = 12,
    action_dim: int = 12,
    push_to_hub: bool = False,
    overwrite: bool = True,
):
    """Convert HDF5 robot data to LeRobot format.
    
    Args:
        data_dir: Directory containing HDF5 files
        output_name: Name for the output dataset (format: "username/dataset_name")
        fps: Frames per second of the robot data
        image_size: Target image size (width, height)
        state_dim: Dimension of state vector (e.g., 12 for 6-DOF arm + 6-DOF hand)
        action_dim: Dimension of action vector
        push_to_hub: Whether to push to Hugging Face Hub
        overwrite: Whether to overwrite existing dataset
    """
    data_dir = Path(data_dir)
    output_path = HF_LEROBOT_HOME / output_name
    
    # Clean up existing dataset if overwrite is True
    if output_path.exists() and overwrite:
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    elif output_path.exists():
        raise ValueError(f"Dataset already exists at {output_path}. Set overwrite=True to replace.")
    
    # Find all HDF5 files
    hdf5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {data_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files to convert")
    
    # Load first episode to determine camera configuration
    first_episode = load_hdf5_episode(hdf5_files[0])
    camera_names = list(first_episode['images'].keys())
    print(f"Detected cameras: {camera_names}")
    
    # Create feature specification for LeRobot dataset
    features = {}
    
    # Add camera features
    for camera_name in camera_names:
        # Map camera names to OpenPI conventions
        # You may need to modify this mapping based on your camera setup
        if 'wrist' in camera_name.lower():
            lerobot_name = "wrist_camera"
        elif 'camera_0' in camera_name or 'cam_0' in camera_name:
            lerobot_name = "camera_0"
        elif 'camera_1' in camera_name or 'cam_1' in camera_name:
            lerobot_name = "camera_1"
        else:
            lerobot_name = camera_name
        
        features[lerobot_name] = {
            "dtype": "image",
            "shape": (image_size[1], image_size[0], 3),  # (height, width, channels)
            "names": ["height", "width", "channel"],
        }
    
    # Add state feature
    features["state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["state"],
    }
    
    # Add action feature
    features["actions"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["actions"],
    }
    
    print(f"Creating LeRobot dataset with features: {list(features.keys())}")
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_name,
        robot_type="custom",  # You can change this to your robot type
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Convert each episode
    for hdf5_path in tqdm(hdf5_files, desc="Converting episodes"):
        try:
            episode_data = load_hdf5_episode(hdf5_path)
            
            # Get episode length
            T = len(episode_data['state'])
            
            # Verify all arrays have same length
            for camera_name, images in episode_data['images'].items():
                assert len(images) == T, f"Camera {camera_name} has {len(images)} frames, expected {T}"
            assert len(episode_data['actions']) == T, f"Actions have {len(episode_data['actions'])} steps, expected {T}"
            
            # Add each frame to the dataset
            for t in range(T):
                frame_dict = {
                    "state": episode_data['state'][t].astype(np.float32),
                    "actions": episode_data['actions'][t].astype(np.float32),
                    "task": episode_data['language'],
                }
                
                # Add resized images
                for camera_name, images in episode_data['images'].items():
                    # Map to LeRobot camera names
                    if 'wrist' in camera_name.lower():
                        lerobot_name = "wrist_camera"
                    elif 'camera_0' in camera_name or 'cam_0' in camera_name:
                        lerobot_name = "camera_0"
                    elif 'camera_1' in camera_name or 'cam_1' in camera_name:
                        lerobot_name = "camera_1"
                    else:
                        lerobot_name = camera_name
                    
                    frame_dict[lerobot_name] = resize_image(images[t], image_size)
                
                dataset.add_frame(frame_dict)
            
            dataset.save_episode()
            
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            continue
    
    print(f"\nDataset created successfully at: {output_path}")
    print(f"Total episodes: {len(dataset.episodes)}")
    print(f"Total frames: {len(dataset)}")
    
    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["robotics", "custom_robot", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to https://huggingface.co/datasets/{output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 robot data to LeRobot format")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing HDF5 files")
    parser.add_argument("--output_name", type=str, default="your_username/custom_robot_dataset",
                        help="Name for output dataset (format: username/dataset_name)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second of the robot data")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Target image width")
    parser.add_argument("--image_height", type=int, default=180,
                        help="Target image height")
    parser.add_argument("--state_dim", type=int, default=12,
                        help="Dimension of state vector")
    parser.add_argument("--action_dim", type=int, default=12,
                        help="Dimension of action vector")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push dataset to Hugging Face Hub")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Don't overwrite existing dataset")
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_name=args.output_name,
        fps=args.fps,
        image_size=(args.image_width, args.image_height),
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        push_to_hub=args.push_to_hub,
        overwrite=not args.no_overwrite,
    )
