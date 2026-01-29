"""
Convert aligned HDF5 robot data to LeRobot format for OpenPI/Pi0.5 training.

This script converts aligned HDF5 files (from align_proprioception_pi.py) 
to the LeRobot dataset format used by OpenPI for training.

Usage:
    cd ~/openpi
    uv run examples/airbot/convert_aligned_to_lerobot.py \
        --data_dir /path/to/aligned/hdf5/files \
        --output_name "your_username/robot_dataset" \
        --fps 10

Input Format (from align_proprioception_pi.py):
        /data/demo_0/
            obs/
                qpos               # (T, N) - Combined arm+hand positions
                images/top         # (T, H, W, 3) - RGB 'top' camera view
                images/wrist       # (T, H, W, 3) - RGB 'wrist' camera view (optional)
            action               # (T, N) - Combined arm+hand actions
            timestamps           # (T,) - Frame timestamps
            task (attribute)     # Language instruction

Output Format (LeRobot):
        Saved to $HF_LEROBOT_HOME/your_username/robot_dataset/
        Structure:
            - state: qpos from input
            - actions: actions from input
            - image_top: top camera images (if requested)
            - image_wrist: wrist camera images (if requested)
            - task: language instruction

Author: Data Collection Pipeline
Date: 2025
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

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


def load_aligned_hdf5(hdf5_path: Path, camera_names: Optional[List[str]] = None) -> dict:
    """Load a single aligned episode from HDF5 file.
    
    Expected structure (from align_proprioception_pi.py):
    /data/demo_0/
      obs/
        qpos              # (T, N) float32
        images/top        # (T, H, W, 3) uint8
      action              # (T, N) float32
      timestamps          # (T,) float64
      task (attribute)    # str
    
    Args:
        hdf5_path: Path to aligned HDF5 file
    
    Returns:
        Dictionary containing:
            - qpos: (T, N) array
            - actions: (T, N) array
            - images: (T, H, W, 3) array
            - language: str
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Check for pi05 compatible format
        if 'pi05_compatible' not in f.attrs or not f.attrs['pi05_compatible']:
            print(f"[WARNING] {hdf5_path.name} may not be in Pi0.5 format. "
                  "Make sure you used align_proprioception_pi.py")
        
        if 'data' not in f or 'demo_0' not in f['data']:
            raise ValueError(
                f"Invalid format in {hdf5_path}. Expected /data/demo_0/ structure. "
                "Did you use align_proprioception_pi.py?"
            )
        
        demo_grp = f['data']['demo_0']
        obs_grp = demo_grp['obs']
        
        # Load data
        qpos = np.array(obs_grp['qpos'])
        actions = np.array(demo_grp['action'])

        # Load images for requested camera streams. If camera_names is None,
        # try to load the 'top' stream as a sensible default (backwards
        # compatibility with existing single-camera data).
        images = {}
        if camera_names is None:
            camera_names = ['top']

        for cam in camera_names:
            if 'images' not in obs_grp or cam not in obs_grp['images']:
                raise ValueError(f"Camera stream '{cam}' not found in {hdf5_path}")
            images[cam] = np.array(obs_grp['images'][cam])
        
        # Load language instruction
        if 'task' in demo_grp.attrs:
            language = demo_grp.attrs['task']
        elif 'task' in f.attrs:
            language = f.attrs['task']
        else:
            language = "Perform task"
            print(f"[WARNING] No language instruction found in {hdf5_path}, using default")
        
        if isinstance(language, bytes):
            language = language.decode('utf-8')
    
    return {
        'qpos': qpos,
        'actions': actions,
        'images': images,  # dict: camera_name -> (T, H, W, 3)
        'language': language
    }


def main(
    data_dir: str,
    output_name: str = "your_username/airbot_robot_dataset",
    fps: int = 30,
    image_size: tuple[int, int] = (224, 224),
    push_to_hub: bool = False,
    overwrite: bool = True,
    cameras: Optional[List[str]] = None,
):
    """Convert aligned HDF5 robot data to LeRobot format.
    
    Args:
        data_dir: Directory containing aligned HDF5 files (from align_proprioception_pi.py)
        output_name: Name for the output dataset (format: "username/dataset_name")
        fps: Frames per second of the robot data
        image_size: Target image size (width, height)
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
    
    # Find all aligned HDF5 files
    hdf5_files = sorted(data_dir.glob("*_pi05_*.h5")) + sorted(data_dir.glob("*_pi05_*.hdf5"))
    
    if not hdf5_files:
        print(f"[WARNING] No *_pi05_*.h5 files found in {data_dir}")
        print(f"[WARNING] Looking for any .h5/.hdf5 files instead...")
        hdf5_files = sorted(data_dir.glob("*.h5")) + sorted(data_dir.glob("*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {data_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files to convert")
    
    # Parse cameras list: if user provided a comma-separated string, split it.
    # Default to two views commonly used in OpenPI collection: 'top' and 'wrist'.
    if isinstance(cameras, str):
        cameras = [c.strip() for c in cameras.split(',') if c.strip()]
    if cameras is None:
        cameras = ['top', 'wrist']

    # Load first episode to determine dimensions
    first_episode = load_aligned_hdf5(hdf5_files[0], camera_names=cameras)
    state_dim = first_episode['qpos'].shape[1]
    action_dim = first_episode['actions'].shape[1]
    
    print(f"Detected dimensions:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    # Create feature specification for LeRobot dataset
    # Pi0.5 expects: "observation/state" -> "state". For images we support
    # one or more camera streams; each camera becomes its own image feature
    # named `image_<camera_name>` (e.g. image_top, image_wrist).

    features = {
        'state': {
            'dtype': 'float32',
            'shape': (state_dim,),
            'names': ['state'],
        },
        'actions': {
            'dtype': 'float32',
            'shape': (action_dim,),
            'names': ['actions'],
        },
    }

    # Add one image feature per camera
    for cam in cameras:
        key = f'image_{cam}'
        features[key] = {
            'dtype': 'image',
            'shape': (image_size[1], image_size[0], 3),
            'names': ['height', 'width', 'channel'],
        }
    
    print(f"Creating LeRobot dataset with features: {list(features.keys())}")
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_name,
        robot_type="airbot",
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Convert each episode
    for hdf5_path in tqdm(hdf5_files, desc="Converting episodes"):
        try:
            episode_data = load_aligned_hdf5(hdf5_path, camera_names=cameras)
            
            # Get episode length
            T = len(episode_data['qpos'])
            
            # Verify all arrays have same length
            assert len(episode_data['actions']) == T, \
                f"Actions have {len(episode_data['actions'])} steps, expected {T}"
            # For each requested camera, make sure image lengths match T
            for cam in cameras:
                if cam not in episode_data['images']:
                    raise ValueError(f"Camera '{cam}' missing in {hdf5_path}")
                assert len(episode_data['images'][cam]) == T, \
                    f"Images for camera '{cam}' have {len(episode_data['images'][cam])} frames, expected {T}"
            
            # Add each frame to the dataset
            for t in range(T):
                frame_dict = {
                    'state': episode_data['qpos'][t].astype(np.float32),
                    'actions': episode_data['actions'][t].astype(np.float32),
                    'task': episode_data['language'],  # Use "task" like LIBERO
                }

                # Add each camera image under its feature key
                for cam in cameras:
                    key = f'image_{cam}'
                    frame_dict[key] = resize_image(episode_data['images'][cam][t], image_size)
                
                dataset.add_frame(frame_dict)
            
            dataset.save_episode()
            
        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDataset created successfully at: {output_path}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    
    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["robotics", "airbot_robot", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to https://huggingface.co/datasets/{output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert aligned HDF5 robot data to LeRobot format")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing aligned HDF5 files (from align_proprioception_pi.py)")
    parser.add_argument("--output_name", type=str, default="your_username/airbot_robot_dataset",
                        help="Name for output dataset (format: username/dataset_name)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second of the robot data")
    parser.add_argument("--image_width", type=int, default=224,
                        help="Target image width")
    parser.add_argument("--image_height", type=int, default=224,
                        help="Target image height")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push dataset to Hugging Face Hub")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="Don't overwrite existing dataset")
    parser.add_argument("--cameras", type=str, default="top,wrist",
                        help="Comma-separated camera names under obs/images (e.g. top,wrist)")
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_name=args.output_name,
        fps=args.fps,
        image_size=(args.image_width, args.image_height),
        push_to_hub=args.push_to_hub,
        overwrite=not args.no_overwrite,
        cameras=args.cameras,
    )
