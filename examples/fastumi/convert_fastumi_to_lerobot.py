"""
Convert FastUMI dataset to LeRobot format for openpi training.

This script converts FastUMI HDF5 data (TCP poses) to joint angles using inverse
kinematics and transforms it into LeRobot dataset format.

Usage:
    uv run examples/fastumi/convert_fastumi_to_lerobot.py \
        --data_dir data/fastumi_raw \
        --config_path examples/fastumi/config.json \
        --output_repo your_hf_username/fastumi_xarm6_pickplace

Optional:
    --push_to_hub: Push the converted dataset to HuggingFace Hub
"""

import json
import shutil
from pathlib import Path

import cv2
import h5py
import ikpy.chain
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def resize_image(image: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """Resize image to target size."""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def tcp_to_joint_angles(
    position: np.ndarray,
    quaternion: np.ndarray,
    chain: ikpy.chain.Chain,
    initial_joint_angles: np.ndarray = None,
    flange_to_tcp_distance: float = 0.0
) -> np.ndarray:
    """
    Convert TCP pose to joint angles using inverse kinematics.
    
    Args:
        position: [x, y, z] TCP position
        quaternion: [qx, qy, qz, qw] TCP orientation
        chain: IKPy kinematic chain
        initial_joint_angles: Starting guess for IK solver
        flange_to_tcp_distance: Distance from flange to TCP along negative Z-axis
    
    Returns:
        Joint angles array (includes fixed base joint at index 0)
    """
    # Adjust position to account for flange-to-TCP offset
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]
    adjusted_position = position - flange_to_tcp_distance * z_axis
    
    if initial_joint_angles is None:
        initial_joint_angles = [0] * len(chain.links)
    
    # Compute IK
    joint_angles = chain.inverse_kinematics(
        adjusted_position,
        rotation_matrix,
        orientation_mode='all',
        initial_position=initial_joint_angles
    )
    
    return joint_angles


def transform_tcp_to_base_frame(
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float,
    base_transform: np.ndarray,
    offset: dict
) -> tuple:
    """
    Transform TCP pose from local frame to base frame.
    
    Args:
        x, y, z: Position in local frame
        qx, qy, qz, qw: Orientation quaternion in local frame
        base_transform: 4x4 transformation matrix from base to local
        offset: dict with 'x' and 'z' offsets for T265 to TCP
    
    Returns:
        (x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base)
    """
    # Apply offset (T265 to TCP)
    x_offset = x + offset['x']
    z_offset = z + offset['z']
    
    # Build local transformation matrix
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x_offset, y, z_offset]
    
    # Transform to base frame
    T_base_r = np.dot(T_local[:3, :3], base_transform[:3, :3])
    pos_base = base_transform[:3, 3] + T_local[:3, 3]
    
    # Convert back to quaternion
    rotation_base = R.from_matrix(T_base_r)
    quat_base = rotation_base.as_quat()  # [qx, qy, qz, qw]
    
    return (*pos_base, *quat_base)


def detect_gripper_width(
    images: np.ndarray,
    config: dict
) -> np.ndarray:
    """
    Detect gripper width from ArUco markers in images.
    
    Args:
        images: (T, H, W, 3) array of RGB images
        config: Configuration dict with ArUco parameters
    
    Returns:
        Array of gripper widths (T,) in mm, interpolated for missing detections
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config['aruco_dict'])
    )
    parameters = cv2.aruco.DetectorParameters()
    
    distances = []
    distances_indices = []
    
    for i in range(len(images)):
        gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if ids is not None:
            marker_centers = []
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id in [config['marker_id_0'], config['marker_id_1']]:
                    marker_corners = corners[idx][0]
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)
            
            if len(marker_centers) >= 2:
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])
                distances.append(distance)
                distances_indices.append(i)
            elif len(marker_centers) == 1:
                # Single marker visible: estimate from image center
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2
                distances.append(distance)
                distances_indices.append(i)
    
    if len(distances) == 0:
        # No markers detected at all - return default closed gripper
        return np.zeros(len(images), dtype=np.float32)
    
    # Convert pixel distances to gripper widths
    distances = np.array(distances)
    marker_min = config['gripper']['marker_min']
    marker_max = config['gripper']['marker_max']
    gripper_max = config['gripper']['gripper_max']
    
    gripper_widths = ((distances - marker_min) / (marker_max - marker_min) * gripper_max)
    gripper_widths = gripper_widths.clip(0, gripper_max)
    
    # Interpolate for missing frames
    result = np.zeros(len(images), dtype=np.float32)
    for i in range(len(distances)):
        if i == 0:
            # Fill from start to first detection
            start_idx = 0
            end_idx = distances_indices[0]
            result[start_idx:end_idx + 1] = gripper_widths[0]
        else:
            # Interpolate between detections
            start_idx = distances_indices[i - 1]
            end_idx = distances_indices[i]
            if end_idx - start_idx == 1:
                result[end_idx] = gripper_widths[i]
            else:
                result[start_idx:end_idx + 1] = np.linspace(
                    gripper_widths[i - 1],
                    gripper_widths[i],
                    end_idx - start_idx + 1
                )
    
    # Fill remaining frames with last value
    if len(distances_indices) > 0:
        last_idx = distances_indices[-1]
        result[last_idx:] = gripper_widths[-1]
    
    # Convert to normalized gripper state (0 = closed, 1 = open)
    normalized_gripper = result / gripper_max
    
    return normalized_gripper


def process_episode(
    hdf5_path: Path,
    chain: ikpy.chain.Chain,
    config: dict,
    base_transform: np.ndarray
) -> dict:
    """
    Process a single FastUMI episode.
    
    Returns dict with:
        - images: (T, 224, 224, 3) uint8
        - joint_positions: (T, 7) float32
        - actions: (T, 7) float32
        - task: str (optional)
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load observations
        images = f['observations/images/front'][:]  # (T, 1920, 1080, 3)
        tcp_poses = f['observations/qpos'][:]  # (T, 7) [x, y, z, qx, qy, qz, qw]
        
        T = len(images)
        
        # Resize images
        images_resized = np.array([resize_image(img) for img in images])
        
        # Detect gripper widths
        gripper_widths = detect_gripper_width(images, config)
        
        # Convert TCP poses to joint angles
        joint_angles_list = []
        prev_joints = np.array(config['start_qpos'])
        
        for i in range(T):
            # Transform TCP pose to base frame
            tcp_base = transform_tcp_to_base_frame(
                *tcp_poses[i],
                base_transform,
                config['offset']
            )
            
            # Compute IK
            position = tcp_base[:3]
            quaternion = tcp_base[3:]
            
            try:
                joint_angles = tcp_to_joint_angles(
                    position,
                    quaternion,
                    chain,
                    initial_joint_angles=prev_joints,
                    flange_to_tcp_distance=config['gripper']['flange_to_tcp']
                )
                # Remove fixed base joint (first element)
                joint_angles = joint_angles[1:]
                prev_joints = joint_angles
            except Exception as e:
                # IK failed - use previous joint angles
                joint_angles = prev_joints
            
            # Append gripper state
            joints_with_gripper = np.append(joint_angles[:6], gripper_widths[i])
            joint_angles_list.append(joints_with_gripper)
        
        joint_positions = np.array(joint_angles_list, dtype=np.float32)
        
        # In FastUMI format, actions = current joint positions
        actions = joint_positions.copy()
        
        return {
            'images': images_resized,
            'joint_positions': joint_positions,
            'actions': actions,
            'task': 'pick and place'  # Default task name
        }


def main(
    data_dir: str,
    config_path: str,
    output_repo: str,
    push_to_hub: bool = False
):
    """
    Convert FastUMI dataset to LeRobot format.
    
    Args:
        data_dir: Path to directory containing FastUMI HDF5 files
        config_path: Path to config.json
        output_repo: Output LeRobot repo name (e.g., 'username/dataset_name')
        push_to_hub: Whether to push to HuggingFace Hub
    """
    # Load config
    config = load_config(config_path)
    
    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / output_repo
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Load kinematic chain
    chain = ikpy.chain.Chain.from_urdf_file(
        config['urdf_path'],
        base_elements=['world']
    )
    print(f"Loaded kinematic chain with {len(chain.links)} links")
    
    # Build base transformation matrix
    base_pos = config['base_position']
    base_ori = config['base_orientation']
    rotation_base = R.from_euler(
        'xyz',
        [base_ori['roll'], base_ori['pitch'], base_ori['yaw']],
        degrees=True
    ).as_matrix()
    
    base_transform = np.eye(4)
    base_transform[:3, :3] = rotation_base
    base_transform[:3, 3] = [base_pos['x'], base_pos['y'], base_pos['z']]
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=output_repo,
        robot_type="xarm6",
        fps=30,  # FastUMI typically runs at ~30 fps
        features={
            "front_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),  # 6 joints + 1 gripper
                "names": ["joint_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Find all HDF5 files
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"))
    
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {data_dir}")
    
    print(f"Found {len(hdf5_files)} episodes to convert")
    
    # Process each episode
    total_frames = 0
    failed_episodes = 0
    
    for hdf5_file in tqdm(hdf5_files, desc="Converting episodes"):
        try:
            episode_data = process_episode(hdf5_file, chain, config, base_transform)
            
            # Add frames to dataset
            for i in range(len(episode_data['images'])):
                dataset.add_frame({
                    'front_image': episode_data['images'][i],
                    'joint_position': episode_data['joint_positions'][i],
                    'actions': episode_data['actions'][i],
                    'task': episode_data['task'],
                })
            
            dataset.save_episode()
            total_frames += len(episode_data['images'])
            
        except Exception as e:
            print(f"\nFailed to process {hdf5_file.name}: {e}")
            failed_episodes += 1
    
    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"  Total episodes: {len(hdf5_files) - failed_episodes}")
    print(f"  Failed episodes: {failed_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Dataset saved to: {output_path}")
    print(f"{'='*80}")
    
    # Optionally push to hub
    if push_to_hub:
        print("Pushing dataset to HuggingFace Hub...")
        dataset.push_to_hub(
            tags=["fastumi", "xarm6", "pick-and-place"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print("Upload complete!")


if __name__ == "__main__":
    tyro.cli(main)
