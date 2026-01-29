import time
import sys
import re
import threading
import numpy as np
import cv2
import rerun as rr
import tyro
import dataclasses
from typing import Optional, Dict
from PIL import Image

# Hardware Libraries
from pyorbbecsdk import Pipeline, Context, Config, OBSensorType, OBFormat, OBFrameType
from openpi_client import websocket_client_policy as _websocket_client_policy

# Robot Library (Adjust import if your airbot_py is located elsewhere)
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile


# =============================================================================
# 1. Simplified Camera Logic (No complex states, just stream & rerun)
# =============================================================================

def process_color_frame(frame):
    """Decodes Orbbec color frame to numpy RGB array."""
    if frame is None:
        return None
    width = frame.get_width()
    height = frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    
    # Resize/Reshape based on format (Assuming MJPG or RGB)
    # If the SDK returns raw RGB:
    if frame.get_format() == OBFormat.RGB:
        data = data.reshape((height, width, 3))
    # If MJPG, decode it
    elif frame.get_format() == OBFormat.MJPG:
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    else:
        # Fallback for other formats (YUYV etc), might need specific handling
        return None
    
    data = resize_image(data, (224, 224))
    return data

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

class DualOrbbecStreamer:
    """
    Manages two Orbbec cameras (Top and Wrist).
    Constantly fetches frames in background threads and logs to Rerun.
    """
    def __init__(self, top_serial: str, wrist_serial: str):
        self.target_serials = {'top': top_serial, 'wrist': wrist_serial}
        self.pipelines = []
        self.latest_frames = {'top': None, 'wrist': None}
        self.lock = threading.Lock()
        self.ctx = Context()
        


    def connect(self):
        device_list = self.ctx.query_devices()
        dev_count = device_list.get_count()
        print(f"[CAM] Found {dev_count} devices.")

        for i in range(dev_count):
            device = device_list.get_device_by_index(i)
            
            # 1. Identify Device Serial
            serial = None
            try:
                # Regex parse the weird repr() string from Orbbec SDK
                info_repr = repr(device.get_device_info())
                m = re.search(r'serial_number=([^,\s\)]+)', info_repr)
                if m: serial = m.group(1).strip()
            except Exception:
                pass

            if not serial:
                print(f"[CAM] Could not identify serial for device {i}")
                continue

            # 2. Match to Role
            role = None
            if serial == self.target_serials['top']: role = 'top'
            elif serial == self.target_serials['wrist']: role = 'wrist'
            
            if role:
                print(f"[CAM] Connecting to {role.upper()} camera (Serial: {serial})")
                self._start_pipeline(device, role)
            else:
                print(f"[CAM] Ignoring device {serial} (not in target list)")

    def _start_pipeline(self, device, role):
        pipeline = Pipeline(device)
        config = Config()
        
        # Try enabling Color stream
        try:
            profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            profile = profiles.get_default_video_stream_profile()
            config.enable_stream(profile)
        except Exception as e:
            print(f"[CAM] Failed to enable color for {role}: {e}")
            return

        # Define Callback
        def callback(frame_set):
            color_frame = frame_set.get_color_frame()
            if color_frame:
                rgb = process_color_frame(color_frame)
                if rgb is not None:
                    # 1. Update latest frame safely
                    with self.lock:
                        self.latest_frames[role] = rgb
                    
                    # 2. Stream to Rerun immediately
                    try:
                        rr.log(f"world/camera/{role}", rr.Image(rgb), static=True)
                    except Exception:
                        pass

        pipeline.start(config, callback)
        self.pipelines.append(pipeline)

    def get_latest(self) -> Dict[str, np.ndarray]:
        """Returns the most recent frames for inference."""
        with self.lock:
            # Return a copy to avoid threading issues during inference
            return {k: v.copy() if v is not None else None for k, v in self.latest_frames.items()}

    def stop(self):
        for p in self.pipelines:
            try:
                p.stop()
            except: pass


# =============================================================================
# 2. Simplified Robot Logic
# =============================================================================

class AirBotArm:
    def __init__(self, port: str):
        self.port = port
        self.robot = None

    def connect(self):
        self.robot = AIRBOTPlay(port = self.port)
        print(f"[ROBOT] Connecting to AirBot on port {self.port}...")
        self.robot.connect()
        self.robot.set_speed_profile(SpeedProfile.SLOW)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)

    def disconnect(self):
        if self.robot: self.robot.disconnect()

    def get_state(self):
        """Returns concatenated [arm_qpos(6), gripper_pos(1)]"""
        if not self.robot: return np.zeros(7)
        
        joints = np.array(self.robot.get_joint_pos(), dtype=np.float32)
        
        # Get raw gripper pos (0.0 - 0.065)
        raw_gripper = np.array(self.robot.get_eef_pos(), dtype=np.float32)
        if raw_gripper.ndim == 0: raw_gripper = raw_gripper.reshape(1)
        
        # Map to model range (0.0 - 0.036)
        scaled_gripper = self._map_to_model(raw_gripper)
        
        return np.concatenate([joints, scaled_gripper])

    def act(self, action):
        """Expects 7D action: [arm(6), gripper(1)]"""
        if not self.robot: return
        
        arm_cmd = action[:6]
        
        # Model outputs command in 0.0 - 0.036 range
        model_gripper_cmd = action[6]
        
        # Map back to robot range (0.0 - 0.065)
        robot_gripper_cmd = self._map_to_robot(model_gripper_cmd)
        
        # Clip to ensure safety
        robot_gripper_cmd = np.clip(robot_gripper_cmd, 0.0, 0.065)

        print(f"[ROBOT] Acting: Arm: {arm_cmd}, Gripper: {robot_gripper_cmd:.4f}")        
        self.robot.servo_joint_pos(arm_cmd.tolist())
        self.robot.servo_eef_pos([robot_gripper_cmd])

    def move_home(self):
        if not self.robot: return
        home = [0.12569619715213776, -1.7263675928115845, 0.2725642919540405, -1.731708288192749, -1.2296863794326782, -0.7524605393409729]
        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.move_to_joint_pos(home, blocking=True)
        self.robot.switch_mode(RobotMode.SERVO_JOINT_POS)
        self.robot.servo_eef_pos([0.065]) # Open gripper

    def _map_to_model(self, gripper_pos):
        """
        Maps robot range [0.0, 0.065] -> model range [0.0, 0.036]
        """
        # Linear mapping: y = (x / max_robot) * max_model
        return (gripper_pos / 0.065) * 0.036

    def _map_to_robot(self, gripper_cmd):
        """
        Maps model range [0.0, 0.036] -> robot range [0.0, 0.065]
        """
        # Linear mapping: y = (x / max_model) * max_robot
        return (gripper_cmd / 0.036) * 0.065

# =============================================================================
# 3. Main Validation Loop
# =============================================================================

@dataclasses.dataclass
class Args:
    # Policy Server
    host: str = "localhost"
    port: int = 8000
    
    # Robot
    arm_port: str = "50001" # Check your USB/Serial port
    
    # Camera Serials
    top_cam_serial: str = "CP7JC42000EY"
    wrist_cam_serial: str = "CP7JC42000F4"
    
    # Tuning
    action_horizon: int =30  # How many actions to execute per inference
    control_freq: int = 30    # Hz
    
    prompt: str = "pick up the red block and place it in the bowl"

def main(args: Args):
    # 1. Setup Rerun
    rr.init("pi0_validation", spawn=True)

    # 2. Connect Hardware
    print("[SYS] Connecting to Cameras...")
    cameras = DualOrbbecStreamer(args.top_cam_serial, args.wrist_cam_serial)
    cameras.connect()
    
    # Wait a second for auto-exposure/white balance
    time.sleep(2.0) 

    print("[SYS] Connecting to Robot...")
    robot = AirBotArm(args.arm_port)
    robot.connect()
    robot.move_home()

    # 3. Connect Policy
    print(f"[SYS] Connecting to Policy Server ({args.host}:{args.port})...")
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    
    # Warmup
    print("[SYS] Warming up model...")
    policy.infer({
        "observation/image": np.zeros((224,224,3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224,224,3), dtype=np.uint8),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": args.prompt
    })

    print("\n" + "="*40)
    print(" READY TO VALIDATE")
    print("="*40)
    input("Press Enter to start episode...")

    try:
        dt = 1.0 / args.control_freq
        
        while True:
            cycle_start = time.time()

            # --- A. GET DATA ---
            frames = cameras.get_latest()
            robot_state = robot.get_state()
            
            if frames['top'] is None or frames['wrist'] is None:
                print("[WARN] Waiting for frames...", end='\r')
                time.sleep(0.1)
                continue

            # --- B. INFERENCE ---
            obs = {
                "observation/image": frames['top'],
                "observation/wrist_image": frames['wrist'],
                "observation/state": robot_state,
                "prompt": args.prompt
            }
            
            result = policy.infer(obs)
            actions = result['actions'] # Shape: [Horizon, 7]

            # --- C. EXECUTE HORIZON ---
            # We execute a chunk of actions (Horizon) open-loop to reduce inference latency effects
            chunk_size = min(args.action_horizon, len(actions))
            
            for i in range(chunk_size):
                step_start = time.time()
                
                robot.act(actions[i])
                
                # Sleep to maintain control frequency
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            print(f"[Run] Executed {chunk_size} actions", end='\r')

    except KeyboardInterrupt:
        print("\n[SYS] Stopping...")
    finally:
        cameras.stop()
        robot.disconnect()
        print("[SYS] Clean shutdown.")

if __name__ == "__main__":
    tyro.cli(main)