# AirBot Pi0.5 Validation Architecture

## Overview

This document explains how to validate your fine-tuned Pi0.5 model with the AirBot arm and Revo2 hand.

## Architecture: Two-Terminal Client-Server (Same as ALOHA)

OpenPI uses a **client-server architecture** where:
- **Policy Server**: Runs model inference (requires `uv` and GPU)
- **Robot Client**: Controls hardware (uses your conda environment with robot dependencies)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YOUR PC                                          │
│                                                                             │
│  ┌─────────────────────────────┐      ┌──────────────────────────────────┐ │
│  │  Terminal 1: Policy Server  │      │  Terminal 2: Robot Client        │ │
│  │  (uv environment)           │      │  (conda environment)             │ │
│  │                             │      │                                  │ │
│  │  ┌─────────────────────┐    │      │  ┌────────────────────────────┐  │ │
│  │  │   Pi0.5 Model       │    │      │  │   openpi-client            │  │ │
│  │  │   (JAX/PyTorch)     │    │      │  │   (lightweight websocket)  │  │ │
│  │  └─────────────────────┘    │      │  └────────────────────────────┘  │ │
│  │           │                 │      │              │                   │ │
│  │           ▼                 │      │              ▼                   │ │
│  │  ┌─────────────────────┐    │      │  ┌────────────────────────────┐  │ │
│  │  │ WebSocket Server    │◄───┼──────┼──│ WebSocket Client           │  │ │
│  │  │ (port 8000)         │    │      │  │                            │  │ │
│  │  └─────────────────────┘    │      │  └────────────────────────────┘  │ │
│  │                             │      │              │                   │ │
│  └─────────────────────────────┘      │              ▼                   │ │
│                                       │  ┌────────────────────────────┐  │ │
│                                       │  │   Robot Control            │  │ │
│                                       │  │   - AirBot arm (airbot_py) │  │ │
│                                       │  │   - Revo2 hand (revo2)     │  │ │
│                                       │  │   - Camera (TODO)          │  │ │
│                                       │  └────────────────────────────┘  │ │
│                                       │                                  │ │
│                                       └──────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why Two Terminals?

1. **Environment Isolation**: Policy needs JAX/PyTorch (uv), robot needs airbot_py/revo2 (conda)
2. **Clean Separation**: Can restart policy server without affecting robot connection
3. **Same as ALOHA**: This is exactly how Physical Intelligence does it in their examples
4. **Easy Debugging**: Can test inference and robot control independently

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LOOP                                  │
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐ │
│  │ Camera  │───▶│ Client  │───▶│ Server  │───▶│ Policy  │───▶│Actions │ │
│  │ + State │    │ (obs)   │    │ (recv)  │    │ (infer) │    │(N, 12) │ │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘ │
│       ▲                                                          │       │
│       │                                                          ▼       │
│       │         ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│       └─────────│ Robot   │◀───│ Execute │◀───│ Client  │◀─────────────┘ │
│                 │ (move)  │    │ Actions │    │ (recv)  │                │
│                 └─────────┘    └─────────┘    └─────────┘                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Start Policy Server (Terminal 1)

```bash
cd ~/openpi

# Serve your fine-tuned checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/20000 \
    --default_prompt="pick up the object"
```

### Step 2: Install openpi-client in Conda (One-time)

```bash
conda activate your_robot_env

# Install the lightweight client package
pip install ~/openpi/packages/openpi-client
```

### Step 3: Run Robot Client (Terminal 2)

```bash
conda activate your_robot_env
cd ~/openpi

# Run validation
python examples/airbot/validate_airbot.py \
    --host=localhost \
    --port=8000 \
    --num_episodes=5
```

## Observation Format

The policy expects observations in this format:

```python
observation = {
    "observation/image": image,      # (H, W, 3) uint8 RGB, typically 224x224 or 480x640
    "observation/state": state,      # (12,) float32 joint positions [arm(6) + hand(6)]
    "prompt": instruction,           # str, e.g., "pick up the red block"
}
```

## Action Format

The policy returns action chunks:

```python
result = policy.infer(observation)
actions = result["actions"]  # Shape: (N, 12) where N is action horizon

# actions[:, 0:6]  = arm joint positions (radians)
# actions[:, 6:12] = hand joint positions (depends on your normalization)
```

## Do You Need Threading Like motion_retargeting?

**Short answer: NO**, for simple validation.

The complex threading in `robot_arm_exec_worker.py` and `robot_hand_exec_worker.py` is needed for:
- Real-time teleoperation with safety interlocks
- State machines (HOME, TELEOP, ERROR, MANUAL)
- Keyboard interfaces for human operators
- Synchronized data logging during data collection

For **policy validation**, you just need a simple loop:

```python
# Simple validation loop (no threads needed)
while not done:
    # 1. Get observation
    image = camera.get_frame()
    arm_joints = arm.get_joint_positions()
    hand_joints = hand.get_joint_positions()
    state = np.concatenate([arm_joints, hand_joints])
    
    # 2. Query policy
    obs = {"observation/image": image, "observation/state": state, "prompt": prompt}
    actions = policy.infer(obs)["actions"]
    
    # 3. Execute action chunk (open-loop horizon)
    for action in actions[:8]:  # Execute 8 actions before re-querying
        arm.set_joint_positions(action[:6])
        hand.set_joint_positions(action[6:12])
        time.sleep(1/30)  # 30 Hz control
```

## Files

| File | Purpose |
|------|---------|
| `validate_airbot.py` | Main validation script using openpi-client |
| `VALIDATION_ARCHITECTURE.md` | This document |
| `SINGLE_PC_SETUP.md` | Alternative in-process approach (no server) |

## Comparison: Client-Server vs In-Process

| Aspect | Client-Server (This Guide) | In-Process (SINGLE_PC_SETUP.md) |
|--------|---------------------------|--------------------------------|
| Terminals | 2 | 1 |
| Environment | Separate (uv + conda) | Single (uv only, need robot deps) |
| Latency | ~10-50ms (websocket) | <1ms (in-memory) |
| Flexibility | Can swap models easily | Need to restart script |
| Debugging | Easier (separate processes) | Single process |
| ALOHA-like? | ✅ Yes, same architecture | ❌ No |

## Troubleshooting

### "Connection refused" error
- Make sure policy server is running in Terminal 1
- Check the port matches (default: 8000)

### "Module not found: openpi_client"
- Run `pip install ~/openpi/packages/openpi-client` in your conda env

### Robot not moving
- Check robot is connected and powered
- Verify `airbot_py` and `revo2` libraries are installed in conda env
- Check joint limits and safety stops

### Slow inference
- Ensure GPU is being used for policy server
- Check `nvidia-smi` for GPU utilization
- First few inferences are slow (model warmup)
