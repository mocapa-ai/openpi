# Revo2 Hand Setup for validate_airbot.py

This document describes the Revo2 hand implementation in `validate_airbot.py`.

## Overview

The `validate_airbot.py` script now includes direct integration with the Revo2 dexterous hand library, without requiring the motion-retargeting adapter layer. The implementation uses the Revo2 Modbus communication protocol directly.

## Setup Instructions

### 1. Clone the revo2_library

Clone the revo2_library directly into the openpi directory:

```bash
cd ~/openpi
git clone <revo2_library_repo_url> revo2_library
```

The expected directory structure will be:
```
openpi/
├── revo2_library/
│   └── python/
│       └── revo2/
│           ├── revo2_utils.py
│           ├── revo2_ctrl.py
│           └── ...
├── examples/
│   └── airbot/
│       └── validate_airbot.py
└── ...
```

### 2. Install Dependencies

The revo2_library requires the `bc_stark_sdk` package:

```bash
cd ~/openpi/revo2_library/python
pip install -r requirements.txt
```

### 3. Configure Hardware

Connect the Revo2 hand via USB serial. By default, the script will auto-detect the port. You can also specify a specific port:

```bash
python examples/airbot/validate_airbot.py --hand_port=/dev/ttyUSB0
```

## Implementation Details

### Revo2Hand Class

The `Revo2Hand` class provides a clean interface matching the expected format:

- **6 DOF control**: Thumb, Index, Middle, Ring, Pinky, Wrist
- **Position format**: Radians (matches training data format)
- **Internal protocol**: Modbus with normalized mode (0-1000 range)
- **Auto-conversion**: Automatically converts between radians and Revo2's normalized format

### Key Methods

```python
hand = Revo2Hand(port="/dev/ttyUSB0", side="right")

# Connect to hardware
hand.connect()

# Get current positions (returns np.ndarray of 6 floats in radians)
positions = hand.get_joint_positions()

# Set positions (accepts np.ndarray of 6 floats in radians)
hand.set_joint_positions(new_positions)

# Convenience methods
hand.set_open()
hand.set_closed()

# Disconnect
hand.disconnect()
```

### Position Mapping

The implementation handles conversion between two coordinate systems:

1. **Policy Output (Radians)**: The format expected by the trained model
   - Thumb: [0, 1.57] rad
   - Index: [0, 1.03] rad
   - Middle/Ring/Pinky/Wrist: [0, 1.41] rad

2. **Revo2 Protocol (Normalized)**: The hardware communication format
   - All fingers: [0, 1000] (0 = open, 1000 = closed)

Conversion is handled automatically by `_radians_to_revo2()` and `_revo2_to_radians()` methods.

### Asyncio Integration

The Revo2 library uses asyncio for communication. Each method in `Revo2Hand` wraps async operations in `asyncio.run()` to provide a synchronous interface compatible with the validation script's control flow.

## Usage

Once setup is complete, run validation normally:

```bash
# Terminal 1: Start policy server
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=airbot_pi05 \
    --policy.dir=checkpoints/airbot_pi05/20000

# Terminal 2: Run validation with Revo2 hand
conda activate your_robot_env
python examples/airbot/validate_airbot.py \
    --host=localhost \
    --port=8000 \
    --hand_port=/dev/ttyUSB0  # or omit for auto-detect
```

## Troubleshooting

### "No module named 'revo2_library'"

Make sure you've cloned the revo2_library into `~/openpi/revo2_library/` and the path structure is correct.

### "Failed to connect to Revo2 hand"

1. Check USB connection: `ls /dev/ttyUSB*`
2. Check permissions: `sudo chmod 666 /dev/ttyUSB0`
3. Try auto-detect: omit `--hand_port` argument
4. Check if device is already in use by another process

### "bc_stark_sdk not found"

Install the SDK from the revo2_library requirements:
```bash
cd ~/openpi/revo2_library/python
pip install -r requirements.txt
```

## Testing Without Hardware

The implementation includes a dummy mode that activates when:
- The revo2_library is not installed, OR
- Hardware is not connected

In dummy mode, all hand operations return zero positions and accept commands silently.

## Differences from Adapter Approach

The original motion-retargeting repository includes a `Revo2HandAdapter` class. This implementation:

1. **Direct integration**: Uses revo2_library directly, no adapter needed
2. **Simplified**: Only includes methods needed for policy validation
3. **Self-contained**: All mapping logic is in validate_airbot.py
4. **No external config**: Doesn't depend on config files from motion-retargeting

This makes the validation script more portable and easier to understand.
