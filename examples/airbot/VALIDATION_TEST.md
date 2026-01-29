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
python examples/airbot/validation_gripper.py \
```
