# YAM ⟵ SO-101 Teleop + Camera Dashboard + ACT Pipeline

Bimanual teleoperation of **i2rt YAM** arms with **SO-101** leader arms, a live
**Rerun** camera dashboard, a one-click web **control panel**, and an
**episode-recording → LeRobot dataset → ACT training** pipeline.

Built for a hackathon on top of [`i2rt`](https://github.com/i2rt-robotics/i2rt)
(YAM arms, DM motors over CAN) and [LeRobot](https://github.com/huggingface/lerobot)
(ACT policy + dataset format).

## System

```
SO-101 leaders (Feetech STS3215, USB) ──► YAM followers (DM motors, CAN)
        │                                          │
        └── joint-space mapping (per-leader calibration by controller serial)
RealSense cameras (top + 2 wrists) ──► Rerun web dashboard
Control panel (:8080) ── one-click connect / teleop / record episodes
Episodes ──► LeRobotDataset ──► lerobot-train (ACT) ──► deploy
```

## Components

| File | What it does |
|------|--------------|
| `control_panel.py` | One-page web control panel (`:8080`): auto-discovers leaders/YAMs/cameras, buttons for Connect / Start Teleop / Stop, episode recording (named datasets, start/stop/save/discard), embeds the Rerun camera view. |
| `so101_teleop.py` | SO-101 leader → YAM follower teleop. Absolute range-to-range joint mapping, slow-move-to-start, velocity clamp. Loads per-leader ranges from `leader_calibration.json` by controller serial. Publishes state to `/dev/shm` for the recorder. |
| `camera_dashboard.py` | Owns the RealSense cameras, streams them to a Rerun web dashboard (scene-top / wrists-bottom layout), and hosts the episode **recorder** (control server on `:8090`). |
| `calibrate_leaders.py` | Multi-arm leader calibration manager: health-checks each connected leader, interactive full-range sweep, saves per-joint tick ranges to `leader_calibration.json` keyed by controller serial. |
| `check_leader.py` | Quick single-leader health check (USB detection, servo power/stability, motion-corruption test). |
| `check_cameras.py` | Snapshot each RealSense camera to verify it works / identify which is which. |
| `episode_writer.py` | Dependency-light episode format: one `mp4` per camera + `npz` of state/action/timestamps, under `episodes/<dataset>/episode_XXXX/`. |
| `convert_to_lerobot.py` | Convert `episodes/<dataset>/` → a `LeRobotDataset` (ACT-ready) and optionally push to the HF Hub. |
| `push_dataset.py` | Resumable HF upload of a local LeRobotDataset (uses `hf_transfer` for speed). |
| `leader_calibration.json` | Per-controller leader joint tick ranges (keyed by USB serial, stable across replug). |

## Requirements

- `i2rt` installed (YAM arm driver, CAN). See the i2rt repo.
- Python deps: `feetech-servo-sdk`, `pyrealsense2`, `opencv-python`, `rerun-sdk`, `numpy`.
- For dataset conversion / ACT training: `lerobot` (+ `accelerate`, `hf_transfer`) — typically a separate env.

## Quickstart

```bash
# 1. Calibrate the leader arms (once per arm)
python calibrate_leaders.py

# 2. Launch the control panel  →  open http://localhost:8080
python control_panel.py
#   - Connect (cameras) → Start Teleop → record episodes into named datasets

# 3. Convert recorded episodes to a LeRobot dataset (in the lerobot env)
python convert_to_lerobot.py --src episodes/<dataset> --repo-id <user>/<name> [--push]

# 4. Train ACT locally
HF_HUB_ENABLE_HF_TRANSFER=1 lerobot-train \
    --dataset.repo_id=<user>/<name> --policy.type=act --policy.device=cuda \
    --policy.push_to_hub=false --batch_size=8 --steps=50000 --output_dir=outputs/act
```

## Hardware notes

- Each SO-101 leader needs its **own** power supply (sharing one causes servo dropouts).
- RealSense cameras: on USB 2.0, use **color-only, low res** (e.g. 424×240@15); move to USB 3.0 for depth/higher res.
- YAM followers on `can0` / `can1` at 1 Mbit/s; leaders enumerate as `/dev/ttyACM*`.
