#!/usr/bin/env python3
"""ACT policy inference / deploy for the bimanual YAM rig.

Run with the conda lerobot python (has lerobot + i2rt + torch/cuda):
    /home/shiv/miniforge3/envs/lerobot/bin/python infer_act.py \
        --checkpoint outputs/act_yams_expert/checkpoints/050000 [--dry-run] [--channels can0,can1]

Observation = 3 cameras (read from /dev/shm/cam_*.jpg, published by camera_dashboard)
            + robot state (14 = [can0(7), can1(7)]).
Policy outputs a 14-dim action (target joint pos for both arms).

  --dry-run : read cams + state from /dev/shm (state from /dev/shm/teleop_<ch>.json),
              predict, and PRINT the action. No YAM opened, no motion. Teleop may stay on.
  live      : opens the YAMs on --channels, reads their state, commands both arms with a
              slow-move-to-first-action and a per-step velocity clamp. Teleop MUST be off.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/shiv/Mission/i2rt")

from lerobot.common.control_utils import predict_action
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CAMS = ["top", "wrist_1", "wrist_2"]
CHANNELS = ["can0", "can1"]      # order MUST match training state/action layout
NJ = 7                           # per arm (6 joints + gripper)


def read_cam(role):
    with open(f"/dev/shm/cam_{role}.jpg", "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # HWC uint8


def state_from_shm(channels):
    s = []
    for ch in channels:
        d = json.load(open(f"/dev/shm/teleop_{ch}.json"))
        s += d["follower"]
    return np.asarray(s, dtype=np.float32)


def build_obs(state14, channels):
    obs = {f"observation.images.{r}": read_cam(r) for r in CAMS}
    obs["observation.state"] = state14
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a checkpoint dir (contains pretrained_model/)")
    ap.add_argument("--channels", default="can0,can1")
    ap.add_argument("--task", default="bimanual teleoperation")
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until stopped")
    ap.add_argument("--engage-vel", type=float, default=0.4, help="rad/s during slow-move-to-first-action")
    ap.add_argument("--stream-vel", type=float, default=1.5, help="max rad/s per joint while running")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    channels = args.channels.split(",")

    pm = os.path.join(args.checkpoint, "pretrained_model")
    if not os.path.isdir(pm):
        pm = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ACT policy from {pm} on {device}...", flush=True)
    policy = ACTPolicy.from_pretrained(pm)
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=pm)
    print("policy loaded. select_action ready.", flush=True)

    robots = None
    if not args.dry_run:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
        from i2rt.utils.utils import override_log_level
        override_log_level()
        robots = {}
        for ch in channels:
            robots[ch] = get_yam_robot(channel=ch, arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310)
        print("YAMs opened:", list(robots.keys()), flush=True)

    def read_state():
        if robots is None:
            return state_from_shm(channels)
        return np.concatenate([np.asarray(robots[ch].get_joint_pos(), dtype=np.float32) for ch in channels])

    def infer(state):
        obs = build_obs(state, channels)
        action = predict_action(obs, policy, device, preprocessor, postprocessor,
                                use_amp=False, task=args.task)
        return np.asarray(action.squeeze(0).cpu(), dtype=np.float32) if action.ndim > 1 \
            else np.asarray(action.cpu(), dtype=np.float32)

    dt = 1.0 / args.hz
    print(f"[{'DRY-RUN' if args.dry_run else 'LIVE'}] inference @ {args.hz}Hz. "
          + ("printing actions, no motion." if args.dry_run else "commanding YAMs."), flush=True)

    # LIVE: slow-move each YAM from current pose to the first predicted action
    prev = read_state()
    if not args.dry_run:
        first = infer(prev)
        print(f"first action: {np.round(first,2)}  (slow-moving there)", flush=True)
        steps = max(1, int(3.0 / dt))
        for i in range(1, steps + 1):
            a = i / steps
            cmd = prev * (1 - a) + first * a
            for k, ch in enumerate(channels):
                robots[ch].command_joint_pos(cmd[k*NJ:(k+1)*NJ])
            time.sleep(dt)
        prev = first

    t0 = time.time()
    frame = 0
    max_step = args.stream_vel * dt
    try:
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            state = read_state()
            action = infer(state)
            if args.dry_run:
                if frame % max(1, int(args.hz)) == 0:  # ~1 Hz print
                    print("action=[" + " ".join(f"{v:+.2f}" for v in action) + "]", flush=True)
            else:
                q = prev + np.clip(action - prev, -max_step, max_step)  # velocity clamp
                for k, ch in enumerate(channels):
                    robots[ch].command_joint_pos(q[k*NJ:(k+1)*NJ])
                prev = q
            frame += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        if robots is not None:
            for ch, r in robots.items():
                try:
                    r.enter_gravity_comp_idle()
                except Exception:
                    pass
            print("YAMs returned to gravity-comp idle.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)


