#!/usr/bin/env python3
"""Convert our mp4+npz episodes into a LeRobotDataset (and optionally push to HF).

Run with the conda lerobot python:
    /home/shiv/miniforge3/envs/lerobot/bin/python convert_to_lerobot.py \
        --src episodes/default --repo-id Shivakumr/yam_default [--limit 2] [--push] [--private]
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMS = ["top", "wrist_1", "wrist_2"]
# state/action layout: recorder concatenates can0 then can1, each 6 joints + gripper
STATE_NAMES = [f"{ch}_{j}" for ch in ("can0", "can1")
               for j in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]


def build_features(h, w):
    feats = {}
    for c in CAMS:
        feats[f"observation.images.{c}"] = {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    feats["observation.state"] = {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES}
    feats["action"] = {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES}
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="episodes/default")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--task", default="bimanual teleoperation")
    ap.add_argument("--robot-type", default="yam_bimanual_so101_leader")
    ap.add_argument("--limit", type=int, default=0, help="only convert first N episodes (0 = all)")
    ap.add_argument("--episodes", default="", help="only these episode indices, e.g. 7,8,12-19")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(args.src, "episode_*")))
    if args.episodes:
        keep = set()
        for part in args.episodes.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                keep.update(range(int(a), int(b) + 1))
            else:
                keep.add(int(part))
        eps = [e for e in eps if int(os.path.basename(e).split("_")[1]) in keep]
        print(f"filtering to {len(eps)} episodes: {sorted(keep)}")
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        print(f"no episodes in {args.src}")
        return
    # resolution from the first video
    cap = cv2.VideoCapture(os.path.join(eps[0], f"{CAMS[0]}.mp4"))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    print(f"converting {len(eps)} episodes @ {w}x{h}, fps={args.fps} -> {args.repo_id}")

    # fresh local dataset dir
    root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    if os.path.exists(root):
        print(f"removing existing local dataset at {root}")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=build_features(h, w),
        robot_type=args.robot_type, use_videos=True,
    )

    for ei, ep in enumerate(eps):
        z = np.load(os.path.join(ep, "data.npz"))
        state, action = z["state"], z["action"]
        caps = {c: cv2.VideoCapture(os.path.join(ep, f"{c}.mp4")) for c in CAMS}
        n = len(state)
        written = 0
        for i in range(n):
            imgs = {}
            ok = True
            for c in CAMS:
                r, bgr = caps[c].read()
                if not r:
                    ok = False
                    break
                imgs[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if not ok:
                break
            frame = {"task": args.task,
                     "observation.state": state[i].astype(np.float32),
                     "action": action[i].astype(np.float32)}
            for c in CAMS:
                frame[f"observation.images.{c}"] = imgs[c]
            ds.add_frame(frame)
            written += 1
        for cap in caps.values():
            cap.release()
        ds.save_episode()
        print(f"  [{ei+1}/{len(eps)}] {os.path.basename(ep)}: {written} frames")

    ds.finalize()
    print(f"done. local dataset: {root}")

    if args.push:
        print(f"pushing to HF hub as {args.repo_id} (private={args.private})...")
        ds.push_to_hub(private=args.private, tags=["yam", "so101", "act", "bimanual"])
        print(f"pushed: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
