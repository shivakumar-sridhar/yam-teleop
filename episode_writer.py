#!/usr/bin/env python3
"""Dependency-light episode writer for teleop demonstration recording.

Writes one mp4 per camera (via OpenCV) + a single npz of state/action/timestamps
per episode. No LeRobot/torch dependency — convert to LeRobotDataset at train time.

Layout:
    <root>/episode_XXXX/
        top.mp4  wrist_1.mp4  wrist_2.mp4     # one per camera that had frames
        data.npz                              # state, action, t, frame counts, fps
        meta.json                             # human-readable summary

Usage:
    w = EpisodeWriter(root, fps=15)
    w.start()
    w.add({"top": rgb, "wrist_1": rgb, ...}, state=vec, action=vec, t=time.time())
    path, counts = w.save()      # or w.discard()
"""
import json
import os
import shutil
import tempfile

import cv2
import numpy as np


def next_episode_index(root):
    os.makedirs(root, exist_ok=True)
    n = -1
    for name in os.listdir(root):
        if name.startswith("episode_"):
            try:
                n = max(n, int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return n + 1


class EpisodeWriter:
    def __init__(self, root, fps=15):
        self.root = root
        self.fps = fps
        self._tmp = None
        self._writers = {}     # cam -> cv2.VideoWriter
        self._counts = {}      # cam -> frames written
        self._states = []
        self._actions = []
        self._ts = []

    @property
    def active(self):
        return self._tmp is not None

    @property
    def counts(self):
        return dict(self._counts)

    @property
    def n_frames(self):
        return len(self._ts)

    def start(self):
        os.makedirs(self.root, exist_ok=True)
        self._tmp = tempfile.mkdtemp(prefix="_rec_", dir=self.root)
        self._writers = {}
        self._counts = {}
        self._states, self._actions, self._ts = [], [], []

    def add(self, frames: dict, state, action, t):
        """frames: {cam_name: HxWx3 uint8 RGB or None}. state/action: 1D arrays."""
        if self._tmp is None:
            return
        for cam, img in frames.items():
            if img is None:
                continue
            w = self._writers.get(cam)
            if w is None:
                h, wd = img.shape[:2]
                path = os.path.join(self._tmp, f"{cam}.mp4")
                w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                    float(self.fps), (wd, h))
                self._writers[cam] = w
                self._counts[cam] = 0
            w.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self._counts[cam] += 1
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._ts.append(float(t))

    def save(self):
        """Finalize and move into episode_XXXX. Returns (path, counts)."""
        if self._tmp is None:
            return None, {}
        for w in self._writers.values():
            w.release()
        np.savez(
            os.path.join(self._tmp, "data.npz"),
            state=np.array(self._states, dtype=np.float32) if self._states else np.zeros((0,)),
            action=np.array(self._actions, dtype=np.float32) if self._actions else np.zeros((0,)),
            t=np.array(self._ts, dtype=np.float64),
        )
        with open(os.path.join(self._tmp, "meta.json"), "w") as f:
            json.dump({"fps": self.fps, "n_frames": self.n_frames,
                       "camera_frame_counts": self._counts,
                       "cameras": list(self._writers.keys())}, f, indent=2)
        ep = next_episode_index(self.root)
        final = os.path.join(self.root, f"episode_{ep:04d}")
        os.rename(self._tmp, final)
        counts = dict(self._counts)
        self._tmp = None
        return final, counts

    def discard(self):
        if self._tmp is None:
            return
        for w in self._writers.values():
            try:
                w.release()
            except Exception:
                pass
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None


if __name__ == "__main__":  # smoke test: write a 30-frame dummy episode
    import time
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes")
    w = EpisodeWriter(root, fps=15)
    w.start()
    for i in range(30):
        img = (np.random.rand(240, 424, 3) * 255).astype(np.uint8)
        w.add({"top": img, "wrist_1": img, "wrist_2": img},
              state=np.zeros(14), action=np.zeros(14), t=time.time())
    path, counts = w.save()
    print("wrote", path, counts)
    print("files:", sorted(os.listdir(path)))
