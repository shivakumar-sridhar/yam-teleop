#!/usr/bin/env python3
"""Stream RealSense cameras to a Rerun web dashboard (top + wrist views).

Logs each camera's color (and depth) to Rerun and serves a browser dashboard.

Run:
    cd ~/Mission/i2rt
    .venv/bin/python camera_dashboard.py                 # web dashboard
    .venv/bin/python camera_dashboard.py --spawn         # local desktop viewer
    .venv/bin/python camera_dashboard.py --no-depth      # color only (lighter)
Then open the printed http URL in a browser.
"""
import argparse
import glob
import json
import os
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr
import rerun.blueprint as rrb

from episode_writer import EpisodeWriter, next_episode_index

RECORD_PORT = 8090
STATE_CHANNELS = ["can0", "can1"]   # /dev/shm/teleop_<ch>.json -> observation.state / action
STATE_DIM = 7                        # per arm (6 joints + gripper)


class EpisodeRecorder:
    """Samples the latest camera frames + teleop states into episodes on command."""

    def __init__(self, record_root, fps):
        self.record_root = record_root
        self.fps = fps
        # sweep orphaned temp dirs left by any process killed mid-record
        for d in glob.glob(os.path.join(record_root, "**", "_rec_*"), recursive=True):
            shutil.rmtree(d, ignore_errors=True)
        self.dataset = "default"
        self.writer = EpisodeWriter(self._dataset_dir(), fps)
        self.recording = False
        self.latest = {}            # role -> last rgb frame (updated by main loop)
        self.last_sample = 0.0
        self.last_saved = None      # (name, counts)

    def _dataset_dir(self):
        return os.path.join(self.record_root, self.dataset)

    def set_dataset(self, name):
        if self.recording:
            return "stop/save the current episode before switching dataset"
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "").strip()) or "default"
        self.dataset = name
        os.makedirs(self._dataset_dir(), exist_ok=True)
        self.writer = EpisodeWriter(self._dataset_dir(), self.fps)
        return f"dataset '{name}' ({next_episode_index(self._dataset_dir())} episodes)"

    def list_datasets(self):
        out = []
        if os.path.isdir(self.record_root):
            for d in sorted(os.listdir(self.record_root)):
                p = os.path.join(self.record_root, d)
                if os.path.isdir(p) and not d.startswith("_rec_"):
                    out.append({"name": d, "episodes": next_episode_index(p)})
        return out

    def update_frame(self, role, img):
        self.latest[role] = img

    def _read_states(self):
        follower, action = [], []
        now = time.time()
        for ch in STATE_CHANNELS:
            try:
                d = json.load(open(f"/dev/shm/teleop_{ch}.json"))
                fresh = (now - d.get("t", 0)) < 0.5
                follower += d["follower"] if fresh else [0.0] * STATE_DIM
                action += d["action"] if fresh else [0.0] * STATE_DIM
            except Exception:
                follower += [0.0] * STATE_DIM
                action += [0.0] * STATE_DIM
        return follower, action

    def maybe_sample(self):
        if not self.recording:
            return
        now = time.time()
        if now - self.last_sample < 1.0 / self.fps:
            return
        self.last_sample = now
        frames = dict(self.latest)          # snapshot of most-recent frame per camera
        follower, action = self._read_states()
        self.writer.add(frames, follower, action, now)

    # --- control actions (return a status string) ---
    def start(self):
        if self.recording:
            return "already recording"
        self.writer.start()
        self.last_sample = 0.0
        self.recording = True
        return "recording started"

    def stop(self):
        self.recording = False
        return f"stopped at {self.writer.n_frames} frames"

    def save(self):
        self.recording = False
        path, counts = self.writer.save()
        name = os.path.basename(path) if path else "-"
        self.last_saved = (name, counts)
        return f"saved {name}: {counts}"

    def discard(self):
        self.recording = False
        self.writer.discard()
        return "discarded (nothing saved)"

    def status(self):
        return {
            "recording": self.recording,
            "n_frames": self.writer.n_frames,
            "counts": self.writer.counts,
            "dataset": self.dataset,
            "episodes_on_disk": next_episode_index(self._dataset_dir()),
            "datasets": self.list_datasets(),
            "last_saved": self.last_saved,
        }


def start_record_server(rec: EpisodeRecorder):
    actions = {
        "/record/start": rec.start,
        "/record/stop": rec.stop,
        "/record/save": rec.save,
        "/record/discard": rec.discard,
    }

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/record/status":
                self._send(rec.status())
            else:
                self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/record/dataset":
                name = parse_qs(parsed.query).get("name", [""])[0]
                self._send({"msg": rec.set_dataset(name), **rec.status()})
                return
            fn = actions.get(parsed.path)
            if not fn:
                self.send_error(404)
                return
            self._send({"msg": fn(), **rec.status()})

    srv = ThreadingHTTPServer(("0.0.0.0", RECORD_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"  record control: http://localhost:{RECORD_PORT}/record/{{start,stop,save,discard,status}}")


# Friendly role names by serial. Unknown cameras get "cam_<serial>".
ROLE_MAP = {
    "420222071940": "top",       # D435
    "353322271521": "wrist_1",   # D405
    "323622270916": "wrist_2",   # D405
}


def discover_cameras():
    """Return [(role, serial, model)] for every connected RealSense camera."""
    out = []
    for d in rs.context().query_devices():
        serial = d.get_info(rs.camera_info.serial_number)
        model = d.get_info(rs.camera_info.name)
        out.append((ROLE_MAP.get(serial, f"cam_{serial}"), serial, model))
    return out


def start_camera(serial, width, height, fps, want_depth):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    if want_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipe.start(cfg)
    depth_scale = None
    if want_depth:
        try:
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        except Exception:
            depth_scale = 0.001
    return pipe, depth_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--grpc-port", type=int, default=9876)
    # Defaults tuned for USB 2.0 (both cams on a 480M bus): color-only, low res.
    # For depth / higher res, move the cameras to USB 3.0 ports and add --depth.
    ap.add_argument("--width", type=int, default=424)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--depth", action="store_true", help="also stream depth (needs USB 3.0 headroom)")
    ap.add_argument("--spawn", action="store_true", help="desktop viewer instead of web")
    ap.add_argument("--record-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes"))
    ap.add_argument("--record-fps", type=int, default=0, help="episode fps (0 = same as --fps)")
    args = ap.parse_args()
    want_depth = args.depth

    rr.init("robot_cameras")

    # Custom dashboard layout (Blueprint): big scene view on top, wrist views
    # side-by-side on the bottom. Falls back to a grid for other camera sets.
    discovered = discover_cameras()
    roles = [r for r, _, _ in discovered]

    def view(r):
        return rrb.Spatial2DView(origin=f"{r}/color", name=r)

    top = [r for r in roles if r == "top" or r.startswith("top")]
    wrists = [r for r in roles if "wrist" in r]
    if top and wrists:
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                view(top[0]),                                  # scene, full width
                rrb.Horizontal(*[view(r) for r in wrists]),    # wrists side-by-side
                rrb.TimeSeriesView(origin="policy", name="ACT policy — predicted action & state"),
                row_shares=[2, 2, 1.6],
            ),
            collapse_panels=True,
        )
    elif roles:
        blueprint = rrb.Blueprint(rrb.Grid(*[view(r) for r in roles]), collapse_panels=True)
    else:
        blueprint = None

    if args.spawn:
        rr.spawn(default_blueprint=blueprint)
    else:
        uri = rr.serve_grpc(grpc_port=args.grpc_port, default_blueprint=blueprint)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=uri)
        print("\n" + "=" * 60)
        print("  Open the dashboard (auto-connects to the live stream):")
        print(f"    http://localhost:{args.web_port}/?url={uri}")
        print(f"  (remote host? swap localhost for the host IP; keep the ?url= part)")
        print("=" * 60 + "\n")

    # start every connected camera (skip any that fail so one bad cam doesn't stop the rest)
    cams = {}
    for role, serial, model in discovered:
        try:
            pipe, scale = start_camera(serial, args.width, args.height, args.fps, want_depth)
            cams[role] = (pipe, scale)
            print(f"  started {role}: {model} {serial}  ({args.width}x{args.height}@{args.fps})")
        except Exception as e:
            print(f"  WARN could not start {role} ({model} {serial}): {e}")
    if not cams:
        print("  no cameras started — check USB / serials.")
        return

    rec = EpisodeRecorder(args.record_dir, args.record_fps or args.fps)
    start_record_server(rec)

    print("  streaming... Ctrl-C to stop.")
    frame = 0
    fcount = {name: 0 for name in cams}
    last_report = time.time()
    try:
        while True:
            rr.set_time("frame", sequence=frame)
            for name, (pipe, scale) in cams.items():
                try:
                    fs = pipe.wait_for_frames(timeout_ms=500)
                except RuntimeError:
                    continue  # no frame within timeout
                if not fs:
                    continue
                fcount[name] += 1
                color = fs.get_color_frame()
                if color:
                    img = np.asanyarray(color.get_data())  # HxWx3 rgb8
                    rr.log(f"{name}/color", rr.Image(img))
                    rec.update_frame(name, img)
                    # publish latest frame to /dev/shm for the inference process
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if ok:
                        tmp = f"/dev/shm/cam_{name}.jpg.tmp"
                        with open(tmp, "wb") as fh:
                            fh.write(buf.tobytes())
                        os.replace(tmp, f"/dev/shm/cam_{name}.jpg")
                depth = fs.get_depth_frame()
                if depth:
                    d = np.asanyarray(depth.get_data())  # z16
                    meter = (1.0 / scale) if scale else 1000.0
                    rr.log(f"{name}/depth", rr.DepthImage(d, meter=meter))
            # overlay live ACT policy predictions (published by infer_act.py) as time-series
            try:
                d = json.load(open("/dev/shm/infer_action.json"))
                if time.time() - d.get("t", 0) < 2.0:
                    jn = [f"{ch}_{j}" for ch in ("can0", "can1")
                          for j in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")]
                    for nm, av in zip(jn, d.get("action", [])):
                        rr.log(f"policy/action/{nm}", rr.Scalars(float(av)))
                    for nm, sv in zip(jn, d.get("state", [])):
                        rr.log(f"policy/state/{nm}", rr.Scalars(float(sv)))
            except Exception:
                pass
            rec.maybe_sample()
            frame += 1
            if time.time() - last_report > 2.0:
                print("  frames logged: " + ", ".join(f"{n}={c}" for n, c in fcount.items()), flush=True)
                last_report = time.time()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n  stopping.")
    finally:
        for pipe, _ in cams.values():
            try:
                pipe.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
