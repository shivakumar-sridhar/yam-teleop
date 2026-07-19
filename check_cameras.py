#!/usr/bin/env python3
"""Snapshot each RealSense camera, one at a time, to verify it works and see
which physical camera it is (top vs wrist).

Opens cameras ONE AT A TIME (no USB-bandwidth conflict) at 640x480 color, grabs
a frame, and saves a labeled JPG you can open. Also reports frame count so you
know the camera is actually streaming.

NOTE: nothing else may hold the cameras (stop camera_dashboard.py first).

Run:
    cd ~/Mission/i2rt
    .venv/bin/python check_cameras.py
    .venv/bin/python check_cameras.py --live 353322271521   # live view one cam in Rerun
Then open the JPGs in ./cam_snapshots/
"""
import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_snapshots")


def list_cams():
    return [
        (d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
        for d in rs.context().query_devices()
    ]


def snapshot(serial, width=640, height=480, fps=15, warmup=20):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipe.start(cfg)
    frame, got = None, 0
    try:
        for _ in range(warmup):
            try:
                fs = pipe.wait_for_frames(2000)
                c = fs.get_color_frame()
                if c:
                    frame = np.asanyarray(c.get_data())
                    got += 1
            except Exception:
                pass
    finally:
        pipe.stop()
    return frame, got


def live_view(serial):
    """Stream one camera into a native Rerun window (needs a display)."""
    import time
    import rerun as rr

    rr.init(f"camera_{serial}", spawn=True)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    pipe.start(cfg)
    print(f"live view of {serial} — Ctrl-C to stop")
    try:
        while True:
            fs = pipe.wait_for_frames(2000)
            c = fs.get_color_frame()
            if c:
                rr.log("color", rr.Image(np.asanyarray(c.get_data())))
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", metavar="SERIAL", default=None, help="live-view one camera by serial")
    args = ap.parse_args()

    cams = list_cams()
    if not cams:
        print("No RealSense cameras found. Check USB / power.")
        return

    if args.live:
        live_view(args.live)
        return

    os.makedirs(OUT, exist_ok=True)
    print(f"Found {len(cams)} RealSense camera(s). Saving snapshots to:\n  {OUT}\n")
    for i, (name, serial) in enumerate(cams):
        print(f"[{i}] {name}  serial={serial} ... ", end="", flush=True)
        try:
            frame, got = snapshot(serial)
            if frame is None:
                print("NO FRAME — camera not delivering (USB bandwidth/power?)")
                continue
            path = os.path.join(OUT, f"cam{i}_{name.split()[-1]}_{serial}.jpg")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"OK ({got} frames) -> {os.path.basename(path)}")
        except Exception as e:
            print(f"ERROR: {e}")
    print(f"\nOpen the JPGs in {OUT} to verify each camera and see which is top vs wrist.")


if __name__ == "__main__":
    main()
