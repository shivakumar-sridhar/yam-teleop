#!/usr/bin/env python3
"""Live one-line progress bar for a running lerobot-train job. Ctrl-C to exit.

    python watch_training.py                 # watches the current ACT run
    python watch_training.py --log <path>    # any other train log
"""
import argparse
import os
import re
import shutil
import subprocess
import time

DEFAULT_LOG = ("/tmp/claude-1000/-home-shiv-Mission-i2rt/"
               "f2b788d1-dcb7-4117-a4c2-7602b91d519b/scratchpad/train_expert.log")


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()[0]
        u, mu, t = [x.strip() for x in out.split(",")]
        return f"GPU {u}% {mu}MB {t}C"
    except Exception:
        return "GPU --"


def parse(path):
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, sz - 80000))
            data = f.read().decode(errors="ignore").replace("\r", "\n")
    except Exception:
        return None
    step = total = rate = eta = loss = None
    for m in re.finditer(r"(\d+)/(\d+) \[[\d:]+<([\d:?]+),\s*([\d.]+)step/s\]", data):
        step, total, eta, rate = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))
    for lm in re.finditer(r"loss:([\d.]+)", data):
        loss = float(lm.group(1))
    return step, total, rate, eta, loss


def bar(frac, width=28):
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()
    print("watching training... (Ctrl-C to exit)")
    try:
        while True:
            p = parse(args.log)
            if not p or p[0] is None:
                line = "  waiting for training output..."
            else:
                step, total, rate, eta, loss = p
                frac = step / total if total else 0.0
                lo = f"{loss:.3f}" if loss is not None else "?"
                line = (f"  ACT [{bar(frac)}] {frac*100:4.1f}%  "
                        f"{step}/{total}  loss {lo}  {rate or 0:.1f} it/s  ETA {eta}  {gpu()}")
            cols = shutil.get_terminal_size((140, 20)).columns
            print("\r" + line[: cols - 1].ljust(cols - 1), end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
