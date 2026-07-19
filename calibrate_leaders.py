#!/usr/bin/env python3
"""Multi-arm SO-101 leader calibration manager.

Polls every connected leader controller (/dev/ttyACM*) and, for each one:
  1. CHECK   - power + servo stability (won't calibrate a flaky/unpowered arm)
  2. CALIBRATE - interactive full-range sweep with a live table
  3. SAVE    - writes per-joint tick ranges into leader_calibration.json,
               keyed by the controller's STABLE serial id (survives replug /
               port renumbering), for the main teleop program to load.

Run:
    cd ~/Mission/i2rt
    .venv/bin/python calibrate_leaders.py                # check + calibrate all
    .venv/bin/python calibrate_leaders.py --check-only   # just health check
    .venv/bin/python calibrate_leaders.py --show         # print saved calibration
"""
import argparse
import glob
import json
import os
import re
import time
from datetime import datetime

from scservo_sdk import PacketHandler, PortHandler

ADDR_POS = 56
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leader_calibration.json")


# --------------------------------------------------------------------------- #
# discovery / identity
# --------------------------------------------------------------------------- #
def list_ports():
    """Return [(port, full_serial_or_None)] for every /dev/ttyACM*."""
    byid = "/dev/serial/by-id"
    links = {}
    if os.path.isdir(byid):
        for name in os.listdir(byid):
            links[os.path.realpath(os.path.join(byid, name))] = name
    out = []
    for p in sorted(glob.glob("/dev/ttyACM*")):
        out.append((p, links.get(os.path.realpath(p))))
    return out


def controller_id(full_serial, port):
    """Stable id for a controller board, e.g. '5B14115162'. Falls back to port."""
    if full_serial:
        m = re.search(r"Serial_([A-Za-z0-9]+)", full_serial)
        if m:
            return m.group(1)
        return full_serial
    return os.path.basename(port)


# --------------------------------------------------------------------------- #
# json store
# --------------------------------------------------------------------------- #
def load_calib():
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_calib(calib):
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def robust_read(pk, ph, i):
    """Median of 5 with >=3 consensus; rejects transient glitches. None if unstable."""
    vals = []
    for _ in range(5):
        raw, comm, _ = pk.read2ByteTxRx(ph, i, ADDR_POS)
        if comm == 0 and 0 <= raw <= 4095:
            vals.append(raw)
    if len(vals) < 3:
        return None
    vals.sort()
    med = vals[len(vals) // 2]
    close = [v for v in vals if abs(v - med) <= 40]
    return sum(close) // len(close) if len(close) >= 3 else None


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def health_check(pk, ph, pings):
    counts = {i: 0 for i in range(1, 7)}
    full = 0
    for _ in range(pings):
        found = [i for i in range(1, 7) if pk.ping(ph, i)[1] == 0]
        for i in found:
            counts[i] += 1
        if len(found) == 6:
            full += 1
        time.sleep(0.2)
    total = sum(counts.values())
    if total == 0:
        status = "NO_POWER"
    elif full < pings:
        status = "FLAKY"
    else:
        status = "OK"
    return status, counts, full


def calibrate_sweep(pk, ph):
    """Interactive full-range sweep. Returns (ranges dict, corrupt_total)."""
    cur = {i: None for i in range(1, 7)}
    mn = {i: 10**9 for i in range(1, 7)}
    mx = {i: -10**9 for i in range(1, 7)}
    corrupt = {i: 0 for i in range(1, 7)}
    for i in range(1, 7):
        v = robust_read(pk, ph, i)
        while v is None:
            v = robust_read(pk, ph, i)
        mn[i] = mx[i] = cur[i] = v

    input("    >> Press ENTER, then sweep EVERY joint to both extremes + gripper. Ctrl-C when done...")
    t0 = time.time()
    last = 0.0
    drawn = [False]

    def render():
        lines = [f"    {'JOINT':<11} {'POS':>6} {'MIN':>5} {'MAX':>5} {'SPAN':>5}   STATUS"]
        for j in range(1, 7):
            v = cur[j]
            poss = f"{v:6d}" if isinstance(v, int) else "   ---"
            lo, hi = mn[j], mx[j]
            span = hi - lo
            if corrupt[j] > 0:
                st = f"CORRUPT x{corrupt[j]}"
            elif span < 50:
                st = "move it"
            else:
                st = "ok"
            lines.append(f"    {NAMES[j-1]:<11} {poss} {lo:5d} {hi:5d} {span:5d}   {st}")
        lines.append(f"    elapsed {time.time()-t0:5.1f}s   corrupt {sum(corrupt.values())}   (Ctrl-C when done)")
        block = "".join("\033[2K" + ln + "\n" for ln in lines)
        if drawn[0]:
            block = f"\033[{len(lines)}A" + block
        drawn[0] = True
        print(block, end="", flush=True)

    try:
        while True:
            for i in range(1, 7):
                raw, comm, _ = pk.read2ByteTxRx(ph, i, ADDR_POS)
                if comm != 0:
                    continue
                if raw < 0 or raw > 4095:
                    corrupt[i] += 1
                    continue
                mn[i] = min(mn[i], raw)
                mx[i] = max(mx[i], raw)
                cur[i] = raw
            if time.time() - last > 0.1:
                render()
                last = time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    render()
    print()
    ranges = {NAMES[i - 1]: [mn[i], mx[i]] for i in range(1, 7)}
    return ranges, sum(corrupt.values())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pings", type=int, default=15)
    ap.add_argument("--check-only", action="store_true", help="health check, don't calibrate")
    ap.add_argument("--show", action="store_true", help="print saved calibration and exit")
    args = ap.parse_args()

    calib = load_calib()

    if args.show:
        print(f"calibration file: {CALIB_PATH}\n")
        if not calib:
            print("  (empty)")
        for cid, e in calib.items():
            print(f"  [{cid}] last_port={e.get('last_port')} calibrated_at={e.get('calibrated_at')}")
            for jn, (lo, hi) in e.get("ranges", {}).items():
                print(f"      {jn:11s} {lo:5d}..{hi:5d}")
        return

    ports = list_ports()
    print("=" * 64)
    print("  SO-101 LEADER CALIBRATION MANAGER")
    print("=" * 64)
    if not ports:
        print("No /dev/ttyACM* controllers found. Check USB / power, then re-run.")
        return
    print(f"Found {len(ports)} controller(s):")
    for p, s in ports:
        print(f"  {p}  ->  id {controller_id(s, p)}")

    for idx, (port, full_serial) in enumerate(ports, 1):
        cid = controller_id(full_serial, port)
        print("\n" + "-" * 64)
        print(f"CONTROLLER {idx}/{len(ports)}   id={cid}   port={port}")
        print("-" * 64)
        try:
            ph = PortHandler(port)
            if not ph.openPort():
                raise RuntimeError("openPort failed")
            ph.setBaudRate(1000000)
        except Exception as e:
            print(f"  ✗ cannot open {port}: {e}  — skipping")
            continue
        pk = PacketHandler(0)

        print(f"  [check] pinging servos x{args.pings} ...")
        status, counts, full = health_check(pk, ph, args.pings)
        for i in range(1, 7):
            mark = "ok" if counts[i] == args.pings else "!!"
            print(f"     servo {i} {NAMES[i-1]:11s} {counts[i]:2d}/{args.pings} {mark}")
        print(f"     full 6/6: {full}/{args.pings}  ->  {status}")

        if status == "NO_POWER":
            print("     ✗ servos unpowered — connect the power brick. Skipping this controller.")
            ph.closePort()
            continue
        if status == "FLAKY":
            print("     ✗ flaky (under-powered or loose cable). Fix before calibrating. Skipping.")
            ph.closePort()
            continue
        print("     ✓ connection good.")

        if args.check_only:
            ph.closePort()
            continue

        ans = input(f"  Calibrate controller {cid}? [Y/skip]: ").strip().lower()
        if ans in ("s", "skip", "n"):
            print("     skipped.")
            ph.closePort()
            continue

        ranges, corrupt_total = calibrate_sweep(pk, ph)
        ph.closePort()

        # report
        print("    captured ranges:")
        for jn, (lo, hi) in ranges.items():
            print(f"      {jn:11s} {lo:5d}..{hi:5d}  span {hi-lo}")
        if corrupt_total > 0:
            print(f"    ⚠ {corrupt_total} corrupt reads during sweep (possible bad cable).")
            keep = input("    Save anyway? [y/N]: ").strip().lower()
            if keep not in ("y", "yes"):
                print("    NOT saved. Fix the cable and re-run.")
                continue

        calib[cid] = {
            "serial_id": full_serial,
            "last_port": port,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "ranges": ranges,
        }
        save_calib(calib)
        print(f"    ✓ saved calibration for [{cid}] -> {os.path.basename(CALIB_PATH)}")

    print("\n" + "=" * 64)
    print(f"Done. Calibrations on file: {list(calib.keys())}")
    print(f"File: {CALIB_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
