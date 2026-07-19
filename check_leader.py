#!/usr/bin/env python3
"""SO-101 leader-arm health check.

Verifies, in order:
  [1] USB DETECTION  - the controller enumerates as /dev/ttyACM*
  [2] SERVOS + POWER - all 6 servos respond, stably (flaky = power/cable)
  [3] MOTION         - no data corruption while the arm is moved (bad cable)

Run it manually:
    cd ~/Mission/i2rt
    .venv/bin/python check_leader.py                 # auto-detect port
    .venv/bin/python check_leader.py --port /dev/ttyACM0
    .venv/bin/python check_leader.py --skip-motion    # just checks [1] and [2]
"""
import argparse
import glob
import os
import time

from scservo_sdk import PacketHandler, PortHandler

ADDR_POS = 56
NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]


def list_ports():
    byid = "/dev/serial/by-id"
    links = {}
    if os.path.isdir(byid):
        for name in os.listdir(byid):
            links[os.path.realpath(os.path.join(byid, name))] = name
    out = []
    for p in sorted(glob.glob("/dev/ttyACM*")):
        out.append((p, links.get(os.path.realpath(p), "?")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="e.g. /dev/ttyACM0 (auto-detect if omitted)")
    ap.add_argument("--pings", type=int, default=20)
    ap.add_argument("--skip-motion", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print("  SO-101 LEADER ARM HEALTH CHECK")
    print("=" * 60)

    # ---- [1] DETECTION ----
    print("\n[1] USB DETECTION")
    ports = list_ports()
    if not ports:
        print("  ✗ FAIL: no /dev/ttyACM* found.")
        print("    -> USB not detected. Check the USB cable and try a different PC port")
        print("       (plug straight into the PC, not through a hub).")
        return
    for p, serial in ports:
        print(f"  • {p}   (serial {serial})")
    port = args.port or ports[0][0]
    if args.port is None and len(ports) > 1:
        print(f"  NOTE: multiple ports found — testing {port}. Use --port to choose another.")
    print(f"  ✓ detected. Testing: {port}")

    try:
        ph = PortHandler(port)
        if not ph.openPort():
            raise RuntimeError("openPort() failed")
        ph.setBaudRate(1000000)
    except Exception as e:
        print(f"  ✗ FAIL: cannot open {port}: {e}")
        return
    pk = PacketHandler(0)

    # ---- [2] SERVOS + POWER ----
    print(f"\n[2] SERVOS + POWER  ({args.pings} pings)")
    counts = {i: 0 for i in range(1, 7)}
    full = 0
    for _ in range(args.pings):
        found = [i for i in range(1, 7) if pk.ping(ph, i)[1] == 0]
        for i in found:
            counts[i] += 1
        if len(found) == 6:
            full += 1
        time.sleep(0.25)
    for i in range(1, 7):
        bar = "OK " if counts[i] == args.pings else "!! "
        print(f"    servo {i} {NAMES[i-1]:11s}: {counts[i]:2d}/{args.pings} {bar}")
    total = sum(counts.values())
    print(f"  full 6/6 pings: {full}/{args.pings}")

    verdict_power = None
    if total == 0:
        verdict_power = "NO_POWER"
        print("  ✗ FAIL: no servo responded.")
        print("    -> SERVOS UNPOWERED. Connect/switch on the leader's power brick")
        print("       (separate from USB). Then re-run.")
        ph.closePort()
        _verdict("NO_POWER", None)
        return
    elif full < args.pings:
        verdict_power = "FLAKY"
        print("  ✗ WARN: servos drop in/out (not 6/6 every ping).")
        print("    -> under-powered supply OR loose servo cable. Reseat cables;")
        print("       use a correctly-rated dedicated power brick.")
    else:
        verdict_power = "OK"
        print("  ✓ all 6 servos stable at rest.")

    # ---- [3] MOTION ----
    verdict_motion = "SKIP"
    if not args.skip_motion:
        print("\n[3] MOTION TEST — corruption while moving")
        input("    >> Press ENTER, then SWEEP EVERY JOINT + gripper. Press Ctrl-C when done...")
        t0 = time.time()
        corrupt = {i: 0 for i in range(1, 7)}
        commfail = {i: 0 for i in range(1, 7)}
        mn = {i: 10**9 for i in range(1, 7)}
        mx = {i: -10**9 for i in range(1, 7)}
        cur = {i: None for i in range(1, 7)}  # last read: int, or ('bad', text)
        last_print = 0.0
        drawn = False
        print()  # leave a blank line above the table

        def render():
            lines = [f"    {'JOINT':<11} {'POS':>6} {'MIN':>5} {'MAX':>5} {'SPAN':>5}   STATUS"]
            for j in range(1, 7):
                v = cur[j]
                if isinstance(v, int):
                    poss = f"{v:6d}"
                elif v is None:
                    poss = "   ---"
                else:
                    poss = f"{v[1]:>6}"
                lo = mn[j] if mx[j] >= mn[j] else 0
                hi = mx[j] if mx[j] >= mn[j] else 0
                span = hi - lo
                if corrupt[j] > 0:
                    status = f"CORRUPT x{corrupt[j]}"
                elif commfail[j] > 5:
                    status = f"drops x{commfail[j]}"
                elif span < 50:
                    status = "move it"
                else:
                    status = "ok"
                lines.append(f"    {NAMES[j-1]:<11} {poss} {lo:5d} {hi:5d} {span:5d}   {status}")
            elapsed = time.time() - t0
            total_bad = sum(corrupt.values())
            lines.append(f"    elapsed: {elapsed:5.1f}s   corrupt reads: {total_bad}   (Ctrl-C when done)")
            block = "".join("\033[2K" + ln + "\n" for ln in lines)
            nonlocal drawn
            if drawn:
                block = f"\033[{len(lines)}A" + block  # move cursor back up over the table
            drawn = True
            print(block, end="", flush=True)

        try:
            while True:  # runs until you press Ctrl-C — no timeout
                for i in range(1, 7):
                    raw, comm, _ = pk.read2ByteTxRx(ph, i, ADDR_POS)
                    if comm != 0:
                        commfail[i] += 1
                        cur[i] = ("bad", "----")
                    elif raw < 0 or raw > 4095:
                        corrupt[i] += 1
                        cur[i] = ("bad", f"!{raw}")
                    else:
                        mn[i] = min(mn[i], raw)
                        mx[i] = max(mx[i], raw)
                        cur[i] = raw
                now = time.time()
                if now - last_print > 0.1:  # redraw table in place ~10 Hz
                    render()
                    last_print = now
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        render()
        print("\n    results:")
        any_corrupt = False
        for i in range(1, 7):
            span = (mx[i] - mn[i]) if mx[i] >= mn[i] else 0
            flags = []
            if corrupt[i] > 0:
                flags.append(f"CORRUPT={corrupt[i]}")
                any_corrupt = True
            if commfail[i] > 5:
                flags.append(f"commfail={commfail[i]}")
            if span < 50:
                flags.append("(didn't move?)")
            print(f"      servo {i} {NAMES[i-1]:11s}: range {mn[i]}..{mx[i]} span {span}  {' '.join(flags)}")
        verdict_motion = "CORRUPT" if any_corrupt else "OK"
        if any_corrupt:
            print("    ✗ corruption under motion -> bad servo cable on the flagged servo(s).")
            print("      Replace that cable (reseating usually won't hold).")
        else:
            print("    ✓ clean under motion.")

    ph.closePort()
    _verdict(verdict_power, verdict_motion)


def _verdict(power, motion):
    print("\n" + "=" * 60)
    print("  VERDICT")
    if power == "NO_POWER":
        print("  ✗ FAIL — servos unpowered.")
    elif power == "FLAKY":
        print("  ✗ FAIL — flaky/under-powered connection.")
    elif motion == "CORRUPT":
        print("  ✗ FAIL — corruption under motion (bad servo cable).")
    elif power == "OK" and motion in ("OK", "SKIP"):
        tail = "(motion skipped)" if motion == "SKIP" else "clean under motion"
        print(f"  ✓ PASS — detected, powered, 6/6 stable, {tail}. Ready to calibrate.")
    else:
        print(f"  ? power={power} motion={motion}")
    print("=" * 60)


if __name__ == "__main__":
    main()
