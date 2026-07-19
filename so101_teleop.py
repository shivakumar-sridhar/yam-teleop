#!/usr/bin/env python3
"""SO-101 leader -> YAM follower, ABSOLUTE full-range teleop (calibrated).

Full range: each leader joint's captured tick range maps to the YAM joint's
full limit range, so a full leader sweep = full YAM motion.

Safety:
  * SLOW-MOVE-TO-START: on engage the YAM eases from its current pose to the
    leader's mapped pose over a distance-proportional duration (never a snap).
  * VELOCITY CLAMP while streaming: each joint may change at most stream_vel*dt
    per step, so a bad serial read can't cause a jump.
  * command_joint_pos clips arm joints to YAM limits; exit -> gravity-comp idle.
  * --dry-run prints commands, no YAM connection/motion.

Usage:
  DRY : .venv/bin/python so101_yam_full.py --dry-run
  LIVE: .venv/bin/python so101_yam_full.py --channel can1 --seconds 60
"""
import argparse
import glob
import json
import math
import os
import re
import time
from dataclasses import dataclass

import numpy as np
from scservo_sdk import PacketHandler, PortHandler

ADDR_POS = 56
TWO_PI = 2 * math.pi
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leader_calibration.json")

# Per-leader tick ranges are loaded at runtime from leader_calibration.json,
# keyed by the controller's serial id (run calibrate_leaders.py to populate it).
LEADER_RANGE = {}


def _controller_id_for_port(port: str):
    """Resolve the stable controller serial id for a /dev/ttyACM* port."""
    byid = "/dev/serial/by-id"
    if os.path.isdir(byid):
        for name in os.listdir(byid):
            if os.path.realpath(os.path.join(byid, name)) == os.path.realpath(port):
                m = re.search(r"Serial_([A-Za-z0-9]+)", name)
                return m.group(1) if m else name
    return None


def load_leader_range(port: str):
    """Return (ranges dict, controller_id) for the leader on `port`, from JSON."""
    if not os.path.exists(CALIB_PATH):
        raise SystemExit(f"No calibration file at {CALIB_PATH}. Run calibrate_leaders.py first.")
    with open(CALIB_PATH) as f:
        calib = json.load(f)
    cid = _controller_id_for_port(port)
    if cid is None or cid not in calib:
        raise SystemExit(
            f"No calibration for the controller on {port} (id={cid}).\n"
            f"  calibrated controllers: {list(calib.keys())}\n"
            f"  -> run: .venv/bin/python calibrate_leaders.py"
        )
    return {k: tuple(v) for k, v in calib[cid]["ranges"].items()}, cid
YAM_LIMITS = {
    "J1": (-2.61799, 3.05433), "J2": (0.0, 3.65000), "J3": (0.0, 3.66519),
    "J4": (-1.57080, 1.57080), "J5": (-1.57080, 1.57080), "J6": (-2.09440, 2.09440),
}
YAM_IDX = {"J1": 0, "J2": 1, "J3": 2, "J4": 3, "J5": 4, "J6": 5}


@dataclass
class JointMap:
    leader: str
    yam: str
    sign: int  # <0 => reversed (leader-min -> YAM-max)


ARM_MAP = [
    JointMap("pan", "J1", -1),   # confirmed reversed
    JointMap("lift", "J2", +1),
    JointMap("elbow", "J3", -1),  # confirmed reversed
    JointMap("wrist_flex", "J4", -1),  # up/down is J4 (J5 is the yaw/left-right joint); flipped direction
    JointMap("wrist_roll", "J6", -1),  # flipped: wrist roll was reversed
]
HELD_JOINT = "J5"  # yaw/left-right — no leader equivalent, held fixed
GRIPPER_SIGN = +1
LEADER_NAMES = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"]

# Forearm roll (J4) has no leader equivalent. Holding it at ~+90deg reorients the
# wrist so J5 (wrist_flex) becomes the up/down axis instead of side-to-side.
J4_HOLD = math.pi / 2


def norm(tick, lo, hi, sign):
    n = (tick - lo) / (hi - lo) if hi != lo else 0.0
    n = min(1.0, max(0.0, n))
    return (1.0 - n) if sign < 0 else n


def read_leader(ph, pk):
    out = {}
    for i, name in enumerate(LEADER_NAMES, start=1):
        raw, comm, _ = pk.read2ByteTxRx(ph, i, ADDR_POS)
        out[name] = raw if comm == 0 else None
    return out


def leader_to_target(ticks):
    q = np.zeros(7)
    for m in ARM_MAP:
        llo, lhi = LEADER_RANGE[m.leader]
        n = norm(ticks[m.leader], llo, lhi, m.sign)
        ylo, yhi = YAM_LIMITS[m.yam]
        q[YAM_IDX[m.yam]] = ylo + n * (yhi - ylo)
    q[YAM_IDX[HELD_JOINT]] = 0.0  # hold the unused yaw/left-right joint at neutral
    q[6] = norm(ticks["gripper"], *LEADER_RANGE["gripper"], GRIPPER_SIGN)
    return q


def publish_state(channel, follower, action):
    """Publish latest follower state + leader action to /dev/shm for the recorder."""
    path = f"/dev/shm/teleop_{channel}.json"
    try:
        with open(path + ".tmp", "w") as f:
            json.dump({"follower": [float(x) for x in follower],
                       "action": [float(x) for x in action],
                       "t": time.time()}, f)
        os.replace(path + ".tmp", path)
    except Exception:
        pass


def main():
    global J4_HOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--engage-vel", type=float, default=0.5, help="rad/s during slow-move-to-start")
    ap.add_argument("--stream-vel", type=float, default=2.0, help="max rad/s per joint while streaming")
    ap.add_argument("--j4-hold", type=float, default=J4_HOLD,
                    help="fixed forearm-roll angle (rad); ~+/-pi/2 makes J5 the up/down axis")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    J4_HOLD = max(-1.5708, min(1.5708, args.j4_hold))  # clip to J4 limits

    global LEADER_RANGE
    LEADER_RANGE, cid = load_leader_range(args.port)
    print(f"loaded calibration for controller [{cid}] from {os.path.basename(CALIB_PATH)}")

    ph = PortHandler(args.port); ph.openPort(); ph.setBaudRate(1000000)
    pk = PacketHandler(0)
    lt = read_leader(ph, pk)
    if any(v is None for v in lt.values()):
        print(f"leader read failed: {lt}"); return
    tgt = leader_to_target(lt)

    robot = None
    if not args.dry_run:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
        from i2rt.utils.utils import override_log_level
        override_log_level()
        robot = get_yam_robot(channel=args.channel, arm_type=ArmType.YAM,
                              gripper_type=GripperType.LINEAR_4310)
        assert robot.num_dofs() == 7
        cur = np.asarray(robot.get_joint_pos(), dtype=float).copy()
    else:
        cur = np.zeros(7)

    dt = 1.0 / args.hz
    print(f"[{'DRY' if args.dry_run else 'LIVE'}] target from leader = ["
          + " ".join(f"{v:+.2f}" for v in tgt) + "]")
    print("current YAM pose            = [" + " ".join(f"{v:+.2f}" for v in cur) + "]")
    arm_dist = float(np.max(np.abs(tgt[:6] - cur[:6])))
    dur = max(3.0, arm_dist / args.engage_vel)
    print(f"SLOW-MOVE to leader pose over {dur:.1f}s (max joint move {arm_dist:.2f} rad). "
          f"Keep clear / hand on e-stop.")

    prev = cur.copy()
    try:
        # --- slow-move-to-start (arm joints interpolate; gripper -> target) ---
        steps = max(1, int(dur / dt))
        for i in range(1, steps + 1):
            a = i / steps
            cmd = cur * (1 - a) + tgt * a
            cmd[6] = tgt[6]
            if not args.dry_run:
                robot.command_joint_pos(cmd)
            prev = cmd
            time.sleep(dt)
        print("engaged — streaming full-range teleop. Move the leader freely. Ctrl-C to stop.\n")

        # --- streaming with per-step velocity clamp ---
        t0 = time.time(); frame = 0
        max_step = args.stream_vel * dt
        run_forever = args.seconds <= 0
        while run_forever or (time.time() - t0 < args.seconds):
            lt = read_leader(ph, pk)
            if any(lt[n] is None for n in LEADER_NAMES):
                time.sleep(dt); continue
            tgt = leader_to_target(lt)
            q = prev.copy()
            for j in range(6):
                d = max(-max_step, min(max_step, tgt[j] - prev[j]))
                q[j] = prev[j] + d
            q[6] = tgt[6]
            if not args.dry_run:
                robot.command_joint_pos(q)
            prev = q
            # publish state+action for the episode recorder (observation.state / action)
            follower = robot.get_joint_pos() if robot is not None else q
            publish_state(args.channel, follower, q)
            frame += 1
            if frame % max(1, int(args.hz * 3)) == 0:  # ~every 3s
                print("q=[" + " ".join(f"{v:+.2f}" for v in q) + "]", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if robot is not None:
            try:
                robot.enter_gravity_comp_idle()
                print("YAM returned to gravity-comp idle (backdrivable).")
            except Exception as e:
                print(f"cleanup warning: {e}")
        ph.closePort()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
    # YAM motor-chain runs a non-daemon control thread that otherwise keeps the
    # process (and the CAN channel) alive after main() returns. Force exit so the
    # channel is released for the next run.
    import os
    os._exit(0)
