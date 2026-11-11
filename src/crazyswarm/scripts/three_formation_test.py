#!/usr/bin/env python3
"""Three-Crazyflie takeoff, follow a precomputed trajectory from a txt log,
then land. Designed for the Crazyswarm flight environment.

The trajectory log (tab-separated, header line starts with '#') has columns:
    t  ref_x ... ref_psi  cf1_x cf1_y cf1_z cf1_psi cf1_vx cf1_vy cf1_vz cf1_vpsi
    cf2_... cf3_...   cf1_err cf2_err cf3_err
Position/velocity columns for each cf are located automatically from the header.
"""

from __future__ import division, print_function

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRAZYSWARM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if CRAZYSWARM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CRAZYSWARM_SCRIPTS_DIR)


CF_KEYS = ["x", "y", "z", "psi", "vx", "vy", "vz", "vpsi"]
CF_NAMES = ["cf%d" % i for i in (1, 2, 3)]


def build_parser() -> argparse.ArgumentParser:
    # Crazyswarm also parses this parent parser and adds --sim/--vis/--dt.
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("three-Crazyflie takeoff + trajectory test")
    group.add_argument(
        "--crazyflies-config",
        default=os.path.join("/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch", "crazyflies_3_test.yaml"),
        help="Crazyswarm YAML containing exactly three Crazyflies.")
    group.add_argument(
        "--traj",
        default=os.path.join(SCRIPT_DIR, "lmpc_crazyswarm_traj.txt"),
        help="Path to the trajectory txt log (one row per timestep, dt fixed).")
    group.add_argument("--height", type=float, default=0.5,
                       help="Takeoff hover height before following the trajectory.")
    group.add_argument("--hover-duration", type=float, default=1.0)
    group.add_argument("--takeoff-duration", type=float, default=3.0)
    group.add_argument("--landing-duration", type=float, default=3.0)
    group.add_argument("--landing-height", type=float, default=0.04)
    group.add_argument(
        "--dry-run", action="store_true",
        help="Parse the trajectory/config and print the plan without flying.")
    return parser


def load_crazyflies_yaml(path: str) -> Tuple[str, List[int]]:
    with open(path, "r") as stream:
        text = stream.read()
    parsed = yaml.safe_load(text)
    entries = parsed.get("crazyflies", []) if isinstance(parsed, dict) else []
    ids = [int(item["id"]) for item in entries]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("crazyflies YAML must contain exactly three distinct IDs")
    return text, ids


def parse_traj(path: str) -> Tuple[List[float], Dict[str, List[Tuple]]]:
    """Return (times, {cfN: [(pos, vel, acc, yaw, omega), ...]}) for each cf."""
    with open(path, "r") as fh:
        lines = fh.readlines()

    # The column-header line is the '#' line that contains the column names
    # (e.g. 'cf1_x'), not the free-text comment lines above it.
    header_index = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            parts = line.lstrip()[1:].strip().split("\t")
            if "cf1_x" in parts:
                header_index = idx
                break
    if header_index is None:
        raise ValueError("trajectory file has no column header line "
                         "(a '#' line containing 'cf1_x')")

    header = lines[header_index].lstrip()[1:].strip().split("\t")
    col = {name: i for i, name in enumerate(header)}

    cf_cols = {}
    for name in CF_NAMES:
        cols = [col[name + "_" + k] for k in CF_KEYS]
        cf_cols[name] = cols

    times: List[float] = []
    states = {name: [] for name in CF_NAMES}
    for line in lines[header_index + 1:]:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if not parts or parts[0].startswith("#"):
            continue
        try:
            row = [float(v) for v in parts]
        except ValueError:
            continue
        times.append(row[col["t"]])
        for name, cols in cf_cols.items():
            x, y, z = row[cols[0]], row[cols[1]], row[cols[2]]
            psi = row[cols[3]]
            vx, vy, vz = row[cols[4]], row[cols[5]], row[cols[6]]
            vpsi = row[cols[7]]
            states[name].append(([x, y, z], [vx, vy, vz], [0.0, 0.0, 0.0],
                                 psi, vpsi))
    return times, states


def stop_individual_cfs(all_cfs: Any) -> None:
    for cf in getattr(all_cfs, "crazyflies", []):
        try:
            cf.cmdStop()
        except Exception:
            pass


def main() -> int:
    parser = build_parser()
    args, _ = parser.parse_known_args()

    crazyflies_yaml, ids = load_crazyflies_yaml(args.crazyflies_config)
    times, states = parse_traj(args.traj)

    print("Three-Crazyflie takeoff + trajectory test")
    print("IDs (must match cf1..cf3 order in the log): {}".format(ids))
    print("traj: {} ({:d} waypoints, t_end={:.2f} s)".format(
        args.traj, len(times), times[-1] if times else 0.0))
    print("takeoff height={:.2f} m, hover={:.1f} s".format(
        args.height, args.hover_duration))
    print("config: {}".format(args.crazyflies_config))
    if args.dry_run:
        print("t=0 cf1:{!s} cf2:{!s} cf3:{!s}".format(
            states["cf1"][0][0], states["cf2"][0][0], states["cf3"][0][0]))
        return 0

    from pycrazyswarm import Crazyswarm

    swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml, parent_parser=parser)
    time_helper = swarm.timeHelper
    all_cfs = swarm.allcfs
    cfs = list(all_cfs.crazyflies)
    if len(cfs) != 3:
        raise RuntimeError("expected exactly three Crazyflies, got %d" % len(cfs))

    try:
        print("Taking off...")
        all_cfs.takeoff(targetHeight=float(args.height),
                        duration=float(args.takeoff_duration))
        time_helper.sleep(float(args.takeoff_duration) + 0.5)

        print("Hovering before trajectory...")
        time_helper.sleep(float(args.hover_duration))

        print("Following trajectory...")
        start_time = time_helper.time()
        for i in range(len(times)):
            target = start_time + times[i]
            while time_helper.time() < target:
                time_helper.sleep(min(0.02, max(0.001, target - time_helper.time())))
            for cf, name in zip(cfs, CF_NAMES):
                cf.cmdFullState(*states[name][i])

        print("Trajectory done; hovering before landing...")
        time_helper.sleep(0.5)

        print("Landing...")
        all_cfs.land(targetHeight=float(args.landing_height),
                     duration=float(args.landing_duration))
        time_helper.sleep(float(args.landing_duration) + 0.5)
        all_cfs.stop()
    except KeyboardInterrupt:
        print("Interrupted; landing now.")
        all_cfs.land(targetHeight=float(args.landing_height),
                     duration=float(args.landing_duration))
        time_helper.sleep(float(args.landing_duration) + 0.5)
        stop_individual_cfs(all_cfs)
        return 130

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())