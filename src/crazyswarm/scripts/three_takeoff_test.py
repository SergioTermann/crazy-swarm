#!/usr/bin/env python3
"""Minimal three-Crazyflie takeoff, hover, and landing test."""

from __future__ import division, print_function

import argparse
import os
import sys
from typing import Any, List, Tuple

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRAZYSWARM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if CRAZYSWARM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CRAZYSWARM_SCRIPTS_DIR)


def build_parser() -> argparse.ArgumentParser:
    # Crazyswarm also parses this parent parser and adds --sim/--vis/--dt.
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("three-Crazyflie takeoff test")
    group.add_argument(
        "--crazyflies-config",
        default=os.path.join("/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch", "crazyflies_3_test.yaml"),
        help="Crazyswarm YAML containing exactly three Crazyflies.")
    group.add_argument("--height", type=float, default=0.60)
    group.add_argument("--hover-duration", type=float, default=5.0)
    group.add_argument("--takeoff-duration", type=float, default=3.0)
    group.add_argument("--landing-duration", type=float, default=3.0)
    group.add_argument("--landing-height", type=float, default=0.04)
    group.add_argument(
        "--dry-run", action="store_true",
        help="Validate the YAML and print the planned test without flying.")
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

    print("Three-Crazyflie takeoff test")
    print("IDs: {}".format(ids))
    print("height={:.2f} m, hover={:.1f} s".format(
        args.height, args.hover_duration))
    print("config: {}".format(args.crazyflies_config))
    if args.dry_run:
        return 0

    from pycrazyswarm import Crazyswarm

    swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml, parent_parser=parser)
    time_helper = swarm.timeHelper
    all_cfs = swarm.allcfs

    try:
        print("Taking off...")
        all_cfs.takeoff(targetHeight=float(args.height),
                        duration=float(args.takeoff_duration))
        time_helper.sleep(float(args.takeoff_duration) + 0.5)

        print("Hovering...")
        time_helper.sleep(float(args.hover_duration))

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
