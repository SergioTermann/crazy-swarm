#!/usr/bin/env python3
"""Single-Crazyflie safety test for the paper obstacle-avoidance term."""

from __future__ import division, print_function

import argparse
import datetime
import math
import os
import sys
from typing import List, Tuple

import numpy as np

from hra_apf_crazyswarm import (
    EmergencyLandingMonitor,
    EmergencyLandingRequested,
    SafetyStop,
    VelocityEstimator,
    apply_live_initial_state,
    apply_flight_guards,
    collect_positions,
    collect_velocities_if_available,
    command_obstacle_uavs,
    command_translational_state,
    load_crazyflies_yaml,
    load_controller_raw_config,
    log_row,
    reorder_crazyflies,
    safe_land,
    validate_obstacle_uav_positions,
    write_buffered_logs,
)
from paper_controller import (
    ControllerConfig,
    HRAAPFController,
    controller_initial_positions,
)


SCRIPT_DIR = "/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch"
CRAZYSWARM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if CRAZYSWARM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CRAZYSWARM_SCRIPTS_DIR)


def embedded_single_config():
    return {
        "control": {
            "rate_hz": 30.0,
            "duration": 16.0,
            "velocity_limit": 0.30,
        },
        "gains": {
            "kp": 1.05,
            "kv": 3.30,
            "k_f": 1.00,
            "lambda_e": 0.32,
            "k_i": 1.00,
            "k_o": 0.08,
            "k_t": 0.06,
        },
        "avoidance": {
            "uav_radius": 0.05,
            "obstacle_buffer": 0.08,
            "inter_uav_buffer": 0.00,
            "obstacle_influence_radius": 0.36,
            "safe_clearance": 0.1,
            "prediction_horizon": 0.35,
            "closing_speed_weight": 0.45,
            "risk_adaptive_gain": 1.10,
            "risk_regularizer": 0.293778,
            "apf_epsilon": 0.08,
            "inter_uav_sensing_radius": 0.30,
            "auxiliary_sensing_radius": 2.0,
        },
        "leader_advantage": {
            "target_weight": 0.50,
            "obstacle_weight": 0.40,
            "continuity_weight": 0.10,
            "target_scale": 1.0,
            "continuity_scale": 1.0,
        },
        "switching": {
            "hysteresis": [0.010],
            "persistence": [0.580],
            "stability_xi": 0.20,
        },
        "formation": {
            "layers": [[0]],
            "offsets": [[0.00, 0.00, 0.00]],
        },
        "mission": {
            "start_anchor": [-1.30, 0.30, 0.00],
            "target": [1.10, 0.30, 0.50],
            "adapt_target_x_to_workspace": False,
            "target_x_margin": 0.20,
            "target_velocity": [0.00, 0.00, 0.00],
        },
        "crazyswarm": {
            "uav_ids": [2],
            "takeoff_height": 0.50,
        },
        "state_estimation": {
            "velocity_filter_alpha": 0.35,
        },
        "flight_guard": {
            "start_position_tolerance": 0.60,
            "minimum_inter_uav_distance": 0.00,
            "minimum_obstacle_clearance": 0.08,
            "obstacle_boundary_margin": 0.25,
            "workspace_min": [-1.50, -2.00, 0.00],
            "workspace_max": [1.50, 1.00, 1.30],
        },
        "initialization": {
            "use_live_start": True,
        },
        "obstacles": [
            {
                "vehicle_id": 3,
                "center": [-0.10, 0.60, 0.50],
                "radius": 0.05,
                "velocity": [-0.075, -0.130, 0.0],
                "height": 0.10,
            },
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    # Crazyswarm also parses this parent parser and adds --sim/--vis/--dt.
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("single-Crazyflie HRA-APF avoidance test")
    group.add_argument(
        "--controller-config",
        default="embedded",
        help="Controller parameters. Default: embedded. Pass a YAML path to override.")
    group.add_argument(
        "--crazyflies-config",
        default=os.path.join(SCRIPT_DIR, "crazyflies.yaml"),
        help="Crazyswarm YAML containing one test UAV and one obstacle UAV.")
    group.add_argument(
        "--log-dir",
        default=os.path.join(SCRIPT_DIR, "logs"),
        help="Directory for CSV logs.")
    group.add_argument(
        "--dry-run", action="store_true",
        help="Validate parameters and print the planned test without flying.")
    group.add_argument(
        "--confirm-real-flight", action="store_true",
        help="Required when running without --sim.")
    group.add_argument(
        "--disable-emergency-key", action="store_true",
        help="Disable terminal emergency landing keys l/q/Esc.")
    group.add_argument("--takeoff-duration", type=float, default=3.0)
    group.add_argument("--landing-duration", type=float, default=3.0)
    group.add_argument("--landing-height", type=float, default=0.04)
    return parser


def validate_single_setup(config: ControllerConfig, ids: List[int]) -> None:
    expected_ids = config.uav_ids + config.obstacle_uav_ids
    if config.n != 1 or len(config.obstacles) != 1:
        raise ValueError(
            "single test requires one controlled UAV and one obstacle UAV")
    if len(config.obstacle_uav_ids) != 1:
        raise ValueError(
            "single test requires obstacles[0].vehicle_id for the real obstacle UAV")
    if ids != expected_ids:
        raise ValueError(
            "crazyflies YAML IDs {} differ from controlled+obstacle IDs {}"
            .format(ids, expected_ids))
    if not math.isclose(config.takeoff_height, config.target[2], abs_tol=1e-9):
        raise ValueError(
            "single test keeps altitude fixed; takeoff_height must equal target z")


def print_setup(config: ControllerConfig, crazyflies_config: str) -> None:
    obstacle = config.obstacles[0]
    inflated = obstacle.radius + config.r_u + config.b_o
    print("Single-Crazyflie HRA-APF avoidance test")
    print("formation UAV IDs={}".format(config.uav_ids))
    print("obstacle UAV IDs={}".format(config.obstacle_uav_ids))
    print("crazyflies config={}".format(crazyflies_config))
    print("rate={:.1f} Hz, duration={:.1f} s, vmax={:.2f} m/s".format(
        config.rate_hz, config.duration, config.velocity_limit))
    print("takeoff/target altitude={:.2f} m".format(config.takeoff_height))
    print("start={}, target={}".format(
        config.start_anchor.tolist(), config.target.tolist()))
    print("dynamic obstacle center={}, velocity={}, radius={:.3f} m, inflated radius={:.3f} m".format(
        obstacle.center0.tolist(), obstacle.velocity.tolist(), obstacle.radius, inflated))
    print("logs are buffered in memory and saved after landing")


def expected_start_after_takeoff(config: ControllerConfig) -> np.ndarray:
    return controller_initial_positions(config)


def validate_start(config: ControllerConfig, positions: np.ndarray,
                   tolerance: float) -> None:
    expected = expected_start_after_takeoff(config)
    error = float(np.linalg.norm(positions[0] - expected[0]))
    if error > tolerance:
        raise SafetyStop(
            "initial position error {:.3f} m exceeds {:.3f} m; check chooser.py/RViz"
            .format(error, tolerance))


def run_controller(args: argparse.Namespace, config: ControllerConfig,
                   raw_config, crazyflies_yaml: str) -> Tuple[str, str, int]:
    from pycrazyswarm import Crazyswarm

    swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml, parent_parser=build_parser())
    time_helper = swarm.timeHelper
    all_cfs = swarm.allcfs
    expected_ids = config.uav_ids + config.obstacle_uav_ids
    all_crazyflies = reorder_crazyflies(all_cfs.crazyflies, expected_ids)
    crazyflies = all_crazyflies[:config.n]
    obstacle_crazyflies = all_crazyflies[config.n:]

    crazyflies[0].takeoff(
        targetHeight=float(config.takeoff_height),
        duration=float(args.takeoff_duration))
    obstacle_crazyflies[0].takeoff(
        targetHeight=float(config.obstacles[0].center0[2]),
        duration=float(args.takeoff_duration))
    time_helper.sleep(float(args.takeoff_duration) + 0.5)

    positions = collect_positions(crazyflies)
    obstacle_positions = collect_positions(obstacle_crazyflies)
    if apply_live_initial_state(config, raw_config, positions, obstacle_positions):
        print("live start applied: start_anchor={}, obstacle_center={}".format(
            config.start_anchor.tolist(), config.obstacles[0].center0.tolist()))
    guard = raw_config["flight_guard"]
    validate_start(
        config, positions, float(guard["start_position_tolerance"]))
    validate_obstacle_uav_positions(config, obstacle_positions, guard)

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, "single_avoidance_{}.csv".format(timestamp))
    switch_path = os.path.join(
        args.log_dir, "single_avoidance_switches_{}.csv".format(timestamp))

    controller = HRAAPFController(config)
    estimator = VelocityEstimator(positions, config.velocity_filter_alpha)
    buffered_rows = []
    buffered_switch_rows = []
    emergency = EmergencyLandingMonitor(
        enabled=not args.disable_emergency_key)
    if emergency.enabled:
        print("Emergency landing key active: press l, q, or Esc to land both UAVs.")
    else:
        print("Emergency landing key inactive; Ctrl+C still lands both UAVs.")
    failure = None
    start_time = time_helper.time()
    previous_time = start_time
    obstacle_hovered = False

    try:
        try:
            while not time_helper.isShutdown():
                if emergency.poll():
                    raise EmergencyLandingRequested(
                        "emergency landing key {} pressed".format(
                            emergency.triggered_key))
                now = time_helper.time()
                elapsed = now - start_time
                if elapsed >= config.duration:
                    break
                measured_dt = max(now - previous_time, 1e-6)
                control_dt = min(max(measured_dt, 0.005), 0.05)
                previous_time = now

                positions = collect_positions(crazyflies)
                velocities = collect_velocities_if_available(crazyflies)
                if velocities is None:
                    velocities = estimator.update(positions, measured_dt)
                else:
                    estimator.previous_positions = positions.copy()
                    estimator.filtered_velocity = velocities.copy()

                obstacle_positions = collect_positions(obstacle_crazyflies)
                obstacle_velocities = collect_velocities_if_available(obstacle_crazyflies)
                _, reference_obstacle_velocities = command_obstacle_uavs(
                    config, obstacle_crazyflies, elapsed, guard)
                if (not obstacle_hovered and
                        np.linalg.norm(config.obstacles[0].velocity) > 1e-9 and
                        np.linalg.norm(reference_obstacle_velocities[0]) < 1e-9):
                    obstacle_hovered = True
                    print("t={:.2f}s obstacle UAV {} reached boundary margin; hovering."
                          .format(elapsed, config.obstacle_uav_ids[0]))
                if obstacle_velocities is None:
                    obstacle_velocities = reference_obstacle_velocities
                diagnostic = controller.step(
                    positions, velocities, elapsed, dt=control_dt,
                    obstacle_positions=obstacle_positions,
                    obstacle_velocities=obstacle_velocities)
                apply_flight_guards(config, positions, diagnostic, guard)
                controls = diagnostic["controls"]
                next_velocity = controller.saturate_rows(
                    velocities + control_dt * controls, config.velocity_limit)
                next_position = positions + control_dt * next_velocity
                command_translational_state(
                    crazyflies, next_position, next_velocity, controls)
                buffered_rows.append(log_row(
                    elapsed, config, controller, positions, velocities, diagnostic,
                    obstacle_positions=obstacle_positions,
                    obstacle_velocities=obstacle_velocities))
                time_helper.sleepForRate(config.rate_hz)
        finally:
            emergency.close()
    except (KeyboardInterrupt, SafetyStop, Exception) as error:
        failure = error

    safe_land(all_cfs, all_crazyflies, time_helper, args.landing_height,
              args.landing_duration)
    write_buffered_logs(
        log_path, switch_path, config, buffered_rows, buffered_switch_rows)
    print("Saved single-UAV log: {}".format(log_path))
    print("Saved switch log: {}".format(switch_path))

    if failure is not None:
        if isinstance(failure, EmergencyLandingRequested):
            print("Emergency landing completed: {}".format(failure))
            return log_path, switch_path, 0
        if isinstance(failure, KeyboardInterrupt):
            print("Interrupted; landing completed.")
            return log_path, switch_path, 130
        if isinstance(failure, SafetyStop):
            print("Safety stop: {}".format(failure), file=sys.stderr)
            return log_path, switch_path, 2
        raise failure
    print("Completed single-UAV avoidance test.")
    return log_path, switch_path, 0


def main() -> int:
    parser = build_parser()
    args, _ = parser.parse_known_args()
    raw_config, config_source = load_controller_raw_config(
        args.controller_config, embedded_single_config)
    config = ControllerConfig(raw_config)
    crazyflies_yaml, ids = load_crazyflies_yaml(args.crazyflies_config)
    validate_single_setup(config, ids)
    print_setup(config, args.crazyflies_config)
    print("controller parameters={}".format(config_source))
    if args.dry_run:
        return 0

    simulation = "--sim" in sys.argv[1:]
    if not simulation and not args.confirm_real_flight:
        raise RuntimeError(
            "real flight is locked. First run with --sim; for real hardware, "
            "check ID/channel/initial position and add --confirm-real-flight.")
    _, _, exit_code = run_controller(args, config, raw_config, crazyflies_yaml)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
