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
            "rate_hz": 50.0,
            "duration": 16.0,
            "velocity_limit": 0.35,
        },
        "gains": {
            "kp": 1.20,
            "kv": 2.80,
            "k_f": 1.00,
            "lambda_e": 0.32,
            "k_i": 1.00,
            "k_o": 0.4,
            "k_t": 0.25,
        },
        "avoidance": {
            "uav_radius": 0.05,
            "obstacle_buffer": 0.08,
            "inter_uav_buffer": 0.00,
            "obstacle_influence_radius": 0.3,
            "safe_clearance": 0.18,
            "prediction_horizon": 2.0,
            "closing_speed_weight": 0.6,
            "risk_adaptive_gain": 1.25,
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
            "target": [1.30, 0.30, 0.50],
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
            "start_position_tolerance": 0.12,
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
                "center": [0.00, 0.0, 0.50],
                "radius": 0.05,
                "velocity": [-0.025, -0.05, 0.0],
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
    group.add_argument(
        "--debug-control", action="store_true",
        help="Print controller output and commanded state during flight.")
    group.add_argument(
        "--debug-control-interval", type=float, default=0.50,
        help="Seconds between --debug-control printouts.")
    group.add_argument(
        "--force-forward-test", action="store_true",
        help="Bypass the controller and command a small forward velocity.")
    group.add_argument(
        "--force-forward-speed", type=float, default=0.15,
        help="Forward x velocity used by --force-forward-test, in m/s.")
    group.add_argument(
        "--position-command", action="store_true",
        help="Use cf.cmdPosition instead of full-state commands.")
    group.add_argument(
        "--velocity-command", action="store_true",
        help="Use cf.cmdVelocityWorld instead of full-state commands.")
    group.add_argument(
        "--full-state-command", action="store_true",
        help="Force the original full-state command path on real hardware.")
    group.add_argument(
        "--position-command-lookahead", type=float, default=0.25,
        help="Maximum cmdPosition lead distance ahead of the measured position.")
    group.add_argument(
        "--position-stream-mode",
        choices=("hybrid", "controller", "direct"),
        default="hybrid",
        help="cmdPosition stream source on real hardware.")
    group.add_argument(
        "--position-stream-min-speed", type=float, default=0.18,
        help="Minimum target-directed speed in hybrid/direct cmdPosition mode.")
    group.add_argument(
        "--position-stream-arrival-radius", type=float, default=0.06,
        help="Stop target-directed streaming inside this target distance.")
    group.add_argument(
        "--position-stream-stop-time", type=float, default=1.0,
        help="Slowdown time constant near the target for cmdPosition streaming.")
    group.add_argument(
        "--position-stream-obstacle-clearance", type=float, default=0.25,
        help="Use target-directed fallback only when obstacle clearance exceeds this value.")
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


def command_position_state(crazyflies, positions: np.ndarray) -> None:
    for cf, position in zip(crazyflies, positions):
        if not hasattr(cf, "cmdPosition"):
            raise AttributeError(
                "Crazyflie object {} has no cmdPosition".format(cf))
        cf.cmdPosition(np.asarray(position, dtype=float).tolist(), yaw=0.0)


def command_velocity_state(crazyflies, velocities: np.ndarray) -> None:
    for cf, velocity in zip(crazyflies, velocities):
        if not hasattr(cf, "cmdVelocityWorld"):
            raise AttributeError(
                "Crazyflie object {} has no cmdVelocityWorld".format(cf))
        cf.cmdVelocityWorld(np.asarray(velocity, dtype=float), yawRate=0.0)


def min_obstacle_clearance(config: ControllerConfig, position: np.ndarray,
                           obstacle_positions: np.ndarray) -> float:
    if obstacle_positions is None or len(obstacle_positions) == 0:
        return float("inf")
    clearances = []
    count = min(len(config.obstacles), len(obstacle_positions))
    if count <= 0:
        return float("inf")
    for idx in range(count):
        obstacle_position = obstacle_positions[idx]
        obstacle = config.obstacles[idx]
        inflated_radius = obstacle.radius + config.r_u + config.b_o
        center_distance = float(
            np.linalg.norm(position[:2] - obstacle_position[:2]))
        clearances.append(center_distance - inflated_radius)
    return min(clearances)


def target_direct_velocity(config: ControllerConfig, position: np.ndarray,
                           min_speed: float, arrival_radius: float,
                           stop_time: float) -> np.ndarray:
    error = config.target - position
    distance = float(np.linalg.norm(error))
    if distance <= max(arrival_radius, 1e-6):
        return np.zeros(3)
    speed = min(float(config.velocity_limit), max(float(min_speed), 0.0))
    speed = min(speed, distance / max(float(stop_time), 1e-3))
    return error / distance * speed


def position_stream_velocities(config: ControllerConfig, args: argparse.Namespace,
                               command_positions: np.ndarray,
                               controller_velocities: np.ndarray,
                               obstacle_positions: np.ndarray) -> np.ndarray:
    stream_velocities = controller_velocities.copy()
    mode = args.position_stream_mode
    if mode == "controller":
        return stream_velocities
    for idx in range(config.n):
        clearance = min_obstacle_clearance(
            config, command_positions[idx], obstacle_positions)
        target_error = config.target - command_positions[idx]
        target_distance = float(np.linalg.norm(target_error))
        controller_speed = float(np.linalg.norm(stream_velocities[idx]))
        use_direct = mode == "direct"
        if mode == "hybrid":
            use_direct = (
                clearance > float(args.position_stream_obstacle_clearance) and
                target_distance > float(args.position_stream_arrival_radius) and
                controller_speed < float(args.position_stream_min_speed))
        if use_direct:
            stream_velocities[idx] = target_direct_velocity(
                config, command_positions[idx],
                args.position_stream_min_speed,
                args.position_stream_arrival_radius,
                args.position_stream_stop_time)
    return stream_velocities


def safe_land_compatible(all_cfs, crazyflies, time_helper, landing_height,
                         landing_duration) -> None:
    for cf in crazyflies:
        if hasattr(cf, "notifySetpointsStop"):
            cf.notifySetpointsStop(100)
    time_helper.sleep(0.10)
    try:
        if hasattr(all_cfs, "land"):
            all_cfs.land(
                targetHeight=float(landing_height),
                duration=float(landing_duration))
        else:
            for cf in crazyflies:
                cf.land(
                    targetHeight=float(landing_height),
                    duration=float(landing_duration))
        time_helper.sleep(float(landing_duration) + 0.50)
    except Exception as error:
        print("landing sequence failed: {}".format(error), file=sys.stderr)
        for cf in crazyflies:
            if hasattr(cf, "cmdStop"):
                cf.cmdStop()


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

    simulation = "--sim" in sys.argv[1:]
    if simulation and hasattr(time_helper, "dt"):
        simulator_dt = float(time_helper.dt)
        if not math.isclose(simulator_dt, config.dt, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "run with --dt {:.5f}; got {:.5f}".format(config.dt, simulator_dt))
    command_mode_count = sum([
        bool(args.position_command),
        bool(args.velocity_command),
        bool(args.full_state_command),
    ])
    if command_mode_count > 1:
        raise RuntimeError(
            "choose only one of --position-command, --velocity-command, "
            "or --full-state-command")
    use_velocity_command = args.velocity_command or (
        not simulation and not args.position_command and
        not args.full_state_command)
    use_position_command = args.position_command
    if use_velocity_command:
        print("command mode: cmdVelocityWorld")
    elif use_position_command:
        print("command mode: cmdPosition")
    else:
        print("command mode: full-state")

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
    next_debug_time = 0.0
    position_command = positions.copy()
    position_command[:, 2] = config.target[2]

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
                    positions, velocities, elapsed, dt=config.dt,
                    obstacle_positions=obstacle_positions,
                    obstacle_velocities=obstacle_velocities)
                apply_flight_guards(config, positions, diagnostic, guard)
                controller_controls = diagnostic["controls"]
                controller_next_velocity = controller.saturate_rows(
                    velocities + config.dt * controller_controls,
                    config.velocity_limit)
                controls = controller_controls
                next_velocity = controller_next_velocity
                if args.force_forward_test:
                    next_velocity = np.zeros_like(positions)
                    next_velocity[:, 0] = float(args.force_forward_speed)
                    next_velocity = controller.saturate_rows(
                        next_velocity, config.velocity_limit)
                    controls = np.zeros_like(positions)
                elif use_position_command or use_velocity_command:
                    stream_reference_positions = (
                        position_command if use_position_command else positions)
                    next_velocity = position_stream_velocities(
                        config, args, stream_reference_positions,
                        controller_next_velocity, obstacle_positions)
                    next_velocity = controller.saturate_rows(
                        next_velocity, config.velocity_limit)
                next_position = positions + config.dt * next_velocity
                sent_position = next_position
                clearance_reference_position = (
                    position_command[0] if use_position_command else positions[0])
                command_clearance = min_obstacle_clearance(
                    config, clearance_reference_position, obstacle_positions)
                if use_position_command:
                    command_dt = min(max(measured_dt, config.dt), 0.10)
                    position_command = position_command + command_dt * next_velocity
                    target_step = config.target - position_command[0]
                    if (np.linalg.norm(target_step) <
                            float(args.position_stream_arrival_radius)):
                        position_command[0] = config.target.copy()
                    lead = position_command - positions
                    lead_norms = np.linalg.norm(lead, axis=1)
                    max_lead = max(float(args.position_command_lookahead), 1e-3)
                    for idx, lead_norm in enumerate(lead_norms):
                        if lead_norm > max_lead:
                            position_command[idx] = (
                                positions[idx] + lead[idx] / lead_norm * max_lead)
                    sent_position = position_command.copy()
                if args.debug_control and elapsed >= next_debug_time:
                    distance_to_target = float(
                        np.linalg.norm(config.target - positions[0]))
                    print(
                        "DEBUG t={:.2f}s pos={} target={} dist={:.3f} "
                        "vel={} ctrl={} ctrl_next_vel={} cmd_vel={} raw_pos={} "
                        "sent_pos={} clearance={:.3f} mode={} "
                        "force_forward={} position_cmd={} velocity_cmd={}".format(
                            elapsed,
                            np.round(positions[0], 4).tolist(),
                            np.round(config.target, 4).tolist(),
                            distance_to_target,
                            np.round(velocities[0], 4).tolist(),
                            np.round(controller_controls[0], 4).tolist(),
                            np.round(controller_next_velocity[0], 4).tolist(),
                            np.round(next_velocity[0], 4).tolist(),
                            np.round(next_position[0], 4).tolist(),
                            np.round(sent_position[0], 4).tolist(),
                            command_clearance,
                            args.position_stream_mode,
                            bool(args.force_forward_test),
                            bool(use_position_command),
                            bool(use_velocity_command)))
                    next_debug_time = elapsed + max(
                        float(args.debug_control_interval), config.dt)
                if use_velocity_command:
                    command_velocity_state(crazyflies, next_velocity)
                elif use_position_command:
                    command_position_state(crazyflies, sent_position)
                else:
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

    if use_velocity_command:
        try:
            command_velocity_state(crazyflies, np.zeros_like(positions))
            time_helper.sleep(0.10)
        except Exception as error:
            print("velocity stop failed: {}".format(error), file=sys.stderr)

    safe_land_compatible(all_cfs, all_crazyflies, time_helper,
                         args.landing_height, args.landing_duration)
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
