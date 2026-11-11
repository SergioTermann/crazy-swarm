#!/usr/bin/env python3
"""Run the paper controller through the original Crazyswarm Python API.

The host program computes translational position, velocity, and acceleration
only.  The required yaw and body-rate fields of ``cmdFullState`` are fixed to
zero and are not part of the paper controller.
"""

from __future__ import division, print_function

import argparse
import copy
import csv
import datetime
import math
import os
import select
import sys
import termios
import tty
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from paper_controller import (
    ControllerConfig,
    HRAAPFController,
    controller_initial_positions,
    load_yaml,
    minimum_inter_uav_margin,
)


SCRIPT_DIR = "/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch"
CRAZYSWARM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if CRAZYSWARM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CRAZYSWARM_SCRIPTS_DIR)


def embedded_lab_config() -> Dict[str, Any]:
    """Default 3+1 hardware experiment parameters.

    The Crazyswarm device YAML is still external because chooser.py owns the
    real radio IDs, channels, and initialPosition entries.
    """
    return copy.deepcopy({
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
            "k_i": 6.00,
            "k_o": 0.40,
            "k_t": 0.25,
        },
        "avoidance": {
            "uav_radius": 0.05,
            "obstacle_buffer": 0.08,
            "inter_uav_buffer": 0.06,
            "obstacle_influence_radius": 0.30,
            "safe_clearance": 0.18,
            "prediction_horizon": 2.0,
            "closing_speed_weight": 0.6,
            "risk_adaptive_gain": 1.25,
            "risk_regularizer": 0.293778,
            "apf_epsilon": 0.08,
            "inter_uav_sensing_radius": 0.26,
            "auxiliary_sensing_radius": 2.0,
        },
        "leader_advantage": {
            "target_weight": 0.45,
            "obstacle_weight": 0.35,
            "continuity_weight": 0.20,
            "target_scale": 1.0,
            "continuity_scale": 1.0,
        },
        "switching": {
            "hysteresis": [0.080, 0.060],
            "persistence": [1.500, 1.000],
            "stability_xi": 0.20,
        },
        "formation": {
            "layers": [[0], [1, 2]],
            "offsets": [
                [0.00, 0.00, 0.00],
                [-0.5, 0.4, 0.00],
                [-0.5, -0.4, 0.00],
            ],
        },
        "mission": {
            "start_anchor": [-1.0, -0.40, 0.00],
            "target": [1.00, 0.40, 0.40],
            "adapt_target_x_to_workspace": False,
            "target_x_margin": 0.20,
            "target_velocity": [0.00, 0.00, 0.00],
        },
        "crazyswarm": {
            "uav_ids": [3, 1, 4],
            "takeoff_height": 0.40,
        },
        "state_estimation": {
            "velocity_filter_alpha": 0.35,
        },
        "flight_guard": {
            "start_position_tolerance": 0.3,
            "minimum_inter_uav_distance": 0.08,
            "minimum_obstacle_clearance": 0.08,
            "obstacle_boundary_margin": 0.25,
            "inter_uav_velocity_guard_distance": 0.24,
            "inter_uav_separation_speed": 0.18,
            "workspace_min": [-1.80, -2.00, 0.00],
            "workspace_max": [1.50, 1.00, 1.30],
        },
        "initialization": {
            "use_live_start": True,
        },
        "obstacles": [
            {
                "vehicle_id": 2,
                "center": [0.00, 0.00, 0.40],
                "radius": 0.05,
                "velocity": [-0.015, -0.015, 0.0],
                "height": 0.10,
            },
        ],
    })


def load_controller_raw_config(path: str, default_factory) -> Tuple[Dict[str, Any], str]:
    if path in (None, "", "embedded"):
        return default_factory(), "embedded"
    return load_yaml(path), path


class SafetyStop(RuntimeError):
    pass


class EmergencyLandingRequested(RuntimeError):
    pass


class EmergencyLandingMonitor(object):
    """Non-blocking terminal key monitor for supervised real flights."""

    def __init__(self, enabled: bool = True,
                 keys: Sequence[str] = ("l", "q", "\x1b")) -> None:
        self.enabled = bool(enabled) and sys.stdin.isatty()
        self.keys = set(keys)
        self.fd = None
        self.previous_attrs = None
        self.triggered_key = None
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.previous_attrs = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def close(self) -> None:
        if self.enabled and self.fd is not None and self.previous_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_attrs)
        self.enabled = False

    def poll(self) -> bool:
        if not self.enabled or self.fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return False
        key = os.read(self.fd, 1).decode("utf-8", "ignore")
        if key.lower() in self.keys or key in self.keys:
            self.triggered_key = "Esc" if key == "\x1b" else key
            return True
        return False


class VelocityEstimator(object):
    """Finite-difference velocity estimate for physical Crazyswarm objects."""

    def __init__(self, positions: np.ndarray, alpha: float) -> None:
        self.previous_positions = np.asarray(positions, dtype=float).copy()
        self.filtered_velocity = np.zeros_like(self.previous_positions)
        self.alpha = float(alpha)

    def update(self, positions: np.ndarray, dt: float) -> np.ndarray:
        raw = (np.asarray(positions) - self.previous_positions) / max(float(dt), 1e-6)
        self.filtered_velocity = (
            self.alpha * raw + (1.0 - self.alpha) * self.filtered_velocity)
        self.previous_positions = np.asarray(positions, dtype=float).copy()
        return self.filtered_velocity.copy()


def build_parser() -> argparse.ArgumentParser:
    # Crazyswarm adds --sim/--vis/--dt and parses the same parent parser.
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("HRA-APF paper controller")
    group.add_argument(
        "--controller-config",
        default="embedded",
        help="Controller parameters. Default: embedded. Pass a YAML path to override.")
    group.add_argument(
        "--crazyflies-config",
        default=os.path.join(SCRIPT_DIR, "crazyflies.yaml"),
        help="Crazyswarm YAML for three formation and one obstacle UAV.")
    group.add_argument(
        "--log-dir",
        default=os.path.join(SCRIPT_DIR, "logs"),
        help="Directory for flight CSV logs.")
    group.add_argument(
        "--dry-run", action="store_true",
        help="Validate parameters and print the controller setup without starting Crazyswarm.")
    group.add_argument(
        "--confirm-real-flight", action="store_true",
        help="Required acknowledgement when --sim is not present.")
    group.add_argument(
        "--disable-emergency-key", action="store_true",
        help="Disable terminal emergency landing keys l/q/Esc.")
    group.add_argument(
        "--no-takeoff", action="store_true",
        help="Do not issue takeoff; intended only for an already airborne, supervised swarm.")
    group.add_argument(
        "--paper-vis", action="store_true",
        help="Show the complete paper scene: obstacle UAVs, safety boundaries, trails, and hierarchy.")
    group.add_argument(
        "--paper-vis-rate", type=float, default=10.0,
        help="Display refresh rate of the complete paper scene.")
    group.add_argument(
        "--paper-vis-trail-seconds", type=float, default=10.0,
        help="Seconds of recent UAV trajectories retained in the paper scene.")
    group.add_argument(
        "--paper-vis-no-realtime", action="store_true",
        help="Do not pace the paper visualization to wall-clock time.")
    group.add_argument(
        "--paper-vis-no-hold", action="store_true",
        help="Close instead of holding the final paper-scene frame.")
    group.add_argument("--takeoff-duration", type=float, default=3.0)
    group.add_argument("--landing-duration", type=float, default=3.0)
    group.add_argument("--landing-height", type=float, default=0.04)
    group.add_argument(
        "--debug-control", action="store_true",
        help="Print controller output and commanded velocity during flight.")
    group.add_argument(
        "--debug-control-interval", type=float, default=0.50,
        help="Seconds between --debug-control printouts.")
    group.add_argument(
        "--velocity-command", action="store_true",
        help="Use cf.cmdVelocityWorld instead of full-state commands.")
    group.add_argument(
        "--position-command", action="store_true",
        help="Use cf.cmdPosition instead of full-state commands.")
    group.add_argument(
        "--full-state-command", action="store_true",
        help="Force the original full-state command path on real hardware.")
    group.add_argument(
        "--force-forward-test", action="store_true",
        help="Bypass the controller and command a small forward velocity.")
    group.add_argument(
        "--force-forward-speed", type=float, default=0.15,
        help="Forward x velocity used by --force-forward-test, in m/s.")
    group.add_argument(
        "--minimum-command-speed", type=float, default=0.08,
        help="Minimum speed used when controller velocity is very small but control direction is clear.")
    group.add_argument(
        "--minimum-command-control-norm", type=float, default=0.05,
        help="Control-norm threshold for applying --minimum-command-speed.")
    return parser


def load_crazyflies_yaml(path: str) -> Tuple[str, List[int]]:
    with open(path, "r") as stream:
        text = stream.read()
    parsed = yaml.safe_load(text)
    entries = parsed.get("crazyflies", []) if isinstance(parsed, dict) else []
    ids = [int(item["id"]) for item in entries]
    if not ids:
        raise ValueError("crazyflies configuration is empty")
    return text, ids


def collect_positions(crazyflies: Sequence[Any]) -> np.ndarray:
    positions = np.asarray([np.asarray(cf.position(), dtype=float) for cf in crazyflies])
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise SafetyStop("invalid position feedback")
    return positions


def collect_velocities_if_available(crazyflies: Sequence[Any]) -> Any:
    if not all(hasattr(cf, "velocity") for cf in crazyflies):
        return None
    velocities = np.asarray([np.asarray(cf.velocity(), dtype=float) for cf in crazyflies])
    if velocities.ndim != 2 or velocities.shape[1] != 3 or not np.all(np.isfinite(velocities)):
        return None
    return velocities


def reorder_crazyflies(crazyflies: Sequence[Any], expected_ids: Sequence[int]) -> List[Any]:
    by_id = {int(cf.id): cf for cf in crazyflies}
    missing = [vehicle_id for vehicle_id in expected_ids if vehicle_id not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError("Crazyswarm IDs do not match controller: missing={}, extra={}".format(
            missing, extra))
    return [by_id[vehicle_id] for vehicle_id in expected_ids]


def validate_start_positions(config: ControllerConfig, positions: np.ndarray,
                             guard: Dict[str, Any], after_takeoff: bool = False) -> None:
    expected = config.eta + config.start_anchor[None, :]
    if after_takeoff:
        expected = controller_initial_positions(config)
    tolerance = float(guard["start_position_tolerance"])
    errors = np.linalg.norm(positions - expected, axis=1)
    if float(np.max(errors)) > tolerance:
        raise SafetyStop(
            "initial formation does not match start_anchor + offsets; max error "
            "{:.3f} m exceeds {:.3f} m".format(float(np.max(errors)), tolerance))


def validate_obstacle_uav_positions(config: ControllerConfig, positions: np.ndarray,
                                    guard: Dict[str, Any]) -> None:
    expected = np.asarray(
        [obstacle.center0 for obstacle in config.obstacles], dtype=float)
    if positions.shape != expected.shape:
        raise SafetyStop("invalid obstacle-UAV position feedback")
    tolerance = float(guard["start_position_tolerance"])
    errors = np.linalg.norm(positions - expected, axis=1)
    if float(np.max(errors)) > tolerance:
        raise SafetyStop(
            "obstacle UAVs do not match configured initial centers; max error "
            "{:.3f} m exceeds {:.3f} m".format(float(np.max(errors)), tolerance))


def apply_live_initial_state(config: ControllerConfig, raw_config: Dict[str, Any],
                             formation_positions: np.ndarray,
                             obstacle_positions: np.ndarray = None) -> bool:
    initialization = raw_config.get("initialization", {})
    if not bool(initialization.get("use_live_start", False)):
        return False

    formation_positions = np.asarray(formation_positions, dtype=float)
    if formation_positions.shape != (config.n, 3):
        raise SafetyStop("invalid live formation positions")
    anchor = config.start_anchor.copy()
    anchor[:2] = np.mean(formation_positions[:, :2] - config.eta[:, :2], axis=0)
    config.start_anchor = anchor

    if obstacle_positions is not None:
        obstacle_positions = np.asarray(obstacle_positions, dtype=float)
        if obstacle_positions.shape != (len(config.obstacles), 3):
            raise SafetyStop("invalid live obstacle positions")
        for index, obstacle in enumerate(config.obstacles):
            obstacle.center0 = obstacle_positions[index].copy()
    return True


def apply_flight_guards(config: ControllerConfig, positions: np.ndarray,
                        diagnostic: Dict[str, Any], guard: Dict[str, Any]) -> None:
    workspace_min = np.asarray(guard["workspace_min"], dtype=float)
    workspace_max = np.asarray(guard["workspace_max"], dtype=float)
    if np.any(positions < workspace_min[None, :]) or np.any(positions > workspace_max[None, :]):
        raise SafetyStop("workspace boundary reached")

    inter_distance = float(guard["minimum_inter_uav_distance"])
    minimum_distance = minimum_inter_uav_margin(positions, 0.0)
    if minimum_distance < inter_distance:
        raise SafetyStop("inter-UAV distance {:.3f} m below guard {:.3f} m".format(
            minimum_distance, inter_distance))

    clearances = diagnostic["current_clearance"]
    if clearances.size:
        minimum_clearance = float(np.min(clearances))
        clearance_guard = float(guard["minimum_obstacle_clearance"])
        if minimum_clearance < clearance_guard:
            raise SafetyStop("obstacle clearance {:.3f} m below guard {:.3f} m".format(
                minimum_clearance, clearance_guard))


def command_translational_state(crazyflies: Sequence[Any], positions: np.ndarray,
                                velocities: np.ndarray, accelerations: np.ndarray) -> None:
    # Crazyswarm requires yaw and omega fields in cmdFullState.  They are fixed
    # interface placeholders; no attitude/yaw law is implemented here.
    zero_omega = np.zeros(3, dtype=float)
    for index, cf in enumerate(crazyflies):
        cf.cmdFullState(
            positions[index], velocities[index], accelerations[index],
            yaw=0.0, omega=zero_omega)


def command_position_state(crazyflies: Sequence[Any], positions: np.ndarray) -> None:
    for index, cf in enumerate(crazyflies):
        if not hasattr(cf, "cmdPosition"):
            raise AttributeError(
                "Crazyflie object {} has no cmdPosition".format(cf))
        cf.cmdPosition(np.asarray(positions[index], dtype=float).tolist(), yaw=0.0)


def command_velocity_state(crazyflies: Sequence[Any], velocities: np.ndarray) -> None:
    for index, cf in enumerate(crazyflies):
        if not hasattr(cf, "cmdVelocityWorld"):
            raise AttributeError(
                "Crazyflie object {} has no cmdVelocityWorld".format(cf))
        cf.cmdVelocityWorld(np.asarray(velocities[index], dtype=float), yawRate=0.0)


def command_state(crazyflies: Sequence[Any], positions: np.ndarray,
                  velocities: np.ndarray, accelerations: np.ndarray,
                  command_mode: str) -> None:
    if command_mode == "velocity":
        command_velocity_state(crazyflies, velocities)
    elif command_mode == "position":
        command_position_state(crazyflies, positions)
    else:
        command_translational_state(crazyflies, positions, velocities, accelerations)


def enforce_minimum_command_speed(velocities: np.ndarray,
                                  controls: np.ndarray,
                                  min_speed: float,
                                  min_control_norm: float) -> np.ndarray:
    adjusted = np.asarray(velocities, dtype=float).copy()
    min_speed = max(float(min_speed), 0.0)
    min_control_norm = max(float(min_control_norm), 0.0)
    if min_speed <= 0.0:
        return adjusted
    for index in range(adjusted.shape[0]):
        speed = float(np.linalg.norm(adjusted[index]))
        control_norm = float(np.linalg.norm(controls[index]))
        if 1e-9 < speed < min_speed and control_norm > min_control_norm:
            adjusted[index] = adjusted[index] / speed * min_speed
        elif speed <= 1e-9 and control_norm > min_control_norm:
            adjusted[index] = controls[index] / control_norm * min_speed
    return adjusted


def limit_velocity_rows(velocities: np.ndarray, velocity_limit: float) -> np.ndarray:
    limited = np.asarray(velocities, dtype=float).copy()
    velocity_limit = max(float(velocity_limit), 0.0)
    if velocity_limit <= 0.0:
        return np.zeros_like(limited)
    norms = np.linalg.norm(limited, axis=1)
    for index, norm in enumerate(norms):
        if norm > velocity_limit:
            limited[index] *= velocity_limit / norm
    return limited


def enforce_inter_uav_velocity_guard(positions: np.ndarray,
                                     velocities: np.ndarray,
                                     guard: Dict[str, Any],
                                     dt: float,
                                     velocity_limit: float) -> np.ndarray:
    adjusted = np.asarray(velocities, dtype=float).copy()
    positions = np.asarray(positions, dtype=float)
    hard_distance = float(guard.get("minimum_inter_uav_distance", 0.12))
    soft_distance = float(guard.get(
        "inter_uav_velocity_guard_distance",
        max(hard_distance + 0.06, hard_distance * 1.5)))
    separation_speed = float(guard.get("inter_uav_separation_speed", 0.12))
    if soft_distance <= hard_distance or separation_speed <= 0.0:
        return limit_velocity_rows(adjusted, velocity_limit)

    dt = max(float(dt), 1e-6)
    for _ in range(2):
        for first in range(positions.shape[0] - 1):
            for second in range(first + 1, positions.shape[0]):
                delta_xy = positions[first, :2] - positions[second, :2]
                distance = float(np.linalg.norm(delta_xy))
                if distance <= 1e-6:
                    unit_xy = np.array([1.0, 0.0], dtype=float)
                else:
                    unit_xy = delta_xy / distance
                relative_velocity_xy = adjusted[first, :2] - adjusted[second, :2]
                distance_rate = float(np.dot(unit_xy, relative_velocity_xy))
                predicted_delta_xy = (
                    delta_xy + dt * relative_velocity_xy)
                predicted_distance = float(np.linalg.norm(predicted_delta_xy))
                if distance >= soft_distance and predicted_distance >= hard_distance:
                    continue
                if distance <= hard_distance or predicted_distance < hard_distance:
                    desired_rate = separation_speed
                else:
                    fraction = (soft_distance - distance) / (
                        soft_distance - hard_distance)
                    desired_rate = separation_speed * max(0.0, min(1.0, fraction))
                correction_rate = desired_rate - distance_rate
                if correction_rate <= 0.0:
                    continue
                correction_xy = 0.5 * correction_rate * unit_xy
                adjusted[first, :2] += correction_xy
                adjusted[second, :2] -= correction_xy
    return limit_velocity_rows(adjusted, velocity_limit)


def obstacle_boundary_limited_state(config: ControllerConfig, obstacle: Any,
                                    time_sec: float,
                                    guard: Dict[str, Any] = None
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    position, velocity, _ = obstacle.state(time_sec)
    if guard is None:
        return position, velocity

    margin = float(guard.get("obstacle_boundary_margin", 0.25))
    if margin <= 0.0:
        return position, velocity
    workspace_min = np.asarray(guard["workspace_min"], dtype=float)
    workspace_max = np.asarray(guard["workspace_max"], dtype=float)
    lower_xy = workspace_min[:2] + margin
    upper_xy = workspace_max[:2] - margin
    if np.any(lower_xy >= upper_xy):
        raise SafetyStop("invalid obstacle boundary margin")

    start_xy = obstacle.center0[:2]
    velocity_xy = obstacle.velocity[:2]
    if np.any(start_xy <= lower_xy) or np.any(start_xy >= upper_xy):
        return obstacle.center0.copy(), np.zeros(3, dtype=float)

    stop_time = math.inf
    for axis in range(2):
        if velocity_xy[axis] > 0.0:
            candidate = (upper_xy[axis] - start_xy[axis]) / velocity_xy[axis]
        elif velocity_xy[axis] < 0.0:
            candidate = (lower_xy[axis] - start_xy[axis]) / velocity_xy[axis]
        else:
            continue
        if candidate >= 0.0:
            stop_time = min(stop_time, float(candidate))

    if time_sec < stop_time:
        return position, velocity
    hover_position = obstacle.center0 + stop_time * obstacle.velocity
    return hover_position, np.zeros(3, dtype=float)


def command_obstacle_uavs(config: ControllerConfig,
                          obstacle_crazyflies: Sequence[Any],
                          time_sec: float,
                          guard: Dict[str, Any] = None,
                          command_mode: str = "full_state"
                          ) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((len(config.obstacles), 3), dtype=float)
    velocities = np.zeros_like(positions)
    for index, obstacle in enumerate(config.obstacles):
        positions[index], velocities[index] = obstacle_boundary_limited_state(
            config, obstacle, time_sec, guard)
    accelerations = np.zeros_like(positions)
    command_state(
        obstacle_crazyflies, positions, velocities, accelerations, command_mode)
    return positions, velocities


def set_simulation_colors(formation_crazyflies: Sequence[Any],
                          obstacle_crazyflies: Sequence[Any]) -> None:
    """Make the two UAV groups visually distinct in the native VisPy view."""
    for cf in formation_crazyflies:
        if hasattr(cf, "setLEDColor"):
            cf.setLEDColor(0.20, 0.45, 1.00)
    for cf in obstacle_crazyflies:
        if hasattr(cf, "setLEDColor"):
            cf.setLEDColor(1.00, 0.25, 0.05)


def safe_land(all_cfs: Any, crazyflies: Sequence[Any], time_helper: Any,
              landing_height: float, landing_duration: float) -> None:
    try:
        hold = collect_positions(crazyflies)
        end_time = time_helper.time() + 0.30
        while time_helper.time() < end_time and not time_helper.isShutdown():
            for index, cf in enumerate(crazyflies):
                cf.cmdPosition(hold[index], yaw=0.0)
            time_helper.sleepForRate(30.0)
        for cf in crazyflies:
            cf.notifySetpointsStop()
        time_helper.sleep(0.15)
        all_cfs.land(targetHeight=landing_height, duration=landing_duration)
        time_helper.sleep(landing_duration + 0.5)
        for cf in crazyflies:
            if hasattr(cf, "stop"):
                cf.stop()
            elif hasattr(cf, "cmdStop"):
                cf.cmdStop()
    except Exception as error:
        print("landing sequence failed: {}".format(error), file=sys.stderr)
        for cf in crazyflies:
            try:
                cf.cmdStop()
            except Exception:
                pass


def log_header(vehicle_count: int, obstacle_count: int = 0,
               layer_count: int = 2) -> List[str]:
    header = ["time", "min_obstacle_clearance", "min_inter_uav_margin", "switch_count"]
    for index in range(vehicle_count):
        header.extend([
            "uav{}_x".format(index + 1), "uav{}_y".format(index + 1),
            "uav{}_z".format(index + 1), "uav{}_vx".format(index + 1),
            "uav{}_vy".format(index + 1), "uav{}_vz".format(index + 1),
            "uav{}_ux".format(index + 1), "uav{}_uy".format(index + 1),
            "uav{}_uz".format(index + 1), "uav{}_risk".format(index + 1),
            "uav{}_advantage".format(index + 1), "uav{}_role".format(index + 1),
        ])
    header.extend(["layer{}_leader".format(index + 1) for index in range(layer_count)])
    for uav_index in range(vehicle_count):
        for obstacle_index in range(obstacle_count):
            header.extend([
                "uav{}_obstacle{}_xy_distance".format(
                    uav_index + 1, obstacle_index + 1),
                "uav{}_obstacle{}_clearance".format(
                    uav_index + 1, obstacle_index + 1),
                "uav{}_obstacle{}_predicted_clearance".format(
                    uav_index + 1, obstacle_index + 1),
            ])
    for index in range(obstacle_count):
        header.extend([
            "obstacle_uav{}_x".format(index + 1),
            "obstacle_uav{}_y".format(index + 1),
            "obstacle_uav{}_z".format(index + 1),
            "obstacle_uav{}_vx".format(index + 1),
            "obstacle_uav{}_vy".format(index + 1),
            "obstacle_uav{}_vz".format(index + 1),
        ])
    return header


def log_row(time_sec: float, config: ControllerConfig, controller: HRAAPFController,
            positions: np.ndarray, velocities: np.ndarray,
            diagnostic: Dict[str, Any], obstacle_positions: np.ndarray = None,
            obstacle_velocities: np.ndarray = None) -> List[float]:
    current_clearance = diagnostic["current_clearance"]
    minimum_obstacle = float(np.min(current_clearance)) if current_clearance.size else math.inf
    minimum_inter = minimum_inter_uav_margin(positions, config.d_f)
    row = [time_sec, minimum_obstacle, minimum_inter, float(controller.switch_count)]
    controls = diagnostic["controls"]
    for index in range(config.n):
        row.extend(list(positions[index]))
        row.extend(list(velocities[index]))
        row.extend(list(controls[index]))
        row.extend([
            float(diagnostic["risk"][index]),
            float(diagnostic["scores"][index]),
            float(diagnostic["roles"][index] + 1),
        ])
    row.extend([float(value + 1) for value in diagnostic["leaders"]])
    predicted_clearance = diagnostic.get("predicted_clearance")
    for uav_index in range(config.n):
        for obstacle_index, obstacle in enumerate(config.obstacles):
            inflated_radius = obstacle.radius + config.r_u + config.b_o
            clearance = current_clearance[uav_index, obstacle_index]
            predicted = (predicted_clearance[uav_index, obstacle_index]
                         if predicted_clearance is not None else math.nan)
            row.extend([
                float(clearance + inflated_radius),
                float(clearance),
                float(predicted),
            ])
    if obstacle_positions is not None and obstacle_velocities is not None:
        for index in range(len(config.obstacles)):
            row.extend(list(obstacle_positions[index]))
            row.extend(list(obstacle_velocities[index]))
    return row


def write_buffered_logs(log_path: str, switch_path: str, config: ControllerConfig,
                        rows: Sequence[Sequence[float]],
                        switch_rows: Sequence[Sequence[float]]) -> None:
    with open(log_path, "w", newline="") as log_stream:
        writer = csv.writer(log_stream)
        writer.writerow(log_header(
            config.n, len(config.obstacles), len(config.layers)))
        writer.writerows(rows)
    with open(switch_path, "w", newline="") as switch_stream:
        switch_writer = csv.writer(switch_stream)
        switch_writer.writerow([
            "time", "layer", "old_leader", "new_leader", "advantage_gain",
            "old_risk", "new_risk"])
        switch_writer.writerows(switch_rows)


def print_configuration(config: ControllerConfig) -> None:
    theorem = config.theorem_report()
    pair_distances = [
        float(np.linalg.norm(config.eta[first] - config.eta[second]))
        for first in range(config.n - 1)
        for second in range(first + 1, config.n)]
    nominal_spacing = min(pair_distances) if pair_distances else math.inf
    print("Validated {}-UAV, {}-layer Proposed controller.".format(
        config.n, len(config.layers)))
    print("rate={:.1f} Hz, duration={:.1f} s, target={}".format(
        config.rate_hz, config.duration, config.target.tolist()))
    print("formation UAV IDs={}, obstacle UAV IDs={}".format(
        config.uav_ids, config.obstacle_uav_ids))
    print("layers={}, initial layer leaders={}".format(
        [[int(index + 1) for index in layer] for layer in config.layers],
        [int(layer[0] + 1) for layer in config.layers]))
    print("uav radius={:.3f} m, nominal nearest-neighbor spacing={:.3f} m".format(
        config.r_u, nominal_spacing))
    print("fixed gains: kp={:.6f}, kv={:.6f}, kF={:.6f}, lambdaE={:.6f}".format(
        config.kp, config.kv, config.k_f, config.lambda_e))
    print("Theorem check: vartheta={:.6f}, alphaT={:.6f}, required dwell={:.3f} s "
          "< min dwell={:.3f} s".format(
              theorem["vartheta"], theorem["alpha_t"], theorem["required_dwell"],
              theorem["minimum_dwell"]))


def main() -> int:
    parser = build_parser()
    own_args, _ = parser.parse_known_args()
    raw_config, config_source = load_controller_raw_config(
        own_args.controller_config, embedded_lab_config)
    config = ControllerConfig(raw_config)
    guard = raw_config["flight_guard"]
    config.raw_flight_guard = guard
    crazyflies_yaml, file_ids = load_crazyflies_yaml(own_args.crazyflies_config)
    expected_ids = config.uav_ids + config.obstacle_uav_ids
    if file_ids != expected_ids:
        raise ValueError(
            "crazyflies YAML IDs {} differ from formation+obstacle IDs {}".format(
                file_ids, expected_ids))
    print_configuration(config)
    print("controller parameters={}".format(config_source))
    if own_args.dry_run:
        return 0

    simulation = "--sim" in sys.argv[1:]
    if not simulation and not own_args.confirm_real_flight:
        raise RuntimeError(
            "real flight is locked; first validate with --sim, then add "
            "--confirm-real-flight after adapting target, obstacles, workspace, and radio channels")

    from pycrazyswarm import Crazyswarm

    swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml, parent_parser=parser)
    time_helper = swarm.timeHelper
    all_cfs = swarm.allcfs
    all_crazyflies = reorder_crazyflies(all_cfs.crazyflies, expected_ids)
    crazyflies = all_crazyflies[:config.n]
    obstacle_crazyflies = all_crazyflies[config.n:]
    if simulation:
        set_simulation_colors(crazyflies, obstacle_crazyflies)

    if simulation and hasattr(time_helper, "dt"):
        simulator_dt = float(time_helper.dt)
        if not math.isclose(simulator_dt, config.dt, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "for equation-level replay, run Crazyswarm with --dt {:.5f}; got {:.5f}"
                .format(config.dt, simulator_dt))
    command_mode_count = sum([
        bool(own_args.velocity_command),
        bool(own_args.position_command),
        bool(own_args.full_state_command),
    ])
    if command_mode_count > 1:
        raise RuntimeError(
            "choose only one of --velocity-command, --position-command, "
            "or --full-state-command")
    if own_args.velocity_command or (
            not simulation and not own_args.position_command and
            not own_args.full_state_command):
        command_mode = "velocity"
    elif own_args.position_command:
        command_mode = "position"
    else:
        command_mode = "full_state"
    if command_mode == "velocity":
        print("command mode: cmdVelocityWorld")
    elif command_mode == "position":
        print("command mode: cmdPosition")
    else:
        print("command mode: full-state")

    if not own_args.no_takeoff:
        for cf in crazyflies:
            cf.takeoff(targetHeight=float(config.takeoff_height),
                       duration=float(own_args.takeoff_duration))
        for cf, obstacle in zip(obstacle_crazyflies, config.obstacles):
            cf.takeoff(targetHeight=float(obstacle.center0[2]),
                       duration=float(own_args.takeoff_duration))
        time_helper.sleep(float(own_args.takeoff_duration) + 0.5)

    positions = collect_positions(crazyflies)
    obstacle_positions = (collect_positions(obstacle_crazyflies)
                          if obstacle_crazyflies else None)
    if apply_live_initial_state(config, raw_config, positions, obstacle_positions):
        print("live start applied: start_anchor={}, obstacle_centers={}".format(
            config.start_anchor.tolist(),
            [obstacle.center0.tolist() for obstacle in config.obstacles]))
    validate_start_positions(
        config, positions, guard, after_takeoff=not own_args.no_takeoff)
    if obstacle_crazyflies:
        validate_obstacle_uav_positions(config, obstacle_positions, guard)
    estimator = VelocityEstimator(positions, config.velocity_filter_alpha)
    controller = HRAAPFController(config)
    scene_visualizer = None
    if own_args.paper_vis:
        from paper_scene_visualizer import PaperSceneVisualizer
        scene_visualizer = PaperSceneVisualizer(
            config,
            display_rate_hz=float(own_args.paper_vis_rate),
            realtime=not own_args.paper_vis_no_realtime,
            trail_history_sec=float(own_args.paper_vis_trail_seconds))

    os.makedirs(own_args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(own_args.log_dir, "hra_apf_{}.csv".format(timestamp))
    switch_path = os.path.join(own_args.log_dir, "hra_apf_switches_{}.csv".format(timestamp))

    print("Starting translational paper controller; buffered log: {}".format(log_path))
    emergency = EmergencyLandingMonitor(
        enabled=not own_args.disable_emergency_key)
    if emergency.enabled:
        print("Emergency landing key active: press l, q, or Esc to land all UAVs.")
    else:
        print("Emergency landing key inactive; Ctrl+C still lands all UAVs.")
    failure = None
    start_time = time_helper.time()
    previous_time = start_time
    buffered_rows = []  # type: List[List[float]]
    buffered_switch_rows = []  # type: List[List[float]]
    obstacle_hovered = np.zeros(len(config.obstacles), dtype=bool)
    next_debug_time = 0.0
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
                    # Keep the estimator synchronized in case the backend changes.
                    estimator.previous_positions = positions.copy()
                    estimator.filtered_velocity = velocities.copy()

                if obstacle_crazyflies:
                    obstacle_positions = collect_positions(obstacle_crazyflies)
                    obstacle_velocities = collect_velocities_if_available(
                        obstacle_crazyflies)
                    _, reference_obstacle_velocities = command_obstacle_uavs(
                        config, obstacle_crazyflies, elapsed, guard,
                        command_mode=command_mode)
                    for obstacle_index, reference_velocity in enumerate(
                            reference_obstacle_velocities):
                        if (not obstacle_hovered[obstacle_index] and
                                np.linalg.norm(config.obstacles[obstacle_index].velocity) > 1e-9 and
                                np.linalg.norm(reference_velocity) < 1e-9):
                            obstacle_hovered[obstacle_index] = True
                            print("t={:.2f}s obstacle UAV {} reached boundary margin; hovering."
                                  .format(
                                      elapsed,
                                      config.obstacle_uav_ids[obstacle_index]))
                    if obstacle_velocities is None:
                        obstacle_velocities = reference_obstacle_velocities
                else:
                    obstacle_positions, obstacle_velocities, _ = (
                        controller.obstacle_state(elapsed))

                diagnostic = controller.step(
                    positions, velocities, elapsed, dt=config.dt,
                    obstacle_positions=obstacle_positions,
                    obstacle_velocities=obstacle_velocities)
                apply_flight_guards(config, positions, diagnostic, guard)
                controls = diagnostic["controls"]
                next_velocity = controller.saturate_rows(
                    velocities + config.dt * controls, config.velocity_limit)
                if own_args.force_forward_test:
                    next_velocity = np.zeros_like(positions)
                    next_velocity[:, 0] = float(own_args.force_forward_speed)
                    controls = np.zeros_like(positions)
                else:
                    if command_mode == "velocity":
                        next_velocity = enforce_minimum_command_speed(
                            next_velocity, controls,
                            own_args.minimum_command_speed,
                            own_args.minimum_command_control_norm)
                        next_velocity = controller.saturate_rows(
                            next_velocity, config.velocity_limit)
                    next_velocity = enforce_inter_uav_velocity_guard(
                        positions, next_velocity, guard, config.dt,
                        config.velocity_limit)
                next_position = positions + config.dt * next_velocity
                if own_args.debug_control and elapsed >= next_debug_time:
                    print(
                        "DEBUG t={:.2f}s mode={} pos1={} target={} "
                        "vel1={} ctrl1={} cmd_vel1={} pos{}={} cmd_vel{}={}".format(
                            elapsed,
                            command_mode,
                            np.round(positions[0], 4).tolist(),
                            np.round(config.target, 4).tolist(),
                            np.round(velocities[0], 4).tolist(),
                            np.round(controls[0], 4).tolist(),
                            np.round(next_velocity[0], 4).tolist(),
                            config.n,
                            np.round(positions[-1], 4).tolist(),
                            config.n,
                            np.round(next_velocity[-1], 4).tolist()))
                    next_debug_time = elapsed + max(
                        float(own_args.debug_control_interval), config.dt)
                command_state(
                    crazyflies, next_position, next_velocity, controls,
                    command_mode)
                if scene_visualizer is not None:
                    scene_visualizer.update(
                        elapsed, positions, diagnostic["leaders"],
                        controller.active_parent_map(), controller.switch_count,
                        obstacle_positions=obstacle_positions)

                buffered_rows.append(log_row(
                    elapsed, config, controller, positions, velocities, diagnostic,
                    obstacle_positions=obstacle_positions,
                    obstacle_velocities=obstacle_velocities))
                for event in diagnostic["switch_events"]:
                    old_index = int(event["old_leader"])
                    new_index = int(event["new_leader"])
                    buffered_switch_rows.append([
                        elapsed, int(event["layer"]) + 1, old_index + 1, new_index + 1,
                        event["advantage_gain"], diagnostic["risk"][old_index],
                        diagnostic["risk"][new_index]])
                    print("t={:.2f}s layer {} leader {} -> {}, gain={:.4f}".format(
                        elapsed, int(event["layer"]) + 1, old_index + 1, new_index + 1,
                        event["advantage_gain"]))
                time_helper.sleepForRate(config.rate_hz)
        finally:
            emergency.close()
    except (KeyboardInterrupt, SafetyStop, Exception) as error:
        # Preserve the traceback for programming errors after attempting to land.
        failure = error

    if command_mode == "velocity":
        try:
            command_velocity_state(crazyflies, np.zeros((len(crazyflies), 3)))
            if obstacle_crazyflies:
                command_velocity_state(
                    obstacle_crazyflies,
                    np.zeros((len(obstacle_crazyflies), 3)))
            time_helper.sleep(0.10)
        except Exception as error:
            print("velocity stop failed: {}".format(error), file=sys.stderr)

    safe_land(all_cfs, all_crazyflies, time_helper, own_args.landing_height,
              own_args.landing_duration)
    write_buffered_logs(
        log_path, switch_path, config, buffered_rows, buffered_switch_rows)
    print("Saved buffered flight log: {}".format(log_path))
    print("Saved buffered switch log: {}".format(switch_path))
    if scene_visualizer is not None:
        scene_visualizer.finish(block=not own_args.paper_vis_no_hold)
    if failure is not None:
        if isinstance(failure, EmergencyLandingRequested):
            print("Emergency landing completed: {}".format(failure))
            return 0
        if isinstance(failure, KeyboardInterrupt):
            print("Flight interrupted; landing completed.")
            return 130
        if isinstance(failure, SafetyStop):
            print("Safety stop: {}".format(failure), file=sys.stderr)
            return 2
        raise failure
    print("Completed. Accepted leader switches: {}".format(controller.switch_count))
    return 0


if __name__ == "__main__":
    sys.exit(main())
