#!/usr/bin/env python3
"""Risk-aware dynamic leader reconfiguration and HRA-APF controller.

This module is a direct Python port of the online ``Proposed`` branch in the
provided MATLAB program.  It contains no PIO, comparison methods, Monte Carlo
evaluation, plotting, or vehicle attitude control.

Internal vehicle and role indices are zero-based.  The configuration file uses
the same convention so that the equations stay readable beside the paper.
"""

from __future__ import division, print_function

import itertools
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


def _array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, result.shape))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains a non-finite value".format(name))
    return result


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not valid:
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError("{} must be a finite {} number".format(name, relation))
    return result


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("controller configuration must be a YAML mapping")
    return data


class Obstacle(object):
    """Obstacle UAV with a known constant horizontal reference velocity."""

    def __init__(self, center: Sequence[float], radius: float,
                 velocity: Sequence[float], height: float,
                 vehicle_id: int = None) -> None:
        center_array = np.asarray(center, dtype=float).reshape(-1)
        velocity_array = np.asarray(velocity, dtype=float).reshape(-1)
        if center_array.size == 2:
            center_array = np.r_[center_array, 1.0]
        if velocity_array.size == 2:
            velocity_array = np.r_[velocity_array, 0.0]
        self.center0 = _array(center_array, (3,), "obstacle.center")
        self.velocity = _array(velocity_array, (3,), "obstacle.velocity")
        self.radius = _positive(radius, "obstacle.radius")
        self.height = _positive(height, "obstacle.height")
        self.vehicle_id = None if vehicle_id is None else int(vehicle_id)
        if self.vehicle_id is not None and self.vehicle_id <= 0:
            raise ValueError("obstacle.vehicle_id must be a positive integer")

    def state(self, time_sec: float) -> Tuple[np.ndarray, np.ndarray, float]:
        return self.center0 + float(time_sec) * self.velocity, self.velocity, self.radius


class ControllerConfig(object):
    """Validated fixed parameters for the paper controller."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        control = raw["control"]
        gains = raw["gains"]
        avoidance = raw["avoidance"]
        advantage = raw["leader_advantage"]
        switching = raw["switching"]
        formation = raw["formation"]
        mission = raw["mission"]
        flight_guard = raw.get("flight_guard", {})

        self.rate_hz = _positive(control["rate_hz"], "control.rate_hz")
        self.dt = 1.0 / self.rate_hz
        self.duration = _positive(control["duration"], "control.duration")
        self.velocity_limit = _positive(control["velocity_limit"], "control.velocity_limit")

        self.kp = _positive(gains["kp"], "gains.kp")
        self.kv = _positive(gains["kv"], "gains.kv")
        self.k_f = _positive(gains["k_f"], "gains.k_f")
        self.lambda_e = _positive(gains["lambda_e"], "gains.lambda_e")
        self.k_i = _positive(gains["k_i"], "gains.k_i")
        self.k_o = _positive(gains["k_o"], "gains.k_o")
        self.k_t = _positive(gains["k_t"], "gains.k_t", allow_zero=True)

        self.r_u = _positive(avoidance["uav_radius"], "avoidance.uav_radius")
        self.b_o = _positive(avoidance["obstacle_buffer"], "avoidance.obstacle_buffer",
                             allow_zero=True)
        self.b_f = _positive(avoidance["inter_uav_buffer"], "avoidance.inter_uav_buffer",
                             allow_zero=True)
        self.d_f = 2.0 * self.r_u + self.b_f
        self.rho_o = _positive(avoidance["obstacle_influence_radius"],
                               "avoidance.obstacle_influence_radius")
        self.d_s = _positive(avoidance["safe_clearance"], "avoidance.safe_clearance",
                             allow_zero=True)
        self.t_p = _positive(avoidance["prediction_horizon"],
                             "avoidance.prediction_horizon", allow_zero=True)
        self.lambda_v = _positive(avoidance["closing_speed_weight"],
                                  "avoidance.closing_speed_weight", allow_zero=True)
        self.alpha_0 = _positive(avoidance["risk_adaptive_gain"],
                                 "avoidance.risk_adaptive_gain")
        self.c_o = _positive(avoidance["risk_regularizer"], "avoidance.risk_regularizer")
        self.apf_epsilon = _positive(avoidance["apf_epsilon"], "avoidance.apf_epsilon")
        self.rho_i = _positive(avoidance["inter_uav_sensing_radius"],
                               "avoidance.inter_uav_sensing_radius")
        self.rho_f = _positive(avoidance["auxiliary_sensing_radius"],
                               "avoidance.auxiliary_sensing_radius")
        if self.rho_o <= self.d_s:
            raise ValueError("obstacle influence radius must exceed safe clearance")

        self.w_d = _positive(advantage["target_weight"], "leader_advantage.target_weight",
                             allow_zero=True)
        self.w_o = _positive(advantage["obstacle_weight"],
                             "leader_advantage.obstacle_weight", allow_zero=True)
        self.w_c = _positive(advantage["continuity_weight"],
                             "leader_advantage.continuity_weight", allow_zero=True)
        if not math.isclose(self.w_d + self.w_o + self.w_c, 1.0, abs_tol=1e-9):
            raise ValueError("leader-advantage weights must sum to one")
        self.lambda_d = _positive(advantage["target_scale"],
                                  "leader_advantage.target_scale", allow_zero=True)
        self.lambda_c = _positive(advantage["continuity_scale"],
                                  "leader_advantage.continuity_scale", allow_zero=True)

        self.layers = [list(map(int, layer)) for layer in formation["layers"]]
        self.eta = np.asarray(formation["offsets"], dtype=float)
        self.n = int(self.eta.shape[0])
        if self.eta.shape != (self.n, 3) or self.n < 1:
            raise ValueError("formation.offsets must be an N-by-3 array")
        flattened = [node for layer in self.layers for node in layer]
        if sorted(flattened) != list(range(self.n)) or len(set(flattened)) != self.n:
            raise ValueError("formation.layers must partition vehicle indices 0..N-1")
        if any(len(layer) < 1 for layer in self.layers):
            raise ValueError("every formation layer must contain at least one vehicle")

        self.delta_h = _array(switching["hysteresis"], (len(self.layers),),
                              "switching.hysteresis")
        self.persistence = _array(switching["persistence"], (len(self.layers),),
                                  "switching.persistence")
        if np.any(self.delta_h < 0.0) or np.any(self.persistence <= 0.0):
            raise ValueError("switching hysteresis must be nonnegative and persistence positive")
        self.stability_xi = _positive(switching["stability_xi"],
                                      "switching.stability_xi")

        self.workspace_min = _array(
            flight_guard.get("workspace_min", [-1.0e12, -1.0e12, -1.0e12]),
            (3,), "flight_guard.workspace_min")
        self.workspace_max = _array(
            flight_guard.get("workspace_max", [1.0e12, 1.0e12, 1.0e12]),
            (3,), "flight_guard.workspace_max")
        if np.any(self.workspace_max <= self.workspace_min):
            raise ValueError("flight_guard.workspace_max must exceed workspace_min")

        self.target = _array(mission["target"], (3,), "mission.target")
        self.adapt_target_x = bool(mission.get("adapt_target_x_to_workspace", False))
        self.target_x_margin = _positive(
            mission.get("target_x_margin", 0.35), "mission.target_x_margin",
            allow_zero=True)
        if self.adapt_target_x:
            if not math.isfinite(float(self.workspace_max[0])):
                raise ValueError(
                    "adaptive target x requires flight_guard.workspace_max")
            self.target[0] = self.workspace_max[0] - self.target_x_margin
        if np.any(self.target < self.workspace_min) or np.any(self.target > self.workspace_max):
            raise ValueError("mission target must lie inside the configured workspace")
        self.target_velocity = _array(mission.get("target_velocity", [0.0, 0.0, 0.0]),
                                      (3,), "mission.target_velocity")
        self.start_anchor = _array(mission["start_anchor"], (3,), "mission.start_anchor")

        self.obstacles = []  # type: List[Obstacle]
        for item in raw.get("obstacles", []):
            self.obstacles.append(Obstacle(item["center"], item["radius"],
                                           item.get("velocity", [0.0, 0.0, 0.0]),
                                           item.get("height", 12.0),
                                           item.get("vehicle_id")))

        self.uav_ids = list(map(int, raw.get("crazyswarm", {}).get(
            "uav_ids", list(range(1, self.n + 1)))))
        if len(self.uav_ids) != self.n or len(set(self.uav_ids)) != self.n:
            raise ValueError("crazyswarm.uav_ids must contain N distinct IDs")
        self.obstacle_uav_ids = [
            obstacle.vehicle_id for obstacle in self.obstacles
            if obstacle.vehicle_id is not None]
        if self.obstacle_uav_ids and len(self.obstacle_uav_ids) != len(self.obstacles):
            raise ValueError(
                "either all or none of the obstacles must define obstacle.vehicle_id")
        if len(set(self.obstacle_uav_ids)) != len(self.obstacle_uav_ids):
            raise ValueError("obstacle.vehicle_id values must be distinct")
        if set(self.uav_ids).intersection(self.obstacle_uav_ids):
            raise ValueError(
                "formation and obstacle UAV IDs must not overlap")
        for obstacle in self.obstacles:
            if not (self.workspace_min[2] <= obstacle.center0[2] <= self.workspace_max[2]):
                raise ValueError(
                    "obstacle UAV flight altitude must lie inside the workspace")
        self.takeoff_height = _positive(
            raw.get("crazyswarm", {}).get("takeoff_height", self.target[2]),
            "crazyswarm.takeoff_height")

        self.velocity_filter_alpha = float(raw.get("state_estimation", {}).get(
            "velocity_filter_alpha", 0.35))
        if not 0.0 < self.velocity_filter_alpha <= 1.0:
            raise ValueError("state_estimation.velocity_filter_alpha must be in (0, 1]")

        self._validate_theorem_conditions()

    @classmethod
    def from_yaml(cls, path: str) -> "ControllerConfig":
        return cls(load_yaml(path))

    def _validate_theorem_conditions(self) -> None:
        vartheta = self.kp - self.lambda_e * (self.kv - self.lambda_e)
        alpha_t = min(self.kv - self.lambda_e, 2.0 * self.lambda_e)
        mu = 1.0 + self.stability_xi
        required = (2.0 * math.log(mu) / alpha_t) if alpha_t > 0.0 else math.inf
        if not (0.0 < self.lambda_e < self.kv and vartheta > 0.0 and
                float(np.min(self.persistence)) > required):
            raise ValueError(
                "fixed gains violate the paper theorem conditions: "
                "vartheta={:.6f}, alpha_T={:.6f}, min(T_h)={:.6f}, required={:.6f}"
                .format(vartheta, alpha_t, float(np.min(self.persistence)), required))

    def theorem_report(self) -> Dict[str, float]:
        vartheta = self.kp - self.lambda_e * (self.kv - self.lambda_e)
        alpha_t = min(self.kv - self.lambda_e, 2.0 * self.lambda_e)
        mu = 1.0 + self.stability_xi
        return {
            "vartheta": vartheta,
            "alpha_t": alpha_t,
            "mu": mu,
            "required_dwell": 2.0 * math.log(mu) / alpha_t,
            "minimum_dwell": float(np.min(self.persistence)),
        }


class HRAAPFController(object):
    """Stateful implementation of the paper's online Proposed controller."""

    def __init__(self, config: ControllerConfig) -> None:
        self.p = config
        self.active_leaders = np.asarray([layer[0] for layer in config.layers], dtype=int)
        self.pending_leaders = self.active_leaders.copy()
        self.pending_times = np.zeros(len(config.layers), dtype=float)
        # role_map[vehicle] = nominal role/slot index currently occupied.
        self.role_map = np.arange(config.n, dtype=int)
        self.switch_count = 0

    @staticmethod
    def smooth_activation(clearance: float, safe_clearance: float,
                          influence_radius: float) -> float:
        if clearance <= safe_clearance:
            return 1.0
        if clearance < influence_radius:
            phase = math.pi * (clearance - safe_clearance) / (
                influence_radius - safe_clearance)
            return 0.5 * (1.0 + math.cos(phase))
        return 0.0

    def obstacle_state(self, time_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.p.obstacles)
        centers = np.zeros((count, 3), dtype=float)
        velocities = np.zeros((count, 3), dtype=float)
        radii = np.zeros(count, dtype=float)
        for index, obstacle in enumerate(self.p.obstacles):
            centers[index], velocities[index], radii[index] = obstacle.state(time_sec)
        return centers, velocities, radii

    def compute_obstacle_risk(
            self, positions: np.ndarray, velocities: np.ndarray, time_sec: float,
            obstacle_centers: np.ndarray = None,
            obstacle_velocities: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.p
        positions = _array(positions, (p.n, 3), "positions")
        velocities = _array(velocities, (p.n, 3), "velocities")
        reference_centers, reference_velocities, radii = self.obstacle_state(time_sec)
        count = radii.size
        centers = (reference_centers if obstacle_centers is None else
                   _array(obstacle_centers, (count, 3), "obstacle_centers"))
        obstacle_velocities = (
            reference_velocities if obstacle_velocities is None else
            _array(obstacle_velocities, (count, 3), "obstacle_velocities"))
        risk = np.zeros(p.n, dtype=float)
        beta = np.zeros((p.n, count), dtype=float)
        current_clearance = np.zeros((p.n, count), dtype=float)
        predicted_clearance = np.zeros((p.n, count), dtype=float)

        for i in range(p.n):
            for obstacle_index in range(count):
                inflated_radius = radii[obstacle_index] + p.r_u + p.b_o
                relative = positions[i] - centers[obstacle_index]
                relative_velocity = velocities[i] - obstacle_velocities[obstacle_index]
                relative_xy = relative[:2]
                relative_velocity_xy = relative_velocity[:2]
                current_clearance[i, obstacle_index] = (
                    np.linalg.norm(relative_xy) - inflated_radius)

                speed_squared = float(np.dot(relative_velocity_xy, relative_velocity_xy))
                if speed_squared < 1e-10:
                    tau_star = 0.0
                else:
                    tau_star = float(np.clip(
                        -np.dot(relative_xy, relative_velocity_xy) / speed_squared,
                        0.0, p.t_p))
                predicted_relative = relative_xy + tau_star * relative_velocity_xy
                predicted_clearance[i, obstacle_index] = (
                    np.linalg.norm(predicted_relative) - inflated_radius)
                beta[i, obstacle_index] = self.smooth_activation(
                    predicted_clearance[i, obstacle_index], p.d_s, p.rho_o)
                closing_speed = max(
                    0.0,
                    -float(np.dot(relative_velocity_xy, relative_xy)) /
                    max(float(np.linalg.norm(relative_xy)), 1e-9),
                )
                risk[i] += beta[i, obstacle_index] * (1.0 + p.lambda_v * closing_speed)

        return risk, beta, current_clearance, predicted_clearance

    def compute_leader_scores(self, positions: np.ndarray, risk: np.ndarray) -> np.ndarray:
        p = self.p
        positions = _array(positions, (p.n, 3), "positions")
        risk = _array(risk, (p.n,), "risk")
        scores = np.zeros(p.n, dtype=float)
        for layer_index, layer in enumerate(p.layers):
            leader = int(self.active_leaders[layer_index])
            for vehicle in layer:
                target_advantage = 1.0 / (
                    1.0 + p.lambda_d * np.linalg.norm(positions[vehicle] - p.target))
                obstacle_advantage = 1.0 / (1.0 + risk[vehicle])
                continuity_advantage = 1.0 / (
                    1.0 + p.lambda_c * np.linalg.norm(
                        positions[vehicle] - positions[leader]))
                scores[vehicle] = (
                    p.w_d * target_advantage +
                    p.w_o * obstacle_advantage +
                    p.w_c * continuity_advantage)
        return scores

    def update_leaders(self, scores: np.ndarray, positions: np.ndarray,
                       dt: float) -> List[Dict[str, float]]:
        scores = _array(scores, (self.p.n,), "scores")
        events = []  # type: List[Dict[str, float]]
        for layer_index, layer in enumerate(self.p.layers):
            candidate_scores = scores[np.asarray(layer, dtype=int)]
            best = int(layer[int(np.argmax(candidate_scores))])
            current = int(self.active_leaders[layer_index])
            if best == current:
                self.pending_leaders[layer_index] = current
                self.pending_times[layer_index] = 0.0
                continue

            condition = scores[best] > scores[current] + self.p.delta_h[layer_index]
            if condition:
                if int(self.pending_leaders[layer_index]) == best:
                    self.pending_times[layer_index] += float(dt)
                else:
                    self.pending_leaders[layer_index] = best
                    self.pending_times[layer_index] = float(dt)
                if self.pending_times[layer_index] + 1e-12 >= self.p.persistence[layer_index]:
                    old = current
                    self.active_leaders[layer_index] = best
                    self.pending_leaders[layer_index] = best
                    self.pending_times[layer_index] = 0.0
                    self._reassign_layer_roles(layer_index, positions)
                    self.switch_count += 1
                    events.append({
                        "layer": float(layer_index),
                        "old_leader": float(old),
                        "new_leader": float(best),
                        "advantage_gain": float(scores[best] - scores[old]),
                    })
            else:
                self.pending_leaders[layer_index] = current
                self.pending_times[layer_index] = 0.0
        return events

    @staticmethod
    def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _point_on_segment(point: np.ndarray, first: np.ndarray,
                          second: np.ndarray, tolerance: float) -> bool:
        return bool(
            point[0] >= min(first[0], second[0]) - tolerance and
            point[0] <= max(first[0], second[0]) + tolerance and
            point[1] >= min(first[1], second[1]) - tolerance and
            point[1] <= max(first[1], second[1]) + tolerance)

    @classmethod
    def _segments_intersect(cls, a: np.ndarray, b: np.ndarray,
                            c: np.ndarray, d: np.ndarray) -> bool:
        tolerance = 1e-10
        o1 = cls._cross_2d(b - a, c - a)
        o2 = cls._cross_2d(b - a, d - a)
        o3 = cls._cross_2d(d - c, a - c)
        o4 = cls._cross_2d(d - c, b - c)
        proper = (((o1 > tolerance and o2 < -tolerance) or
                   (o1 < -tolerance and o2 > tolerance)) and
                  ((o3 > tolerance and o4 < -tolerance) or
                   (o3 < -tolerance and o4 > tolerance)))
        if proper:
            return True
        return bool(
            (abs(o1) <= tolerance and cls._point_on_segment(c, a, b, tolerance)) or
            (abs(o2) <= tolerance and cls._point_on_segment(d, a, b, tolerance)) or
            (abs(o3) <= tolerance and cls._point_on_segment(a, c, d, tolerance)) or
            (abs(o4) <= tolerance and cls._point_on_segment(b, c, d, tolerance)))

    @classmethod
    def _assignment_has_crossing(cls, positions: np.ndarray,
                                 followers: Sequence[int], desired: np.ndarray) -> bool:
        for first in range(len(followers) - 1):
            for second in range(first + 1, len(followers)):
                if cls._segments_intersect(
                        positions[followers[first]], desired[first],
                        positions[followers[second]], desired[second]):
                    return True
        return False

    def _reassign_layer_roles(self, layer_index: int, positions: np.ndarray) -> None:
        p = self.p
        layer = p.layers[layer_index]
        leader_node = int(self.active_leaders[layer_index])
        leader_role = int(layer[0])
        followers = [vehicle for vehicle in layer if vehicle != leader_node]
        follower_roles = list(layer[1:])
        self.role_map[leader_node] = leader_role
        if not followers:
            return

        best_cost = math.inf
        best_roles = None
        leader_position = positions[leader_node]
        for roles in itertools.permutations(follower_roles):
            desired = np.zeros((len(followers), 3), dtype=float)
            cost = 0.0
            for index, vehicle in enumerate(followers):
                desired[index] = (
                    leader_position + p.eta[roles[index]] - p.eta[leader_role])
                cost += float(np.linalg.norm(positions[vehicle] - desired[index]) ** 2)
            if self._assignment_has_crossing(positions, followers, desired):
                continue
            if cost < best_cost:
                best_cost = cost
                best_roles = roles
        if best_roles is None:
            raise RuntimeError("no non-crossing slot assignment for layer {}".format(layer_index))
        for vehicle, role in zip(followers, best_roles):
            self.role_map[vehicle] = int(role)

    def active_parent_map(self) -> np.ndarray:
        parents = np.full(self.p.n, -1, dtype=int)
        for layer_index, layer in enumerate(self.p.layers):
            leader = int(self.active_leaders[layer_index])
            upper_leader = -1 if layer_index == 0 else int(
                self.active_leaders[layer_index - 1])
            parents[leader] = upper_leader
            for follower in layer:
                if follower != leader:
                    parents[follower] = leader
        return parents

    def _layer_for_vehicle(self, vehicle: int) -> int:
        for layer_index, layer in enumerate(self.p.layers):
            if vehicle in layer:
                return layer_index
        raise ValueError("vehicle {} is not in a layer".format(vehicle))

    def _follower_aid_edges(self) -> List[Tuple[int, int]]:
        edges = []  # type: List[Tuple[int, int]]
        for layer_index, layer in enumerate(self.p.layers):
            leader = int(self.active_leaders[layer_index])
            followers = [vehicle for vehicle in layer if vehicle != leader]
            edges.extend(zip(followers[:-1], followers[1:]))
            if len(followers) >= 3:
                edges.append((followers[0], followers[-1]))
        return edges

    def follower_auxiliary(self, positions: np.ndarray, velocities: np.ndarray,
                           vehicle: int) -> np.ndarray:
        p = self.p
        layer_index = self._layer_for_vehicle(vehicle)
        if vehicle == int(self.active_leaders[layer_index]):
            return np.zeros(3, dtype=float)
        result = np.zeros(3, dtype=float)
        for first, second in self._follower_aid_edges():
            if first == vehicle:
                neighbor = second
            elif second == vehicle:
                neighbor = first
            else:
                continue
            relative = positions[neighbor] - positions[vehicle]
            distance = float(np.linalg.norm(relative))
            if 1e-8 < distance < p.rho_f:
                slot_error = relative - (
                    p.eta[self.role_map[neighbor]] - p.eta[self.role_map[vehicle]])
                result += p.k_f * (
                    velocities[neighbor] - velocities[vehicle] +
                    p.lambda_e * slot_error)
        return result

    def obstacle_avoidance(self, positions: np.ndarray, velocities: np.ndarray,
                           time_sec: float, beta: np.ndarray, risk: np.ndarray,
                           vehicle: int, reference: np.ndarray,
                           obstacle_centers: np.ndarray = None,
                           obstacle_radii: np.ndarray = None) -> np.ndarray:
        del velocities  # Risk already contains the relative-velocity dependence.
        p = self.p
        reference_centers, _, reference_radii = self.obstacle_state(time_sec)
        centers = reference_centers if obstacle_centers is None else obstacle_centers
        radii = reference_radii if obstacle_radii is None else obstacle_radii
        repulsive = np.zeros(3, dtype=float)
        tangential = np.zeros(3, dtype=float)
        for obstacle_index, radius in enumerate(radii):
            relative = positions[vehicle] - centers[obstacle_index]
            relative_xy = relative[:2]
            distance = float(np.linalg.norm(relative_xy))
            if distance < 1e-8:
                normal = np.array([1.0, 0.0, 0.0])
            else:
                normal = np.array([relative_xy[0] / distance,
                                   relative_xy[1] / distance, 0.0])
            inflated_radius = radius + p.r_u + p.b_o
            clearance = distance - inflated_radius
            activation = float(beta[vehicle, obstacle_index])
            if activation <= 0.0:
                continue
            nonnegative_clearance = max(clearance, 0.0)
            denominator = nonnegative_clearance + p.apf_epsilon
            normal_gain = (
                p.k_o * activation * (1.0 / denominator - 1.0 / p.rho_o) /
                (denominator ** 2))
            repulsive += max(0.0, normal_gain) * normal

            tangent = np.array([-normal[1], normal[0], 0.0])
            sign = 1.0 if float(np.dot(tangent, reference - positions[vehicle])) >= 0.0 else -1.0
            tangential += p.k_t * activation * sign * tangent

        alpha = p.alpha_0 * risk[vehicle] / (
            risk[vehicle] + p.c_o + np.finfo(float).eps)
        return alpha * (repulsive + tangential)

    def inter_uav_safety(self, positions: np.ndarray, vehicle: int) -> np.ndarray:
        p = self.p
        result = np.zeros(3, dtype=float)
        for neighbor in range(p.n):
            if neighbor == vehicle:
                continue
            relative = positions[vehicle] - positions[neighbor]
            distance = float(np.linalg.norm(relative))
            if distance < 1e-8 or distance >= p.rho_i:
                continue
            normal = relative / distance
            gap = p.rho_i - distance
            result += p.k_i * (gap ** 2) * normal
        return result

    def compute_controls(self, positions: np.ndarray, velocities: np.ndarray,
                         time_sec: float, beta: np.ndarray,
                         risk: np.ndarray, obstacle_centers: np.ndarray = None,
                         obstacle_radii: np.ndarray = None) -> np.ndarray:
        p = self.p
        parents = self.active_parent_map()
        controls = np.zeros((p.n, 3), dtype=float)
        active_set = set(map(int, self.active_leaders))
        for vehicle in range(p.n):
            parent = int(parents[vehicle])
            if parent < 0:
                reference = p.target
                tracking = (
                    p.kp * (p.target - positions[vehicle]) +
                    p.kv * (p.target_velocity - velocities[vehicle]))
                auxiliary = np.zeros(3, dtype=float)
            else:
                reference_offset = (
                    p.eta[self.role_map[vehicle]] - p.eta[self.role_map[parent]])
                reference = positions[parent] + reference_offset
                tracking = (
                    p.kp * (reference - positions[vehicle]) +
                    p.kv * (velocities[parent] - velocities[vehicle]))
                auxiliary = (np.zeros(3, dtype=float) if vehicle in active_set else
                             self.follower_auxiliary(positions, velocities, vehicle))
            obstacle = self.obstacle_avoidance(
                positions, velocities, time_sec, beta, risk, vehicle, reference,
                obstacle_centers=obstacle_centers,
                obstacle_radii=obstacle_radii)
            safety = self.inter_uav_safety(positions, vehicle)
            controls[vehicle] = tracking + auxiliary + obstacle + safety
        return controls

    @staticmethod
    def saturate_rows(values: np.ndarray, maximum_norm: float) -> np.ndarray:
        result = np.asarray(values, dtype=float).copy()
        norms = np.linalg.norm(result, axis=1)
        selected = norms > maximum_norm
        if np.any(selected):
            result[selected] *= (maximum_norm / norms[selected])[:, None]
        return result

    def step(self, positions: np.ndarray, velocities: np.ndarray,
             time_sec: float, dt: float = None,
             obstacle_positions: np.ndarray = None,
             obstacle_velocities: np.ndarray = None) -> Dict[str, Any]:
        dt = self.p.dt if dt is None else float(dt)
        _, _, obstacle_radii = self.obstacle_state(time_sec)
        risk, beta, current_clearance, predicted_clearance = (
            self.compute_obstacle_risk(
                positions, velocities, time_sec,
                obstacle_centers=obstacle_positions,
                obstacle_velocities=obstacle_velocities))
        scores = self.compute_leader_scores(positions, risk)
        events = self.update_leaders(scores, positions, dt)
        controls = self.compute_controls(
            positions, velocities, time_sec, beta, risk,
            obstacle_centers=obstacle_positions,
            obstacle_radii=obstacle_radii)
        return {
            "controls": controls,
            "risk": risk,
            "beta": beta,
            "current_clearance": current_clearance,
            "predicted_clearance": predicted_clearance,
            "scores": scores,
            "leaders": self.active_leaders.copy(),
            "roles": self.role_map.copy(),
            "switch_events": events,
        }


def controller_initial_positions(config: ControllerConfig) -> np.ndarray:
    """Positions at paper-controller handover after the low takeoff phase."""
    positions = config.eta + config.start_anchor[None, :]
    positions[:, 2] = config.takeoff_height + config.eta[:, 2]
    return positions


def minimum_inter_uav_margin(positions: np.ndarray, safe_distance: float) -> float:
    minimum = math.inf
    for first in range(positions.shape[0] - 1):
        for second in range(first + 1, positions.shape[0]):
            margin = float(np.linalg.norm(positions[first] - positions[second]) - safe_distance)
            minimum = min(minimum, margin)
    return minimum
