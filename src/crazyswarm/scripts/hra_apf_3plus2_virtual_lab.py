#!/usr/bin/env python3
"""Run the embedded HRA-APF controller through the original Crazyswarm Python API.

The host program computes translational position, velocity, and acceleration
only.  The required yaw and body-rate fields of ``cmdFullState`` are fixed to
zero and are not part of the paper controller.
"""

from __future__ import division, print_function

import argparse
import copy
import csv
import datetime
import itertools
import math
import os
import select
import sys
import termios
import tty
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml


# Embedded controller code: keep this experiment script independent of external modules.
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
        self.obstacle_force_limit = _positive(
            avoidance.get("obstacle_force_limit", 1.20),
            "avoidance.obstacle_force_limit", allow_zero=True)
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
        self.tangent_signs = np.zeros((config.n, len(config.obstacles)), dtype=float)
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
                self.tangent_signs[vehicle, obstacle_index] = 0.0
                continue
            nonnegative_clearance = max(clearance, 0.0)
            denominator = nonnegative_clearance + p.apf_epsilon
            normal_gain = (
                p.k_o * activation * (1.0 / denominator - 1.0 / p.rho_o) /
                (denominator ** 2))
            repulsive += max(0.0, normal_gain) * normal

            tangent = np.array([-normal[1], normal[0], 0.0])
            raw_sign = 1.0 if float(np.dot(tangent, reference - positions[vehicle])) >= 0.0 else -1.0
            stored_sign = float(self.tangent_signs[vehicle, obstacle_index])
            if stored_sign == 0.0:
                stored_sign = raw_sign
                self.tangent_signs[vehicle, obstacle_index] = stored_sign
            sign = stored_sign
            tangential += p.k_t * activation * sign * tangent

        alpha = p.alpha_0 * risk[vehicle] / (
            risk[vehicle] + p.c_o + np.finfo(float).eps)
        force = alpha * (repulsive + tangential)
        force_norm = float(np.linalg.norm(force[:2]))
        if p.obstacle_force_limit > 0.0 and force_norm > p.obstacle_force_limit:
            force[:2] *= p.obstacle_force_limit / force_norm
        return force

    def inter_uav_safety(self, positions: np.ndarray, vehicle: int) -> np.ndarray:
        p = self.p
        result = np.zeros(3, dtype=float)
        for neighbor in range(p.n):
            if neighbor == vehicle:
                continue
            relative_xy = positions[vehicle, :2] - positions[neighbor, :2]
            distance = float(np.linalg.norm(relative_xy))
            if distance < 1e-8 or distance >= p.rho_i:
                continue
            normal = np.array([relative_xy[0] / distance,
                               relative_xy[1] / distance, 0.0])
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


SCRIPT_DIR = "/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch"
CRAZYSWARM_SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
if CRAZYSWARM_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CRAZYSWARM_SCRIPTS_DIR)


def embedded_lab_config() -> Dict[str, Any]:
    """Default 3-UAV, 2-virtual-obstacle hardware experiment parameters.

    The Crazyswarm device YAML is still external because chooser.py owns the
    real radio IDs, channels, and initialPosition entries.
    """
    return copy.deepcopy({
        "control": {
            "rate_hz": 50.0,
            "duration": 15.0,
            "velocity_limit": 0.20,
            "xy_velocity_slew_rate": 0.12,
            "reset_command_velocity_on_switch": True,
        },
        "gains": {
            "kp": 0.80,
            "kv": 2.80,
            "k_f": 1.00,
            "lambda_e": 0.32,
            "k_i": 4.00,
            "k_o": 0.25,
            "k_t": 0.40,
        },
        "avoidance": {
            "uav_radius": 0.05,
            "obstacle_buffer": 0.05,
            "inter_uav_buffer": 0.04,
            "obstacle_influence_radius": 0.5,
            "safe_clearance": 0.13,
            "prediction_horizon": 2.0,
            "closing_speed_weight": 0.70,
            "risk_adaptive_gain": 0.90,
            "risk_regularizer": 0.22,
            "apf_epsilon": 0.18,
            "obstacle_force_limit": 0.35,
            "inter_uav_sensing_radius": 0.10,
            "auxiliary_sensing_radius": 1.20,
        },
        "leader_advantage": {
            "target_weight": 0.35,
            "obstacle_weight": 0.55,
            "continuity_weight": 0.10,
            "target_scale": 1.0,
            "continuity_scale": 1.0,
        },
        "switching": {
            "hysteresis": [0.025],
            "persistence": [0.70],
            "stability_xi": 0.20,
        },
        "formation": {
            "layers": [[0, 1, 2]],
            "offsets": [
                [0.00, 0.00, 0.00],
                [0.10, -0.15, 0.00],
                [0.10, 0.15, 0.00],
            ],
        },
        "mission": {
            "start_anchor": [1.00, 0.00, 0.00],
            "target": [-1.00, 0.30, 0.20],
            "adapt_target_x_to_workspace": False,
            "target_x_margin": 0.20,
            "target_velocity": [0.00, 0.00, 0.00],
        },
        "crazyswarm": {
            "uav_ids": [3, 4, 7],
            "takeoff_height": 0.20,
        },
        "state_estimation": {
            "velocity_filter_alpha": 0.35,
        },
        "flight_guard": {
            "start_position_tolerance": 0.20,
            "minimum_inter_uav_distance": 0.10,
            "minimum_obstacle_clearance": 0.02,
            "obstacle_boundary_margin": 0.05,
            "inter_uav_velocity_guard_distance": 0.18,
            "inter_uav_separation_speed": 0.14,
            "workspace_min": [-2.00, -1.50, 0.00],
            "workspace_max": [1.50, 1.50, 1.50],
        },
        "initialization": {
            "use_live_start": True,
            "use_live_offsets": True,
            "validate_start_positions": False,
        },
        "virtual_obstacles": [
            {
                "center": [0.00, 0.80, 0.20],
                "radius": 0.1,
                "velocity": [0.00, 0.00, 0.00],
                "height": 2.00,
            },
            {
                "center": [-0.60, -1.50, 0.20],
                "radius": 0.1,
                "velocity": [0.00, 0.05, 0.00],
                "height": 2.00,
            },
        ],
        "obstacles": [],
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
        help="Crazyswarm YAML for three formation UAVs; obstacles are virtual.")
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
    group.add_argument("--takeoff-duration", type=float, default=2.0)
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
        help="Use cf.cmdVelocityWorld; this is the default command mode.")
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


def obstacle_vehicle_ids_from_raw(raw_config: Dict[str, Any]) -> List[Any]:
    ids = []
    for obstacle in raw_config.get("obstacles", []):
        vehicle_id = obstacle.get("vehicle_id", None)
        if obstacle.get("virtual", False) or vehicle_id is None or int(vehicle_id) <= 0:
            ids.append(None)
        else:
            ids.append(int(vehicle_id))
    return ids


def dynamic_obstacle_indices(vehicle_ids: Sequence[Any]) -> List[int]:
    return [index for index, vehicle_id in enumerate(vehicle_ids)
            if vehicle_id is not None]


def append_virtual_obstacles(config: ControllerConfig,
                             raw_config: Dict[str, Any]) -> None:
    for item in raw_config.get("virtual_obstacles", []):
        config.obstacles.append(Obstacle(
            item["center"], item["radius"],
            item.get("velocity", [0.0, 0.0, 0.0]),
            item.get("height", 12.0),
            None))


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
                                    guard: Dict[str, Any],
                                    obstacle_indices: Sequence[int] = None) -> None:
    if obstacle_indices is None:
        obstacle_indices = list(range(len(config.obstacles)))
    expected = np.asarray(
        [config.obstacles[index].center0 for index in obstacle_indices],
        dtype=float)
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
                             obstacle_positions: np.ndarray = None,
                             obstacle_indices: Sequence[int] = None) -> bool:
    initialization = raw_config.get("initialization", {})
    if not bool(initialization.get("use_live_start", False)):
        return False

    formation_positions = np.asarray(formation_positions, dtype=float)
    if formation_positions.shape != (config.n, 3):
        raise SafetyStop("invalid live formation positions")
    anchor = config.start_anchor.copy()
    if bool(initialization.get("use_live_offsets", False)):
        root_index = int(config.layers[0][0]) if config.layers else 0
        anchor[:2] = formation_positions[root_index, :2] - config.eta[root_index, :2]
        live_eta = config.eta.copy()
        live_eta[:, :2] = formation_positions[:, :2] - anchor[:2]
        live_eta[:, 2] = 0.0
        config.eta = live_eta
    else:
        anchor[:2] = np.mean(formation_positions[:, :2] - config.eta[:, :2], axis=0)
    config.start_anchor = anchor

    if obstacle_positions is not None:
        obstacle_positions = np.asarray(obstacle_positions, dtype=float)
        if obstacle_indices is None:
            obstacle_indices = list(range(len(config.obstacles)))
        if obstacle_positions.shape != (len(obstacle_indices), 3):
            raise SafetyStop("invalid live obstacle positions")
        for row_index, obstacle_index in enumerate(obstacle_indices):
            config.obstacles[obstacle_index].center0 = obstacle_positions[row_index].copy()
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


def limit_xy_velocity_slew(previous: np.ndarray,
                           desired: np.ndarray,
                           dt: float,
                           slew_rate: float) -> np.ndarray:
    limited = np.asarray(desired, dtype=float).copy()
    slew_rate = max(float(slew_rate), 0.0)
    max_delta = slew_rate * max(float(dt), 1e-6)
    if max_delta <= 0.0:
        return limited
    previous = np.asarray(previous, dtype=float)
    for index in range(limited.shape[0]):
        delta_xy = limited[index, :2] - previous[index, :2]
        delta_norm = float(np.linalg.norm(delta_xy))
        if delta_norm > max_delta:
            limited[index, :2] = (
                previous[index, :2] + delta_xy * (max_delta / delta_norm))
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
                          command_mode: str = "full_state",
                          obstacle_indices: Sequence[int] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((len(config.obstacles), 3), dtype=float)
    velocities = np.zeros_like(positions)
    for index, obstacle in enumerate(config.obstacles):
        positions[index], velocities[index] = obstacle_boundary_limited_state(
            config, obstacle, time_sec, guard)
    accelerations = np.zeros_like(positions)
    if obstacle_indices is None:
        obstacle_indices = list(range(len(obstacle_crazyflies)))
    if obstacle_crazyflies:
        commanded_positions = np.asarray(
            [positions[index] for index in obstacle_indices], dtype=float)
        commanded_velocities = np.asarray(
            [velocities[index] for index in obstacle_indices], dtype=float)
        commanded_accelerations = np.zeros_like(commanded_positions)
        command_state(
            obstacle_crazyflies, commanded_positions, commanded_velocities,
            commanded_accelerations, command_mode)
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
            "uav{}_cmd_vx".format(index + 1), "uav{}_cmd_vy".format(index + 1),
            "uav{}_cmd_vz".format(index + 1),
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
            diagnostic: Dict[str, Any], command_velocities: np.ndarray = None,
            obstacle_positions: np.ndarray = None,
            obstacle_velocities: np.ndarray = None) -> List[float]:
    current_clearance = diagnostic["current_clearance"]
    minimum_obstacle = float(np.min(current_clearance)) if current_clearance.size else math.inf
    minimum_inter = minimum_inter_uav_margin(positions, config.d_f)
    row = [time_sec, minimum_obstacle, minimum_inter, float(controller.switch_count)]
    controls = diagnostic["controls"]
    if command_velocities is None:
        command_velocities = np.zeros_like(velocities)
    for index in range(config.n):
        row.extend(list(positions[index]))
        row.extend(list(velocities[index]))
        row.extend(list(command_velocities[index]))
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
    append_virtual_obstacles(config, raw_config)
    guard = raw_config["flight_guard"]
    config.raw_flight_guard = guard
    planar_control_height = float(config.takeoff_height)
    config.target[2] = planar_control_height
    crazyflies_yaml, file_ids = load_crazyflies_yaml(own_args.crazyflies_config)
    obstacle_vehicle_ids = obstacle_vehicle_ids_from_raw(raw_config)
    dynamic_indices = dynamic_obstacle_indices(obstacle_vehicle_ids)
    dynamic_obstacle_ids = [
        obstacle_vehicle_ids[index] for index in dynamic_indices]
    if hasattr(config, "obstacle_uav_ids"):
        config.obstacle_uav_ids = dynamic_obstacle_ids
    expected_ids = config.uav_ids + dynamic_obstacle_ids
    if sorted(file_ids) != sorted(expected_ids):
        raise ValueError(
            "crazyflies YAML IDs {} differ from formation+obstacle IDs {}".format(
                file_ids, expected_ids))
    if file_ids != expected_ids:
        print(
            "crazyflies YAML IDs {} will be remapped to controller order {}."
            .format(file_ids, expected_ids))
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
            not own_args.position_command and not own_args.full_state_command):
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
        for cf, obstacle_index in zip(obstacle_crazyflies, dynamic_indices):
            obstacle = config.obstacles[obstacle_index]
            cf.takeoff(targetHeight=float(obstacle.center0[2]),
                       duration=float(own_args.takeoff_duration))
        time_helper.sleep(float(own_args.takeoff_duration) + 0.5)

    positions = collect_positions(crazyflies)
    obstacle_positions = (collect_positions(obstacle_crazyflies)
                          if obstacle_crazyflies else None)
    if apply_live_initial_state(
            config, raw_config, positions, obstacle_positions,
            obstacle_indices=dynamic_indices):
        print("live start applied: start_anchor={}, obstacle_centers={}".format(
            config.start_anchor.tolist(),
            [obstacle.center0.tolist() for obstacle in config.obstacles]))
    if bool(raw_config.get("initialization", {}).get(
            "validate_start_positions", True)):
        validate_start_positions(
            config, positions, guard, after_takeoff=not own_args.no_takeoff)
    if obstacle_crazyflies:
        validate_obstacle_uav_positions(
            config, obstacle_positions, guard,
            obstacle_indices=dynamic_indices)
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
    last_command_velocity = np.zeros((config.n, 3), dtype=float)
    xy_velocity_slew_rate = float(
        raw_config.get("control", {}).get("xy_velocity_slew_rate", 0.0))
    reset_command_velocity_on_switch = bool(
        raw_config.get("control", {}).get(
            "reset_command_velocity_on_switch", True))
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
                    measured_obstacle_positions = collect_positions(obstacle_crazyflies)
                    measured_obstacle_velocities = collect_velocities_if_available(
                        obstacle_crazyflies)
                    obstacle_positions, reference_obstacle_velocities = command_obstacle_uavs(
                        config, obstacle_crazyflies, elapsed, guard,
                        command_mode=command_mode,
                        obstacle_indices=dynamic_indices)
                    obstacle_velocities = reference_obstacle_velocities.copy()
                    for row_index, obstacle_index in enumerate(dynamic_indices):
                        obstacle_positions[obstacle_index] = (
                            measured_obstacle_positions[row_index])
                        if measured_obstacle_velocities is not None:
                            obstacle_velocities[obstacle_index] = (
                                measured_obstacle_velocities[row_index])
                    for obstacle_index in dynamic_indices:
                        reference_velocity = reference_obstacle_velocities[obstacle_index]
                        if (not obstacle_hovered[obstacle_index] and
                                np.linalg.norm(config.obstacles[obstacle_index].velocity) > 1e-9 and
                                np.linalg.norm(reference_velocity) < 1e-9):
                            obstacle_hovered[obstacle_index] = True
                            print("t={:.2f}s obstacle UAV {} reached boundary margin; hovering."
                                  .format(
                                      elapsed,
                                      obstacle_vehicle_ids[obstacle_index]))
                else:
                    obstacle_positions, obstacle_velocities, _ = (
                        controller.obstacle_state(elapsed))

                planar_positions = positions.copy()
                planar_velocities = velocities.copy()
                planar_positions[:, 2] = planar_control_height
                planar_velocities[:, 2] = 0.0
                planar_obstacle_positions = obstacle_positions
                planar_obstacle_velocities = obstacle_velocities
                if obstacle_positions is not None:
                    planar_obstacle_positions = obstacle_positions.copy()
                    planar_obstacle_positions[:, 2] = planar_control_height
                if obstacle_velocities is not None:
                    planar_obstacle_velocities = obstacle_velocities.copy()
                    planar_obstacle_velocities[:, 2] = 0.0

                diagnostic = controller.step(
                    planar_positions, planar_velocities, elapsed, dt=config.dt,
                    obstacle_positions=planar_obstacle_positions,
                    obstacle_velocities=planar_obstacle_velocities)
                apply_flight_guards(config, positions, diagnostic, guard)
                controls = diagnostic["controls"].copy()
                switched_velocity_indices = set()
                if command_mode == "velocity" and reset_command_velocity_on_switch:
                    for event in diagnostic["switch_events"]:
                        layer_index = int(event["layer"])
                        switched_velocity_indices.update(config.layers[layer_index])
                velocity_base = (
                    last_command_velocity.copy() if command_mode == "velocity"
                    else planar_velocities.copy())
                for vehicle_index in switched_velocity_indices:
                    velocity_base[int(vehicle_index)] = 0.0
                next_velocity = controller.saturate_rows(
                    velocity_base + config.dt * controls, config.velocity_limit)
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
                        slew_reference = last_command_velocity.copy()
                        for vehicle_index in switched_velocity_indices:
                            slew_reference[int(vehicle_index)] = 0.0
                        next_velocity = limit_xy_velocity_slew(
                            slew_reference, next_velocity,
                            config.dt, xy_velocity_slew_rate)
                    next_velocity = enforce_inter_uav_velocity_guard(
                        positions, next_velocity, guard, config.dt,
                        config.velocity_limit)
                next_velocity[:, 2] = 0.0
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
                if command_mode == "velocity":
                    last_command_velocity = next_velocity.copy()
                if scene_visualizer is not None:
                    scene_visualizer.update(
                        elapsed, positions, diagnostic["leaders"],
                        controller.active_parent_map(), controller.switch_count,
                        obstacle_positions=obstacle_positions)

                buffered_rows.append(log_row(
                    elapsed, config, controller, positions, velocities, diagnostic,
                    command_velocities=next_velocity,
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
