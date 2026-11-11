# -*- coding: utf-8 -*-
"""Crazyswarm1：三架跟随机 + 虚拟领导机 + 编队切换 + LMPC。

仿真：
    python3 lmpc_crazyswarm1_sim_real.py --sim

实飞：
    先启动配置好的 Crazyswarm launch 文件，再执行：
    python3 lmpc_crazyswarm1_sim_real.py

仿真和实飞使用同一份 launch/crazyflies.yaml。
本脚本不会创建或修改 crazyflies.yaml。
"""

from __future__ import print_function

import csv
import math
import sys
import time
import warnings
from datetime import datetime

import numpy as np
from scipy.optimize import minimize


# ============================= 需要检查的参数 ==============================

# 必须与 crazyflies.yaml 中的 ID 一致
CF_IDS = (1, 2, 3)

# 控制周期与任务时间
DT = 0.10
MISSION_TIME = 40.0
SETTLE_TIME = 5.0

# 起降高度均相对于启动前测得的地面高度
TAKEOFF_DZ = 0.30
LAND_DZ = 0.04

# 虚拟领导机轨迹
LEADER_RADIUS = 0.10
LEADER_OMEGA = 0.25
LEADER_RISE = 0.08

# 三角形外接圆半径
FORMATION_R = 0.15
ENCIRCLE_R = 0.18
CAPTURE_R = 0.12
OBSTACLE_R = 0.24

# 编队切换时间
ENCIRCLE_AT = 10.0
CAPTURE_AT = 20.0
RADIUS_RATE = 0.05

# 如果已经知道障碍物在 Lighthouse /world 下的坐标，改成：
# OBSTACLE_WORLD = np.array([x, y, z], dtype=float)
#
# 保持 None 时，使用相对于任务中心的虚拟障碍物。
OBSTACLE_WORLD = None
OBSTACLE_OFFSET = np.array([-0.10, 0.10, 0.10], dtype=float)

OBSTACLE_TRIGGER = 0.12
OBSTACLE_RELEASE = 0.18
OBSTACLE_SAFE = 0.13
OBSTACLE_WARNING = 0.20

# 无人机间距
MIN_START_SEP = 0.18
MIN_COMMAND_SEP = 0.13
SEPARATION_WARNING = 0.22
CRITICAL_SEP = 0.08

# 速度、加速度与定位保护
MAX_SPEED = 0.25
MAX_ACCEL = 0.80
MAX_POSITION_JUMP = 0.20

# 以起飞后三机中心为中心的飞行安全框
BOX_HALF_X = 0.65
BOX_HALF_Y = 0.43
BOX_DOWN = 0.06
BOX_UP = 0.16

# LMPC参数
N_PRED = 4
MPC_MAXITER = 20
MPC_FTOL = 1e-3

# 反步控制器和LMPC权重
K1 = 1.5
K2 = 1.8

W_FORM = 35.0
W_VEL = 5.0
W_U = 0.15
W_OBS = 250.0
W_SEP = 180.0

# 有向环拓扑：
# cf1 <- cf3
# cf2 <- cf1
# cf3 <- cf2
# cf3能够获取虚拟领导机信息
A_TOPO = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

B_INFO = np.array([0.0, 0.0, 1.0])

# 三架飞机在三角形中的方向
ANGLES = np.deg2rad([90.0, 210.0, 330.0])

FORMATION_UNIT = np.column_stack((
    np.cos(ANGLES),
    np.sin(ANGLES),
    np.zeros(3),
))


# =============================== 数学函数 =================================

def clip_norm(vector, limit):
    """限制向量模长。"""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm <= limit or norm < 1e-12:
        return vector

    return vector * (limit / norm)


def smoothstep01(value):
    """0到1之间的平滑过渡函数。"""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def leader_state(t_now, center):
    """计算虚拟领导机的位置、速度和加速度。"""
    if t_now <= SETTLE_TIME:
        return center.copy(), np.zeros(3), np.zeros(3)

    tau = t_now - SETTLE_TIME
    theta = LEADER_OMEGA * tau
    duration = max(MISSION_TIME - SETTLE_TIME, 1.0)
    s = np.clip(tau / duration, 0.0, 1.0)

    position = center + np.array([
        LEADER_RADIUS * (math.cos(theta) - 1.0),
        LEADER_RADIUS * math.sin(theta),
        LEADER_RISE * smoothstep01(s),
    ])

    velocity = np.array([
        -LEADER_RADIUS * LEADER_OMEGA * math.sin(theta),
        LEADER_RADIUS * LEADER_OMEGA * math.cos(theta),
        LEADER_RISE * 6.0 * s * (1.0 - s) / duration,
    ])

    acceleration = np.array([
        -LEADER_RADIUS * LEADER_OMEGA ** 2 * math.cos(theta),
        -LEADER_RADIUS * LEADER_OMEGA ** 2 * math.sin(theta),
        LEADER_RISE * 6.0 * (1.0 - 2.0 * s) / duration ** 2,
    ])

    return position, velocity, acceleration


def desired_states(t_now, center, radius, radius_dot=0.0):
    """计算三架跟随机的期望状态。"""
    leader_p, leader_v, leader_a = leader_state(t_now, center)

    desired_p = leader_p + radius * FORMATION_UNIT
    desired_v = leader_v + radius_dot * FORMATION_UNIT
    desired_a = np.repeat(leader_a[None, :], 3, axis=0)

    return desired_p, desired_v, desired_a, leader_p


def graph_error(
        index,
        position,
        velocity,
        positions,
        velocities,
        desired_p,
        desired_v):
    """根据有向通信拓扑计算编队误差。"""
    own_p = position - desired_p[index]
    own_v = velocity - desired_v[index]

    error_p = B_INFO[index] * own_p
    error_v = B_INFO[index] * own_v

    neighbors = np.nonzero(A_TOPO[index])[0]

    for neighbor in neighbors:
        neighbor_p = positions[neighbor] - desired_p[neighbor]
        neighbor_v = velocities[neighbor] - desired_v[neighbor]

        error_p += A_TOPO[index, neighbor] * (own_p - neighbor_p)
        error_v += A_TOPO[index, neighbor] * (own_v - neighbor_v)

    return error_p, error_v


# ============================== 反步控制器 ================================

def backstepping(
        index,
        positions,
        velocities,
        desired_p,
        desired_v,
        desired_a):
    """反步辅助控制器，LMPC失败时使用该控制量。"""
    error_p, error_v = graph_error(
        index,
        positions[index],
        velocities[index],
        positions,
        velocities,
        desired_p,
        desired_v,
    )

    z2 = error_v + K1 * error_p

    acceleration = (
        desired_a[index]
        - K1 * error_v
        - K2 * z2
        - error_p
    )

    return clip_norm(acceleration, MAX_ACCEL), error_p, z2


# ================================ LMPC =====================================

def step_model(position, velocity, acceleration, dt=DT):
    """三维二阶运动学预测模型。"""
    next_position = (
        position
        + velocity * dt
        + 0.5 * acceleration * dt * dt
    )

    next_velocity = velocity + acceleration * dt

    return next_position, next_velocity


def barrier_cost(point, other_points, obstacle):
    """障碍物和机间距离代价。"""
    cost = 0.0

    distance = np.linalg.norm(point - obstacle)

    if distance < OBSTACLE_WARNING:
        cost += W_OBS * (OBSTACLE_WARNING - distance) ** 2

    if distance < OBSTACLE_SAFE:
        cost += 1e5 * (OBSTACLE_SAFE - distance) ** 2

    for other in other_points:
        distance = np.linalg.norm(point - other)

        if distance < SEPARATION_WARNING:
            cost += W_SEP * (SEPARATION_WARNING - distance) ** 2

        if distance < MIN_COMMAND_SEP:
            cost += 1e5 * (MIN_COMMAND_SEP - distance) ** 2

    return cost


def predicted_value(
        index,
        acceleration,
        t_now,
        positions,
        velocities,
        center,
        radius):
    """计算一个预测步后的Lyapunov候选函数。"""
    position, velocity = step_model(
        positions[index],
        velocities[index],
        acceleration,
    )

    predicted_positions = positions + velocities * DT
    predicted_velocities = velocities.copy()

    predicted_positions[index] = position
    predicted_velocities[index] = velocity

    desired_p, desired_v, _, _ = desired_states(
        t_now + DT,
        center,
        radius,
    )

    error_p, error_v = graph_error(
        index,
        position,
        velocity,
        predicted_positions,
        predicted_velocities,
        desired_p,
        desired_v,
    )

    z2 = error_v + K1 * error_p

    return 0.5 * (
        np.dot(error_p, error_p)
        + np.dot(z2, z2)
    )


def mpc_cost(
        sequence,
        index,
        t_now,
        positions,
        velocities,
        center,
        radius,
        obstacle):
    """LMPC预测时域目标函数。"""
    controls = sequence.reshape(N_PRED, 3)

    position = positions[index].copy()
    velocity = velocities[index].copy()
    total = 0.0

    for step, acceleration in enumerate(controls, 1):
        position, velocity = step_model(
            position,
            velocity,
            acceleration,
        )

        predicted_positions = (
            positions + velocities * (step * DT)
        )
        predicted_velocities = velocities.copy()

        predicted_positions[index] = position
        predicted_velocities[index] = velocity

        desired_p, desired_v, _, _ = desired_states(
            t_now + step * DT,
            center,
            radius,
        )

        error_p, error_v = graph_error(
            index,
            position,
            velocity,
            predicted_positions,
            predicted_velocities,
            desired_p,
            desired_v,
        )

        other_points = np.delete(
            predicted_positions,
            index,
            axis=0,
        )

        total += (
            W_FORM * np.dot(error_p, error_p)
            + W_VEL * np.dot(error_v, error_v)
            + W_U * np.dot(acceleration, acceleration)
            + barrier_cost(position, other_points, obstacle)
        )

    return total


def solve_lmpc(
        index,
        t_now,
        positions,
        velocities,
        center,
        radius,
        obstacle,
        auxiliary,
        warm_start):
    """求解LMPC；失败或不满足条件时回退反步控制器。"""
    if warm_start is None:
        initial = np.tile(auxiliary, N_PRED)
    else:
        initial = warm_start

    bounds = [
        (-MAX_ACCEL, MAX_ACCEL)
        for _ in range(3 * N_PRED)
    ]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds*",
                category=RuntimeWarning,
            )

            result = minimize(
                mpc_cost,
                initial,
                args=(
                    index,
                    t_now,
                    positions,
                    velocities,
                    center,
                    radius,
                    obstacle,
                ),
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": MPC_MAXITER,
                    "ftol": MPC_FTOL,
                    "disp": False,
                },
            )

    except Exception:
        return auxiliary, False, None

    if (
        not result.success
        or result.x is None
        or not np.all(np.isfinite(result.x))
    ):
        return auxiliary, False, None

    candidate = clip_norm(
        result.x[:3],
        MAX_ACCEL,
    )

    candidate_value = predicted_value(
        index,
        candidate,
        t_now,
        positions,
        velocities,
        center,
        radius,
    )

    auxiliary_value = predicted_value(
        index,
        auxiliary,
        t_now,
        positions,
        velocities,
        center,
        radius,
    )

    candidate_position, _ = step_model(
        positions[index],
        velocities[index],
        candidate,
    )

    auxiliary_position, _ = step_model(
        positions[index],
        velocities[index],
        auxiliary,
    )

    safer = (
        np.linalg.norm(candidate_position - obstacle)
        >
        np.linalg.norm(auxiliary_position - obstacle) + 0.005
    )

    accepted = (
        candidate_value <= 1.05 * auxiliary_value + 1e-6
        or safer
    )

    if not accepted:
        return auxiliary, False, None

    shifted = np.concatenate((
        result.x[3:],
        result.x[-3:],
    ))

    return candidate, True, shifted


# =============================== 安全控制层 ================================

def outward_unit(delta, fallback):
    """返回向外方向，避免距离为零时除零。"""
    norm = np.linalg.norm(delta)

    if norm > 1e-9:
        return delta / norm

    return fallback


def safety_acceleration(index, acceleration, positions, obstacle):
    """对LMPC或反步控制量增加障碍与机间排斥。"""
    safe = np.array(acceleration, dtype=float)

    delta = positions[index] - obstacle
    distance = np.linalg.norm(delta)

    if distance < OBSTACLE_WARNING:
        safe += (
            6.0
            * (OBSTACLE_WARNING - distance)
            * outward_unit(
                delta,
                np.array([1.0, 0.0, 0.0]),
            )
        )

    for neighbor in range(3):
        if neighbor == index:
            continue

        delta = positions[index] - positions[neighbor]
        distance = np.linalg.norm(delta)

        if distance < SEPARATION_WARNING:
            fallback = (
                FORMATION_UNIT[index]
                - FORMATION_UNIT[neighbor]
            )

            safe += (
                7.0
                * (SEPARATION_WARNING - distance)
                * outward_unit(
                    delta,
                    outward_unit(
                        fallback,
                        np.array([1.0, 0.0, 0.0]),
                    ),
                )
            )

    return clip_norm(safe, MAX_ACCEL)


def limit_commands(
        positions,
        velocities,
        accelerations,
        lower,
        upper,
        obstacle):
    """生成最终位置、速度和加速度指令。"""
    position_commands = np.zeros((3, 3))
    velocity_commands = np.zeros((3, 3))
    acceleration_commands = np.zeros((3, 3))

    for index in range(3):
        velocity = clip_norm(
            velocities[index] + accelerations[index] * DT,
            MAX_SPEED,
        )

        point = positions[index] + velocity * DT

        # 障碍物安全距离
        delta = point - obstacle
        distance = np.linalg.norm(delta)

        if distance < OBSTACLE_SAFE:
            point = (
                obstacle
                + OBSTACLE_SAFE
                * outward_unit(
                    delta,
                    positions[index] - obstacle,
                )
            )

        # 飞行区域限制
        point = np.clip(point, lower, upper)

        # 单周期最大位移限制
        point = (
            positions[index]
            + clip_norm(
                point - positions[index],
                MAX_SPEED * DT,
            )
        )

        position_commands[index] = point

    # 最终机间指令距离保护
    for _ in range(2):
        for first in range(3):
            for second in range(first + 1, 3):
                delta = (
                    position_commands[second]
                    - position_commands[first]
                )

                distance = np.linalg.norm(delta)

                if distance < MIN_COMMAND_SEP:
                    fallback = (
                        FORMATION_UNIT[second]
                        - FORMATION_UNIT[first]
                    )

                    unit = outward_unit(
                        delta,
                        outward_unit(
                            fallback,
                            np.array([1.0, 0.0, 0.0]),
                        ),
                    )

                    correction = (
                        0.5
                        * (MIN_COMMAND_SEP - distance)
                        * unit
                    )

                    position_commands[first] -= correction
                    position_commands[second] += correction

        position_commands = np.clip(
            position_commands,
            lower,
            upper,
        )

        for index in range(3):
            position_commands[index] = (
                positions[index]
                + clip_norm(
                    position_commands[index] - positions[index],
                    MAX_SPEED * DT,
                )
            )

    # 保证位置、速度和加速度指令相互一致
    for index in range(3):
        velocity_commands[index] = (
            position_commands[index] - positions[index]
        ) / DT

        acceleration_commands[index] = (
            velocity_commands[index] - velocities[index]
        ) / DT

        acceleration_commands[index] = clip_norm(
            acceleration_commands[index],
            MAX_ACCEL,
        )

    return (
        position_commands,
        velocity_commands,
        acceleration_commands,
    )


def minimum_pair_distance(points):
    """计算三架无人机的最小两两距离。"""
    return min(
        np.linalg.norm(points[first] - points[second])
        for first in range(3)
        for second in range(first + 1, 3)
    )


def validate_positions(
        positions,
        previous=None,
        lower=None,
        upper=None):
    """检查Lighthouse位置、跳变、间距和飞行边界。"""
    if (
        positions.shape != (3, 3)
        or not np.all(np.isfinite(positions))
    ):
        raise RuntimeError(
            "Lighthouse position is invalid"
        )

    if minimum_pair_distance(positions) < CRITICAL_SEP:
        raise RuntimeError(
            "critical inter-drone distance"
        )

    if previous is not None:
        jumps = np.linalg.norm(
            positions - previous,
            axis=1,
        )

        if np.max(jumps) > MAX_POSITION_JUMP:
            raise RuntimeError(
                "Lighthouse position jump"
            )

    if lower is not None:
        margin = 0.10

        if (
            np.any(positions < lower - margin)
            or np.any(positions > upper + margin)
        ):
            raise RuntimeError(
                "drone left the configured flight box"
            )


# ================================ 数据保存 =================================

def save_log(rows):
    """保存CSV和NPZ飞行数据。"""
    if not rows:
        return

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = "lmpc_flight_%s.csv" % stamp
    npz_path = "lmpc_flight_%s.npz" % stamp

    header = [
        "t",
        "mode",
        "formation_radius",
        "leader_x",
        "leader_y",
        "leader_z",
    ]

    for cf_id in CF_IDS:
        header += [
            "cf%d_x" % cf_id,
            "cf%d_y" % cf_id,
            "cf%d_z" % cf_id,
            "cf%d_vx" % cf_id,
            "cf%d_vy" % cf_id,
            "cf%d_vz" % cf_id,
            "cf%d_error" % cf_id,
            "cf%d_lmpc" % cf_id,
        ]

    with open(csv_path, "w") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    np.savez(
        npz_path,
        data=np.asarray(rows, dtype=float),
        header=np.asarray(header),
    )

    print(
        "[done] saved %s and %s"
        % (csv_path, npz_path)
    )


# ================================ 自检函数 =================================

def self_test():
    """不连接无人机的基本几何自检。"""
    center = np.array([1.0, 0.5, 0.3])

    desired, _, _, leader = desired_states(
        0.0,
        center,
        FORMATION_R,
    )

    assert np.allclose(leader, center)

    assert np.allclose(
        np.linalg.norm(desired[0] - desired[1]),
        math.sqrt(3.0) * FORMATION_R,
        atol=1e-6,
    )

    assert (
        minimum_pair_distance(desired)
        > MIN_COMMAND_SEP
    )

    print("self-test passed")


# ================================ 主程序 ===================================

def read_positions(crazyflies):
    """读取Crazyswarm1位置反馈。"""
    return np.asarray(
        [cf.position() for cf in crazyflies],
        dtype=float,
    )


def main():
    if "--self-test" in sys.argv:
        self_test()
        return

    simulation = "--sim" in sys.argv

    from pycrazyswarm import Crazyswarm

    print(
        "[init] Crazyswarm1 %s"
        % (
            "simulation"
            if simulation
            else "real flight"
        )
    )

    if not simulation:
        print(
            "[init] waiting for the "
            "already-running crazyswarm_server..."
        )

    # --sim由Crazyswarm1自身解析
    swarm = Crazyswarm()
    time_helper = swarm.timeHelper

    by_id = {
        cf.id: cf
        for cf in swarm.allcfs.crazyflies
    }

    missing = [
        cf_id
        for cf_id in CF_IDS
        if cf_id not in by_id
    ]

    if missing:
        raise RuntimeError(
            "missing Crazyflie IDs: %s"
            % missing
        )

    crazyflies = [
        by_id[cf_id]
        for cf_id in CF_IDS
    ]

    airborne = False
    streaming = False
    rows = []

    ground_z = 0.0

    try:
        # 起飞前Lighthouse/仿真位置检查
        print(
            "[check] reading position estimates..."
        )

        samples = []

        for _ in range(10):
            samples.append(
                read_positions(crazyflies)
            )
            time_helper.sleep(0.10)

        initial = np.mean(samples, axis=0)

        validate_positions(initial)

        jitter = np.max(
            np.linalg.norm(
                np.asarray(samples) - initial,
                axis=2,
            )
        )

        if jitter > 0.06:
            raise RuntimeError(
                "position estimate is unstable: "
                "%.3f m" % jitter
            )

        if (
            minimum_pair_distance(initial)
            < MIN_START_SEP
        ):
            raise RuntimeError(
                "start positions are too close"
            )

        # 防止误执行实飞
        if not simulation:
            answer = input(
                "[check] type FLY to arm this mission: "
            ).strip()

            if answer != "FLY":
                print(
                    "[stop] real flight not armed"
                )
                return

        ground_z = float(
            np.mean(initial[:, 2])
        )

        takeoff_z = ground_z + TAKEOFF_DZ

        print(
            "[takeoff] target z = %.3f m"
            % takeoff_z
        )

        for cf in crazyflies:
            cf.takeoff(
                targetHeight=takeoff_z,
                duration=3.0,
            )

        airborne = True

        time_helper.sleep(4.0)

        positions = read_positions(crazyflies)

        validate_positions(positions)

        # 使用起飞后的实际中心作为任务中心
        mission_center = np.mean(
            positions,
            axis=0,
        )

        mission_center[2] = takeoff_z

        if OBSTACLE_WORLD is None:
            obstacle = (
                mission_center
                + OBSTACLE_OFFSET
            )
        else:
            obstacle = np.asarray(
                OBSTACLE_WORLD,
                dtype=float,
            )

        lower = (
            mission_center
            + np.array([
                -BOX_HALF_X,
                -BOX_HALF_Y,
                -BOX_DOWN,
            ])
        )

        upper = (
            mission_center
            + np.array([
                BOX_HALF_X,
                BOX_HALF_Y,
                BOX_UP,
            ])
        )

        print(
            "[mission] center   = %s"
            % np.round(mission_center, 3)
        )

        print(
            "[mission] obstacle = %s"
            % np.round(obstacle, 3)
        )

        print(
            "[mission] box low/high = %s / %s"
            % (
                np.round(lower, 3),
                np.round(upper, 3),
            )
        )

        velocities = np.zeros((3, 3))
        previous = positions.copy()

        radius = FORMATION_R

        obstacle_mode = False
        release_since = None

        warm_starts = [
            None,
            None,
            None,
        ]

        start_time = time_helper.time()
        last_measurement_time = start_time
        last_print = -10.0

        while True:
            loop_wall = time.monotonic()

            now = time_helper.time()
            t_now = float(now - start_time)

            if t_now >= MISSION_TIME:
                break

            # 读取真实位置或仿真位置
            positions = read_positions(
                crazyflies
            )

            validate_positions(
                positions,
                previous,
                lower,
                upper,
            )

            # 使用相邻两次真实位置计算速度并低通滤波
            measure_dt = float(
                np.clip(
                    now - last_measurement_time,
                    0.03,
                    0.50,
                )
            )

            measured_velocity = (
                positions - previous
            ) / measure_dt

            velocities = (
                0.65 * velocities
                + 0.35 * measured_velocity
            )

            previous = positions.copy()
            last_measurement_time = now

            # 已知障碍接近与离开判断
            leader_probe = leader_state(
                t_now,
                mission_center,
            )[0]

            leader_obstacle = np.linalg.norm(
                leader_probe - obstacle
            )

            if (
                not obstacle_mode
                and leader_obstacle
                < OBSTACLE_TRIGGER
            ):
                obstacle_mode = True
                release_since = None

                print(
                    "[mode] obstacle expansion"
                )

            elif obstacle_mode:
                if (
                    leader_obstacle
                    > OBSTACLE_RELEASE
                ):
                    if release_since is None:
                        release_since = now

                    if now - release_since >= 1.5:
                        obstacle_mode = False
                        release_since = None

                        print(
                            "[mode] obstacle released"
                        )
                else:
                    release_since = None

            # mode:
            # 0 = 基础三角形
            # 1 = 展开环绕
            # 2 = 收紧围捕
            # 3 = 障碍规避扩张
            if obstacle_mode:
                target_radius = OBSTACLE_R
                mode = 3

            elif t_now >= CAPTURE_AT:
                target_radius = CAPTURE_R
                mode = 2

            elif t_now >= ENCIRCLE_AT:
                target_radius = ENCIRCLE_R
                mode = 1

            else:
                target_radius = FORMATION_R
                mode = 0

            # 平滑改变编队尺寸
            radius_change = float(
                np.clip(
                    target_radius - radius,
                    -RADIUS_RATE * DT,
                    RADIUS_RATE * DT,
                )
            )

            radius += radius_change
            radius_dot = radius_change / DT

            desired_p, desired_v, desired_a, leader_p = (
                desired_states(
                    t_now,
                    mission_center,
                    radius,
                    radius_dot,
                )
            )

            accelerations = np.zeros((3, 3))
            solved = np.zeros(3, dtype=int)

            # 每架飞机独立计算LMPC
            for index in range(3):
                auxiliary, _, _ = backstepping(
                    index,
                    positions,
                    velocities,
                    desired_p,
                    desired_v,
                    desired_a,
                )

                acceleration, success, warm_starts[index] = (
                    solve_lmpc(
                        index,
                        t_now,
                        positions,
                        velocities,
                        mission_center,
                        radius,
                        obstacle,
                        auxiliary,
                        warm_starts[index],
                    )
                )

                accelerations[index] = (
                    safety_acceleration(
                        index,
                        acceleration,
                        positions,
                        obstacle,
                    )
                )

                solved[index] = int(success)

            # 最终安全指令
            (
                position_commands,
                velocity_commands,
                acceleration_commands,
            ) = limit_commands(
                positions,
                velocities,
                accelerations,
                lower,
                upper,
                obstacle,
            )

            if (
                minimum_pair_distance(
                    position_commands
                )
                < CRITICAL_SEP
            ):
                raise RuntimeError(
                    "safe command separation "
                    "could not be maintained"
                )

            # 发送低层全状态指令
            streaming = True

            for index, cf in enumerate(crazyflies):
                cf.cmdFullState(
                    position_commands[index],
                    velocity_commands[index],
                    acceleration_commands[index],
                    0.0,
                    np.zeros(3),
                )

            errors = np.linalg.norm(
                positions - desired_p,
                axis=1,
            )

            row = [
                t_now,
                mode,
                radius,
                leader_p[0],
                leader_p[1],
                leader_p[2],
            ]

            for index in range(3):
                row += list(positions[index])
                row += list(velocities[index])
                row += [
                    errors[index],
                    solved[index],
                ]

            rows.append(row)

            if t_now - last_print >= 2.0:
                print(
                    "[run] t=%5.1f  mode=%d  "
                    "err=%.3f  sep=%.3f  LMPC=%d/3"
                    % (
                        t_now,
                        mode,
                        float(np.mean(errors)),
                        minimum_pair_distance(
                            positions
                        ),
                        int(np.sum(solved)),
                    )
                )

                last_print = t_now

            # 仿真时间和实时时间分别处理
            if simulation:
                time_helper.sleep(DT)

            else:
                remaining = (
                    DT
                    - (
                        time.monotonic()
                        - loop_wall
                    )
                )

                if remaining > 0.0:
                    time_helper.sleep(remaining)

                elif t_now - last_print >= 1.0:
                    print(
                        "[warn] control loop overrun"
                    )

    except KeyboardInterrupt:
        print("[stop] user interrupted")

    except Exception as error:
        print("[stop] %s" % error)

    finally:
        if airborne:
            # cmdFullState结束后，必须先停止低层指令
            if streaming:
                for cf in crazyflies:
                    cf.notifySetpointsStop(
                        remainValidMillisecs=100
                    )

                time_helper.sleep(0.20)

            print("[land]")

            for cf in crazyflies:
                cf.land(
                    targetHeight=ground_z + LAND_DZ,
                    duration=3.0,
                )

            time_helper.sleep(4.0)

        save_log(rows)


if __name__ == "__main__":
    main()