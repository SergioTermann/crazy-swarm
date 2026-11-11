#!/usr/bin/env python3

from pycrazyswarm import Crazyswarm

import uav_trajectory

import numpy as np
import math
import time
import sys


# ============================================================
# 1. 全局实验参数
# ============================================================

TRAJECTORY_ID = 0

# ------------------------------------------------------------
# 房间尺寸
#
# 假定世界坐标原点位于房间地面中心：
#
# X: [-1.5, 1.5]
# Y: [-1.0, 1.0]
# Z: [ 0.0, 2.0]
# ------------------------------------------------------------

ROOM_MIN = np.array([
    -1.5,
    -1.0,
     0.0
])

ROOM_MAX = np.array([
     1.5,
     1.0,
     2.0
])


# ------------------------------------------------------------
# 给墙壁和天花板留下安全距离
# ------------------------------------------------------------

SAFE_MIN = np.array([
    -1.30,
    -0.80,
     0.05
])

SAFE_MAX = np.array([
     1.30,
     0.80,
     1.70
])


# ------------------------------------------------------------
# 飞行安全限制
#
# 第一阶段故意设置得比较保守
# ------------------------------------------------------------

MAX_SPEED = 0.80          # m/s
MAX_ACCELERATION = 1.20  # m/s^2


# ------------------------------------------------------------
# 起降参数
# ------------------------------------------------------------

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 2.5

LAND_HEIGHT = 0.05
LAND_DURATION = 2.5


# ------------------------------------------------------------
# 圆弧参数
#
# 半径从之前的 0.25 m 收缩到 0.20 m，
# 增加与 +X 墙面的安全距离。
# ------------------------------------------------------------

ARC_RADIUS = 0.20
ARC_DURATION = 4.0
ARC_SEGMENTS = 12


# ============================================================
# 2. 状态打印
# ============================================================

def report(timeHelper, cf, label):

    p = np.asarray(
        cf.position(),
        dtype=float
    )

    print(
        "[{:7.2f} s] {:<28s}"
        "x={:6.3f}, y={:6.3f}, z={:6.3f}".format(
            timeHelper.time(),
            label,
            p[0],
            p[1],
            p[2]
        )
    )


# ============================================================
# 3. 七次多项式 Hermite 插值
# ============================================================

def hermite7_coefficients(
    duration,
    p0,
    v0,
    a0,
    j0,
    p1,
    v1,
    a1,
    j1
):

    """
    构造七次多项式：

        p(t) =
        c0 +
        c1*t +
        c2*t^2 +
        ...
        c7*t^7

    同时满足：

    t = 0:
        position     = p0
        velocity     = v0
        acceleration = a0
        jerk         = j0

    t = T:
        position     = p1
        velocity     = v1
        acceleration = a1
        jerk         = j1

    这样可以保证轨迹段之间具有较高阶连续性。
    """

    T = float(duration)

    if T <= 0.0:
        raise ValueError(
            "Polynomial duration must be > 0."
        )

    c = np.zeros(
        8,
        dtype=float
    )

    # t = 0 时直接能够确定的四个系数
    c[0] = p0
    c[1] = v0
    c[2] = a0 / 2.0
    c[3] = j0 / 6.0

    # 求 c4 ~ c7
    A = np.array([
        [
            T**4,
            T**5,
            T**6,
            T**7
        ],
        [
            4*T**3,
            5*T**4,
            6*T**5,
            7*T**6
        ],
        [
            12*T**2,
            20*T**3,
            30*T**4,
            42*T**5
        ],
        [
            24*T,
            60*T**2,
            120*T**3,
            210*T**4
        ]
    ], dtype=float)

    b = np.array([
        p1 - (
            c[0]
            + c[1]*T
            + c[2]*T**2
            + c[3]*T**3
        ),

        v1 - (
            c[1]
            + 2*c[2]*T
            + 3*c[3]*T**2
        ),

        a1 - (
            2*c[2]
            + 6*c[3]*T
        ),

        j1 - (
            6*c[3]
        )

    ], dtype=float)

    c[4:8] = np.linalg.solve(
        A,
        b
    )

    return c


# ============================================================
# 4. 静止 -> 静止的平滑轨迹段
# ============================================================

def create_rest_to_rest_piece(
    start_position,
    end_position,
    duration,
    start_yaw,
    end_yaw
):

    """
    生成一个：

        静止
          ↓
        平滑运动
          ↓
        静止

    的七次多项式轨迹。

    起点和终点：

        velocity     = 0
        acceleration = 0
        jerk         = 0
    """

    p0 = np.asarray(
        start_position,
        dtype=float
    )

    p1 = np.asarray(
        end_position,
        dtype=float
    )

    px = hermite7_coefficients(
        duration,
        p0[0],
        0.0,
        0.0,
        0.0,
        p1[0],
        0.0,
        0.0,
        0.0
    )

    py = hermite7_coefficients(
        duration,
        p0[1],
        0.0,
        0.0,
        0.0,
        p1[1],
        0.0,
        0.0,
        0.0
    )

    pz = hermite7_coefficients(
        duration,
        p0[2],
        0.0,
        0.0,
        0.0,
        p1[2],
        0.0,
        0.0,
        0.0
    )

    pyaw = hermite7_coefficients(
        duration,
        start_yaw,
        0.0,
        0.0,
        0.0,
        end_yaw,
        0.0,
        0.0,
        0.0
    )

    return uav_trajectory.Polynomial4D(
        duration,
        px,
        py,
        pz,
        pyaw
    )


# ============================================================
# 5. 悬停轨迹段
# ============================================================

def create_hover_piece(
    position,
    duration,
    yaw
):

    position = np.asarray(
        position,
        dtype=float
    )

    def constant_poly(value):

        return np.array([
            value,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])

    return uav_trajectory.Polynomial4D(
        duration,

        constant_poly(
            position[0]
        ),

        constant_poly(
            position[1]
        ),

        constant_poly(
            position[2]
        ),

        constant_poly(
            yaw
        )
    )


# ============================================================
# 6. 七阶 smootherstep
# ============================================================

def smootherstep7(s):

    """
    七阶平滑函数：

    q(s) =
        35s^4
        -84s^5
        +70s^6
        -20s^7

    在 s=0 和 s=1 处：

        velocity     = 0
        acceleration = 0
        jerk         = 0
    """

    q = (
        35*s**4
        - 84*s**5
        + 70*s**6
        - 20*s**7
    )

    dq = (
        140*s**3
        - 420*s**4
        + 420*s**5
        - 140*s**6
    )

    ddq = (
        420*s**2
        - 1680*s**3
        + 2100*s**4
        - 840*s**5
    )

    dddq = (
        840*s
        - 5040*s**2
        + 8400*s**3
        - 4200*s**4
    )

    return (
        q,
        dq,
        ddq,
        dddq
    )


# ============================================================
# 7. 理想圆弧状态
# ============================================================

def arc_state(
    t,
    duration,
    radius
):

    """
    半圆掉头：

    起点：
        (1.0, 0.0, 0.5)

    圆心：
        (1.0, radius, 0.5)

    最远点：
        (1.0 + radius, radius, 0.5)

    终点：
        (1.0, 2*radius, 0.5)

    yaw:
        0 -> pi
    """

    T = float(duration)

    s = t / T

    s = min(
        max(
            s,
            0.0
        ),
        1.0
    )

    q, dq, ddq, dddq = smootherstep7(
        s
    )

    theta = (
        -math.pi / 2.0
        + math.pi * q
    )

    # 对时间的导数
    theta_dot = (
        math.pi
        * dq
        / T
    )

    theta_ddot = (
        math.pi
        * ddq
        / (T*T)
    )

    theta_dddot = (
        math.pi
        * dddq
        / (T*T*T)
    )


    cx = 1.0
    cy = radius
    z = 0.5


    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    x = (
        cx
        + radius * math.cos(theta)
    )

    y = (
        cy
        + radius * math.sin(theta)
    )


    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    vx = (
        -radius
        * math.sin(theta)
        * theta_dot
    )

    vy = (
        radius
        * math.cos(theta)
        * theta_dot
    )


    # --------------------------------------------------------
    # Acceleration
    # --------------------------------------------------------

    ax = -radius * (
        math.cos(theta)
        * theta_dot**2
        +
        math.sin(theta)
        * theta_ddot
    )

    ay = radius * (
        -math.sin(theta)
        * theta_dot**2
        +
        math.cos(theta)
        * theta_ddot
    )


    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    jx = (
        radius
        * math.sin(theta)
        * theta_dot**3

        - 3.0
        * radius
        * math.cos(theta)
        * theta_dot
        * theta_ddot

        - radius
        * math.sin(theta)
        * theta_dddot
    )

    jy = (
        -radius
        * math.cos(theta)
        * theta_dot**3

        - 3.0
        * radius
        * math.sin(theta)
        * theta_dot
        * theta_ddot

        + radius
        * math.cos(theta)
        * theta_dddot
    )


    position = np.array([
        x,
        y,
        z
    ])

    velocity = np.array([
        vx,
        vy,
        0.0
    ])

    acceleration = np.array([
        ax,
        ay,
        0.0
    ])

    jerk = np.array([
        jx,
        jy,
        0.0
    ])


    # 飞机机头沿圆弧切线方向旋转
    yaw = (
        theta
        + math.pi / 2.0
    )

    yaw_rate = theta_dot

    yaw_acc = theta_ddot

    yaw_jerk = theta_dddot


    return (
        position,
        velocity,
        acceleration,
        jerk,
        yaw,
        yaw_rate,
        yaw_acc,
        yaw_jerk
    )


# ============================================================
# 8. 创建半圆 Poly4D 轨迹
# ============================================================

def create_arc_pieces(
    duration,
    radius,
    segments
):

    pieces = []

    dt = (
        duration
        / float(segments)
    )

    for i in range(segments):

        t0 = i * dt
        t1 = (i + 1) * dt

        state0 = arc_state(
            t0,
            duration,
            radius
        )

        state1 = arc_state(
            t1,
            duration,
            radius
        )

        p0, v0, a0, j0 = (
            state0[0],
            state0[1],
            state0[2],
            state0[3]
        )

        p1, v1, a1, j1 = (
            state1[0],
            state1[1],
            state1[2],
            state1[3]
        )


        px = hermite7_coefficients(
            dt,
            p0[0],
            v0[0],
            a0[0],
            j0[0],
            p1[0],
            v1[0],
            a1[0],
            j1[0]
        )

        py = hermite7_coefficients(
            dt,
            p0[1],
            v0[1],
            a0[1],
            j0[1],
            p1[1],
            v1[1],
            a1[1],
            j1[1]
        )

        pz = hermite7_coefficients(
            dt,
            p0[2],
            v0[2],
            a0[2],
            j0[2],
            p1[2],
            v1[2],
            a1[2],
            j1[2]
        )


        pyaw = hermite7_coefficients(
            dt,

            state0[4],
            state0[5],
            state0[6],
            state0[7],

            state1[4],
            state1[5],
            state1[6],
            state1[7]
        )


        pieces.append(
            uav_trajectory.Polynomial4D(
                dt,
                px,
                py,
                pz,
                pyaw
            )
        )

    return pieces


# ============================================================
# 9. 处理 yaw ±pi 跳变
# ============================================================

def unwrap_yaw_near(
    target_yaw,
    reference_yaw
):

    delta = math.atan2(
        math.sin(
            target_yaw
            - reference_yaw
        ),
        math.cos(
            target_yaw
            - reference_yaw
        )
    )

    return (
        reference_yaw
        + delta
    )


# ============================================================
# 10. 构建完整任务轨迹
# ============================================================

def build_mission_trajectory():

    pieces = []

    phases = []


    # --------------------------------------------------------
    # 轨迹关键点
    # --------------------------------------------------------

    P0 = np.array([
        0.0,
        0.0,
        1.0
    ])

    P1 = np.array([
        0.5,
        0.0,
        1.0
    ])

    P2 = np.array([
        0.5,
        0.0,
        0.3
    ])

    P3 = np.array([
        0.5,
        0.0,
        0.5
    ])

    P4 = np.array([
        1.0,
        0.0,
        0.5
    ])

    P5 = np.array([
        1.0,
        2.0 * ARC_RADIUS,
        0.5
    ])

    P6 = np.array([
        0.0,
        0.0,
        0.5
    ])


    # ========================================================
    # PHASE 1
    # 前进0.5m
    # ========================================================

    duration = 2.5

    pieces.append(
        create_rest_to_rest_piece(
            P0,
            P1,
            duration,
            0.0,
            0.0
        )
    )

    phases.append(
        (
            "Forward 0.5 m",
            duration
        )
    )


    # ========================================================
    # PHASE 2
    # 从1.0m下降到0.3m
    # ========================================================

    duration = 2.5

    pieces.append(
        create_rest_to_rest_piece(
            P1,
            P2,
            duration,
            0.0,
            0.0
        )
    )

    phases.append(
        (
            "Descend to 0.3 m",
            duration
        )
    )


    # ========================================================
    # PHASE 3
    # 0.3m悬停3秒
    # ========================================================

    duration = 3.0

    pieces.append(
        create_hover_piece(
            P2,
            duration,
            0.0
        )
    )

    phases.append(
        (
            "Hover at 0.3 m",
            duration
        )
    )


    # ========================================================
    # PHASE 4
    # 上升到0.5m
    # ========================================================

    duration = 1.5

    pieces.append(
        create_rest_to_rest_piece(
            P2,
            P3,
            duration,
            0.0,
            0.0
        )
    )

    phases.append(
        (
            "Climb to 0.5 m",
            duration
        )
    )


    # ========================================================
    # PHASE 5
    # 再向前0.5m
    # ========================================================

    duration = 2.5

    pieces.append(
        create_rest_to_rest_piece(
            P3,
            P4,
            duration,
            0.0,
            0.0
        )
    )

    phases.append(
        (
            "Forward another 0.5 m",
            duration
        )
    )


    # ========================================================
    # PHASE 6
    # 半圆掉头
    # ========================================================

    arc_pieces = create_arc_pieces(
        duration=ARC_DURATION,
        radius=ARC_RADIUS,
        segments=ARC_SEGMENTS
    )

    pieces.extend(
        arc_pieces
    )

    phases.append(
        (
            "180-degree arc turn",
            ARC_DURATION
        )
    )


    # ========================================================
    # PHASE 7
    # 返回原点上空
    # ========================================================

    return_duration = 4.5

    start_yaw = math.pi

    raw_return_yaw = math.atan2(
        P6[1] - P5[1],
        P6[0] - P5[0]
    )

    return_yaw = unwrap_yaw_near(
        raw_return_yaw,
        start_yaw
    )

    pieces.append(
        create_rest_to_rest_piece(
            P5,
            P6,
            return_duration,
            start_yaw,
            return_yaw
        )
    )

    phases.append(
        (
            "Return above origin",
            return_duration
        )
    )


    # ========================================================
    # 创建Trajectory对象
    # ========================================================

    trajectory = (
        uav_trajectory.Trajectory()
    )

    trajectory.polynomials = pieces

    trajectory.duration = sum(
        piece.duration
        for piece in pieces
    )


    return (
        trajectory,
        phases
    )


# ============================================================
# 11. 轨迹安全审计
# ============================================================

def validate_trajectory(
    trajectory,
    sample_dt=0.01
):

    print("")
    print(
        "======================================"
    )

    print(
        " Pre-flight trajectory validation"
    )

    print(
        "======================================"
    )


    sample_count = int(
        math.ceil(
            trajectory.duration
            / sample_dt
        )
    ) + 1


    times = np.linspace(
        0.0,
        trajectory.duration,
        sample_count
    )


    min_position = np.array([
        np.inf,
        np.inf,
        np.inf
    ])

    max_position = np.array([
        -np.inf,
        -np.inf,
        -np.inf
    ])


    maximum_speed = 0.0
    maximum_acceleration = 0.0


    for t in times:

        state = trajectory.eval(
            float(t)
        )

        p = state.pos
        v = state.vel
        a = state.acc


        min_position = np.minimum(
            min_position,
            p
        )

        max_position = np.maximum(
            max_position,
            p
        )


        speed = np.linalg.norm(
            v
        )

        acceleration = np.linalg.norm(
            a
        )


        maximum_speed = max(
            maximum_speed,
            speed
        )

        maximum_acceleration = max(
            maximum_acceleration,
            acceleration
        )


        # ----------------------------------------------------
        # 空间边界检查
        # ----------------------------------------------------

        if np.any(
            p < SAFE_MIN
        ) or np.any(
            p > SAFE_MAX
        ):

            raise RuntimeError(
                "\nTrajectory safety violation!\n"
                "Time = {:.3f} s\n"
                "Position = {}\n"
                "Safe minimum = {}\n"
                "Safe maximum = {}\n".format(
                    t,
                    p,
                    SAFE_MIN,
                    SAFE_MAX
                )
            )


        # ----------------------------------------------------
        # 最大速度检查
        # ----------------------------------------------------

        if speed > MAX_SPEED:

            raise RuntimeError(
                "\nMaximum speed exceeded!\n"
                "Time = {:.3f} s\n"
                "Speed = {:.3f} m/s\n"
                "Limit = {:.3f} m/s\n".format(
                    t,
                    speed,
                    MAX_SPEED
                )
            )


        # ----------------------------------------------------
        # 最大加速度检查
        # ----------------------------------------------------

        if acceleration > MAX_ACCELERATION:

            raise RuntimeError(
                "\nMaximum acceleration exceeded!\n"
                "Time = {:.3f} s\n"
                "Acceleration = {:.3f} m/s^2\n"
                "Limit = {:.3f} m/s^2\n".format(
                    t,
                    acceleration,
                    MAX_ACCELERATION
                )
            )


    print(
        "Trajectory duration: "
        "{:.2f} s".format(
            trajectory.duration
        )
    )

    print(
        "Position minimum:",
        np.round(
            min_position,
            3
        )
    )

    print(
        "Position maximum:",
        np.round(
            max_position,
            3
        )
    )

    print(
        "Maximum speed: "
        "{:.3f} m/s".format(
            maximum_speed
        )
    )

    print(
        "Maximum acceleration: "
        "{:.3f} m/s^2".format(
            maximum_acceleration
        )
    )

    print(
        "Safety boundary check: PASS"
    )

    print(
        "Velocity check: PASS"
    )

    print(
        "Acceleration check: PASS"
    )


# ============================================================
# 12. 检查初始位置
# ============================================================

def validate_initial_position(cf):

    initial_position = np.asarray(
        cf.initialPosition,
        dtype=float
    )

    expected = np.array([
        0.0,
        0.0,
        0.0
    ])

    error = np.linalg.norm(
        initial_position
        - expected
    )

    print("")
    print(
        "Crazyflie initial position:",
        initial_position
    )


    if error > 0.05:

        raise RuntimeError(
            "\nInitial position is not close "
            "to [0, 0, 0].\n"
            "Current initialPosition = {}\n"
            "Please check crazyflies.yaml "
            "before flight.".format(
                initial_position
            )
        )


# ============================================================
# 13. 安全降落
# ============================================================

def safe_land(
    timeHelper,
    cf
):

    print("")
    print(
        "Attempting controlled safety landing..."
    )

    try:

        cf.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION
        )

        timeHelper.sleep(
            LAND_DURATION
            + 0.8
        )

        print(
            "Safety landing command completed."
        )

    except Exception as landing_error:

        print(
            "WARNING: controlled landing failed:"
        )

        print(
            landing_error
        )

        print(
            "Manual intervention may be required."
        )


# ============================================================
# 14. Vispy任务结束后保持30个现实秒
# ============================================================

def keep_visualization_open(
    timeHelper,
    seconds=30.0,
    refresh_rate=50.0
):

    print("")
    print(
        "Keeping visualization active for "
        "{:.0f} real seconds...".format(
            seconds
        )
    )


    module_name = (
        timeHelper
        .__class__
        .__module__
    )

    is_simulation = (
        "crazyflieSim"
        in module_name
    )


    # --------------------------------------------------------
    # 实机模式
    #
    # timeHelper.sleep 本身已经是真实时间。
    # --------------------------------------------------------

    if not is_simulation:

        timeHelper.sleep(
            seconds
        )

        return


    # --------------------------------------------------------
    # 仿真模式
    #
    # Crazyswarm仿真时间可能比现实时间快，
    # 所以一边刷新Vispy，一边使用monotonic校准现实时间。
    # --------------------------------------------------------

    dt = (
        1.0
        / refresh_rate
    )

    start_real = (
        time.monotonic()
    )

    next_tick = start_real


    while (
        time.monotonic()
        - start_real
        < seconds
    ):

        # 推进仿真并刷新Vispy
        timeHelper.sleep(
            dt
        )

        next_tick += dt

        remaining = (
            next_tick
            - time.monotonic()
        )

        if remaining > 0.0:

            time.sleep(
                remaining
            )


# ============================================================
# 15. 主任务
# ============================================================

def main():

    print("")
    print(
        "======================================"
    )

    print(
        " Crazyswarm Indoor Poly4D Mission"
    )

    print(
        "======================================"
    )


    # ========================================================
    # STEP 0
    # 在无人机初始化前先生成并检查轨迹
    # ========================================================

    trajectory, phases = (
        build_mission_trajectory()
    )

    validate_trajectory(
        trajectory
    )


    # ========================================================
    # 初始化Crazyswarm
    # ========================================================

    swarm = Crazyswarm()

    timeHelper = swarm.timeHelper

    allcfs = swarm.allcfs


    if len(
        allcfs.crazyflies
    ) != 1:

        raise RuntimeError(
            "\nThis experiment requires exactly "
            "ONE Crazyflie.\n"
            "Current number = {}\n".format(
                len(
                    allcfs.crazyflies
                )
            )
        )


    cf = allcfs.crazyflies[0]


    print("")
    print(
        "Crazyflie ID:",
        cf.id
    )


    validate_initial_position(
        cf
    )


    airborne = False


    try:

        # ====================================================
        # STEP 1
        # 上传完整任务轨迹
        # ====================================================

        print("")
        print(
            "STEP 1: Upload Poly4D trajectory"
        )


        cf.uploadTrajectory(
            TRAJECTORY_ID,
            0,
            trajectory
        )


        print(
            "Trajectory uploaded successfully."
        )


        # ====================================================
        # STEP 2
        # 起飞到1m
        # ====================================================

        print("")
        print(
            "STEP 2: Takeoff to 1.0 m"
        )


        cf.takeoff(
            targetHeight=TAKEOFF_HEIGHT,
            duration=TAKEOFF_DURATION
        )


        airborne = True


        timeHelper.sleep(
            TAKEOFF_DURATION
            + 0.5
        )


        report(
            timeHelper,
            cf,
            "Takeoff complete"
        )


        # ====================================================
        # STEP 3
        # 启动整条Poly4D轨迹
        # ====================================================

        print("")
        print(
            "STEP 3: Start uploaded trajectory"
        )


        cf.startTrajectory(
            TRAJECTORY_ID,

            timescale=1.0,

            reverse=False,

            # 我们使用绝对世界坐标
            relative=False
        )


        # ====================================================
        # 分阶段等待并打印状态
        #
        # 注意：
        # 这里没有再次发飞控命令。
        #
        # 无人机一直在执行同一条上传轨迹。
        # ====================================================

        for index, phase in enumerate(
            phases,
            start=1
        ):

            name = phase[0]
            duration = phase[1]

            print("")
            print(
                "Mission phase {}: {}".format(
                    index,
                    name
                )
            )


            timeHelper.sleep(
                duration
            )


            report(
                timeHelper,
                cf,
                name + " complete"
            )


        # 给轨迹末端一点时间余量
        timeHelper.sleep(
            0.2
        )


        report(
            timeHelper,
            cf,
            "Trajectory complete"
        )


        # ====================================================
        # STEP 4
        # 降落
        # ====================================================

        print("")
        print(
            "STEP 4: Landing"
        )


        cf.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION
        )


        timeHelper.sleep(
            LAND_DURATION
            + 0.5
        )


        airborne = False


        report(
            timeHelper,
            cf,
            "Landing complete"
        )


        print("")
        print(
            "======================================"
        )

        print(
            " MISSION COMPLETED SUCCESSFULLY"
        )

        print(
            "======================================"
        )


        # ====================================================
        # 任务完成后保持Vispy 30个现实秒
        # ====================================================

        keep_visualization_open(
            timeHelper,
            seconds=30.0,
            refresh_rate=50.0
        )


        print("")
        print(
            "Visualization hold finished."
        )

        print(
            "Program exits normally."
        )


    # ========================================================
    # Ctrl+C
    # ========================================================

    except KeyboardInterrupt:

        print("")
        print(
            "KeyboardInterrupt detected."
        )


        if airborne:

            safe_land(
                timeHelper,
                cf
            )


        print(
            "Mission interrupted safely."
        )


    # ========================================================
    # 其他异常
    # ========================================================

    except Exception as error:

        print("")
        print(
            "======================================"
        )

        print(
            " MISSION ERROR"
        )

        print(
            "======================================"
        )

        print(
            repr(error)
        )


        if airborne:

            safe_land(
                timeHelper,
                cf
            )


        raise


# ============================================================
# Python入口
# ============================================================

if __name__ == "__main__":

    main()
