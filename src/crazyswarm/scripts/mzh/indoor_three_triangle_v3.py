#!/usr/bin/env python3

"""
Crazyswarm Indoor Three-UAV Triangle Formation V3

任务：
1. 三架Crazyflie组成边长0.30m的等边三角形
2. 编队质心位于房间中心(0,0)
3. 同步起飞到1.0m
4. 编队质心直飞到(0.6,0.5,1.0)
5. 以(0.7,0.5,1.0)为圆心，半径0.10m绕行一圈
6. 编队保持固定等边三角形形状
7. 圆周结束后直线返回质心(0,0,1.0)
8. 同步降落
9. 开启碰撞规避和碰撞椭球显示
10. 任务完成后Vispy保持30个现实秒
"""

from pycrazyswarm import Crazyswarm
import uav_trajectory

import numpy as np
import itertools
import math


# ============================================================
# 从V1复用已经验证过的Poly4D数学工具
# ============================================================

from indoor_single_poly4d_v1 import (
    hermite7_coefficients,
    create_rest_to_rest_piece,
    keep_visualization_open
)


# ============================================================
# 1. 基本参数
# ============================================================

TRAJECTORY_ID = 1

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 3.0

LAND_HEIGHT = 0.05
LAND_DURATION = 3.0

POST_FLIGHT_DISPLAY_SECONDS = 2.0
# ============================================================
# 2. 等边三角形
# ============================================================

FORMATION_SIDE = 0.30

FORMATION_RADIUS = (
    FORMATION_SIDE
    / math.sqrt(3.0)
)


FORMATION_OFFSETS = {

    1: np.array([
        0.0,
        FORMATION_RADIUS,
        0.0
    ]),

    2: np.array([
        -FORMATION_SIDE / 2.0,
        -FORMATION_RADIUS / 2.0,
        0.0
    ]),

    3: np.array([
        FORMATION_SIDE / 2.0,
        -FORMATION_RADIUS / 2.0,
        0.0
    ])
}


EXPECTED_IDS = [
    1,
    2,
    3
]


# ============================================================
# 3. 房间安全区域
# ============================================================

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


# ============================================================
# 4. 圆周轨迹
# ============================================================

ORBIT_CENTER = np.array([
    0.70,
    0.50,
    1.00
])

ORBIT_RADIUS = 0.10

# 左侧作为圆周起点
ORBIT_START = np.array([
    ORBIT_CENTER[0] - ORBIT_RADIUS,
    ORBIT_CENTER[1],
    1.0
])

STRAIGHT_DURATION = 4.0

ORBIT_DURATION = 8.0

ORBIT_SEGMENTS = 20


# ============================================================
# 5. 安全限制
# ============================================================

MAX_SPEED = 0.60          # m/s
MAX_ACCELERATION = 0.80   # m/s^2

MIN_SAFE_DISTANCE = 0.22

FORMATION_TOLERANCE = 0.05


# ============================================================
# 6. 碰撞规避
# ============================================================

ENABLE_COLLISION_AVOIDANCE = True

SHOW_COLLISION_ELLIPSOIDS = True


COLLISION_RADII = np.array([
    0.10,
    0.10,
    0.30
])


# ============================================================
# 7. 七阶平滑时间函数
# ============================================================

def smootherstep7(s):

    q = (
        35.0 * s**4
        - 84.0 * s**5
        + 70.0 * s**6
        - 20.0 * s**7
    )

    dq = (
        140.0 * s**3
        - 420.0 * s**4
        + 420.0 * s**5
        - 140.0 * s**6
    )

    ddq = (
        420.0 * s**2
        - 1680.0 * s**3
        + 2100.0 * s**4
        - 840.0 * s**5
    )

    dddq = (
        840.0 * s
        - 5040.0 * s**2
        + 8400.0 * s**3
        - 4200.0 * s**4
    )

    return q, dq, ddq, dddq


# ============================================================
# 8. 理想圆周状态
# ============================================================

def circle_state(t):

    T = ORBIT_DURATION

    s = t / T

    s = min(
        max(s, 0.0),
        1.0
    )

    q, dq, ddq, dddq = smootherstep7(s)


    # --------------------------------------------------------
    # 从圆的左端点开始：
    #
    # theta = pi
    #
    # 逆时针旋转完整2*pi
    # --------------------------------------------------------

    theta = (
        math.pi
        + 2.0 * math.pi * q
    )


    theta_dot = (
        2.0
        * math.pi
        * dq
        / T
    )


    theta_ddot = (
        2.0
        * math.pi
        * ddq
        / (T * T)
    )


    theta_dddot = (
        2.0
        * math.pi
        * dddq
        / (T * T * T)
    )


    r = ORBIT_RADIUS

    cx = ORBIT_CENTER[0]
    cy = ORBIT_CENTER[1]


    # ========================================================
    # Position
    # ========================================================

    x = cx + r * math.cos(theta)

    y = cy + r * math.sin(theta)

    z = 1.0


    # ========================================================
    # Velocity
    # ========================================================

    vx = (
        -r
        * math.sin(theta)
        * theta_dot
    )

    vy = (
        r
        * math.cos(theta)
        * theta_dot
    )


    # ========================================================
    # Acceleration
    # ========================================================

    ax = -r * (
        math.cos(theta)
        * theta_dot**2

        +

        math.sin(theta)
        * theta_ddot
    )


    ay = r * (
        -math.sin(theta)
        * theta_dot**2

        +

        math.cos(theta)
        * theta_ddot
    )


    # ========================================================
    # Jerk
    # ========================================================

    jx = (
        r
        * math.sin(theta)
        * theta_dot**3

        - 3.0
        * r
        * math.cos(theta)
        * theta_dot
        * theta_ddot

        - r
        * math.sin(theta)
        * theta_dddot
    )


    jy = (
        -r
        * math.cos(theta)
        * theta_dot**3

        - 3.0
        * r
        * math.sin(theta)
        * theta_dot
        * theta_ddot

        + r
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


    return (
        position,
        velocity,
        acceleration,
        jerk
    )


# ============================================================
# 9. 创建完整圆周Poly4D
# ============================================================

def create_circle_pieces():

    pieces = []

    dt = (
        ORBIT_DURATION
        / ORBIT_SEGMENTS
    )


    for i in range(
        ORBIT_SEGMENTS
    ):

        t0 = i * dt
        t1 = (i + 1) * dt


        p0, v0, a0, j0 = (
            circle_state(t0)
        )

        p1, v1, a1, j1 = (
            circle_state(t1)
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


        # ----------------------------------------------------
        # 第一版保持所有Crazyflie yaw = 0
        #
        # 即：
        # 三角形编队方向固定，
        # 只让整个三角形平移做圆周运动。
        # ----------------------------------------------------

        pyaw = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])


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
# 10. 构建完整质心轨迹
# ============================================================

def build_centroid_trajectory():

    pieces = []

    phases = []


    origin_air = np.array([
        0.0,
        0.0,
        1.0
    ])


    # ========================================================
    # Phase 1
    # 原点 -> 圆周起点
    # ========================================================

    pieces.append(
        create_rest_to_rest_piece(
            origin_air,
            ORBIT_START,
            STRAIGHT_DURATION,
            0.0,
            0.0
        )
    )


    phases.append(
        (
            "Straight flight to orbit",
            STRAIGHT_DURATION
        )
    )


    # ========================================================
    # Phase 2
    # 完整圆周运动
    # ========================================================

    circle_pieces = (
        create_circle_pieces()
    )


    pieces.extend(
        circle_pieces
    )


    phases.append(
        (
            "One complete orbit",
            ORBIT_DURATION
        )
    )


    # ========================================================
    # Phase 3
    # 圆周起点 -> 原点
    # ========================================================

    pieces.append(
        create_rest_to_rest_piece(
            ORBIT_START,
            origin_air,
            STRAIGHT_DURATION,
            0.0,
            0.0
        )
    )


    phases.append(
        (
            "Straight return to origin",
            STRAIGHT_DURATION
        )
    )


    trajectory = (
        uav_trajectory.Trajectory()
    )


    trajectory.polynomials = pieces


    trajectory.duration = sum(
        p.duration
        for p in pieces
    )


    return trajectory, phases


# ============================================================
# 11. 检查等边三角形
# ============================================================

def validate_formation_geometry():

    print("")
    print(
        "=============================================="
    )

    print(
        " Triangle formation validation"
    )

    print(
        "=============================================="
    )


    minimum_distance = math.inf


    for id1, id2 in itertools.combinations(
        EXPECTED_IDS,
        2
    ):

        p1 = FORMATION_OFFSETS[
            id1
        ]

        p2 = FORMATION_OFFSETS[
            id2
        ]


        distance = np.linalg.norm(
            p1 - p2
        )


        minimum_distance = min(
            minimum_distance,
            distance
        )


        print(
            "CF{} - CF{} = {:.6f} m".format(
                id1,
                id2,
                distance
            )
        )


        if abs(
            distance
            - FORMATION_SIDE
        ) > 1e-5:

            raise RuntimeError(
                "Triangle geometry is incorrect."
            )


    if (
        minimum_distance
        < MIN_SAFE_DISTANCE
    ):

        raise RuntimeError(
            "Initial formation too close."
        )


    print(
        "Equilateral triangle check: PASS"
    )


# ============================================================
# 12. 完整轨迹安全检查
# ============================================================

def validate_trajectory(
    trajectory,
    dt=0.01
):

    print("")
    print(
        "=============================================="
    )

    print(
        " Three-UAV trajectory safety validation"
    )

    print(
        "=============================================="
    )


    times = np.arange(
        0.0,
        trajectory.duration + dt,
        dt
    )


    global_min = np.array([
        np.inf,
        np.inf,
        np.inf
    ])


    global_max = np.array([
        -np.inf,
        -np.inf,
        -np.inf
    ])


    max_speed = 0.0

    max_acceleration = 0.0


    for t in times:

        t_eval = min(
            float(t),
            trajectory.duration
        )


        state = trajectory.eval(
            t_eval
        )


        speed = np.linalg.norm(
            state.vel
        )


        acceleration = np.linalg.norm(
            state.acc
        )


        max_speed = max(
            max_speed,
            speed
        )


        max_acceleration = max(
            max_acceleration,
            acceleration
        )


        if speed > MAX_SPEED:

            raise RuntimeError(
                "Maximum speed exceeded: "
                "{:.3f} m/s".format(
                    speed
                )
            )


        if (
            acceleration
            > MAX_ACCELERATION
        ):

            raise RuntimeError(
                "Maximum acceleration exceeded: "
                "{:.3f} m/s^2".format(
                    acceleration
                )
            )


        # ====================================================
        # 检查每架Crazyflie的真实空间轨迹
        # ====================================================

        for cf_id in EXPECTED_IDS:

            position = (
                state.pos
                + FORMATION_OFFSETS[
                    cf_id
                ]
            )


            global_min = np.minimum(
                global_min,
                position
            )


            global_max = np.maximum(
                global_max,
                position
            )


            if (
                np.any(
                    position < SAFE_MIN
                )
                or
                np.any(
                    position > SAFE_MAX
                )
            ):

                raise RuntimeError(
                    "\nRoom safety boundary exceeded.\n"
                    "CF ID = {}\n"
                    "Time = {:.3f}s\n"
                    "Position = {}\n".format(
                        cf_id,
                        t,
                        position
                    )
                )


    print(
        "Trajectory duration: "
        "{:.2f} s".format(
            trajectory.duration
        )
    )


    print(
        "Swarm minimum position:",
        np.round(
            global_min,
            3
        )
    )


    print(
        "Swarm maximum position:",
        np.round(
            global_max,
            3
        )
    )


    print(
        "Maximum speed: "
        "{:.3f} m/s".format(
            max_speed
        )
    )


    print(
        "Maximum acceleration: "
        "{:.3f} m/s^2".format(
            max_acceleration
        )
    )


    print(
        "Room boundary check: PASS"
    )

    print(
        "Velocity check: PASS"
    )

    print(
        "Acceleration check: PASS"
    )


# ============================================================
# 13. 输出三机状态
# ============================================================

def report_swarm(
    timeHelper,
    cfs,
    label
):

    print("")
    print(
        "----------------------------------------------"
    )

    print(
        "[{:7.2f}s] {}".format(
            timeHelper.time(),
            label
        )
    )


    for cf in cfs:

        p = np.asarray(
            cf.position()
        )

        print(
            "CF{}: "
            "x={:7.3f} "
            "y={:7.3f} "
            "z={:7.3f}".format(
                cf.id,
                p[0],
                p[1],
                p[2]
            )
        )


# ============================================================
# 14. 实时检查机间距离
# ============================================================

def check_runtime_formation(
    cfs
):

    distances = []


    for cf1, cf2 in itertools.combinations(
        cfs,
        2
    ):

        p1 = np.asarray(
            cf1.position()
        )

        p2 = np.asarray(
            cf2.position()
        )


        d = np.linalg.norm(
            p1 - p2
        )


        distances.append(
            d
        )


        print(
            "CF{}-CF{} distance = "
            "{:.3f} m".format(
                cf1.id,
                cf2.id,
                d
            )
        )


        if d < MIN_SAFE_DISTANCE:

            raise RuntimeError(
                "Unsafe inter-UAV distance!"
            )


    average_distance = (
        np.mean(
            distances
        )
    )


    print(
        "Average triangle side = "
        "{:.3f} m".format(
            average_distance
        )
    )


    if abs(
        average_distance
        - FORMATION_SIDE
    ) > FORMATION_TOLERANCE:

        print(
            "WARNING: triangle formation "
            "has noticeable deformation."
        )


# ============================================================
# 15. 开启碰撞规避
# ============================================================

def enable_collision_avoidance(
    timeHelper,
    cfs
):

    if not ENABLE_COLLISION_AVOIDANCE:

        print(
            "Collision avoidance: DISABLED"
        )

        return


    print("")
    print(
        "Collision avoidance: ENABLED"
    )


    for index, cf in enumerate(
        cfs
    ):

        others = (
            cfs[:index]
            +
            cfs[index + 1:]
        )


        cf.enableCollisionAvoidance(
            others,
            COLLISION_RADII
        )


    if SHOW_COLLISION_ELLIPSOIDS:

        timeHelper.visualizer.showEllipsoids(
            0.95
            * COLLISION_RADII
        )


        print(
            "Collision ellipsoids: ENABLED"
        )


    print(
        "Collision avoidance configuration: PASS"
    )


# ============================================================
# 16. 安全降落
# ============================================================

def safe_land(
    timeHelper,
    allcfs
):

    print("")
    print(
        "Attempting safety landing..."
    )


    try:

        allcfs.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION
        )


        timeHelper.sleep(
            LAND_DURATION
            + 1.0
        )


    except Exception as error:

        print(
            "Safety landing failed:",
            repr(error)
        )


# ============================================================
# 17. 主程序
# ============================================================

def main():

    print("")
    print(
        "=============================================="
    )

    print(
        " Three-UAV Equilateral Triangle Mission V3"
    )

    print(
        "=============================================="
    )


    # ========================================================
    # 起飞前数学验证
    # ========================================================

    validate_formation_geometry()


    trajectory, phases = (
        build_centroid_trajectory()
    )


    validate_trajectory(
        trajectory
    )


    # ========================================================
    # 使用独立三机YAML
    # ========================================================

    swarm = Crazyswarm(
        crazyflies_yaml=(
            "../launch/crazyflies_three.yaml"
        )
    )


    timeHelper = (
        swarm.timeHelper
    )


    allcfs = (
        swarm.allcfs
    )


    cfs = sorted(
        allcfs.crazyflies,
        key=lambda cf: cf.id
    )


    actual_ids = [
        cf.id
        for cf in cfs
    ]


    if actual_ids != EXPECTED_IDS:

        raise RuntimeError(
            "Expected Crazyflie IDs {}, "
            "but found {}.".format(
                EXPECTED_IDS,
                actual_ids
            )
        )


    airborne = False


    try:

        # ====================================================
        # STEP 1
        # 上传同一条Poly4D轨迹
        # ====================================================

        print("")
        print(
            "STEP 1: Upload trajectory"
        )


        for cf in cfs:

            cf.uploadTrajectory(
                TRAJECTORY_ID,
                0,
                trajectory
            )


        print(
            "Trajectory upload: PASS"
        )


        # ====================================================
        # STEP 2
        # 三机同步起飞
        # ====================================================

        print("")
        print(
            "STEP 2: Synchronized takeoff"
        )


        allcfs.takeoff(
            targetHeight=TAKEOFF_HEIGHT,
            duration=TAKEOFF_DURATION
        )


        airborne = True


        timeHelper.sleep(
            TAKEOFF_DURATION
            + 0.5
        )


        report_swarm(
            timeHelper,
            cfs,
            "Takeoff complete"
        )


        check_runtime_formation(
            cfs
        )


        # ====================================================
        # STEP 3
        # 开启碰撞规避
        # ====================================================

        print("")
        print(
            "STEP 3: Collision avoidance"
        )


        enable_collision_avoidance(
            timeHelper,
            cfs
        )


        # ====================================================
        # STEP 4
        # 启动同步轨迹
        #
        # relative=True是关键：
        #
        # 三架机执行完全相同轨迹，
        # 但分别从自己的三角形顶点开始。
        # ====================================================

        print("")
        print(
            "STEP 4: Start synchronized mission"
        )


        allcfs.startTrajectory(
            TRAJECTORY_ID,
            timescale=1.0,
            reverse=False,
            relative=True
        )


        # ====================================================
        # 分阶段观察
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


            report_swarm(
                timeHelper,
                cfs,
                name + " complete"
            )


            check_runtime_formation(
                cfs
            )


        # ====================================================
        # STEP 5
        # 返回原点上空后同步降落
        # ====================================================

        print("")
        print(
            "STEP 5: Synchronized landing"
        )


        allcfs.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION
        )


        timeHelper.sleep(
            LAND_DURATION
            + 0.5
        )


        airborne = False


        report_swarm(
            timeHelper,
            cfs,
            "Landing complete"
        )


        print("")
        print(
            "=============================================="
        )

        print(
            " THREE-UAV MISSION COMPLETED SUCCESSFULLY"
        )

        print(
            "=============================================="
        )


        # ====================================================
        # Vispy继续保持30现实秒
        # ====================================================

        keep_visualization_open(
            timeHelper,
            seconds=POST_FLIGHT_DISPLAY_SECONDS,
            refresh_rate=50.0
        )


    except KeyboardInterrupt:

        print("")
        print(
            "KeyboardInterrupt detected."
        )


        if airborne:

            safe_land(
                timeHelper,
                allcfs
            )


    except Exception as error:

        print("")
        print(
            "MISSION ERROR:"
        )

        print(
            repr(error)
        )


        if airborne:

            safe_land(
                timeHelper,
                allcfs
            )


        raise


if __name__ == "__main__":

    main()
