#!/usr/bin/env python3

import numpy as np
import rospy
from pycrazyswarm import Crazyswarm


# ============================================================
# 参数设置
# ============================================================

# 飞行高度
takeoff_height = 0.5

# 降落目标高度
land_height = 0.02

# 认为已经接近地面的高度
land_stop_height = 0.08


# XYZ 三个方向的位置比例控制增益
Kpx = 0.5
Kpy = 0.5
Kpz = 0.5


# 最大速度
leader_max_vel = 0.10       # CF3 最大速度 0.1 m/s
follower_max_vel = 0.25     # CF4 / CF12 最大速度 0.25 m/s


# Leader 最终目标 x
leader_target_x = 2.0


# CF3 起飞完成后，等待 5 秒启动 followers
follower_delay_time = 3.0


# 第一编队保持时间
formation_hold_time = 3.0


# CF3 到达 x=2 后悬停时间
leader_hover_time = 35.0


# 到达目标的允许误差
position_tolerance = 0.05


# 控制频率
control_rate = 20


# Crazyflie ID
leader_id = 3
follower1_id = 4
follower2_id = 12


# ============================================================
# 速度限幅
# ============================================================

def limit_velocity(vel, max_vel):
    """
    对三维合速度进行限幅。

    保证:
        ||vel|| <= max_vel

    同时保持速度方向不变。
    """

    speed = np.linalg.norm(vel)

    if speed > max_vel and speed > 0.0:
        vel = vel / speed * max_vel

    return vel


# ============================================================
# 三维位置控制
# ============================================================

def position_control(cf, target, max_vel):
    """
    三维位置 P 控制器。

    target:
        [x_target, y_target, z_target]

    返回:
        三维位置误差范数
    """

    current = np.array(cf.position(), dtype=float)

    target = np.array(target, dtype=float)

    error = target - current


    # 三维位置误差 -> 三维速度
    vel = np.array([
        Kpx * error[0],
        Kpy * error[1],
        Kpz * error[2]
    ])


    # 合速度限幅
    vel = limit_velocity(
        vel,
        max_vel
    )


    # 世界坐标系速度控制
    cf.cmdVelocityWorld(
        vel=vel,
        yawRate=0
    )


    return np.linalg.norm(error)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 初始化 Crazyswarm
    # --------------------------------------------------------

    swarm = Crazyswarm()

    timeHelper = swarm.timeHelper

    allcfs = swarm.allcfs


    # --------------------------------------------------------
    # 根据 ID 找到三架飞机
    # --------------------------------------------------------

    cf3 = None
    cf4 = None
    cf12 = None


    for cf in allcfs.crazyflies:

        if cf.id == leader_id:
            cf3 = cf

        elif cf.id == follower1_id:
            cf4 = cf

        elif cf.id == follower2_id:
            cf12 = cf


    # 检查是否找到
    if cf3 is None:
        print("ERROR: Cannot find CF3")
        exit()

    if cf4 is None:
        print("ERROR: Cannot find CF4")
        exit()

    if cf12 is None:
        print("ERROR: Cannot find CF12")
        exit()


    print("================================")
    print("CF3  : Leader")
    print("CF4  : Follower 1")
    print("CF12 : Follower 2")
    print("================================")


    # ========================================================
    # 初始位置
    # ========================================================

    # 起飞前记录 CF3 的初始 XY
    # 起飞过程中保持这个位置
    initial_pos3 = np.array(
        cf3.position(),
        dtype=float
    )


    print("CF3 initial position:", initial_pos3)


    # ========================================================
    # Leader 状态
    # ========================================================

    # 0 = 起飞
    # 1 = 飞向 x=2
    # 2 = 到达目标，悬停

    leader_state = 0


    # CF3 起飞完成时的位置
    leader_takeoff_pos = None

    # CF3 起飞完成时间
    leader_takeoff_time = None

    # CF3 起飞完成后固定使用的 y
    leader_target_y = None

    # CF3 到达目标的时间
    leader_arrive_time = None


    # ========================================================
    # Follower 状态
    # ========================================================

    # 0 = 等待
    # 1 = 同时起飞
    # 2 = 第一编队
    # 3 = 第一编队保持5秒
    # 4 = 第二编队
    # 5 = 第二编队完成，持续动态跟随

    follower_state = 0


    # followers 起飞时的位置
    follower4_takeoff_target = None
    follower12_takeoff_target = None


    # 第一编队开始保持的时间
    formation1_hold_start = None


    # ========================================================
    # 降落相关变量
    # ========================================================

    landing = False


    # 降落时固定当前位置 XY
    land_target3 = None
    land_target4 = None
    land_target12 = None


    landed3 = False
    landed4 = False
    landed12 = False


    print("Mission start")


    # ========================================================
    # 主控制循环
    # ========================================================

    while not rospy.is_shutdown():

        timeHelper.sleepForRate(control_rate)

        current_time = timeHelper.time()


        # ====================================================
        # 判断是否应该开始降落
        # ====================================================

        if (
            leader_arrive_time is not None
            and not landing
        ):

            if (
                current_time - leader_arrive_time
                >= leader_hover_time
            ):

                print("================================")
                print("Leader hover 15 s finished")
                print("All Crazyflies start landing")
                print("================================")


                landing = True


                # --------------------------------------------
                # 记录开始降落时三架飞机的 XY
                #
                # 降落过程中保持 XY，不允许横向漂移
                # --------------------------------------------

                pos3 = np.array(cf3.position(), dtype=float)
                pos4 = np.array(cf4.position(), dtype=float)
                pos12 = np.array(cf12.position(), dtype=float)


                land_target3 = np.array([
                    pos3[0],
                    pos3[1],
                    land_height
                ])


                land_target4 = np.array([
                    pos4[0],
                    pos4[1],
                    land_height
                ])


                land_target12 = np.array([
                    pos12[0],
                    pos12[1],
                    land_height
                ])


        # ====================================================
        # 如果已经进入降落阶段
        # ====================================================

        if landing:

            # ------------------------------------------------
            # CF3 降落
            # ------------------------------------------------

            if not landed3:

                position_control(
                    cf3,
                    land_target3,
                    leader_max_vel
                )

                pos3 = np.array(
                    cf3.position(),
                    dtype=float
                )


                if pos3[2] < land_stop_height:

                    cf3.cmdStop()

                    landed3 = True

                    print("CF3 landed")


            # ------------------------------------------------
            # CF4 降落
            # ------------------------------------------------

            if not landed4:

                position_control(
                    cf4,
                    land_target4,
                    follower_max_vel
                )

                pos4 = np.array(
                    cf4.position(),
                    dtype=float
                )


                if pos4[2] < land_stop_height:

                    cf4.cmdStop()

                    landed4 = True

                    print("CF4 landed")


            # ------------------------------------------------
            # CF12 降落
            # ------------------------------------------------

            if not landed12:

                position_control(
                    cf12,
                    land_target12,
                    follower_max_vel
                )

                pos12 = np.array(
                    cf12.position(),
                    dtype=float
                )


                if pos12[2] < land_stop_height:

                    cf12.cmdStop()

                    landed12 = True

                    print("CF12 landed")


            # ------------------------------------------------
            # 全部降落完成
            # ------------------------------------------------

            if landed3 and landed4 and landed12:

                print("================================")
                print("Mission complete")
                print("All Crazyflies landed")
                print("================================")

                break


            # 已经进入降落状态后
            # 不再执行下面的正常飞行控制
            continue


        # ====================================================
        #
        # Leader 控制
        #
        # 注意：
        #
        # Leader 和 follower 是两个独立状态机。
        #
        # 无论 follower 正在起飞还是编队，
        # 这里每一个周期都会继续控制 CF3。
        #
        # ====================================================


        # ----------------------------------------------------
        # Leader State 0
        #
        # CF3 起飞
        # ----------------------------------------------------

        if leader_state == 0:

            leader_takeoff_target = np.array([
                initial_pos3[0],
                initial_pos3[1],
                takeoff_height
            ])


            err3 = position_control(
                cf3,
                leader_takeoff_target,
                leader_max_vel
            )


            if err3 < position_tolerance:

                # --------------------------------------------
                # 起飞完成后重新读取位置
                # --------------------------------------------

                leader_takeoff_pos = np.array(
                    cf3.position(),
                    dtype=float
                )


                # --------------------------------------------
                # 关键：
                #
                # 记录起飞完成后的 y
                #
                # 后面 CF3 飞 x=2 时，
                # 始终使用这个 y 作为目标
                # --------------------------------------------

                leader_target_y = leader_takeoff_pos[1]


                # 起飞完成时间
                leader_takeoff_time = current_time


                leader_state = 1


                print("================================")

                print(
                    "CF3 takeoff finished:",
                    leader_takeoff_pos
                )

                print(
                    "CF3 fixed target y:",
                    leader_target_y
                )

                print("CF3 starts moving to x = 2")

                print("================================")


        # ----------------------------------------------------
        # Leader State 1
        #
        # CF3 飞向 x = 2
        # ----------------------------------------------------

        elif leader_state == 1:

            leader_target = np.array([
                leader_target_x,
                leader_target_y,
                takeoff_height
            ])


            # --------------------------------------------
            # 关键：
            #
            # 即使 followers 此时正在：
            #
            # 起飞
            # 第一编队
            # 第二编队
            #
            # CF3 仍然每一个循环执行这里。
            # --------------------------------------------

            err3 = position_control(
                cf3,
                leader_target,
                leader_max_vel
            )


            # 到达完整三维目标
            if err3 < position_tolerance:

                leader_arrive_time = current_time

                leader_state = 2


                print("================================")
                print("CF3 reached destination")
                print("CF3 starts 15 s hover")
                print("================================")


        # ----------------------------------------------------
        # Leader State 2
        #
        # CF3 在 x=2 悬停
        # ----------------------------------------------------

        elif leader_state == 2:

            leader_target = np.array([
                leader_target_x,
                leader_target_y,
                takeoff_height
            ])


            # 持续 XYZ 三维闭环
            position_control(
                cf3,
                leader_target,
                leader_max_vel
            )


        # ====================================================
        #
        # Follower 控制
        #
        # ====================================================


        # ----------------------------------------------------
        # Follower State 0
        #
        # 等待 CF3 起飞完成 5 秒
        # ----------------------------------------------------

        if follower_state == 0:

            if leader_takeoff_time is not None:

                if (
                    current_time - leader_takeoff_time
                    >= follower_delay_time
                ):

                    # ----------------------------------------
                    # followers 开始起飞时
                    # 记录自己的 XY
                    # ----------------------------------------

                    pos4 = np.array(
                        cf4.position(),
                        dtype=float
                    )

                    pos12 = np.array(
                        cf12.position(),
                        dtype=float
                    )


                    follower4_takeoff_target = np.array([
                        pos4[0],
                        pos4[1],
                        takeoff_height
                    ])


                    follower12_takeoff_target = np.array([
                        pos12[0],
                        pos12[1],
                        takeoff_height
                    ])


                    follower_state = 1


                    print("================================")
                    print("5 s after CF3 takeoff")
                    print("CF4 and CF12 start takeoff")
                    print("================================")


        # ----------------------------------------------------
        # Follower State 1
        #
        # CF4 / CF12 同时起飞
        # ----------------------------------------------------

        elif follower_state == 1:

            err4 = position_control(
                cf4,
                follower4_takeoff_target,
                follower_max_vel
            )


            err12 = position_control(
                cf12,
                follower12_takeoff_target,
                follower_max_vel
            )


            if (
                err4 < position_tolerance
                and
                err12 < position_tolerance
            ):

                follower_state = 2


                print("================================")
                print("CF4 and CF12 takeoff finished")
                print("Followers move to formation 1")
                print("================================")


        # ----------------------------------------------------
        # Follower State 2
        #
        # 第一编队
        #
        # 动态读取 Leader 当前位置
        # ----------------------------------------------------

        elif follower_state == 2:

            leader_pos = np.array(
                cf3.position(),
                dtype=float
            )


            # CF4:
            # leader.x - 0.6
            # leader.y + 0.6
            # z = 0.5

            target4 = np.array([
                leader_pos[0] - 0.6,
                leader_pos[1] + 0.6,
                takeoff_height
            ])


            # CF12:
            # leader.x - 0.6
            # leader.y - 1.0
            # z = 0.5

            target12 = np.array([
                leader_pos[0] - 0.6,
                leader_pos[1] - 0.6,
                takeoff_height
            ])


            err4 = position_control(
                cf4,
                target4,
                follower_max_vel
            )


            err12 = position_control(
                cf12,
                target12,
                follower_max_vel
            )


            if (
                err4 < position_tolerance
                and
                err12 < position_tolerance
            ):

                formation1_hold_start = current_time

                follower_state = 3


                print("================================")
                print("Formation 1 reached")
                print("Hold formation 1 for 5 seconds")
                print("================================")


        # ----------------------------------------------------
        # Follower State 3
        #
        # 第一编队动态保持 5 秒
        # ----------------------------------------------------

        elif follower_state == 3:

            # --------------------------------------------
            # 这里仍然实时读取 leader
            #
            # 所以不是固定悬停点，
            # 而是动态保持相对编队。
            # --------------------------------------------

            leader_pos = np.array(
                cf3.position(),
                dtype=float
            )


            target4 = np.array([
                leader_pos[0] - 0.6,
                leader_pos[1] + 0.6,
                takeoff_height
            ])


            target12 = np.array([
                leader_pos[0] - 0.6,
                leader_pos[1] - 0.6,
                takeoff_height
            ])


            position_control(
                cf4,
                target4,
                follower_max_vel
            )


            position_control(
                cf12,
                target12,
                follower_max_vel
            )


            if (
                current_time - formation1_hold_start
                >= formation_hold_time
            ):

                follower_state = 4


                print("================================")
                print("Formation 1 hold finished")
                print("Followers move to formation 2")
                print("================================")


        # ----------------------------------------------------
        # Follower State 4
        #
        # 第二编队
        # ----------------------------------------------------

        elif follower_state == 4:

            leader_pos = np.array(
                cf3.position(),
                dtype=float
            )


            # CF4:
            #
            # leader.x - 0.3
            # leader.y + 0.3

            target4 = np.array([
                leader_pos[0] - 0.3,
                leader_pos[1] + 0.3,
                takeoff_height
            ])


            # CF12:
            #
            # leader.x - 0.3
            # leader.y - 0.3

            target12 = np.array([
                leader_pos[0] - 0.3,
                leader_pos[1] - 0.3,
                takeoff_height
            ])


            err4 = position_control(
                cf4,
                target4,
                follower_max_vel
            )


            err12 = position_control(
                cf12,
                target12,
                follower_max_vel
            )


            if (
                err4 < position_tolerance
                and
                err12 < position_tolerance
            ):

                follower_state = 5


                print("================================")
                print("Formation 2 reached")
                print("Followers continue tracking CF3")
                print("================================")


        # ----------------------------------------------------
        # Follower State 5
        #
        # 第二编队完成后
        # 持续动态跟随 Leader
        # ----------------------------------------------------

        elif follower_state == 5:

            leader_pos = np.array(
                cf3.position(),
                dtype=float
            )


            target4 = np.array([
                leader_pos[0] - 0.3,
                leader_pos[1] + 0.3,
                takeoff_height
            ])


            target12 = np.array([
                leader_pos[0] - 0.3,
                leader_pos[1] - 0.3,
                takeoff_height
            ])


            position_control(
                cf4,
                target4,
                follower_max_vel
            )


            position_control(
                cf12,
                target12,
                follower_max_vel
            )
