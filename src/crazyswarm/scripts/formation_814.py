#!/usr/bin/env python3

### 功能介绍 : 实现论文中二阶固定时间分布式观测器 + 二阶固定时间编队控制协议(纯仿真)
### 一致性协议作为加速度控制指令, 积分得到速度控制指令作用于一阶积分器模型
### 采用跟随者期望状态二阶分布式观测器
### 通过二阶分布式观测器来计算每个跟随者的期望速度和跟随者期望位置

### 领导-跟随编队, 领导者的参考轨迹为圆形
### 设置领导者运动一圈的时间为T, 用 2*pi/T 计算角速率w
### 领导者轨迹生成为 : Px = xcirc + R * cos(wt), Py = ycirc - R * sin(wt)
### 设置集群内包含 : 1个领导者 + 4个跟随者

### 拓扑矩阵中, wij = 1 表示 节点 i 能获取 节点 j 的状态信息

import numpy as np
from pycrazyswarm import Crazyswarm

# 起飞降落参数
takeoff_height = 0.5
land_height = 0.02

takeoff_duration = 2.0
land_duration = 2.0

Kpz = 0.2

total_time_hold = 2.0

# 三角形目标位置
target_postion_T = np.array([[0.7,-0.4,0.5],  # 1号
                            [0,-0.6,0.5],  # 2号
                            [0,0.2,0.5],  # 3号
                            [-0.7,0.4,0.5],  # 4号
                            [0.7,0.0,0.5],  # 5号
                            [1.3,-0.2,0.5],  # 7号
                            [-0.7,-0.8,0.5],  # 9号
                            [-0.7,-0.2,0.5]])  # 10号

# V形目标位置
target_postion_V = np.array([[0.7,-0.4,0.5],  # 1号
                            [0,-0.6,0.5],  # 2号
                            [0,0.2,0.5],  # 3号
                            [-0.7,0.4,0.5],  # 4号
                            [0.7,0.0,0.5],  # 5号
                            [1.3,-0.2,0.5],  # 7号
                            [-0.7,-0.8,0.5],  # 9号
                            [-1.3,0.6,0.5]])  # 10号

# 圆形目标位置
target_postion_C = np.array([[0.7,-0.7,0.5],  # 1号
                            [0,-1.0,0.5],  # 2号
                            [0,1.0,0.5],  # 3号
                            [-0.7,0.7,0.5],  # 4号
                            [0.7,0.7,0.5],  # 5号
                            [1.0,0.0,0.5],  # 7号
                            [-0.7,-0.7,0.5],  # 9号
                            [-1.0,0.0,0.5]])  # 10号

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # 控制集群全部无人机起飞
    flag_all_uav_takeoff = False
    for cf in allcfs.crazyflies:
        cf.flag_takeoff = False
    while flag_all_uav_takeoff == False:
        timeHelper.sleepForRate(10)
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, Kpz * (takeoff_height - cf.current_uav_pos[2])]), yawRate = 0)
            if cf.current_uav_pos[2] > takeoff_height - 0.1 and cf.current_uav_pos[2] < takeoff_height + 0.1:
                cf.flag_takeoff = True
  
        cnt_uav_takeoff = 0
        for cf in allcfs.crazyflies:
            if cf.flag_takeoff == True:
                cnt_uav_takeoff = cnt_uav_takeoff + 1
        if cnt_uav_takeoff == len(allcfs.crazyflies):
            flag_all_uav_takeoff = True
    
    # 在跟随者期望速度观测器收敛后, 集群悬停等待二阶观测器中跟随者期望位置观测器收敛
    time_stamp_begin = timeHelper.time()
    time_stamp_last = timeHelper.time()
    time_stamp_now = timeHelper.time()

    while time_stamp_now - time_stamp_begin <= total_time_hold:
        timeHelper.sleepForRate(10)
        time_stamp_now = timeHelper.time()
        for cf in allcfs.crazyflies:
            # 控制跟随者无人机悬停
            cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, 0.0]), yawRate = 0)
        
    # 三角形
    flag_all_uav_ready_T = False
    time_stamp_begin = timeHelper.time()
    time_stamp_now = timeHelper.time()
    while flag_all_uav_ready_T == False:
        timeHelper.sleepForRate(10)
        time_stamp_now = timeHelper.time()
        id_cnt = 0
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            bia = target_postion_T[id_cnt]-cf.current_uav_pos
            cf.cmdVelocityWorld(vel = Kpz * bia, yawRate = 0)
            id_cnt = id_cnt + 1
        if time_stamp_now - time_stamp_begin >= 8:
            flag_all_uav_ready_T = True
            
    # 圆形
    flag_all_uav_ready_C = False
    time_stamp_begin = timeHelper.time()
    time_stamp_now = timeHelper.time()
    while flag_all_uav_ready_C == False:
        timeHelper.sleepForRate(10)
        time_stamp_now = timeHelper.time()
        id_cnt = 0
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            bia = target_postion_C[id_cnt]-cf.current_uav_pos
            cf.cmdVelocityWorld(vel = Kpz * bia, yawRate = 0)
            # print(0.5*bia)
            id_cnt = id_cnt + 1
        if time_stamp_now - time_stamp_begin >= 8:
            flag_all_uav_ready_C = True

    # 控制集群全部无人机降落
    flag_all_uav_land = False
    for cf in allcfs.crazyflies:
        cf.flag_uav_land = False
    while flag_all_uav_land == False:
        timeHelper.sleepForRate(10)
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            if cf.current_uav_pos[2] < 0.1:
                cf.flag_uav_land = True
                cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, 0.0]), yawRate = 0)
            else:
                cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, Kpz * (land_height - cf.current_uav_pos[2])]), yawRate = 0)
        cnt_uav_land = 0
        for cf in allcfs.crazyflies:
            if cf.flag_uav_land == True:
                cnt_uav_land = cnt_uav_land + 1
        if cnt_uav_land == len(allcfs.crazyflies):
            flag_all_uav_land = True
    
    # 电机停转
    for cf in allcfs.crazyflies:
        cf.cmdStop()
