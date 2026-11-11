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

Kpz = 0.5

total_time_hold = 2.0

# 三角形位置
target_postion_T = np.array([[0.5,0.0,takeoff_height],  # 1号
                          [-0.25,-0.43,takeoff_height],  # 2号
                          [-0.25,0.43,takeoff_height]]  # 3号
                            )

goal_postion = np.array([1.35,0.2,takeoff_height])
obs_position = np.array([0.0,0.0,takeoff_height])

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

    time_stamp_begin = timeHelper.time()
    time_stamp_now = timeHelper.time()
    arrive_flag = False
    while arrive_flag == False \
            and time_stamp_now - time_stamp_begin <= 20:
        timeHelper.sleepForRate(10)
        time_stamp_now = timeHelper.time()
        
        # 实时计算3架飞机中心位置
        center = np.array([0.0,0.0,0.0])
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            center += cf.current_uav_pos
        center = center/3.0

        # 目标点吸引
        attr_dir = goal_postion - center 
        norm = np.linalg.norm(attr_dir)
        att_f = min(0.5,norm)
        att = att_f*attr_dir/norm
        
        # 障碍物排斥
        feq_dir = center - obs_position
        obs_norm = np.linalg.norm(feq_dir)
        feq = np.array([0.0,0.0,0.0])
        if obs_norm < 1.0:
            obs_norm_reverse = 1.0/obs_norm
            feq_f = min(0.5,obs_norm_reverse)
            feq = feq_f*feq_dir/obs_norm

        # 编队
        id_cnt = 0
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            bia = center + target_postion_T[id_cnt] - cf.current_uav_pos
            desired_velocity = Kpz * bia + att + feq
            cf.cmdVelocityWorld(vel = np.array([desired_velocity[0],
                                                desired_velocity[1],
                                                Kpz * (takeoff_height - cf.current_uav_pos[2])]), yawRate = 0)
            id_cnt = id_cnt + 1
        
        if np.linalg.norm(center - goal_postion) <= 0.2:
            arrive_flag = True
    
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
