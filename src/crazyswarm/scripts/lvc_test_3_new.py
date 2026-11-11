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
from fractions import Fraction

import rospy
from crazyswarm.msg import GenericLogData


# 定义 s_func 函数
def s_func(x):
    """符号函数，输入向量，返回单位向量或零向量"""
    nn = np.linalg.norm(x)
    if nn == 0:
        return np.zeros_like(x)
    else:
        return x / nn
# 定义四元数辅助函数
def quatnormalize(q):
    """四元数归一化"""
    return q / np.linalg.norm(q)

def quatconj(q):
    """四元数共轭"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quatmultiply(q1, q2):
    """四元数乘法"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def operator_x(v):
    """向量叉乘矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def operator_R(q):
    """四元数转旋转矩阵（使用提供的公式）"""
    q0 = q[0]  # 标量部分
    qv = q[1:4]  # 向量部分
    # 使用提供的公式计算旋转矩阵
    R = (q0 ** 2 - np.dot(qv, qv)) * np.eye(3) + \
        2 * np.outer(qv, qv) - \
        2 * q0 * operator_x(qv)

    return R

# 起飞降落参数
takeoff_height = 0.5
land_height = 0.02

takeoff_duration = 2.0
land_duration = 2.0

# control parameter
Kpz = 0.5
total_time_hold = 100.0

k_gama = 0.1 #1.0
k_a = 0.2 #0.8 #4
k_p = 0.2 #1 #8
k_v = 0.2 #1 #8
k_s = 0.1 #0.8 #4
k_eta = 0.1 #0.8 #4
l_p = 0.1 #1.0
l_v = 0.1 #1.0
g = 9.8

dt = 0.1


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    #rate = rospy.Rate(10)

    # 为每个crazyflie对象添加新属性
    for i, cf in enumerate(allcfs.crazyflies):
        cf.a_hat = np.zeros(3)  # 估计的加速度，3维向量
        cf.v_hat = np.zeros(3)  # 估计的速度，3维向量
        cf.v_hat_dot = np.zeros(3)
        cf.p_hat = np.zeros(3)  # 估计的位置，3维向量
        cf.p_hat_dot = np.zeros(3)
        cf.p_eb = np.zeros(3)
        cf.v_eb = np.zeros(3)
        cf.p_et = np.zeros(3)
        cf.v_et = np.zeros(3)
        cf.w = np.zeros(3)
        cf.Q = np.array([1.0,0.0,0.0,0.0])


        cf.gama = np.zeros(3)  # 3维向量
        cf.gama_dot = np.zeros(3)  # 3维向量

        cf.eta = np.zeros(3)  # 扰动估计，3维向量
        cf.eta_dot = np.zeros(3)  # 扰动估计导数，3维向量
        cf.eta_dot2 = np.zeros(3)  # 扰动估计二阶导数，3维向量
        cf.u = np.zeros(3)  # 控制输入，3维向量
        cf.T = 0.0  # 推力标量
        cf.Qc = np.zeros(4)  # 四元数命令，4维向量
        cf.delta = np.zeros(3)  # 偏移量，3维向量
        cf.m = 0.03  # 无人机质量（kg）

        e_3 = np.array([0, 0, 1])

        if i == 0:  # UAVs(1)
            cf.delta = np.array([2, 2, 0])
        elif i == 1:  # UAVs(2)
            cf.delta = np.array([2, -2, 0])
        elif i == 2:  # UAVs(3)
            cf.delta = np.array([-2, -2, 0])
        elif i == 3:  # UAVs(4)
            cf.delta = np.array([-2, 2, 0])
        else:
            # 如果有多于4架无人机，为额外的无人机设置默认值
            cf.delta = np.zeros(3)


    # 控制集群全部无人机起飞
    flag_all_uav_takeoff = False
    for cf in allcfs.crazyflies:
        cf.flag_takeoff = False
    while flag_all_uav_takeoff == False:
        #rate.sleep()
        timeHelper.sleepForRate(10)
        for cf in allcfs.crazyflies:
            cf.current_uav_pos = cf.position()
            if cf.current_uav_pos[2] > takeoff_height - 0.1:
                cf.flag_takeoff = True
                cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, 0.0]), yawRate = 0)
            else:
                cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, Kpz * (takeoff_height - cf.current_uav_pos[2])]), yawRate = 0)
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

    # 连接矩阵（实际应用中应该根据无人机之间的通信关系定义）
    N = len(allcfs.crazyflies)
    d = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    m = np.array([
        [ 1, 0, 0, 0],
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ])

    # 创建位置记录矩阵，形状为 (T, N, 3)
    position_matrix = np.zeros((5000, N, 3))

    t_step = 0
    while time_stamp_now - time_stamp_begin <= total_time_hold:
        #rate.sleep()
        timeHelper.sleepForRate(10)
        time_stamp_now = timeHelper.time()
        # 计算当前时间步数
        #t_step = int((time_stamp_now - time_stamp_begin) / dt)
        t_step += 1

        # 计算参考轨迹
        pr = np.array([
            0.5 * np.cos(0.2 * dt * t_step),
            0.5 * np.sin(0.2 * dt * t_step),
            0.1*dt * t_step
        ])

        pr_dot = np.array([
            -0.1*np.sin(0.2 * dt * t_step),
            0.1*np.cos(0.2 * dt * t_step),
            0.1
        ])

        pr_dot2 = np.array([
            -0.02 * np.cos(0.2 * dt * t_step),
            -0.02 * np.sin(0.2 * dt * t_step),
            0
        ])

        for i, cf in enumerate(allcfs.crazyflies):
            F = np.diag([
                np.tanh(cf.gama_dot[0]) * cf.gama_dot[0],
                np.tanh(cf.gama_dot[1]) * cf.gama_dot[1],
                np.tanh(cf.gama_dot[2]) * cf.gama_dot[2]
            ])
            print(cf.gama)
            F1 = k_gama * np.diag([
                1/(1 - np.tanh(cf.gama[0])**2),
                1/(1 - np.tanh(cf.gama[1])**2),
                1/(1 - np.tanh(cf.gama[2])**2)
            ])

            # 计算 Ma, Mv, Mp
            Ma = np.zeros(3)
            Mv = np.zeros(3)
            Mp = np.zeros(3)

            for j in range(N):
                other_cf = allcfs.crazyflies[j]
                Ma += m[i, j] * (other_cf.a_hat - pr_dot2)
                Mv += m[i, j] * (other_cf.v_hat - pr_dot)
                Mp += m[i, j] * (other_cf.p_hat - pr)
            # 更新 gama_dot 和 gama
            cf.gama_dot = -np.dot(F1 , (-k_a * Ma - k_s * s_func(Mv)))
            cf.gama += cf.gama_dot * dt

            # 更新 a_hat,v_hat 和 p_hat
            cf.a_hat = k_gama * np.tanh(cf.gama)
            cf.v_hat_dot = cf.a_hat - k_v*Mv
            cf.v_hat += cf.v_hat_dot * dt
            cf.p_hat_dot = cf.v_hat - k_p*Mp
            cf.p_hat += cf.p_hat_dot * dt

            # 对于第一个无人机，可以设置特殊值（可选）
            if i == 0:
                # 示例：设置第一个无人机为圆形轨迹
                cf.a_hat = pr_dot2
                cf.v_hat = pr_dot
                cf.p_hat = pr

            cf.current_uav_pos = cf.position()
            cf.current_uav_vel = cf.velocity()

            # p_eb, v_eb
            cf.p_eb = cf.current_uav_pos - cf.p_hat - cf.delta
            cf.v_eb = cf.current_uav_vel - cf.v_hat
            cf.p_et = cf.p_eb - cf.eta
            cf.v_et = cf.v_eb - cf.eta_dot

            # eta
            cf.eta_dot2 = -k_eta * (np.tanh(cf.eta + cf.eta_dot)) + \
                               np.tanh(l_p * cf.p_et + l_v * cf.v_et)
            cf.eta_dot += cf.eta_dot2 * dt
            cf.eta += cf.eta_dot * dt

            # u
            cf.u = g * e_3 + cf.a_hat - k_eta * (np.tanh(cf.eta_dot + cf.eta))

            # T
            cf.T = cf.m * np.linalg.norm(cf.u)

            # Qc
            norm_u = np.linalg.norm(cf.u)
            cf.Qc[0] = np.sqrt(0.5 + 0.5 * (g - cf.u[2]) / norm_u)
            cf.Qc[1:4] = (1 / (2 * norm_u * cf.Qc[0])) * np.array([cf.u[1], -cf.u[0], 0])

            # 新增：四元数控制和动力学更新
            # 四元数归一化和误差计算
            Qc_quat = quatnormalize(cf.Qc)
            Q_quat = quatnormalize(cf.Q)
            Q_quat_conj = quatconj(Q_quat)
            Q_error = quatmultiply(Qc_quat, Q_quat_conj)

            # 计算转矩
            cf.tao = 200 * Q_error[1:4]  # 只取后三个元素 (x, y, z)

            # 更新角速度和角加速度
            cf.w_dot = cf.tao  # 简化模型，忽略惯性矩阵
            cf.w += cf.w_dot * dt

            # 更新四元数导数
            qx = operator_x(cf.Q[1:4])
            # 构建矩阵 [-Q(2:4), Q(1)*eye(3)-qx]
            part1 = -cf.Q[1:4].reshape(3, 1)  # 3x1 矩阵
            part2 = cf.Q[0] * np.eye(3) - qx  # 3x3 矩阵
            # 水平拼接两个部分，得到 3x4 矩阵
            matrix_3x4 = np.hstack((part1, part2))
            # 转置得到 4x3 矩阵，并乘以角速度向量
            cf.Q_dot = 0.5 * np.dot(matrix_3x4.T, cf.w)

            # 更新四元数
            cf.Q += np.int64(Q_error)* dt  # 使用误差四元数更新

            # 归一化四元数
            cf.Q = quatnormalize(cf.Q)

            # 计算旋转矩阵
            R_Q = operator_R(Q_error)

            # 更新加速度、速度和位置
            cf.v_dot = -g * e_3 + cf.T / cf.m * R_Q.dot(e_3)
            cf.current_uav_vel += cf.v_dot *dt
            cf.current_uav_pos = cf.position()
            position_matrix[t_step, i] = cf.current_uav_pos + cf.current_uav_vel *dt
            cf.cmdVelocityWorld(vel = cf.current_uav_vel, yawRate = 0)

    # 仿真结束后保存数据到txt文件
    for i in range(N):
        # 为每个无人机创建一个文件
        filename = f"drone_{i}_positions.txt"
    
        # 提取该无人机的所有位置数据
        drone_positions = position_matrix[:, i, :]
    
        # 添加时间步作为第一列
        time_steps = np.arange(5000).reshape(-1, 1)
        data_to_save = np.hstack((time_steps, drone_positions))
    
        # 保存到文件
        np.savetxt(filename, data_to_save, 
                    fmt='%.6f', 
                    delimiter='\t',
                    header='t_step\tx\ty\tz',
                    comments='')
    
        print(f"Saved positions for drone {i} to {filename}")        

        

    # 控制集群全部无人机降落
    flag_all_uav_land = False
    for cf in allcfs.crazyflies:
        cf.flag_uav_land = False
    while flag_all_uav_land == False:
        #rate.sleep()
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
