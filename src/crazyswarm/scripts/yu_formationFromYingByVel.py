import numpy as np
from pycrazyswarm import *
import sys
import time
from typing import List

controller_mode = "velocity"  # "position" or "velocity"

class APF_Planner(object):
    def __init__(self, inta=1, k=3, p0=1.0):
        self.inta = inta     #引力系数
        self.k = k        #斥力系数
        self.p0 = p0      #初始位置

    def plan(self, pos_cur, pos_target, neighbor: List[np.ndarray]):
        Fatt = -self.inta*self.dist(pos_cur, pos_target)*(pos_cur-pos_target)/self.dist(pos_cur, pos_target)
        Freq = np.zeros(3)
        for pos_i in neighbor:
            Freq1 = self.k*(1/self.dist(pos_i, pos_cur) - 1/self.p0)*(self.dist2(pos_cur, pos_target)/self.dist2(pos_i, pos_cur))*(pos_i-pos_cur)/self.dist(pos_i, pos_cur) if self.dist(pos_i, pos_cur)<self.p0 else 0
            Freq2 = 1/2*self.k*(1/self.dist(pos_i, pos_cur) - 1/self.p0)*self.dist(pos_cur, pos_target)
            Freq += Freq1 + Freq2
        F = Fatt + Freq
        return F

    def dist(self, pos1: np.ndarray, pos2: np.ndarray):
        return np.linalg.norm(pos1-pos2)
    
    def dist2(self, pos1,pos2):
        return np.sum((pos1-pos2)**2)

def formation_8(ids):
    x = np.array(ids) - 3.5
    result = np.zeros((2,len(ids)))
    result[0, :] = np.array(x)
    result[1, :] = np.array([i if i <=0 else -i for i in x])
    return result

def main():
    global controller_mode
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # Takeoff Stage
    allcfs.takeoff(targetHeight=1.0, duration=3.0)
    timeHelper.sleep(5.0)
    step_n=500
    dt=0.03 #仿真步长（采样时间）
    Timeall=step_n*dt
    N=6  #无人机数量，不包括虚拟领导者，其中一架无人机的位置可以与虚拟领导者重合

    neighbor=np.zeros((N,N))  #邻居矩阵
    r_communicate=5  #邻居交互半径
    height = 2.0  #无人机高度
    land = 0.1  #降落高度
    theta_tmp = np.linspace(0, 2*np.pi, N+1)[0:N]
    r2 = np.zeros((4,N)) 
    ids = [i+1 for i in range(N)]
    r2[0:2] = formation_8(ids)
    r2[2,:] = 0
    r2[3,:] = 0

    r1 = np.zeros((4,N)) 
    r1[0, :] = 2*np.sin(theta_tmp)  
    r1[1, :] = 2*np.cos(theta_tmp)  
    r1[2, :] = 0
    r1[3, :] = 0

    sum_x_bia=np.zeros((3,N)) 
    sum_v_bia=np.zeros((3,N)) 

    u=np.zeros((3,N)) 
    v=np.zeros((3,N)) 
    X=np.zeros((3,N)) 

    u_gamma=np.array([0, 0, 0, 0], dtype=float)  #虚拟领导者的加速度
    v_gamma=np.array([0, 0, 0, 0], dtype=float)  #虚拟领导者的初始速度
    x_gamma=np.array([0, 0, 2, 0], dtype=float)  #虚拟领导者的初始位置


    k_1=2 
    k_2=5 
    k_3=2 
    k_4=5 
    
    #初始化无人机状态
    pos_target = np.zeros((3,N)) 
    vel_target = np.zeros((3,N)) 
    acc_target = np.zeros((3,N)) 
    pos_cur = np.zeros((3,N)) 
    vel_cur = np.zeros((3,N)) 
    acc_cur = np.zeros((3,N)) 
    pos_err = np.zeros((3,N)) 
    vel_err = np.zeros((3,N)) 
    acc_err = np.zeros((3,N)) 
    for cf in allcfs.crazyflies:
        pos_cur[:, cf.id-1] = cf.position()
        vel_cur[:, cf.id-1] = cf.velocity()
        acc_cur[:, cf.id-1] = cf.acceleration()
    pos_target[:, 0:N] = r2[0:3, 0:N] + np.tile(np.array([[0], [0], [height]]), (1, N))
    vel_target[:, 0:N] = vel_cur[:, 0:N]
    acc_target[:, 0:N] = acc_cur[:, 0:N]
    pos_err[:, 0:N] = pos_target[:, 0:N] - pos_cur[:, 0:N]
    vel_err[:, 0:N] = vel_target[:, 0:N] - vel_cur[:, 0:N]
    acc_err[:, 0:N] = acc_target[:, 0:N] - acc_cur[:, 0:N]
    while np.linalg.norm(pos_err[:, 0:N])/N > 0.01:
        for cf in allcfs.crazyflies:
            cf.cmdPosition(pos=pos_target[:, cf.id-1], yaw=0)
            timeHelper.sleep(dt)
        for cf in allcfs.crazyflies:
            pos_cur[:, cf.id-1] = cf.position()
            vel_cur[:, cf.id-1] = cf.velocity()
            acc_cur[:, cf.id-1] = cf.acceleration()
        pos_err[:, 0:N] = pos_target[:, 0:N] - pos_cur[:, 0:N]
        vel_err[:, 0:N] = vel_target[:, 0:N] - vel_cur[:, 0:N]
        acc_err[:, 0:N] = acc_target[:, 0:N] - acc_cur[:, 0:N]
    for t_k in range(step_n):
        if t_k>=step_n/2:
            formation_offset = r1
        else:
            formation_offset = r2

        formation_offset_copy = np.copy(formation_offset)
     
    #------↓找到(t_k*dt)时刻每架无人机各自的邻居↓------ #
        for i in range(N):
            for j in range(N):
                if np.linalg.norm(pos_cur[:, i]-pos_cur[:, j])<r_communicate:
                    neighbor[i,j]=1
                elif i==j:
                    neighbor[i,j]=0
                else:
                    neighbor[i,j]=0
    #------↑找到(t_k*dt)时刻每架无人机各自的邻居↑------ #
    #------↓计算(t_k*dt)时刻每架无人机的控制量↓------ #
        for i in range(N):
            neighbor_list=np.where(neighbor[i,:]==1)[0]
            for j in neighbor_list:
                sum_x_bia[:,i] = (pos_cur[0:, i] - pos_cur[:, j]) - (formation_offset[0:3, i] - formation_offset[0:3, j])
                sum_v_bia[:,i] = (vel_cur[:, i] - vel_cur[:, j])
            u_1 = -k_1*sum_x_bia[:, i]    #机间位置控制分量
            u_2 = -k_2*(pos_cur[:, i] - x_gamma[0:3] - formation_offset[0:3, i])  #无人机与虚拟领导者相对位置控制分量
            u_3 = -k_3*sum_v_bia[:, i]    #机间速度控制分量
            u_4 = -k_4*(vel_cur[:, i] - v_gamma[0:3])  #无人机与虚拟领导者相对速度控制分量

            u[:, i] = u_1 + u_2 + u_3 + u_4 + u_gamma[0:3]  #总控制量

      #-------↓更新虚拟领导者以及每架无人机的速度和位置↓------#
        t = t_k*dt
        # u_gamma(1) = 5/sqrt(2)*log(abs((sqrt(2)+cos(0.1*t))/(sqrt(2)-cos(0.1*t)))) + 10*cos(0.1*t) 
        # u_gamma(2) = 10*atan(sin(0.1*t)) - 10*sin(0.1*t) 
        # u_gamma = 0.5*u_gamma 
        # v_gamma=v_gamma+u_gamma(1:2)*dt 
        # v_gamma=max(min(v_gamma,5),-5)  #对虚拟领导者速度限幅[-5,5]
        # x_gamma=x_gamma+v_gamma*dt 
        v_gamma[0:3] = np.array([-3*np.sin(t), 3*np.cos(t), 0])
        v_gamma[3] = 1
        x_gamma = x_gamma + v_gamma*dt

        if controller_mode == "position":
            #-----------↓更新formation_offset↓-#
            rotation_matrix = np.array([
                [np.cos(x_gamma[3]), -np.sin(x_gamma[3])],
                [np.sin(x_gamma[3]), np.cos(x_gamma[3])]])
            formation_offset_copy[0:2] = rotation_matrix.dot(formation_offset[0:2])
            for cf in allcfs.crazyflies:
                #-----------↑控制指令解算器↑-----------#
                pos_target[:, cf.id-1]= x_gamma[0:3] + formation_offset_copy[0:3, cf.id-1]
                #-----------↓控制指令限幅器↓-#
                pos_target[:, cf.id-1] = np.clip(pos_target[:, cf.id-1], -10, 10)
                
                pos_cur[:, cf.id-1] = cf.position()
                vel_cur[:, cf.id-1] = cf.velocity()
                acc_cur[:, cf.id-1] = cf.acceleration()
                pos_err[:, cf.id-1] = abs(pos_target[:, cf.id-1] - pos_cur[:, cf.id-1])
                cf.cmdPosition(pos_target[:, cf.id-1], yaw=x_gamma[3])
                timeHelper.sleep(dt)
                #------------打印当前时刻各无人机的位置和速度-#
                if t_k%100==0:
                    print("cf{} pos:{}, vel:{}".format(cf.id, pos_cur[:, cf.id-1], vel_cur[:, cf.id-1]))
        elif controller_mode == "velocity":
            pass
        else:
            print("Invalid controller mode!")
    allcfs.land(targetHeight = land, duration = 5.0)
    timeHelper.sleep(5.0)
    print("Mission Complete!")

if __name__ == "__main__":
    main()
