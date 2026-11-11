import numpy as np
from pycrazyswarm import *
import sys
import time
from typing import List

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
 

if __name__ == "__main__":
    main()
