# -*- coding: utf-8 -*-
"""
LMPC 编队跟踪-围捕控制 —— Crazyswarm 仿真主程序(单文件, 自包含)
================================================================
由 MATLAB 工程(main0115.m / MPC_obs)移植。按实飞验证需求做的简化(不影响
编队/围捕效果):
  1. 贝叶斯风险评估链(calculate_bayes_risk / dynamic_weights 等)简化为
     "障碍接近时间 EFD + 编队收敛误差" 的简单切换判据; 切换函数 G 的收缩/
     膨胀机制与原算法完全一致(收缩=围捕, 膨胀=编队跟踪);
  2. 障碍代价中恒为 0 的对数项(MATLAB 原代码 log(1)=0)删除, 只保留安全
     半径内的强斥力项(1e10);
  3. LMPC 的 Lyapunov 约束与优化变量无关(原 fmincon 实为常数约束), 移植为
     求解前可行性预判: 不满足时直接回退反步控制器 eta_i, 与原 fmincon
     失败回退行为一致;
  4. 原工程六机 beta/gamma 参数完全相同, 统一为常量矩阵;
  5. 圆形构型改为三机等边三角形构型(3 架跟随机): 1 架领导机被 3 架跟随无人机
     以等边三角形顶角编队跟踪/围捕, 三角形边长 0.26m(顶点在半径 R=0.15m
     的圆上, 方位角 90/210/330°), 通信拓扑为有向环 1←3←2←1, 3号机获得
     领导机信息(B_INFO);
  6. 切换逻辑与原工程一致: 风险触发 -> 三角形展开环绕, 保持 8s 后收紧围捕;
  7. 全场景按实飞场地(2m x 1m, 高度可用到 1.5m)缩放: 编队顶角半径 R=0.15m
     (边长 0.26m, 切换后外扩/内缩), 目标轨迹为场地中心 0.12m 半径圆
     (0.042m/s, 60s 约 3.3 圈) + 0.25~1.45m 螺旋爬升(不增圈数、只拉高度,
     顶点 = 1.5m 天花板减 5cm 安全边距),
     参考偏航恒为 0, 限速 0.3m/s, 指令位置带场地边界硬限幅 + 障碍安全层 +
     机间间距安全层, 总时长 60s;
  8. 避障: 障碍置于圆轨迹 90° 相位点, 群体接近时 EFD 触发展开环绕;
     LMPC 代价含 0.30m 内指数软斥力(优化器提前绕行), 指令层距障碍不足
     0.15m 时径向推出(反步回退通道也有硬保障);
  9. 保留: LMPC(SLSQP) + 反步辅助控制器 + RK4 预测模型 + 群内避碰代价 +
     通信拓扑 A/B + 等边三角形编队构型 M + 切换函数 G;
 10. 后端对接的三个关键实现(实测曾因此"无人机根本不跟踪"):
     a) 速度状态用后端 cf.velocity()(仿真器维护的真实速度), 无此接口的
        后端回退内部模型预测速度; 绝不用指令位置差分——那构成自引用回路,
        不动点 ~ 0.5*dt*TAU_MAX/L, 限速调小后实际速度塌缩到近零;
     b) 速度指令在所有安全层作用后统一按 (pos_cmd-实测位置)/DT 计算,
        与位置指令运动学自洽, 兼容"瞬移到 pos"与"按 vel 积分"两种后端语义;
     c) 控制周期 DT 自动与后端仿真 tick 对齐(默认 0.1s), 避免指令混叠。
运行结束后保存: 结果图 png / 数据 npz / 轨迹 csv+txt(Excel/MATLAB 可导入)。

运行方式(需要 Ubuntu 环境; 本脚本自动选择已安装的后端):
  * Crazyswarm1 pycrazyswarm 仿真模式:
      装好后(见官方 installation.rst 的 Simulation Only 一节)把本脚本拷到
      ros_ws/src/crazyswarm/scripts/ 下, 在该目录运行:
      python3 lmpc_crazyswarm_sim.py --sim
    说明:
      - 首次运行自动生成 crazyflies.yaml(3 跟随机 + 1 领导机, 与脚本同目录),
        并通过 crazyflies_yaml 参数显式传给后端。注意: pycrazyswarm 默认读的
        是 ../launch/crazyflies.yaml(官方 demo 配置), 不显式传入会加载错误!
      - "Warning: No joystick found!" 属正常提示(仿真无手柄, 自动切键盘)。
      - 可选: --vis vispy(交互 3D 窗口, 默认 matplotlib);
        --dt 0.05(同时收紧仿真 tick 与控制周期, 默认 0.1, 脚本会自动对齐)。
  * Crazyswarm2 (ROS2, 需先启动仿真后端):
      ros2 launch crazyswarm2 backend.py --sim crazyflie_types:=cf2
      python3 lmpc_crazyswarm_sim.py

依赖: numpy, scipy, matplotlib
"""

import os
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Crazyswarm 后端自动选择(三家 API 兼容)
try:
    from crazyflie_sim import CrazySwarm          # Bitcraze crazyflie-simulation
    BACKEND = "crazyflie_sim"
except ImportError:
    try:
        from crazyswarm2 import CrazySwarm        # Crazyswarm2 (ROS2)
        BACKEND = "crazyswarm2"
    except ImportError:
        from pycrazyswarm import Crazyswarm as CrazySwarm   # Crazyswarm1
        BACKEND = "pycrazyswarm"

# ============================================================================
# 1. 参数设置(对应 main0115.m 第1~6节)
# ============================================================================
PI = np.pi
# ---- 实飞小场地(2m x 1m x 1m): 全部尺度缩到 ~1.3m x 0.6m x 0.45m 内 ----
ROOM_X, ROOM_Y, ROOM_Z = 2.0, 1.0, 1.5      # 实飞场地尺寸 [m](高度可用到 1.5)
DT       = 0.10          # 采样时间 [s](=pycrazyswarm 默认仿真 tick; main() 中
                         #  会自动读取后端实际 tick 并同步, 保证指令不混叠)
T_SIM    = 60.0          # 总仿真时间 [s](完整任务链: 收敛->避障展环->收紧->多圈螺旋跟踪)
N_STEPS  = int(round(T_SIM / DT))          # 主循环步数
N_PRED   = 4             # LMPC 预测步数
TAU_MAX  = 0.3           # 最大控制输入; V_ss = K*tau ~= tau, 即限速 0.3 m/s
NUM_CF   = 3             # 跟随无人机数量(等边三角形编队)
USE_TARGET_CF = True     # 用一架真实 CF 做领导机(第 4 架), 3 跟随机 + 1 领导机

# 四旋翼简化模型参数(beta/gamma, 原工程六机相同)
_L = np.array([0.83, 0.77, 0.50, 0.51])
_K = np.array([1.0, 1.0, 1.0, PI / 180.0])
BETA  = np.diag(-_L ** -1.0)               # diag(-1/L)
GAMMA = np.diag(_K / _L)                   # diag(K/L)

# LMPC 权重
Q_W = np.diag([2e4, 2e4, 1e4, 1e2])        # 形成误差权重
F_W = np.diag([1e2, 1e2, 1e2, 1e2])        # 控制输入权重

# 编队构型 M_i(相对领导机的期望偏移, 3xN: x,y,z,psi)
# 等边三角形: 边长 0.26m, 顶点在半径 R = 0.26/sqrt(3) = 0.15m 圆上, 方位角 90/210/330°
#   (0,    0.15)         顶部顶点
#   (-0.13, -0.075)      左下顶点
#   ( 0.13, -0.075)      右下顶点
M_FORM = np.array([
    [0.0,     0.150, 0, 0],   # 无人机1: 顶部
    [-0.1299, -0.075, 0, 0],  # 无人机2: 左下角
    [ 0.1299, -0.075, 0, 0],  # 无人机3: 右下角
], dtype=float).T

# 通信拓扑(有向环形): 1←3, 2←1, 3←2; 领导信息矩阵 B(3号机获得领导机信息)
A_TOPO = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)
B_INFO = np.diag([0.0, 0, 1.0])   # 3号机获得领导机信息

# 反步控制器增益
K1_BS = np.diag([7.0, 7, 7, 6])
K2_BS = np.diag([5.0, 5, 5, 5])

# 切换函数参数
DELTA_SW   = 0.5          # 收敛速率参数
DELTA_TE   = 2.0          # 切换过渡时长 [s]
# h_bar 的几何语义(与原工程一致): G = -h*sign(M), 顶角偏移 = M + G
#   h = -0.03: 三角形顶角半径 |R=0.15| -> |R'=0.18|, 半径增大(展开环绕目标)
#   h = +0.03: 三角形顶角半径 |R=0.15| -> |R'=0.12|, 半径收紧(围捕捕获)
ENCIRCLE_H  = -0.03       # 第一次切换: 展开环绕(R: 0.15 -> 0.18)
CAPTURE_H   = +0.03       # 保持 SWITCH_HOLD 后: 收紧围捕(R: 0.15 -> 0.12)
SWITCH_HOLD = 8.0         # 切换保持时间 [s]

# 简化风险判据阈值
EFD_DANGER   = 3.0        # 障碍接近时间 < 3s 视为危险
CONV_ERR     = 0.10       # 平均编队误差 < 0.10m 视为已收敛

# 障碍(3 x n_obs): 虚拟位置(非真实无人机, 共 4 架: 3 跟随机 + 1 领导机)。
# 置于圆轨迹 90° 相位点, 编队每圈经过一次, 群体接近时 EFD 触发展开环绕,
# 通过时靠斥力代价 + 安全层绕开(单位 m)
OBS = np.array([[1.0, 0.62, 0.35]], dtype=float).T
D_SAFE_CMD = 0.15         # 指令层安全半径: 距障碍不足时径向推出 [m]
D_MIN_CMD  = 0.13         # 指令层机间最小间距: 两两推离 [m]

# 初始位置不预设, 飞行时先获取实际位置: 此处仅给仿真后端一个"落点/悬停高度",
# 主循环每步都用 cf.position() 读取实测位置作为状态反馈, 编队在领导机附近
# 自发收敛到三角形构型, 不依赖初始落点(实飞时无人机停在场地任意位置均适用)。
_INIT_XY = np.array([[0.30, 0.35], [0.55, 0.55], [0.35, 0.75]], float)  # yaml 仿真落点
                                                                      # (避开虚拟障碍 1.0,0.62)
HOVER_Z  = np.array([0.25, 0.30, 0.25])   # 起飞悬停高度 [m]


# ============================================================================
# 2. 简化无人机模型(对应 update_drone_state_full.m, RK4)
#    状态 X = [x, y, z, psi, vx, vy, vz, vpsi](速度为机体系)
# ============================================================================
def wrap_to_pi(a):
    return (a + PI) % (2 * PI) - PI


def R4(psi):
    """机体系->世界系的 4x4 旋转(与 MATLAB R_ib 一致)"""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]])


def drone_step(X, tau, dt):
    """RK4 一步积分: p_dot = R(psi)V, V_dot = beta V + gamma tau"""
    def f(Xu):
        p, V = Xu[0:4], Xu[4:8]
        pd = R4(p[3]) @ V
        Vd = BETA @ V + GAMMA @ tau
        return np.concatenate([pd, Vd])
    k1 = f(X)
    k2 = f(X + 0.5 * dt * k1)
    k3 = f(X + 0.5 * dt * k2)
    k4 = f(X + dt * k3)
    Xn = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    Xn[3] = wrap_to_pi(Xn[3])
    return Xn


# ============================================================================
# 3. 目标轨迹(实飞小场地版: 场地中心 0.12m 半径圆 + 0.25~1.5m 高螺旋爬升)
# ============================================================================
REF_CX, REF_CY = ROOM_X / 2.0, ROOM_Y / 2.0   # 圆心 = 场地中心 (1.0, 0.5)
REF_R   = 0.12            # 圆半径 [m]
REF_W   = 0.35            # 角速度 [rad/s] (线速度 0.042 m/s, 60s 转约 3.3 圈)
REF_Z0, REF_Z1 = 0.25, 1.45  # 高度 60s 内从 0.25m 缓升至 1.45m(高螺旋;
                              #  顶点=天花板1.5m减去5cm安全边距)

def reference(tk):
    """返回 (Yd, Yd_dot, Yd_ddot), 各为 4 维 [x,y,z,psi]
    小场地版: 偏航取常值 0(偏航通道增益小, 旋转偏航难以跟踪且无必要)"""
    T = T_SIM
    xr  = REF_CX + REF_R * np.cos(REF_W * tk)
    yr  = REF_CY + REF_R * np.sin(REF_W * tk)
    dxr = -REF_R * REF_W * np.sin(REF_W * tk)
    dyr =  REF_R * REF_W * np.cos(REF_W * tk)
    ddxr = -REF_R * REF_W**2 * np.cos(REF_W * tk)
    ddyr = -REF_R * REF_W**2 * np.sin(REF_W * tk)
    # z: smoothstep 从 REF_Z0 升到 REF_Z1
    dz = REF_Z1 - REF_Z0
    s1 = (3 * tk**2 / T**2) - (2 * tk**3 / T**3)      # 0 -> 1
    zr  = REF_Z0 + dz * s1
    dzr = dz * (6 * tk / T**2) - dz * (6 * tk**2 / T**3)
    ddzr = dz * (6 / T**2) - dz * (12 * tk / T**3)
    psir = 0.0
    vpsi = 0.0
    Yd      = np.array([xr, yr, zr, psir])
    Yd_dot  = np.array([dxr, dyr, dzr, vpsi])
    Yd_ddot = np.array([ddxr, ddyr, ddzr, 0.0])
    return Yd, Yd_dot, Yd_ddot


# ============================================================================
# 4. 切换函数(对应 compute_switching_functions_full.m)
# ============================================================================
def switching_functions(t0, T_switch, h_bar, M):
    """返回 G, G_dot, G_ddot (4xN)"""
    n = M.shape[1]
    G = np.zeros((4, n)); Gd = np.zeros((4, n)); Gdd = np.zeros((4, n))
    for i in range(n):
        h = h_bar[i] if T_switch > 0 else 0.0
        if T_switch > 0 and T_switch <= t0 < T_switch + DELTA_TE:
            # 过渡段: G 从 0 平滑过渡到满幅
            tan_arg = PI * (T_switch + DELTA_TE - t0) / (2 * DELTA_TE)
            with np.errstate(over="ignore", under="ignore"):
                tt = np.tan(tan_arg)
                G_ia = h * np.exp(-(tt**4) / (2 * DELTA_SW**4))
            for jj in range(2):
                G[jj, i] = -G_ia * np.sign(M[jj, i])
            # 一阶导
            sec = 1.0 / np.cos(tan_arg)**2
            Gd_ia = PI * G[0, i] / (DELTA_SW**4 * DELTA_TE) * tt**3 * sec
            Gd[0, i] = Gd[1, i] = Gd_ia
        elif T_switch > 0 and t0 >= T_switch + DELTA_TE:
            # 保持段
            for jj in range(2):
                G[jj, i] = -h * np.sign(M[jj, i])
    return G, Gd, Gdd


# ============================================================================
# 5. 反步辅助控制器(对应 compute_backstepping_controller_full.m)
# ============================================================================
def Rdot4(psi, vpsi):
    s, c = np.sin(psi), np.cos(psi)
    return np.array([[-s * vpsi, -c * vpsi, 0, 0],
                     [ c * vpsi, -s * vpsi, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])


def backstepping(i, X_all, U, M, G, Gd, Gdd, Yd, Yd_dot, Yd_ddot):
    """返回 eta_i, S_i1, S_i1dot, S_i2, S_i2dot"""
    p_i, V_i = X_all[0:4, i], X_all[4:8, i]
    psi = p_i[3]
    Rib = R4(psi)
    d_i = A_TOPO[i].sum()
    b_i = B_INFO[i, i]
    nbrs = np.nonzero(A_TOPO[i] > 0)[0]

    # --- S_i1: 编队形成误差 ---
    S1 = np.zeros(4)
    for j in nbrs:
        S1 += A_TOPO[i, j] * (X_all[0:4, j] - M[:, j] - G[:, j]
                               - p_i + M[:, i] + G[:, i])
    S1 += b_i * (Yd - p_i + M[:, i] + G[:, i])

    # --- S_i1 导数 ---
    nsum1 = np.zeros(4)
    for j in nbrs:
        nsum1 += A_TOPO[i, j] * (R4(X_all[3, j]) @ X_all[4:8, j] - Gd[:, j])
    S1d = -(d_i + b_i) * (Rib @ V_i - Gd[:, i]) + nsum1 + b_i * Yd_dot

    # --- S_i2(带 sigma=1.3 的分数幂项) ---
    sigma = 1.3
    Phi = np.sign(S1) * np.abs(S1)**sigma
    kphi = np.diag(sigma * np.abs(S1)**(sigma - 1.0))
    Phid = kphi @ S1d
    S2 = S1d + K1_BS @ Phi

    # --- S_i2 导数 ---
    Ribd = Rdot4(psi, V_i[3])
    nsum2 = np.zeros(4)
    for j in nbrs:
        Vj = X_all[4:8, j]
        Rjb, Rjbd = R4(X_all[3, j]), Rdot4(X_all[3, j], Vj[3])
        Vjd = BETA @ Vj + GAMMA @ U[:, j]
        nsum2 += A_TOPO[i, j] * (Rjbd @ Vj + Rjb @ Vjd - Gdd[:, j])
    Vid = BETA @ V_i + GAMMA @ U[:, i]
    S2d = -(d_i + b_i) * (Ribd @ V_i + Rib @ Vid - Gdd[:, i]) \
        + nsum2 + b_i * Yd_ddot + K1_BS @ (S2 - S1d)

    # --- 反步控制量 eta_i ---
    complex_term = (d_i + b_i) * Gdd[:, i] + b_i * Yd_ddot \
        - (d_i + b_i) * (Ribd @ V_i) + nsum2 + S1 + K2_BS @ S2 + K1_BS @ Phid
    g_inv = np.linalg.inv(GAMMA)
    eta = -g_inv @ BETA @ V_i + g_inv @ Rib.T @ (complex_term / (d_i + b_i))
    eta = np.clip(eta, -TAU_MAX, TAU_MAX)
    return eta, S1, S1d, S2, S2d


# ============================================================================
# 6. LMPC 优化(对应 lmpc_cost_function.m / solve_lmpc_optimization.m)
# ============================================================================
def _collision_cost(X_pred, X_all, i, nbrs, D_bar_ij=0.12):
    """群内避碰: sum a_ij * exp(tan(D_bar - D_ij))"""
    cost = 0.0
    p_i = X_pred[0:3]
    for j in nbrs:
        D = np.linalg.norm(p_i - X_all[0:3, j])
        with np.errstate(over="ignore", under="ignore"):
            v = np.tan(D_bar_ij - D)
            cost += A_TOPO[i, j] * np.exp(np.clip(v, -700, 700))
    return cost


def _obstacle_cost(X_pred, obs, D_bar=0.30, D_safe=0.10):
    """群外避碰: D_bar 内软斥力(指数增强, 让优化器提前绕行),
    D_safe 内硬墙(1e10); 原对数项恒为0, 已删"""
    cost = 0.0
    p = X_pred[0:3]
    for l in range(obs.shape[1]):
        D = np.linalg.norm(p - obs[:, l])
        if D <= D_safe:
            cost += 1e10
        elif D < D_bar:
            cost += 50.0 * np.exp(8.0 * (D_bar - D))
    return cost


def lmpc_cost(x, X_cur, S1, X_all, i, nbrs, dt, n_pred, obs):
    """预测时域总代价: 形成误差 + 控制量 + 避碰 + 障碍"""
    J = 0.0
    Xp = X_cur.copy()
    taus = x.reshape(n_pred, 4)
    for k in range(n_pred):
        tau = taus[k]
        formation = S1 @ Q_W @ S1
        control = tau @ F_W @ tau
        col = _collision_cost(Xp, X_all, i, nbrs)
        ob = _obstacle_cost(Xp, obs)
        J += (formation + control + col + ob) * dt
        Xp = drone_step(Xp, tau, dt)
    return J


def solve_lmpc(X_cur, S1, S1d, S2, S2d, X_all, i, nbrs, eta, x0_prev):
    """SLSQP 求解; Lyapunov 约束为常数, 不满足时直接回退 eta(同 fmincon 行为)"""
    # Lyapunov 可行性预判: Vdot_mpc <= Vdot_aux
    Vdot_mpc = S1 @ S1d + S2 @ S2d
    Vdot_aux = -(S1 @ K1_BS @ S1) - (S2 @ K2_BS @ S2)
    if Vdot_mpc - Vdot_aux > 0:
        return eta.copy(), np.inf, False, x0_prev

    n_var = 4 * N_PRED
    x0 = x0_prev if x0_prev is not None else np.zeros(n_var)
    bounds = [(-TAU_MAX, TAU_MAX)] * n_var
    try:
        res = minimize(lmpc_cost, x0,
                       args=(X_cur, S1, X_all, i, nbrs, DT, N_PRED, OBS),
                       method="SLSQP", bounds=bounds,
                       options={"maxiter": 100, "ftol": 1e-6})
    except Exception:
        return eta.copy(), np.inf, False, None
    if res.success and res.x is not None:
        tau0 = res.x[0:4].copy()
        # 热启动: 序列左移一位
        x_next = np.concatenate([res.x[4:], res.x[-4:]])
        return tau0, res.fun, True, x_next
    return eta.copy(), np.inf, False, None


# ============================================================================
# 7. 简化风险判据(替代 calculate_bayes_risk_probability 链)
# ============================================================================
def risk_metrics(X_next, Yd_next, M, G, obs):
    """返回 (平均编队误差, 最近障碍 EFD); danger 判据在主循环中给出"""
    n = X_next.shape[1]
    err = 0.0
    for i in range(n):
        err += np.linalg.norm(X_next[0:3, i] - Yd_next[0:3] - M[0:3, i] - G[0:3, i])
    err /= n
    center = X_next[0:3, :].mean(axis=1)
    v_swarm = X_next[4:7, :].mean(axis=1)
    d_min, efd = np.inf, np.inf
    for l in range(obs.shape[1]):
        d = np.linalg.norm(obs[:, l] - center)
        if d < d_min:
            p_so = center - obs[:, l]
            nrm = np.linalg.norm(p_so)
            v_close = np.dot(v_swarm, p_so) / nrm if nrm > 1e-9 else 0.0
            if v_close > 0.05:
                efd = d / v_close
            d_min = d
    return err, efd


# ============================================================================
# 8. Crazyswarm 主程序
# ============================================================================
YAML_TAG = "lmpc_crazyswarm_sim08:triangle-v3"
YAML_CONTENT = """\
# {tag}
crazyflies:
{entries}
"""

# yaml 与脚本放在同一目录; pycrazyswarm 默认读 ../launch/crazyflies.yaml,
# 必须通过 crazyflies_yaml 参数显式传入本文件, 否则加载的是官方 demo 配置!
YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "crazyflies.yaml")


def ensure_yaml():
    """生成 crazyflies.yaml(3 跟随机 + 1 领导机 + 虚拟障碍=无真实障碍机)。
    版本标记不符时自动重建; 初始落点仅供参考(控制运行时读取实际位置)"""
    need = True
    if os.path.exists(YAML_PATH):
        try:
            with open(YAML_PATH, "r") as f:
                need = YAML_TAG not in f.read()
        except OSError:
            pass
        if not need:
            return
    entries = []
    ids = list(range(1, NUM_CF + 1)) + ([NUM_CF + 1] if USE_TARGET_CF else [])
    for idx, cid in enumerate(ids):
        if cid <= NUM_CF:
            xy = _INIT_XY[idx]
        else:
            xy = np.array([REF_CX + REF_R, REF_CY])   # 领导机初始位于轨迹起点
        entries.append(
            "- id: {i}\n  channel: {ch}\n  initialPosition: [{x:.2f}, {y:.2f}, 0.0]\n  type: default\n"
            .format(i=cid, ch=100 + idx, x=xy[0], y=xy[1]))
    with open(YAML_PATH, "w") as f:
        f.write(YAML_CONTENT.format(tag=YAML_TAG, entries="".join(entries)))
    print("[init] 已生成 crazyflies.yaml (%d 架: 3 跟随机 + 1 领导机)" % len(ids))


def get_cf_list(swarm):
    cfs_obj = getattr(swarm, "crazyflies", None)
    if cfs_obj is None:
        cfs_obj = swarm.allcfs
    cf_list = list(getattr(cfs_obj, "crazyflies", cfs_obj))
    cf_list.sort(key=lambda c: c.id)
    return cf_list


def save_trajectories_txt(X_hist, err_hist, t_switch_log,
                           prefix="lmpc_crazyswarm_traj"):
    """轨迹结果保存: txt(易读带表头) + csv(Excel/MATLAB 可直接导入)。
    每行一个采样时刻, 各机状态平铺为列:
      t, ref_x/y/z/psi, cf1_x...cf1_vpsi, cf2_..., ..., err1..errN
    X_hist 形状 (NUM_CF, n+1, 8), 状态 [x,y,z,psi,vx,vy,vz,vpsi]"""
    n_steps = X_hist.shape[1] - 1
    n_cf = X_hist.shape[0]
    refs = np.array([reference(k * DT)[0] for k in range(n_steps + 1)])
    cols = [refs]
    for i in range(n_cf):
        cols.append(X_hist[i])
    err_pad = np.vstack([err_hist.T, err_hist.T[-1]])
    cols.append(err_pad)
    data = np.hstack(cols)
    tcol = np.arange(n_steps + 1).reshape(-1, 1) * DT
    data = np.hstack([tcol, data])

    state_names = ["x", "y", "z", "psi", "vx", "vy", "vz", "vpsi"]
    header = ["t", "ref_x", "ref_y", "ref_z", "ref_psi"]
    for i in range(n_cf):
        header += ["cf%d_%s" % (i + 1, s) for s in state_names]
    header += ["cf%d_err" % (i + 1) for i in range(n_cf)]

    csv_path = prefix + ".csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        np.savetxt(f, data, fmt="%.6f", delimiter=",")
    txt_path = prefix + ".txt"
    with open(txt_path, "w") as f:
        f.write("# LMPC formation-encirclement trajectory log\n")
        f.write("# dt = %.3f s, duration = %.1f s, %d followers + 1 target\n"
                % (DT, T_SIM, n_cf))
        f.write("# switch times [s]: %s\n"
                % (", ".join("%.2f" % s for s in t_switch_log) or "none"))
        f.write("# state per cf: x y z psi [m, m, m, rad] + vx vy vz vpsi [m/s, rad/s]\n")
        f.write("# " + "\t".join(header) + "\n")
        np.savetxt(f, data, fmt="%.6f", delimiter="\t")
    print("[done] 轨迹已保存: %s / %s (%d 行 x %d 列)"
          % (csv_path, txt_path, data.shape[0], data.shape[1]))


def main():
    ensure_yaml()
    # pycrazyswarm 默认读取 ../launch/crazyflies.yaml(官方 demo 配置), 必须显式
    # 传入本项目生成的 yaml; 其他后端不支持该参数时自动回退
    try:
        swarm = CrazySwarm(crazyflies_yaml=YAML_PATH)
    except TypeError:
        swarm = CrazySwarm()
    timeHelper = swarm.timeHelper
    # 控制周期与后端仿真 tick 对齐(不同步会造成指令混叠/半步滞后, 表现为
    # 跟踪振荡与残余误差); 实机后端无 dt 属性时保持默认 DT
    sim_dt = getattr(timeHelper, "dt", None)
    global DT, N_STEPS
    if sim_dt is not None and abs(sim_dt - DT) > 1e-9:
        print("[init] 控制周期 DT %.3fs -> 与后端仿真 tick %.3fs 对齐"
              % (DT, sim_dt))
        DT = float(sim_dt)
        N_STEPS = int(round(T_SIM / DT))
    cf_list = get_cf_list(swarm)
    by_id = {c.id: c for c in cf_list}
    # 防护: 若加载的机体不全, 给出明确提示而不是 KeyError
    expected_ids = list(range(1, NUM_CF + (2 if USE_TARGET_CF else 1)))
    missing = [cid for cid in expected_ids if cid not in by_id]
    if missing:
        raise SystemExit(
            "[error] 加载的机体缺少 id=%s (现有: %s)。\n"
            "原因: 后端加载了错误的 crazyflies.yaml(如 launch/ 目录的官方配置)。\n"
            "请确认脚本同目录的 crazyflies.yaml 含 %d 架(id 1~%d)。"
            % (missing, sorted(by_id), len(expected_ids), expected_ids[-1]))
    cfs = [by_id[i + 1] for i in range(NUM_CF)]     # 跟随机 1..3
    cf_target = by_id.get(NUM_CF + 1) if USE_TARGET_CF else None   # 领导机 id=4
    print("[init] 后端: %s | 跟随机 %d 架%s" %
          (BACKEND, NUM_CF, " | 领导机 1 架" if cf_target else ""))

    # ---- 起飞 ----
    for cf, hz in zip(cfs, HOVER_Z):
        cf.takeoff(targetHeight=float(hz), duration=2.5)
    if cf_target is not None:
        cf_target.takeoff(targetHeight=0.25, duration=2.5)
    timeHelper.sleep(3.5)

    # ---- 飞行开始时获取各机实际初始位置(不预设) ----
    # 编队不依赖初始落点: 领导机跟踪参考轨迹, 跟随机收敛到三角形顶角绕其编队。
    print("[init] 实测初始位置:")
    for i, cf in enumerate(cfs):
        p = np.array(cf.position(), dtype=float)
        print("  follower%d: x=%.3f y=%.3f z=%.3f" % (i + 1, p[0], p[1], p[2]))
    if cf_target is not None:
        plt_pos = np.array(cf_target.position(), dtype=float)
        print("  leader  : x=%.3f y=%.3f z=%.3f" % (plt_pos[0], plt_pos[1], plt_pos[2]))

    # ---- 状态与记录 ----
    X_hist  = np.zeros((NUM_CF, N_STEPS + 1, 8))    # 测量状态
    J_hist  = np.full((NUM_CF, N_STEPS), np.inf)
    feas    = np.zeros((NUM_CF, N_STEPS), dtype=bool)
    err_hist = np.zeros((NUM_CF, N_STEPS))
    risk_err = np.zeros(N_STEPS); risk_efd = np.zeros(N_STEPS)
    steps_done = [0]             # 实际完成的步数(Ctrl+C 提前结束时用于截断保存)

    last_pos = [np.array(cf.position(), dtype=float) for cf in cfs]
    vel_model = [np.zeros(3) for _ in range(NUM_CF)]  # 内部模型速度(体坐标, 后备)
    psi_state  = np.zeros(NUM_CF)                   # 内部偏航状态(指令值)
    vpsi_state = np.zeros(NUM_CF)
    x0_warm = [None] * NUM_CF                       # SLSQP 热启动
    # 速度来源: 后端是否提供 cf.velocity()?
    # (pycrazyswarm 仿真器维护真实速度状态; 实机后端一般没有, 回退模型速度)
    has_vel_api = all(callable(getattr(cf, "velocity", None)) for cf in cfs)

    T_switch = 0.0
    h_bar = ENCIRCLE_H * np.ones(NUM_CF)
    switched = False
    switched_captured = False   # 是否已进入收紧围捕阶段
    t_switch_log = []

    U_k = np.zeros((4, NUM_CF))
    print("[run] 开始 LMPC 仿真: %d 步, dt=%.2fs, 总时长 %.1fs" % (N_STEPS, DT, T_SIM))

    try:
        for k in range(N_STEPS):
            t_now = k * DT
            Yd, Yd_dot, Yd_ddot = reference(t_now)
            Yd_next, Yd_dot_next, _ = reference(t_now + DT)

            # ---- 测量: 位置实测 + 速度来源(后端真实速度或模型速度) ----
            # 关键修复: 不可对指令位置做差分获取速度! 位置跟踪后端下, 差分速度
            # 是"自己指令的导数", 构成自引用回路, 不动点 ~ 0.5*dt*TAU_MAX/L
            # (TAU_MAX 小时实际速度塌缩到接近 0, 无人机"根本不跟踪")。
            # 优先用后端 cf.velocity()(仿真器维护的真实速度状态);
            # 后端无此接口(如实机)时用上一步内部模型预测速度。
            X_k = np.zeros((8, NUM_CF))
            for i in range(NUM_CF):
                pos = np.array(cfs[i].position(), dtype=float)
                if has_vel_api:
                    v_world = np.array(cfs[i].velocity(), dtype=float)
                    R2m = R4(psi_state[i])[0:2, 0:2]
                    v_body = np.array([*(R2m.T @ v_world[0:2]), v_world[2]])
                    vel_model[i] = v_body
                else:
                    v_body = vel_model[i]
                X_k[0:3, i] = pos
                X_k[3, i] = psi_state[i]
                X_k[4:7, i] = v_body
                X_k[7, i] = vpsi_state[i]
                last_pos[i] = pos
            X_hist[:, k, :] = X_k.T

            # ---- 简化风险判据 -> 切换逻辑(与原工程语义一致) ----
            # 第一次风险触发: 三角形展开环绕(半径 R: 0.15 -> 0.18);
            # 保持 SWITCH_HOLD 后: 三角形收紧围捕(半径 R: 0.15 -> 0.12)
            err_avg, efd = risk_metrics(X_k, Yd, M_FORM,
                                        switching_functions(t_now, T_switch, h_bar, M_FORM)[0], OBS)
            risk_err[k] = err_avg; risk_efd[k] = efd
            # 编队未收敛时(err>0.15)不触发 EFD 切换, 保证"先收敛再展环"链路
            danger = (err_avg < CONV_ERR) or (efd < EFD_DANGER and err_avg < 0.15)
            first_trig = (not switched and danger)
            hold_done = (switched and not switched_captured
                         and t_now - T_switch > SWITCH_HOLD)
            if first_trig or hold_done:
                T_switch = t_now if t_now > 0 else DT
                if not switched:
                    # 第一次风险触发: 展开环绕
                    h_bar = ENCIRCLE_H * np.ones(NUM_CF)
                    switched = True
                else:
                    # 展开保持 SWITCH_HOLD 后: 收紧围捕(此后保持围捕跟踪)
                    h_bar = CAPTURE_H * np.ones(NUM_CF)
                    switched_captured = True
                t_switch_log.append(T_switch)
                print("[sw ] t=%.2fs -> %s (err=%.3fm, efd=%s)"
                      % (T_switch, "收紧围捕(±0.10m)" if switched_captured else "展开环绕(±0.16m)",
                         err_avg, ("%.1fs" % efd) if np.isfinite(efd) else "inf"))

            G, Gd, Gdd = switching_functions(t_now, T_switch, h_bar, M_FORM)

            # ---- 逐机: 反步控制器 + LMPC 优化 + 状态预测 ----
            U_k = np.zeros((4, NUM_CF))
            X_next = X_k.copy()
            cmd_list = []
            for i in range(NUM_CF):
                eta, S1, S1d, S2, S2d = backstepping(
                    i, X_k, U_k, M_FORM, G, Gd, Gdd, Yd, Yd_dot, Yd_ddot)
                nbrs = np.nonzero(A_TOPO[i] > 0)[0]
                tau, J, ok, x0_warm[i] = solve_lmpc(
                    X_k[:, i], S1, S1d, S2, S2d, X_k, i, nbrs, eta, x0_warm[i])
                U_k[:, i] = tau
                J_hist[i, k] = J
                feas[i, k] = ok
                err_hist[i, k] = np.linalg.norm(
                    X_k[0:4, i] - (Yd + M_FORM[:, i] + G[:, i]))

                # 内部模型预测下一步, 得到全状态指令
                Xp = drone_step(X_k[:, i], tau, DT)
                X_next[:, i] = Xp
                psi_state[i], vpsi_state[i] = Xp[3], Xp[7]
                vel_model[i] = Xp[4:7].copy()     # 供无 velocity() 后端作速度状态

                pos_cmd = Xp[0:3].copy()
                pos_cmd[2] = max(pos_cmd[2], 0.05)          # 地面保护
                # 场地边界硬限幅(实飞安全墙): 指令位置夹在场地内
                pos_cmd[0] = np.clip(pos_cmd[0], 0.05, ROOM_X - 0.05)
                pos_cmd[1] = np.clip(pos_cmd[1], 0.05, ROOM_Y - 0.05)
                pos_cmd[2] = np.clip(pos_cmd[2], 0.05, ROOM_Z - 0.05)
                # 障碍安全层: 距障碍不足 D_SAFE_CMD 时沿径向推出
                # (无论 LMPC 可行还是反步回退, 指令层都保证安全间距)
                for l in range(OBS.shape[1]):
                    dvec = pos_cmd[0:3] - OBS[0:3, l]
                    d = np.linalg.norm(dvec)
                    if d < D_SAFE_CMD:
                        pos_cmd[0:3] = OBS[0:3, l] + dvec / max(d, 1e-6) * D_SAFE_CMD
                R2 = R4(Xp[3])[0:2, 0:2]
                yaw_cmd = float(Xp[3])
                omega = np.array([0.0, 0.0, Xp[7]])
                # 速度指令在各安全层作用完之后统一计算(见下发段), 保证
                # pos/vel 指令始终运动学自洽: vel = (pos_cmd - 实测位置)/DT。
                cmd_list.append([pos_cmd, X_k[0:3, i].copy(), np.zeros(3),
                                 yaw_cmd, omega])

            # ---- 机间最小间距安全层: 两两不足 D_MIN_CMD 时沿连线推离 ----
            for a in range(NUM_CF):
                for b in range(a + 1, NUM_CF):
                    dv = cmd_list[b][0] - cmd_list[a][0]
                    d = np.linalg.norm(dv)
                    if d < D_MIN_CMD:
                        u = dv / max(d, 1e-6)
                        push = 0.5 * (D_MIN_CMD - d) * u
                        cmd_list[a][0] = cmd_list[a][0] - push
                        cmd_list[b][0] = cmd_list[b][0] + push

            # ---- 下发指令: 速度 = (最终位置指令 - 实测位置)/DT ----
            # 所有安全层(边界/障碍/机间)作用完之后再计算速度指令, 保证
            # pos/vel 运动学自洽: 无论后端按"瞬移到 pos"还是"按 vel 积分"
            # 推进, 无人机都精确落到 pos_cmd, 位置环不依赖模型匹配。
            for i in range(NUM_CF):
                pos_cmd, pos_meas_i, _, yaw, omega = cmd_list[i]
                vel_cmd = (pos_cmd - pos_meas_i) / DT
                cfs[i].cmdFullState(pos_cmd, vel_cmd, np.zeros(3),
                                    yaw, omega)

            # ---- 领导机跟踪参考轨迹(实际飞行的领队机, 指令运动学自洽) ----
            if cf_target is not None:
                t_pos = np.array(cf_target.position(), dtype=float)
                t_vel = (Yd_next[0:3] - t_pos) / DT
                cf_target.cmdFullState(
                    Yd_next[0:3], t_vel, Yd_ddot[0:3],
                    float(Yd_next[3]), np.array([0.0, 0.0, Yd_dot_next[3]]))

            if k % 50 == 0:
                print("[run] t=%5.2fs  err=%.3fm  LMPC可行率(近50步)=%.0f%%"
                      % (t_now, err_avg,
                         100.0 * feas[:, max(0, k - 49):k + 1].mean()))
            timeHelper.sleep(DT)
            steps_done[0] = k + 1

        X_hist[:, N_STEPS, :] = X_hist[:, N_STEPS - 1, :]
        steps_done[0] = N_STEPS

    except KeyboardInterrupt:
        print("\n[run] 用户中断, 准备降落...")

    # ---- 降落 ----
    for cf in cfs + ([cf_target] if cf_target is not None else []):
        cf.land(targetHeight=0.04, duration=3.5)
    timeHelper.sleep(4.5)

    # ---- 结果保存与绘图 ----
    # Ctrl+C 提前结束时只保存实际完成的步数(避免零填充行)
    n_done = max(steps_done[0], 1)
    if n_done < N_STEPS:
        X_hist = X_hist[:, :n_done + 1, :].copy()
        X_hist[:, n_done, :] = X_hist[:, n_done - 1, :]
        err_hist = err_hist[:, :n_done].copy()
        print("[done] 提前结束于 t=%.2fs, 保存前 %d 步数据" % (n_done * DT, n_done))
    np.savez("lmpc_crazyswarm_result.npz",
             X=X_hist, J=J_hist, feasible=feas, err=err_hist,
             risk_err=risk_err, risk_efd=risk_efd, t_switch=np.array(t_switch_log))
    save_trajectories_txt(X_hist, err_hist, t_switch_log)
    try:
        import matplotlib.pyplot as plt
        n_plot = X_hist.shape[1] - 1     # 实际保存的步数(可能与 N_STEPS 不同)
        tt = np.arange(n_plot + 1) * DT
        tk = np.arange(n_plot) * DT
        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        for i in range(NUM_CF):
            ax1.plot(X_hist[i, :, 0], X_hist[i, :, 1], X_hist[i, :, 2], lw=0.8)
        ref = np.array([reference(k2 * DT)[0] for k2 in range(n_plot + 1)])
        ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", lw=1.5, label="target")
        ax1.scatter(OBS[0], OBS[1], OBS[2], marker="x", c="r", s=80, label="obstacle")
        ax1.set_xlim(0, ROOM_X); ax1.set_ylim(0, ROOM_Y); ax1.set_zlim(0, ROOM_Z)
        ax1.set_title("3D trajectories (room %gx%gx%g m)" % (ROOM_X, ROOM_Y, ROOM_Z))
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(tk, err_hist.T, lw=0.8)
        ax2.set_title("formation error per drone [m]"); ax2.grid(True)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(tk, risk_err[:n_plot], label="mean formation err")
        efd_p = np.where(np.isfinite(risk_efd[:n_plot]), risk_efd[:n_plot], 50.0)
        ax3.plot(tk, efd_p, label="obstacle EFD (capped 50s)")
        ax3.axhline(EFD_DANGER, color="r", ls="--", lw=0.8)
        ax3.set_title("risk metrics (simplified)"); ax3.legend(fontsize=8); ax3.grid(True)

        ax4 = fig.add_subplot(2, 2, 4)
        Jm = np.where(np.isfinite(J_hist[:, :n_plot]), J_hist[:, :n_plot], np.nan).mean(axis=0)
        ax4.semilogy(tk, Jm, lw=0.8)
        ax4.set_title("mean LMPC cost"); ax4.grid(True)

        fig.suptitle("LMPC formation-encirclement in Crazyswarm "
                     "(switches: %s)" % (["%.1f" % s for s in t_switch_log] or "none"))
        fig.tight_layout()
        fig.savefig("lmpc_crazyswarm_result.png", dpi=150)
        print("[done] 结果已保存: lmpc_crazyswarm_result.png / .npz")
        plt.show()
    except Exception as e:
        print("[done] 绘图跳过(%s), 数据已保存至 npz/txt/csv" % e)

    print("[done] 仿真结束: 切换时刻 %s" % (["%.2fs" % s for s in t_switch_log] or "无"))


if __name__ == "__main__":
    main()
