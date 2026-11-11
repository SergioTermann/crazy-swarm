#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三机延迟起飞、直线飞行、追近编队。

坐标单位 m，时间单位 s；x 为前进方向，y 为横向。
任务时刻：0 中机起飞，3 中机前进，10 两侧起飞，13 两侧前进，
38 中机到达 x=3.5 并悬停、两侧开始追近，46 三机进入最终悬停。
35 秒是中机的水平运动时间，不包括 3 秒起飞时间。
默认持续悬停，Ctrl+C 降落；--hover-duration 可指定悬停后降落。
cmdPosition 持续发送位置参考，由飞控闭环保持高度及横向位置。
0.25 m 是要求下限，0.35 m 提前触发终止；实机误差/延迟仍需验证。
"""
from __future__ import division, print_function

import argparse
import os
import sys
import signal
import numpy as np

HEIGHT = 0.7
MAX_HEIGHT = 1.2
Z_KP = 1.2
Z_KI = 0.08
Z_KD = 0.25
MAX_VERTICAL_SPEED = 0.35
TAKEOFF_DURATION = 3.0
FOLLOWER_TAKEOFF_DELAY = 10.0
CRUISE_SPEED = 0.1
TRAVEL_DISTANCE = 3.5
CHASE_DURATION = 8.0
CHASE_MAX_SPEED = 0.30
FOLLOWER_BEHIND = 0.30
FOLLOWER_SIDE = 0.45
MIN_DISTANCE = 0.25
ABORT_DISTANCE = 0.35
MAX_TRACKING_ERROR = 0.15
CONTROL_HZ = 30.0
LAND_HEIGHT = 0.05
LAND_DURATION = 3.0
REQUIRED_IDS = {3, 4, 12}
MOVE_END = TAKEOFF_DURATION + TRAVEL_DISTANCE / CRUISE_SPEED
CHASE_END = MOVE_END + CHASE_DURATION


def reference(index, elapsed, start_y):
    """返回位置参考；未轮到起飞的无人机返回 None，不发送启动指令。"""
    delay = 0.0 if index == 1 else FOLLOWER_TAKEOFF_DELAY
    age = elapsed - delay
    if age < 0:
        return None
    pos = np.array([0.0, start_y[index], HEIGHT])
    if age < TAKEOFF_DURATION:
        # 起飞由高层 takeoff 执行，不用低层指令覆盖。
        return None
    if index == 1:
        pos[0] = min(CRUISE_SPEED * (age - TAKEOFF_DURATION), TRAVEL_DISTANCE)
    elif elapsed < MOVE_END:
        pos[0] = CRUISE_SPEED * (age - TAKEOFF_DURATION)
    else:
        # 三次 Hermite：起点速度为巡航速度，终点速度为零，平滑加速追近。
        start = pos.copy()
        start[0] = CRUISE_SPEED * (MOVE_END - delay - TAKEOFF_DURATION)
        end = np.array([TRAVEL_DISTANCE - FOLLOWER_BEHIND,
                        np.sign(start_y[index]) * FOLLOWER_SIDE, HEIGHT])
        s = np.clip((elapsed - MOVE_END) / CHASE_DURATION, 0.0, 1.0)
        pos = ((2*s**3 - 3*s**2 + 1) * start
               + (s**3 - 2*s**2 + s) * CHASE_DURATION
               * np.array([CRUISE_SPEED, 0.0, 0.0])
               + (-2*s**3 + 3*s**2) * end)
    return pos


def read_positions(cfs):
    positions = np.array([cf.position() for cf in cfs], dtype=float)
    if positions.shape != (3, 3) or not np.all(np.isfinite(positions)):
        raise RuntimeError("定位数据无效，终止任务")
    return positions


def check_separation(positions):
    for i in range(3):
        for j in range(i + 1, 3):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < ABORT_DISTANCE:
                raise RuntimeError("无人机索引 {} / {} 间距 {:.3f} m，小于提前终止阈值 {:.2f} m"
                                   .format(i, j, distance, ABORT_DISTANCE))


def z_pid_velocity(z, z_error_integral, z_error_previous, dt):
    """PID 高度控制；返回垂直速度及更新后的积分、误差。"""
    error = HEIGHT - float(z)
    integral = np.clip(z_error_integral + error * dt, -0.5, 0.5)
    derivative = (error - z_error_previous) / dt if dt > 0.0 else 0.0
    velocity = Z_KP * error + Z_KI * integral + Z_KD * derivative
    if z >= MAX_HEIGHT:
        velocity = min(velocity, -0.10)
    return float(np.clip(velocity, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)), integral, error


def emergency_land(cfs, active, th):
    """Best-effort emergency landing; ignore a second Ctrl+C while cleaning up."""
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for i in sorted(active):
            try:
                cfs[i].notifySetpointsStop(100)
            except Exception as exc:
                print('ID {} 控制切换失败：{}'.format(cfs[i].id, exc))
        try:
            th.sleep(0.15)
        except Exception:
            pass
        for i in sorted(active):
            try:
                cfs[i].land(targetHeight=LAND_HEIGHT, duration=LAND_DURATION)
            except Exception as exc:
                print('ID {} 紧急降落发送失败：{}'.format(cfs[i].id, exc))
        try:
            th.sleep(LAND_DURATION + 0.5)
        except Exception:
            pass
        for i in sorted(active):
            try:
                cfs[i].cmdStop()
            except Exception:
                pass
    finally:
        signal.signal(signal.SIGINT, old_handler)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--hover-duration', type=float, default=None,
                        help='最终悬停秒数；不指定则持续悬停，Ctrl+C 降落')
    parser.add_argument('--crazyflies-config', default=None,
                        help='可选 YAML 路径；实机必须与 ROS 服务端配置一致')
    args, _ = parser.parse_known_args()
    if args.hover_duration is not None and (not np.isfinite(args.hover_duration)
                                           or args.hover_duration < 0):
        raise ValueError('hover-duration 必须是非负有限数')
    from pycrazyswarm import Crazyswarm
    config = args.crazyflies_config
    if config is None and '--sim' in sys.argv[1:]:
        config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'three_uav_formation_sim.yaml')
    swarm = Crazyswarm(crazyflies_yaml=config, parent_parser=parser)
    th = swarm.timeHelper
    cfs = list(swarm.allcfs.crazyflies)
    if len(cfs) != 3:
        raise RuntimeError('配置必须恰好包含三架无人机')
    ids = {int(cf.id) for cf in cfs}
    if ids != REQUIRED_IDS:
        raise RuntimeError('当前配置的无人机 ID 必须是 3、4、12，实际为 {}'.format(sorted(ids)))
    leader_cf = next(cf for cf in cfs if int(cf.id) == 4)
    followers = [cf for cf in cfs if int(cf.id) != 4]
    followers.sort(key=lambda cf: float(cf.position()[1]))
    cfs = [followers[0], leader_cf, followers[1]]
    initial = read_positions(cfs)
    check_separation(initial)
    start_y = initial[:, 1].copy()
    print('按 y 从小到大排列的 ID：{}；目标机固定为 ID：{}'.format(
        [cf.id for cf in cfs], cfs[1].id))
    print('中机先起飞；10 s 两侧起飞；38 s 开始追近；46 s 最终悬停。')
    active = set()
    takeoff_times = {}
    hover_start = None
    z_integrals = np.zeros(3)
    z_previous_errors = np.zeros(3)
    try:
        start_time = th.time()
        last_control_time = start_time
        last_status_print = -1.0
        while True:
            elapsed = th.time() - start_time
            dt = max(th.time() - last_control_time, 1.0 / CONTROL_HZ)
            last_control_time = th.time()
            positions = read_positions(cfs)
            check_separation(positions)
            targets = [reference(i, elapsed, start_y) for i in range(3)]
            for i in active:
                if elapsed - takeoff_times[i] < TAKEOFF_DURATION:
                    targets[i] = None
            for i, cf in enumerate(cfs):
                delay = 0.0 if i == 1 else FOLLOWER_TAKEOFF_DELAY
                if elapsed >= delay and i not in active:
                    active.add(i)
                    takeoff_times[i] = elapsed
                    cf.takeoff(targetHeight=HEIGHT, duration=TAKEOFF_DURATION)
                    targets[i] = None
                if targets[i] is not None:
                    z_velocity, z_integrals[i], z_previous_errors[i] = z_pid_velocity(
                        positions[i, 2], z_integrals[i], z_previous_errors[i], dt)
                    # 直线阶段使用速度指令，严格限制为 0.1 m/s。
                    age = elapsed - (0.0 if i == 1 else FOLLOWER_TAKEOFF_DELAY) - TAKEOFF_DURATION
                    if age >= 0.0 and elapsed < MOVE_END:
                        cf.cmdVelocityWorld(vel=np.array([CRUISE_SPEED, 0.0, z_velocity]), yawRate=0.0)
                    elif i == 1:
                        # 目标机到达 3.5 m 后保持悬停。
                        cf.cmdVelocityWorld(vel=np.array([0.0, 0.0, z_velocity]), yawRate=0.0)
                    else:
                        # 跟随机在目标机停止后才加速追近，速度上限为 0.30 m/s。
                        target = targets[i]
                        error = target - positions[i]
                        speed = min(CHASE_MAX_SPEED, np.linalg.norm(error[:2]))
                        velocity = np.array([0.0, 0.0, z_velocity])
                        if speed > 1e-3:
                            velocity[:2] = error[:2] / np.linalg.norm(error[:2]) * speed
                        cf.cmdVelocityWorld(vel=velocity, yawRate=0.0)

            # 每秒打印一次实时位置、阶段和本轮控制状态，便于实飞诊断。
            if elapsed - last_status_print >= 1.0:
                last_status_print = elapsed
                if elapsed < TAKEOFF_DURATION:
                    phase = '目标机起飞'
                elif elapsed < FOLLOWER_TAKEOFF_DELAY:
                    phase = '目标机前进/等待跟随机起飞'
                elif elapsed < MOVE_END:
                    phase = '三机直线前进'
                elif elapsed < CHASE_END:
                    phase = '跟随机加速追近'
                else:
                    phase = '编队悬停'
                print('[状态 t={:.1f}s 阶段={}]'.format(elapsed, phase))
                for i, cf in enumerate(cfs):
                    pos = positions[i]
                    if i not in active:
                        command = '等待起飞'
                    elif targets[i] is None:
                        command = '起飞控制中'
                    elif elapsed < MOVE_END:
                        command = '速度 [0.100, 0.000, PID-z]'
                    elif i == 1:
                        command = '悬停 [0.000, 0.000, PID-z]'
                    else:
                        command = '追踪 [水平<=0.300, PID-z]'
                    print('  ID {}: pos=({:.3f}, {:.3f}, {:.3f}), {}'.format(
                        cf.id, pos[0], pos[1], pos[2], command))
            if elapsed >= CHASE_END:
                settled = all(np.linalg.norm(positions[i] - targets[i]) < 0.08 for i in range(3))
                if settled and hover_start is None:
                    hover_start = th.time()
                    print('三机已靠拢，保持悬停。Ctrl+C 将降落。')
                if not settled:
                    hover_start = None
                if (hover_start is not None and args.hover_duration is not None
                        and th.time() - hover_start >= args.hover_duration):
                    break
            th.sleepForRate(CONTROL_HZ)
    except KeyboardInterrupt:
        print('收到中断，开始降落。')
    finally:
        emergency_land(cfs, active, th)


if __name__ == '__main__':
    main()
