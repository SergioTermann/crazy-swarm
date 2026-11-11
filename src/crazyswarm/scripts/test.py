#!/usr/bin/env python3
"""单机起飞，悬停30秒，降落。"""
from pycrazyswarm import Crazyswarm

TAKEOFF_HEIGHT = 1.0      # 起飞高度(米)，第一次建议改成 0.5
TAKEOFF_DURATION = 2.5    # 起飞用时(秒)
HOVER_DURATION = 30.0     # 悬停时间(秒)
LAND_DURATION = 2.5       # 降落用时(秒)

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]  # 只有一架飞机

    print(">>> 起飞!")
    cf.takeoff(targetHeight=TAKEOFF_HEIGHT, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION)

    print(">>> 悬停 30 秒...")
    timeHelper.sleep(HOVER_DURATION)

    print(">>> 降落!")
    cf.land(targetHeight=0.04, duration=LAND_DURATION)
    timeHelper.sleep(LAND_DURATION)

    print(">>> 完成!")

if __name__ == "__main__":
    main()