"""Takeoff -> straight line -> land for one Crazyflie."""

import numpy as np
from pycrazyswarm import Crazyswarm


TAKEOFF_HEIGHT = 0.5
TAKEOFF_DURATION = 2.5
MOVE_DURATION = 3.0
LAND_DURATION = 2.5


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper

    # 取配置中的第一架 Crazyflie
    cf = swarm.allcfs.crazyflies[0]

    # 1. 起飞到 0.5 m
    cf.takeoff(
        targetHeight=TAKEOFF_HEIGHT,
        duration=TAKEOFF_DURATION
    )
    timeHelper.sleep(TAKEOFF_DURATION + 1.0)

    # 2. 沿 x 轴正方向飞 1 m
    cf.goTo(
        goal=np.array([1.0, 0.0, TAKEOFF_HEIGHT]),
        yaw=0.0,
        duration=MOVE_DURATION
    )
    timeHelper.sleep(MOVE_DURATION + 1.0)

    # 3. 降落
    cf.land(
        targetHeight=0.04,
        duration=LAND_DURATION
    )
    timeHelper.sleep(LAND_DURATION + 1.0)


if __name__ == "__main__":
    main()
