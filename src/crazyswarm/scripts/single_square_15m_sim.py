#!/usr/bin/env python3

import numpy as np
from pycrazyswarm import Crazyswarm


# ============================================================
# 1. Flight parameters
# ============================================================/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/scripts

SIDE_LENGTH = 1.50
HALF_SIDE = SIDE_LENGTH / 2.0

TAKEOFF_HEIGHT = 0.50
TAKEOFF_DURATION = 3.0

CENTER_TO_CORNER_DURATION = 6.0
SIDE_DURATION = 8.0
CORNER_HOVER = 1.0

HOVER_AFTER_TAKEOFF = 3.0
HOVER_AFTER_SQUARE = 3.0

LAND_HEIGHT = 0.05
LAND_DURATION = 3.0


# ============================================================
# 2. Simulation configuration
# ============================================================

CRAZYFLIES_YAML = (
    "/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch/crazyflies_single_15m_sim.yaml"
)


def main():

    # --------------------------------------------------------
    # Initialize Crazyswarm
    # --------------------------------------------------------

    swarm = Crazyswarm(
        crazyflies_yaml=CRAZYFLIES_YAML
    )

    time_helper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies

    if len(cfs) != 1:
        raise RuntimeError(
            "This program requires exactly ONE Crazyflie."
        )

    cf = cfs[0]

    # The configured initial position is the square center
    center = np.array(cf.initialPosition, dtype=float)

    flight_z = center[2] + TAKEOFF_HEIGHT

    center_air = np.array([
        center[0],
        center[1],
        flight_z
    ])

    # --------------------------------------------------------
    # Square waypoints
    # --------------------------------------------------------

    p1 = np.array([
        center[0] - HALF_SIDE,
        center[1] - HALF_SIDE,
        flight_z
    ])

    p2 = np.array([
        center[0] + HALF_SIDE,
        center[1] - HALF_SIDE,
        flight_z
    ])

    p3 = np.array([
        center[0] + HALF_SIDE,
        center[1] + HALF_SIDE,
        flight_z
    ])

    p4 = np.array([
        center[0] - HALF_SIDE,
        center[1] + HALF_SIDE,
        flight_z
    ])

    square_points = [p1, p2, p3, p4, p1]

    print(">>> SIMULATION")
    print(">>> Square center:", center)
    print(">>> Square side length: %.2f m" % SIDE_LENGTH)

    # --------------------------------------------------------
    # Camera
    # Keep approximately the view we adjusted previously
    # --------------------------------------------------------

    visualizer = time_helper.visualizer

    if hasattr(visualizer, "view"):

        camera = visualizer.view.camera

        camera.center = (
            center[0],
            center[1],
            flight_z / 2.0
        )

        camera.azimuth = 105.0
        camera.elevation = 20.0
        camera.scale_factor = 3.0
        camera.fov = 45.0

    # --------------------------------------------------------
    # Takeoff
    # --------------------------------------------------------

    print(">>> Takeoff")

    cf.takeoff(
        targetHeight=flight_z,
        duration=TAKEOFF_DURATION
    )

    time_helper.sleep(
        TAKEOFF_DURATION + HOVER_AFTER_TAKEOFF
    )

    # --------------------------------------------------------
    # Move from center to first corner
    # --------------------------------------------------------

    print(">>> Move from center to P1")

    cf.goTo(
        p1,
        yaw=0.0,
        duration=CENTER_TO_CORNER_DURATION,
        relative=False
    )

    time_helper.sleep(
        CENTER_TO_CORNER_DURATION + CORNER_HOVER
    )

    # --------------------------------------------------------
    # Fly square
    # --------------------------------------------------------

    for i, point in enumerate(square_points[1:], start=1):

        print(
            ">>> Square edge %d, target = %s"
            % (i, point)
        )

        cf.goTo(
            point,
            yaw=0.0,
            duration=SIDE_DURATION,
            relative=False
        )

        time_helper.sleep(
            SIDE_DURATION + CORNER_HOVER
        )

    print(">>> Square completed")

    time_helper.sleep(HOVER_AFTER_SQUARE)

    # --------------------------------------------------------
    # Return to square center
    # --------------------------------------------------------

    print(">>> Return to center")

    cf.goTo(
        center_air,
        yaw=0.0,
        duration=CENTER_TO_CORNER_DURATION,
        relative=False
    )

    time_helper.sleep(
        CENTER_TO_CORNER_DURATION + 2.0
    )

    # --------------------------------------------------------
    # Land
    # --------------------------------------------------------

    print(">>> Landing")

    cf.land(
        targetHeight=center[2] + LAND_HEIGHT,
        duration=LAND_DURATION
    )

    time_helper.sleep(
        LAND_DURATION + 1.0
    )

    print(">>> Finished")


if __name__ == "__main__":
    main()
