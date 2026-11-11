#!/usr/bin/env python3

import os
import sys
import numpy as np

from pycrazyswarm import Crazyswarm


# ============================================================
# 1. REAL FLIGHT PARAMETERS
# ============================================================

SIDE_LENGTH = 1.50
HALF_SIDE = SIDE_LENGTH / 2.0

TAKEOFF_HEIGHT = 0.50
TAKEOFF_DURATION = 3.0

CENTER_TO_CORNER_DURATION = 6.0

# 1.5 m / 8 s gives a deliberately slow trajectory.
SIDE_DURATION = 8.0

CORNER_HOVER = 1.0

HOVER_AFTER_TAKEOFF = 5.0
HOVER_AFTER_SQUARE = 3.0

LAND_HEIGHT = 0.05
LAND_DURATION = 3.0


# ============================================================
# 2. Workspace safety limits
#
# These values follow the conservative starting point used in
# our Crazyswarm tutorial.
#
# Modify ONLY after verifying the real Lighthouse workspace.
# ============================================================

X_MIN = -2
X_MAX =  2

Y_MIN = -2
Y_MAX =  2

Z_MIN =  0.20
Z_MAX =  1.40


# ============================================================
# 3. Real Crazyflie configuration
# ============================================================

CRAZYFLIES_YAML = (
    "/home/chimu/chimu_ws/src/crazyswarm/ros_ws/src/crazyswarm/launch/crazyflies_3_test.yaml"
)


def check_point(point, name):

    x, y, z = point

    if not (X_MIN <= x <= X_MAX):
        raise RuntimeError(
            "%s x=%.2f exceeds workspace [%.2f, %.2f]"
            % (name, x, X_MIN, X_MAX)
        )

    if not (Y_MIN <= y <= Y_MAX):
        raise RuntimeError(
            "%s y=%.2f exceeds workspace [%.2f, %.2f]"
            % (name, y, Y_MIN, Y_MAX)
        )

    if not (Z_MIN <= z <= Z_MAX):
        raise RuntimeError(
            "%s z=%.2f exceeds workspace [%.2f, %.2f]"
            % (name, z, Z_MIN, Z_MAX)
        )


def main():

    # --------------------------------------------------------
    # Prevent accidental simulation/real-mode confusion
    # --------------------------------------------------------

    if "--sim" in sys.argv:
        raise RuntimeError(
            "This is the REAL-FLIGHT program. "
            "Use single_square_15m_sim.py for simulation."
        )

    # --------------------------------------------------------
    # Real-flight software lock
    # --------------------------------------------------------


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
            "This real-flight program requires exactly ONE Crazyflie. "
            "Current configuration contains %d."
            % len(cfs)
        )

    cf = cfs[0]

    # initialPosition is treated as the center of the square
    center = np.array(cf.initialPosition, dtype=float)

    flight_z = center[2] + TAKEOFF_HEIGHT
    land_z = center[2] + LAND_HEIGHT

    center_air = np.array([
        center[0],
        center[1],
        flight_z
    ])

    # --------------------------------------------------------
    # Generate 1.5 m square
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

    # --------------------------------------------------------
    # Workspace validation
    # --------------------------------------------------------

    check_point(center_air, "CENTER")

    check_point(p1, "P1")
    check_point(p2, "P2")
    check_point(p3, "P3")
    check_point(p4, "P4")

    # --------------------------------------------------------
    # Show plan before takeoff
    # --------------------------------------------------------

    print("")
    print("==========================================")
    print("        CRAZYSWARM REAL FLIGHT")
    print("==========================================")
    print("Crazyflie ID :", cf.id)
    print("Center       :", center)
    print("Flight Z     : %.2f m" % flight_z)
    print("Square side  : %.2f m" % SIDE_LENGTH)
    print("Side duration: %.2f s" % SIDE_DURATION)
    print("")
    print("P1:", p1)
    print("P2:", p2)
    print("P3:", p3)
    print("P4:", p4)
    print("")
    print("The real Crazyflie must be physically")
    print("placed at the configured center position.")
    print("==========================================")
    print("")

    confirmation = input(
        "Type FLY to start the real flight: "
    )

    if confirmation.strip() != "FLY":
        print(">>> Flight cancelled.")
        return

    # --------------------------------------------------------
    # Takeoff
    # --------------------------------------------------------

    print(">>> TAKEOFF")

    cf.takeoff(
        targetHeight=flight_z,
        duration=TAKEOFF_DURATION
    )

    time_helper.sleep(
        TAKEOFF_DURATION + HOVER_AFTER_TAKEOFF
    )

    # --------------------------------------------------------
    # Move from center to P1
    # --------------------------------------------------------

    print(">>> Move to first corner P1")

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
    # Fly four sides
    # --------------------------------------------------------

    targets = [p2, p3, p4, p1]

    for i, point in enumerate(targets, start=1):

        print(
            ">>> Edge %d / 4, target = %s"
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

    # --------------------------------------------------------
    # Square complete
    # --------------------------------------------------------

    print(">>> Square completed")

    time_helper.sleep(HOVER_AFTER_SQUARE)

    # --------------------------------------------------------
    # Return to center
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

    print(">>> LANDING")

    cf.land(
        targetHeight=land_z,
        duration=LAND_DURATION
    )

    time_helper.sleep(
        LAND_DURATION + 1.0
    )

    print(">>> REAL FLIGHT FINISHED")


if __name__ == "__main__":
    main()
