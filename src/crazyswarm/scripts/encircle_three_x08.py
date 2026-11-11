#!/usr/bin/env python2
from __future__ import print_function

import math
import numpy as np

from pycrazyswarm import Crazyswarm

# ============================================================
# Scene configuration
# Field center is (0, 0)
# x direction: 3 m  -> [-1.5, 1.5]
# y direction: 2 m  -> [-1.0, 1.0]
# ============================================================

FIELD_X = (-1.5, 1.5)
FIELD_Y = (-1.0, 1.0)
FIELD_MARGIN = 0.08

# The Crazyflies are expected to start near x=0.8, but the planner does not
# impose an extra mission x-limit. Only the unchanged physical field boundary
# x in [-1.5, 1.5], y in [-1.0, 1.0] is enforced.
NOMINAL_TAKEOFF_X = 0.80

# Circular obstacle in XY plane.
# Obstacle center is shifted to x=0.2, y=0.
OBSTACLE_CENTER = np.array([0.20, 0.00])
OBSTACLE_RADIUS = 0.12
OBSTACLE_DIAMETER = 2.0 * OBSTACLE_RADIUS
OBSTACLE_MARGIN = 0.18             # extra clearance around the obstacle
PLAN_RADIUS = OBSTACLE_RADIUS + OBSTACLE_MARGIN

# Encirclement target, on the opposite side of the obstacle from the x≈0.8 takeoff area.
TARGET = np.array([-0.80, 0.00])
ENCIRCLE_RADIUS = 0.20

FLIGHT_HEIGHT = 0.25
TAKEOFF_DURATION = 2.0
LAND_DURATION = 2.0
HOLD_TIME = 3.0

# Smooth position streaming
CMD_RATE = 50.0                    # Hz
MAX_PATH_SPEED = 0.35              # m/s, approximate peak path speed
MIN_MOVE_DURATION = 4.0

# Three obstacle-bypass corridors.
# Lowest-starting CF uses bottom outer lane,
# highest-starting CF uses top outer lane,
# middle CF automatically selects a shorter inner lane.
OUTER_LANE_Y = 0.82
INNER_LANE_Y = 0.58

# Optional Crazyflie-Crazyflie collision avoidance.
# This API exists in both simulation and real Crazyswarm.
ENABLE_CF_COLLISION_AVOIDANCE = False
CF_COLLISION_RADII = np.array([0.12, 0.12, 0.22])

AUTO_LAND = True



def safe_land(cfs, timeHelper, reason=None):
    """
    Best-effort vertical landing at each Crazyflie's current XY position.

    This function is used for every abnormal exit after Crazyswarm has
    been created. It first leaves low-level streaming mode (if active),
    then sends a high-level land command to every available Crazyflie.
    Individual failures are caught so that one problematic vehicle does
    not prevent the others from receiving a landing command.
    """
    if reason:
        print("\n[SAFETY] {}".format(reason))

    if cfs is None or len(cfs) == 0:
        print("[SAFETY] No Crazyflies are available for landing.")
        return

    print("[SAFETY] Landing all available Crazyflies in place...")

    # If cmdPosition() streaming was active, hand control back to the
    # high-level commander before requesting land(). This is harmless if
    # streaming had not started yet.
    for cf in cfs:
        try:
            cf.notifySetpointsStop()
        except Exception as e:
            print("[SAFETY] cf{} notifySetpointsStop failed: {}".format(
                getattr(cf, 'id', '?'), e))

    try:
        timeHelper.sleep(0.2)
    except Exception:
        pass

    # land() changes altitude while keeping the current XY location.
    for cf in cfs:
        try:
            cf.land(targetHeight=0.03, duration=LAND_DURATION)
        except Exception as e:
            print("[SAFETY] cf{} land command failed: {}".format(
                getattr(cf, 'id', '?'), e))

    try:
        timeHelper.sleep(LAND_DURATION + 1.0)
    except Exception:
        pass

    for cf in cfs:
        try:
            cf.stop()
        except Exception as e:
            print("[SAFETY] cf{} stop failed: {}".format(
                getattr(cf, 'id', '?'), e))

    print("[SAFETY] Landing sequence finished.")

def point_segment_distance_2d(p, a, b):
    """Distance from point p to line segment a-b in 2D."""
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return np.linalg.norm(p - q)


def direct_path_is_clear(start_xy, goal_xy):
    """True if the straight segment does not enter the inflated obstacle."""
    d = point_segment_distance_2d(
        OBSTACLE_CENTER,
        np.asarray(start_xy),
        np.asarray(goal_xy))
    return d >= PLAN_RADIUS


def inside_field_xy(p):
    return (FIELD_X[0] + FIELD_MARGIN <= p[0] <= FIELD_X[1] - FIELD_MARGIN and
            FIELD_Y[0] + FIELD_MARGIN <= p[1] <= FIELD_Y[1] - FIELD_MARGIN)


def hermite_curve(points, samples_per_segment=80):
    """
    C1-continuous cubic Hermite curve through all points.
    Endpoint tangents are zero; internal tangents use centered differences.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return points.copy()

    tangents = np.zeros_like(points)
    for i in range(1, n - 1):
        tangents[i] = 0.45 * (points[i + 1] - points[i - 1])

    out = []
    for i in range(n - 1):
        p0 = points[i]
        p1 = points[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]

        # Do not duplicate shared endpoints.
        count = samples_per_segment
        for k in range(count):
            u = float(k) / float(count)
            u2 = u * u
            u3 = u2 * u

            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = u3 - u2

            p = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            out.append(p)

    out.append(points[-1])
    return np.asarray(out)


def curve_is_safe(curve, allow_start_outside=False):
    """
    Check static-obstacle clearance and nominal field boundary.

    If allow_start_outside=True, the curve may begin outside the nominal
    2 x 3 m field. Once it enters the field, it is not allowed to leave
    again. This keeps the takeoff position free instead of aborting when
    mocap/simulation reports a start a few centimeters outside the field.
    """
    entered_field = False

    for p in curve:
        inside = inside_field_xy(p[:2])

        if inside:
            entered_field = True
        elif not allow_start_outside or entered_field:
            return False

        d = np.linalg.norm(p[:2] - OBSTACLE_CENTER)
        if d < PLAN_RADIUS:
            return False

    # Goal should be inside the nominal field.
    return entered_field


def build_smooth_path(start, goal, lane_y):
    """
    Build a smooth 3D path.
    If the direct path intersects the obstacle, use two bypass waypoints.
    lane_y is an absolute world-frame y coordinate.
    """
    start = np.asarray(start, dtype=float).copy()
    goal = np.asarray(goal, dtype=float).copy()

    start[2] = FLIGHT_HEIGHT
    goal[2] = FLIGHT_HEIGHT

    # The takeoff position is intentionally NOT fixed. A start point may be
    # slightly outside the nominal field; the generated path is allowed to
    # enter the field smoothly. The final encirclement goal must remain inside.
    if not inside_field_xy(goal[:2]):
        raise RuntimeError("Goal point outside field: {}".format(goal))

    if np.linalg.norm(start[:2] - OBSTACLE_CENTER) < PLAN_RADIUS:
        raise RuntimeError("Start point is inside obstacle safety region.")
    if np.linalg.norm(goal[:2] - OBSTACLE_CENTER) < PLAN_RADIUS:
        raise RuntimeError("Goal point is inside obstacle safety region.")

    if direct_path_is_clear(start[:2], goal[:2]):
        curve = hermite_curve([start, goal])
        if curve_is_safe(curve, allow_start_outside=True):
            return curve

    direction = 1.0 if goal[0] >= start[0] else -1.0
    x_offset = PLAN_RADIUS + 0.13

    # If smoothing cuts too close to the obstacle, automatically move
    # the lane farther outward, up to the field limit.
    sign = 1.0 if lane_y >= OBSTACLE_CENTER[1] else -1.0
    lane_mag = abs(lane_y - OBSTACLE_CENTER[1])

    for _ in range(10):
        y = OBSTACLE_CENTER[1] + sign * lane_mag

        if direction > 0:
            x1 = OBSTACLE_CENTER[0] - x_offset
            x2 = OBSTACLE_CENTER[0] + x_offset
        else:
            x1 = OBSTACLE_CENTER[0] + x_offset
            x2 = OBSTACLE_CENTER[0] - x_offset

        p1 = np.array([x1, y, FLIGHT_HEIGHT])
        p2 = np.array([x2, y, FLIGHT_HEIGHT])

        control_points = [start, p1, p2, goal]
        curve = hermite_curve(control_points)

        if curve_is_safe(curve, allow_start_outside=True):
            return curve

        lane_mag += 0.04

        max_lane = min(
            abs(FIELD_Y[1] - FIELD_MARGIN - OBSTACLE_CENTER[1]),
            abs(OBSTACLE_CENTER[1] - (FIELD_Y[0] + FIELD_MARGIN)))
        if lane_mag > max_lane:
            break

    raise RuntimeError(
        "Could not find a safe smooth route around the obstacle. "
        "Try reducing OBSTACLE_MARGIN or changing scene geometry.")


def curve_arclength(curve):
    ds = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s


def sample_curve_by_fraction(curve, s, fraction):
    """Arc-length interpolation, fraction in [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    total = s[-1]
    if total < 1e-9:
        return curve[-1].copy()

    target_s = fraction * total
    idx = np.searchsorted(s, target_s)

    if idx <= 0:
        return curve[0].copy()
    if idx >= len(s):
        return curve[-1].copy()

    s0 = s[idx - 1]
    s1 = s[idx]
    alpha = (target_s - s0) / max(s1 - s0, 1e-12)
    return (1.0 - alpha) * curve[idx - 1] + alpha * curve[idx]


def quintic_smoothstep(tau):
    """0->1 progress with zero velocity and acceleration at both ends."""
    tau = max(0.0, min(1.0, tau))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def encircle_goals():
    """
    Three equally spaced positions around the target.
    0, 120, 240 degrees -> pairwise spacing is sqrt(3)*R.
    """
    angles = [0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0]
    goals = []
    for a in angles:
        goals.append(np.array([
            TARGET[0] + ENCIRCLE_RADIUS * math.cos(a),
            TARGET[1] + ENCIRCLE_RADIUS * math.sin(a),
            FLIGHT_HEIGHT
        ]))

    # Sort by y so start-goal assignment preserves vertical ordering.
    goals.sort(key=lambda p: p[1])
    return goals


def choose_lanes(starts, assigned_goals):
    """
    Avoid unnecessary path crossings:
      lowest start  -> lower outer corridor
      highest start -> upper outer corridor
      middle start  -> shorter upper/lower inner corridor
    """
    order = sorted(range(3), key=lambda i: starts[i][1])
    lanes = [0.0, 0.0, 0.0]

    lanes[order[0]] = OBSTACLE_CENTER[1] - OUTER_LANE_Y
    lanes[order[2]] = OBSTACLE_CENTER[1] + OUTER_LANE_Y

    mid = order[1]
    top_lane = OBSTACLE_CENTER[1] + INNER_LANE_Y
    bottom_lane = OBSTACLE_CENTER[1] - INNER_LANE_Y

    # Approximate route cost through the two bypass waypoints.
    def bypass_cost(lane_y):
        s = starts[mid]
        g = assigned_goals[mid]
        xoff = PLAN_RADIUS + 0.13
        if g[0] >= s[0]:
            p1 = np.array([OBSTACLE_CENTER[0] - xoff, lane_y])
            p2 = np.array([OBSTACLE_CENTER[0] + xoff, lane_y])
        else:
            p1 = np.array([OBSTACLE_CENTER[0] + xoff, lane_y])
            p2 = np.array([OBSTACLE_CENTER[0] - xoff, lane_y])

        return (np.linalg.norm(s[:2] - p1) +
                np.linalg.norm(p1 - p2) +
                np.linalg.norm(p2 - g[:2]))

    lanes[mid] = top_lane if bypass_cost(top_lane) <= bypass_cost(bottom_lane) else bottom_lane
    return lanes


def validate_three_cfs(cfs):
    if len(cfs) != 3:
        raise RuntimeError(
            "This script requires exactly 3 Crazyflies, but found {}. "
            "Edit launch/crazyflies.yaml.".format(len(cfs)))


def main():
    swarm = None
    timeHelper = None
    cfs = []

    try:
        swarm = Crazyswarm()
        timeHelper = swarm.timeHelper
        cfs = swarm.allcfs.crazyflies

        validate_three_cfs(cfs)

        print("Field: x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}]".format(
            FIELD_X[0], FIELD_X[1], FIELD_Y[0], FIELD_Y[1]))
        print("Nominal takeoff x: {:.2f} m (no extra mission x-limit)".format(
            NOMINAL_TAKEOFF_X))
        print("Obstacle center={}, radius={:.2f} m, diameter={:.2f} m, planning radius={:.2f} m".format(
            OBSTACLE_CENTER.tolist(), OBSTACLE_RADIUS, OBSTACLE_DIAMETER, PLAN_RADIUS))
        print("Target={}, encircle radius={:.2f} m".format(
            TARGET.tolist(), ENCIRCLE_RADIUS))

        # Optional inter-Crazyflie collision avoidance.
        if ENABLE_CF_COLLISION_AVOIDANCE:
            for cf in cfs:
                cf.enableCollisionAvoidance(cfs, CF_COLLISION_RADII)

        # 1) Take off vertically from wherever each Crazyflie currently is.
        print("Taking off...")
        for cf in cfs:
            cf.takeoff(targetHeight=FLIGHT_HEIGHT, duration=TAKEOFF_DURATION)
        timeHelper.sleep(TAKEOFF_DURATION + 1.0)

        # 2) Read ACTUAL positions after takeoff.
        starts = [np.asarray(cf.position(), dtype=float) for cf in cfs]

        for i, cf in enumerate(cfs):
            print("cf{} current position: {}".format(cf.id, starts[i]))
            if not inside_field_xy(starts[i][:2]):
                print("Warning: cf{} starts outside the nominal field; "
                      "the planner will allow it to enter the field smoothly.".format(cf.id))

        # 3) Assign final encirclement goals according to current y ordering.
        goals_sorted = encircle_goals()
        start_order = sorted(range(3), key=lambda i: starts[i][1])

        assigned_goals = [None, None, None]
        for rank, idx in enumerate(start_order):
            assigned_goals[idx] = goals_sorted[rank]

        # 4) Select obstacle-bypass corridors and build smooth paths.
        lanes = choose_lanes(starts, assigned_goals)

        curves = []
        arc_tables = []
        lengths = []

        for i, cf in enumerate(cfs):
            curve = build_smooth_path(starts[i], assigned_goals[i], lanes[i])
            s = curve_arclength(curve)
            curves.append(curve)
            arc_tables.append(s)
            lengths.append(s[-1])

            print("cf{} -> goal {}, lane y={:.2f}, path length={:.2f} m".format(
                cf.id,
                np.round(assigned_goals[i], 3),
                lanes[i],
                lengths[i]))

        # Use one common duration so all three arrive together.
        move_duration = max(MIN_MOVE_DURATION, 1.875 * max(lengths) / MAX_PATH_SPEED)
        print("Executing smooth encirclement trajectory, duration={:.2f} s".format(
            move_duration))

        # 5) Stream smooth absolute position setpoints at 50 Hz.
        steps = int(math.ceil(move_duration * CMD_RATE))
        for k in range(steps + 1):
            if timeHelper.isShutdown():
                safe_land(cfs, timeHelper,
                          "Crazyswarm shutdown requested during trajectory.")
                return

            # Runtime guard at about 10 Hz. The extra x<=1.0 constraint has
            # been removed; only the unchanged physical field boundary is hard.
            if k % max(1, int(CMD_RATE / 10.0)) == 0:
                for cf in cfs:
                    actual = np.asarray(cf.position(), dtype=float)
                    if (actual[0] < FIELD_X[0] or actual[0] > FIELD_X[1] or
                            actual[1] < FIELD_Y[0] or actual[1] > FIELD_Y[1]):
                        raise RuntimeError(
                            "cf{} exceeded physical field boundary: position={}".format(
                                cf.id, np.round(actual, 3)))

            tau = float(k) / float(max(steps, 1))
            progress = quintic_smoothstep(tau)

            for i, cf in enumerate(cfs):
                p = sample_curve_by_fraction(curves[i], arc_tables[i], progress)
                cf.cmdPosition(p, yaw=0.0)

            timeHelper.sleepForRate(CMD_RATE)

        # Hold exact encirclement points.
        hold_steps = int(HOLD_TIME * CMD_RATE)
        for hold_k in range(hold_steps):
            if timeHelper.isShutdown():
                safe_land(cfs, timeHelper,
                          "Crazyswarm shutdown requested while holding formation.")
                return

            if hold_k % max(1, int(CMD_RATE / 10.0)) == 0:
                for cf in cfs:
                    actual = np.asarray(cf.position(), dtype=float)
                    if (actual[0] < FIELD_X[0] or actual[0] > FIELD_X[1] or
                            actual[1] < FIELD_Y[0] or actual[1] > FIELD_Y[1]):
                        raise RuntimeError(
                            "cf{} exceeded physical field boundary while holding: position={}".format(
                                cf.id, np.round(actual, 3)))

            for i, cf in enumerate(cfs):
                cf.cmdPosition(assigned_goals[i], yaw=0.0)
            timeHelper.sleepForRate(CMD_RATE)

        print("Encirclement complete.")

        if AUTO_LAND:
            safe_land(cfs, timeHelper, "Mission complete.")

    except KeyboardInterrupt:
        # Ctrl+C no longer leaves the vehicles in the air.
        if timeHelper is not None and cfs:
            safe_land(cfs, timeHelper, "Keyboard interrupt (Ctrl+C).")
        else:
            print("\n[SAFETY] Keyboard interrupt before Crazyflie control became available.")
        return

    except Exception as e:
        # Any planning, configuration, command, or runtime error after
        # Crazyswarm initialization is converted into an in-place landing.
        print("\n[SAFETY] Task error: {}".format(e))
        if timeHelper is not None and cfs:
            safe_land(cfs, timeHelper, "Aborting mission safely.")
        else:
            print("[SAFETY] Crazyflie control was not initialized, so no landing command can be sent.")
        return


if __name__ == "__main__":
    main()
