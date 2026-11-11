#!/usr/bin/env python3
"""Plan and, only after explicit confirmation, fly an M5-derived Crazyflie route.

The four geographic transfer legs are evidence-selected from the rank-1 M5
action sequence for 2019-05-15--2019-05-18.  Longitude/latitude coordinates
are converted to local metric coordinates and uniformly scaled into an indoor
3 m x 2 m x 2 m coordinate frame.  The default mode performs planning and
safety validation only; it never imports pycrazyswarm or connects to hardware.

Coordinate convention:
    x in [-1.5, 1.5] m, y in [-1.0, 1.0] m, z in [0.0, 2.0] m
    room-floor center and configured initial position: (0, 0, 0) m
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOM_MIN = (-1.50, -1.00, 0.00)
ROOM_MAX = (1.50, 1.00, 2.00)

# Deliberately smaller than the physical room to retain nominal clearance.
TRAJECTORY_MIN = (-1.25, -0.75, 0.45)
TRAJECTORY_MAX = (1.25, 0.75, 1.45)
MAP_HALF_WIDTH_X = 1.15
MAP_HALF_WIDTH_Y = 0.65

START_FLOOR = (-1.3, 0.5, 0.0)
CRUISE_Z = 0.90
SERVICE_Z = 0.55
LAND_Z = 0.05

TAKEOFF_SECONDS = 3.0
LAND_SECONDS = 3.0
BOARD_SECONDS = 1.5
ALIGHT_SECONDS = 1.5
DAY_SEPARATOR_SECONDS = 0.8
FINAL_HOVER_SECONDS = 1.0

MAX_SPEED = 0.45          # m/s, nominal planned-trajectory limit
MAX_ACCELERATION = 0.90   # m/s^2, nominal planned-trajectory limit
INITIAL_POSITION_TOLERANCE = 0.05
VALIDATION_STEP = 0.01
TRAJECTORY_ID = 0
CONFIRMATION_PHRASE = "FLY M5 ROUTE"

# Maxima of the first two derivatives of the seventh-order smootherstep.
Q1_MAX = 2.1875
Q2_MAX = 7.5131884044


@dataclass(frozen=True)
class GeoTask:
    date: str
    split: str
    route_map_id: str
    source_segment_id: int
    pickup_station: str
    pickup_lon: float
    pickup_lat: float
    receiving_station: str
    receiving_lon: float
    receiving_lat: float
    loaded_distance_km: float
    transferred_passengers: int


@dataclass(frozen=True)
class IndoorTask:
    source: GeoTask
    pickup_xy: tuple[float, float]
    receiving_xy: tuple[float, float]


@dataclass(frozen=True)
class Segment:
    label: str
    phase: str
    duration: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    yaw_start: float = 0.0
    yaw_end: float = 0.0


SOURCE_TASKS: tuple[GeoTask, ...] = (
    GeoTask(
        date="2019-05-15",
        split="validation",
        route_map_id="episode_2249__2019-05-15__composite__rank_01",
        source_segment_id=443,
        pickup_station="NS_0759",
        pickup_lon=116.374580,
        pickup_lat=39.897166,
        receiving_station="NS_0494",
        receiving_lon=116.363300,
        receiving_lat=39.895870,
        loaded_distance_km=0.9730199543330486,
        transferred_passengers=15,
    ),
    GeoTask(
        date="2019-05-16",
        split="validation",
        route_map_id="episode_2249__2019-05-16__composite__rank_01",
        source_segment_id=453,
        pickup_station="NS_0876",
        pickup_lon=116.384320,
        pickup_lat=39.897840,
        receiving_station="NS_0708",
        receiving_lon=116.382610,
        receiving_lat=39.904280,
        loaded_distance_km=0.7308021488969801,
        transferred_passengers=15,
    ),
    GeoTask(
        date="2019-05-17",
        split="frozen_test",
        route_map_id="episode_2249__2019-05-17__composite__rank_01",
        source_segment_id=447,
        pickup_station="NS_0759",
        pickup_lon=116.374580,
        pickup_lat=39.897166,
        receiving_station="NS_1197",
        receiving_lon=116.374080,
        receiving_lat=39.904570,
        loaded_distance_km=0.8243924648516193,
        transferred_passengers=15,
    ),
    GeoTask(
        date="2019-05-18",
        split="frozen_test",
        route_map_id="episode_2249__2019-05-18__composite__rank_01",
        source_segment_id=441,
        pickup_station="NS_1368",
        pickup_lon=116.337800,
        pickup_lat=39.907650,
        receiving_station="NS_1100",
        receiving_lon=116.328910,
        receiving_lat=39.909590,
        loaded_distance_km=0.7883538032043917,
        transferred_passengers=15,
    ),
)


def local_xy_m(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Equirectangular projection appropriate for this small local extent."""
    earth_radius_m = 6_371_000.0
    x = earth_radius_m * math.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = earth_radius_m * math.radians(lat - lat0)
    return x, y


def map_tasks(tasks: Sequence[GeoTask]) -> tuple[list[IndoorTask], dict[str, float]]:
    all_lon = [value for task in tasks for value in (task.pickup_lon, task.receiving_lon)]
    all_lat = [value for task in tasks for value in (task.pickup_lat, task.receiving_lat)]
    lon0 = 0.5 * (min(all_lon) + max(all_lon))
    lat0 = 0.5 * (min(all_lat) + max(all_lat))

    projected: list[tuple[tuple[float, float], tuple[float, float]]] = []
    extrema_x: list[float] = []
    extrema_y: list[float] = []
    for task in tasks:
        pickup = local_xy_m(task.pickup_lon, task.pickup_lat, lon0, lat0)
        receiving = local_xy_m(task.receiving_lon, task.receiving_lat, lon0, lat0)
        projected.append((pickup, receiving))
        extrema_x.extend((pickup[0], receiving[0]))
        extrema_y.extend((pickup[1], receiving[1]))

    max_abs_x = max(abs(value) for value in extrema_x)
    max_abs_y = max(abs(value) for value in extrema_y)
    scale = min(MAP_HALF_WIDTH_X / max_abs_x, MAP_HALF_WIDTH_Y / max_abs_y)

    mapped: list[IndoorTask] = []
    for task, (pickup, receiving) in zip(tasks, projected):
        mapped.append(
            IndoorTask(
                source=task,
                pickup_xy=(pickup[0] * scale, pickup[1] * scale),
                receiving_xy=(receiving[0] * scale, receiving[1] * scale),
            )
        )
    return mapped, {"reference_lon": lon0, "reference_lat": lat0, "scale": scale}


def distance_3d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((right - left) ** 2 for left, right in zip(a, b)))


def motion_duration(start: Sequence[float], end: Sequence[float]) -> float:
    """Duration satisfying nominal smootherstep speed and acceleration limits."""
    distance = distance_3d(start, end)
    if distance <= 1e-12:
        return 0.60
    speed_duration = Q1_MAX * distance / MAX_SPEED
    acceleration_duration = math.sqrt(Q2_MAX * distance / MAX_ACCELERATION)
    return max(0.80, speed_duration, acceleration_duration)


def add_motion(
    segments: list[Segment], label: str, phase: str,
    start: tuple[float, float, float], end: tuple[float, float, float],
) -> tuple[float, float, float]:
    segments.append(Segment(label, phase, motion_duration(start, end), start, end))
    return end


def add_hover(
    segments: list[Segment], label: str, phase: str,
    position: tuple[float, float, float], duration: float,
) -> tuple[float, float, float]:
    segments.append(Segment(label, phase, duration, position, position))
    return position


def build_segments(tasks: Sequence[IndoorTask]) -> list[Segment]:
    segments: list[Segment] = []
    current = (0.0, 0.0, CRUISE_Z)

    for index, task in enumerate(tasks, start=1):
        prefix = f"day{index}_{task.source.date}"
        pickup_cruise = (task.pickup_xy[0], task.pickup_xy[1], CRUISE_Z)
        pickup_service = (task.pickup_xy[0], task.pickup_xy[1], SERVICE_Z)
        receiving_cruise = (task.receiving_xy[0], task.receiving_xy[1], CRUISE_Z)
        receiving_service = (task.receiving_xy[0], task.receiving_xy[1], SERVICE_Z)

        current = add_motion(segments, prefix + "_empty_reposition", "empty", current, pickup_cruise)
        current = add_motion(segments, prefix + "_descend_pickup", "service", current, pickup_service)
        current = add_hover(segments, prefix + "_boarding", "boarding", current, BOARD_SECONDS)
        current = add_motion(segments, prefix + "_climb_pickup", "service", current, pickup_cruise)
        current = add_motion(segments, prefix + "_loaded_transfer", "loaded", current, receiving_cruise)
        current = add_motion(segments, prefix + "_descend_receiving", "service", current, receiving_service)
        current = add_hover(segments, prefix + "_alighting", "alighting", current, ALIGHT_SECONDS)
        current = add_motion(segments, prefix + "_climb_receiving", "service", current, receiving_cruise)
        current = add_hover(segments, prefix + "_separator", "separator", current, DAY_SEPARATOR_SECONDS)

    center_cruise = (0.0, 0.0, CRUISE_Z)
    current = add_motion(segments, "return_to_room_center", "return", current, center_cruise)
    add_hover(segments, "final_hover", "hover", current, FINAL_HOVER_SECONDS)
    return segments


def smootherstep(s: float) -> float:
    return 35.0 * s**4 - 84.0 * s**5 + 70.0 * s**6 - 20.0 * s**7


def smootherstep_d1(s: float) -> float:
    return 140.0 * s**3 - 420.0 * s**4 + 420.0 * s**5 - 140.0 * s**6


def smootherstep_d2(s: float) -> float:
    return 420.0 * s**2 - 1680.0 * s**3 + 2100.0 * s**4 - 840.0 * s**5


def interpolate(segment: Segment, elapsed: float) -> tuple[tuple[float, float, float], float, float]:
    s = min(1.0, max(0.0, elapsed / segment.duration))
    q = smootherstep(s)
    q1 = smootherstep_d1(s) / segment.duration
    q2 = smootherstep_d2(s) / (segment.duration**2)
    delta = tuple(end - start for start, end in zip(segment.start, segment.end))
    position = tuple(start + q * change for start, change in zip(segment.start, delta))
    speed = math.sqrt(sum((q1 * change) ** 2 for change in delta))
    acceleration = math.sqrt(sum((q2 * change) ** 2 for change in delta))
    return position, speed, acceleration


def validate_plan(segments: Sequence[Segment]) -> dict[str, float | int]:
    if not segments:
        raise ValueError("The trajectory contains no segments.")
    expected_start = (0.0, 0.0, CRUISE_Z)
    if distance_3d(segments[0].start, expected_start) > 1e-9:
        raise ValueError("Uploaded trajectory does not begin above the room center.")

    max_speed = 0.0
    max_acceleration = 0.0
    min_xyz = [math.inf, math.inf, math.inf]
    max_xyz = [-math.inf, -math.inf, -math.inf]
    sample_count = 0

    previous_end = segments[0].start
    for segment in segments:
        if distance_3d(previous_end, segment.start) > 1e-8:
            raise ValueError(f"Discontinuous segment boundary before {segment.label}.")
        steps = max(2, math.ceil(segment.duration / VALIDATION_STEP))
        for sample_index in range(steps + 1):
            elapsed = segment.duration * sample_index / steps
            position, speed, acceleration = interpolate(segment, elapsed)
            sample_count += 1
            max_speed = max(max_speed, speed)
            max_acceleration = max(max_acceleration, acceleration)
            for axis in range(3):
                min_xyz[axis] = min(min_xyz[axis], position[axis])
                max_xyz[axis] = max(max_xyz[axis], position[axis])
                if position[axis] < TRAJECTORY_MIN[axis] - 1e-9 or position[axis] > TRAJECTORY_MAX[axis] + 1e-9:
                    raise ValueError(
                        f"Safe trajectory bound violated by {segment.label}: "
                        f"axis={axis}, value={position[axis]:.6f}."
                    )
                if position[axis] < ROOM_MIN[axis] - 1e-9 or position[axis] > ROOM_MAX[axis] + 1e-9:
                    raise ValueError(f"Physical room bound violated by {segment.label}.")
        previous_end = segment.end

    if distance_3d(segments[-1].end, expected_start) > 1e-9:
        raise ValueError("Uploaded trajectory does not return above the room center.")
    if max_speed > MAX_SPEED + 1e-6:
        raise ValueError(f"Speed audit failed: {max_speed:.6f} m/s.")
    if max_acceleration > MAX_ACCELERATION + 1e-6:
        raise ValueError(f"Acceleration audit failed: {max_acceleration:.6f} m/s^2.")

    return {
        "segments": len(segments),
        "samples": sample_count,
        "duration_s": sum(segment.duration for segment in segments),
        "min_x": min_xyz[0], "max_x": max_xyz[0],
        "min_y": min_xyz[1], "max_y": max_xyz[1],
        "min_z": min_xyz[2], "max_z": max_xyz[2],
        "max_speed_mps": max_speed,
        "max_acceleration_mps2": max_acceleration,
    }


def polynomial_coefficients(start: float, end: float, duration: float) -> list[float]:
    delta = end - start
    return [
        start, 0.0, 0.0, 0.0,
        35.0 * delta / duration**4,
        -84.0 * delta / duration**5,
        70.0 * delta / duration**6,
        -20.0 * delta / duration**7,
    ]


def make_crazyswarm_trajectory(segments: Sequence[Segment], uav_trajectory_module):
    trajectory = uav_trajectory_module.Trajectory()
    trajectory.polynomials = []
    for segment in segments:
        x_coefficients = polynomial_coefficients(segment.start[0], segment.end[0], segment.duration)
        y_coefficients = polynomial_coefficients(segment.start[1], segment.end[1], segment.duration)
        z_coefficients = polynomial_coefficients(segment.start[2], segment.end[2], segment.duration)
        yaw_coefficients = polynomial_coefficients(segment.yaw_start, segment.yaw_end, segment.duration)
        # Match the Polynomial4D(duration, x, y, z, yaw) interface used by
        # the user-supplied legacy pycrazyswarm reference program.
        trajectory.polynomials.append(
            uav_trajectory_module.Polynomial4D(
                segment.duration,
                x_coefficients,
                y_coefficients,
                z_coefficients,
                yaw_coefficients,
            )
        )
    trajectory.duration = sum(segment.duration for segment in segments)
    return trajectory


def export_plan(path: Path, segments: Sequence[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "segment_index", "label", "phase", "duration_s",
            "start_x_m", "start_y_m", "start_z_m",
            "end_x_m", "end_y_m", "end_z_m",
        ])
        for index, segment in enumerate(segments, start=1):
            writer.writerow([
                index, segment.label, segment.phase, f"{segment.duration:.9f}",
                *(f"{value:.9f}" for value in segment.start),
                *(f"{value:.9f}" for value in segment.end),
            ])


def print_plan(tasks: Sequence[IndoorTask], metadata: dict[str, float], audit: dict[str, float | int]) -> None:
    print("MODE: PLAN AND VALIDATE ONLY")
    print("No CrazySwarm connection has been opened and no vehicle command has been sent.")
    print(f"Physical room: x={ROOM_MIN[0]:.2f}..{ROOM_MAX[0]:.2f} m, "
          f"y={ROOM_MIN[1]:.2f}..{ROOM_MAX[1]:.2f} m, z={ROOM_MIN[2]:.2f}..{ROOM_MAX[2]:.2f} m")
    print(f"Uploaded-route safety envelope: x={TRAJECTORY_MIN[0]:.2f}..{TRAJECTORY_MAX[0]:.2f} m, "
          f"y={TRAJECTORY_MIN[1]:.2f}..{TRAJECTORY_MAX[1]:.2f} m, "
          f"z={TRAJECTORY_MIN[2]:.2f}..{TRAJECTORY_MAX[2]:.2f} m")
    print(f"Projection reference: lon={metadata['reference_lon']:.7f}, lat={metadata['reference_lat']:.7f}")
    print(f"Uniform indoor/source scale: {metadata['scale']:.12f} m per source metre")
    for task in tasks:
        print(
            f"{task.source.date} [{task.source.split}] "
            f"{task.source.pickup_station} -> {task.source.receiving_station}: "
            f"pickup=({task.pickup_xy[0]:.4f},{task.pickup_xy[1]:.4f}), "
            f"receiving=({task.receiving_xy[0]:.4f},{task.receiving_xy[1]:.4f})"
        )
    print("AUDIT: PASS")
    for key, value in audit.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def execute_flight(
    segments: Sequence[Segment],
    crazyswarm_args=None
) -> None:
    try:
        from pycrazyswarm import Crazyswarm
        import uav_trajectory
    except ImportError as exc:
        raise RuntimeError(
            "Physical execution requires the legacy pycrazyswarm and "
            "uav_trajectory packages used by the supplied reference script."
        ) from exc

    swarm = Crazyswarm(
    args=crazyswarm_args
)
    time_helper = swarm.timeHelper
    crazyflies = swarm.allcfs.crazyflies
    if len(crazyflies) != 1:
        raise RuntimeError(f"Expected exactly one Crazyflie, detected {len(crazyflies)}.")

    cf = crazyflies[0]
    initial = tuple(float(value) for value in cf.initialPosition)
    if distance_3d(initial, START_FLOOR) > INITIAL_POSITION_TOLERANCE:
        raise RuntimeError(
            f"Configured initialPosition {initial} is not within "
            f"{INITIAL_POSITION_TOLERANCE:.2f} m of the room-floor center {START_FLOOR}."
        )

    trajectory = make_crazyswarm_trajectory(segments, uav_trajectory)
    cf.uploadTrajectory(TRAJECTORY_ID, 0, trajectory)
    airborne = False
    try:
        cf.takeoff(targetHeight=CRUISE_Z, duration=TAKEOFF_SECONDS)
        airborne = True
        time_helper.sleep(TAKEOFF_SECONDS + 0.5)
        cf.startTrajectory(TRAJECTORY_ID, timescale=1.0, reverse=False, relative=False)
        time_helper.sleep(trajectory.duration + 0.5)
        cf.land(targetHeight=LAND_Z, duration=LAND_SECONDS)
        time_helper.sleep(LAND_SECONDS + 0.5)
        airborne = False
    finally:
        if airborne:
            print("Execution interrupted; requesting controlled landing.", file=sys.stderr)
            cf.land(targetHeight=LAND_Z, duration=LAND_SECONDS)
            time_helper.sleep(LAND_SECONDS + 0.5)


def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )


    # ========================================================
    # 本程序自己的参数
    # ========================================================

    parser.add_argument(
        "--export-plan",
        type=Path,
        help="Write the audited segment plan to CSV."
    )


    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow physical Crazyflie execution after all guards pass."
    )


    parser.add_argument(
        "--yes-i-checked-room",
        action="store_true",
        help=(
            "Confirm that the room, mocap frame, props, "
            "batteries, and emergency stop were checked."
        ),
    )


    # ========================================================
    # 关键修改：
    #
    # 不再使用：
    #
    #     parser.parse_args()
    #
    # 而使用：
    #
    #     parser.parse_known_args()
    #
    # 本程序认识的参数进入 args
    #
    # Crazyswarm 参数：
    #
    # --sim
    # --vis
    # --dt
    # --video
    # --writecsv
    #
    # 等进入 crazyswarm_args
    # ========================================================

    args, crazyswarm_args = (
        parser.parse_known_args()
    )


    return (
        args,
        crazyswarm_args
    )

def main() -> int:

    args, crazyswarm_args = (
        parse_args()
    )


    mapped_tasks, metadata = (
        map_tasks(
            SOURCE_TASKS
        )
    )


    segments = build_segments(
        mapped_tasks
    )


    audit = validate_plan(
        segments
    )


    print_plan(
        mapped_tasks,
        metadata,
        audit
    )


    # ========================================================
    # 可选：输出轨迹规划CSV
    # ========================================================

    if args.export_plan:

        export_plan(
            args.export_plan,
            segments
        )

        print(
            "Audited segment plan written to: {}".format(
                args.export_plan.resolve()
            )
        )


    # ========================================================
    # 判断当前是不是仿真模式
    # ========================================================

    simulation_requested = (
        "--sim"
        in crazyswarm_args
    )


    # ========================================================
    # 仿真模式
    #
    # 不要求：
    #
    # --execute
    # --yes-i-checked-room
    # 输入 FLY M5 ROUTE
    #
    # 因为这里不会连接真实无人机。
    # ========================================================

    if simulation_requested:

        print("")
        print(
            "=============================================="
        )

        print(
            " SIMULATION EXECUTION REQUESTED"
        )

        print(
            "=============================================="
        )

        print(
            "Crazyswarm arguments:",
            crazyswarm_args
        )


        execute_flight(
            segments,
            crazyswarm_args
        )


        return 0


    # ========================================================
    # 如果既没有--sim，也没有--execute
    #
    # 只进行规划和审计
    # ========================================================

    if not args.execute:

        return 0


    # ========================================================
    # 以下才是真实无人机执行
    # ========================================================

    if not args.yes_i_checked_room:

        print(
            "REFUSED: --execute also requires "
            "--yes-i-checked-room.",
            file=sys.stderr
        )

        return 2


    print("")
    print(
        "PHYSICAL EXECUTION REQUESTED."
    )

    print(
        "Keep the emergency-stop control in hand "
        "and clear the entire flight volume."
    )


    confirmation = input(
        "Type exactly '{}' to continue: ".format(
            CONFIRMATION_PHRASE
        )
    ).strip()


    if confirmation != CONFIRMATION_PHRASE:

        print(
            "Physical execution cancelled; "
            "no CrazySwarm connection was opened."
        )

        return 2


    execute_flight(
        segments,
        crazyswarm_args
    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
