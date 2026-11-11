# ros_ws

ROS catkin workspace containing Crazyswarm scripts, Crazyflie tools, motion-capture dependencies, and experiment data.

## Contents

- `src/crazyswarm`: ROS package, Python control/simulation scripts, configurations, and experiment outputs.
- `src/crazyflie_tools`: Crazyflie utilities and C++ dependency sources.
- `src/externalDependencies`: object-tracking dependency sources.
- `src/crazyflie-firmware`: locally available firmware files.

This repository preserves the available workspace sources and data as a snapshot. Included third-party sources retain their original license files. Dependencies are stored directly in this repository rather than as Git submodules; nested `.gitmodules` files are retained as upstream metadata.

## Environment

The workspace's `src/CMakeLists.txt` points to the ROS Noetic catkin installation at `/opt/ros/noetic/share/catkin/cmake/toplevel.cmake`. Building requires an appropriate ROS environment and the dependencies declared by the packages. Generated `build`, `devel`, and `install` directories are excluded from version control.

Build and hardware operation have not been validated as part of this upload. Review script configuration and hardware requirements before running flight-control scripts.
