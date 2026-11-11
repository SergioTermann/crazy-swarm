# Crazy Swarm

A Crazyflie swarm experimentation workspace built on ROS 1 and Crazyswarm, with code for formation control, trajectory tracking, obstacle avoidance, dynamic leader reconfiguration, and simulation and experiment data.

This repository brings together scripts and dependencies for developing swarm control experiments. Python scripts compute reference positions, velocities, or accelerations and connect to simulation or ROS control interfaces through `pycrazyswarm`. The workspace also includes Crazyflie communication tools, motion capture interfaces, and object tracking dependencies.

## Features

- **Basic flight and trajectory control:** takeoff, hover, landing, position and velocity commands, full-state commands, and examples using CSV and polynomial trajectories.
- **Formation control and following:** delayed takeoff, straight-line following, and formation convergence for three UAVs, along with additional formation scripts and swarm configurations.
- **Obstacle avoidance and dynamic reconfiguration:** artificial potential field experiments, risk-aware dynamic leader reconfiguration, and an HRA-APF controller implementation.
- **LMPC simulation experiments:** several iterations covering formation tracking, encirclement, predictive control, and an auxiliary backstepping controller.
- **Simulation and data recording:** `pycrazyswarm` supports Matplotlib, VisPy, and headless visualization backends. Selected scripts save CSV, TXT, and NPZ data, as well as PNG result plots.
- **Localization and hardware integration:** motion capture interfaces, object tracking libraries, and launch configurations for Vicon and Lighthouse setups.

These features describe the code and configurations currently included in the repository. The scripts have not all been validated in a single, consistent environment.

## Repository Structure

```text
crazy-swarm/
├── src/
│   ├── CMakeLists.txt           # Symlink to ROS Noetic catkin
│   ├── crazyswarm/
│   │   ├── scripts/             # Controllers, simulations, trajectories, and data
│   │   │   ├── pycrazyswarm/    # Python API and simulation backend
│   │   │   └── outputs/         # Selected experiment outputs
│   │   ├── launch/              # ROS launch files and swarm YAML configurations
│   │   ├── src/                 # C++ server, teleoperation, and localization nodes
│   │   ├── msg/                 # ROS message definitions
│   │   ├── srv/                 # ROS service definitions
│   │   └── externalDependencies/
│   ├── crazyflie_tools/         # Crazyflie utilities and crazyflie_cpp
│   ├── externalDependencies/    # libobjecttracker
│   └── crazyflie-firmware/      # Partial firmware build artifacts in this snapshot
└── README.md
```

## Code Guide

| Entry point | Description |
| --- | --- |
| [`hello_world.py`](src/crazyswarm/scripts/hello_world.py) | Basic takeoff, hover, and landing example |
| [`three_uav_formation_follow.py`](src/crazyswarm/scripts/three_uav_formation_follow.py) | Delayed takeoff, following, and formation convergence for three UAVs |
| [`paper_controller.py`](src/crazyswarm/scripts/paper_controller.py) | Risk-aware dynamic leader reconfiguration and HRA-APF controller |
| [`hra_apf_crazyswarm.py`](src/crazyswarm/scripts/hra_apf_crazyswarm.py) | Controller integration with the Crazyswarm Python API |
| [`lmpc_crazyswarm_sim16.py`](src/crazyswarm/scripts/lmpc_crazyswarm_sim16.py) | An LMPC formation tracking and encirclement simulation version |
| [`pycrazyswarm`](src/crazyswarm/scripts/pycrazyswarm) | Shared interfaces for hardware and simulation |
| [`hover_swarm.launch`](src/crazyswarm/launch/hover_swarm.launch) | Server, localization, joystick, and RViz configuration |
| [`launch`](src/crazyswarm/launch) | UAV IDs, initial positions, and experiment configurations |

## Clone the Repository

```bash
git clone git@github.com:SergioTermann/crazy-swarm.git
cd crazy-swarm
```

The repository is currently private, so cloning requires access. Included third-party dependencies are stored directly as files and do not require submodule initialization. Nested `.gitmodules` files are retained as upstream metadata.

## Environment and Setup

The project uses a ROS 1 catkin workspace layout. Its `src/CMakeLists.txt` currently points to `/opt/ros/noetic/share/catkin/cmake/toplevel.cmake`. Use a Linux environment compatible with ROS Noetic when preparing dependencies. The C++ components depend on catkin, PCL, Crazyflie communication libraries, and motion capture libraries. Python dependencies vary by script and include NumPy and PyYAML, with SciPy, Matplotlib, or VisPy used by selected simulations.

This repository is a snapshot of an existing experiment workspace, not yet a complete installation package. Before building or running it, address the following setup requirements:

1. **Restore firmware sources and check paths.** `src/crazyflie-firmware` currently contains only a subset of object files. The C++ build and `pycrazyswarm/cfsim` also reference firmware headers and C sources outside this workspace. Obtain compatible Crazyflie firmware sources and check the relative paths.
2. **Update machine-specific paths.** The default YAML path in `pycrazyswarm/crazyswarm_py.py` and paths in some experiment scripts still refer to the original experiment machine. Update them for your environment or explicitly pass `crazyflies_yaml` where supported.
3. **Choose an experiment configuration.** Review the YAML files in `launch/` and the scripts directory. UAV IDs, radio channels, and initial positions must match the selected control script.
4. **Select the backend.** The shared API supports `--sim`, with visualization options including `--vis mpl`, `--vis vispy`, and `--vis null`. Argument handling and default backends can differ between scripts; inspect the selected entry point first.

After preparing dependencies, firmware paths, and configuration, proceed with the catkin build and script validation. Hardware experiments also require compatible Crazyflie vehicles, radio hardware, and a localization system. Match the localization type, host, and flight controller parameters in the launch files to the experiment setup.

## Experiment Data

The `scripts/` directory includes trajectory CSV files, TXT and NPZ data, result images, and selected experiment outputs. The `scripts/outputs/` directory contains data for `human_uav_three_trial`. These files preserve the connection between existing experiment code and its outputs; they are not a standardized benchmark dataset. Output fields, locations, and parameters vary by script, so interpret each dataset alongside the script that generated it.

Version control excludes the workspace-root `build/`, `devel/`, `install/`, and log directories, as well as Python caches and local environment files. Flight experiments and historical results were not rerun or reproduced as part of this repository preparation.

## Upstream Projects and Licensing

This workspace builds on [Crazyswarm](https://github.com/USC-ACTLab/crazyswarm) and its dependencies. Refer to the [Crazyswarm documentation](https://crazyswarm.readthedocs.io/) for API and environment setup guidance.

Third-party components retain their original licenses and author information; consult their respective directories. The repository root does not currently declare a single license covering all added experiment code.
