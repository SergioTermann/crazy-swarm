# Crazy Swarm

基于 ROS 1 与 Crazyswarm 的 Crazyflie 多无人机实验工作空间，汇集编队控制、轨迹跟踪、避障、动态领导者重配置以及仿真与实验数据。

本项目用于保存和迭代无人机集群控制实验：上层 Python 脚本计算参考位置、速度或加速度，通过 `pycrazyswarm` 接入仿真后端或 ROS 控制接口；底层包含 Crazyflie 通信工具、运动捕捉接口与目标跟踪依赖。

## 主要内容

- **基础飞行与轨迹控制**：起飞、悬停、降落、位置/速度指令、全状态指令，以及 CSV 轨迹和多项式轨迹相关示例。
- **多机编队与跟随**：三机延迟起飞、直线跟随、追近编队，以及多个编队实验脚本与机群配置。
- **避障与动态重配置**：人工势场相关实验，以及风险感知动态领导者重配置与 HRA-APF 控制器实现。
- **LMPC 仿真实验**：包含多个迭代版本，涉及编队跟踪、围捕、预测控制与反步辅助控制。
- **仿真与结果记录**：`pycrazyswarm` 提供仿真入口，可选择 Matplotlib、VisPy 或无可视化后端；部分脚本保存 CSV、TXT、NPZ 数据及 PNG 结果图。
- **定位与硬件接入**：包含运动捕捉接口、目标跟踪库，以及 Vicon 和 Lighthouse 相关启动配置。

以上描述对应仓库现有代码与配置，不代表所有脚本已在统一环境下完成验证。

## 目录结构

```text
crazy-swarm/
├── src/
│   ├── CMakeLists.txt           # 指向 ROS Noetic catkin 的符号链接
│   ├── crazyswarm/
│   │   ├── scripts/            # 控制算法、仿真脚本、轨迹与实验数据
│   │   │   ├── pycrazyswarm/   # Python API 与仿真后端
│   │   │   └── outputs/        # 部分实验输出
│   │   ├── launch/             # ROS 启动文件与机群 YAML 配置
│   │   ├── src/                # C++ 服务端、遥控及定位辅助节点
│   │   ├── msg/                # ROS 消息定义
│   │   ├── srv/                # ROS 服务定义
│   │   └── externalDependencies/
│   ├── crazyflie_tools/        # Crazyflie 工具与 crazyflie_cpp
│   ├── externalDependencies/   # libobjecttracker
│   └── crazyflie-firmware/     # 当前快照中的局部固件构建文件
└── README.md
```

## 代码阅读入口

| 入口 | 内容 |
| --- | --- |
| [`hello_world.py`](src/crazyswarm/scripts/hello_world.py) | 起飞、悬停与降落基础示例 |
| [`three_uav_formation_follow.py`](src/crazyswarm/scripts/three_uav_formation_follow.py) | 三机延迟起飞、跟随和追近编队 |
| [`paper_controller.py`](src/crazyswarm/scripts/paper_controller.py) | 风险感知动态领导者重配置与 HRA-APF 控制器 |
| [`hra_apf_crazyswarm.py`](src/crazyswarm/scripts/hra_apf_crazyswarm.py) | 将控制器接入 Crazyswarm Python API |
| [`lmpc_crazyswarm_sim16.py`](src/crazyswarm/scripts/lmpc_crazyswarm_sim16.py) | LMPC 编队跟踪与围捕仿真版本 |
| [`pycrazyswarm`](src/crazyswarm/scripts/pycrazyswarm) | 真实飞行与仿真的公共接口 |
| [`hover_swarm.launch`](src/crazyswarm/launch/hover_swarm.launch) | 服务端、定位、手柄与 RViz 配置 |
| [`launch`](src/crazyswarm/launch) | 机群 ID、初始位置及不同实验配置 |

## 获取项目

```bash
git clone git@github.com:SergioTermann/crazy-swarm.git
cd crazy-swarm
```

仓库当前为私有仓库，克隆需要具备访问权限。已包含的第三方依赖按普通文件保存，无需为这些文件执行子模块初始化；目录内保留的 `.gitmodules` 是上游元数据。

## 环境与运行准备

项目使用 ROS 1 的 catkin 工作空间结构，当前 `src/CMakeLists.txt` 指向 `/opt/ros/noetic/share/catkin/cmake/toplevel.cmake`。建议在与 ROS Noetic 配套的 Linux 环境中整理依赖。C++ 部分涉及 catkin、PCL、Crazyflie 通信及运动捕捉库；Python 依赖随脚本变化，主要包括 NumPy、PyYAML，以及部分仿真使用的 SciPy、Matplotlib 或 VisPy。

当前仓库是已有实验工作空间的快照，尚不是可直接一键安装的完整发行包。配置环境时需要处理以下已知事项：

1. **补齐固件源码及路径**：`src/crazyflie-firmware` 当前仅有局部目标文件；C++ 构建和 `pycrazyswarm/cfsim` 还引用工作空间之外的固件头文件与 C 源码。需要准备兼容的 Crazyflie 固件源码，并核对相对路径。
2. **调整本机路径**：`pycrazyswarm/crazyswarm_py.py` 的默认 YAML 路径及部分实验脚本仍含原实验机器的绝对路径。应改为当前环境中的路径，或在脚本支持时显式传入 `crazyflies_yaml`。
3. **选择实验配置**：核对 `launch/` 和脚本目录下的 YAML，确保无人机 ID、无线信道、初始位置与控制脚本要求一致。
4. **选择运行后端**：公共 API 支持 `--sim`，并提供 `--vis mpl`、`--vis vispy`、`--vis null` 等仿真参数；各实验脚本对参数的解析与默认后端可能不同，应先阅读对应入口。

完成依赖、固件路径和配置适配后，再进行 catkin 构建与脚本验证。真实飞行还需要匹配的 Crazyflie、无线通信设备和定位系统；启动文件中的定位类型、主机及飞控参数需与实验环境对应。

## 实验数据

`scripts/` 中保留了轨迹 CSV、TXT、NPZ、结果图片和部分实验输出，`scripts/outputs/` 下包含 `human_uav_three_trial` 数据。它们用于关联现有实验代码与结果，不构成统一整理的评测数据集。不同脚本的输出字段、保存位置与参数可能不同，请结合生成脚本解释数据。

版本控制排除了工作空间根目录下的 `build/`、`devel/`、`install/`、日志目录，以及 Python 缓存和本地环境文件。此次整理未重新运行飞行实验或复现历史结果。

## 上游项目与许可

本工作空间基于 [Crazyswarm](https://github.com/USC-ACTLab/crazyswarm) 及其相关依赖，接口和环境配置可参考 [Crazyswarm 文档](https://crazyswarm.readthedocs.io/)。

各第三方组件保留其原有许可证与作者信息，请分别查阅对应目录。仓库根目录目前未声明覆盖全部新增实验代码的统一许可证。
