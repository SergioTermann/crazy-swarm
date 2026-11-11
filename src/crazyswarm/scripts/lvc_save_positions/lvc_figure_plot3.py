import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取四个无人机的位置数据
def read_drone_data(filename):
    """读取无人机位置数据文件，跳过标题行"""
    data = np.loadtxt(filename, skiprows=1)  # 跳过第一行标题
    t_steps = data[:, 0]  # 时间步
    positions = data[:, 1:4]  # 位置数据 (x, y, z)
    return t_steps, positions

# 读取四个无人机的数据
try:
    t1, pos1 = read_drone_data("drone_0_positions.txt")
    t2, pos2 = read_drone_data("drone_1_positions.txt")
    t3, pos3 = read_drone_data("drone_2_positions.txt")
    t4, pos4 = read_drone_data("drone_3_positions.txt")
    
    print(f"成功读取数据:")
    print(f"无人机1: {len(pos1)} 个数据点")
    print(f"无人机2: {len(pos2)} 个数据点")
    print(f"无人机3: {len(pos3)} 个数据点")
    print(f"无人机4: {len(pos4)} 个数据点")
    
except Exception as e:
    print(f"读取文件时出错: {e}")
    # 如果文件不存在或格式错误，创建示例数据用于演示
    print("创建示例数据用于演示...")
    t = np.arange(0, 100)
    pos1 = np.column_stack((np.sin(t/10), np.cos(t/10), t/100))
    pos2 = np.column_stack((np.sin(t/10 + 0.5), np.cos(t/10 + 0.5), t/100 + 0.2))
    pos3 = np.column_stack((np.sin(t/10 + 1.0), np.cos(t/10 + 1.0), t/100 + 0.4))
    pos4 = np.column_stack((np.sin(t/10 + 1.5), np.cos(t/10 + 1.5), t/100 + 0.6))

# 创建3D图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制四个无人机的轨迹，使用不同的标记和颜色
marker_size = 30  # 标记大小
line_width = 2.0  # 线宽

# 计算采样间隔，避免标记过于密集
sample_interval = max(1, len(pos1) // 20)

# 无人机1 - 红色圆形
ax.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], 'r-', linewidth=line_width, alpha=0.7)
ax.scatter(pos1[::sample_interval, 0], pos1[::sample_interval, 1], pos1[::sample_interval, 2], 
           c='red', marker='o', s=marker_size, label='无人机1', edgecolors='black')

# 无人机2 - 蓝色方形
ax.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], 'b-', linewidth=line_width, alpha=0.7)
ax.scatter(pos2[::sample_interval, 0], pos2[::sample_interval, 1], pos2[::sample_interval, 2], 
           c='blue', marker='s', s=marker_size, label='无人机2', edgecolors='black')

# 无人机3 - 绿色三角形
ax.plot(pos3[:, 0], pos3[:, 1], pos3[:, 2], 'g-', linewidth=line_width, alpha=0.7)
ax.scatter(pos3[::sample_interval, 0], pos3[::sample_interval, 1], pos3[::sample_interval, 2], 
           c='green', marker='^', s=marker_size, label='无人机3', edgecolors='black')

# 无人机4 - 橙色星形
ax.plot(pos4[:, 0], pos4[:, 1], pos4[:, 2], color='orange', linewidth=line_width, alpha=0.7)
ax.scatter(pos4[::sample_interval, 0], pos4[::sample_interval, 1], pos4[::sample_interval, 2], 
           c='orange', marker='*', s=marker_size, label='无人机4', edgecolors='black')

# 设置坐标轴标签
ax.set_xlabel('X 位置 (m)', fontsize=12)
ax.set_ylabel('Y 位置 (m)', fontsize=12)
ax.set_zlabel('Z 位置 (m)', fontsize=12)

# 添加标题
ax.set_title('四无人机三维轨迹图', fontsize=16)

# 添加图例
ax.legend(loc='best', fontsize=12)

# 设置等比例坐标轴
all_pos = np.vstack([pos1, pos2, pos3, pos4])
max_range = np.array([all_pos[:, 0].max()-all_pos[:, 0].min(), 
                      all_pos[:, 1].max()-all_pos[:, 1].min(), 
                      all_pos[:, 2].max()-all_pos[:, 2].min()]).max() / 2.0

mid_x = (all_pos[:, 0].max()+all_pos[:, 0].min()) * 0.5
mid_y = (all_pos[:, 1].max()+all_pos[:, 1].min()) * 0.5
mid_z = (all_pos[:, 2].max()+all_pos[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# 添加网格
ax.grid(True, alpha=0.3)

# 调整视角
ax.view_init(elev=20, azim=45)

# 保存图像
plt.savefig('drone_trajectories_3d.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()