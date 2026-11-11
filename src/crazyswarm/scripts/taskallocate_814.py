#!/usr/bin/env python3
import numpy as np
import math
from pycrazyswarm import Crazyswarm
from typing import List, Tuple, Dict
from dataclasses import dataclass

# 飞行参数
takeoff_height = 0.5
land_height = 0.02
Kp = 0.2
hover_time = 2.0

def takeoff_all_drones(allcfs, timeHelper, target_height, kp):
    """控制所有无人机垂直起飞到目标高度"""
    print("开始起飞...")
    # 记录每架无人机的初始XY位置
    for cf in allcfs.crazyflies:
        cf.flag_takeoff = False
        initial_pos = cf.position()
        cf.takeoff_x = initial_pos[0]
        cf.takeoff_y = initial_pos[1]
    
    # 起飞控制循环
    while True:
        timeHelper.sleepForRate(10)
        # 控制每架无人机
        for cf in allcfs.crazyflies:
            current_pos = cf.position()
            
            if abs(current_pos[2]-target_height)<=0.05:
                cf.flag_takeoff = True
                cf.cmdVelocityWorld(vel=np.array([0.0, 0.0, 0.0]), yawRate=0)
            else:
                vz = kp * (target_height - current_pos[2])
                vx = kp * (cf.takeoff_x - current_pos[0])
                vy = kp * (cf.takeoff_y - current_pos[1])
                
                cf.cmdVelocityWorld(vel=np.array([vx, vy, vz]), yawRate=0)
        
        # 检查是否所有无人机完成起飞
        completed_count = sum(1 for cf in allcfs.crazyflies if cf.flag_takeoff)
        if completed_count == len(allcfs.crazyflies):
            break
    
    print(f"起飞完成！所有无人机到达 {target_height}m 高度")

def execute_task_assignment(allcfs, timeHelper, assignment, tasks, kp=0.5):
    """执行任务分配：有任务的无人机飞往目标，无任务的悬停"""
    print("开始执行任务分配...")
    
    # 初始化标志
    for i, cf in enumerate(allcfs.crazyflies):
        cf.task_completed = False
        cf.has_task = i in assignment and assignment[i] != -1
        
        if cf.has_task:
            task_id = assignment[i]
            cf.target_pos = tasks[task_id].position
            print(f"无人机{i} -> 任务{task_id} 目标位置: {cf.target_pos}")
        else:
            # 无任务的无人机保持当前位置
            current_pos = cf.position()
            cf.target_pos = current_pos
            print(f"无人机{i} -> 保持悬停位置: {cf.target_pos}")
    
    # 执行任务循环
    while True:
        timeHelper.sleepForRate(10)
        
        for i, cf in enumerate(allcfs.crazyflies):
            current_pos = cf.position()
            target_pos = cf.target_pos
            
            # 计算到目标位置的距离
            distance = math.sqrt(
                (current_pos[0] - target_pos[0])**2 + 
                (current_pos[1] - target_pos[1])**2 + 
                (current_pos[2] - target_pos[2])**2
            )
            
            # 判断是否到达目标
            if distance <= 0.1:  # 距离目标小于10cm
                cf.task_completed = True
                cf.cmdVelocityWorld(vel=np.array([0.0, 0.0, 0.0]), yawRate=0)
            else:
                # 使用比例控制器飞往目标位置
                vx = kp * (target_pos[0] - current_pos[0])
                vy = kp * (target_pos[1] - current_pos[1])
                vz = kp * (target_pos[2] - current_pos[2])
                
                cf.cmdVelocityWorld(vel=np.array([vx, vy, vz]), yawRate=0)
        
        # 检查是否所有无人机完成任务
        completed_count = sum(1 for cf in allcfs.crazyflies if cf.task_completed)
        if completed_count == len(allcfs.crazyflies):
            break
    
    print("所有任务执行完成！")

def land_all_drones(allcfs, timeHelper, target_height, kp):
    """控制所有无人机垂直降落到目标高度"""
    print("开始降落...")
    
    for cf in allcfs.crazyflies:
        cf.flag_land = False
    
    # 降落控制循环
    while True:
        timeHelper.sleepForRate(10)
        
        # 控制每架无人机
        for cf in allcfs.crazyflies:
            current_pos = cf.position()
            
            if current_pos[2] < 0.1:  # 接近地面
                cf.flag_land = True
                cf.cmdVelocityWorld(vel=np.array([0.0, 0.0, 0.0]), yawRate=0)
            else:
                vz = kp * (target_height - current_pos[2])  
                cf.cmdVelocityWorld(vel=np.array([0.0, 0.0, vz]), yawRate=0)
        
        completed_count = sum(1 for cf in allcfs.crazyflies if cf.flag_land)
        if completed_count == len(allcfs.crazyflies):
            break
    
    print("降落完成！")

def stop_all_drones(allcfs):
    """停止所有无人机电机"""
    print("关闭所有电机...")
    for cf in allcfs.crazyflies:
        cf.cmdStop()
    print("任务完成！")

def get_current_drone_positions(allcfs):
    """获取当前所有无人机的位置"""
    positions = []
    for i, cf in enumerate(allcfs.crazyflies):
        pos = cf.position()
        positions.append((i, pos))
    return positions


@dataclass
class Task:
    """任务定义"""
    id: int
    position: Tuple[float, float, float]  # (x, y, z)
    priority: float  # 优先级权重 (0-1, 1为最高优先级)

@dataclass
class Drone:
    """无人机定义"""
    id: int
    position: Tuple[float, float, float]  # (x, y, z)
    max_speed: float  # 最大飞行速度 m/s

class COATaskAllocator:
    """基于狼群优化算法的无人机任务分配器"""

    def __init__(self, drones: List[Drone], tasks: List[Task],
                 max_iter: int = 50, num_groups: int = 5, group_size: int = 6):
        self.drones = drones
        self.tasks = tasks
        self.num_drones = len(drones)
        self.num_tasks = len(tasks)
        self.num_assignments = min(self.num_drones, self.num_tasks)

        # COA参数
        self.max_iter = max_iter
        self.num_groups = num_groups
        self.group_size = group_size
        self.population_size = num_groups * group_size

        # 预计算矩阵
        self.distance_matrix = self._calculate_distance_matrix()
        self.cost_matrix = self._calculate_cost_matrix()
        self.best_fitness_history = []

    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算无人机到任务的距离矩阵"""
        distances = np.zeros((self.num_drones, self.num_tasks))

        for i, drone in enumerate(self.drones):
            for j, task in enumerate(self.tasks):
                dx = drone.position[0] - task.position[0]
                dy = drone.position[1] - task.position[1]
                dz = drone.position[2] - task.position[2]
                distances[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)

        return distances

    def _calculate_cost_matrix(self) -> np.ndarray:
        """计算成本矩阵（考虑距离、优先级）"""
        costs = np.zeros((self.num_drones, self.num_tasks))

        for i, drone in enumerate(self.drones):
            for j, task in enumerate(self.tasks):
                # 距离成本（归一化）
                max_distance = np.max(self.distance_matrix)
                distance_cost = self.distance_matrix[i, j] / max_distance if max_distance > 0 else 0

                # 飞行时间成本
                flight_time = self.distance_matrix[i, j] / drone.max_speed
                flight_time_cost = flight_time / 60.0

                # 优先级奖励（高优先级任务成本更低）
                priority_reward = (1.0 - task.priority) * 0.5

                # 总成本
                costs[i, j] = distance_cost + flight_time_cost + priority_reward

        return costs

    def _objective_function(self, solution: np.ndarray) -> float:
        """目标函数：最小化总分配成本"""
        assignment = self._decode_solution(solution)
        total_cost = 0.0

        for drone_id, task_id in assignment.items():
            if task_id != -1:
                total_cost += self.cost_matrix[drone_id, task_id]

        assigned_tasks = set(assignment.values())
        unassigned_tasks = set(range(self.num_tasks)) - assigned_tasks - {-1}
        for task_id in unassigned_tasks:
            total_cost += 100.0 * self.tasks[task_id].priority

        unassigned_drones = set(range(self.num_drones)) - set(assignment.keys())
        total_cost += len(unassigned_drones) * 10.0

        return total_cost

    def _decode_solution(self, solution: np.ndarray) -> Dict[int, int]:
        """将解向量解码为一对一任务分配方案"""
        solution_matrix = solution.reshape(self.num_drones, self.num_tasks)
        assignment = {}
        assigned_tasks = set()

        task_priorities = [(i, self.tasks[i].priority) for i in range(self.num_tasks)]
        task_priorities.sort(key=lambda x: x[1], reverse=True)

        for task_id, priority in task_priorities:
            if task_id in assigned_tasks:
                continue

            best_drone = -1
            best_score = float('inf')

            for drone_id in range(self.num_drones):
                if drone_id in assignment:
                    continue

                # 计算综合得分（解向量概率 + 成本）
                prob_score = 1.0 - solution_matrix[drone_id, task_id]
                cost_score = self.cost_matrix[drone_id, task_id]
                total_score = prob_score + cost_score

                if total_score < best_score:
                    best_score = total_score
                    best_drone = drone_id

            if best_drone != -1:
                assignment[best_drone] = task_id
                assigned_tasks.add(task_id)

        # 处理剩余无人机和任务
        unassigned_drones = [i for i in range(self.num_drones) if i not in assignment]
        unassigned_tasks = [i for i in range(self.num_tasks) if i not in assigned_tasks]

        for i, drone_id in enumerate(unassigned_drones):
            if i < len(unassigned_tasks):
                task_id = unassigned_tasks[i]
                assignment[drone_id] = task_id
            else:
                assignment[drone_id] = -1

        return assignment

    def _normalize_solution(self, solution: np.ndarray) -> np.ndarray:
        """归一化解向量"""
        solution_matrix = solution.reshape(self.num_drones, self.num_tasks)

        for drone_id in range(self.num_drones):
            drone_sum = np.sum(solution_matrix[drone_id])
            if drone_sum > 0:
                solution_matrix[drone_id] /= drone_sum
            else:
                solution_matrix[drone_id] = 1.0 / self.num_tasks

        return solution_matrix.flatten()

    def _find_median_index(self, fitness_values: np.ndarray) -> int:
        """找到中位数索引"""
        sorted_indices = np.argsort(fitness_values)
        median_idx = len(sorted_indices) // 2
        return sorted_indices[median_idx]

    def optimize(self) -> Tuple[Dict[int, int], float]:
        """执行COA优化"""
        # 初始化种群
        population = np.random.rand(self.population_size, self.num_drones * self.num_tasks)

        # 归一化初始种群
        for i in range(self.population_size):
            population[i] = self._normalize_solution(population[i])

        # 计算初始适应度
        fitness = np.array([self._objective_function(ind) for ind in population])

        # 随机分组
        groups = np.random.permutation(self.population_size).reshape(self.num_groups, self.group_size)

        # 找到全局最优
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        self.best_fitness_history = [best_fitness]

        # 主优化循环
        for iteration in range(self.max_iter):
            # 对每个群组进行优化
            for g in range(self.num_groups):
                group_indices = groups[g]
                group_population = population[group_indices]
                group_fitness = fitness[group_indices]

                best_group_idx = np.argmin(group_fitness)
                median_group_idx = self._find_median_index(group_fitness)

                group_best = group_population[best_group_idx]
                group_median = group_population[median_group_idx]

                # 更新群组内每个个体
                for c in range(self.group_size):
                    random_indices = np.random.choice(self.group_size, 2, replace=False)
                    coy1 = group_population[random_indices[0]]
                    coy2 = group_population[random_indices[1]]

                    r1, r2 = np.random.rand(), np.random.rand()
                    new_solution = (group_population[c] +
                                    r1 * (group_best - coy1) +
                                    r2 * (group_median - coy2))

                    new_solution = np.clip(new_solution, 0, 1)
                    new_solution = self._normalize_solution(new_solution)
                    new_fitness = self._objective_function(new_solution)

                    if new_fitness < group_fitness[c]:
                        group_population[c] = new_solution
                        group_fitness[c] = new_fitness

                # 社会条件：创建新个体
                parent_indices = np.random.choice(self.group_size, 2, replace=False)
                parent1 = group_population[parent_indices[0]]
                parent2 = group_population[parent_indices[1]]

                new_individual = np.zeros_like(parent1)
                crossover_prob = 1.0 / len(parent1)

                for x in range(len(parent1)):
                    beta = np.random.rand()
                    if beta < crossover_prob:
                        new_individual[x] = parent1[x]
                    elif beta < 0.5 + crossover_prob:
                        new_individual[x] = parent2[x]
                    else:
                        new_individual[x] = np.random.rand()

                new_individual = self._normalize_solution(new_individual)
                new_individual_fitness = self._objective_function(new_individual)

                worst_idx = np.argmax(group_fitness)
                if new_individual_fitness < group_fitness[worst_idx]:
                    group_population[worst_idx] = new_individual
                    group_fitness[worst_idx] = new_individual_fitness

                population[group_indices] = group_population
                fitness[group_indices] = group_fitness

            # 群组间交换
            exchange_prob = 0.01 * self.group_size ** 2 / 2
            if np.random.rand() < exchange_prob:
                group_indices = np.random.choice(self.num_groups, 2, replace=False)
                coy_indices = np.random.choice(self.group_size, 2, replace=False)

                idx1 = groups[group_indices[0], coy_indices[0]]
                idx2 = groups[group_indices[1], coy_indices[1]]
                groups[group_indices[0], coy_indices[0]] = idx2
                groups[group_indices[1], coy_indices[1]] = idx1

            # 更新全局最优
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            self.best_fitness_history.append(best_fitness)

            # 早停条件
            if len(self.best_fitness_history) > 10:
                recent_improvement = (self.best_fitness_history[-10] -
                                      self.best_fitness_history[-1])
                if recent_improvement < 1e-6:
                    break

        # 解码最终结果
        final_assignment = self._decode_solution(best_solution)
        return final_assignment, best_fitness

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    
    try:
        # 1. 起飞
        takeoff_all_drones(allcfs, timeHelper, takeoff_height, Kp)
        
        # 2. 获取当前无人机位置，创建Drone对象
        current_positions = get_current_drone_positions(allcfs)
        drones = []
        for i, pos in current_positions:
            drones.append(Drone(id=i, position=pos, max_speed=1.0))
        
        # 3. 创建任务列表（可以根据需要修改任务位置）
        tasks = [
            Task(id=0, position=(0.0, -0.9, 0.6), priority=0.9),
            Task(id=1, position=(-1.4, 0, 0.6), priority=0.8),
        ]
             
        # 4. 执行任务分配优化
        print("开始任务分配优化...")
        allocator = COATaskAllocator(drones, tasks, max_iter=30, num_groups=4, group_size=5)
        assignment, best_cost = allocator.optimize()
        
        # 5. 输出分配结果
        print("\n=== 任务分配结果 ===")
        for drone_id, task_id in assignment.items():
            if task_id != -1:
                task_pos = tasks[task_id].position
                print(f"无人机{drone_id} -> 任务{task_id} (目标: {task_pos})")
            else:
                print(f"无人机{drone_id} -> 保持悬停")
        print(f"分配总成本: {best_cost:.3f}\n")
        
        # 6. 执行任务分配
        execute_task_assignment(allcfs, timeHelper, assignment, tasks, Kp)
        
        # 7. 任务完成后悬停一段时间
        print("任务完成，悬停等待...")
        # timeHelper.sleep(hover_time)
        
        # # 8. 降落
        # land_all_drones(allcfs, timeHelper, land_height, Kp)
        
        # # 9. 停机
        # stop_all_drones(allcfs)

        # 控制集群全部无人机降落
        print("降落...")
        flag_all_uav_land = False
        for cf in allcfs.crazyflies:
            cf.flag_uav_land = False
        while flag_all_uav_land == False:
            timeHelper.sleepForRate(10)
            for cf in allcfs.crazyflies:
                cf.current_uav_pos = cf.position()
                if cf.current_uav_pos[2] < 0.1:
                    cf.flag_uav_land = True
                    cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, 0.0]), yawRate = 0)
                else:
                    cf.cmdVelocityWorld(vel = np.array([0.0, 0.0, 0.5 * (land_height - cf.current_uav_pos[2])]), yawRate = 0)
            cnt_uav_land = 0
            for cf in allcfs.crazyflies:
                if cf.flag_uav_land == True:
                    cnt_uav_land = cnt_uav_land + 1
            if cnt_uav_land == len(allcfs.crazyflies):
                flag_all_uav_land = True
    
        # 电机停转
        print("急停...")
        for cf in allcfs.crazyflies:
            cf.cmdStop()
        
    except KeyboardInterrupt:
        print("用户中断，紧急停机...")
        stop_all_drones(allcfs)
    except Exception as e:
        print(f"发生错误: {e}")
        print("紧急停机...")
        stop_all_drones(allcfs)