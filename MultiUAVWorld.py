from entity import UAV, User
import numpy as np
import os
from numpy import linalg as LA
from rural_world import Rural_world
import radio_map_A2G as A2G
import radio_map_G2A as G2A

class MultiUAVWorld(object):
    """多无人机协同巡检环境"""
    
    def __init__(self, 
                 length=10, 
                 width=10, 
                 uav_num=3,  # 🔥 多无人机数量
                 user_num=10,
                 dist_max=0.10,
                 delta_t=0.5,
                 t=200, 
                 uav_h=1,
                 data_size=250,
                 ini_loc=[10.93, 4.4],
                 end_loc=[15.5, 18.8],
                 users_name='Users.txt',
                 BS_loc=[],
                 traverse_sequence=[],
                 safe_distance=2.0,  # 🔥 安全距离
                 comm_range=5.0,     # 🔥 通信范围
                 cooperative_mode='sequential'):  # 'sequential' or 'parallel'
        
        # 基础参数
        self.length = length
        self.width = width
        self.uav_num = uav_num
        self.users_path = users_name
        self.user_num = user_num
        self.Users = []
        self.UAVs = []
        self.T = t
        self.t = 0
        
        # 边界
        self.max_x = length
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h
        
        # 飞行参数
        self.dist_max = dist_max
        self.delta_T = delta_t
        
        # 任务参数
        self.initial_loc = ini_loc
        self.end_loc = end_loc
        self.distance = 0.5  # 到达目标阈值
        self.data_size_ini = data_size
        self.BandWidth = 1
        
        # 🔥 多无人机新增参数
        self.safe_distance = safe_distance  # 最小安全距离
        self.comm_range = comm_range  # 通信/感知范围
        self.cooperative_mode = cooperative_mode
        
        # 通信阈值
        self.SIR_THRESHOLD_COMM = 3
        self.SIR_THRESHOLD_COVER = 2
        
        # 奖励权重
        self.NON_COMM_PENALTY = -5
        self.NON_COVER_PENALTY = -10
        self.Engy_w = 0.05
        self.COLLISION_PENALTY = -500  # 🔥 碰撞惩罚
        
        # 巡检序列，暂时还没确定
        # self.Traverse = traverse_sequence
        
        # 加载基站位置
        self.BS_loc = BS_loc
        self.set_users()
        
        # 环境地图
        self.urban_world = Rural_world(self.BS_loc)
        self.HeightMapMatrix = self.urban_world.Buliding_construct()
        
        # 🔥 多无人机任务分配
        self.uav_targets = [None for _ in range(self.uav_num)]   # 每个无人机的当前目标
        self.uav_traverse = [[] for _ in range(self.uav_num)]  # 每个无人机的巡检点序列
        self.uav_reach_final = [False for _ in range(self.uav_num)] # 每个无人机是否到达终点
        
        # self.uav_data_sizes = [0.0] * uav_num  # 每个无人机的数据传输进度
        # self.uav_transmit_flags = [False] * uav_num  # 传输完成标志
        self.completed_targets = set()  # 已完成的目标点
        
        # 统计信息
        self.fa = 0.0
        self.r = 0.0
        self.terminal = False
        self.total_engy = 0.0
        self.out_time = 0.0
        self.collision_count = 0
        
        # print(f"[MultiUAVWorld] 初始化完成")
        # print(f"  - 无人机数量: {uav_num}")
        # print(f"  - 检查点数量: {len(traverse_sequence)}")
        # print(f"  - 安全距离: {safe_distance}")
        # print(f"  - 通信范围: {comm_range}")
        # print(f"  - 协作模式: {cooperative_mode}")

    def set_users(self):
        """加载用户位置（检查点）"""
        self.Users =[]
        if os.path.exists(self.users_path): # 读写文件 # 读入用户位置
            f = open(self.users_path, 'r')
            if f:
                user_loc = f.readline()
                user_loc = user_loc.split(' ')
                self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                # self.GT_loc[len(self.Users) - 1] = np.array(
                #     [float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                while user_loc:
                    user_loc = f.readline()
                    if user_loc:
                        user_loc = user_loc.split(' ')
                        self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                        # self.GT_loc[len(self.Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                f.close()
        else:
            assert False, "Users file not found: " + self.users_path
        
    """初始化所有无人机位置"""
    def set_uavs_loc(self):
        self.UAVs = []
        #所有无人机初始位置相同
        for i in range(self.uav_num):
            x = self.initial_loc[0]
            y = self.initial_loc[1]
            
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
            
    """为每个无人机分配巡检序列，这里也是我们作文章的一个点"""
    def assign_traverse(self):
        # 顺序模式：按traverse_sequence依次分配
        for i in range(self.uav_num):
            #每个无人机均匀地分配user_num/uav_num个目标
            for j in range (i*(self.user_num//self.uav_num),(i+1)*(self.user_num//self.uav_num)):
                self.uav_traverse[i].append(j)
            # self.uav_data_sizes[i] = self.data_size_ini
            # self.uav_transmit_flags[i] = False
            
    """为每个无人机分配第一个飞行目标点"""
    def assign_targets(self):
        for i in range (self.uav_num):
            self.uav_targets[i] = self.uav_traverse[i][0]
            

    def reset(self):
        """重置环境"""
        self.set_uavs_loc() #重置无人机位置
        self.assign_traverse() #重置巡检序列
        self.assign_targets() #分配初始目标
        self.completed_targets = set() #重置已完成目标集
        self.uav_reach_final = [False for _ in range(self.uav_num)] #重置每个无人机到达终点标志
        
        self.t = 0
        self.fa = 0
        self.out_time = 0
        self.total_engy = 0.0
        self.terminal = False
        self.collision_count = 0
        
        # 返回初始观测列表
        obs_list = self.get_observations()
        return obs_list

    def get_observations(self):
        """
        获取所有无人机的局部观测
        
        返回: List[np.array] - 每个无人机的局部观测
        """
        obs_list = []
        
        for i, uav in enumerate(self.UAVs):
            obs = self._get_local_observation(i)
            obs_list.append(obs)
        
        return obs_list

    def _get_local_observation(self, uav_id):
        """
        构造单个无人机的局部观测
        
        观测空间设计：
        1. 自身状态: [x, y] (2维)
        2. 目标位置: [pos_x, pos_y] (2维)
        
        
        总维度: 2+2
        """
        uav = self.UAVs[uav_id]
        obs = []
        
        # 1. 自身位置 (2维)
        obs.extend([uav.x, uav.y])
        
        # 2. 目标位置 (2维)
        if self.uav_targets[uav_id] is not None:
            target = self.Users[self.uav_targets[uav_id]]
            target_pos = np.array([target.x, target.y])
            # uav_pos = np.array([uav.x, uav.y])
            # relative_target = target_pos - uav_pos
            # distance_to_target = LA.norm(relative_target)
            obs.extend([target_pos[0],target_pos[1]])
        else:
            # 如果没有目标，前往终点
            # relative_end = np.array(self.end_loc) - np.array([uav.x, uav.y])
            # distance_to_end = LA.norm(relative_end)
            obs.extend([self.end_loc[0],self.end_loc[1]])
        
        # 3. 数据传输进度 (2维)
        # obs.append(self.uav_data_sizes[uav_id])
        # obs.append(1.0 if self.uav_transmit_flags[uav_id] else 0.0)
        
        # 4. 最近邻居信息 (最多K=3个邻居，每个4维)
        # K_neighbors = 3
        # neighbors_info = self._get_neighbors_info(uav_id, K=K_neighbors)
        # obs.extend(neighbors_info)
        
        # 5. 时间信息 (1维)
        # remaining_time = (self.T - self.t) / self.T  # 归一化
        # obs.append(remaining_time)
        
        return np.array(obs, dtype=np.float32)

    # def _get_neighbors_info(self, uav_id, K=3):
    #     """
    #     获取最近的K个邻居信息,这个函数暂时没用上
        
    #     返回: list - 长度为K×4的列表，填充或截断
    #     """
    #     uav = self.UAVs[uav_id]
    #     uav_pos = np.array([uav.x, uav.y])
        
    #     neighbors = []
    #     for i, other_uav in enumerate(self.UAVs):
    #         if i != uav_id:
    #             #获取邻居的位置和距离
    #             other_pos = np.array([other_uav.x, other_uav.y])
    #             distance = LA.norm(other_pos - uav_pos)
                
    #             if distance < self.comm_range:  # 通信范围内
    #                 relative_pos = other_pos - uav_pos
    #                 # 相对速度（简化为0）
    #                 relative_vel = [0.0, 0.0]
                    
    #                 neighbors.append({
    #                     'distance': distance,
    #                     'info': list(relative_pos) + relative_vel
    #                 })
        
    #     # 按距离排序
    #     neighbors.sort(key=lambda x: x['distance'])
        
    #     # 提取特征
    #     neighbor_features = []
    #     for n in neighbors[:K]:
    #         neighbor_features.extend(n['info'])
        
    #     # 填充到固定长度
    #     while len(neighbor_features) < K * 4:
    #         neighbor_features.append(0.0)
        
    #     return neighbor_features[:K * 4]

    def step(self, actions):
        """
        执行一步
        
        Args:
            actions: List[np.array] - 每个无人机的动作 [[phi, dist], [phi, dist], ...]
        
        Returns:
            obs_list: 观测列表
            rewards: 奖励列表
            dones: 完成标志列表
            info: 额外信息
        """
        self.t += 1
        
        # 存储上一时刻位置
        uav_locations_pre = np.array([[uav.x, uav.y] for uav in self.UAVs])
        
        # 执行动作
        self._execute_actions(actions, uav_locations_pre)
        
        # 当前位置
        uav_locations = np.array([[uav.x, uav.y] for uav in self.UAVs])
        
        # 更新数据传输
        # self._update_data_transmission(uav_locations)
        
        # 检查目标完成情况 这个函数有点多此一举了
        # self._check_target_completion(uav_locations)
        
        # 计算奖励
        rewards = self._compute_rewards(
            uav_locations, uav_locations_pre)
        
        # 检查是否完成
        dones = self._check_done()
        
        # 获取新观测
        obs_list = self.get_observations()
        
        # 额外信息
        info = {
            'success': self.terminal,
            # 'collision': collision_occurred,
            'completed_targets': len(self.completed_targets),
            'total_targets': self.user_num,
            'collision_count': self.collision_count
        }
        
        return obs_list, rewards, dones, info, self.uav_reach_final

    def _execute_actions(self, actions, uav_locations_pre):
        """执行所有无人机的动作"""
        # collision_occurred = False
        fa_total = 0.0
        
        # 执行动作
        for i, uav in enumerate(self.UAVs):
            if len(actions[i]) >= 2:
                uav.move_inside(actions[i][0], actions[i][1], self.dist_max)
                
                # 检查边界
                penalty, bound = self.boundary_margin(uav)
                fa_total += penalty
                
                if not bound:  # 出界，取消动作
                    self.fa += 1
                    uav.x = uav_locations_pre[i][0]
                    uav.y = uav_locations_pre[i][1]
        
        # 🔥 检查碰撞
        # collision_occurred = self._check_collisions()
        # if collision_occurred:
        #     self.collision_count += 1
        #     # 碰撞后回退到上一位置
        #     for i, uav in enumerate(self.UAVs):
        #         uav.x = uav_locations_pre[i][0]
        #         uav.y = uav_locations_pre[i][1]
        
        # return collision_occurred

    def _check_collisions(self):
        """检查无人机之间是否碰撞"""
        for i in range(self.uav_num):
            for j in range(i + 1, self.uav_num):
                pos_i = np.array([self.UAVs[i].x, self.UAVs[i].y])
                pos_j = np.array([self.UAVs[j].x, self.UAVs[j].y])
                distance = LA.norm(pos_i - pos_j)
                
                if distance < self.safe_distance:
                    return True
        return False

    def _update_data_transmission(self, uav_locations):
        """更新数据传输进度"""
        for i, uav_loc in enumerate(uav_locations):
            if not self.uav_transmit_flags[i] and self.uav_targets[i] is not None:
                # 计算A2G信噪比
                MaxSINR_A2G = A2G.getPointDateRate(uav_loc)
                data_rate = self.BandWidth * np.log2(1 + 10**(MaxSINR_A2G/10.0))
                
                # 更新数据量
                self.uav_data_sizes[i] -= data_rate
                
                if self.uav_data_sizes[i] <= 0:
                    self.uav_data_sizes[i] = 0
                    self.uav_transmit_flags[i] = True

    def _check_target_completion(self, uav_locations):
        """检查目标完成情况"""
        for i, uav_loc in enumerate(uav_locations):
            if self.uav_targets[i] is not None:
                target = self.Users[self.uav_targets[i]]
                target_pos = np.array([target.x, target.y])
                distance = LA.norm(uav_loc - target_pos)
                
                if distance <= self.distance:  # 到达巡检点
                    # if self.uav_transmit_flags[i]:  # 且传输完成
                        # 标记目标完成
                    self.completed_targets.add(self.uav_targets[i])
                    # 分配下一个目标
                    self._assign_next_target(i)
            else:
                # 前往终点
                end_pos = np.array(self.end_loc)
                distance = LA.norm(uav_loc - end_pos)
                if distance <= self.distance:
                    self.uav_reach_final[i] = True

    def _assign_next_target(self, uav_id):
        
        """为无人机分配下一个目标"""
        # 找到未完成的目标
        remaining_targets = [
            self.uav_traverse[uav_id][idx]
            for idx in range(len(self.uav_traverse[uav_id]))
            if self.uav_traverse[uav_id][idx] not in self.completed_targets
        ]
        
        if remaining_targets: 
            # 分配最近的未完成目标
            uav_pos = np.array([self.UAVs[uav_id].x, self.UAVs[uav_id].y])
            min_dist = float('inf')
            next_target = None
            
            for target_idx in remaining_targets:
                target_pos = np.array([self.Users[target_idx].x, self.Users[target_idx].y])
                dist = LA.norm(uav_pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    next_target = target_idx
            
            self.uav_targets[uav_id] = next_target
            # self.uav_data_sizes[uav_id] = self.data_size_ini
            # self.uav_transmit_flags[uav_id] = False
        else:
            # 所有目标完成，前往终点
            self.uav_targets[uav_id] = None

    def _compute_rewards(self, uav_locations, uav_locations_pre):
        """计算每个无人机的奖励"""
        rewards = []
        
        for i in range(self.uav_num):
            reward = 0.0
            
            # 1. 前进奖励
            if self.uav_targets[i] is not None:
                target_pos = np.array([
                    self.Users[self.uav_targets[i]].x,
                    self.Users[self.uav_targets[i]].y
                ])
                
                dist_cur = LA.norm(uav_locations[i] - target_pos)
                dist_pre = LA.norm(uav_locations_pre[i] - target_pos)
                progress = dist_pre - dist_cur
                
                reward += max(0, progress * 100)
                
                # 停留惩罚
                if abs(progress) < 0.05:
                    reward -= 10
            
            # 2. 碰撞惩罚
            # if collision:
            #     reward += self.COLLISION_PENALTY
            
            # 3. 接近其他无人机的惩罚（软避碰）
            for j in range(self.uav_num):
                if i != j:
                    dist_to_other = LA.norm(uav_locations[i] - uav_locations[j])
                    if dist_to_other < self.safe_distance * 2:
                        penalty = (self.safe_distance * 2 - dist_to_other) / self.safe_distance
                        reward -= 20 * penalty
            
            # 4. 到达目标奖励
            if self.uav_targets[i] is not None:
                # 到达巡检点奖励
                target_pos = np.array([
                    self.Users[self.uav_targets[i]].x,
                    self.Users[self.uav_targets[i]].y
                ])
                if LA.norm(uav_locations[i] - target_pos) <= self.distance:
                    print("UAV {} reached target {}".format(i, self.uav_targets[i]))
                    reward += 500 #到达目标点的奖励
                    # 标记目标完成
                    self.completed_targets.add(self.uav_targets[i])
                    # 分配下一个目标
                    self._assign_next_target(i)    
            else:
                # 前往终点奖励
                end_pos = np.array(self.end_loc)
                distance = LA.norm(uav_locations[i] - end_pos)
                if distance <= self.distance and not self.uav_reach_final[i]:
                    print("UAV {} heading to end".format(i))
                    self.uav_reach_final[i] = True
                    reward += 1000
            
            # 5. 时间惩罚
            # reward -= 0.1
            rewards.append(reward)
        
        # 6. 团队奖励（所有目标完成）
        if sum(self.uav_reach_final) == self.uav_num:
            print("All UAVs reached the end!")
            team_bonus = 2000 + (self.T - self.t) * 10
            rewards = [r + team_bonus / self.uav_num for r in rewards]
            self.terminal = True
        
        return rewards

    def _check_done(self):
        
        """检查每个无人机是否完成"""
        dones = False
        
        # 任务完成或超时
        if self.terminal or self.t >= self.T:
            dones = True
        else :
            dones = False
        
        return dones    
        
    def boundary_margin(self, uav):
        """检查无人机是否出界"""
        margin = 1.0
        penalty_factor = 100 / self.uav_num
        
        center_x = (self.max_x + self.min_x) / 2
        center_y = (self.max_y + self.min_y) / 2
        
        x_exceed = max(abs(uav.x - center_x) - margin * (self.max_x - self.min_x) / 2, 0.0)
        y_exceed = max(abs(uav.y - center_y) - margin * (self.max_y - self.min_y) / 2, 0.0)
        
        if x_exceed == 0.0 and y_exceed == 0.0:
            return 0.0, True
        else:
            penalty = penalty_factor * (x_exceed**2 + y_exceed**2)
            return penalty, False

    @property
    def local_obs_dim(self):
        """单个无人机的观测维度"""
        # 2(自身) + 2(目标)
        return 2+2
    
    @property
    def action_dim(self):
        """单个无人机的动作维度"""
        return 2  # [phi, dist]