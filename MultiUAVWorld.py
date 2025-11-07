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
        
        # 动态分配相关结构
        # target_owner: map target_idx -> uav_id (reservation)
        self.target_owner = {}
        # assigned_time: map target_idx -> time step when assigned (for timeout)
        self.assigned_time = {}
        # assignment timeout (steps) - 可根据任务时长调整
        self.assignment_timeout = max((self.T/self.user_num)*2, int(self.T * 0.1))
        # sentinel value: UAV is waiting for an available (unassigned) target
        self.WAIT_TARGET = -1
        
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
            
    """为每个无人机分配第一个飞行目标点,需要提前指定巡检序列self.uav_traverse"""
    def assign_targets(self):
        for i in range (self.uav_num):
            self.uav_targets[i] = self.uav_traverse[i][0]
    
    def assign_initial_targets(self):
        """为每个无人机分配初始目标点,不需要提前指定巡检序列self.uav_traverse"""
        # 可选：打乱顺序以避免固定优先级
        uav_order = list(range(self.uav_num))
        # import random
        # random.shuffle(uav_order)
        for uav_id in uav_order:
            self._assign_next_target(uav_id)
            

    def reset(self):
        """重置环境"""
        self.set_uavs_loc() #重置无人机位置
        # 重置分配/完成状态（先清空再分配）
        self.completed_targets = set() #重置已完成目标集
        self.target_owner = {}
        self.assigned_time = {}
        self.uav_targets = [None for _ in range(self.uav_num)]
        # self.assign_traverse() #重置巡检序列（如需静态分配可启用）
        self.assign_initial_targets() #分配初始目标,不需要提前指定巡检序列
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
        tgt = self.uav_targets[uav_id]
        if tgt is not None and tgt != self.WAIT_TARGET:
            # 正常分配到某个巡检点（索引）
            target = self.Users[tgt]
            target_pos = np.array([target.x, target.y])
            obs.extend([target_pos[0], target_pos[1]])
        elif tgt == self.WAIT_TARGET:
            # 等待分配：将目标位置设置为自身位置，表示当前无可分配巡检点
            obs.extend([uav.x, uav.y])
        else:
            # 所有目标已完成，前往终点
            obs.extend([self.end_loc[0], self.end_loc[1]])
        
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
        # 回收超时的分配，避免死占用
        self._reclaim_stale_assignments()

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

    # def _check_target_completion(self, uav_locations):
    #     """检查目标完成情况"""
    #     for i, uav_loc in enumerate(uav_locations):
    #         if self.uav_targets[i] is not None:
    #             target = self.Users[self.uav_targets[i]]
    #             target_pos = np.array([target.x, target.y])
    #             distance = LA.norm(uav_loc - target_pos)
                
    #             if distance <= self.distance:  # 到达巡检点
    #                 # if self.uav_transmit_flags[i]:  # 且传输完成
    #                     # 标记目标完成
    #                 self.completed_targets.add(self.uav_targets[i])
    #                 # 分配下一个目标
    #                 self._assign_next_target(i)
    #         else:
    #             # 前往终点
    #             end_pos = np.array(self.end_loc)
    #             distance = LA.norm(uav_loc - end_pos)
    #             if distance <= self.distance:
    #                 self.uav_reach_final[i] = True

    # def _assign_next_target(self, uav_id):
    #     """为无人机分配下一个目标"""
    #     """需要提前指定每个无人机的巡检序列self.uav_traverse"""
    #     # 找到未完成的目标
    #     remaining_targets = [
    #         self.uav_traverse[uav_id][idx]
    #         for idx in range(len(self.uav_traverse[uav_id]))
    #         if self.uav_traverse[uav_id][idx] not in self.completed_targets
    #     ]
        
    #     if remaining_targets: 
    #         # 分配最近的未完成目标
    #         uav_pos = np.array([self.UAVs[uav_id].x, self.UAVs[uav_id].y])
    #         min_dist = float('inf')
    #         next_target = None
            
    #         for target_idx in remaining_targets:
    #             target_pos = np.array([self.Users[target_idx].x, self.Users[target_idx].y])
    #             dist = LA.norm(uav_pos - target_pos)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 next_target = target_idx
            
    #         self.uav_targets[uav_id] = next_target
    #         # self.uav_data_sizes[uav_id] = self.data_size_ini
    #         # self.uav_transmit_flags[uav_id] = False
    #     else:
    #         # 所有目标完成，前往终点
    #         self.uav_targets[uav_id] = None
    
    def _assign_next_target(self, uav_id):
        """为无人机分配下一个目标"""
        """无需提前指定每个无人机的巡检序列，按照就近分配原则分配。
        仅预占（reservation）目标：将 target_owner[target] = uav_id
        完成（visited）由 UAV 真正到达时记录到 completed_targets。
        """
        # 剔除已完成与已被占用的目标
        occupied = set(self.target_owner.keys())
        remaining_targets = [idx for idx in range(self.user_num)
                             if idx not in self.completed_targets and idx not in occupied]

        if remaining_targets:
            # 分配最近的未完成未被占用目标
            uav_pos = np.array([self.UAVs[uav_id].x, self.UAVs[uav_id].y])
            min_dist = float('inf')
            next_target = None

            for target_idx in remaining_targets:
                target_pos = np.array([self.Users[target_idx].x, self.Users[target_idx].y])
                dist = LA.norm(uav_pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    next_target = target_idx

            # 预占该目标，记录分配时间
            if next_target is not None:
                self.target_owner[next_target] = uav_id
                self.assigned_time[next_target] = self.t
                self.uav_targets[uav_id] = next_target
        else:
            # 没有可立即分配的目标
            # 如果还有未完成的目标，但都被占用，则进入等待状态；
            # 只有当所有目标都被完成时，才前往终点（uav_targets=None）
            if len(self.completed_targets) < self.user_num:
                # 仍有未完成的目标，但当前无可分配，等待被分配
                self.uav_targets[uav_id] = self.WAIT_TARGET
            else:
                # 所有目标已完成，前往终点
                self.uav_targets[uav_id] = None

        return self.uav_targets[uav_id]

    def _on_reach_target(self, uav_id, target_idx):
        """处理 UAV 真正到达并完成目标时的逻辑"""
        if target_idx is None:
            return
        # 标记完成
        self.completed_targets.add(target_idx)
        # 释放 reservation（如果存在）
        if target_idx in self.target_owner:
            try:
                del self.target_owner[target_idx]
            except KeyError:
                pass
        if target_idx in self.assigned_time:
            try:
                del self.assigned_time[target_idx]
            except KeyError:
                pass
        # 分配下一个目标给该 UAV
        self._assign_next_target(uav_id)

    def _reclaim_stale_assignments(self):
        """回收长期未完成的分配，避免目标被永久占用"""
        now = self.t
        stale = [t for t, ts in list(self.assigned_time.items()) if now - ts > self.assignment_timeout]
        for t in stale:
            owner = self.target_owner.get(t)
            # 释放 reservation
            if t in self.target_owner:
                try:
                    del self.target_owner[t]
                except KeyError:
                    pass
            if t in self.assigned_time:
                try:
                    del self.assigned_time[t]
                except KeyError:
                    pass
            # 如果原 owner 仍指向该目标，则清空其当前目标，允许重分配
            if owner is not None and 0 <= owner < self.uav_num and self.uav_targets[owner] == t:
                # 原 owner 的当前目标已被回收，将其标记为等待状态，等待重新分配
                self.uav_targets[owner] = self.WAIT_TARGET

        # 尝试为处于等待状态的 UAV 分配刚刚回收出来的目标
        for uid in range(self.uav_num):
            if self.uav_targets[uid] == self.WAIT_TARGET:
                self._assign_next_target(uid)


    def _compute_rewards(self, uav_locations, uav_locations_pre):
        """计算每个无人机的奖励"""
        rewards = []
        for i in range(self.uav_num):
            reward = 0.0

            # 对于已到达终点的 UAV，保持 reward 为 0（或可自定义），但仍需 append
            if not self.uav_reach_final[i]:
                # 1. 前进奖励
                tgt = self.uav_targets[i]
                if tgt is not None and tgt != self.WAIT_TARGET:
                    target_pos = np.array([
                        self.Users[tgt].x,
                        self.Users[tgt].y
                    ])
                elif tgt == self.WAIT_TARGET:
                    # 等待状态：以当前位置信息作为“虚拟目标”，不会产生前进奖励
                    target_pos = np.array(uav_locations[i])
                else:
                    # 所有目标已完成，朝终点飞行
                    target_pos = np.array(self.end_loc)

                dist_cur = LA.norm(uav_locations[i] - target_pos)
                dist_pre = LA.norm(uav_locations_pre[i] - target_pos)
                progress = dist_pre - dist_cur

                reward += max(0, progress * 100)

                # 停留惩罚
                if abs(progress) < 0.05:
                    reward -= 10

                # 2. 接近其他无人机的惩罚（软避碰）
                for j in range(self.uav_num):
                    if i != j:
                        dist_to_other = LA.norm(uav_locations[i] - uav_locations[j])
                        if dist_to_other < self.safe_distance * 2:
                            penalty = (self.safe_distance * 2 - dist_to_other) / self.safe_distance
                            reward -= 20 * penalty

                # 3. 到达目标奖励（实际完成时调用 _on_reach_target 做标记和重新分配）
                if tgt is not None and tgt != self.WAIT_TARGET:
                    target_pos2 = np.array([
                        self.Users[tgt].x,
                        self.Users[tgt].y
                    ])
                    if LA.norm(uav_locations[i] - target_pos2) <= self.distance:
                        print("UAV {} reached target {}".format(i, tgt))
                        reward += 500 #到达目标点的奖励
                        # 处理真正到达：标记完成并分配下一个目标
                        self._on_reach_target(i, tgt)
                else:
                    # 当前 UAV 指向终点（tgt is None）或处于等待状态（tgt == WAIT）
                    # 仅当指向终点且到达时标记到达终点
                    if tgt is None:
                        end_pos = np.array(self.end_loc)
                        distance = LA.norm(uav_locations[i] - end_pos)
                        if distance <= self.distance and not self.uav_reach_final[i]:
                            print("UAV {} heading to end".format(i))
                            self.uav_reach_final[i] = True
                            reward += 1000

            # 无论是否到达终点，都要 append，以保证 rewards 长度为 uav_num
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