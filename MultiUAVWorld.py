from entity import UAV, User
import numpy as np
import os
from numpy import linalg as LA
from rural_world import Rural_world
import radio_map_A2G as A2G
import radio_map_G2A as G2A
from utils import wrap_angle

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
                 sequence_path=None, 
                 safe_distance=0.1,  # 🔥 安全距离 10m
                 comm_range=5.0,
                 cooperative_mode='sequential',
                 drop_targets_per_uav=0,
                 random_layout=False,
                 comm_alpha=0.5,
                 comm_beta=0.3):
        
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

        # 通信奖励权重
        self.comm_alpha = comm_alpha   # G2A 中断惩罚权重
        self.comm_beta = comm_beta     # A2G 速率奖励权重

        # 加载基站位置
        self.BS_loc = BS_loc
        self.set_users()

        # 鲁棒性测试：随机化巡检点空间分布（保持序列索引不变）
        self.random_layout = random_layout
        if self.random_layout:
            for user in self.Users:
                user.x = np.random.uniform(0, self.length)
                user.y = np.random.uniform(0, self.width)
            print(f"已随机化 {len(self.Users)} 个巡检点位置 (范围: [0,{self.length}] x [0,{self.width}])")

        # 环境地图
        self.urban_world = Rural_world(self.BS_loc)
        self.HeightMapMatrix = self.urban_world.Buliding_construct()

        # 预计算最大 A2G 速率用于归一化
        # 注意: getPointDateRate() 返回 SINR (dB)，需要转换为数据速率
        area_km = self.length / 10.0
        np.random.seed(42)  # 固定随机种子，确保可复现
        sample_x = np.random.uniform(0, area_km, 20)
        sample_y = np.random.uniform(0, area_km, 20)
        max_rate = 0.0
        for sx, sy in zip(sample_x, sample_y):
            loc_km = np.array([[sx, sy, self.uav_h / 10]])
            try:
                sinr_db = A2G.getPointDateRate(loc_km)
                rate_mbps = self.BandWidth * np.log2(1 + 10 ** (float(sinr_db) / 10.0))
                max_rate = max(max_rate, rate_mbps)
            except Exception:
                pass
        self.a2g_max_rate = max_rate if max_rate > 0 else 1.0
        
        # 🔥 多无人机任务分配
        self.uav_targets = [None for _ in range(self.uav_num)]   # 每个无人机的当前目标
        self.uav_reach_final = [False for _ in range(self.uav_num)] # 每个无人机是否到达终点
        # 分配方式：如果 sequence_path 提供，则加载预定义的巡检序列；否则使用空序列，后续按需分配
        self.sequence_path = sequence_path
        if self.sequence_path is not None:
            print(f"找到序列文件，加载巡检序列: {sequence_path}")
            npz_data = np.load(sequence_path, allow_pickle=True)
            self.uav_traverse = npz_data['result'].item()
        else:
            print("未找到序列文件，使用空序列")
            self.uav_traverse = {i: [] for i in range(self.uav_num)}

        # 鲁棒性测试：保存原始序列副本，支持每回合随机丢弃目标点
        self.original_uav_traverse = {i: list(seq) for i, seq in self.uav_traverse.items()}
        self.drop_targets_per_uav = drop_targets_per_uav
        self.dropped_targets_log = {}
        self._on_reset_callback = None  # reset 回调（用于 random_layout 下动态规划序列）

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
        self.prev_actions = [None for _ in range(self.uav_num)]
        self.prev_action_targets = [None for _ in range(self.uav_num)]
        self.current_actions = [None for _ in range(self.uav_num)]
        self.current_action_targets = [None for _ in range(self.uav_num)]
        
        # print(f"[MultiUAVWorld] 初始化完成")
        # print(f"  - 无人机数量: {uav_num}")
        # print(f"  - 检查点数量: {len(traverse_sequence)}")


    def set_users(self):
        """加载用户位置（检查点）"""
        assert os.path.exists(self.users_path), f"Users file not found: {self.users_path}"
        data = np.loadtxt(self.users_path)
        self.Users = [User(float(row[0]), float(row[1]), float(row[2])) for row in data]
        
    """初始化所有无人机位置"""
    def set_uavs_loc(self):
        self.UAVs = []
        #所有无人机初始位置相同
        for i in range(self.uav_num):
            x = self.initial_loc[0]
            y = self.initial_loc[1]
            
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
            
           
    """为每个无人机分配第一个飞行目标点,需要提前指定巡检序列self.uav_traverse"""
    def assign_targets(self):
        self.target_owner = {}
        self.assigned_time = {}
        for i in range(self.uav_num):
            if len(self.uav_traverse[i]) > 0:
                target_idx = self.uav_traverse[i][0]
                self.uav_targets[i] = target_idx
                self.target_owner[target_idx] = i
                self.assigned_time[target_idx] = self.t
            else:
                self.uav_targets[i] = None
                
    def assign_initial_targets(self):
        """为每个无人机分配初始目标点（不需要提前指定巡检序列）"""
        for uav_id in range(self.uav_num):
            self._assign_next_target(uav_id)

    def set_uav_traverse(self, uav_traverse_dict):
        """动态更新巡检序列（用于 random_layout 模式下每轮重新规划）。

        Args:
            uav_traverse_dict: dict[int, list[int]] — {uav_id: [target_idx, ...]}
        """
        self.uav_traverse = uav_traverse_dict
        self.original_uav_traverse = {i: list(seq) for i, seq in uav_traverse_dict.items()}

    def set_on_reset_callback(self, callback):
        """设置 reset 回调，在位置随机化之后、目标分配之前调用。

        callback 签名: callback(world) -> None
        典型用途：在 random_layout 模式下，根据随机化后的位置重新计算巡检序列。
        """
        self._on_reset_callback = callback
            

    def reset(self):
        """重置环境"""
        self.set_uavs_loc() #重置无人机位置

        # 鲁棒性测试：每回合重新随机化巡检点位置
        if self.random_layout:
            for user in self.Users:
                user.x = np.random.uniform(0, self.length)
                user.y = np.random.uniform(0, self.width)

        # 回调：在位置随机化后、目标分配前，允许外部重新计算巡检序列
        if self._on_reset_callback is not None:
            self._on_reset_callback(self)

        # 重置分配/完成状态（先清空再分配）
        self.completed_targets = set() #重置已完成目标集
        self.target_owner = {}
        self.assigned_time = {}
        self.uav_targets = [None for _ in range(self.uav_num)]
        self.prev_actions = [None for _ in range(self.uav_num)]
        self.prev_action_targets = [None for _ in range(self.uav_num)]
        self.current_actions = [None for _ in range(self.uav_num)]
        self.current_action_targets = [None for _ in range(self.uav_num)]
        self.uav_reach_final = [False for _ in range(self.uav_num)] #重置每个无人机到达终点标志

        # 鲁棒性测试：每回合随机丢弃目标点
        if self.sequence_path is not None and self.drop_targets_per_uav > 0:
            self._apply_target_drops()

        # 重置巡检序列/任务分配
        if self.sequence_path is None:
            self.assign_initial_targets()
        else:
            self.assign_targets()

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
        构造单个无人机的局部观测。

        使用“自身归一化位置 + 相对目标信息”的表示，
        让策略更容易学会朝目标收敛，而不是仅凭绝对坐标硬拟合。

        观测维度:
        1. 自身位置: [x_norm, y_norm]
        2. 相对目标向量: [dx_norm, dy_norm]
        3. 目标距离: [dist_norm]
        4. 指向目标的单位方向: [dir_x, dir_y]
        5. 剩余时间比例: [remaining_time]
        6. 是否存在有效目标: [has_active_target]
        """
        uav = self.UAVs[uav_id]
        obs = []

        max_dist = np.sqrt(self.length ** 2 + self.width ** 2) + 1e-8
        remaining_time = max(0.0, (self.T - self.t) / max(self.T, 1))

        # 1. 自身归一化位置
        obs.extend([
            uav.x / max(self.length, 1e-8),
            uav.y / max(self.width, 1e-8)
        ])

        # 2. 解析当前目标
        tgt = self.uav_targets[uav_id]
        has_active_target = 0.0
        if tgt is not None and tgt != self.WAIT_TARGET:
            target = self.Users[tgt]
            target_pos = np.array([target.x, target.y])
            has_active_target = 1.0
        elif tgt == self.WAIT_TARGET:
            target_pos = np.array([uav.x, uav.y])
        else:
            if self.uav_reach_final[uav_id]:
                target_pos = np.array([uav.x, uav.y])
            else:
                target_pos = np.array([self.end_loc[0], self.end_loc[1]])
                has_active_target = 1.0

        rel_vec = target_pos - np.array([uav.x, uav.y])
        dist_to_target = LA.norm(rel_vec)
        if dist_to_target > 1e-8:
            dir_vec = rel_vec / dist_to_target
        else:
            dir_vec = np.zeros(2, dtype=np.float32)

        # 3. 相对目标信息
        obs.extend([
            rel_vec[0] / max(self.length, 1e-8),
            rel_vec[1] / max(self.width, 1e-8),
            dist_to_target / max_dist,
            dir_vec[0],
            dir_vec[1],
            remaining_time,
            has_active_target
        ])

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
        self.current_actions = []
        self.current_action_targets = []
        for i in range(self.uav_num):
            if len(actions[i]) >= 2:
                self.current_actions.append(
                    np.array([float(actions[i][0]), float(actions[i][1])], dtype=np.float32)
                )
            else:
                self.current_actions.append(None)

            tgt = self.uav_targets[i]
            if tgt is not None and tgt != self.WAIT_TARGET:
                self.current_action_targets.append(int(tgt))
            else:
                self.current_action_targets.append(tgt)
        
        # 执行动作
        self._execute_actions(actions, uav_locations_pre)
        
        # 当前位置
        uav_locations = np.array([[uav.x, uav.y] for uav in self.UAVs])
        
        # 只有允许协同重分配时，才回收超时分配
        if self.cooperative_mode in ('dynamic', 'hybrid'):
            self._reclaim_stale_assignments()

        # 计算奖励
        rewards = self._compute_rewards(
            uav_locations, uav_locations_pre)
        self.prev_actions = [
            None if action is None else action.copy()
            for action in self.current_actions
        ]
        self.prev_action_targets = list(self.current_action_targets)

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
        fa_total = 0.0
     
        # 执行动作
        for i, uav in enumerate(self.UAVs):
            
            # 如果该 UAV 已完成任务，直接切断动力，原地悬停
            if self.uav_reach_final[i]:
                continue
            
            if len(actions[i]) >= 2:
                uav.move_inside(actions[i][0], actions[i][1], self.dist_max)
                
                # 检查边界
                penalty, bound = self.boundary_margin(uav)
                fa_total += penalty
                
                if not bound:  # 出界，取消动作
                    self.fa += 1
                    uav.x = uav_locations_pre[i][0]
                    uav.y = uav_locations_pre[i][1]

    def _reclaim_stale_assignments(self):
        """回收长期未完成的分配，避免目标被永久占用"""
        '''动态贪心情况下使用'''
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

    def _assign_nearest_available_target(self, uav_id):
        occupied = {
            tgt for tgt, owner in self.target_owner.items()
            if owner != uav_id and tgt not in self.completed_targets
        }
        remaining_targets = [
            idx for idx in range(self.user_num)
            if idx not in self.completed_targets and idx not in occupied
        ]

        if remaining_targets:
            uav_pos = np.array([self.UAVs[uav_id].x, self.UAVs[uav_id].y])
            min_dist = float('inf')
            next_target = None

            for target_idx in remaining_targets:
                target_pos = np.array([self.Users[target_idx].x, self.Users[target_idx].y])
                dist = LA.norm(uav_pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    next_target = target_idx

            if next_target is not None:
                self.target_owner[next_target] = uav_id
                self.assigned_time[next_target] = self.t
                self.uav_targets[uav_id] = next_target
                self.uav_reach_final[uav_id] = False
                return next_target

        if len(self.completed_targets) < self.user_num:
            self.uav_targets[uav_id] = self.WAIT_TARGET
            self.uav_reach_final[uav_id] = False
        else:
            self.uav_targets[uav_id] = None
            self.uav_reach_final[uav_id] = True

        return self.uav_targets[uav_id]

    def _assign_next_target(self, uav_id):
        if self.uav_reach_final[uav_id]:
            return None

        if self.sequence_path is None:
            return self._assign_nearest_available_target(uav_id)

        remaining_targets = [
            tgt for tgt in self.uav_traverse[uav_id]
            if tgt not in self.completed_targets
        ]

        if remaining_targets:
            next_target = remaining_targets[0]
            owner = self.target_owner.get(next_target)
            if owner is None or owner == uav_id:
                self.uav_targets[uav_id] = next_target
                self.target_owner[next_target] = uav_id
                self.assigned_time[next_target] = self.t
            else:
                self.uav_targets[uav_id] = self.WAIT_TARGET
            self.uav_reach_final[uav_id] = False
            return self.uav_targets[uav_id]

        if self.cooperative_mode == 'hybrid':
            return self._assign_nearest_available_target(uav_id)

        self.uav_targets[uav_id] = None
        self.uav_reach_final[uav_id] = True
        return None

    def _on_reach_target(self, uav_id, target_idx):
        if target_idx is None:
            return

        self.completed_targets.add(target_idx)

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

        self._assign_next_target(uav_id)

    def _apply_target_drops(self):
        """随机丢弃每架无人机巡检序列中的 drop_targets_per_uav 个目标点。
        每回合开始时在 reset() 中调用，用于鲁棒性测试。
        """
        self.dropped_targets_log = {}
        for i in range(self.uav_num):
            original_seq = list(self.original_uav_traverse[i])
            n_drop = min(self.drop_targets_per_uav, len(original_seq))
            if n_drop > 0:
                drop_indices = set(np.random.choice(len(original_seq), n_drop, replace=False))
                dropped = [original_seq[j] for j in drop_indices]
                keep = [original_seq[j] for j in range(len(original_seq)) if j not in drop_indices]
                self.uav_traverse[i] = keep
                self.dropped_targets_log[i] = dropped
            else:
                self.uav_traverse[i] = list(original_seq)
                self.dropped_targets_log[i] = []

    def _get_g2a_outage(self, x, y):
        """获取位置 (x, y) 处的 G2A 最小中断概率。坐标单位: 100m。"""
        loc_km = np.zeros((1, 3))
        loc_km[0, 0] = x / 10
        loc_km[0, 1] = y / 10
        loc_km[0, 2] = self.uav_h / 10
        try:
            outage = G2A.getPointMiniOutage(loc_km)
            return float(outage[0][0])
        except Exception:
            return 0.0

    def _get_a2g_rate(self, x, y):
        """获取位置 (x, y) 处的 A2G 数据速率 (Mbps)。坐标单位: 100m。

        注意: A2G.getPointDateRate() 实际返回的是最大 SINR (dB)，
        这里使用公式 R = B * log2(1 + 10^(SINR/10)) 转换为数据速率。
        """
        loc_km = np.zeros((1, 3))
        loc_km[0, 0] = x / 10
        loc_km[0, 1] = y / 10
        loc_km[0, 2] = self.uav_h / 10
        try:
            sinr_db = A2G.getPointDateRate(loc_km)  # 实际返回 SINR (dB)
            # 转换为数据速率: R = B * log2(1 + 10^(SINR/10))
            # B = 1 MHz (BandWidth), SINR 单位为 dB
            rate_mbps = self.BandWidth * np.log2(1 + 10 ** (float(sinr_db) / 10.0))
            return rate_mbps
        except Exception:
            return 0.0

    def _compute_rewards(self, uav_locations, uav_locations_pre):
        """计算每个无人机的奖励。

        设计原则:
        1. 稀疏奖励只负责“是否到达目标”。
        2. 稠密奖励只负责“是否更接近目标”，尽量避免相互打架。
        3. 目标附近增加专门的绕圈/冲过目标惩罚，解决最后一段不收敛的问题。
        """
        rewards = []

        for i in range(self.uav_num):
            reward = 0.0

            if self.uav_reach_final[i]:
                rewards.append(reward)
                continue

            tgt = self.uav_targets[i]
            active_target = tgt is not None and tgt != self.WAIT_TARGET

            if active_target:
                target_pos = np.array([self.Users[tgt].x, self.Users[tgt].y])
            else:
                target_pos = np.array(uav_locations[i])

            dist_cur = LA.norm(uav_locations[i] - target_pos)
            dist_pre = LA.norm(uav_locations_pre[i] - target_pos)
            progress = dist_pre - dist_cur

            move_vec = uav_locations[i] - uav_locations_pre[i]
            move_norm = LA.norm(move_vec)
            dir_to_target = target_pos - uav_locations_pre[i]
            dir_norm = LA.norm(dir_to_target)

            if active_target:
                # 小的时间代价，鼓励更短时间完成任务。
                reward -= 0.2

                # 1. 归一化进度奖励: 每一步是否真的更接近目标。
                norm_progress = progress / max(self.dist_max, 1e-8)
                reward += 24.0 * norm_progress

                # 2. 势能差塑形: 越接近目标，最后几步的改进越值钱。
                potential_pre = 1.0 / (dist_pre + 1.0)
                potential_cur = 1.0 / (dist_cur + 1.0)
                reward += 35.0 * (potential_cur - potential_pre)

                # 3. 对齐奖励: 鼓励朝目标方向运动，反方向则受罚。
                if move_norm > 1e-8 and dir_norm > 1e-8:
                    forward = np.dot(move_vec, dir_to_target) / (move_norm * dir_norm)
                    forward = float(np.clip(forward, -1.0, 1.0))
                    reward += 4.0 * forward

                    # 4. 目标附近惩罚切向绕圈。
                    if dist_pre < 4.0:
                        close_factor = (4.0 - dist_pre) / 4.0
                        tangential_ratio = np.sqrt(max(0.0, 1.0 - forward ** 2))
                        reward -= 6.0 * close_factor * tangential_ratio

                # 5. 目标附近若没有实质推进，追加惩罚。
                if dist_pre < 4.0:
                    close_factor = (4.0 - dist_pre) / 4.0
                    if progress < 0.0:
                        reward -= 8.0 * close_factor
                    elif progress < 0.02:
                        reward -= 2.5 * close_factor

                    # 距离很近时，仍然大步飞行通常会导致越过目标或绕圈。
                    if dist_pre < 2.0:
                        step_ratio = min(move_norm / max(self.dist_max, 1e-8), 1.0)
                        reward -= 3.0 * ((2.0 - dist_pre) / 2.0) * step_ratio

                    # 6. 给最后一段额外收敛奖励，帮助稳定钻进到达阈值。
                    reward += 8.0 * max(0.0, 1.5 - dist_cur) / 1.5
                elif move_norm < 1e-3:
                    reward -= 0.5

                # 到达目标奖励（真正触发任务完成的唯一稀疏奖励）
                prev_action = self.prev_actions[i]
                curr_action = self.current_actions[i]
                same_target_action = (
                    prev_action is not None
                    and curr_action is not None
                    and self.prev_action_targets[i] == tgt
                    and self.current_action_targets[i] == tgt
                )
                if same_target_action:
                    turn_delta = abs(wrap_angle(float(curr_action[0]) - float(prev_action[0])))
                    step_delta = abs(float(curr_action[1]) - float(prev_action[1])) / max(self.dist_max, 1e-8)
                    if dist_pre < 4.0:
                        close_factor = (4.0 - dist_pre) / 4.0
                        reward -= 1.8 * close_factor * (turn_delta / np.pi)
                        reward -= 0.8 * close_factor * step_delta
                    else:
                        reward -= 0.25 * (turn_delta / np.pi)
                        reward -= 0.08 * step_delta

                if dist_cur <= self.distance:
                    print("UAV {} reached target {}".format(i, tgt))
                    reward += 400.0
                    self._on_reach_target(i, tgt)

            # 通信奖励（仅对有活跃目标的 UAV 生效）
            if active_target:
                # G2A 中断惩罚: 中断概率越高，惩罚越大
                g2a_outage = self._get_g2a_outage(uav_locations[i][0], uav_locations[i][1])
                r_g2a = -self.comm_alpha * g2a_outage
                reward += r_g2a

                # A2G 速率奖励: 速率越高，奖励越大（归一化到 [0, 1]）
                a2g_rate = self._get_a2g_rate(uav_locations[i][0], uav_locations[i][1])
                r_a2g = self.comm_beta * a2g_rate / max(self.a2g_max_rate, 1e-8)
                reward += r_a2g

            # 接近其他无人机的惩罚
            for j in range(self.uav_num):
                if i == j or self.uav_reach_final[j]:
                    continue

                if self.uav_targets[i] is None and self.uav_targets[j] is None:
                    continue

                dist_to_other = LA.norm(uav_locations[i] - uav_locations[j])
                if dist_to_other < self.safe_distance * 2:
                    penalty = (self.safe_distance * 2 - dist_to_other) / self.safe_distance
                    reward -= 20.0 * penalty

            rewards.append(reward)

        # 团队奖励（所有目标完成）
        if sum(self.uav_reach_final) == self.uav_num:
            print("All UAVs reached all targets!")
            team_bonus = 1000.0 + max(self.T - self.t, 0) * 2.0
            rewards = [r + team_bonus / self.uav_num for r in rewards]
            self.terminal = True

        return rewards

    def _check_done(self):
        """检查是否完成（任务完成或超时）"""
        return self.terminal or self.t >= self.T    
        
    def boundary_margin(self, uav):
        """检查无人机是否出界"""
        margin = 1.0
        penalty_factor = 100 / self.uav_num
        half_x = self.length / 2
        half_y = self.width / 2

        x_exceed = max(abs(uav.x - half_x) - margin * half_x, 0.0)
        y_exceed = max(abs(uav.y - half_y) - margin * half_y, 0.0)

        if x_exceed == 0.0 and y_exceed == 0.0:
            return 0.0, True
        return penalty_factor * (x_exceed**2 + y_exceed**2), False

    @property
    def local_obs_dim(self):
        """单个无人机的观测维度"""
        # [x, y, dx, dy, dist, dir_x, dir_y, remaining_time, has_target]
        return 9
    
    @property
    def action_dim(self):
        """单个无人机的动作维度"""
        return 2  # [phi, dist]
