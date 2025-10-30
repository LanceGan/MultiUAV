"""
多智能体经验回放池 (Multi-Agent Replay Buffer)
支持集中式训练所需的全局状态存储
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAReplayBuffer:
    """
    多智能体经验回放池
    
    存储格式：
    - global_state: 所有智能体的局部观测拼接 [n_agents * local_state_dim]
    - all_actions: 所有智能体的动作拼接 [n_agents * action_dim]
    - reward: 团队奖励或平均奖励 (单个标量)
    - next_global_state: 下一时刻的全局状态
    - done: 终止标志
    """
    
    def __init__(self, max_size, global_state_dim, total_action_dim, n_agents):
        """
        Args:
            max_size: 缓冲区最大容量
            global_state_dim: 全局状态维度 (n_agents * local_state_dim)
            total_action_dim: 总动作维度 (n_agents * action_dim)
            n_agents: 智能体数量
        """
        self.max_size = max_size
        self.ptr = 0  # 当前存储指针
        self.size = 0  # 当前已存储数量
        self.n_agents = n_agents
        
        # 初始化存储空间
        self.global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.all_actions = np.zeros((max_size, total_action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        print(f"[MAReplayBuffer] 初始化完成")
        print(f"  - 最大容量: {max_size}")
        print(f"  - 全局状态维度: {global_state_dim}")
        print(f"  - 总动作维度: {total_action_dim}")
        print(f"  - 智能体数量: {n_agents}")
    
    def add(self, global_state, all_actions, reward, next_global_state, done):
        """
        添加一条经验
        
        Args:
            global_state: np.array [global_state_dim] - 所有智能体状态拼接
            all_actions: np.array [total_action_dim] - 所有智能体动作拼接
            reward: float - 团队奖励或平均奖励
            next_global_state: np.array [global_state_dim]
            done: bool or float - 终止标志
        """
        self.global_states[self.ptr] = global_state
        self.all_actions[self.ptr] = all_actions
        self.rewards[self.ptr] = reward
        self.next_global_states[self.ptr] = next_global_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        随机采样一个batch
        
        Args:
            batch_size: 批次大小
        
        Returns:
            tuple: (global_states, all_actions, rewards, next_global_states, dones)
                所有返回值都是 torch.FloatTensor 在 device 上
        """
        # 随机采样索引
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 转换为Tensor并移到device
        return (
            torch.FloatTensor(self.global_states[indices]).to(device),
            torch.FloatTensor(self.all_actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_global_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )
    
    def __len__(self):
        """返回当前存储的经验数量"""
        return self.size
    
    def is_ready(self, batch_size):
        """检查是否有足够的经验进行训练"""
        return self.size >= batch_size


class MAReplayBufferIndividualReward:
    """
    多智能体经验回放池 (单独奖励版本)
    
    如果每个智能体有独立的奖励信号，使用此版本
    存储每个智能体的奖励：rewards [n_agents]
    """
    
    def __init__(self, max_size, global_state_dim, total_action_dim, n_agents):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        
        # 存储空间
        self.global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.all_actions = np.zeros((max_size, total_action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, n_agents), dtype=np.float32)  # 每个智能体的奖励
        self.next_global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, n_agents), dtype=np.float32)  # 每个智能体的done
        
        print(f"[MAReplayBuffer-IndividualReward] 初始化完成")
        print(f"  - 最大容量: {max_size}")
        print(f"  - 全局状态维度: {global_state_dim}")
        print(f"  - 总动作维度: {total_action_dim}")
        print(f"  - 智能体数量: {n_agents}")
        print(f"  - 奖励模式: 每个智能体独立奖励")
    
    def add(self, global_state, all_actions, rewards, next_global_state, dones):
        """
        Args:
            rewards: np.array [n_agents] - 每个智能体的奖励
            dones: np.array [n_agents] - 每个智能体的done标志
        """
        self.global_states[self.ptr] = global_state
        self.all_actions[self.ptr] = all_actions
        self.rewards[self.ptr] = rewards
        self.next_global_states[self.ptr] = next_global_state
        self.dones[self.ptr] = dones
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, agent_id=None):
        """
        Args:
            agent_id: 如果指定，返回特定智能体的奖励和done；否则返回团队平均
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 选择奖励和done
        if agent_id is not None:
            # 特定智能体
            rewards = self.rewards[indices, agent_id:agent_id+1]
            dones = self.dones[indices, agent_id:agent_id+1]
        else:
            # 团队平均
            rewards = self.rewards[indices].mean(axis=1, keepdims=True)
            dones = self.dones[indices].max(axis=1, keepdims=True)  # 任一智能体done则为done
        
        return (
            torch.FloatTensor(self.global_states[indices]).to(device),
            torch.FloatTensor(self.all_actions[indices]).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(self.next_global_states[indices]).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size):
        return self.size >= batch_size


class PrioritizedMAReplayBuffer:
    """
    优先经验回放 (Prioritized Experience Replay) 多智能体版本
    
    优先级基于TD-error，重要经验被更频繁采样
    适用于样本效率要求高的场景
    """
    
    def __init__(self, max_size, global_state_dim, total_action_dim, n_agents, 
                 alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Args:
            alpha: 优先级指数 (0=均匀采样, 1=完全优先)
            beta: 重要性采样权重 (初始值)
            beta_increment: beta的增量，逐渐趋向1
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # 存储空间
        self.global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.all_actions = np.zeros((max_size, total_action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_global_states = np.zeros((max_size, global_state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        # 优先级数组
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        
        print(f"[PrioritizedMAReplayBuffer] 初始化完成")
        print(f"  - Alpha: {alpha}, Beta: {beta}")
    
    def add(self, global_state, all_actions, reward, next_global_state, done):
        """新经验使用最大优先级"""
        self.global_states[self.ptr] = global_state
        self.all_actions[self.ptr] = all_actions
        self.rewards[self.ptr] = reward
        self.next_global_states[self.ptr] = next_global_state
        self.dones[self.ptr] = done
        
        # 新经验使用最大优先级，保证至少被采样一次
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """基于优先级采样"""
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # Beta annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = (
            torch.FloatTensor(self.global_states[indices]).to(device),
            torch.FloatTensor(self.all_actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_global_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device),
            torch.FloatTensor(weights).unsqueeze(1).to(device),  # IS weights
            indices  # 用于更新优先级
        )
        
        return batch
    
    def update_priorities(self, indices, td_errors):
        """
        更新采样经验的优先级
        
        Args:
            indices: 采样的索引
            td_errors: TD-error (绝对值)
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # 避免0优先级
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return self.size
    
    def is_ready(self, batch_size):
        return self.size >= batch_size
