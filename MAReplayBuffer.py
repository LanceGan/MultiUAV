"""
多智能体经验回放池 (Multi-Agent Replay Buffer)
支持集中式训练所需的全局状态存储
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
