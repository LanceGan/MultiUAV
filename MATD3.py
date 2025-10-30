import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================== Actor Network (分散式) ======================
class MActor(nn.Module):
    """多智能体Actor网络 - 每个智能体使用局部观测独立决策"""
    def __init__(self, local_state_dim, action_dim, net_width, maxaction):
        super(MActor, self).__init__()
        self.l1 = nn.Linear(local_state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, action_dim)
        self.maxaction = maxaction

    def forward(self, local_state):
        """
        输入: local_state [batch, local_state_dim] - 单个智能体的局部观测
        输出: action [batch, action_dim]
        """
        a = torch.tanh(self.l1(local_state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = torch.tanh(self.l4(a)) * self.maxaction
        return a


# ====================== Critic Network (集中式) ======================
class MAQ_Critic(nn.Module):
    """多智能体Critic网络 - 使用全局状态和所有智能体的动作"""
    def __init__(self, global_state_dim, total_action_dim, net_width):
        super(MAQ_Critic, self).__init__()
        input_dim = global_state_dim + total_action_dim
        
        # Q1 architecture (Twin Critic 1)
        self.l1 = nn.Linear(input_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width)
        self.l4 = nn.Linear(net_width, 1)

        # Q2 architecture (Twin Critic 2)
        self.l5 = nn.Linear(input_dim, net_width)
        self.l6 = nn.Linear(net_width, net_width)
        self.l7 = nn.Linear(net_width, net_width)
        self.l8 = nn.Linear(net_width, 1)

    def forward(self, global_state, all_actions):
        """
        输入: 
            global_state: [batch, global_state_dim] - 所有智能体状态拼接
            all_actions: [batch, total_action_dim] - 所有智能体动作拼接
        输出: (Q1, Q2) - 两个Q值估计
        """
        sa = torch.cat([global_state, all_actions], dim=1)
        
        # Q1
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        # Q2
        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        
        return q1, q2

    def Q1(self, global_state, all_actions):
        """仅返回Q1，用于Actor更新"""
        sa = torch.cat([global_state, all_actions], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1


# ====================== MA-TD3 Agent ======================
class MATD3Agent(object):
    """单个智能体的MA-TD3实现"""
    def __init__(
        self,
        agent_id,
        local_state_dim,
        action_dim,
        max_action,
        min_action,
        net_width=256,
        a_lr=1e-4,
        c_lr=1e-3,
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        
        # 转换max_action为tensor
        if isinstance(max_action, np.ndarray):
            max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
            min_action = torch.tensor(min_action, dtype=torch.float32).to(device)
        elif not isinstance(max_action, torch.Tensor):
            max_action = torch.tensor([max_action], dtype=torch.float32).to(device)
            min_action = torch.tensor(min_action, dtype=torch.float32).to(device)
        self.max_action = max_action
        self.min_action = min_action

        # Actor网络 (分散式 - 仅使用局部观测)
        self.actor = MActor(local_state_dim, action_dim, net_width, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        # TD3参数
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005

    def select_action(self, local_state, add_noise=False, noise_scale=0.1):
        """
        分散式执行：仅使用局部观测选择动作
        
        Args:
            local_state: 该智能体的局部观测
            add_noise: 是否添加探索噪声
            noise_scale: 噪声比例
        """
        with torch.no_grad():
            if not isinstance(local_state, torch.Tensor):
                local_state = torch.FloatTensor(local_state.reshape(1, -1)).to(device)
            action = self.actor(local_state).cpu().numpy().flatten()
            
            if add_noise:
                noise = np.random.normal(0, noise_scale * self.max_action.cpu().numpy(), 
                                       size=action.shape)
                action += noise 
                    
        return action


# ====================== MA-TD3 Controller ======================
class MATD3(object):
    """多智能体TD3控制器 - 管理所有智能体的训练"""
    def __init__(
        self,
        n_agents,
        local_state_dim,
        action_dim,
        max_action,
        min_action,
        env_with_Dead=True,
        gamma=0.99,
        net_width=256,
        critic_net_width=512,
        a_lr=1e-4,
        c_lr=1e-3,
        Q_batchsize=256,
        train_path=None
    ):
        """
        Args:
            n_agents: 智能体数量
            local_state_dim: 单个智能体的局部观测维度
            action_dim: 单个智能体的动作维度
            max_action: 动作最大值
            env_with_Dead: 环境是否有终止状态
            gamma: 折扣因子
            net_width: Actor网络宽度
            critic_net_width: Critic网络宽度
            a_lr: Actor学习率
            c_lr: Critic学习率
            Q_batchsize: 批次大小
        """
        self.n_agents = n_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.env_with_Dead = env_with_Dead
        self.gamma = gamma
        self.Q_batchsize = Q_batchsize
        
        # 计算全局维度
        self.global_state_dim = n_agents * local_state_dim
        self.total_action_dim = n_agents * action_dim
        
        # 转换max_action
        if isinstance(max_action, np.ndarray):
            max_action = torch.tensor(max_action, dtype=torch.float32).to(device)
        elif not isinstance(max_action, torch.Tensor):
            max_action = torch.tensor([max_action], dtype=torch.float32).to(device)
        self.max_action = max_action
        
        # TD3参数
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.delay_counter = -1
        self.delay_freq = 2  # 延迟策略更新频率
        
        # TensorBoard
        self.writer = SummaryWriter(train_path) if train_path else None
        self.q_iteration = 0
        self.a_iteration = 0
        
        # 创建所有智能体 (参数共享模式)
        print(f"[MATD3] 初始化 {n_agents} 个智能体...")
        print(f"  - 局部状态维度: {local_state_dim}")
        print(f"  - 全局状态维度: {self.global_state_dim}")
        print(f"  - 单智能体动作维度: {action_dim}")
        print(f"  - 总动作维度: {self.total_action_dim}")
        
        self.agents = []
        for i in range(n_agents):
            agent = MATD3Agent(
                agent_id=i,
                local_state_dim=local_state_dim,
                action_dim=action_dim,
                max_action=max_action,
                min_action=min_action,
                net_width=net_width,
                a_lr=a_lr,
                c_lr=c_lr
            )
            self.agents.append(agent)
        
        # 集中式Critic (所有智能体共享)
        self.q_critic = MAQ_Critic(
            self.global_state_dim, 
            self.total_action_dim, 
            critic_net_width
        ).to(device)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        
        print(f"[MATD3] 初始化完成！")

    def select_actions(self, local_states, add_noise=False, noise_scale=0.1):
        """
        所有智能体选择动作 (分散式执行)
        
        Args:
            local_states: list of [local_state_dim] - 每个智能体的局部观测
            add_noise: 是否添加探索噪声
            noise_scale: 噪声比例
        
        Returns:
            actions: list of [action_dim] - 每个智能体的动作
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(local_states[i], add_noise, noise_scale)
            actions.append(action)
        return actions

    def train(self, replay_buffer):
        """
        集中式训练：使用全局信息更新所有智能体
        
        Args:
            replay_buffer: 经验回放池，应包含全局状态和所有智能体的动作/奖励
        """
        self.delay_counter += 1
        
        # 从replay buffer采样
        # 期望格式: (global_state, all_actions, rewards, global_next_state, dones)
        batch = replay_buffer.sample(self.Q_batchsize)
        
        if len(batch) == 5:
            global_s, all_a, r, global_s_prime, dead_mask = batch
        else:
            raise ValueError("Replay buffer返回格式错误，期望5个元素")
        
        # ==================== 更新 Critic ====================
        with torch.no_grad():
            # 1. 计算所有智能体的目标动作
            next_actions = []
            for i, agent in enumerate(self.agents):
                # 提取每个智能体的局部观测
                start_idx = i * self.local_state_dim
                end_idx = (i + 1) * self.local_state_dim
                local_next_state = global_s_prime[:, start_idx:end_idx]
                
                # 使用目标Actor生成动作
                next_action = agent.actor_target(local_next_state)
                
                # Target Policy Smoothing (TD3关键技巧)
                noise = torch.clamp(
                    torch.randn_like(next_action) * self.policy_noise,
                    -self.noise_clip, self.noise_clip
                )
                next_action = torch.clamp(next_action + noise, 
                                        -self.max_action, self.max_action)
                next_actions.append(next_action)
            
            # 拼接所有动作
            all_next_actions = torch.cat(next_actions, dim=1)
            
            # 2. Twin Q-targets (取最小值)
            target_Q1, target_Q2 = self.q_critic_target(global_s_prime, all_next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # 3. 计算TD target
            if self.env_with_Dead:
                target_Q = r + (1 - dead_mask) * self.gamma * target_Q
            else:
                target_Q = r + self.gamma * target_Q
        
        # 4. 当前Q值
        current_Q1, current_Q2 = self.q_critic(global_s, all_a)
        
        # 5. Critic loss (Twin Q的MSE)
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # 6. 更新Critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), 0.5)
        self.q_critic_optimizer.step()
        
        if self.writer:
            self.writer.add_scalar('Loss/critic', q_loss.item(), self.q_iteration)
            self.writer.add_scalar('Q_value/mean', current_Q1.mean().item(), self.q_iteration)
        self.q_iteration += 1
        
        # ==================== 延迟更新 Actor ====================
        if self.delay_counter == self.delay_freq:
            total_actor_loss = 0.0
            
            for i, agent in enumerate(self.agents):
                # 1. 提取该智能体的局部观测
                start_idx = i * self.local_state_dim
                end_idx = (i + 1) * self.local_state_dim
                local_state = global_s[:, start_idx:end_idx]
                
                # 2. 构造所有智能体的动作（仅更新当前智能体的动作）
                current_actions = []
                for j, other_agent in enumerate(self.agents):
                    if j == i:
                        # 当前智能体使用其Actor (需要梯度)
                        action = agent.actor(local_state)
                    else:
                        # 其他智能体使用固定动作 (不需要梯度)
                        other_start = j * self.local_state_dim
                        other_end = (j + 1) * self.local_state_dim
                        other_local_state = global_s[:, other_start:other_end]
                        with torch.no_grad():
                            action = other_agent.actor(other_local_state)
                    current_actions.append(action)
                
                all_current_actions = torch.cat(current_actions, dim=1)
                
                # 3. Actor loss: 最大化Q1值
                actor_loss = -self.q_critic.Q1(global_s, all_current_actions).mean()
                
                # 4. 更新该智能体的Actor
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
                agent.actor_optimizer.step()
                
                total_actor_loss += actor_loss.item()
            
            # 记录平均Actor loss
            if self.writer:
                self.writer.add_scalar('Loss/actor', total_actor_loss / self.n_agents, 
                                     self.a_iteration)
            self.a_iteration += 1
            
            # 5. 软更新所有目标网络
            self._soft_update_targets()
            
            self.delay_counter = -1
        
        return {
            'critic_loss': q_loss.item(),
            'q_value_mean': current_Q1.mean().item()
        }

    def _soft_update_targets(self):
        """软更新所有目标网络"""
        # 更新Critic目标网络
        for param, target_param in zip(self.q_critic.parameters(), 
                                       self.q_critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        # 更新每个智能体的Actor目标网络
        for agent in self.agents:
            for param, target_param in zip(agent.actor.parameters(), 
                                          agent.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, episode, model_path):
        """保存所有智能体的模型"""
        # 保存共享的Critic
        torch.save(self.q_critic.state_dict(), 
                  f"{model_path}matd3_critic_ep{episode}.pth")
        
        # 保存每个智能体的Actor
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), 
                      f"{model_path}matd3_actor{i}_ep{episode}.pth")
        
        print(f"[MATD3] 模型已保存到 {model_path} (Episode {episode})")

    def load(self, episode, model_path):
        """加载所有智能体的模型"""
        # 加载Critic
        self.q_critic.load_state_dict(
            torch.load(f"{model_path}matd3_critic_ep{episode}.pth")
        )
        self.q_critic_target = copy.deepcopy(self.q_critic)
        
        # 加载每个智能体的Actor
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(
                torch.load(f"{model_path}matd3_actor{i}_ep{episode}.pth")
            )
            agent.actor_target = copy.deepcopy(agent.actor)
        
        print(f"[MATD3] 模型已从 {model_path} 加载 (Episode {episode})")