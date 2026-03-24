#此文件中的距离为100m为单位
"""多无人机协同巡检训练脚本基于MA-TD3算法"""
import torch
from MATD3 import MATD3
from MAReplayBuffer import MAReplayBufferIndividualReward
from MultiUAVWorld import MultiUAVWorld
import numpy as np
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import argparse


def create_parser():
    
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav_num", help="无人机数量", type=int, default=3)
    parser.add_argument("--uav_h", help="无人机飞行高度", type=float, default=1.0)
    parser.add_argument("--gamma", help="折扣因子", type=float, default=0.99)
    parser.add_argument("--buffer", help="replay buffer大小", type=int, default=500000)
    parser.add_argument("--net_width", help="Actor网络宽度", type=int, default=256)
    parser.add_argument("--critic_width", help="Critic网络宽度", type=int, default=512)
    parser.add_argument("--exploration_strategy", help="探索策略", type=str, default="adaptive")
    parser.add_argument("--min_exploration", help="最小探索噪声", type=float, default=0.2)
    parser.add_argument("--max_exploration", help="最大探索噪声", type=float, default=0.5)
    parser.add_argument("--safe_distance", help="安全距离", type=float, default=0.1)
    parser.add_argument("--comm_range", help="通信范围", type=float, default=5.0)
    parser.add_argument("--total_episode", help="总训练回合数", type=int, default=3000)
    return parser


class AdaptiveExploration:
    """自适应探索策略"""
    def __init__(self, initial_noise=0.3, min_noise=0.05, decay_rate=0.9995, 
                 stagnation_threshold=50, boost_factor=1.5):
        self.initial_noise = initial_noise
        self.min_noise = min_noise
        self.current_noise = initial_noise
        self.decay_rate = decay_rate
        self.stagnation_threshold = stagnation_threshold
        self.boost_factor = boost_factor
        
        self.reward_history = []
        self.stagnation_count = 0
        self.last_avg_reward = -float('inf')
        
    def update(self, episode_reward):
        self.reward_history.append(episode_reward)
        
        if len(self.reward_history) >= self.stagnation_threshold:
            current_avg = np.mean(self.reward_history[-self.stagnation_threshold:])
            
            if current_avg <= self.last_avg_reward + 1e-3:
                self.stagnation_count += 1
                if self.stagnation_count >= 3:
                    self.current_noise = min(self.initial_noise, 
                                           self.current_noise * self.boost_factor)
                    self.stagnation_count = 0
                    print(f"检测到性能停滞，增加探索噪声至: {self.current_noise:.4f}")
            else:
                self.stagnation_count = 0
                
            self.last_avg_reward = current_avg
        
        self.current_noise = max(self.min_noise, self.current_noise * self.decay_rate)
        
    def get_noise(self):
        return self.current_noise


def mkdir(path):
    """创建目录"""
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_moving_average(mylist, N):
    """计算移动平均"""
    if len(mylist) < N:
        return mylist
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            moving_aves.append(moving_ave)
    return moving_aves


# ==================== 主训练函数 ====================
def main():
    # 参数解析
    parser = create_parser()
    args = parser.parse_args()
    
    # 提取参数
    uav_num = args.uav_num
    uav_h = args.uav_h
    gamma = args.gamma
    buffer_size = args.buffer
    net_width = args.net_width
    critic_width = args.critic_width
    exploration_strategy = args.exploration_strategy
    min_exploration = args.min_exploration
    max_exploration = args.max_exploration
    safe_distance = args.safe_distance
    comm_range = args.comm_range
    total_episode = args.total_episode
    
    # 创建日志目录
    logs_path = f'logs/MATD3_uav_{uav_num}/'
    mkdir(logs_path)
    
    
    writer = SummaryWriter(logs_path)
    
    # 环境参数
    user_num = 20
    T = 2000
    Length = 40
    Width = 40
    data_size = 300
    
    # 无人机参数
    V_max = 0.50
    delta_t = 0.5
    dist_max = delta_t * V_max
    max_action = np.array([math.pi, dist_max])
    min_action = np.array([-math.pi, 0])
    
    # 巡检序列文件路径
    sequence_path = None
    
    if uav_num == 2:
        user_num = 20
        Length = 40
        Width = 40
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_PSO_%d.npz' % (user_num, uav_num)
        ini_loc = [14.76, 14.83]
        end_loc = [27.62, 23.47]
        BS_loc=np.array([[15.03,8.27,0.25],[26.98,8.25,0.25],[7.43,20.36,0.25],
                        [20.01,20.36,0.25],[32.47,20.36,0.25],[15.10,32.48,0.25],[27.02,32.48,0.25]]) 
    elif uav_num == 3:
        # 5kmx5km area,30 users, 3 UAVs
        user_num = 30
        Length = 50
        Width = 50
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_PSO_%d.npz' % (user_num, uav_num)
        ini_loc = [32.88, 22.67]
        end_loc = [21.62, 48.47]
        BS_loc=np.array([[1.879, 1.034, 0.025],[3.373, 1.031, 0.025],[0.929, 2.545, 0.025],
                        [2.501, 2.545, 0.025],[4.059, 2.545, 0.025],[1.888, 4.060, 0.025],[3.378, 4.060, 0.025]])
    elif uav_num == 4:
        # 6kmx6km area,40 users, 4 UAVs
        user_num = 40
        Length = 60
        Width = 60
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_PSO_%d.npz' % (user_num, uav_num)
        ini_loc = [34.12, 28.79]
        end_loc = [38.46, 45.23]
        BS_loc=np.array([[2.255, 1.241, 0.025],[4.047, 1.238, 0.025],[1.115, 3.054, 0.025],
                         [3.002, 3.054, 0.025],[4.871, 3.054, 0.025],[2.265, 4.872, 0.025],[4.053, 4.872, 0.025]])
    
    
    print("="*70)
    print("MA-TD3 多无人机协同巡检训练")
    print("="*70)
    print(f"无人机数量: {uav_num}")
    print(f"检查点数量: {user_num}")
    print(f"安全距离: {safe_distance} m")
    print(f"通信范围: {comm_range} m")
    print(f"探索策略: {exploration_strategy}")
    print(f"总训练回合: {total_episode}")
    print("="*70 + "\n")
    
    # 创建多无人机环境
    world = MultiUAVWorld(
        length=Length, 
        width=Width, 
        uav_num=uav_num,
        user_num=user_num,
        dist_max=dist_max, 
        delta_t=delta_t, 
        t=T, 
        uav_h=uav_h,
        data_size=data_size, 
        ini_loc=ini_loc, 
        end_loc=end_loc,
        users_name=f'results/datas/Users_{user_num}.txt',
        BS_loc=BS_loc, 
        sequence_path=sequence_path,
        safe_distance=safe_distance,
        comm_range=comm_range,
        cooperative_mode='sequential'
    )
    
    # 状态和动作维度
    local_state_dim = world.local_obs_dim  # 单个无人机的观测维度
    action_dim = world.action_dim  # 单个无人机的动作维度
    global_state_dim = uav_num * local_state_dim
    total_action_dim = uav_num * action_dim
  
  
    # 初始化MA-TD3
    matd3 = MATD3(
        n_agents=uav_num,
        local_state_dim=local_state_dim,
        action_dim=action_dim,
        max_action=max_action,
        min_action=min_action,
        env_with_Dead=True,
        gamma=gamma,
        net_width=net_width,
        critic_net_width=critic_width,
        a_lr=1e-4,
        c_lr=1e-3,
        Q_batchsize=256,
        train_path=logs_path
    )
    
    # 初始化Replay Buffer (使用每智能体独立奖励存储)
    replay_buffer = MAReplayBufferIndividualReward(
        max_size=buffer_size,
        global_state_dim=global_state_dim,
        total_action_dim=total_action_dim,
        n_agents=uav_num
    )
    
    # 模型保存路径
    model_path = './results/models/MA-TD3/UAV_%d/' % uav_num
    mkdir(model_path)
    
    
    # 初始化探索策略
    if exploration_strategy == "adaptive":
        explorer = AdaptiveExploration(
            initial_noise=max_exploration, 
            min_noise=min_exploration
        )
    else:
        expl_noise = max_exploration
    
    # 训练参数
    train_memory_size = 5000
    train_freq = 2
    
    # 早停参数
    best_reward = -float('inf')
    best_success_rate = 0.0
    patience = 300
    no_improve_count = 0
    
    # 记录变量
    ep_rewards = []
    ep_exploration_noise = []
    ep_collision = []
    ep_completed_targets = []
    
    # 轨迹记录
    x_uav = np.zeros([total_episode+1, T+1, uav_num])
    y_uav = np.zeros([total_episode+1, T+1, uav_num])
    z_uav = np.zeros([total_episode+1, T+1, uav_num])
    
    print("开始训练...\n")
    
    # ==================== 主训练循环 ====================
    for episode in tqdm(range(1, total_episode+1), ascii=True, unit='episodes'):
        t_start = time.time()
        
        # 重置环境
        obs_list = world.reset()
        
        # 记录初始位置
        for i, uav in enumerate(world.UAVs):
            x_uav[episode][0][i] = uav.x
            y_uav[episode][0][i] = uav.y
            z_uav[episode][0][i] = uav.h
        
        episode_reward = 0
        done = False
        step_count = 0
        
        # 单回合循环
        while not done:
            step_count += 1
            
            # 噪声计算
            if exploration_strategy == "adaptive":
                current_noise = explorer.get_noise()
            else:
                current_noise = max(min_exploration, 
                                   max_exploration * (1 - episode / total_episode))
            
            # MA-TD3选择动作 策略网络 + 探索噪声
            actions = matd3.select_actions(
                obs_list,
                add_noise=True,
                noise_scale=current_noise
            )
            
            #对每一个动作进行裁剪(虽说在MATD3里已经裁剪过了，这里再保险一次)
            for i,a in enumerate(actions):
                actions[i] = np.clip(a, min_action, max_action)
            
            # 执行动作
            next_obs_list, rewards, done, info, uav_reach = world.step(actions)
            
            # 构造全局状态和动作
            global_state = np.concatenate(obs_list)
            next_global_state = np.concatenate(next_obs_list)
            all_actions = np.concatenate(actions)
            
            # 团队奖励: 使用未到达终点的无人机的平均奖励，
            # 避免已完成的无人机（其即时奖励可能变小）把团队平均拉低，
            # 从而干扰仍在巡检的无人机的学习信号。
            # alive_indices = [i for i in range(world.uav_num) if not world.uav_reach_final[i]]
            # if len(alive_indices) > 0:
            #     team_reward = np.mean([rewards[i] for i in alive_indices])
            # else:
            #     # 若所有无人机都已到达（极端情况），退回到对所有奖励取平均
            #     team_reward = np.mean(rewards)
            
            
            # 存储经验 (保存每个智能体的奖励和done标志)
            replay_buffer.add(
                global_state,
                all_actions,
                np.array(rewards, dtype=np.float32),
                next_global_state,
                np.array(uav_reach, dtype=np.float32)
            )
            
            # 训练网络,按一定频率训练
            if replay_buffer.size >= train_memory_size and step_count % train_freq == 0:
                train_metrics = matd3.train(replay_buffer) 
 
            episode_reward += np.sum(rewards)
            obs_list = next_obs_list
            # done = all_done
            
            # 记录所有UAV位置
            for i, uav in enumerate(world.UAVs):
                x_uav[episode][step_count][i] = uav.x
                y_uav[episode][step_count][i] = uav.y
                z_uav[episode][step_count][i] = uav.h
        
        print(f"Episode {episode} 完成，奖励: {episode_reward:.2f}" 
              f"完成目标: {info['completed_targets']}/{user_num}")
        
        # 回合结束统计
        ep_rewards.append(episode_reward)
        # ep_collision.append(1 if info.get('collision', False) else 0)
        ep_completed_targets.append(info.get('completed_targets', 0))
        ep_exploration_noise.append(current_noise)
        
        # 🔥 新增：将每一轮的总奖励和完成目标数写入 TensorBoard
        writer.add_scalar('Metrics/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Completed_Targets', info.get('completed_targets', 0), episode)
        
        # 更新探索策略
        if exploration_strategy == "adaptive":
            explorer.update(episode_reward)
            current_noise = explorer.get_noise()
        else:
            current_noise = max(min_exploration, 
                               max_exploration * (1 - episode / total_episode))
 
       # 定期保存
        if episode % 100 == 0 and episode >= 100:
            matd3.save(episode, model_path)
        
        # 🔥 新增：保存最佳模型并更新早停计数器
        if episode_reward > best_reward:
            best_reward = episode_reward
            matd3.save('best', model_path)  # 保存后缀为 epbest 的模型
            print(f"🌟 发现新最佳模型！当前最高奖励刷新为: {best_reward:.2f}")
            no_improve_count = 0  # 奖励有提升，重置早停计数器
        else:
            no_improve_count += 1 # 奖励未提升，增加计数
           
        # 早停检查
        if no_improve_count > patience:
            print(f"\n早停触发于 Episode {episode}")
            print(f"最佳奖励: {best_reward:.2f}")
            break
    
    # ==================== 训练结束 ====================
    matd3.save('final', model_path)
    print(f"✅ 模型已保存到: {model_path}\n")
    # 🔥 新增：关闭 TensorBoard writer
    writer.close()
    
    # ==================== 绘制训练奖励曲线 (仅 reward) ====================
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(ep_rewards) + 1)
    plt.plot(episodes, ep_rewards, label='Episode Reward', alpha=0.6)

    # 可选：绘制移动平均以显示趋势
    if len(ep_rewards) > 1:
        N = min(100, max(1, len(ep_rewards) // 4))
        if N > 0 and len(ep_rewards) >= N:
            smoothed = get_moving_average(ep_rewards, N)
            plt.plot(np.arange(N, N + len(smoothed)), smoothed, 'r-', linewidth=2, label=f'MA({N})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('./results/reward_curve/training_reward_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()