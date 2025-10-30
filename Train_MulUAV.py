#此文件中的距离为100m为单位
"""多无人机协同巡检训练脚本基于MA-TD3算法"""
import torch
from MATD3 import MATD3
from MAReplayBuffer import MAReplayBuffer
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
    parser.add_argument("--uav_num", help="无人机数量", type=int, default=2)
    parser.add_argument("--uav_h", help="无人机飞行高度", type=float, default=1.0)
    parser.add_argument("--gamma", help="折扣因子", type=float, default=0.99)
    parser.add_argument("--buffer", help="replay buffer大小", type=int, default=500000)
    parser.add_argument("--net_width", help="Actor网络宽度", type=int, default=256)
    parser.add_argument("--critic_width", help="Critic网络宽度", type=int, default=512)
    parser.add_argument("--exploration_strategy", help="探索策略", type=str, default="adaptive")
    parser.add_argument("--min_exploration", help="最小探索噪声", type=float, default=0.2)
    parser.add_argument("--max_exploration", help="最大探索噪声", type=float, default=0.5)
    parser.add_argument("--safe_distance", help="安全距离", type=float, default=2.0)
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


def draw_multi_uav_location(x_uav, y_uav, t, world, savepath):
    """绘制多无人机轨迹"""
    plt.figure(facecolor='w', figsize=(20, 20))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 绘制每架无人机的轨迹
    for i in range(world.uav_num):
        color = colors[i % len(colors)]
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], 
                c=color, marker='.', linewidth=3.5, markersize=7.5,
                label=f'UAV {i+1}')
        # 当前位置
        plt.plot(x_uav[i][t], y_uav[i][t], 
                c=color, marker='o', markersize=15, 
                markeredgecolor='black', markeredgewidth=2)
    
    # 绘制检查点
    for i, user in enumerate(world.Users):
        plt.plot(user.x, user.y, 'r*', markersize=20, 
                markeredgecolor='black', markeredgewidth=1)
        plt.text(user.x, user.y, f'P{i+1}', fontsize=12, 
                ha='center', va='bottom')
    
    # 绘制建筑物
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        XList = [x1, x2, x2, x1, x1]
        YList = [y1, y1, y2, y2, y1]
        plt.plot(XList, YList, 'k-', linewidth=2)
    
    plt.title(f'Multi-UAV Trajectory (t={t})', fontsize=30)
    plt.xlabel('X (m)', fontsize=20)
    plt.ylabel('Y (m)', fontsize=20)
    plt.xlim((0, world.length))
    plt.ylim((0, world.width))
    plt.legend(fontsize=15, loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()


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
    train_path = f'logs/MATD3_uav{uav_num}/'
    # mkdir(train_path)
    # writer = SummaryWriter(train_path)
    
    # 环境参数
    user_num = 20
    T = 1000
    Length = 40
    Width = 40
    
    V_max = 0.50
    delta_t = 0.5
    dist_max = delta_t * V_max
    
    max_action = np.array([math.pi, dist_max])
    min_action = np.array([-math.pi, 0])
    
    data_size = 300
    
    ini_loc = [14.76, 14.83]
    end_loc = [27.62, 23.47]
    BS_loc=np.array([[15.03,8.27,0.25],[26.98,8.25,0.25],[7.43,20.36,0.25],
                     [20.01,20.36,0.25],[32.47,20.36,0.25],[15.10,32.48,0.25],[27.02,32.48,0.25]]) # 4kmx4km area, 7 BSs

    
    # 巡检顺序
    trave_order = [1, 6, 2, 3, 5, 4]  # GA
    
    print("="*70)
    print("MA-TD3 多无人机协同巡检训练")
    print("="*70)
    print(f"无人机数量: {uav_num}")
    print(f"检查点数量: {len(trave_order)}")
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
        traverse_sequence=trave_order,
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
        train_path=train_path
    )
    
    # 初始化Replay Buffer
    replay_buffer = MAReplayBuffer(
        max_size=buffer_size,
        global_state_dim=global_state_dim,
        total_action_dim=total_action_dim,
        n_agents=uav_num
    )
    
    # 模型保存路径
    model_path = f'Model/MATD3_uav{uav_num}/'
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
            
            # 团队奖励（平均）
            team_reward = np.mean(rewards)
            # all_done = all(dones)
            
            # 存储经验
            replay_buffer.add(
                global_state,
                all_actions,
                team_reward,
                next_global_state,
                float(all(uav_reach))
            )
            
            # 训练网络,按一定频率训练
            if replay_buffer.size >= train_memory_size and step_count % train_freq == 0:
                train_metrics = matd3.train(replay_buffer) 
                # 记录到TensorBoard
                # if step_count % 10 == 0:
                #     writer.add_scalar('Loss/critic', 
                #                     train_metrics['critic_loss'], 
                #                     episode * T + step_count)
                #     writer.add_scalar('Q_value/mean', 
                #                     train_metrics['q_value_mean'], 
                #                     episode * T + step_count)
            
            episode_reward += team_reward
            obs_list = next_obs_list
            # done = all_done
            
            # 记录所有UAV位置
            for i, uav in enumerate(world.UAVs):
                x_uav[episode][step_count][i] = uav.x
                y_uav[episode][step_count][i] = uav.y
                z_uav[episode][step_count][i] = uav.h
        
        print(f"Episode {episode} 完成，奖励: {episode_reward:.2f}，"
              f"完成目标: {info['completed_targets']}/{user_num}，"
              f"碰撞次数: {info['collision_count']}")
        
        # 回合结束统计
        ep_rewards.append(episode_reward)
        ep_collision.append(1 if info.get('collision', False) else 0)
        ep_completed_targets.append(info.get('completed_targets', 0))
        ep_exploration_noise.append(current_noise)
        
        # 更新探索策略
        if exploration_strategy == "adaptive":
            explorer.update(episode_reward)
            current_noise = explorer.get_noise()
 
        # 定期保存
        if episode % 100 == 0 and episode >= 100:
            matd3.save(episode, model_path)
           
        # 早停检查
        if no_improve_count > patience:
            print(f"\n早停于Episode {episode}")
            print(f"最佳奖励: {best_reward:.2f}")
            print(f"最佳成功率: {best_success_rate*100:.1f}%")
            break
    
    # ==================== 训练结束 ====================
    matd3.save('final', model_path)
    # writer.close()
    
    # ==================== 保存训练数据 ====================
    # np.savez(f'{train_path}training_data.npz',
    #          ep_rewards=ep_rewards,
    #          ep_collision=ep_collision,
    #          ep_completed_targets=ep_completed_targets,
    #          ep_exploration_noise=ep_exploration_noise,
    #          x_uav=x_uav[:episode+1],
    #          y_uav=y_uav[:episode+1],
    #          z_uav=z_uav[:episode+1])
    
    print(f"✅ 训练数据已保存到: {train_path}training_data.npz")
    print(f"✅ 训练曲线已保存到: {train_path}training_results.png")
    print(f"✅ 模型已保存到: {model_path}\n")
    
    # ==================== 绘制训练曲线 ====================
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 奖励曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(len(ep_rewards)), ep_rewards, alpha=0.3, label='Episode Reward')
    N = min(100, len(ep_rewards) // 4)
    if N > 0:
        smoothed = get_moving_average(ep_rewards, N)
        ax1.plot(np.arange(len(smoothed)) + N, smoothed, 'r-', 
                linewidth=2, label=f'MA({N})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 3. 完成目标数
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(ep_completed_targets, alpha=0.3, label='Completed')
    if len(ep_completed_targets) > N:
        smoothed = get_moving_average(ep_completed_targets, N)
        ax3.plot(np.arange(len(smoothed)) + N, smoothed, 'b-',
                linewidth=2, label=f'MA({N})')
    ax3.axhline(y=len(trave_order), color='r', linestyle='--', 
               label='Total Targets')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Completed Targets')
    ax3.set_title('Completed Targets per Episode')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 探索噪声
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(ep_exploration_noise, 'orange', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Exploration Noise')
    ax4.set_title('Exploration Noise Decay')
    ax4.grid(alpha=0.3)
    
    # 5. 碰撞率
    ax5 = plt.subplot(2, 3, 5)
    collision_rate = [np.mean(ep_collision[max(0,i-100):i+1]) 
                     for i in range(len(ep_collision))]
    ax5.plot(collision_rate, 'r-', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Collision Rate')
    ax5.set_title('Collision Rate (100-episode MA)')
    ax5.grid(alpha=0.3)
    
    # 6. 奖励分布
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(ep_rewards, bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(x=best_reward, color='r', linestyle='--', 
               linewidth=2, label=f'Best: {best_reward:.2f}')
    ax6.set_xlabel('Reward')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Reward Distribution')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{train_path}training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
   


if __name__ == "__main__":
    main()