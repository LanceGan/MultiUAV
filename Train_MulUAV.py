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
from collections import deque


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
    parser.add_argument("--min_exploration", help="最小探索噪声", type=float, default=0.05)
    parser.add_argument("--max_exploration", help="最大探索噪声", type=float, default=0.25)
    parser.add_argument("--safe_distance", help="安全距离", type=float, default=0.1)
    parser.add_argument("--comm_range", help="通信范围", type=float, default=5.0)
    parser.add_argument("--total_episode", help="总训练回合数", type=int, default=3000)
    parser.add_argument("--T", help="每回合最大时间步长", type=int, default=2500)
    parser.add_argument("--warmup", help="用启发式策略填充回放缓冲的热启动回合数", type=int, default=80)
    parser.add_argument("--train_memory_size", help="开始训练前回放池的最小样本数", type=int, default=4000)
    parser.add_argument("--train_freq", help="环境步到训练步的频率", type=int, default=2)
    parser.add_argument("--warmup_train_steps", help="warmup完成后额外执行的bootstrap训练步数", type=int, default=1500)
    parser.add_argument("--guided_action_prob_start", help="训练初期使用启发式动作的概率", type=float, default=0.25)
    parser.add_argument("--guided_action_decay_episodes", help="启发式动作概率衰减到0所需回合数", type=int, default=1000)
    parser.add_argument("--guidance_close_radius", help="目标附近启发式混合半径", type=float, default=3.0)
    parser.add_argument("--stable_window", help="稳定成功率统计窗口", type=int, default=50)
    parser.add_argument("--stable_success_threshold", help="触发stable模型保存的成功率阈值", type=float, default=0.95)
    return parser


class AdaptiveExploration:
    """更稳健的自适应探索策略

    特性:
    - 使用滑动窗口平均检测性能变化
    - 支持最大噪声上限、容忍阈值和耐心值
    - 在检测到持续停滞时放大噪声，在性能改善时衰减噪声
    - 提供 reset() 接口
    """
    def __init__(
        self,
        initial_noise=0.3,
        min_noise=0.05,
        max_noise=None,
        decay_rate=0.995,
        window_size=50,
        patience=3,
        boost_factor=1.5,
        improvement_tol=0.01,
    ):
        self.initial_noise = float(initial_noise)
        self.min_noise = float(min_noise)
        self.max_noise = float(max_noise) if max_noise is not None else float(initial_noise)
        self.current_noise = float(initial_noise)
        self.decay_rate = float(decay_rate)
        self.window_size = int(window_size)
        self.patience = int(patience)
        self.boost_factor = float(boost_factor)
        self.improvement_tol = float(improvement_tol)

        self.rewards = deque(maxlen=max(1000, self.window_size * 4))
        self.stagnation_count = 0
        self.last_avg_reward = None

    def update(self, episode_reward):
        self.rewards.append(float(episode_reward))

        if len(self.rewards) < self.window_size:
            self.current_noise = max(self.min_noise, self.current_noise * self.decay_rate)
            return

        recent = list(self.rewards)[-self.window_size:]
        current_avg = float(np.mean(recent))

        if self.last_avg_reward is None:
            self.last_avg_reward = current_avg
            self.current_noise = max(self.min_noise, self.current_noise * self.decay_rate)
            return

        improved = current_avg > self.last_avg_reward * (1.0 + self.improvement_tol) + 1e-6

        if improved:
            self.stagnation_count = 0
            self.current_noise = max(self.min_noise, self.current_noise * self.decay_rate)
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= self.patience:
                new_noise = min(self.max_noise, self.current_noise * self.boost_factor)
                if new_noise > self.current_noise + 1e-6:
                    self.current_noise = new_noise
                    print(f"检测到性能停滞，增加探索噪声至: {self.current_noise:.4f}")
                self.stagnation_count = 0

        self.last_avg_reward = current_avg

    def get_noise(self):
        return float(self.current_noise)

    def reset(self):
        self.current_noise = float(self.initial_noise)
        self.rewards.clear()
        self.stagnation_count = 0
        self.last_avg_reward = None


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
    train_memory_size = args.train_memory_size
    train_freq = args.train_freq
    warmup_train_steps = args.warmup_train_steps
    guided_action_prob_start = args.guided_action_prob_start
    guided_action_decay_episodes = args.guided_action_decay_episodes
    guidance_close_radius = args.guidance_close_radius
    stable_window = args.stable_window
    stable_success_threshold = args.stable_success_threshold
    
    # 创建日志目录
    logs_path = f'logs/MATD3_uav_{uav_num}/'
    mkdir(logs_path)
    
    
    writer = SummaryWriter(logs_path)
    
    # 环境参数
    user_num = 20
    T = args.T # 每回合时间步长，可通过命令行控制
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
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_GAEQTSP_%d.npz' % (user_num, uav_num)
        ini_loc = [14.76, 14.83]
        end_loc = [27.62, 23.47]
        BS_loc=np.array([[15.03,8.27,0.25],[26.98,8.25,0.25],[7.43,20.36,0.25],
                        [20.01,20.36,0.25],[32.47,20.36,0.25],[15.10,32.48,0.25],[27.02,32.48,0.25]]) 
    elif uav_num == 3:
        # 5kmx5km area,30 users, 3 UAVs
        user_num = 30
        Length = 50
        Width = 50
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_GAEQTSP_%d.npz' % (user_num, uav_num)
        ini_loc = [32.88, 22.67]
        end_loc = [21.62, 48.47]
        BS_loc=np.array([[1.879, 1.034, 0.025],[3.373, 1.031, 0.025],[0.929, 2.545, 0.025],
                        [2.501, 2.545, 0.025],[4.059, 2.545, 0.025],[1.888, 4.060, 0.025],[3.378, 4.060, 0.025]])
    elif uav_num == 4:
        # 6kmx6km area,40 users, 4 UAVs
        user_num = 40
        Length = 60
        Width = 60
        sequence_path = './results/datas/sequence/Users_%d_Clusteredsave_path_PathUAV_GAEQTSP_%d.npz' % (user_num, uav_num)
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
    print(f"训练启动样本数: {train_memory_size}")
    print(f"训练频率: 每 {train_freq} 步训练一次")
    print(f"Warmup回合数: {args.warmup}")
    print(f"Warmup后Bootstrap步数: {warmup_train_steps}")
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
    # 为了更快开始训练并便于调试，减小训练启动阈值并增频训练
    train_memory_size = args.train_memory_size
    train_freq = args.train_freq
    
    # 早停参数
    best_reward = -float('inf')
    best_completed_targets = -1
    patience = 300
    no_improve_count = 0
    stable_success_history = deque(maxlen=max(1, stable_window))
    
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
    # ------------------ 一次性 Warmup（只在训练开始时执行） ------------------
    def heuristic_action(uav_pos, target_pos, dist_max):
        vec = target_pos - uav_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return np.array([0.0, 0.0])
        phi = np.arctan2(vec[1], vec[0])
        step = min(dist, dist_max)
        return np.array([phi, step])

    def build_heuristic_actions(world, close_step_scale=1.0):
        actions = []
        for i in range(world.uav_num):
            tgt = world.uav_targets[i]
            uav_pos = np.array([world.UAVs[i].x, world.UAVs[i].y])

            if tgt is not None and tgt != world.WAIT_TARGET:
                target_pos = np.array([world.Users[tgt].x, world.Users[tgt].y])
                dist_to_target = np.linalg.norm(target_pos - uav_pos)
                step_scale = close_step_scale if dist_to_target < guidance_close_radius else 1.0
                a = heuristic_action(uav_pos, target_pos, world.dist_max * step_scale)
            else:
                end_pos = np.array(world.end_loc)
                a = heuristic_action(uav_pos, end_pos, world.dist_max * 0.5)

            actions.append(a)
        return actions

    warmup_episodes = args.warmup
    if warmup_episodes > 0 and replay_buffer.size < train_memory_size:
        print(f"Warmup: 生成 {warmup_episodes} 个启发式回合以填充回放池 (仅一次)...")
        for w in range(warmup_episodes):
            obs_list = world.reset()
            done = False
            step_count_w = 0
            while not done and step_count_w < T:
                step_count_w += 1
                actions = build_heuristic_actions(world, close_step_scale=0.6)

                next_obs_list, rewards, done, info, uav_reach = world.step(actions)

                global_state = np.concatenate(obs_list)
                next_global_state = np.concatenate(next_obs_list)
                all_actions = np.concatenate(actions)
                done_flags = np.full(world.uav_num, float(done), dtype=np.float32)

                replay_buffer.add(
                    global_state,
                    all_actions,
                    np.array(rewards, dtype=np.float32),
                    next_global_state,
                    done_flags
                )

                obs_list = next_obs_list
        print(f"Warmup complete. Replay size = {replay_buffer.size}")

    if replay_buffer.size >= train_memory_size and warmup_train_steps > 0:
        print(f"Bootstrap training: 使用 warmup 经验先训练 {warmup_train_steps} 步...")
        for _ in tqdm(range(warmup_train_steps), ascii=True, unit='updates', leave=False):
            matd3.train(replay_buffer)
        print("Bootstrap training complete.")
    
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
        guide_prob = 0.0
        
        # 单回合循环
        while not done:
            step_count += 1
            
            # 噪声计算
            if exploration_strategy == "adaptive":
                current_noise = explorer.get_noise()
            else:
                current_noise = max(min_exploration, 
                                   max_exploration * (1 - episode / total_episode))

            if guided_action_decay_episodes > 0:
                guide_prob = max(
                    0.0,
                    guided_action_prob_start * (1.0 - (episode - 1) / guided_action_decay_episodes)
                )
            else:
                guide_prob = 0.0
            
            # MA-TD3选择动作 策略网络 + 探索噪声
            actions = matd3.select_actions(
                obs_list,
                add_noise=True,
                noise_scale=current_noise
            )

            heuristic_actions = build_heuristic_actions(world, close_step_scale=0.6)

            # 对每一个动作进行裁剪，并在训练前期混入一定比例的启发式动作。
            for i, a in enumerate(actions):
                guided_blend = 0.0

                tgt = world.uav_targets[i]
                if tgt is not None and tgt != world.WAIT_TARGET:
                    uav_pos = np.array([world.UAVs[i].x, world.UAVs[i].y])
                    target_pos = np.array([world.Users[tgt].x, world.Users[tgt].y])
                    dist_to_target = np.linalg.norm(target_pos - uav_pos)
                else:
                    dist_to_target = np.inf

                if guide_prob > 0.0 and np.random.rand() < guide_prob:
                    guided_blend = 1.0

                # 最后一段是当前 baseline 最容易失败的地方，训练前期额外混入一部分启发式动作。
                if dist_to_target < guidance_close_radius and episode <= guided_action_decay_episodes:
                    guided_blend = max(guided_blend, 0.5)

                if guided_blend > 0.0:
                    actions[i] = (1.0 - guided_blend) * a + guided_blend * heuristic_actions[i]

                actions[i] = np.clip(actions[i], min_action, max_action)

                # 训练时正常记录（无临时调试输出）
            
            # 执行动作
            next_obs_list, rewards, done, info, uav_reach = world.step(actions)
            
            # 构造全局状态和动作
            global_state = np.concatenate(obs_list)
            next_global_state = np.concatenate(next_obs_list)
            all_actions = np.concatenate(actions)
            done_flags = np.full(world.uav_num, float(done), dtype=np.float32)
            
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
                done_flags
            )
            
            # 训练网络,按一定频率训练
            if replay_buffer.size >= train_memory_size and step_count % train_freq == 0:
                try:
                    train_metrics = matd3.train(replay_buffer)
                    # 打印/记录训练指标，便于调试
                    critic_loss = train_metrics.get('critic_loss', None)
                    q_mean = train_metrics.get('q_value_mean', None)
                    # 注释掉训练时的实时打印，避免大量终端输出
                    # print(f"Train @ Ep{episode} Step{step_count}: critic_loss={critic_loss:.4f}, q_mean={q_mean:.4f}")
                    writer.add_scalar('Train/critic_loss', critic_loss, episode * T + step_count)
                    writer.add_scalar('Train/q_value_mean', q_mean, episode * T + step_count)
                except Exception as e:
                    print(f"训练过程中发生错误: {e}")
 
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
        completed_targets = info.get('completed_targets', 0)
        episode_success = 1 if completed_targets >= user_num else 0
        stable_success_history.append(episode_success)
        stable_success_rate = float(np.mean(stable_success_history))
        ep_completed_targets.append(completed_targets)
        ep_exploration_noise.append(current_noise)
        
        # 🔥 新增：将每一轮的总奖励和完成目标数写入 TensorBoard
        writer.add_scalar('Metrics/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Completed_Targets', completed_targets, episode)
        writer.add_scalar('Metrics/Success', episode_success, episode)
        writer.add_scalar('Metrics/Stable_Success_Rate', stable_success_rate, episode)
        writer.add_scalar('Metrics/Guide_Prob', guide_prob, episode)
        writer.add_scalar('Metrics/Episode_TimeSec', time.time() - t_start, episode)
        
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
        if completed_targets > best_completed_targets or (
            completed_targets == best_completed_targets and episode_reward > best_reward
        ):
            best_completed_targets = completed_targets
            best_reward = episode_reward
            matd3.save('best', model_path)
            print(
                f"发现新最优模型！完成目标数: {best_completed_targets}/{user_num}, "
                f"奖励: {best_reward:.2f}"
            )
            no_improve_count = 0
        else:
            no_improve_count += 1

        if len(stable_success_history) == stable_window and stable_success_rate >= stable_success_threshold:
            matd3.save('stable', model_path)
            print(
                f"\n稳定完成阈值已达到：最近 {stable_window} 回合成功率 "
                f"{stable_success_rate:.2%}，已保存 stable 模型并提前结束训练。"
            )
            break
           
        # 早停检查
        # if no_improve_count > patience:
        #     print(f"\n早停触发于 Episode {episode}")
        #     print(f"最佳奖励: {best_reward:.2f}")
        #     break
    
    # ==================== 训练结束 ====================
    matd3.save('final', model_path)
    print(f"✅ 模型已保存到: {model_path}\n")

    # 🔥 新增：关闭 TensorBoard writer
    writer.close()

    # ==================== 保存训练奖励数据并绘制训练奖励曲线 ====================
    # 确保输出目录存在
    reward_dir = './results/reward_curve/'
    mkdir(reward_dir)

    # 将 ep_rewards 写入文件，保证后续可复现和单独绘图
    rewards_arr = np.array(ep_rewards, dtype=float)

    # 如果实际记录条数少于 total_episode，则用 NaN 补齐，保持索引一致性
    if rewards_arr.size < total_episode:
        pad = np.full((total_episode - rewards_arr.size,), np.nan, dtype=float)
        rewards_arr = np.concatenate([rewards_arr, pad])

    np.save(os.path.join(reward_dir, f'ep_rewards_uav{uav_num}.npy'), rewards_arr)

    # 绘图（忽略 NaN 值）
    valid_idx = ~np.isnan(rewards_arr)
    if np.any(valid_idx):
        plt.figure(figsize=(12, 6))
        episodes = np.arange(1, rewards_arr.size + 1)[valid_idx]
        plt.plot(episodes, rewards_arr[valid_idx], label='Episode Reward', alpha=0.6)

        # 可选：绘制移动平均以显示趋势（基于已有有效点）
        valid_rewards = rewards_arr[valid_idx].tolist()
        if len(valid_rewards) > 1:
            N = min(100, max(1, len(valid_rewards) // 4))
            if N > 0 and len(valid_rewards) >= N:
                smoothed = get_moving_average(valid_rewards, N)
                plt.plot(np.arange(N, N + len(smoothed)), smoothed, 'r-', linewidth=2, label=f'MA({N})')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        out_png = os.path.join(reward_dir, f'training_reward_curve_uav{uav_num}.png')
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练奖励曲线已保存到: {out_png}")
    else:
        print("警告：没有可用的训练奖励数据，未生成图像。")


if __name__ == "__main__":
    main()
