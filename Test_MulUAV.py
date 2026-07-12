"""
多无人机MA-TD3测试脚本
用于评估训练好的模型并保存每个智能体的飞行轨迹
"""

import torch
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from MATD3 import MATD3
from MultiUAVWorld import MultiUAVWorld
from utils import wrap_angle, heuristic_action, refine_action, mkdir
from scenario_config import get_scenario
from sequence_utils import compute_sequence, load_cluster_labels, recluster_kmeans


TRAJECTORY_PROFILES = {
    'smooth': {'guidance_radius': 7.0, 'near_target_radius': 2.2, 'max_turn_deg': 16.0},
    'balanced': {'guidance_radius': 4.5, 'near_target_radius': 1.5, 'max_turn_deg': 26.0},
    'agile': {'guidance_radius': 3.0, 'near_target_radius': 1.0, 'max_turn_deg': 35.0},
}


def resolve_model_episode(model_episode, model_path):
    """解析要加载的模型版本。"""
    if model_episode != 'auto':
        return model_episode
    for candidate in ['stable', 'best', 'final']:
        if os.path.exists(os.path.join(model_path, f'matd3_critic_ep{candidate}.pth')):
            return candidate
    return 'final'


def has_model_files(model_path):
    """判断目录中是否存在 MATD3 模型文件。"""
    if not os.path.isdir(model_path):
        return False
    return any(
        name.startswith('matd3_critic_ep') and name.endswith('.pth')
        for name in os.listdir(model_path)
    )


def resolve_trajectory_profile(profile_name, guidance_radius, near_target_radius, max_turn_deg):
    """解析轨迹后处理配置，支持预设档位和手动覆盖。"""
    profile = dict(TRAJECTORY_PROFILES.get(profile_name, TRAJECTORY_PROFILES['smooth']))
    if guidance_radius is not None:
        profile['guidance_radius'] = guidance_radius
    if near_target_radius is not None:
        profile['near_target_radius'] = near_target_radius
    if max_turn_deg is not None:
        profile['max_turn_deg'] = max_turn_deg
    return profile


def get_effective_traj_end(x_series, y_series, t, move_eps=1e-6):
    """裁掉原地悬停造成的静止尾巴，避免轨迹图出现一团密集点。"""
    if t <= 0:
        return 0

    points = np.stack([x_series[:t + 1], y_series[:t + 1]], axis=1)
    move = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving_idx = np.where(move > move_eps)[0]
    if moving_idx.size == 0:
        return 0
    return int(moving_idx[-1] + 1)


def compute_path_stats(x_uav, y_uav, t):
    """统计有效轨迹长度、位移与路径效率。"""
    stats = []
    for i in range(x_uav.shape[0]):
        effective_t = get_effective_traj_end(x_uav[i], y_uav[i], t)
        points = np.stack([x_uav[i][:effective_t + 1], y_uav[i][:effective_t + 1]], axis=1)
        if len(points) <= 1:
            path_len = 0.0
            displacement = 0.0
        else:
            path_len = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            displacement = float(np.linalg.norm(points[-1] - points[0]))

        efficiency = displacement / max(path_len, 1e-8) if path_len > 1e-8 else 1.0
        stats.append({
            'effective_t': int(effective_t),
            'path_len': path_len,
            'displacement': displacement,
            'efficiency': float(efficiency),
        })
    return stats


def draw_multi_uav_trajectory(x_uav, y_uav, t, world, savepath, episode):
    """
    绘制多无人机轨迹图
    
    Args:
        x_uav: [uav_num, T+1] - 各无人机x坐标
        y_uav: [uav_num, T+1] - 各无人机y坐标
        t: 当前时间步
        world: 环境对象
        savepath: 保存路径
        episode: 回合数
    """
    print(f"绘制Episode {episode}轨迹, 时间步: {t}")
    
    plt.figure(figsize=(20, 20))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # 绘制每架无人机的轨迹
    for i in range(world.uav_num):
        color = colors[i % len(colors)]
        effective_t = get_effective_traj_end(x_uav[i], y_uav[i], t)
        path_x = x_uav[i][0:effective_t+1]
        path_y = y_uav[i][0:effective_t+1]
        sample_stride = max(1, len(path_x) // 40)
        
        # 轨迹线
        plt.plot(
            path_x,
            path_y,
            c=color,
            linewidth=3.5,
            alpha=0.85,
            label=f'UAV {i+1}'
        )
        plt.scatter(
            path_x[::sample_stride],
            path_y[::sample_stride],
            c=color,
            s=18,
            alpha=0.7
        )
        
        # 起点 (方块)
        plt.plot(x_uav[i][0], y_uav[i][0],
                marker='s', markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=2)
        
        # 当前位置 (圆圈)
        plt.plot(path_x[-1], path_y[-1],
                marker='o', markersize=15, color=color,
                markeredgecolor='black', markeredgewidth=2)
    
    # 绘制检查点 (星号)
    for i, user in enumerate(world.Users):
        plt.scatter(user.x, user.y, 
                   c='red', marker='*', s=400, 
                   edgecolors='black', linewidths=2,
                   zorder=5, label='Target' if i == 0 else '')
        plt.text(user.x, user.y, f'{i}', 
                fontsize=12, ha='center', va='center',
                color='white', weight='bold')
    
    # 绘制起点和终点
    plt.scatter(world.initial_loc[0], world.initial_loc[1],
               c='green', marker='D', s=300,
               edgecolors='black', linewidths=2,
               label='Start', zorder=5)
    
    plt.scatter(world.end_loc[0], world.end_loc[1],
               c='yellow', marker='D', s=300,
               edgecolors='black', linewidths=2,
               label='End', zorder=5)
    
    
    # plt.title(f'Multi-UAV Trajectory - Episode {episode} (t={t})', fontsize=24)
    plt.xlabel('X (100m)', fontsize=18)
    plt.ylabel('Y (100m)', fontsize=18)
    plt.xlim((0, world.length))
    plt.ylim((0, world.width))
    plt.legend(fontsize=14, loc='best', ncol=2)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory saved: {savepath}")


def test_matd3_model(
    model_episode='auto',
    uav_num=3,
    test_episodes=10,
    model_root='./results/models/MA-TD3',
    model_subdir='Ours',
    T=2500,
    safe_distance=0.1,
    comm_range=5.0,
    assignment_mode='sequence',
    pure_policy=False,
    trajectory_profile='smooth',
    guidance_radius=None,
    near_target_radius=None,
    max_turn_deg=None,
    drop_targets_per_uav=0,
    random_layout=False,
    sequence_algorithm='PSO',
    recluster=False,
    custom_sequence_path=None,
    result_suffix='',
):
    """
    测试MA-TD3模型
    """
    # 动态设定模型路径和结果保存路径
    base_model_path = os.path.join(model_root, f"UAV_{uav_num}")
    if model_subdir:
        model_path = os.path.join(base_model_path, model_subdir)
        if not has_model_files(model_path) and has_model_files(base_model_path):
            print(f"未在子目录 {model_path} 找到模型，回退到旧目录 {base_model_path}")
            model_path = base_model_path
            model_subdir = ''
    else:
        model_path = base_model_path
    resolved_model_episode = resolve_model_episode(model_episode, model_path)
    trajectory_cfg = resolve_trajectory_profile(
        trajectory_profile,
        guidance_radius,
        near_target_radius,
        max_turn_deg,
    )
    guidance_radius = trajectory_cfg['guidance_radius']
    near_target_radius = trajectory_cfg['near_target_radius']
    max_turn_deg = trajectory_cfg['max_turn_deg']
    result_dir_name = f'{resolved_model_episode}{result_suffix}'
    result_path = f'./results/test/UAV_{uav_num}/{result_dir_name}/'
    mkdir(result_path)
    mkdir('./results/figs/trajectory/')
    
    # 基础环境参数 (与训练时保持一致)
    uav_h = 1.0
    V_max = 0.50
    delta_t = 0.5
    dist_max = V_max * delta_t
    
    max_action = np.array([math.pi, dist_max])
    min_action = np.array([-math.pi, 0])
    
    # 根据无人机数量动态配置场景参数
    assignment_mode = str(assignment_mode).strip().lower()
    if assignment_mode not in ('sequence', 'hybrid', 'dynamic'):
        raise ValueError(f"不支持的 assignment_mode: {assignment_mode}")

    # 随机布局下：若指定了序列算法则保持 sequence 模式并动态规划，否则回退到 dynamic
    if random_layout and assignment_mode != 'dynamic':
        if sequence_algorithm:
            print(f"[提示] random_layout=True + sequence_algorithm={sequence_algorithm}，使用动态规划序列")
        else:
            print(f"[提示] random_layout=True 且未指定序列算法，自动切换 assignment_mode: {assignment_mode} -> dynamic")
            assignment_mode = 'dynamic'
    elif sequence_algorithm and not random_layout:
        print(f"[提示] sequence_algorithm={sequence_algorithm}，使用动态规划序列（保持原始巡检点位置）")

    cfg = get_scenario(uav_num, assignment_mode, sequence_algorithm=sequence_algorithm)
    user_num = cfg['user_num']
    Length = cfg['length']
    Width = cfg['width']
    data_size = cfg['data_size']
    sequence_path = custom_sequence_path if custom_sequence_path else cfg['sequence_path']
    ini_loc = cfg['ini_loc']
    end_loc = cfg['end_loc']
    BS_loc = cfg['BS_loc']

    print("="*80)
    print("MA-TD3 多无人机模型测试")
    print("="*80)
    print(f"模型加载路径: {model_path}")
    print(f"模型子目录: {model_subdir if model_subdir else '(none)'}")
    print(f"模型版本: {resolved_model_episode}")
    print(f"无人机数量: {uav_num}")
    print(f"测试回合数: {test_episodes}")
    print(f"检查点数量: {user_num}")
    print(f"安全距离: {safe_distance}")
    print(f"通信范围: {comm_range}")
    print(f"任务分配模式: {assignment_mode}")
    print(f"单回合最大步数: {T}")
    print(f"纯策略测试: {pure_policy}")
    print(f"轨迹档位: {trajectory_profile}")
    print(f"轨迹修正半径: {guidance_radius}")
    print(f"近目标半径: {near_target_radius}")
    print(f"最大转向角: {max_turn_deg}")
    print(f"每UAV丢弃目标数: {drop_targets_per_uav}")
    print(f"随机布局: {random_layout}")
    print(f"序列算法: {sequence_algorithm}")
    print(f"重新聚类: {recluster}")
    print("="*80 + "\n")

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
        sequence_path=sequence_path,  # 🔥 确保传入序列路径
        safe_distance=safe_distance,
        comm_range=comm_range,
        cooperative_mode=assignment_mode,
        drop_targets_per_uav=drop_targets_per_uav,
        random_layout=random_layout
    )

    # sequence_algorithm: 每轮 reset 时动态规划序列（可与 random_layout 组合或独立使用）
    if sequence_algorithm:
        cluster_labels = load_cluster_labels(uav_num)
        seq_alg = sequence_algorithm
        _ini_loc = np.array(ini_loc)
        _end_loc = np.array(end_loc)
        _recluster = recluster
        _uav_num = uav_num

        def _on_reset_callback(w):
            """在位置随机化后，根据新位置重新计算巡检序列。"""
            positions = np.array([[u.x, u.y, u.z] for u in w.Users])
            if _recluster:
                cur_labels = recluster_kmeans(positions, _uav_num)
            else:
                cur_labels = cluster_labels
            new_traverse = compute_sequence(positions, cur_labels, seq_alg, _ini_loc, _end_loc)
            w.set_uav_traverse(new_traverse)

        world.set_on_reset_callback(_on_reset_callback)

    # 初始化MA-TD3
    matd3 = MATD3(
        n_agents=uav_num,
        local_state_dim=world.local_obs_dim,
        action_dim=world.action_dim,
        max_action=max_action,
        min_action=min_action,
        env_with_Dead=True,
        gamma=0.99,
        net_width=256,
        critic_net_width=512,
        a_lr=1e-4,
        c_lr=1e-3,
        Q_batchsize=256
    )
    
    # 加载模型权重
    try:
        matd3.load(resolved_model_episode, model_path)
        print(f"Model loaded: {model_path}\n")
    except Exception as e:
        print(f"Model load FAILED: {e}")
        print(f"Ensure model files exist: {model_path}matd3_critic_ep{resolved_model_episode}.pth")
        return
    
    # 初始化统计变量
    success_count = 0
    Complete_time = np.zeros(test_episodes)
    Total_rewards = np.zeros(test_episodes)
    Completed_targets = np.zeros(test_episodes)
    UAV_Completion_time = np.full((test_episodes, uav_num), np.nan)  # 每UAV完成时间
    UAV_Energy = np.zeros((test_episodes, uav_num))  # 每UAV能耗 (飞行距离)

    # 多无人机轨迹记录
    x_uav_all = np.zeros([test_episodes, uav_num, T+1])
    y_uav_all = np.zeros([test_episodes, uav_num, T+1])
    z_uav_all = np.zeros([test_episodes, uav_num, T+1])
    
    # 测试循环
    print("开始测试...\n")

    for episode in range(test_episodes):
        print(f"{'='*80}")
        print(f"测试回合 {episode+1}/{test_episodes}")
        print(f"{'='*80}")

        # 重置环境
        obs_list = world.reset()

        # 读取当前用户位置（random_layout 时每轮不同）
        x_user = np.zeros(world.user_num)
        y_user = np.zeros(world.user_num)
        z_user = np.zeros(world.user_num)
        for i, user in enumerate(world.Users):
            x_user[i] = user.x
            y_user[i] = user.y
            z_user[i] = user.z if hasattr(user, 'z') else 0
        if world.drop_targets_per_uav > 0:
            print(f"  丢弃目标: {world.dropped_targets_log}")
        prev_actions = [None for _ in range(uav_num)]
        prev_targets = [None for _ in range(uav_num)]
        prev_distances = [None for _ in range(uav_num)]
        stagnation_counts = [0 for _ in range(uav_num)]
        
        # 记录初始位置
        for i, uav in enumerate(world.UAVs):
            x_uav_all[episode][i][0] = uav.x
            y_uav_all[episode][i][0] = uav.y
            z_uav_all[episode][i][0] = uav.h
        
        episode_reward = 0
        done = False
        step_count = 0
        
        # 单回合测试
        while not done and step_count < T:
            step_count += 1
            
            # 使用MA-TD3选择动作（无探索噪声）
            actions = matd3.select_actions(obs_list, add_noise=False)
            
            # 动作裁剪 / 几何修正
            for i, a in enumerate(actions):
                actions[i] = np.clip(a, min_action, max_action)

                if not pure_policy:
                    tgt = world.uav_targets[i]
                    uav_pos = np.array([world.UAVs[i].x, world.UAVs[i].y])

                    if tgt is not None and tgt != world.WAIT_TARGET:
                        target_pos = np.array([world.Users[tgt].x, world.Users[tgt].y])
                        track_key = tgt
                    elif tgt == world.WAIT_TARGET or world.uav_reach_final[i]:
                        target_pos = uav_pos.copy()
                        track_key = 'wait'
                    else:
                        target_pos = np.array(world.end_loc)
                        track_key = 'end'

                    dist_to_target = float(np.linalg.norm(target_pos - uav_pos))
                    heuristic = heuristic_action(uav_pos, target_pos, world.dist_max)
                    same_target = prev_targets[i] == track_key

                    if same_target and prev_distances[i] is not None:
                        progress = prev_distances[i] - dist_to_target
                        if dist_to_target > near_target_radius and progress < 0.01:
                            stagnation_counts[i] += 1
                        else:
                            stagnation_counts[i] = 0
                    else:
                        prev_actions[i] = None
                        prev_distances[i] = None
                        stagnation_counts[i] = 0

                    actions[i] = refine_action(
                        actions[i],
                        heuristic,
                        prev_actions[i],
                        dist_to_target,
                        world.dist_max,
                        guidance_radius,
                        near_target_radius,
                        np.deg2rad(max_turn_deg),
                        same_target=same_target,
                        stagnation_steps=stagnation_counts[i],
                    )
                    prev_targets[i] = track_key
                    prev_distances[i] = dist_to_target

                prev_actions[i] = np.array(actions[i], dtype=np.float32)
            
            # 执行动作
            next_obs_list, rewards, done, info, uav_reach = world.step(actions)
            
            # 使用所有智能体的奖励总和（与训练时的统计保持一致）
            episode_reward += np.sum(rewards)
            obs_list = next_obs_list
            
            # 记录轨迹
            for i, uav in enumerate(world.UAVs):
                x_uav_all[episode][i][step_count] = uav.x
                y_uav_all[episode][i][step_count] = uav.y
                z_uav_all[episode][i][step_count] = uav.h
            

        # 统计结果
        Complete_time[episode] = step_count
        Total_rewards[episode] = episode_reward
        Completed_targets[episode] = info.get('completed_targets', 0)

        if info.get('success', False):
            success_count += 1
            print("SUCCESS")
        else:
            print("FAILED")
        
        # 显示每架无人机到达的目标点
        print(f"各无人机完成情况:")
        for i in range(uav_num):
            completed = len([t for t in world.uav_traverse[i] if t in world.completed_targets])
            total = len(world.uav_traverse[i])
            original_total = len(world.original_uav_traverse[i])
            suffix = f" (原始: {original_total})" if world.drop_targets_per_uav > 0 else ""
            print(f"  UAV {i}: {completed}/{total} 目标点{suffix}")
        print()

        path_stats = compute_path_stats(
            x_uav_all[episode],
            y_uav_all[episode],
            step_count,
        )
        effective_steps = np.array([s['effective_t'] for s in path_stats], dtype=np.int32)
        path_lengths = np.array([s['path_len'] for s in path_stats], dtype=np.float32)
        path_efficiency = np.array([s['efficiency'] for s in path_stats], dtype=np.float32)
        UAV_Completion_time[episode] = effective_steps
        UAV_Energy[episode] = path_lengths
        print(f"有效轨迹步数: {effective_steps.tolist()}")
        print(f"路径效率: {[round(float(v), 3) for v in path_efficiency]}")
        print(f"UAV能耗: {[round(float(v), 2) for v in path_lengths]} (100m单位)")
        
        # 保存单个episode的轨迹数据
        trajectory_file = f'results/datas/trajectory/MultiUAV_uav{uav_num}_ep{episode}.npz'
        mkdir('./results/datas/trajectory/')
        np.savez(
            trajectory_file,
            x_uav=x_uav_all[episode][:, :step_count+1],
            y_uav=y_uav_all[episode][:, :step_count+1],
            z_uav=z_uav_all[episode][:, :step_count+1],
            x_user=x_user,
            y_user=y_user,
            z_user=z_user,
            steps=step_count,
            active_steps=effective_steps,
            path_lengths=path_lengths,
            path_efficiency=path_efficiency,
            reward=episode_reward,
            success=info.get('success', False),
            completed_targets=info['completed_targets'],
            uav_traverse=[world.uav_traverse[i] for i in range(uav_num)],
            completed_set=list(world.completed_targets),
            dropped_targets=str(world.dropped_targets_log) if world.drop_targets_per_uav > 0 else '',
            random_layout=world.random_layout,
            uav_completion_time=effective_steps,
            uav_energy=path_lengths,
        )
        print(f"Trajectory data saved: {trajectory_file}")
        
        # 绘制轨迹图
        traj_img_path = f'{result_path}Trajectory_Episode_{str(episode).zfill(2)}.png'
        draw_multi_uav_trajectory(
            x_uav_all[episode], 
            y_uav_all[episode], 
            step_count,
            world,
            traj_img_path,
            episode
        )
    
    # 计算统计结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"成功率: {success_count}/{test_episodes} ({success_count/test_episodes*100:.1f}%)")
    print(f"平均步数: {np.mean(Complete_time):.1f} ± {np.std(Complete_time):.1f}")
    print(f"平均奖励: {np.mean(Total_rewards):.2f} ± {np.std(Total_rewards):.2f}")
    print(f"平均完成目标: {np.mean(Completed_targets):.1f}/{user_num}")
    print(f"最佳完成: {int(np.max(Completed_targets))}/{user_num}")
    print(f"最差完成: {int(np.min(Completed_targets))}/{user_num}")
    print(f"\n平均UAV完成时间 (步数):")
    uav_avg_time = np.nanmean(UAV_Completion_time, axis=0)
    uav_std_time = np.nanstd(UAV_Completion_time, axis=0)
    for i in range(uav_num):
        print(f"  UAV {i}: {uav_avg_time[i]:.1f} ± {uav_std_time[i]:.1f}")
    print(f"\n平均UAV能耗 (100m单位):")
    uav_avg_energy = np.mean(UAV_Energy, axis=0)
    uav_std_energy = np.std(UAV_Energy, axis=0)
    for i in range(uav_num):
        print(f"  UAV {i}: {uav_avg_energy[i]:.2f} ± {uav_std_energy[i]:.2f}")
    if drop_targets_per_uav > 0:
        print(f"\n每UAV丢弃目标数: {drop_targets_per_uav}")
    print("="*80)

   # 绘制统计图表，保存到当前 UAV 的 result 目录下
    plot_test_statistics(
        Complete_time, Total_rewards, Completed_targets,
        success_count, test_episodes, user_num,
        result_path,
        UAV_Completion_time=UAV_Completion_time,
        UAV_Energy=UAV_Energy,
    )
    
    # 保存完整测试数据
    test_data_file = f'{result_path}test_results_uav{uav_num}.npz'
    np.savez(
        test_data_file,
        x_uav_all=x_uav_all,
        y_uav_all=y_uav_all,
        z_uav_all=z_uav_all,
        x_user=x_user,
        y_user=y_user,
        z_user=z_user,
        Complete_time=Complete_time,
        Total_rewards=Total_rewards,
        Completed_targets=Completed_targets,
        UAV_Completion_time=UAV_Completion_time,
        UAV_Energy=UAV_Energy,
        success_count=success_count,
        test_episodes=test_episodes
    )
    
    print(f"\n✓ 完整测试数据已保存: {test_data_file}")
    print(f"✓ 轨迹图像已保存到: {result_path}")
    print(f"✓ 单独轨迹数据已保存到: results/trajectory/\n")
    
    return {
        'success_rate': success_count / test_episodes,
        'avg_steps': np.mean(Complete_time),
        'avg_reward': np.mean(Total_rewards),
        'avg_completed': np.mean(Completed_targets)
    }


def plot_test_statistics(Complete_time, Total_rewards, Completed_targets,
                         success_count, test_episodes, total_targets,
                         save_path, UAV_Completion_time=None, UAV_Energy=None):
    """绘制测试统计图表"""
    # 支持中文标题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 完成时间
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(test_episodes), Complete_time, 
                    color='skyblue', edgecolor='black', linewidth=1.5)
    ax1.axhline(y=np.mean(Complete_time), color='r', linestyle='--', 
               linewidth=2, label=f'平均: {np.mean(Complete_time):.1f}')
    
    # 标注最值
    max_idx = np.argmax(Complete_time)
    min_idx = np.argmin(Complete_time)
    ax1.text(max_idx, Complete_time[max_idx], f'{Complete_time[max_idx]:.0f}',
            ha='center', va='bottom', fontsize=10, color='red')
    ax1.text(min_idx, Complete_time[min_idx], f'{Complete_time[min_idx]:.0f}',
            ha='center', va='top', fontsize=10, color='green')
    
    ax1.set_xlabel('回合', fontsize=12)
    ax1.set_ylabel('步数', fontsize=12)
    ax1.set_title('完成时间统计', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. 总奖励
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(test_episodes), Total_rewards, 
                    color='lightgreen', edgecolor='black', linewidth=1.5)
    ax2.axhline(y=np.mean(Total_rewards), color='r', linestyle='--',
               linewidth=2, label=f'平均: {np.mean(Total_rewards):.2f}')
    ax2.set_xlabel('回合', fontsize=12)
    ax2.set_ylabel('总奖励', fontsize=12)
    ax2.set_title('总奖励统计', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. 完成目标数
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(test_episodes), Completed_targets, 
                    color='lightcoral', edgecolor='black', linewidth=1.5)
    ax3.axhline(y=np.mean(Completed_targets), color='r', linestyle='--',
               linewidth=2, label=f'平均: {np.mean(Completed_targets):.1f}')
    ax3.axhline(y=total_targets, color='g', linestyle=':', 
               linewidth=2, label=f'目标: {total_targets}')
    
    # 标注完成数
    for i, v in enumerate(Completed_targets):
        ax3.text(i, v, f'{int(v)}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xlabel('回合', fontsize=12)
    ax3.set_ylabel('完成目标数', fontsize=12)
    ax3.set_title('完成目标统计', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_ylim([0, total_targets + 2])
    
    # 4. 综合指标
    ax4 = axes[1, 1]
    
    # 计算指标
    success_rate = success_count / test_episodes * 100
    avg_completion_rate = np.mean(Completed_targets) / total_targets * 100
    avg_efficiency = 100 - (np.mean(Complete_time) / 1000 * 100)  # 假设1000为最大步数
    
    metrics = ['成功率\n(%)', '平均完成率\n(%)', '效率\n(%)']
    values = [success_rate, avg_completion_rate, avg_efficiency]
    colors_bar = ['green', 'blue', 'orange']
    
    bars4 = ax4.bar(metrics, values, color=colors_bar, 
                    alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for bar, value in zip(bars4, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    ax4.set_ylabel('百分比 (%)', fontsize=12)
    ax4.set_title('综合性能指标', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 110])
    ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_path}test_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 单独绘制 per-UAV 统计图
    if UAV_Completion_time is not None and UAV_Energy is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

        uav_num_val = UAV_Completion_time.shape[1]
        uav_labels = [f'UAV {i}' for i in range(uav_num_val)]
        uav_avg_times = np.nanmean(UAV_Completion_time, axis=0)
        uav_avg_energies = np.mean(UAV_Energy, axis=0)

        ax_t = axes2[0]
        bars_t = ax_t.bar(uav_labels, uav_avg_times,
                          color=['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000'][:uav_num_val],
                          edgecolor='black', linewidth=1.5)
        for bar, v in zip(bars_t, uav_avg_times):
            ax_t.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                      f'{v:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax_t.set_ylabel('步数', fontsize=12)
        ax_t.set_title('平均UAV完成时间', fontsize=14, fontweight='bold')
        ax_t.grid(alpha=0.3, axis='y')

        ax_e = axes2[1]
        bars_e = ax_e.bar(uav_labels, uav_avg_energies,
                          color=['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000'][:uav_num_val],
                          edgecolor='black', linewidth=1.5)
        for bar, v in zip(bars_e, uav_avg_energies):
            ax_e.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                      f'{v:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax_e.set_ylabel('距离 (100m单位)', fontsize=12)
        ax_e.set_title('平均UAV能耗', fontsize=14, fontweight='bold')
        ax_e.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{save_path}test_uav_stats.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"UAV stats saved: {save_path}test_uav_stats.png")

    print(f"Stats saved: {save_path}test_statistics.png")


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试MA-TD3多无人机模型')
    parser.add_argument('--model_episode', type=str, default='best',
                       help='模型版本 (auto/stable/best/final/数字)')
    parser.add_argument('--uav_num', type=int, default=3,   # 🔥 默认修改为 3
                       help='无人机数量')
    parser.add_argument('--test_episodes', type=int, default=10,
                       help='测试回合数')
    parser.add_argument('--model_root', type=str, default='./results/models/MA-TD3',
                       help='模型根目录')
    parser.add_argument('--model_subdir', type=str, default='Ours',
                       help='模型子目录；若留空则直接读取 UAV 目录')
    parser.add_argument('--T', type=int, default=2500,
                       help='单回合最大步数')
    parser.add_argument('--safe_distance', type=float, default=0.1,
                       help='安全距离，单位为100m')
    parser.add_argument('--comm_range', type=float, default=5.0,
                       help='通信范围，单位为100m')
    parser.add_argument('--assignment_mode', type=str, default='sequence',
                       help='任务分配模式: sequence / hybrid / dynamic')
    parser.add_argument('--pure_policy', action='store_true',
                       help='禁用测试时轨迹修正，使用纯策略动作')
    parser.add_argument('--trajectory_profile', type=str, default='smooth',
                       choices=list(TRAJECTORY_PROFILES.keys()),
                       help='轨迹后处理档位')
    parser.add_argument('--guidance_radius', type=float, default=None,
                       help='手动覆盖轨迹修正半径，单位为100m')
    parser.add_argument('--near_target_radius', type=float, default=None,
                       help='手动覆盖近目标强修正半径，单位为100m')
    parser.add_argument('--max_turn_deg', type=float, default=None,
                       help='手动覆盖单步最大转向角，单位为度')
    parser.add_argument('--drop_targets', type=int, default=0,
                       help='每个无人机随机丢弃的目标点数量 (鲁棒性测试，0=不丢弃)')
    parser.add_argument('--random_layout', action='store_true',
                       help='随机化巡检点空间分布 (鲁棒性测试)')
    parser.add_argument('--sequence_algorithm', type=str, default='PSO',
                       help='random_layout模式下的序列优化算法 (GA/GA_EQTSP/PSO/ACO)')
    parser.add_argument('--recluster', action='store_true',
                       help='random_layout模式下重新聚类（K-means），而非保持原聚类分配')
    parser.add_argument('--custom_sequence_path', type=str, default=None,
                       help='自定义序列文件路径（覆盖 get_scenario 自动路径）')
    parser.add_argument('--result_suffix', type=str, default='',
                       help='结果目录后缀（区分不同方法的测试结果）')

    args = parser.parse_args()
    
    # 运行测试
    results = test_matd3_model(
        model_episode=args.model_episode,
        uav_num=args.uav_num,
        test_episodes=args.test_episodes,
        model_root=args.model_root,
        model_subdir=args.model_subdir,
        T=args.T,
        safe_distance=args.safe_distance,
        comm_range=args.comm_range,
        assignment_mode=args.assignment_mode,
        pure_policy=args.pure_policy,
        trajectory_profile=args.trajectory_profile,
        guidance_radius=args.guidance_radius,
        near_target_radius=args.near_target_radius,
        max_turn_deg=args.max_turn_deg,
        drop_targets_per_uav=args.drop_targets,
        random_layout=args.random_layout,
        sequence_algorithm=args.sequence_algorithm,
        recluster=args.recluster,
        custom_sequence_path=args.custom_sequence_path,
        result_suffix=args.result_suffix,
    )
    
    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)
    print(f"成功率: {results['success_rate']*100:.1f}%")
    print(f"平均步数: {results['avg_steps']:.1f}")
    print(f"平均奖励: {results['avg_reward']:.2f}")
    print(f"平均完成: {results['avg_completed']:.1f}")
    print("="*80 + "\n")
