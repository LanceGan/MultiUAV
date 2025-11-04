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


def mkdir(path):
    """创建目录"""
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


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
        
        # 轨迹线
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], 
                c=color, marker='.', linewidth=3.5, markersize=5,
                alpha=0.7, label=f'UAV {i+1}')
        
        # 起点 (方块)
        plt.plot(x_uav[i][0], y_uav[i][0],
                marker='s', markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=2)
        
        # 当前位置 (圆圈)
        plt.plot(x_uav[i][t], y_uav[i][t],
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
    
    print(f"✓ 轨迹图已保存: {savepath}")


def test_matd3_model(
    model_path='Model/MATD3_uav2/',
    model_episode='final',
    uav_num=2,
    test_episodes=10,
    result_path='Test_MultiUAV/',
):
    """
    测试MA-TD3模型
    
    Args:
        model_path: 模型路径
        model_episode: 模型版本 ('final', 'best', 或数字)
        uav_num: 无人机数量
        test_episodes: 测试回合数
        result_path: 结果保存路径
    """
    
    # 创建结果目录
    mkdir(result_path)
    mkdir('results/trajectory')
    
    # 环境参数 (与训练时保持一致)
    user_num = 20
    uav_h = 1.0
    T = 2000
    Length = 40
    Width = 40
    
    V_max = 0.50
    delta_t = 0.5
    dist_max = V_max * delta_t
    
    max_action = np.array([math.pi, dist_max])
    min_action = np.array([-math.pi, 0])
    
    data_size = 300
    
    ini_loc = [14.76, 14.83]
    end_loc = [27.62, 23.47]
    BS_loc = np.array([[15.03,8.27,0.25], [26.98,8.25,0.25], [7.43,20.36,0.25],
                       [20.01,20.36,0.25], [32.47,20.36,0.25], [15.10,32.48,0.25], 
                       [27.02,32.48,0.25]])
    
    print("="*80)
    print("MA-TD3 多无人机模型测试")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"模型版本: {model_episode}")
    print(f"无人机数量: {uav_num}")
    print(f"测试回合数: {test_episodes}")
    print(f"检查点数量: {user_num}")
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
        safe_distance=2.0,
        comm_range=5.0,
        cooperative_mode='sequential'
    )
    
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
        matd3.load(model_episode, model_path)
        print(f"✓ 模型加载成功: {model_path}\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print(f"请确保模型文件存在: {model_path}matd3_critic_ep{model_episode}.pth")
        return
    
    # 初始化统计变量
    success_count = 0
    Complete_time = np.zeros(test_episodes)
    Total_rewards = np.zeros(test_episodes)
    Completed_targets = np.zeros(test_episodes)
    
    # 多无人机轨迹记录
    x_uav_all = np.zeros([test_episodes, uav_num, T+1])
    y_uav_all = np.zeros([test_episodes, uav_num, T+1])
    z_uav_all = np.zeros([test_episodes, uav_num, T+1])
    
    # 用户位置
    x_user = np.zeros(world.user_num)
    y_user = np.zeros(world.user_num)
    z_user = np.zeros(world.user_num)
    
    for i, user in enumerate(world.Users):
        x_user[i] = user.x
        y_user[i] = user.y
        z_user[i] = user.z if hasattr(user, 'z') else 0
    
    # 测试循环
    print("开始测试...\n")
    
    for episode in range(test_episodes):
        print(f"{'='*80}")
        print(f"测试回合 {episode+1}/{test_episodes}")
        print(f"{'='*80}")
        
        # 重置环境
        obs_list = world.reset()
        
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
            
            # 动作裁剪
            for i, a in enumerate(actions):
                actions[i] = np.clip(a, min_action, max_action)
            
            # 执行动作
            next_obs_list, rewards, done, info, uav_reach = world.step(actions)
            
            episode_reward += np.mean(rewards)
            obs_list = next_obs_list
            
            # 记录轨迹
            for i, uav in enumerate(world.UAVs):
                x_uav_all[episode][i][step_count] = uav.x
                y_uav_all[episode][i][step_count] = uav.y
                z_uav_all[episode][i][step_count] = uav.h
            
            # 每100步显示进度
            # if step_count % 100 == 0:
            #     print(f"  步数: {step_count}, 已完成: {info['completed_targets']}/{user_num}")
        
        # 统计结果
        Complete_time[episode] = step_count
        Total_rewards[episode] = episode_reward
        Completed_targets[episode] = info.get('completed_targets', 0)
        
        if info.get('success', False):
            success_count += 1
            print("✓ 任务成功!")
        else:
            print("✗ 任务失败")
        
        # print(f"总步数: {step_count}")
        # print(f"总奖励: {episode_reward:.2f}")
        # print(f"完成目标: {info['completed_targets']}/{user_num}")
        
        # 显示每架无人机到达的目标点
        print(f"各无人机完成情况:")
        for i in range(uav_num):
            completed = len([t for t in world.uav_traverse[i] if t in world.completed_targets])
            total = len(world.uav_traverse[i])
            print(f"  UAV {i}: {completed}/{total} 目标点")
        print()
        
        # 保存单个episode的轨迹数据
        trajectory_file = f'results/trajectory/MultiUAV_uav{uav_num}_ep{episode}.npz'
        np.savez(
            trajectory_file,
            x_uav=x_uav_all[episode][:, :step_count+1],
            y_uav=y_uav_all[episode][:, :step_count+1],
            z_uav=z_uav_all[episode][:, :step_count+1],
            x_user=x_user,
            y_user=y_user,
            z_user=z_user,
            steps=step_count,
            reward=episode_reward,
            success=info.get('success', False),
            completed_targets=info['completed_targets'],
            uav_traverse=[world.uav_traverse[i] for i in range(uav_num)],
            completed_set=list(world.completed_targets)
        )
        print(f"✓ 轨迹数据已保存: {trajectory_file}")
        
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
    print("="*80)
    
    # 绘制统计图表
    plot_test_statistics(
        Complete_time, Total_rewards, Completed_targets,
        success_count, test_episodes, user_num,
        result_path
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
                         save_path):
    """绘制测试统计图表"""
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
    
    print(f"✓ 统计图表已保存: {save_path}test_statistics.png")


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试MA-TD3多无人机模型')
    parser.add_argument('--model_path', type=str, default='Model/MATD3_uav2/',
                       help='模型保存路径')
    parser.add_argument('--model_episode', type=str, default='700',
                       help='模型版本 (final/best/数字)')
    parser.add_argument('--uav_num', type=int, default=2,
                       help='无人机数量')
    parser.add_argument('--test_episodes', type=int, default=10,
                       help='测试回合数')
    parser.add_argument('--result_path', type=str, default='Test_MultiUAV/',
                       help='结果保存路径')
    
    args = parser.parse_args()
    
    # 运行测试
    results = test_matd3_model(
        model_path=args.model_path,
        model_episode=args.model_episode,
        uav_num=args.uav_num,
        test_episodes=args.test_episodes,
        result_path=args.result_path
    )
    
    print("\n" + "="*80)
    print("✅ 测试完成!")
    print("="*80)
    print(f"成功率: {results['success_rate']*100:.1f}%")
    print(f"平均步数: {results['avg_steps']:.1f}")
    print(f"平均奖励: {results['avg_reward']:.2f}")
    print(f"平均完成: {results['avg_completed']:.1f}")
    print("="*80 + "\n")