"""轻量调试脚本：不依赖 torch，运行环境并用启发式动作验证奖励/分配逻辑"""
import os
import numpy as np
from MultiUAVWorld import MultiUAVWorld


def make_users_file(path, n, length, width):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        # 生成 n 个均匀分布的点（x y z）
        xs = np.linspace(1, length - 1, n)
        ys = np.linspace(1, width - 1, n)
        for i in range(n):
            f.write(f"{xs[i]:.3f} {ys[i]:.3f} 0.0\n")


def heuristic_action(uav_pos, target_pos, dist_max):
    vec = target_pos - uav_pos
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return np.array([0.0, 0.0])
    phi = np.arctan2(vec[1], vec[0])
    step = min(dist, dist_max)
    return np.array([phi, step])


def run_debug():
    # 配置
    uav_num = 2
    user_num = 10
    Length = 40
    Width = 40
    data_size = 100
    dist_max = 0.5

    users_path = f'results/datas/Users_{user_num}_debug.txt'
    make_users_file(users_path, user_num, Length, Width)

    # 创建环境（不从文件加载序列），我们手动设置 uav_traverse 模拟预定义序列
    world = MultiUAVWorld(
        length=Length,
        width=Width,
        uav_num=uav_num,
        user_num=user_num,
        dist_max=dist_max,
        delta_t=0.5,
        t=200,
        uav_h=1,
        data_size=data_size,
        ini_loc=[5.0, 5.0],
        end_loc=[35.0, 35.0],
        users_name=users_path,
        BS_loc=[],
        sequence_path=None,
        safe_distance=0.5,
        comm_range=5.0,
        cooperative_mode='sequential'
    )

    # 设置为预定义序列模式并给出简单序列
    world.sequence_path = 'manual'  # non-None 表示使用预定义分配分支
    world.uav_traverse = {
        0: list(range(0, user_num, 2)),
        1: list(range(1, user_num, 2))
    }

    episodes = 5
    for ep in range(1, episodes + 1):
        obs = world.reset()
        done = False
        step = 0
        print(f"\n=== Episode {ep} ===")
        while not done and step < 200:
            step += 1
            actions = []
            for i in range(uav_num):
                tgt = world.uav_targets[i]
                uav_pos = np.array([world.UAVs[i].x, world.UAVs[i].y])
                if tgt is not None and tgt != world.WAIT_TARGET:
                    target_pos = np.array([world.Users[tgt].x, world.Users[tgt].y])
                    a = heuristic_action(uav_pos, target_pos, world.dist_max)
                else:
                    # 指向终点或等待：向终点小步前进
                    end_pos = np.array(world.end_loc)
                    a = heuristic_action(uav_pos, end_pos, world.dist_max * 0.5)
                actions.append(a)

            next_obs, rewards, done, info, uav_reach = world.step(actions)

            print(f"Step {step}: targets={world.uav_targets}, rewards={[round(r,2) for r in rewards]}, completed={len(world.completed_targets)}")

        print(f"Episode {ep} finished in {step} steps, completed_targets={len(world.completed_targets)}")


if __name__ == '__main__':
    run_debug()
