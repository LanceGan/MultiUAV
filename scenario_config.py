"""共享场景配置：根据 UAV 数量确定巡检点数和环境参数。"""
import numpy as np

# 通用环境参数
LENGTH = 40
WIDTH = 40
DATA_SIZE = 300
INI_LOC = [14.76, 14.83]
END_LOC = [27.62, 23.47]
BS_LOC = np.array([
    [15.03, 8.27, 0.25], [26.98, 8.25, 0.25], [7.43, 20.36, 0.25],
    [20.01, 20.36, 0.25], [32.47, 20.36, 0.25], [15.10, 32.48, 0.25],
    [27.02, 32.48, 0.25],
])

# Communication reward weights
COMM_REWARD_ALPHA = 0.5  # G2A outage penalty weight
COMM_REWARD_BETA = 0.3   # A2G rate reward weight

# UAV 数量 -> 巡检点数
UAV_USER_MAP = {2: 20, 3: 30, 4: 40}


def get_scenario(uav_num, assignment_mode='sequence', sequence_algorithm='PSO'):
    """根据 UAV 数量返回场景配置。

    Returns:
        dict: user_num, length, width, sequence_path, ini_loc, end_loc, BS_loc, data_size
    """
    if uav_num not in UAV_USER_MAP:
        raise ValueError(f"不支持的无人机数量: {uav_num}")

    user_num = UAV_USER_MAP[uav_num]

    if assignment_mode == 'dynamic':
        sequence_path = None
    else:
        sequence_path = f'./results/datas/sequence/Users_{user_num}_Clusteredsave_path_PathUAV_{sequence_algorithm}_{uav_num}.npz'

    return {
        'user_num': user_num,
        'length': LENGTH,
        'width': WIDTH,
        'data_size': DATA_SIZE,
        'ini_loc': INI_LOC,
        'end_loc': END_LOC,
        'BS_loc': BS_LOC,
        'sequence_path': sequence_path,
    }
