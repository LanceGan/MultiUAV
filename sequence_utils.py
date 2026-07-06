"""序列规划工具：根据目标点位置和聚类标签，调用序列优化算法计算巡检顺序。"""
import sys
import os
import numpy as np

# 确保 sequence_algorithm 目录下的模块可以导入 radio_map 等依赖
_seq_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sequence_algorithm')
if _seq_dir not in sys.path:
    sys.path.insert(0, _seq_dir)

# 算法名 -> 模块名
_ALGORITHM_MODULES = {
    'GA': 'GA',
    'GA_EQTSP': 'GA_EQTSP',
    'PSO': 'PSO',
    'ACO': 'ACO',
}


def compute_sequence(positions, cluster_labels, algorithm, ini_loc, end_loc):
    """根据目标点位置和聚类标签，为每个 UAV 计算最优巡检序列。

    Args:
        positions: np.array, shape (N, 2 or 3) — 所有目标点坐标
        cluster_labels: np.array, shape (N,) — 每个目标点的 UAV 归属 (0-indexed)
        algorithm: str — 'GA' / 'GA_EQTSP' / 'PSO' / 'ACO'
        ini_loc: list/array — 起点坐标 [x, y] 或 [x, y, z]
        end_loc: list/array — 终点坐标 [x, y] 或 [x, y, z]

    Returns:
        dict: {uav_id: [target_idx, ...]} — 与 uav_traverse 格式相同
    """
    if algorithm not in _ALGORITHM_MODULES:
        raise ValueError(f"不支持的序列算法: {algorithm}，可选: {list(_ALGORITHM_MODULES.keys())}")

    # 动态导入算法模块
    module_name = _ALGORITHM_MODULES[algorithm]
    import importlib
    algo_module = importlib.import_module(module_name)

    # 确保坐标是 3D
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[1] == 2:
        positions = np.hstack([positions, np.zeros((positions.shape[0], 1))])
    ini_loc = np.asarray(ini_loc, dtype=np.float64)
    if len(ini_loc) == 2:
        ini_loc = np.append(ini_loc, 0.0)
    end_loc = np.asarray(end_loc, dtype=np.float64)
    if len(end_loc) == 2:
        end_loc = np.append(end_loc, 0.0)

    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = np.unique(cluster_labels)
    res_indices = {}

    for cid in unique_clusters:
        mask = cluster_labels == cid
        cluster_positions = positions[mask]
        global_indices = np.where(mask)[0]

        # 构造完整路径数据：起点 + 目标点 + 终点
        full_data = np.vstack([ini_loc, cluster_positions, end_loc])
        num_nodes = full_data.shape[0]

        # 实例化算法
        if algorithm in ('GA', 'GA_EQTSP'):
            model = algo_module.GA(num_city=num_nodes, num_total=25, iteration=200, data=full_data.copy())
        elif algorithm == 'PSO':
            model = algo_module.PSO(num_city=num_nodes, data=full_data.copy())
        elif algorithm == 'ACO':
            model = algo_module.ACO(num_city=num_nodes, data=full_data.copy())

        result = model.run()
        # GA/GA_EQTSP 返回 3 个值 (coords, length, indices)，PSO/ACO 返回 2 个 (coords, length)
        if len(result) == 3:
            best_indices = result[2]
        else:
            best_indices = model.best_path

        # 映射回全局索引（排除起点 idx=0 和终点 idx=num_nodes-1）
        orig_order = []
        for idx in best_indices:
            if idx != 0 and idx != num_nodes - 1:
                local_idx = idx - 1  # 减去起点偏移
                orig_order.append(int(global_indices[local_idx]))

        res_indices[int(cid)] = orig_order

    return res_indices


def load_cluster_labels(uav_num):
    """加载聚类标签文件。

    Args:
        uav_num: UAV 数量

    Returns:
        np.array: 每个目标点的聚类标签 (0-indexed)
    """
    user_num = uav_num * 10
    cluster_file = f'results/datas/cluster/Users_{user_num}_Clustered_comm_4DUAV_{uav_num}.txt'
    cluster_labels = np.loadtxt(cluster_file, dtype=int)
    return cluster_labels


def recluster_kmeans(positions, n_clusters, max_iter=100):
    """对目标点位置运行 K-means 聚类。

    Args:
        positions: np.array, shape (N, 2 or 3) — 目标点坐标
        n_clusters: int — 聚类数（等于 UAV 数量）
        max_iter: int — 最大迭代次数

    Returns:
        np.array, shape (N,) — 每个目标点的聚类标签 (0-indexed)
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[1] > 2:
        positions = positions[:, :2]  # 只用 x, y
    n = positions.shape[0]

    # 随机初始化聚类中心（从数据点中选取）
    rng = np.random.RandomState()
    indices = rng.choice(n, n_clusters, replace=False)
    centers = positions[indices].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # 分配：每个点到最近中心
        dists = np.linalg.norm(positions[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)

        # 收敛检查
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 更新中心
        for k in range(n_clusters):
            members = positions[labels == k]
            if len(members) > 0:
                centers[k] = members.mean(axis=0)

    return labels
