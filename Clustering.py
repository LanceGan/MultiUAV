# 四维方差聚类算法，结合地理位置、G2A通信特征、A2G通信特征和数据卸载量进行聚类
from typing import Tuple
import numpy as np
import radio_map_G2A as G2A_rm
import radio_map_A2G as A2G_rm  # 引入 A2G 地图

def _euclidean_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算 a 和 b 之间的平方距离。
    a: (n, d), b: (m, d) -> 返回 (n, m) 的距离矩阵
    """
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)

def _init_centers_kmeans_pp(points: np.ndarray, k: int, random_state: np.random.RandomState) -> np.ndarray:
    n_samples = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=points.dtype)
    idx = random_state.randint(0, n_samples)
    centers[0] = points[idx]
    closest_dist_sq = _euclidean_dist_sq(points, centers[0:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        r = random_state.rand()
        cumulative = np.cumsum(probs)
        next_idx = np.searchsorted(cumulative, r)
        centers[i] = points[next_idx]
        dist_sq_to_new = _euclidean_dist_sq(points, centers[i:i+1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers

def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    """对特征进行 Min-Max 归一化，使其分布在 [0, 1] 区间"""
    ptp = np.ptp(arr)
    if ptp < 1e-8:
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / ptp

def kmeans_4d(points: np.ndarray,
              k: int,
              offload_volumes: np.ndarray = None,
              weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
              max_iters: int = 100,
              tol: float = 1e-4,
              init: str = 'kmeans++',
              random_state: int = None,
              uav_height: float = 0.025,
              point_scale: float = 1.0,
              balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    四维方差聚类算法:
      points: 地理坐标 (n_samples, 2)
      k: 无人机数量 (聚类数)
      offload_volumes: 每个巡航点的数据卸载量 (n_samples,)
      weights: 四维特征的权重 (w_spatial, w_g2a, w_a2g, w_vol)
    """
    if k <= 0:
        raise ValueError('k must be > 0')
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError('points must be 2D array')

    rng = np.random.RandomState(random_state)
    n_samples, n_features = points.shape

    # 1. 地理空间特征 (Spatial)
    points_scaled = points * point_scale
    dataset_max_dist = np.sqrt(np.max(_euclidean_dist_sq(points_scaled, points_scaled))) + 1e-8
    spatial_feats = points_scaled / dataset_max_dist

    # 如果未提供卸载数据量，为了测试生成随机数据
    if offload_volumes is None:
        offload_volumes = rng.uniform(50, 150, n_samples)
    vol_feats = _min_max_norm(offload_volumes).reshape(-1, 1)

    w_s, w_g, w_a, w_v = weights

    # 初始化 G2A 和 A2G 归一化特征
    g2a_feats = np.zeros((n_samples, 1))
    a2g_feats = np.zeros((n_samples, 1))

    # 2 & 3. 提取双地图通信特征 (G2A SINR & A2G Rate)
    if w_g > 0.0 or w_a > 0.0:
        loc_vec = np.hstack([points_scaled, np.full((n_samples, 1), uav_height)])
        
        if w_g > 0.0:
            g2a_list = np.array([float(G2A_rm.getPointDateRate(loc_vec[i:i+1, :])) for i in range(n_samples)])
            g2a_feats = _min_max_norm(g2a_list).reshape(-1, 1)
            
        if w_a > 0.0:
            # 调用 A2G 地图获取上行速率或 SINR
            try:
                a2g_list = np.array([float(A2G_rm.getPointDateRate(loc_vec[i:i+1, :])) for i in range(n_samples)])
            except AttributeError:
                # 若 A2G 函数名不同，请根据实际接口修改
                a2g_list = np.zeros(n_samples) 
            a2g_feats = _min_max_norm(a2g_list).reshape(-1, 1)

    # 4. 构建高维增强特征空间 (Augmented Feature Space)
    # 在这个空间中计算欧氏距离，等价于同时考虑地理距离、G2A连通性差异、A2G速率差异、数据量差异
    augmented_points = np.hstack([
        spatial_feats * w_s,
        g2a_feats * w_g,
        a2g_feats * w_a,
        vol_feats * w_v
    ])

    # K-means 聚类主循环（高维空间）
    if init == 'kmeans++':
        centers_aug = _init_centers_kmeans_pp(augmented_points, k, rng)
    elif init == 'random':
        indices = rng.choice(n_samples, size=k, replace=False)
        centers_aug = augmented_points[indices].copy()

    labels = np.full(n_samples, -1, dtype=int)
    
    for it in range(max_iters):
        # 分配簇：最小化高维空间的距离（即最小化四维方差）
        dist_sq = _euclidean_dist_sq(augmented_points, centers_aug)
        new_labels = np.argmin(dist_sq, axis=1)

        # 更新簇中心
        new_centers_aug = np.zeros_like(centers_aug)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers_aug[idx] = augmented_points[mask].mean(axis=0)
            else:
                new_centers_aug[idx] = augmented_points[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers_aug - new_centers_aug) ** 2, axis=1))
        centers_aug = new_centers_aug
        labels = new_labels

        if np.max(center_shift) <= tol:
            break

    final_dist_sq = _euclidean_dist_sq(augmented_points, centers_aug)
    
    # 后处理：容量/数量约束的平衡分配
    if balanced:
        cost = final_dist_sq  # 代价就是高维空间下的平方距离

        base = n_samples // k
        rest = n_samples % k
        capacities = [base + 1 if i < rest else base for i in range(k)]

        pref_order = np.argsort(cost, axis=1)
        min_cost = np.min(cost, axis=1)
        pts_order = np.argsort(min_cost)
        
        labels_bal = np.full(n_samples, -1, dtype=int)
        cap = capacities.copy()
        
        for p in pts_order:
            for c in pref_order[p]:
                if cap[c] > 0:
                    labels_bal[p] = c
                    cap[c] -= 1
                    break

        unassigned = np.where(labels_bal == -1)[0]
        for p in unassigned:
            labels_bal[p] = int(pref_order[p, 0])

        labels = labels_bal

    # 计算最终物理空间的中心点（还原为输入单位）
    spatial_centers = np.zeros((k, points.shape[1]))  # <--- 动态获取列数
    for idx in range(k):
        mask = (labels == idx)
        if mask.sum() > 0:
            spatial_centers[idx] = points[mask].mean(axis=0)
        else:
            # 极低概率情况下的容错
            spatial_centers[idx] = points[rng.randint(0, n_samples)]

    # 计算整体的方差误差（Inertia）
    inertia = float(np.sum(_euclidean_dist_sq(augmented_points, centers_aug)[np.arange(n_samples), labels]))

    return labels, spatial_centers, inertia

if __name__ == '__main__':
    Users_num = 40
    pts = np.loadtxt('results/datas/Users_%d.txt' % Users_num)
    
    # 假设有一些随机的数据包大小（单位: MB 或 bit）
    np.random.seed(42)
    offloads = np.random.uniform(10, 100, size=Users_num) 
    
    UAV_num = 4
    mode = ('origin','comm_4D')[1]
    random_seed = int(np.random.uniform(0,100))
    
    if mode == 'origin':
        labels, centers, inertia = kmeans_4d(pts, UAV_num, weights=(1.0, 0, 0, 0), max_iters=1000, random_state=random_seed, point_scale=1)
        np.savetxt('results/datas/cluster/Users_%d_Clustered'% Users_num+'UAV_'+str(UAV_num)+'.txt', labels, fmt='%d')
        
    elif mode == 'comm_4D':
        # 权重设置：(地理距离, G2A概率, A2G速率, 卸载量)
        # 根据实际情况调整这四个超参数，使得簇群负载达到最优均衡
        w_tuple = (1.0, 0.4, 0.4, 0.5) 
        
        labels, centers, inertia = kmeans_4d(pts, UAV_num, offload_volumes=offloads, weights=w_tuple, max_iters=1000, uav_height=0.1, random_state=random_seed, point_scale=0.1)
        
        print("Labels:", labels)
        print("Spatial Centers:", centers)
        print("Overall Augmented Inertia:", inertia)
        np.savetxt('results/datas/cluster/Users_%d_Clustered_comm_4D'% Users_num+'UAV_'+str(UAV_num)+'.txt', labels, fmt='%d')