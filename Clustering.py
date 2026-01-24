from typing import Tuple
import numpy as np
import radio_map_G2A as G2A_rm

def _euclidean_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""计算 a 和 b 之间的平方距离。
	a: (n, d), b: (m, d) -> 返回 (n, m) 的距离矩阵
	"""
	# (n,1,d) - (1,m,d) -> (n,m,d)
	diff = a[:, None, :] - b[None, :, :]
	return np.sum(diff * diff, axis=2)


def _init_centers_kmeans_pp(points: np.ndarray, k: int, random_state: np.random.RandomState) -> np.ndarray:
	n_samples = points.shape[0]
	centers = np.empty((k, points.shape[1]), dtype=points.dtype)
	# choose first center uniformly
	idx = random_state.randint(0, n_samples)
	centers[0] = points[idx]
	# distances to nearest center
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


def kmeans(points: np.ndarray,
		   k: int,
		   max_iters: int = 100,
		   tol: float = 1e-4,
		   init: str = 'kmeans++',
		   random_state: int = None,
		   comm_weight: float = 0.0,
		   uav_height: float = 0.025,
		   point_scale: float = 1.0,
		   balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
	"""
	参数:
	  points: ndarray (n_samples, n_features)
	  k: 聚类数
	  max_iters: 最大迭代次数
	  tol: 中心移动阈值（收敛判定）
	  init: 'kmeans++' 或 'random'
	  random_state: int or None
	返回:
	  labels, centers, inertia
	"""
	if k <= 0:
		raise ValueError('k must be > 0')
	points = np.asarray(points)
	if points.ndim != 2:
		raise ValueError('points must be 2D array')

	rng = np.random.RandomState(random_state)
	n_samples, n_features = points.shape

	# 支持输入坐标的单位缩放（例如输入单位为百米，point_scale=0.1 将其转换为 km）
	if point_scale != 1.0:
		points_scaled = points * point_scale
	else:
		points_scaled = points

	# 如果需要考虑通信质量，预计算每个点的 SINR 并归一化到 [0,1]
	if comm_weight is not None and comm_weight > 0.0:
		# radio_map_G2A.getPointDateRate 接受 (n,3) 的坐标矩阵，使用缩放后的坐标（应为 km）
		loc_vec = np.hstack([points_scaled, np.full((n_samples, 1), uav_height)])
		sinr_list = np.zeros(n_samples, dtype=float)
		for i in range(n_samples):
			# getPointDateRate 返回单点的最大 SINR（dB）
			sinr_list[i] = float(G2A_rm.getPointDateRate(loc_vec[i:i+1, :]))
		# 归一化（min-max），确保数值稳定
		min_s, max_s = np.min(sinr_list), np.max(sinr_list)
		if max_s - min_s < 1e-8:
			sinr_norm = np.zeros_like(sinr_list)
		else:
			sinr_norm = (sinr_list - min_s) / (max_s - min_s)
	else:
		sinr_norm = None

	if init == 'kmeans++':
		centers = _init_centers_kmeans_pp(points_scaled, k, rng)
	elif init == 'random':
		indices = rng.choice(n_samples, size=k, replace=False)
		centers = points_scaled[indices].astype(points.dtype)
	else:
		raise ValueError("Unsupported init method: {}".format(init))

	labels = np.full(n_samples, -1, dtype=int)
	# 预计算数据集尺度用于空间距离归一化（基于缩放后的坐标）
	eps = 1e-8
	# 始终基于缩放后的坐标计算最大距离，避免后续 None 错误
	dataset_max_dist = np.sqrt(np.max(_euclidean_dist_sq(points_scaled, points_scaled))) + eps
	for it in range(max_iters):
		# assign
		dist_sq = _euclidean_dist_sq(points_scaled, centers)  # (n_samples, k)
		# 如果启用了通信权重，使用归一化空间距离与归一化 SINR 的线性组合作为代价
		if sinr_norm is not None and comm_weight is not None and comm_weight > 0.0:
			spatial_dist = np.sqrt(dist_sq)
			spatial_norm = spatial_dist / (dataset_max_dist + eps)
			# sinr_norm 范围在 [0,1]，更高的 sinr -> 更低的代价
			combined_cost = (1.0 - comm_weight) * spatial_norm + comm_weight * (1.0 - sinr_norm[:, None])
			new_labels = np.argmin(combined_cost, axis=1)
		else:
			new_labels = np.argmin(dist_sq, axis=1)

		# update centers
		new_centers = np.zeros_like(centers)
		counts = np.zeros(k, dtype=int)
		for idx in range(k):
			mask = (new_labels == idx)
			cnt = mask.sum()
			counts[idx] = cnt
			if cnt > 0:
				new_centers[idx] = points_scaled[mask].mean(axis=0)
			else:
				# empty cluster: reinitialize to a random point
				new_centers[idx] = points[rng.randint(0, n_samples)]

		center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
		centers = new_centers
		labels = new_labels

		if np.max(center_shift) <= tol:
			break

	# inertia: 如果使用通信权重，基于混合代价计算，否则使用平方欧氏距离
	final_dist_sq = _euclidean_dist_sq(points_scaled, centers)
	if sinr_norm is not None and comm_weight is not None and comm_weight > 0.0:
		final_spatial = np.sqrt(final_dist_sq) / (dataset_max_dist + eps)
		final_combined = (1.0 - comm_weight) * final_spatial + comm_weight * (1.0 - sinr_norm[:, None])
		inertia = np.sum(final_combined[np.arange(n_samples), labels])
	else:
		inertia = np.sum(final_dist_sq[np.arange(n_samples), labels])

	# 如果要求平衡簇大小，则进行容量约束分配
	if balanced:
		# 计算分配代价矩阵，使用之前的混合代价定义
		if sinr_norm is not None and comm_weight is not None and comm_weight > 0.0:
			spatial = np.sqrt(final_dist_sq) / (dataset_max_dist + eps)
			cost = (1.0 - comm_weight) * spatial + comm_weight * (1.0 - sinr_norm[:, None])
		else:
			# 仅空间距离（归一化）
			cost = np.sqrt(final_dist_sq) / (dataset_max_dist + eps)

		# capacities：尽量均匀分配点
		base = n_samples // k
		rest = n_samples % k
		capacities = [base + 1 if i < rest else base for i in range(k)]

		# 贪心分配：先按点的最小代价强度排序，依次为每个点分配首个有容量的簇
		pref_order = np.argsort(cost, axis=1)  # (n_samples, k)
		min_cost = np.min(cost, axis=1)
		pts_order = np.argsort(min_cost)  # points with stronger preference first
		labels_bal = np.full(n_samples, -1, dtype=int)
		cap = capacities.copy()
		for p in pts_order:
			for c in pref_order[p]:
				if cap[c] > 0:
					labels_bal[p] = c
					cap[c] -= 1
					break

		# 对于极端情况（理论上不应出现），如果仍有未分配点，直接按最小代价分配
		unassigned = np.where(labels_bal == -1)[0]
		if unassigned.size > 0:
			for p in unassigned:
				labels_bal[p] = int(pref_order[p, 0])

		labels = labels_bal
		# 重新计算 centers（基于缩放后的坐标）
		new_centers = np.zeros_like(centers)
		for idx in range(k):
			mask = (labels == idx)
			if mask.sum() > 0:
				new_centers[idx] = points_scaled[mask].mean(axis=0)
			else:
				new_centers[idx] = centers[idx]
		centers = new_centers
		# 重新计算 inertia（基于 cost）
		if sinr_norm is not None and comm_weight is not None and comm_weight > 0.0:
			inertia = float(np.sum(((1.0 - comm_weight) * (np.sqrt(_euclidean_dist_sq(points_scaled, centers)) / (dataset_max_dist + eps)) + comm_weight * (1.0 - sinr_norm[:, None]))[np.arange(n_samples), labels]))
		else:
			inertia = float(np.sum(_euclidean_dist_sq(points_scaled, centers)[np.arange(n_samples), labels]))

	# 将 centers 转换回输入坐标单位（除以 point_scale），以保持返回的 centers 与输入单位一致
	if point_scale != 1.0:
		centers = centers / point_scale
	return labels, centers, float(inertia)


if __name__ == '__main__':
	Users_num = 20
	pts = np.loadtxt('results/datas/Users_%d.txt' % Users_num)
	UAV_num = 2
	mode = ('origin','comm')[1]
	random_seed = int(np.random.uniform(0,100))
	if mode == 'origin':
		labels, centers, inertia = kmeans(pts, UAV_num, max_iters=1000,random_state=random_seed,comm_weight=0, point_scale=1)
		print("Labels:", labels)
		print("Centers:", centers)
		print("Inertia:", inertia)
		np.savetxt('results/datas/cluster/Users_%d_Clustered'% Users_num+'UAV_'+str(UAV_num)+'.txt', labels, fmt='%d')
	elif mode == 'comm':
		labels, centers, inertia = kmeans(pts, UAV_num, max_iters=1000,comm_weight=0.3, uav_height=0.1, random_state=random_seed, point_scale=0.1)
		print("Labels:", labels)
		print("Centers:", centers)
		print("Inertia:", inertia)
		np.savetxt('results/datas/cluster/Users_%d_Clustered_comm'% Users_num+'UAV_'+str(UAV_num)+'.txt', labels, fmt='%d')