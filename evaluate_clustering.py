"""评估不同聚类算法的 per-cluster 指标。

指标：
  1. 每个 cluster 的总数据量（每点 200-300MB 随机值）
  2. 每个 cluster 的平均 G2A outage probability 和 A2G SNR
  3. 每个 cluster 的平均成对距离（空间紧凑度）
"""
import sys
import os
import json
import time
import numpy as np

# Ensure imports work
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import radio_map_G2A as G2A_rm
import radio_map_A2G as A2G_rm
from Clustering import kmeans_4d
from baselines import (
    naive_kmeans_clustering,
    balanced_naive_kmeans,
    jia2025_balanced_clustering,
)

# ── Config ──────────────────────────────────────────────────
UAV_CONFIGS = {2: 20, 3: 30, 4: 40}
DATA_VOL_RANGE = (200, 300)  # MB per point
RANDOM_SEED = 42
UAV_HEIGHT = 0.1  # km (100m), used for radio map queries
POINT_SCALE = 0.1  # coordinate scaling factor
WEIGHTS_4D = (1.0, 0.4, 0.4, 0.5)
OUTPUT_PATH = 'results/paper_experiments/clustering/cluster_metrics.json'

rng = np.random.RandomState(RANDOM_SEED)

# ── Helpers ─────────────────────────────────────────────────

def get_communication_features(points_2d, uav_height):
    """Query radio maps for all points, return per-point G2A outage and A2G SNR."""
    n = len(points_2d)
    loc_vec = np.hstack([points_2d * POINT_SCALE, np.full((n, 1), uav_height)])

    # G2A: outage probability (lower = better)
    g2a_outage, g2a_sinr = G2A_rm.getPointMiniOutage(loc_vec)

    # A2G: max SINR in dB (higher = better)
    a2g_outage, a2g_sinr = A2G_rm.getPointMiniOutage(loc_vec)

    return np.array(g2a_outage), np.array(a2g_sinr)


def average_pairwise_distance(points_2d):
    """簇内所有点对之间的平均欧氏距离。"""
    if len(points_2d) <= 1:
        return 0.0
    diff = points_2d[:, None, :] - points_2d[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # 只取上三角（不含对角线）
    iu = np.triu_indices(len(points_2d), k=1)
    return float(np.mean(dist[iu]))


def compute_cluster_metrics(points_2d, labels, data_volumes,
                            g2a_outages, a2g_sinrs, k):
    """对聚类结果计算 per-cluster 指标。"""
    # Normalize A2G SINR to [0, 1] for composite cost
    a2g_min, a2g_max = np.min(a2g_sinrs), np.max(a2g_sinrs)
    a2g_norm = (a2g_sinrs - a2g_min) / max(a2g_max - a2g_min, 1e-8)

    clusters = []
    for cid in range(k):
        mask = labels == cid
        pts = points_2d[mask]
        n_pts = len(pts)
        g2a_c = g2a_outages[mask]
        a2g_c = a2g_sinrs[mask]
        a2g_norm_c = a2g_norm[mask]

        # Composite communication cost (lower = better)
        # cost = 0.5 * G2A_outage + 0.5 * (1 - A2G_rate_normalized)
        composite_cost = 0.5 * np.mean(g2a_c) + 0.5 * (1.0 - np.mean(a2g_norm_c))

        clusters.append({
            'cluster_id': int(cid),
            'n_points': n_pts,
            'total_data_MB': float(np.sum(data_volumes[mask])),
            'mean_data_MB': float(np.mean(data_volumes[mask])) if n_pts > 0 else 0.0,
            'mean_g2a_outage': float(np.mean(g2a_c)),
            'worst_g2a_outage': float(np.max(g2a_c)),
            'n_dangerous_g2a': int(np.sum(g2a_c > 0.5)),
            'best_g2a_outage': float(np.min(g2a_c)),
            'mean_a2g_sinr_dB': float(np.mean(a2g_c)),
            'worst_a2g_sinr_dB': float(np.min(a2g_c)),
            'best_a2g_sinr_dB': float(np.max(a2g_c)),
            'composite_comm_cost': float(composite_cost),
            'avg_pairwise_dist': average_pairwise_distance(pts),
        })
    return clusters

def compute_comm_balance(clusters):
    """簇间通信均衡性指标。"""
    g2a_means = [c['mean_g2a_outage'] for c in clusters]
    g2a_worsts = [c['worst_g2a_outage'] for c in clusters]
    a2g_means = [c['mean_a2g_sinr_dB'] for c in clusters]
    cost_vals = [c['composite_comm_cost'] for c in clusters]
    dangerous = [c['n_dangerous_g2a'] for c in clusters]

    return {
        'g2a_mean_max_min_ratio': max(g2a_means) / min(g2a_means) if min(g2a_means) > 0 else float('inf'),
        'g2a_worst_max_min_ratio': max(g2a_worsts) / min(g2a_worsts) if min(g2a_worsts) > 0 else float('inf'),
        'g2a_mean_std': float(np.std(g2a_means)),
        'a2g_mean_std_dB': float(np.std(a2g_means)),
        'composite_cost_std': float(np.std(cost_vals)),
        'total_dangerous_points': int(sum(dangerous)),
        'per_cluster_dangerous': dangerous,
    }


def compute_balance_metrics(labels, k):
    """计算簇间均衡性指标。"""
    counts = [int(np.sum(labels == i)) for i in range(k)]
    return {
        'cluster_sizes': counts,
        'std_dev': float(np.std(counts)),
        'variance': float(np.var(counts)),
        'max_min_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
    }


# ── Main ────────────────────────────────────────────────────

print("=" * 70)
print("Clustering Metrics Evaluation")
print("=" * 70)
print(f"Data volume range: {DATA_VOL_RANGE[0]}-{DATA_VOL_RANGE[1]} MB")
print(f"Random seed: {RANDOM_SEED}")
print(f"4D weights: {WEIGHTS_4D}")
print()

results = {}

for uav_num, user_num in UAV_CONFIGS.items():
    key = f'uav{uav_num}'
    print(f"\n{'='*70}")
    print(f"UAV = {uav_num}, Users = {user_num}")
    print(f"{'='*70}")

    # Load coordinates
    user_file = f'results/datas/Users_{user_num}.txt'
    pts = np.loadtxt(user_file)
    pts_2d = pts[:, :2]

    # Fixed random data volumes (200-300 MB)
    data_volumes = rng.uniform(*DATA_VOL_RANGE, size=user_num)

    # Query radio maps
    g2a_outage, a2g_sinr = get_communication_features(pts_2d, UAV_HEIGHT)
    print(f"G2A outage range: [{g2a_outage.min():.4f}, {g2a_outage.max():.4f}]")
    print(f"A2G SINR  range: [{a2g_sinr.min():.1f}, {a2g_sinr.max():.1f}] dB")
    print(f"Data volumes: min={data_volumes.min():.1f}, max={data_volumes.max():.1f}, sum={data_volumes.sum():.1f} MB")

    results[key] = {
        'n_users': user_num,
        'data_volume_range_MB': list(DATA_VOL_RANGE),
        'total_data_MB': float(data_volumes.sum()),
        'global_g2a_outage_mean': float(np.mean(g2a_outage)),
        'global_a2g_sinr_mean_dB': float(np.mean(a2g_sinr)),
        'methods': {},
    }

    k = uav_num
    methods = {}

    # --- 1) Naive K-means ---
    print("\n[1/4] Naive K-means (spatial only)...")
    t0 = time.time()
    labels_naive, _ = naive_kmeans_clustering(pts_2d, k, random_state=RANDOM_SEED)
    np.savetxt(f'results/paper_experiments/clustering/labels_naive_uav{uav_num}.txt', labels_naive, fmt='%d')
    elapsed = time.time() - t0
    clusters = compute_cluster_metrics(pts_2d, labels_naive, data_volumes,
                                       g2a_outage, a2g_sinr, k)
    balance = compute_balance_metrics(labels_naive, k)
    comm_bal = compute_comm_balance(clusters)
    methods['naive_kmeans'] = {
        'time_s': round(elapsed, 4),
        'balance': balance,
        'comm_balance': comm_bal,
        'clusters': clusters,
    }
    print(f"  time={elapsed:.3f}s | sizes={balance['cluster_sizes']}")

    # --- 2) Balanced K-means ---
    print("[2/4] Balanced Naive K-means (spatial only)...")
    t0 = time.time()
    labels_bal, _ = balanced_naive_kmeans(pts_2d, k, random_state=RANDOM_SEED)
    np.savetxt(f'results/paper_experiments/clustering/labels_bal_kmeans_uav{uav_num}.txt', labels_bal, fmt='%d')
    elapsed = time.time() - t0
    clusters = compute_cluster_metrics(pts_2d, labels_bal, data_volumes,
                                       g2a_outage, a2g_sinr, k)
    balance = compute_balance_metrics(labels_bal, k)
    comm_bal = compute_comm_balance(clusters)
    methods['balanced_kmeans'] = {
        'time_s': round(elapsed, 4),
        'balance': balance,
        'comm_balance': comm_bal,
        'clusters': clusters,
    }
    print(f"  time={elapsed:.3f}s | sizes={balance['cluster_sizes']}")

    # --- 3) Jia2025 ---
    print("[3/4] Jia2025 balanced clustering (spatial + workload)...")
    t0 = time.time()
    labels_jia, _ = jia2025_balanced_clustering(
        pts_2d, data_volumes, k, random_state=RANDOM_SEED,
    )
    np.savetxt(f'results/paper_experiments/clustering/labels_jia2025_uav{uav_num}.txt', labels_jia, fmt='%d')
    elapsed = time.time() - t0
    clusters = compute_cluster_metrics(pts_2d, labels_jia, data_volumes,
                                       g2a_outage, a2g_sinr, k)
    balance = compute_balance_metrics(labels_jia, k)
    comm_bal = compute_comm_balance(clusters)
    methods['jia2025'] = {
        'time_s': round(elapsed, 4),
        'balance': balance,
        'comm_balance': comm_bal,
        'clusters': clusters,
    }
    print(f"  time={elapsed:.3f}s | sizes={balance['cluster_sizes']}")

    # --- 4) Proposed 4D ---
    print("[4/4] Proposed 4D clustering (spatial + G2A + A2G + volume)...")
    t0 = time.time()
    labels_4d, centers_4d, inertia = kmeans_4d(
        pts_2d, k,
        offload_volumes=data_volumes,
        weights=WEIGHTS_4D,
        max_iters=1000,
        uav_height=UAV_HEIGHT,
        point_scale=POINT_SCALE,
        balanced=True,
        random_state=RANDOM_SEED,
    )
    np.savetxt(f'results/paper_experiments/clustering/labels_4d_uav{uav_num}.txt', labels_4d, fmt='%d')
    elapsed = time.time() - t0
    clusters = compute_cluster_metrics(pts_2d, labels_4d, data_volumes,
                                       g2a_outage, a2g_sinr, k)
    balance = compute_balance_metrics(labels_4d, k)
    comm_bal = compute_comm_balance(clusters)
    methods['proposed_4d'] = {
        'time_s': round(elapsed, 4),
        'inertia': float(inertia),
        'balance': balance,
        'comm_balance': comm_bal,
        'clusters': clusters,
    }
    print(f"  time={elapsed:.3f}s | inertia={inertia:.4f} | sizes={balance['cluster_sizes']}")

    results[key]['methods'] = methods


# ── Save ────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print(f"Results saved to: {OUTPUT_PATH}")
print(f"{'='*70}")

# ── Print Summary Table ─────────────────────────────────────

print("\n\n=== PER-CLUSTER METRICS SUMMARY ===\n")
for uav_key in ['uav2', 'uav3', 'uav4']:
    data = results[uav_key]
    print(f"\n{'─'*80}")
    print(f"  {uav_key.upper()} ({data['n_users']} points, total data = {data['total_data_MB']:.0f} MB)")
    print(f"  Global avg G2A outage = {data['global_g2a_outage_mean']:.4f}")
    print(f"  Global avg A2G SNR    = {data['global_a2g_sinr_mean_dB']:.1f} dB")
    print(f"{'─'*80}")

    header = f"{'Method':<20} {'Clust':>6} {'TotalData':>10} {'G2A_Out':>9} {'A2G_SNR':>9} {'PairDist':>9}"
    print(header)
    print('-' * len(header))

    for method_name, method_data in data['methods'].items():
        for c in method_data['clusters']:
            print(f"{method_name:<20} {c['cluster_id']:>5}  "
                  f"{c['total_data_MB']:>9.0f}  "
                  f"{c['mean_g2a_outage']:>8.4f}  "
                  f"{c['mean_a2g_sinr_dB']:>8.1f}  "
                  f"{c['avg_pairwise_dist']:>8.2f}")

    # Balance summary
    print(f"\n  Load balance:")
    for method_name, method_data in data['methods'].items():
        b = method_data['balance']
        print(f"    {method_name:<18} sizes={b['cluster_sizes']}  "
              f"std={b['std_dev']:.2f}  max/min={b['max_min_ratio']:.2f}")

print("\nDone.")
