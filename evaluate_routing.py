"""序列算法对比评估 — 密集采样路径沿途通信质量。"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import radio_map_G2A as G2A_rm
import radio_map_A2G as A2G_rm

UAV_CONFIGS = {2: 20, 3: 30, 4: 40}
UAV_HEIGHT = 0.1; POINT_SCALE = 0.1
ROUTING_DIR = 'results/paper_experiments/routing'
CLUSTER_DIR = 'results/paper_experiments/clustering'
OUTPUT = os.path.join(ROUTING_DIR, 'routing_comparison.json')
INI = np.array([14.76, 14.83]); END = np.array([27.62, 23.47])
ALGOS = ['TSP', 'GA', 'PSO', 'ACO', 'SingleMapGA', 'GA_EQTSP']
STEP_KM = 0.03  # sampling step along edges (km)

def sample_path(indices, route_pts, uav_h):
    """Densely sample points along path edges."""
    all_pts = []
    for k in range(len(indices) - 1):
        a = route_pts[indices[k], :2]
        b = route_pts[indices[k+1], :2]
        seg = np.linalg.norm(b - a)
        n = max(2, int(seg / STEP_KM))
        for t in np.linspace(0, 1, n):
            pt = a + t * (b - a)
            all_pts.append([pt[0] * POINT_SCALE, pt[1] * POINT_SCALE, uav_h])
    return np.array(all_pts)

print("=" * 60)
print("Routing Evaluation (dense path sampling)")
print("=" * 60)
results = {}
for uav_num, user_num in UAV_CONFIGS.items():
    key = f'uav{uav_num}'
    pts = np.loadtxt(f'results/datas/Users_{user_num}.txt')
    ini3, end3 = np.append(INI, 0.0), np.append(END, 0.0)
    labels = np.loadtxt(f'{CLUSTER_DIR}/labels_4d_uav{uav_num}.txt', dtype=int)
    results[key] = {'n_users': user_num, 'algorithms': {}}
    print(f"\n--- UAV={uav_num} ({user_num} pts) ---")
    for algo in ALGOS:
        rf = os.path.join(ROUTING_DIR, f'routing_{algo}_uav{uav_num}.json')
        if not os.path.exists(rf): continue
        with open(rf) as f: rd = json.load(f)
        clusters, tot_len, all_g2a, all_a2g, tot_time = [], 0.0, [], [], 0.0
        for cid in sorted(np.unique(labels)):
            ck = f'cluster_{cid}'
            if ck not in rd: continue
            mask = labels == cid
            rp = np.vstack([ini3, pts[mask], end3])
            idxs = rd[ck]['best_indices']
            samples = sample_path(idxs, rp, UAV_HEIGHT)
            # Query radio maps
            g2a_out, _ = G2A_rm.getPointMiniOutage(samples)
            _, a2g_snr = A2G_rm.getPointMiniOutage(samples)
            g2a_out = np.array(g2a_out); a2g_snr = np.array(a2g_snr)
            eucl = float(np.sum(np.linalg.norm(rp[idxs[1:], :2] - rp[idxs[:-1], :2], axis=1)))
            tot_len += eucl; tot_time += rd[ck].get('time_s', 0)
            all_g2a.extend(g2a_out.tolist()); all_a2g.extend(a2g_snr.tolist())
            clusters.append({
                'cluster_id': int(cid), 'n_nodes': len(idxs)-2,
                'euclidean_length': eucl,
                'g2a_outage_mean': float(np.mean(g2a_out)),
                'a2g_sinr_mean_dB': float(np.mean(a2g_snr)),
                'time_s': rd[ck].get('time_s', 0),
            })
        all_g2a = np.array(all_g2a); all_a2g = np.array(all_a2g)
        results[key]['algorithms'][algo] = {
            'clusters': clusters,
            'total_euclidean_length': float(tot_len),
            'mean_g2a_outage': float(np.mean(all_g2a)),
            'mean_a2g_sinr_dB': float(np.mean(all_a2g)),
            'total_time_s': float(tot_time),
        }
        print(f"  {algo:15s} len={tot_len:7.1f} | G2A={np.mean(all_g2a):.4f} | A2G={np.mean(all_a2g):.1f} dB | t={tot_time:.0f}s")
with open(OUTPUT, 'w') as f: json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT}")
