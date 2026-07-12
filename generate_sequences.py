"""为 Balanced K-means / Jia2025 / Proposed 4D 三种聚类生成 PSO 序列 NPZ。"""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'sequence_algorithm'))

from PSO import PSO

UAV_CONFIGS = {3: 30}
INI_LOC = np.array([14.76, 14.83, 0.0])
END_LOC = np.array([27.62, 23.47, 0.0])

# Proposed 4D 已经有了 PSO 序列 (Users_30_...PSO_3.npz)，只生成前两个方法的
METHODS = [
    ('balanced_kmeans', 'labels_bal_kmeans_uav', 'PSO_bal_kmeans'),
    ('jia2025',         'labels_jia2025_uav',     'PSO_jia2025'),
]

CLUSTER_DIR = 'results/paper_experiments/clustering'
SEQ_DIR = 'results/datas/sequence'

for uav_num, user_num in UAV_CONFIGS.items():
    pts = np.loadtxt(f'results/datas/Users_{user_num}.txt')
    for method_key, label_prefix, algo_key in METHODS:
        label_file = os.path.join(CLUSTER_DIR, f'{label_prefix}{uav_num}.txt')
        labels = np.loadtxt(label_file, dtype=int)

        res_indices = {}
        for cid in sorted(np.unique(labels)):
            mask = labels == cid
            cluster_pts = pts[mask]
            full_data = np.vstack([INI_LOC, cluster_pts, END_LOC])
            num_nodes = full_data.shape[0]

            model = PSO(num_city=num_nodes, data=full_data.copy())
            _, _ = model.run()
            best_indices = model.best_path

            global_indices = np.where(mask)[0]
            orig_order = []
            for idx in best_indices:
                if idx != 0 and idx != num_nodes - 1:
                    orig_order.append(int(global_indices[idx - 1]))
            res_indices[int(cid)] = orig_order

        out_path = os.path.join(SEQ_DIR, f'Users_{user_num}_Clusteredsave_path_PathUAV_{algo_key}_{uav_num}.npz')
        np.savez(out_path, result=res_indices)
        print(f'{method_key} uav{uav_num}: {out_path}  |  sizes={[len(v) for v in res_indices.values()]}')

print("Done.")
