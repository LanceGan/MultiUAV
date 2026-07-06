"""聚类结果空间可视化（论文用）— N=3 场景，三种方法对比。"""
import os, numpy as np, matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, 'results/paper_experiments/clustering')
USER_FILE = os.path.join(PROJECT_ROOT, 'results/datas/Users_30.txt')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

pts = np.loadtxt(USER_FILE)[:, :2]

METHODS = [
    ('labels_bal_kmeans_uav3.txt', 'Balanced K-means'),
    ('labels_jia2025_uav3.txt',    'Jia 2025'),
    ('labels_4d_uav3.txt',         'Proposed 4D'),
]

CLUSTER_COLORS = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
MARKERS = ['o', 's', 'D', '^']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for col, (label_file, method_name) in enumerate(METHODS):
    ax = axes[col]
    labels = np.loadtxt(os.path.join(SAVE_DIR, label_file), dtype=int)

    for cid in np.unique(labels):
        mask = labels == cid
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   c=CLUSTER_COLORS[cid], marker=MARKERS[cid],
                   s=120, edgecolors='white', linewidth=0.8,
                   label=f'UAV {cid+1} ({mask.sum()} pts)',
                   alpha=0.9, zorder=3)

    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel('X (100 m)', fontsize=11)
    ax.set_ylabel('Y (100 m)', fontsize=11)
    ax.set_title(method_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)

fig.suptitle('Clustering Results Comparison (N=3, 30 Points)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'clustering_visualization.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(SAVE_DIR, 'clustering_visualization.png')}")
