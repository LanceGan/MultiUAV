"""序列算法路径可视化 — N=3, Proposed 4D 聚类。"""
import os, json, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDIR = os.path.join(ROOT, 'results/paper_experiments/routing')
CDIR = os.path.join(ROOT, 'results/paper_experiments/clustering')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 9

pts = np.loadtxt(os.path.join(ROOT, 'results/datas/Users_30.txt'))
labels = np.loadtxt(os.path.join(CDIR, 'labels_4d_uav3.txt'), dtype=int)
ini = np.array([14.76, 14.83])
end = np.array([27.62, 23.47])

ALGOS = ['TSP', 'GA', 'PSO', 'ACO', 'GA_EQTSP']
ALG_NAMES = ['TSP (NN+2-opt)', 'Standard GA', 'PSO', 'ACO', 'GA-EQTSP (Proposed)']
CLUSTER_COLORS = ['#4472C4', '#ED7D31', '#70AD47']
UAV_LABELS = ['UAV 1', 'UAV 2', 'UAV 3']

fig, axes = plt.subplots(2, len(ALGOS), figsize=(5*len(ALGOS), 10))

for col, (algo, name) in enumerate(zip(ALGOS, ALG_NAMES)):
    rf = os.path.join(RDIR, f'routing_{algo}_uav3.json')
    with open(rf) as f:
        rd = json.load(f)

    # --- Top row: cluster-colored points ---
    ax = axes[0, col]
    for cid in range(3):
        mask = labels == cid
        ax.scatter(pts[mask, 0], pts[mask, 1], c=CLUSTER_COLORS[cid],
                   s=60, edgecolors='white', linewidth=0.5,
                   label=UAV_LABELS[cid], alpha=0.85, zorder=3)

    # Draw path for each cluster
    for cid in range(3):
        ck = f'cluster_{cid}'
        mask = labels == cid
        cluster_pts = pts[mask]
        idxs = rd[ck]['best_indices']
        route_pts = np.vstack([np.append(ini, 0), cluster_pts, np.append(end, 0)])
        path_xy = route_pts[idxs, :2]
        ax.plot(path_xy[:, 0], path_xy[:, 1], '-', color=CLUSTER_COLORS[cid],
                linewidth=2, alpha=0.7, zorder=2)
        # arrows
        for i in range(0, len(path_xy) - 1, max(1, len(path_xy)//4)):
            ax.annotate('', xy=path_xy[i+1], xytext=path_xy[i],
                        arrowprops=dict(arrowstyle='->', color=CLUSTER_COLORS[cid],
                                        lw=1.2, alpha=0.6))

    ax.scatter(*ini, c='green', marker='s', s=100, edgecolors='black', linewidth=1, label='Start', zorder=5)
    ax.scatter(*end, c='red', marker='D', s=100, edgecolors='black', linewidth=1, label='End', zorder=5)
    ax.set_xlim(0, 40); ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (100 m)'); ax.set_ylabel('Y (100 m)')
    if col == 0: ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.15, linestyle='--')

    # --- Bottom row: cluster-colored paths only ---
    ax2 = axes[1, col]
    for cid in range(3):
        mask = labels == cid
        cluster_pts = pts[mask]
        idxs = rd[f'cluster_{cid}']['best_indices']
        route_pts = np.vstack([np.append(ini, 0), cluster_pts, np.append(end, 0)])
        path_xy = route_pts[idxs, :2]
        ax2.plot(path_xy[:, 0], path_xy[:, 1], '-o', color=CLUSTER_COLORS[cid],
                 linewidth=1.8, markersize=5, alpha=0.85, zorder=2,
                 label=UAV_LABELS[cid])
        # node index labels
        for i, (x_, y_) in enumerate(path_xy[1:-1]):
            ax2.text(x_, y_, str(i), fontsize=7, ha='center', va='center',
                     color='white', fontweight='bold', zorder=4)

    ax2.scatter(*ini, c='green', marker='s', s=100, edgecolors='black', linewidth=1, zorder=5)
    ax2.scatter(*end, c='red', marker='D', s=100, edgecolors='black', linewidth=1, zorder=5)
    ax2.set_xlim(0, 40); ax2.set_ylim(0, 40)
    ax2.set_aspect('equal')
    ax2.set_title(name, fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (100 m)'); ax2.set_ylabel('Y (100 m)')
    if col == 0: ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(alpha=0.15, linestyle='--')

fig.suptitle('Routing Algorithm Path Visualization (N=3, Proposed 4D Clustering)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RDIR, 'routing_paths.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(RDIR, 'routing_paths.png')}")
