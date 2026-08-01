"""序列算法端到端对比图 — 各算法真实best_indices + 直线连接。"""
import os, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIR = os.path.join(ROOT, 'results/paper_experiments/routing')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

with open(os.path.join(EDIR, 'routing_e2e_comparison.json'), 'r') as f:
    all_results = json.load(f)['uav3']

ALG_KEYS   = ['TSP', 'GA', 'PSO', 'ACO', 'GA_EQTSP']
ALG_NAMES  = ['TSP', 'GA', 'PSO', 'ACO', 'GA-EQTSP\n(Proposed)']
ALG_COLORS = ['#B0B0B0', '#8DB6CE', '#70AD47', '#FFC000', '#2C5F8A']
UAV_COLORS = ['#4472C4', '#ED7D31', '#70AD47']

pts   = np.loadtxt(os.path.join(ROOT, 'results/datas/Users_30.txt'))
ini   = np.array([14.76, 14.83]); end = np.array([27.62, 23.47])
labels = np.loadtxt(os.path.join(ROOT, 'results/paper_experiments/clustering', 'labels_4d_uav3.txt'), dtype=int)

# =====================================================================
# FIGURE 1: 1x4 comparison
# =====================================================================
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
mf = [
    ('avg_completion_steps', 'Mission Completion\nTime', 'Steps', '{:.0f}'),
    ('avg_energy',           'Total Flight\nEnergy',     'Path (100m)', '{:.0f}'),
    ('mean_g2a_outage',      'Mean G2A\nOutage',         'Probability', '{:.4f}'),
    ('mean_a2g_sinr_dB',     'Mean A2G\nSINR',           'dB', '{:.1f}'),
]
for col, (field, title, ylabel, fmt) in enumerate(mf):
    ax = axes[col]
    vals = [all_results[m][field] for m in ALG_KEYS]
    x = np.arange(len(ALG_KEYS))
    bars = ax.bar(x, vals, 0.55, color=ALG_COLORS, edgecolor='white', linewidth=1.2, alpha=0.92)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                fmt.format(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([n.replace('\n',' ') for n in ALG_NAMES], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Routing Algorithm Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig(os.path.join(EDIR,'routing_e2e_comparison.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: routing_e2e_comparison.png")

# =====================================================================
# FIGURE 2: Trajectories (1x5) — each algorithm's own best_indices, straight lines
# =====================================================================
fig, axes = plt.subplots(1, 5, figsize=(30, 6.2))

for col, algo in enumerate(ALG_KEYS):
    ax = axes[col]; rd = all_results[algo]
    rfile = os.path.join(ROOT, f'results/paper_experiments/routing/routing_{algo}_uav3.json')
    with open(rfile) as f: routing_data = json.load(f)

    # Cluster points
    for cid in range(3):
        mask = labels == cid
        ax.scatter(pts[mask,0], pts[mask,1], c=UAV_COLORS[cid], s=45,
                   edgecolors='white', linewidth=0.5, alpha=0.85, zorder=3, label=f'UAV {cid+1}')

    # Straight-line paths using algorithm's own best_indices
    for cid in range(3):
        mask = labels == cid; cpts_xy = pts[mask, :2]
        ck = f'cluster_{cid}'
        bi = routing_data[ck]['best_indices']  # each algorithm's unique visit order
        all_pts_2d = np.vstack([ini.reshape(1,2), cpts_xy, end.reshape(1,2)])
        path = all_pts_2d[bi, :]
        ax.plot(path[:,0], path[:,1], '-o', color=UAV_COLORS[cid], linewidth=2.0,
                markersize=5, alpha=0.88, zorder=4, markerfacecolor='white')

    ax.scatter(*ini, c='green', marker='s', s=100, edgecolors='black', linewidth=1, zorder=5, label='Start')
    ax.scatter(*end, c='red', marker='D', s=100, edgecolors='black', linewidth=1, zorder=5, label='End')
    ax.set_xlim(-1,41); ax.set_ylim(-1,41); ax.set_aspect('equal')
    ax.set_title(ALG_NAMES[col].replace('\n',' '), fontsize=11, fontweight='bold')
    ax.set_xlabel('X (100 m)'); ax.set_ylabel('Y (100 m)')
    if col==0: ax.legend(fontsize=6, loc='lower right')
    ax.grid(alpha=0.08, linestyle='--')
    sv=rd['avg_completion_steps']; ev=rd['avg_energy']
    gv=rd['mean_g2a_outage']; av=rd['mean_a2g_sinr_dB']
    ax.text(0.02,0.98, f'Steps: {sv:.0f}\nEnergy: {ev:.0f}\nG2A: {gv:.4f}\nA2G: {av:.1f} dB',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='#CCC'))

fig.suptitle('Flight Trajectories Comparison', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDIR,'routing_e2e_trajectory.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: routing_e2e_trajectory.png")

# =====================================================================
# FIGURE 3: Per-UAV (1x2)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x=np.arange(3); width=0.15
for col,(field,title) in enumerate([('uav_times','Per-UAV Completion Time'),('uav_energies','Per-UAV Energy')]):
    ax=axes[col]
    for i,(algo,color) in enumerate(zip(ALG_KEYS, ALG_COLORS)):
        vals=np.mean([ep[field] for ep in all_results[algo]['episodes']],axis=0)
        ax.bar(x+(i-2)*width, vals, width, color=color, edgecolor='white', linewidth=0.6, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels(['UAV 1','UAV 2','UAV 3'], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
legend = [Patch(facecolor=c, edgecolor='white', label=n.replace('\n',' ')) for c,n in zip(ALG_COLORS, ALG_NAMES)]
fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5,-0.09))
fig.suptitle('Per-UAV Breakdown', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout(rect=[0,0.09,1,0.95])
plt.savefig(os.path.join(EDIR,'routing_e2e_per_uav.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: routing_e2e_per_uav.png")
print("All done")
