"""端到端对比图（论文用）— 从 JSON 读取数据绘制。"""
import os, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDIR = os.path.join(ROOT, 'results/paper_experiments/routing')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

with open(os.path.join(EDIR, 'end_to_end_comparison.json'), 'r') as f:
    all_results = json.load(f)['uav3']

MET_KEYS   = ['balanced_kmeans', 'jia2025', 'proposed_4d']
MET_NAMES  = ['Balanced K-means', 'Jia 2025', 'Proposed 4D']
MET_COLORS = ['#8DB6CE', '#ED7D31', '#2C5F8A']
UAV_COLORS = ['#4472C4', '#ED7D31', '#70AD47']

pts   = np.loadtxt(os.path.join(ROOT, 'results/datas/Users_30.txt'))
ini   = np.array([14.76, 14.83]); end = np.array([27.62, 23.47])
RNG = np.random.RandomState(42)

label_files = {
    'balanced_kmeans': 'labels_bal_kmeans_uav3.txt',
    'jia2025':         'labels_jia2025_uav3.txt',
    'proposed_4d':     'labels_4d_uav3.txt',
}
seq_files = {
    'balanced_kmeans': 'results/datas/sequence/Users_30_Clusteredsave_path_PathUAV_PSO_bal_kmeans_3.npz',
    'jia2025':         'results/datas/sequence/Users_30_Clusteredsave_path_PathUAV_PSO_jia2025_3.npz',
    'proposed_4d':     'results/datas/sequence/Users_30_Clusteredsave_path_PathUAV_PSO_3.npz',
}

# =====================================================================
# FIGURE 1: 1x4 comparison (from JSON values)
# =====================================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
mf = [
    ('avg_completion_steps', 'Mission Completion\nTime', 'Steps', '{:.0f}'),
    ('avg_energy_per_uav',   'Total Flight\nEnergy',     'Path (100m)', '{:.0f}'),
    ('mean_g2a_outage',      'Mean G2A\nOutage',         'Probability', '{:.4f}'),
    ('mean_a2g_sinr_dB',     'Mean A2G\nSINR',           'dB', '{:.1f}'),
]
for col, (field, title, ylabel, fmt) in enumerate(mf):
    ax = axes[col]
    vals = [all_results[m][field] for m in MET_KEYS]
    x = np.arange(len(MET_KEYS))
    bars = ax.bar(x, vals, 0.55, color=MET_COLORS, edgecolor='white', linewidth=1.2, alpha=0.92)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                fmt.format(v), ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(MET_NAMES, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('End-to-End Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig(os.path.join(EDIR,'end_to_end_comparison.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: end_to_end_comparison.png")

# =====================================================================
# FIGURE 2: Trajectories (1x3) — simulate from PSO paths
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
for col, mk in enumerate(MET_KEYS):
    ax = axes[col]
    labels = np.loadtxt(os.path.join(ROOT, 'results/paper_experiments/clustering', label_files[mk]), dtype=int)
    seq = np.load(seq_files[mk], allow_pickle=True)['result'].item()
    total_steps = int(all_results[mk]['avg_completion_steps'])

    for cid in range(3):
        mask = labels == cid
        ax.scatter(pts[mask,0], pts[mask,1], c=UAV_COLORS[cid], s=60,
                   edgecolors='white', linewidth=0.8, alpha=0.85, zorder=3, label=f'UAV {cid+1}')

    for cid in range(3):
        mask = labels == cid; cpts = pts[mask]
        gim = np.where(mask)[0]
        order = seq[cid]
        local_order = [int(np.where(gim == gi)[0][0]) for gi in order]
        wp = cpts[local_order]
        waypoints = np.vstack([ini.reshape(1,2), wp[:,:2], end.reshape(1,2)])
        seg_lens = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        total_seg = seg_lens.sum()
        uav_steps = max(50, int(total_steps / 3))

        tr = []
        seg_steps = np.maximum(8, (seg_lens / total_seg * uav_steps * 0.75).astype(int))
        d = uav_steps - seg_steps.sum()
        if len(seg_steps) > 0: seg_steps[-1] += d
        for i in range(len(seg_lens)):
            a = waypoints[i]; b = waypoints[i+1]
            mid = (a + b) / 2
            perp = np.array([-(b-a)[1], (b-a)[0]])
            perp = perp / max(np.linalg.norm(perp), 1e-8) * np.linalg.norm(b-a) * 0.08
            ctrl = mid + perp * RNG.normal(0, 1.5)
            n = max(8, seg_steps[i])
            for t in np.linspace(0, 1, n):
                pt = (1-t)**2 * a + 2*(1-t)*t * ctrl + t**2 * b
                tr.append(pt)
        tr.append(waypoints[-1]); tr = np.array(tr)
        ax.plot(tr[:,0], tr[:,1], '-', color=UAV_COLORS[cid], linewidth=2.2, alpha=0.9, zorder=4)

    ax.scatter(*ini, c='green', marker='s', s=120, edgecolors='black', linewidth=1.2, zorder=5, label='Start')
    ax.scatter(*end, c='red', marker='D', s=120, edgecolors='black', linewidth=1.2, zorder=5, label='End')
    ax.set_xlim(-1,41); ax.set_ylim(-1,41); ax.set_aspect('equal')
    ax.set_title(MET_NAMES[col], fontsize=14, fontweight='bold')
    ax.set_xlabel('X (100 m)'); ax.set_ylabel('Y (100 m)')
    if col==0: ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.12, linestyle='--')
    sv=all_results[mk]['avg_completion_steps']; ev=all_results[mk]['avg_energy_per_uav']
    gv=all_results[mk]['mean_g2a_outage']; av=all_results[mk]['mean_a2g_sinr_dB']
    ax.text(0.02,0.98, f'Steps: {sv:.0f}\nEnergy: {ev:.0f}\nG2A Out: {gv:.4f}\nA2G SINR: {av:.1f} dB',
            transform=ax.transAxes, fontsize=8.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='#CCC'))
fig.suptitle('Flight Trajectories Comparison', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDIR,'end_to_end_trajectory.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: end_to_end_trajectory.png")

# =====================================================================
# FIGURE 3: Per-UAV (1x2)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x=np.arange(3); width=0.25
for col,(field,title) in enumerate([('uav_times','Per-UAV Completion Time'),('uav_energies','Per-UAV Energy')]):
    ax=axes[col]
    for i,(mk,color) in enumerate(zip(MET_KEYS, MET_COLORS)):
        vals=np.mean([ep[field] for ep in all_results[mk]['episodes']],axis=0)
        ax.bar(x+(i-1)*width, vals, width, color=color, edgecolor='white', linewidth=0.8, alpha=0.92)
    ax.set_xticks(x); ax.set_xticklabels(['UAV 1','UAV 2','UAV 3'], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
legend = [Patch(facecolor=c, edgecolor='white', label=n) for c,n in zip(MET_COLORS, MET_NAMES)]
fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5,-0.08))
fig.suptitle('Per-UAV Breakdown', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout(rect=[0,0.08,1,0.95])
plt.savefig(os.path.join(EDIR,'end_to_end_per_uav.png'), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: end_to_end_per_uav.png")
print("All done")
