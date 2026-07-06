"""序列算法对比图（论文用）— N=3，Proposed 4D 聚类，去除 SingleMapGA。"""
import os, json, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDIR = os.path.join(ROOT, 'results/paper_experiments/routing')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10

with open(os.path.join(RDIR, 'routing_comparison.json'), 'r') as f:
    data = json.load(f)

UAV_KEY = 'uav3'
ALG_KEYS = ['TSP', 'GA', 'PSO', 'ACO', 'GA_EQTSP']
ALG_NAMES = ['TSP', 'GA', 'PSO', 'ACO', 'GA-EQTSP\n(Proposed)']
ALG_COLORS = ['#B0B0B0', '#8DB6CE', '#70AD47', '#FFC000', '#2C5F8A']
ALG_MARKERS = ['s', 'D', 'o', '^', '*']  # for scatter plot
N_UAV = 3

d_algs = data[UAV_KEY]['algorithms']

# =====================================================================
# FIGURE 1: 三项指标综合对比 (1x3)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = [
    ('total_euclidean_length', 'Total Path Length',      '{:.0f}'),
    ('mean_g2a_outage',         'Mean G2A Outage',       '{:.4f}'),
    ('mean_a2g_sinr_dB',        'Mean A2G SINR (dB)',    '{:.1f}'),
]

for col, (field, title, fmt) in enumerate(metrics):
    ax = axes[col]
    vals = [d_algs[a][field] for a in ALG_KEYS]
    x = np.arange(len(ALG_KEYS))
    bars = ax.bar(x, vals, color=ALG_COLORS, edgecolor='white', linewidth=1.2, alpha=0.92)
    ymax = max(vals)
    for bar, v in zip(bars, vals):
        offset = max(ymax * 0.015, 0.0005)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                fmt.format(v), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('\n', ' ') for n in ALG_NAMES], fontsize=8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Routing Algorithm Comparison (N=3, Proposed 4D Clustering)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RDIR, 'routing_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: routing_comparison.png")

# =====================================================================
# FIGURE 2: Per-cluster breakdown (1x3)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cm = [
    ('euclidean_length',  'Path Length per UAV'),
    ('g2a_outage_mean',   'G2A Outage per UAV'),
    ('a2g_sinr_mean_dB',  'A2G SINR per UAV (dB)'),
]
x = np.arange(N_UAV)
width = 0.15

for col, (field, title) in enumerate(cm):
    ax = axes[col]
    for i, (alg, color) in enumerate(zip(ALG_KEYS, ALG_COLORS)):
        vals = [c[field] for c in d_algs[alg]['clusters']]
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, color=color, edgecolor='white', linewidth=0.5, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([f'UAV {j+1}' for j in range(N_UAV)], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

legend = [Patch(facecolor=c, edgecolor='white', label=n.replace('\n', ' '))
          for c, n in zip(ALG_COLORS, ALG_NAMES)]
fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle('Per-Cluster Breakdown (N=3, Proposed 4D Clustering)',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(os.path.join(RDIR, 'routing_comm_quality.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: routing_comm_quality.png")

# =====================================================================
# FIGURE 3: Path Length vs A2G SINR scatter (with jitter to avoid overlap)
# =====================================================================
fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

# Add small jitter to separate overlapping points
rng = np.random.RandomState(42)
for alg, color, name, marker in zip(ALG_KEYS, ALG_COLORS, ALG_NAMES, ALG_MARKERS):
    dd = d_algs[alg]
    jx = rng.uniform(-3, 3)
    jy = rng.uniform(-0.08, 0.08)
    ax.scatter(dd['total_euclidean_length'] + jx, dd['mean_a2g_sinr_dB'] + jy,
               s=180, c=color, marker=marker, edgecolors='black', linewidth=1.0,
               label=name.replace('\n', ' '), zorder=5)
    # Offset annotation
    label = name.replace('\n', ' ')
    offsets = {
        'TSP': (12, -6), 'GA': (-30, 12), 'PSO': (14, -10),
        'ACO': (10, 8), 'GA-EQTSP (Proposed)': (10, -8),
    }
    ox, oy = offsets.get(label, (8, 6))
    ax.annotate(label, (dd['total_euclidean_length'] + jx, dd['mean_a2g_sinr_dB'] + jy),
                textcoords="offset points", xytext=(ox, oy), fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=0.8, alpha=0.5))

ax.set_xlabel('Total Path Length', fontsize=12)
ax.set_ylabel('Mean A2G SINR (dB)', fontsize=12)
ax.set_title('Path Length vs Communication Quality Trade-off (N=3)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(fontsize=8, loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
plt.tight_layout()
plt.savefig(os.path.join(RDIR, 'routing_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: routing_summary.png")

print("\nAll figures saved to:", RDIR)
