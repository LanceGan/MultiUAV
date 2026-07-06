"""聚类评估对比图（论文用）— 仅展示 N=3 场景，全英文。"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, 'results/paper_experiments/clustering')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

with open(os.path.join(SAVE_DIR, 'cluster_metrics.json'), 'r') as f:
    data = json.load(f)

METHOD_KEYS   = ['balanced_kmeans', 'jia2025', 'proposed_4d']
METHOD_NAMES  = ['Balanced K-means', 'Jia 2025', 'Proposed 4D']
METHOD_COLORS = ['#7EA8C4', '#E8923F', '#2C5F8A']          # more distinct palette
METHOD_EDGE   = ['#4B7A96', '#C06D1F', '#1A3D5C']
METHOD_HATCH  = ['', '//', '']                              # Jia2025 has hatch for contrast
UAV_KEY  = 'uav3'
N_UAV    = 3

legend_elements = [Patch(facecolor=METHOD_COLORS[i], edgecolor=METHOD_EDGE[i],
                          label=METHOD_NAMES[i], linewidth=1.2)
                   for i in range(len(METHOD_KEYS))]

def extract(field):
    return [[c[field] for c in data[UAV_KEY]['methods'][mk]['clusters']] for mk in METHOD_KEYS]

uav_labels = ['UAV 1', 'UAV 2', 'UAV 3']
x = np.arange(N_UAV)
width = 0.25
offsets = [-width, 0, width]


# =====================================================================
# FIGURE 1: 总数据量 + 通信质量 (1x3 分组柱状图)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

panels = [
    ('total_data_MB',    'Total Data Volume (MB)',   '{:.0f}'),
    ('mean_g2a_outage',  'Mean G2A Outage',          '{:.3f}'),
    ('mean_a2g_sinr_dB', 'Mean A2G SINR (dB)',       '{:.1f}'),
]

for col, (field, title, fmt) in enumerate(panels):
    ax = axes[col]
    vals_all = extract(field)

    for i, (vals, offset) in enumerate(zip(vals_all, offsets)):
        bars = ax.bar(x + offset, vals, width,
                      color=METHOD_COLORS[i], edgecolor=METHOD_EDGE[i],
                      linewidth=1.2, alpha=0.92, hatch=METHOD_HATCH[i])

        # Annotate values above bars
        for bar, v in zip(bars, vals):
            y_offset = (max(vals_all[0]+vals_all[1]+vals_all[2]) - min(vals_all[0]+vals_all[1]+vals_all[2])) * 0.03
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    fmt.format(v), ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=METHOD_EDGE[i])

    ax.set_xticks(x)
    ax.set_xticklabels(uav_labels, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=10, frameon=True, fancybox=False, edgecolor='#CCCCCC',
           bbox_to_anchor=(0.5, -0.07))
fig.suptitle('Per-Cluster Metrics Comparison (N=3, 30 Inspection Points)',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, 'cluster_metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: cluster_metrics_comparison.png")


# =====================================================================
# FIGURE 2: 空间紧凑度 + 负载均衡 (1x2)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Left: Spatial Compactness
ax = axes[0]
vals_all = extract('avg_pairwise_dist')
for i, (vals, offset) in enumerate(zip(vals_all, offsets)):
    ax.bar(x + offset, vals, width, color=METHOD_COLORS[i],
           edgecolor=METHOD_EDGE[i], linewidth=1.2, alpha=0.92, hatch=METHOD_HATCH[i])
ax.set_xticks(x)
ax.set_xticklabels(uav_labels, fontsize=11)
ax.set_ylabel('Avg Pairwise Distance', fontsize=11)
ax.set_title('Spatial Compactness', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: Load Balance
ax = axes[1]
sizes_all = [data[UAV_KEY]['methods'][mk]['balance']['cluster_sizes'] for mk in METHOD_KEYS]
for i, (sz, offset) in enumerate(zip(sizes_all, offsets)):
    ax.bar(x + offset, sz, width, color=METHOD_COLORS[i],
           edgecolor=METHOD_EDGE[i], linewidth=1.2, alpha=0.92, hatch=METHOD_HATCH[i])
ax.set_xticks(x)
ax.set_xticklabels(uav_labels, fontsize=11)
ax.set_ylabel('Number of Points', fontsize=11)
ax.set_title('Load Balance', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=10, frameon=True, fancybox=False, edgecolor='#CCCCCC',
           bbox_to_anchor=(0.5, -0.09))
fig.suptitle('Spatial Compactness and Load Balance (N=3, 30 Points)',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout(rect=[0, 0.09, 1, 0.95])
plt.savefig(os.path.join(SAVE_DIR, 'cluster_balance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: cluster_balance_comparison.png")


# =====================================================================
# FIGURE 3: 簇间不均衡度 CV (1x1)
# =====================================================================
fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.2))

metric_pairs = [
    ('total_data_MB',      'Data Load'),
    ('mean_g2a_outage',    'G2A Outage'),
    ('mean_a2g_sinr_dB',   'A2G SINR'),
    ('avg_pairwise_dist',  'Spatial'),
]
xv = np.arange(len(metric_pairs))
cv_all = []
for mk in METHOD_KEYS:
    cv = []
    for field, _ in metric_pairs:
        vals = [c[field] for c in data[UAV_KEY]['methods'][mk]['clusters']]
        cv.append(np.std(vals) / max(abs(np.mean(vals)), 1e-8))
    cv_all.append(cv)

for i, (cv, offset) in enumerate(zip(cv_all, offsets)):
    bars = ax.bar(xv + offset, cv, width, color=METHOD_COLORS[i],
                  edgecolor=METHOD_EDGE[i], linewidth=1.2, alpha=0.92, hatch=METHOD_HATCH[i])
    for bar, v in zip(bars, cv):
        if v > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{v:.3f}', ha='center', fontsize=8, fontweight='bold',
                    color=METHOD_EDGE[i])

ax.set_xticks(xv)
ax.set_xticklabels([p[1] for p in metric_pairs], fontsize=11)
ax.set_ylabel('Coefficient of Variation (CV = std / |mean|)', fontsize=11)
ax.set_title('Inter-Cluster Imbalance', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=10, frameon=True, fancybox=False, edgecolor='#CCCCCC',
           bbox_to_anchor=(0.5, -0.11))
plt.tight_layout(rect=[0, 0.11, 1, 0.93])
plt.savefig(os.path.join(SAVE_DIR, 'cluster_imbalance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: cluster_imbalance_comparison.png")
print("\nAll figures saved to:", SAVE_DIR)
