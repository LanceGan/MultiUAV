"""Generate algorithm comparison trajectory figures.

This script creates side-by-side trajectory comparisons for
GA, PSO, ACO, GA_EQTSP, and MA-TD3 algorithms.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scenario_config import UAV_USER_MAP, INI_LOC, END_LOC


TRAJECTORY_COLORS = [
    "#08306B",
    "#67000D",
    "#3F007D",
    "#7F2704",
    "#00441B",
    "#084594",
]


def load_routing_sequence(algo_name, n_uav):
    """Load routing sequence from experiment results."""
    routing_path = os.path.join(
        _project_root, "results", "paper_experiments", "routing",
        f"routing_{algo_name}_uav{n_uav}.json"
    )
    if not os.path.exists(routing_path):
        return None

    with open(routing_path) as f:
        data = json.load(f)

    sequences = {}
    for key, val in data.items():
        if key.startswith("cluster_"):
            cid = int(key.split("_")[1])
            sequences[cid] = val["best_indices"]
    return sequences


def load_matd3_trajectory(n_uav, episode=0):
    """Load MA-TD3 trajectory from test results."""
    traj_path = os.path.join(
        _project_root, "results", "test", f"UAV_{n_uav}", "stable",
        f"test_results_uav{n_uav}.npz"
    )
    if not os.path.exists(traj_path):
        return None

    data = np.load(traj_path, allow_pickle=True)
    if "x_uav_all" in data and "y_uav_all" in data:
        x_uav = data["x_uav_all"][episode]
        y_uav = data["y_uav_all"][episode]
        complete_time = int(data["Complete_time"][episode])
        return {
            "x_uav": x_uav,
            "y_uav": y_uav,
            "steps": complete_time,
        }
    return None


def compute_trajectory_from_sequence(sequence, all_points, cluster_labels, cluster_id, ini_loc, end_loc):
    """Compute trajectory coordinates from a routing sequence.

    Args:
        sequence: list of indices from routing result (includes start=0 and end=num_nodes-1)
        all_points: all inspection points
        cluster_labels: cluster labels for all points
        cluster_id: current cluster ID
        ini_loc: start location
        end_loc: end location
    """
    # Get global indices for this cluster
    global_indices = np.where(cluster_labels == cluster_id)[0]

    # Build full path: start -> sequence (skip first and last) -> end
    full_path = [ini_loc]
    for idx in sequence[1:-1]:  # Skip start (0) and end (num_nodes-1)
        # idx is local index in the cluster (1-based because start is 0)
        local_idx = idx - 1  # Convert to 0-based index in cluster
        if 0 <= local_idx < len(global_indices):
            global_idx = global_indices[local_idx]
            full_path.append(all_points[global_idx])
    full_path.append(end_loc)
    return np.array(full_path)


def draw_radio_map(ax, radio_map):
    """Draw radio map background on axes."""
    if radio_map == "G2A":
        npzfile = np.load("results/datas/radiomap/Radio_datas.npz")
        value = 1 - npzfile["arr_0"]
        x_vec = npzfile["arr_2"]
        y_vec = npzfile["arr_3"]
        levels = np.linspace(0, 1.0, 11, endpoint=True)
        cmap = plt.get_cmap("viridis", len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        ax.contourf(
            np.array(x_vec) * 10,
            np.array(y_vec) * 10,
            value,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="max",
        )
        return True
    return False


def generate_comparison_figure(n_uav, output_dir):
    """Generate algorithm comparison figure for a given UAV count."""
    n_users = UAV_USER_MAP[n_uav]
    points = np.loadtxt(f"results/datas/Users_{n_users}.txt")
    ini_loc = np.array(INI_LOC + [0.0])
    end_loc = np.array(END_LOC + [0.0])

    # Load 4D clustering labels
    labels_path = os.path.join(
        _project_root, "results", "paper_experiments", "clustering",
        f"labels_4d_uav{n_uav}.txt"
    )
    if not os.path.exists(labels_path):
        print(f"[SKIP] Labels not found for N={n_uav}")
        return
    labels = np.loadtxt(labels_path, dtype=int)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    algorithms = ["GA", "PSO", "ACO", "GA_EQTSP"]
    algo_labels = ["GA", "PSO", "ACO", "DRM-GA (Ours)"]

    for idx, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        ax = axes[idx // 2, idx % 2]

        # Draw G2A radio map background
        draw_radio_map(ax, "G2A")

        # Load routing sequence
        sequences = load_routing_sequence(algo, n_uav)
        if sequences is None:
            ax.set_title(f"{label} (No Data)")
            continue

        # Plot each UAV's trajectory
        for cid in range(n_uav):
            if cid not in sequences:
                continue
            seq = sequences[cid]
            traj = compute_trajectory_from_sequence(seq, points, labels, cid, ini_loc, end_loc)

            color = TRAJECTORY_COLORS[cid % len(TRAJECTORY_COLORS)]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, label=f'UAV {cid+1}')
            ax.scatter(traj[1:-1, 0], traj[1:-1, 1], c=color, marker='^', s=40, edgecolors='white')

        # Plot start and end
        ax.scatter([ini_loc[0]], [ini_loc[1]], c='red', marker='o', s=80, label='Start', zorder=5)
        ax.scatter([end_loc[0]], [end_loc[1]], c='gold', marker='s', s=80, label='End', zorder=5)

        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 40)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper left')

    # Plot MA-TD3 trajectory
    ax = axes[1, 2]
    draw_radio_map(ax, "G2A")

    matd3_data = load_matd3_trajectory(n_uav)
    if matd3_data is not None:
        x_uav = matd3_data["x_uav"]
        y_uav = matd3_data["y_uav"]
        steps = matd3_data["steps"]

        for i in range(n_uav):
            color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
            ax.plot(x_uav[i][:steps+1], y_uav[i][:steps+1], '-', color=color, linewidth=2, label=f'UAV {i+1}')

        ax.scatter([ini_loc[0]], [ini_loc[1]], c='red', marker='o', s=80, label='Start', zorder=5)
        ax.scatter([end_loc[0]], [end_loc[1]], c='gold', marker='s', s=80, label='End', zorder=5)

    ax.set_title('MA-TD3 (Ours)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')

    plt.suptitle(f'Algorithm Comparison (N={n_uav})', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'trajectory_comparison_uav{n_uav}.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    output_dir = r'F:\Projects\Latex\MultiUAV\figs'

    for n_uav in [2, 3, 4]:
        print(f'\n--- Generating comparison for N={n_uav} ---')
        generate_comparison_figure(n_uav, output_dir)

    print('\nAll comparison figures generated.')
