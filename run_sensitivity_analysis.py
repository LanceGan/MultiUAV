"""Sensitivity analysis experiments for paper.

Usage:
    python run_sensitivity_analysis.py --analysis ga_population
    python run_sensitivity_analysis.py --analysis data_volume
    python run_sensitivity_analysis.py --analysis all
"""
import sys
import os
import json
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scenario_config import UAV_USER_MAP, INI_LOC, END_LOC
from baselines import run_ga_eqtsp


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_points(n_users: int) -> np.ndarray:
    """Load inspection-point coordinates from the data file."""
    path = os.path.join(_project_root, f"results/datas/Users_{n_users}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.loadtxt(path)


def load_cluster_labels(n_users: int, n_uav: int, method: str = "4d") -> np.ndarray:
    """Load saved clustering labels from file."""
    if method == "4d":
        fname = f"labels_4d_uav{n_uav}.txt"
    else:
        fname = f"labels_naive_uav{n_uav}.txt"
    path = os.path.join(
        _project_root, "results", "paper_experiments", "clustering", fname
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")
    return np.loadtxt(path, dtype=int)


def split_clusters(points: np.ndarray, labels: np.ndarray) -> dict:
    """Split points into a dict keyed by cluster id."""
    clustered = {}
    for cid in np.unique(labels):
        clustered[int(cid)] = points[labels == cid]
    return clustered


def get_ini_end_3d():
    """Return INI_LOC and END_LOC extended to 3D with z=0.0."""
    ini = np.array(INI_LOC + [0.0], dtype=np.float64)
    end = np.array(END_LOC + [0.0], dtype=np.float64)
    return ini, end


# ---------------------------------------------------------------------------
#  Experiment 1: GA Population Sensitivity
# ---------------------------------------------------------------------------

def run_ga_population_sensitivity():
    """Test GA_EQTSP sensitivity to population size with N=3 UAVs, data_size=300."""
    print("=" * 60)
    print("  GA POPULATION SENSITIVITY ANALYSIS")
    print("=" * 60)

    out_dir = os.path.join(_project_root, "results", "sensitivity", "ga_population")
    ensure_dir(out_dir)

    n_uav = 3
    n_users = UAV_USER_MAP[n_uav]
    population_sizes = [25, 50, 100, 150, 200]

    points = load_points(n_users)
    labels = load_cluster_labels(n_users, n_uav, method="4d")
    clusters = split_clusters(points, labels)
    ini_loc, end_loc = get_ini_end_3d()

    results = {"n_uav": n_uav, "n_users": n_users, "population_sizes": {}}

    for pop in population_sizes:
        print(f"\n--- Population size = {pop} ---")
        pop_results = {}
        total_length = 0.0
        total_time = 0.0

        for cid in sorted(clusters.keys()):
            cpts = clusters[cid]
            print(f"  Cluster {cid} ({cpts.shape[0]} points) ... ", end="", flush=True)
            t0 = time.time()
            best_coords, best_length, best_indices = run_ga_eqtsp(
                cpts, ini_loc, end_loc, num_total=pop, iteration=200, data_size=300,
            )
            elapsed = time.time() - t0
            total_length += best_length
            total_time += elapsed
            pop_results[f"cluster_{cid}"] = {
                "path_length": round(float(best_length), 6),
                "time_s": round(elapsed, 2),
            }
            print(f"length={best_length:.4f}  time={elapsed:.1f}s")

        pop_results["total_path_length"] = round(total_length, 6)
        pop_results["total_time_s"] = round(total_time, 2)
        results["population_sizes"][str(pop)] = pop_results

    # Save summary
    summary_path = os.path.join(out_dir, "ga_population_sensitivity.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGA population sensitivity results saved to {summary_path}")
    return results


# ---------------------------------------------------------------------------
#  Experiment 2: Data Volume Sensitivity
# ---------------------------------------------------------------------------

def run_data_volume_sensitivity():
    """Test GA_EQTSP sensitivity to data volume with N=3 UAVs, pop=25."""
    print("=" * 60)
    print("  DATA VOLUME SENSITIVITY ANALYSIS")
    print("=" * 60)

    out_dir = os.path.join(_project_root, "results", "sensitivity", "data_volume")
    ensure_dir(out_dir)

    n_uav = 3
    n_users = UAV_USER_MAP[n_uav]
    data_sizes = [50, 100, 150, 200, 250, 300, 350, 400]

    points = load_points(n_users)
    labels = load_cluster_labels(n_users, n_uav, method="4d")
    clusters = split_clusters(points, labels)
    ini_loc, end_loc = get_ini_end_3d()

    results = {"n_uav": n_uav, "n_users": n_users, "data_sizes": {}}

    for ds in data_sizes:
        print(f"\n--- Data size = {ds} MB ---")
        ds_results = {}
        total_length = 0.0
        total_time = 0.0

        for cid in sorted(clusters.keys()):
            cpts = clusters[cid]
            print(f"  Cluster {cid} ({cpts.shape[0]} points) ... ", end="", flush=True)
            t0 = time.time()
            best_coords, best_length, best_indices = run_ga_eqtsp(
                cpts, ini_loc, end_loc, num_total=25, iteration=200, data_size=ds,
            )
            elapsed = time.time() - t0
            total_length += best_length
            total_time += elapsed
            ds_results[f"cluster_{cid}"] = {
                "path_length": round(float(best_length), 6),
                "time_s": round(elapsed, 2),
            }
            print(f"length={best_length:.4f}  time={elapsed:.1f}s")

        ds_results["total_path_length"] = round(total_length, 6)
        ds_results["total_time_s"] = round(total_time, 2)
        results["data_sizes"][str(ds)] = ds_results

    # Save summary
    summary_path = os.path.join(out_dir, "data_volume_sensitivity.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nData volume sensitivity results saved to {summary_path}")
    return results


# ---------------------------------------------------------------------------
#  Figure Generation
# ---------------------------------------------------------------------------

def generate_sensitivity_figures():
    """Generate dual-axis plots from saved sensitivity analysis results."""
    print("=" * 60)
    print("  GENERATING SENSITIVITY FIGURES")
    print("=" * 60)

    fig_dir = os.path.join(_project_root, "results", "sensitivity", "figures")
    ensure_dir(fig_dir)

    # --- Figure 1: GA Population Sensitivity ---
    pop_path = os.path.join(
        _project_root, "results", "sensitivity", "ga_population",
        "ga_population_sensitivity.json",
    )
    if os.path.isfile(pop_path):
        with open(pop_path, "r", encoding="utf-8") as f:
            pop_data = json.load(f)

        pop_sizes = []
        path_lengths = []
        run_times = []
        for pop_str in sorted(pop_data["population_sizes"].keys(), key=int):
            pop_sizes.append(int(pop_str))
            entry = pop_data["population_sizes"][pop_str]
            path_lengths.append(entry["total_path_length"])
            run_times.append(entry["total_time_s"])

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#1f77b4"
        color2 = "#d62728"

        ax1.set_xlabel("GA Population Size", fontsize=13)
        ax1.set_ylabel("Total Path Length", fontsize=13, color=color1)
        ax1.plot(pop_sizes, path_lengths, "o-", color=color1, linewidth=2,
                 markersize=8, label="Path Length")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Computation Time (s)", fontsize=13, color=color2)
        ax2.plot(pop_sizes, run_times, "s--", color=color2, linewidth=2,
                 markersize=8, label="Comp. Time")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

        plt.title("Sensitivity to GA Population Size (N=3, Data=300 MB)", fontsize=14)
        fig.tight_layout()
        save_path = os.path.join(fig_dir, "ga_population_sensitivity.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        print(f"  [SKIP] Population sensitivity data not found at {pop_path}")

    # --- Figure 2: Data Volume Sensitivity ---
    vol_path = os.path.join(
        _project_root, "results", "sensitivity", "data_volume",
        "data_volume_sensitivity.json",
    )
    if os.path.isfile(vol_path):
        with open(vol_path, "r", encoding="utf-8") as f:
            vol_data = json.load(f)

        data_sizes = []
        path_lengths = []
        run_times = []
        for ds_str in sorted(vol_data["data_sizes"].keys(), key=int):
            data_sizes.append(int(ds_str))
            entry = vol_data["data_sizes"][ds_str]
            path_lengths.append(entry["total_path_length"])
            run_times.append(entry["total_time_s"])

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "#1f77b4"
        color2 = "#d62728"

        ax1.set_xlabel("Data Volume (MB)", fontsize=13)
        ax1.set_ylabel("Total Path Length", fontsize=13, color=color1)
        ax1.plot(data_sizes, path_lengths, "o-", color=color1, linewidth=2,
                 markersize=8, label="Path Length")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Computation Time (s)", fontsize=13, color=color2)
        ax2.plot(data_sizes, run_times, "s--", color=color2, linewidth=2,
                 markersize=8, label="Comp. Time")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

        plt.title("Sensitivity to Data Volume (N=3, Pop=25)", fontsize=14)
        fig.tight_layout()
        save_path = os.path.join(fig_dir, "data_volume_sensitivity.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        print(f"  [SKIP] Data volume sensitivity data not found at {vol_path}")

    print("Figure generation complete.")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

ANALYSIS_MAP = {
    "ga_population": run_ga_population_sensitivity,
    "data_volume": run_data_volume_sensitivity,
    "generate_figures": generate_sensitivity_figures,
}


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis experiments")
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        choices=list(ANALYSIS_MAP.keys()) + ["all"],
        help="Which analysis to run (default: all)",
    )
    args = parser.parse_args()

    if args.analysis == "all":
        for name, func in ANALYSIS_MAP.items():
            func()
    else:
        ANALYSIS_MAP[args.analysis]()


if __name__ == "__main__":
    main()
