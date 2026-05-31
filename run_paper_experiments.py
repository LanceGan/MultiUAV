"""Master script to run all paper experiments.

Usage:
    python run_paper_experiments.py --experiment all
    python run_paper_experiments.py --experiment clustering
    python run_paper_experiments.py --experiment routing
    python run_paper_experiments.py --experiment multi_data
"""
import sys
import os
import json
import argparse
import time

import numpy as np

# Ensure project root is on the path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scenario_config import UAV_USER_MAP, INI_LOC, END_LOC
from baselines import balanced_naive_kmeans, run_ga_eqtsp
from Clustering import kmeans_4d


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
    """Load saved clustering labels from file.

    Args:
        method: '4d' for kmeans_4d labels, 'naive' for balanced_naive_kmeans labels.
    """
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


def compute_balance_metrics(labels: np.ndarray, n_clusters: int) -> dict:
    """Compute load-balance metrics for a set of cluster labels."""
    counts = np.array([np.sum(labels == c) for c in range(n_clusters)], dtype=float)
    return {
        "cluster_sizes": counts.tolist(),
        "std_dev": float(np.std(counts)),
        "variance": float(np.var(counts)),
        "max_min_ratio": float(np.max(counts) / max(np.min(counts), 1.0)),
    }


def get_ini_end_3d():
    """Return INI_LOC and END_LOC extended to 3D with z=0.0."""
    ini = np.array(INI_LOC + [0.0], dtype=np.float64)
    end = np.array(END_LOC + [0.0], dtype=np.float64)
    return ini, end


# ---------------------------------------------------------------------------
#  Experiment 1: Clustering
# ---------------------------------------------------------------------------

def run_clustering_experiment():
    """Compare balanced_naive_kmeans vs kmeans_4d across UAV counts."""
    print("=" * 60)
    print("  CLUSTERING EXPERIMENT")
    print("=" * 60)

    out_dir = os.path.join(_project_root, "results", "paper_experiments", "clustering")
    ensure_dir(out_dir)

    results = {}

    for n_uav in [2, 3, 4]:
        n_users = UAV_USER_MAP[n_uav]
        print(f"\n--- UAV={n_uav}, Users={n_users} ---")

        points = load_points(n_users)
        print(f"  Loaded {points.shape[0]} points from Users_{n_users}.txt")

        # --- Naive balanced K-means (spatial only) ---
        t0 = time.time()
        labels_naive, centers_naive = balanced_naive_kmeans(points, n_uav)
        t_naive = time.time() - t0
        metrics_naive = compute_balance_metrics(labels_naive, n_uav)
        print(f"  balanced_naive_kmeans: {t_naive:.2f}s  {metrics_naive}")

        # --- 4D K-means (spatial + comm + volume) ---
        t0 = time.time()
        labels_4d, centers_4d, inertia_4d = kmeans_4d(
            points, n_uav, weights=(1.0, 0.4, 0.4, 0.5),
            max_iters=1000, random_state=42, point_scale=0.1,
        )
        t_4d = time.time() - t0
        metrics_4d = compute_balance_metrics(labels_4d, n_uav)
        print(f"  kmeans_4d:            {t_4d:.2f}s  {metrics_4d}")

        # Collect results
        results[f"uav{n_uav}"] = {
            "n_users": n_users,
            "naive": {
                "time_s": round(t_naive, 3),
                "inertia": None,
                **metrics_naive,
            },
            "4d": {
                "time_s": round(t_4d, 3),
                "inertia": round(float(inertia_4d), 6),
                **metrics_4d,
            },
        }

        # Save labels
        np.savetxt(
            os.path.join(out_dir, f"labels_naive_uav{n_uav}.txt"),
            labels_naive, fmt="%d",
        )
        np.savetxt(
            os.path.join(out_dir, f"labels_4d_uav{n_uav}.txt"),
            labels_4d, fmt="%d",
        )

    # Save summary JSON
    summary_path = os.path.join(out_dir, "clustering_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nClustering results saved to {summary_path}")


# ---------------------------------------------------------------------------
#  Experiment 2: Routing
# ---------------------------------------------------------------------------

def _run_routing_algo(algo_name: str, cluster_points: np.ndarray,
                      ini_loc: np.ndarray, end_loc: np.ndarray) -> dict:
    """Run a single routing algorithm on one cluster and return metrics."""
    full_data = np.vstack([ini_loc, cluster_points, end_loc])
    num_city = full_data.shape[0]

    if algo_name == "GA":
        from sequence_algorithm.GA import GA
        model = GA(num_city=num_city, num_total=100, iteration=300,
                   data=full_data.copy())
        best_coords, best_length, best_indices = model.run()

    elif algo_name == "PSO":
        from sequence_algorithm.PSO import PSO
        model = PSO(num_city=num_city, data=full_data.copy())
        best_coords, best_length = model.run()
        best_indices = model.best_path

    elif algo_name == "ACO":
        from sequence_algorithm.ACO import ACO
        model = ACO(num_city=num_city, data=full_data.copy(),
                    start_node=0, end_node=num_city - 1)
        # ACO.run() does not expose best_path; call aco() directly
        best_length, best_indices = model.aco()
        best_coords = model.location[best_indices]

    elif algo_name == "GA_EQTSP":
        from sequence_algorithm.GA_EQTSP import GA
        model = GA(num_city=num_city, num_total=25, iteration=200,
                   data=full_data.copy())
        best_coords, best_length, best_indices = model.run()

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    return {
        "path_length": round(float(best_length), 6),
        "num_nodes": num_city,
        "best_indices": [int(i) for i in best_indices],
    }


def run_routing_experiment():
    """Run routing algorithms on clustered data for each UAV count.

    Primary comparison uses N=3 (30 points).
    """
    print("=" * 60)
    print("  ROUTING EXPERIMENT")
    print("=" * 60)

    out_dir = os.path.join(_project_root, "results", "paper_experiments", "routing")
    ensure_dir(out_dir)

    ini_loc, end_loc = get_ini_end_3d()
    algorithms = ["GA", "PSO", "ACO", "GA_EQTSP"]

    # Primary comparison: N=3 (30 points), also run for N=2 and N=4
    for n_uav in [3, 2, 4]:
        n_users = UAV_USER_MAP[n_uav]
        print(f"\n=== UAV={n_uav}, Users={n_users} ===")

        points = load_points(n_users)

        # Load 4D clustering labels
        labels_path = os.path.join(
            _project_root, "results", "paper_experiments", "clustering",
            f"labels_4d_uav{n_uav}.txt",
        )
        if not os.path.isfile(labels_path):
            print(f"  [SKIP] 4D labels not found at {labels_path}. "
                  "Run clustering experiment first.")
            continue
        labels = np.loadtxt(labels_path, dtype=int)
        clusters = split_clusters(points, labels)

        summary = {"n_uav": n_uav, "n_users": n_users, "algorithms": {}}

        for algo in algorithms:
            print(f"\n  --- {algo} ---")
            algo_results = {}
            total_length = 0.0

            for cid in sorted(clusters.keys()):
                cpts = clusters[cid]
                print(f"    Cluster {cid} ({cpts.shape[0]} points) ... ", end="", flush=True)
                t0 = time.time()
                metrics = _run_routing_algo(algo, cpts, ini_loc, end_loc)
                elapsed = time.time() - t0
                metrics["time_s"] = round(elapsed, 2)
                total_length += metrics["path_length"]
                algo_results[f"cluster_{cid}"] = metrics
                print(f"length={metrics['path_length']:.4f}  time={elapsed:.1f}s")

            algo_results["total_path_length"] = round(total_length, 6)
            summary["algorithms"][algo] = algo_results

            # Save per-algorithm results
            algo_path = os.path.join(out_dir, f"routing_{algo}_uav{n_uav}.json")
            with open(algo_path, "w", encoding="utf-8") as f:
                json.dump(algo_results, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_path = os.path.join(out_dir, f"routing_summary_uav{n_uav}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Routing summary saved to {summary_path}")


# ---------------------------------------------------------------------------
#  Experiment 3: Multi-Data-Volume
# ---------------------------------------------------------------------------

def run_multi_data_volume_experiment():
    """Run GA_EQTSP across different data volumes with N=3 UAVs and 4D clustering."""
    print("=" * 60)
    print("  MULTI-DATA-VOLUME EXPERIMENT")
    print("=" * 60)

    out_dir = os.path.join(_project_root, "results", "paper_experiments", "multi_data")
    ensure_dir(out_dir)

    n_uav = 3
    n_users = UAV_USER_MAP[n_uav]
    data_sizes = [100, 200, 300]

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
                "best_indices": [int(i) for i in best_indices],
            }
            print(f"length={best_length:.4f}  time={elapsed:.1f}s")

        ds_results["total_path_length"] = round(total_length, 6)
        ds_results["total_time_s"] = round(total_time, 2)
        results["data_sizes"][str(ds)] = ds_results

        # Save per-data-size results
        ds_path = os.path.join(out_dir, f"ga_eqtsp_data{ds}_uav{n_uav}.json")
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(ds_results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary_path = os.path.join(out_dir, "multi_data_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMulti-data-volume results saved to {summary_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

EXPERIMENT_MAP = {
    "clustering": run_clustering_experiment,
    "routing": run_routing_experiment,
    "multi_data": run_multi_data_volume_experiment,
}


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENT_MAP.keys()) + ["all"],
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    if args.experiment == "all":
        for name, func in EXPERIMENT_MAP.items():
            func()
    else:
        EXPERIMENT_MAP[args.experiment]()


if __name__ == "__main__":
    main()
