"""Baseline schemes for paper comparison experiments."""
import sys
import os

# Add project root to sys.path so we can import from sequence_algorithm/
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.append(_current_dir)

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
#  Helper utilities (mirrored from Clustering.py)
# ---------------------------------------------------------------------------

def _euclidean_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix: (n, d) x (m, d) -> (n, m)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def _init_centers_kmeans_pp(points: np.ndarray, k: int,
                            rng: np.random.RandomState) -> np.ndarray:
    """K-means++ initialisation (same logic as Clustering.py)."""
    n_samples = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=points.dtype)
    idx = rng.randint(0, n_samples)
    centers[0] = points[idx]
    closest_dist_sq = _euclidean_dist_sq(points, centers[0:1]).reshape(-1)
    for i in range(1, k):
        probs = closest_dist_sq / closest_dist_sq.sum()
        r = rng.rand()
        cumulative = np.cumsum(probs)
        next_idx = np.searchsorted(cumulative, r)
        centers[i] = points[next_idx]
        dist_sq_to_new = _euclidean_dist_sq(points, centers[i:i + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_to_new)
    return centers


# ---------------------------------------------------------------------------
#  Clustering baselines
# ---------------------------------------------------------------------------

def naive_kmeans_clustering(points, n_clusters, random_state=42):
    """Standard K-means using only spatial coordinates (x, y).

    Args:
        points: (N, D) array; only the first two columns are used.
        n_clusters: number of clusters (UAVs).
        random_state: seed for reproducibility.

    Returns:
        labels: (N,) cluster label array.
        cluster_centers: (n_clusters, 2) centroid array.
    """
    pts_2d = points[:, :2]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(pts_2d)
    return kmeans.labels_, kmeans.cluster_centers_


def balanced_naive_kmeans(points, n_clusters, max_iters=100, tol=1e-4,
                          random_state=42):
    """Naive K-means with balanced post-processing (equal-size clusters).

    Uses K-means++ initialisation on spatial-only features, then applies the
    same capacity-constrained assignment as Clustering.py (lines 139-164) to
    enforce balanced cluster sizes.

    Args:
        points: (N, 2) or (N, D) spatial coordinates.
        n_clusters: number of clusters (UAVs).
        max_iters: maximum K-means iterations.
        tol: convergence tolerance on centre shift.
        random_state: seed for reproducibility.

    Returns:
        labels: (N,) balanced cluster label array.
        cluster_centers: (n_clusters, 2) spatial centroids.
    """
    pts_2d = np.asarray(points[:, :2], dtype=np.float64)
    rng = np.random.RandomState(random_state)
    n_samples = pts_2d.shape[0]
    k = n_clusters

    # K-means++ initialisation
    centers = _init_centers_kmeans_pp(pts_2d, k, rng)
    labels = np.full(n_samples, -1, dtype=int)

    # Standard K-means iterations
    for _ in range(max_iters):
        dist_sq = _euclidean_dist_sq(pts_2d, centers)
        new_labels = np.argmin(dist_sq, axis=1)

        new_centers = np.zeros_like(centers)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers[idx] = pts_2d[mask].mean(axis=0)
            else:
                new_centers[idx] = pts_2d[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        centers = new_centers
        labels = new_labels

        if np.max(center_shift) <= tol:
            break

    # Capacity-constrained balanced post-processing (Clustering.py lines 139-164)
    final_dist_sq = _euclidean_dist_sq(pts_2d, centers)

    base = n_samples // k
    rest = n_samples % k
    capacities = [base + 1 if i < rest else base for i in range(k)]

    pref_order = np.argsort(final_dist_sq, axis=1)  # (N, k) preferred cluster order
    min_cost = np.min(final_dist_sq, axis=1)
    pts_order = np.argsort(min_cost)  # assign most-confident points first

    labels_bal = np.full(n_samples, -1, dtype=int)
    cap = capacities.copy()

    for p in pts_order:
        for c in pref_order[p]:
            if cap[c] > 0:
                labels_bal[p] = c
                cap[c] -= 1
                break

    # Safety net for any unassigned points
    unassigned = np.where(labels_bal == -1)[0]
    for p in unassigned:
        labels_bal[p] = int(pref_order[p, 0])

    # Compute spatial centroids
    spatial_centers = np.zeros((k, pts_2d.shape[1]))
    for idx in range(k):
        mask = (labels_bal == idx)
        if mask.sum() > 0:
            spatial_centers[idx] = pts_2d[mask].mean(axis=0)
        else:
            spatial_centers[idx] = pts_2d[rng.randint(0, n_samples)]

    return labels_bal, spatial_centers


# ---------------------------------------------------------------------------
#  TSP / GA baselines
# ---------------------------------------------------------------------------

def run_standard_ga(cluster_points, ini_loc, end_loc,
                    num_total=100, iteration=300):
    """Run standard GA (Euclidean distance only) on a single cluster.

    Args:
        cluster_points: (M, D) inspection-point coordinates for this cluster.
        ini_loc: 1-D start location (length D).
        end_loc: 1-D end location (length D).
        num_total: GA population size.
        iteration: GA generations.

    Returns:
        best_coords: (M+2, D) ordered path coordinates (start -> points -> end).
        best_length: best path length (float).
        best_indices: index list used to recover the ordering.
    """
    from sequence_algorithm.GA import GA

    full_data = np.vstack([np.asarray(ini_loc), np.asarray(cluster_points),
                           np.asarray(end_loc)])
    num_city = full_data.shape[0]
    model = GA(num_city=num_city, num_total=num_total,
               iteration=iteration, data=full_data.copy())
    best_coords, best_length, best_indices = model.run()
    return best_coords, best_length, best_indices


def run_ga_eqtsp(cluster_points, ini_loc, end_loc,
                 num_total=25, iteration=200, data_size=300):
    """Run GA_EQTSP (communication-aware GA) on a single cluster.

    Args:
        cluster_points: (M, D) inspection-point coordinates for this cluster.
        ini_loc: 1-D start location (length D).
        end_loc: 1-D end location (length D).
        num_total: GA population size.
        iteration: GA generations.
        data_size: data volume parameter (bits) for the communication weight.

    Returns:
        best_coords: (M+2, D) ordered path coordinates.
        best_length: communication-weighted best path length (float).
        best_indices: index list used to recover the ordering.
    """
    from sequence_algorithm.GA_EQTSP import GA

    full_data = np.vstack([np.asarray(ini_loc), np.asarray(cluster_points),
                           np.asarray(end_loc)])
    num_city = full_data.shape[0]
    model = GA(num_city=num_city, num_total=num_total,
               iteration=iteration, data=full_data.copy())
    # Override Data_size if caller specifies a different value
    if data_size != 300:
        model.Data_size = data_size
    best_coords, best_length, best_indices = model.run()
    return best_coords, best_length, best_indices
