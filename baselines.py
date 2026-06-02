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


# ---------------------------------------------------------------------------
#  Jia2025 baseline: balanced clustering (spatial + workload, no G2A/A2G)
# ---------------------------------------------------------------------------

def jia2025_balanced_clustering(points, offload_volumes, n_clusters,
                                 max_iters=100, tol=1e-4, random_state=42):
    """Jia2025 balanced task assignment: spatial + workload balance.

    This is the baseline from Jia et al. (2025) that considers:
    - Spatial proximity (Euclidean distance)
    - Workload balance (data volume variance)
    But does NOT consider G2A/A2G radio map information.

    Args:
        points: (N, 2) or (N, D) spatial coordinates.
        offload_volumes: (N,) data volume for each inspection point.
        n_clusters: number of clusters (UAVs).
        max_iters: maximum K-means iterations.
        tol: convergence tolerance.
        random_state: seed for reproducibility.

    Returns:
        labels: (N,) balanced cluster label array.
        cluster_centers: (n_clusters, 2) spatial centroids.
    """
    pts_2d = np.asarray(points[:, :2], dtype=np.float64)
    rng = np.random.RandomState(random_state)
    n_samples = pts_2d.shape[0]
    k = n_clusters

    # Normalize spatial coordinates and workload
    spatial_max = np.max(pts_2d, axis=0) - np.min(pts_2d, axis=0) + 1e-8
    pts_norm = pts_2d / spatial_max

    vol_norm = (offload_volumes - np.min(offload_volumes)) / \
               (np.max(offload_volumes) - np.min(offload_volumes) + 1e-8)

    # Augmented feature: [spatial, workload]
    augmented = np.hstack([pts_norm, vol_norm.reshape(-1, 1)])

    # K-means++ initialisation
    centers = _init_centers_kmeans_pp(augmented, k, rng)
    labels = np.full(n_samples, -1, dtype=int)

    # K-means iterations in augmented space
    for _ in range(max_iters):
        dist_sq = _euclidean_dist_sq(augmented, centers)
        new_labels = np.argmin(dist_sq, axis=1)

        new_centers = np.zeros_like(centers)
        for idx in range(k):
            mask = (new_labels == idx)
            if mask.sum() > 0:
                new_centers[idx] = augmented[mask].mean(axis=0)
            else:
                new_centers[idx] = augmented[rng.randint(0, n_samples)]

        center_shift = np.sqrt(np.sum((centers - new_centers) ** 2, axis=1))
        centers = new_centers
        labels = new_labels

        if np.max(center_shift) <= tol:
            break

    # Capacity-constrained balanced post-processing
    final_dist_sq = _euclidean_dist_sq(augmented, centers)

    base = n_samples // k
    rest = n_samples % k
    capacities = [base + 1 if i < rest else base for i in range(k)]

    pref_order = np.argsort(final_dist_sq, axis=1)
    min_cost = np.min(final_dist_sq, axis=1)
    pts_order = np.argsort(min_cost)

    labels_bal = np.full(n_samples, -1, dtype=int)
    cap = capacities.copy()

    for p in pts_order:
        for c in pref_order[p]:
            if cap[c] > 0:
                labels_bal[p] = c
                cap[c] -= 1
                break

    unassigned = np.where(labels_bal == -1)[0]
    for p in unassigned:
        labels_bal[p] = int(pref_order[p, 0])

    # Compute spatial centroids
    spatial_centers = np.zeros((k, 2))
    for idx in range(k):
        mask = (labels_bal == idx)
        if mask.sum() > 0:
            spatial_centers[idx] = pts_2d[mask].mean(axis=0)
        else:
            spatial_centers[idx] = pts_2d[rng.randint(0, n_samples)]

    return labels_bal, spatial_centers


# ---------------------------------------------------------------------------
#  TSP baseline: nearest-neighbor + 2-opt
# ---------------------------------------------------------------------------

def run_tsp_routing(cluster_points, ini_loc, end_loc):
    """Run TSP routing using nearest-neighbor heuristic + 2-opt improvement.

    This is the baseline routing method that uses Euclidean distance only.

    Args:
        cluster_points: (M, D) inspection-point coordinates for this cluster.
        ini_loc: 1-D start location (length D).
        end_loc: 1-D end location (length D).

    Returns:
        best_coords: (M+2, D) ordered path coordinates (start -> points -> end).
        best_length: best path length (float).
        best_indices: index list used to recover the ordering.
    """
    full_data = np.vstack([np.asarray(ini_loc), np.asarray(cluster_points),
                           np.asarray(end_loc)])
    n = full_data.shape[0]

    # Compute distance matrix
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_mat[i, j] = np.linalg.norm(full_data[i] - full_data[j])

    # Nearest-neighbor heuristic (start from node 0, end at node n-1)
    def nearest_neighbor():
        visited = [0]
        unvisited = set(range(1, n - 1))
        current = 0

        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_mat[current, x])
            visited.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        visited.append(n - 1)  # Add end node
        return visited

    # 2-opt improvement
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Calculate cost change
                    old_cost = (dist_mat[route[i-1], route[i]] +
                               dist_mat[route[j], route[j+1]])
                    new_cost = (dist_mat[route[i-1], route[j]] +
                               dist_mat[route[i], route[j+1]])
                    if new_cost < old_cost:
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
        return route

    # Run nearest-neighbor + 2-opt
    route = nearest_neighbor()
    route = two_opt(route)

    # Calculate path length
    best_length = sum(dist_mat[route[i], route[i+1]] for i in range(len(route)-1))

    return full_data[route], best_length, route


# ---------------------------------------------------------------------------
#  Single-map GA baseline (A2G only, no G2A penalty)
# ---------------------------------------------------------------------------

def run_single_map_ga(cluster_points, ini_loc, end_loc,
                      num_total=25, iteration=200, data_size=300):
    """Run GA with A2G-only radio map (no G2A penalty).

    This baseline uses GA_EQTSP but disables the G2A penalty to evaluate
    the importance of considering both radio maps.

    Args:
        cluster_points: (M, D) inspection-point coordinates for this cluster.
        ini_loc: 1-D start location (length D).
        end_loc: 1-D end location (length D).
        num_total: GA population size.
        iteration: GA generations.
        data_size: data volume parameter.

    Returns:
        best_coords: (M+2, D) ordered path coordinates.
        best_length: path length (float).
        best_indices: index list used to recover the ordering.
    """
    from sequence_algorithm.GA_EQTSP import GA

    full_data = np.vstack([np.asarray(ini_loc), np.asarray(cluster_points),
                           np.asarray(end_loc)])
    num_city = full_data.shape[0]

    # Create GA model with disabled G2A penalty
    model = GA(num_city=num_city, num_total=num_total,
               iteration=iteration, data=full_data.copy())

    # Override G2A penalty to 0 (disable G2A consideration)
    model.compute_weight_mec = lambda n, ds, th, g2a: _a2g_only_weight(n, ds, th)

    if data_size != 300:
        model.Data_size = data_size

    # Recompute matrices with A2G-only weights
    model.weight_mec_mat = model.compute_weight_mec(
        num_city, model.Data_size, model.throught_mat, model.g2a_outage_mat)
    model.dis_mat = model.compute_dis_mat(num_city, full_data)

    best_coords, best_length, best_indices = model.run()
    return best_coords, best_length, best_indices


def _a2g_only_weight(num_city, Data_size, Throught):
    """Compute A2G-only weight matrix (no G2A penalty)."""
    matrix_weight = np.zeros([num_city, num_city])
    for i in range(num_city):
        for j in range(num_city):
            if i == j or Throught[i][j] == 0:
                matrix_weight[i][j] = np.inf
                continue
            # Only A2G transmission cost, no G2A penalty
            matrix_weight[i][j] = Data_size / Throught[i][j]
    return matrix_weight
