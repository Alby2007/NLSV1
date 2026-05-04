"""
Phase 1 — Step 2: Cluster activation vectors into candidate primitives.

Reads data/activations.npy, runs MiniBatch K-Means, saves:
  data/centroids.npy      — centroid matrix C ∈ ℝ^(k × d)
  data/assignments.npy    — cluster assignment per vector
"""

import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    KMEANS_K, KMEANS_BATCH_SIZE, KMEANS_N_INIT,
    ACTIVATIONS_PATH, CENTROIDS_PATH, ASSIGNMENTS_PATH,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Phase 1 — cluster_primitives starting")
    if not ACTIVATIONS_PATH.exists():
        raise RuntimeError(f"Activations not found at {ACTIVATIONS_PATH}. Run extract_activations.py first.")
    logger.info(f"Loading activations from {ACTIVATIONS_PATH}...")
    A = np.load(ACTIVATIONS_PATH)
    logger.info(f"Activation matrix shape: {A.shape}")

    logger.info(f"Running MiniBatch K-Means: k={KMEANS_K}, batch_size={KMEANS_BATCH_SIZE}, n_init={KMEANS_N_INIT}")
    km = MiniBatchKMeans(
        n_clusters=KMEANS_K,
        batch_size=KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
        random_state=42,
        verbose=1,
    )
    assignments = km.fit_predict(A)
    centroids = km.cluster_centers_.astype(np.float32)

    np.save(CENTROIDS_PATH, centroids)
    np.save(ASSIGNMENTS_PATH, assignments)

    logger.info(f"Saved centroids to {CENTROIDS_PATH}  shape={centroids.shape}")
    logger.info(f"Saved assignments to {ASSIGNMENTS_PATH}  shape={assignments.shape}")

    cluster_sizes = np.bincount(assignments, minlength=KMEANS_K)
    logger.info(
        f"Cluster size stats — min: {cluster_sizes.min()}, max: {cluster_sizes.max()}, "
        f"mean: {cluster_sizes.mean():.1f}, empty: {(cluster_sizes == 0).sum()}"
    )


if __name__ == "__main__":
    main()
