"""
Manual K-Means clustering implementation (Lloyd's algorithm + K-Means++ init).
Used for the "Auto Playlist" feature — grouping songs into mood clusters.
No sklearn dependency.
"""
import numpy as np


class ManualKMeans:
    """
    K-Means from scratch.

    Algorithm:
    1. K-Means++ initialization — choose initial centroids that are spread out
       (first centroid is random; each subsequent centroid is drawn with
       probability proportional to squared distance from the nearest existing
       centroid). This avoids the poor local minima that random init suffers from.

    2. Lloyd's algorithm — iterate:
       (a) Assign each point to its nearest centroid (Euclidean distance).
       (b) Update each centroid to the mean of points assigned to it.
       Stop when centroids stop moving (or max_iter reached).

    3. Multiple restarts — run n_init times with different seeds and keep the
       clustering with the lowest inertia (sum of squared distances to centroids).
       K-Means is sensitive to initialization; restarts mitigate that.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, # Initialize constructor with hyperparameters
                 n_init: int = 10, tol: float = 1e-4, random_state: int = 42): # Additional hyperparameters for stopping criteria and seeds
        self.n_clusters = n_clusters # Set the target number of clusters (K)
        self.max_iter = max_iter # Set the maximum number of iterations before forcing a stop
        self.n_init = n_init # Set the number of restarts to find the best global minimum
        self.tol = tol # Set the tolerance threshold for early stopping if centroids don't move
        self.random_state = random_state # Store the random seed for reproducible results

    def _init_centroids_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray: # Define internal K-Means++ initialization logic
        """K-Means++ initialization."""
        n = X.shape[0] # Get the total number of data points
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype) # Pre-allocate an empty array for K centroids

        # First centroid: pick one point uniformly at random from X
        centroids[0] = X[rng.integers(n)] # Randomly select the first centroid from the dataset

        # Subsequent centroids: choose points with probability proportional to squared distance
        for i in range(1, self.n_clusters): # Loop to pick the remaining K-1 centroids
            # Calculate squared Euclidean distance from every point to its closest already-chosen centroid
            # X[:, None, :] (n, 1, d) minus centroids[:i][None, :, :] (1, i, d) -> broadcasted distances
            dists_sq = np.min( # Find the minimum squared distance for each point to the existing centroids
                np.sum((X[:, None, :] - centroids[:i][None, :, :]) ** 2, axis=2), # Compute squared distances using broadcasting
                axis=1, # Reduce along the centroid axis to get the minimum distance per point
            )
            total = dists_sq.sum() # Sum all minimum squared distances to use for normalization
            if total == 0: # Check if all remaining points perfectly overlap with existing centroids
                # Fallback to random if all points overlap with chosen centroids
                centroids[i] = X[rng.integers(n)] # Randomly pick a fallback centroid
            else:
                probs = dists_sq / total # Create a probability distribution favoring points further away
                centroids[i] = X[rng.choice(n, p=probs)] # Select the next centroid probabilistically

        return centroids # Return the initialized K-Means++ centroids

    def _lloyds(self, X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]: # Define Lloyd's algorithm for centroid optimization
        """Lloyd's algorithm inner loop."""
        labels = np.zeros(X.shape[0], dtype=np.int64) # Initialize an array to hold cluster assignments for each point

        for _ in range(self.max_iter): # Loop until maximum iterations are reached
            # Assignment: Compute distance from every point to every centroid
            dists_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2) # Calculate pairwise squared distances
            new_labels = np.argmin(dists_sq, axis=1) # Assign each point to the index of its nearest centroid

            # Update: New centroid = mean coordinates of all points assigned to it
            new_centroids = np.empty_like(centroids) # Pre-allocate an array for the updated centroids
            for k in range(self.n_clusters): # Loop over each cluster index
                mask = new_labels == k # Create a boolean mask for points assigned to cluster k
                if mask.any(): # Check if the cluster has at least one assigned point
                    new_centroids[k] = X[mask].mean(axis=0) # Update the centroid to the mean of its assigned points
                else:
                    new_centroids[k] = centroids[k] # Retain the old centroid position if it has no points

            # Check convergence: total movement of centroids
            shift = np.linalg.norm(new_centroids - centroids) # Calculate how much the centroids moved in total
            centroids = new_centroids # Overwrite old centroids with the updated ones
            labels = new_labels # Overwrite old labels with the new assignments
            if shift < self.tol: # Check if the centroid movement is below the tolerance threshold
                break # Exit the loop early because convergence is achieved

        # Calculate final inertia: total intra-cluster sum of squares
        inertia = float(np.sum(np.sum((X - centroids[labels]) ** 2, axis=1))) # Compute final Sum of Squared Errors (SSE)
        return labels, centroids, inertia # Return the final cluster assignments, centroids, and inertia

    def fit(self, X: np.ndarray) -> "ManualKMeans": # Define the public fit method to run the clustering
        """Run K-Means with n_init restarts; keep the best by inertia."""
        X = np.asarray(X, dtype=np.float64) # Ensure the input data is a double-precision numpy array
        rng = np.random.default_rng(self.random_state) # Initialize the main random number generator

        best_labels = None # Initialize a variable to track the best assignments
        best_centroids = None # Initialize a variable to track the best centroids
        best_inertia = np.inf # Set initial best inertia to infinity

        for i in range(self.n_init): # Loop for the specified number of random restarts
            # Each restart gets its own sub-rng for reproducibility
            sub_rng = np.random.default_rng(self.random_state + i) # Create a seeded sub-generator for this specific run
            centroids = self._init_centroids_pp(X, sub_rng) # Initialize centroids using K-Means++ for this run
            labels, centroids, inertia = self._lloyds(X, centroids) # Optimize the centroids using Lloyd's algorithm

            if inertia < best_inertia: # Check if this run produced a lower inertia than previous runs
                best_inertia = inertia # Update the best recorded inertia
                best_labels = labels # Save the best cluster assignments
                best_centroids = centroids # Save the best centroid coordinates

        self.labels_ = best_labels # Store the best labels as a class attribute
        self.centroids_ = best_centroids # Store the best centroids as a class attribute
        self.inertia_ = best_inertia # Store the best inertia as a class attribute
        return self # Return the fitted model instance

    def predict(self, X: np.ndarray) -> np.ndarray: # Define the public predict method to classify new points
        """Assign new points to the nearest learned centroid."""
        X = np.asarray(X, dtype=np.float64) # Ensure the new input data is correctly formatted
        dists_sq = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2) # Calculate squared distances to learned centroids
        return np.argmin(dists_sq, axis=1) # Return the index of the closest centroid for each new point


def manual_silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 500) -> float: # Define function to calculate Silhouette score
    """
    Approximate silhouette score via random subsampling (no sklearn).

    For each sampled point i:
      a(i) = mean distance to other points in the same cluster
      b(i) = min mean distance to points in any other cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns mean s over sampled points. Range [-1, 1]; higher = better.

    Reference ranges:
      Iris (4D, 3 clean classes) : ~0.55  (gold standard)
      8D metadata only           : ~0.20–0.35
      64D (meta + audio, K=2–10) : ~0.13–0.26  ← this dataset (observed)
      Pure random 64D noise      : ~0.00–0.05

    Why lower than Iris: "concentration of measure" — as d grows, pairwise
    distances concentrate around their mean, so a(i) ≈ b(i) → s(i) → 0.
    This dataset scores well above the random floor because genre structure
    is genuine, but still below Iris because 64D dilutes the signal.
    Any positive mean score indicates real cluster structure.
    """
    rng = np.random.default_rng(42) # Initialize random number generator with a fixed seed
    n = len(X) # Get the total number of samples in the dataset
    idx = rng.choice(n, size=min(sample_size, n), replace=False) # Randomly select indices for subsampling to speed up calculation
    X_s = X[idx].astype(np.float64) # Extract the random subsample of the data points
    L_s = labels[idx] # Extract the corresponding cluster labels for the subsample
    unique = np.unique(L_s) # Find the unique cluster IDs present in the subsample
    m = len(X_s) # Record the actual size of the subsample

    # Pairwise Euclidean distance matrix (m × m) — vectorized
    diff = X_s[:, None, :] - X_s[None, :, :]   # Compute 3D differences between all pairs in the subsample
    D = np.sqrt(np.sum(diff ** 2, axis=2))       # Compute the Euclidean distances (m x m matrix)

    scores = np.empty(m) # Pre-allocate an array to hold the silhouette score for each sampled point
    for i in range(m): # Loop over each point in the subsample
        c = L_s[i] # Get the cluster assignment of the current point
        same_mask = L_s == c # Create a mask identifying all other points in the same cluster
        same_mask[i] = False  # Exclude the current point itself from the same-cluster mask

        if not same_mask.any(): # Check if the current point is the only member of its cluster in the subsample
            scores[i] = 0.0 # Assign a zero score if there are no intra-cluster neighbors to compare against
            continue # Move to the next point

        a = D[i, same_mask].mean() # Calculate 'a': average distance to all other points in the same cluster

        b = np.inf # Initialize 'b' (min inter-cluster distance) to infinity
        for oc in unique: # Loop through all other possible clusters
            if oc == c: # Skip the loop if it's evaluating the point's own cluster
                continue # Move to the next candidate cluster
            other_mask = L_s == oc # Create a mask identifying points in the other cluster
            if not other_mask.any(): # Check if the other cluster is empty in the subsample
                continue # Skip to the next candidate cluster
            mean_d = D[i, other_mask].mean() # Calculate the mean distance to all points in this other cluster
            if mean_d < b: # Check if this distance is the smallest seen so far
                b = mean_d # Update 'b' with the new minimum inter-cluster mean distance

        if np.isinf(b): # Check if 'b' remained infinity (e.g. only one cluster exists)
            scores[i] = 0.0 # Assign a zero score if there are no other clusters to compare against
        else:
            denom = max(a, b) # Determine the maximum of intra-cluster and nearest inter-cluster distance for normalization
            scores[i] = (b - a) / denom if denom > 0 else 0.0 # Calculate the final silhouette score for point i

    return float(scores.mean()) # Return the average silhouette score across all sampled points
