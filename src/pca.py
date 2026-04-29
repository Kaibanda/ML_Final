"""
Manual PCA implementation using covariance matrix + SVD.
Used for 2D visualization of audio embeddings in the Streamlit Evaluation tab.
No sklearn dependency.
"""
import numpy as np # Import numpy for matrix and array operations


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize X to zero mean and unit variance (z-score normalization).
    This is critical before PCA: without it, high-variance features
    (e.g. tempo in BPM) would dominate the principal components purely
    because of their scale, not because they carry more information.
    """
    mean = X.mean(axis=0) # Compute the average value for each feature column
    std = X.std(axis=0) # Compute the standard deviation for each feature column
    std = np.where(std == 0, 1.0, std)  # Replace zero std with 1.0 to prevent division by zero errors
    return (X - mean) / std, mean, std # Return z-score normalized array, along with computed mean and std


def manual_pca(X: np.ndarray, n_components: int = 2):
    """
    Manual PCA via covariance matrix eigendecomposition using SVD.

    Steps:
    1. Center X (subtract mean) so the covariance is computed around the origin.
    2. Compute the covariance matrix: C = X^T @ X / (n-1).
       Each entry C[i,j] measures how much feature i and j vary together.
    3. Apply SVD to C: C = U S V^T.
       - U: eigenvectors (principal component directions)
       - S: eigenvalues (variance explained per component)
    4. Project X onto the top-k principal components: X_pca = X_centered @ U[:, :k]

    Returns (X_pca, explained_variance_ratio, components, mean).
      - X_pca: (n, n_components) — points projected into PC space
      - explained_variance_ratio: (n_components,) — fraction of total variance captured by each component
      - components: (d, n_components) — PC basis vectors (columns are PCs),
        reusable for projecting new points: Y = (new - mean) @ components
      - mean: (d,) — data mean used for centering (for projecting new points)
    """
    n = X.shape[0] # Retrieve the total number of data samples (rows)

    mean_vec = X.mean(axis=0) # Calculate the average value of each feature across all samples
    X_centered = X - mean_vec # Center the data around origin by subtracting the mean

    # Compute the Covariance Matrix: (X^T @ X) / (n - 1)
    cov = X_centered.T @ X_centered / (n - 1) # Calculate covariance matrix capturing feature correlations

    # Perform Singular Value Decomposition (SVD) on the Covariance matrix
    U, S, _ = np.linalg.svd(cov) # Decompose covariance to get eigenvectors (U) and eigenvalues (S)

    components = U[:, :n_components] # Extract the top 'n_components' eigenvectors as the principal components
    X_pca = X_centered @ components # Project the high-dimensional centered data down to the selected PCA dimensions
    explained_variance_ratio = S[:n_components] / S.sum() # Calculate the percentage of total data variance retained

    return X_pca, explained_variance_ratio, components, mean_vec # Return the transformed data and mathematical metadata
