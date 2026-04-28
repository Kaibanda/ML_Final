import numpy as np
import pandas as pd


METADATA_COLS = [ # Define the list of metadata feature columns extracted from Spotify
    'danceability', 'energy', 'speechiness', 'acousticness', # Audio features representing mood and style
    'instrumentalness', 'liveness', 'valence', 'tempo' # Audio features representing instrumentation and speed
]

# 56D audio feature names — order must match extract_audio_features() in audio_feature_extractor.py
AUDIO_FEATURE_NAMES = ( # Construct the list of names for the 56 audio features extracted via Librosa
    [f"mfcc{i}_mean" for i in range(1, 14)] # Generate names for the 13 MFCC mean features
    + [f"mfcc{i}_std" for i in range(1, 14)] # Generate names for the 13 MFCC standard deviation features
    + ["chroma_C", "chroma_C#", "chroma_D", "chroma_D#", "chroma_E", # List the 12 Chroma pitch classes (C to B)
       "chroma_F", "chroma_F#", "chroma_G", "chroma_G#", "chroma_A", # List the 12 Chroma pitch classes (C to B)
       "chroma_A#", "chroma_B"] # List the 12 Chroma pitch classes (C to B)
    + ["centroid_mean", "centroid_std", # Names for spectral centroid mean and std
       "bandwidth_mean", "bandwidth_std", # Names for spectral bandwidth mean and std
       "rolloff_mean", "rolloff_std", # Names for spectral rolloff mean and std
       "zcr_mean", "zcr_std", # Names for zero-crossing rate mean and std
       "rms_mean", "rms_std"] # Names for root-mean-square energy mean and std
    + ["audio_tempo"] # Name for the estimated audio tempo feature
    + [f"spec_contrast_{i}" for i in range(1, 8)] # Generate names for the 7 spectral contrast bands
)

# 64D full feature space used by K-Means (meta + audio)
FULL_FEATURE_NAMES = METADATA_COLS + AUDIO_FEATURE_NAMES # Combine metadata and audio feature names into a single list


def build_embedding_std(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # Define a function to standardize embeddings
    """
    Returns (emb_std, feat_mean, feat_std) where emb_std is the embedding matrix
    z-score standardized per feature dimension using the training distribution.

    Use this — not the L2-normalized emb_matrix — when matching a new song that
    has no Spotify metadata. Unlike cosine similarity (which is magnitude-invariant),
    Euclidean distance on z-scored features respects absolute feature magnitudes:
    [0.1, 0.1, ...] and [0.9, 0.9, ...] correctly produce a non-zero distance
    instead of being treated as identical unit vectors.
    """
    raw = np.stack(df['embedding'].values).astype(np.float32) # Convert the list of embedding arrays into a 2D numpy matrix
    mean = raw.mean(axis=0) # Calculate the mean vector across all embeddings
    std = raw.std(axis=0) # Calculate the standard deviation vector across all embeddings
    std = np.where(std == 0, 1.0, std) # Replace zero standard deviations with 1.0 to prevent division by zero
    return (raw - mean) / std, mean, std # Return the z-score standardized embeddings, along with the mean and std vectors


def build_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: # Define a function to prepare normalized matrices for recommendations
    """Build normalized metadata and embedding matrices from master dataframe."""
    meta = df[METADATA_COLS].copy() # Extract the metadata columns from the dataframe into a new object
    # Normalize tempo to [0, 1] so MSE is comparable across features
    meta['tempo'] = meta['tempo'] / 200.0 # Scale tempo assuming a max of 200 BPM to align with other [0, 1] bounded features
    meta_matrix = meta.values.astype(np.float32) # Convert the metadata dataframe into a 32-bit float numpy matrix

    emb_matrix = np.stack(df['embedding'].values).astype(np.float32) # Convert the list of audio embeddings into a 32-bit float numpy matrix

    # 1. Z-score Standardize per feature dimension to prevent Cosine Collapse
    # This ensures large features (like MFCC_0) do not dominate the Euclidean shape.
    feat_mean = emb_matrix.mean(axis=0) # Calculate the mean for each audio feature dimension
    feat_std = emb_matrix.std(axis=0) # Calculate the standard deviation for each audio feature dimension
    feat_std = np.where(feat_std == 0, 1.0, feat_std) # Prevent division by zero for constant features
    emb_matrix = (emb_matrix - feat_mean) / feat_std # Apply z-score standardization to the audio embeddings

    # 2. L2-normalize each embedding row so cosine similarity = dot product
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) # Calculate the L2 norm (magnitude) of each embedding vector
    norms = np.where(norms == 0, 1.0, norms) # Prevent division by zero for zero-vectors
    emb_matrix = emb_matrix / norms # Normalize each vector to unit length

    return meta_matrix, emb_matrix # Return the prepared metadata and embedding matrices


def cosine_similarity(A: np.ndarray, b: np.ndarray) -> np.ndarray: # Define a function to compute cosine similarities
    """
    Compute cosine similarity between each row of A and vector b.
    Cosine similarity measures the angle between vectors, not magnitude,
    making it robust in high-dimensional spaces where MSE suffers from
    the curse of dimensionality.
    """
    # L2-normalize each row of A and vector b
    A_norms = np.linalg.norm(A, axis=1, keepdims=True) # Calculate the L2 norm of each row in matrix A
    A_norms = np.where(A_norms == 0, 1.0, A_norms) # Replace zero norms with 1.0 to avoid division by zero
    b_norm = np.linalg.norm(b) # Calculate the L2 norm of vector b
    b_norm = b_norm if b_norm != 0 else 1.0 # Replace zero norm with 1.0 to avoid division by zero

    # Dot product of normalized vectors = cosine similarity
    return (A / A_norms) @ (b / b_norm) # Return the dot product of the L2-normalized matrices, which yields cosine similarity


def recommend( # Define the core recommendation function
    query_idx: int, # The index of the seed song in the dataset
    df: pd.DataFrame, # The main dataframe containing song metadata
    meta_matrix: np.ndarray, # The normalized metadata matrix
    emb_matrix: np.ndarray, # The normalized audio embedding matrix
    lambda_weight: float = 0.5, # The weight balancing metadata vs audio distance
    top_k: int = 5, # The number of recommendations to return
    genre_filter: str = None, # An optional genre string to filter recommendations
) -> list[dict]: # Returns a list of dictionaries containing recommendation details
    """
    Recommend songs using a hybrid multimodal distance metric.
    
    Formula: score = MSE(metadata) + λ * (1 - CosineSimilarity(audio))
    
    Mathematical Rationale:
    1. MSE on Metadata (8D):
       Metadata features (e.g., energy, valence) are bounded [0, 1]. In such low-dimensional 
       spaces, Euclidean/MSE distance is a valid metric for absolute "vibe" differences.
       
    2. Cosine on Audio (56D):
       High-dimensional vectors suffer from the "Curse of Dimensionality" where Euclidean 
       distances tend to concentrate (making songs appear equally distant). Cosine Similarity 
       circumvents this by focusing on the *angle* (semantic direction) between the 
       acoustic signatures rather than their magnitude.
       
    3. λ (Lambda):
       Balances the two distance systems. A tuned λ=0.1 ensures that metadata acts as 
       the primary filter while audio embeddings provide fine-grained timbral matching.
    """
    q_meta = meta_matrix[query_idx] # Retrieve the metadata vector for the seed song
    q_emb = emb_matrix[query_idx] # Retrieve the audio embedding vector for the seed song

    # --- Step 1: Metadata Distance (MSE) ---
    # We measure absolute deviation in track characteristics. 
    # Small differences (e.g., 0.1 vs 0.12 tempo) yield near-zero scores.
    meta_dists = np.mean((meta_matrix - q_meta) ** 2, axis=1) # Calculate the Mean Squared Error (MSE) between the seed metadata and all other songs

    # --- Step 2: Audio Embedding Distance (1 - Cosine) ---
    # We focus on the "Acoustic Signature" direction.
    # Note: emb_matrix was Z-scored and L2-normalized during build_matrices, 
    # so (1 - dot_product) is equivalent to 0.5 * squared_euclidean(normalized_vectors).
    cos_sim = cosine_similarity(emb_matrix, q_emb) # Calculate the cosine similarity between the seed audio and all other songs
    emb_dists = 1.0 - cos_sim # Convert similarity to distance by subtracting from 1

    # --- Step 3: Hybrid Fusion ---
    scores = meta_dists + lambda_weight * emb_dists # Combine the two distances using the lambda weight parameter

    if genre_filter and genre_filter != "All": # Check if a specific genre filter was requested
        mask = df['playlist_genre'] != genre_filter # Create a boolean mask identifying songs that DO NOT match the genre
        scores[mask] = np.inf # Set the score of non-matching songs to infinity to exclude them

    scores[query_idx] = np.inf # Set the score of the seed song itself to infinity so it isn't recommended

    top_indices = np.argsort(scores)[:top_k] # Sort the scores ascending and retrieve the indices of the top-k lowest scores

    return [ # Construct and return the list of recommendation dictionaries
        { # Create a dictionary for each recommended song
            'idx': int(idx), # Store the index of the recommended song
            'score': float(scores[idx]), # Store the final hybrid distance score
            'meta_dist': float(meta_dists[idx]), # Store the metadata component of the distance
            'emb_dist': float(emb_dists[idx]), # Store the audio component of the distance
        }
        for idx in top_indices # Iterate over the indices of the top matches
    ]


def sanity_check( # Define a function to test the distance metric's logic
    song_a: str, # The name of the anchor song
    song_b: str, # The name of the conceptually closer target song
    song_c: str, # The name of the conceptually further target song
    df: pd.DataFrame, # The dataset dataframe
    meta_matrix: np.ndarray, # The metadata matrix
    emb_matrix: np.ndarray, # The audio embedding matrix
    lambda_weight: float = 0.5, # The hybrid balancing weight
) -> dict: # Returns a dictionary containing the test results
    """
    Verify that song_a is closer to song_b than to song_c.
    Used to evaluate whether the distance metric makes musical sense.
    """
    def find_idx(name): # Helper function to find a song's index by its name
        matches = df[df['track_name'].str.lower() == name.lower()] # Search for exact case-insensitive matches in the dataframe
        return matches.index[0] if not matches.empty else None # Return the first matching index, or None if not found

    idx_a, idx_b, idx_c = find_idx(song_a), find_idx(song_b), find_idx(song_c) # Look up the indices for all three input songs
    if any(i is None for i in [idx_a, idx_b, idx_c]): # Check if any of the songs could not be found
        return {"error": "One or more songs not found in dataset."} # Return an error dictionary if lookup failed

    def score(i, j): # Helper function to compute the hybrid score between two song indices
        m = float(np.mean((meta_matrix[i] - meta_matrix[j]) ** 2)) # Compute the metadata MSE distance
        cos = float(cosine_similarity(emb_matrix[i:i+1], emb_matrix[j])[0]) # Compute the audio cosine similarity
        e = 1.0 - cos # Convert audio similarity to distance
        return m + lambda_weight * e, m, e # Return the hybrid score along with the individual components

    score_ab, m_ab, e_ab = score(idx_a, idx_b) # Calculate the distance between song A and song B
    score_ac, m_ac, e_ac = score(idx_a, idx_c) # Calculate the distance between song A and song C

    return { # Construct and return the dictionary of sanity check results
        'song_a': song_a, 'song_b': song_b, 'song_c': song_c, # Include the names of the tested songs
        'score_ab': score_ab, 'score_ac': score_ac, # Include the total hybrid scores
        'meta_ab': m_ab, 'meta_ac': m_ac, # Include the metadata distance components
        'emb_ab': e_ab, 'emb_ac': e_ac, # Include the audio distance components
        'passed': score_ab < score_ac, # Evaluate whether A is closer to B than to C (the test condition)
    }
