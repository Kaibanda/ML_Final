"""
Evaluation + λ tuning for the recommendation score:
    score = MSE_metadata + λ × (1 − cos_sim_embedding)

Produces three outputs:
  1. λ sweep — pick the best λ based on same-artist vs random-pair discrimination
  2. Ablation study — compare hybrid to each modality alone, showing both are needed
  3. Precision@5 — for 100 random seed songs whose artist has ≥2 tracks in the DB,
     measure what fraction of the top-5 recommendations share the seed's artist.
     Higher = the metric actually surfaces the "right" neighbors.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys # Import sys to manipulate the Python runtime environment
sys.path.append(str(Path(__file__).parent.parent)) # Add the project root to the PYTHONPATH so 'src' module imports work correctly

from src.recommender import build_matrices, cosine_similarity # Import the core mathematical functions from our recommender engine

PROJECT_ROOT = Path(__file__).parent.parent # Define the root directory of the project
N_PAIRS = 100 # Define the number of song pairs to use for the discrimination tests
N_SEEDS = 100 # Define the number of seed songs to test Precision@K against
K = 5 # Define the 'K' in Precision@K (how many recommendations to retrieve)
RANDOM_SEED = 42 # Define a fixed seed for reproducible random sampling


def build_same_artist_pairs(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]: # Define function to generate pairs of songs by the same artist
    artist_groups = df.groupby('track_artist').indices # Group the dataframe by artist and get the row indices for each group
    valid_artists = [a for a, idxs in artist_groups.items() if len(idxs) >= 2] # Filter for artists that have at least two songs in the dataset
    pairs = [] # Initialize an empty list to store the generated index pairs
    for _ in range(n): # Loop n times to generate n pairs
        artist = rng.choice(valid_artists) # Randomly select an eligible artist
        idxs = artist_groups[artist] # Get all row indices for that selected artist
        i, j = rng.choice(idxs, size=2, replace=False) # Randomly pick two distinct songs by that artist
        pairs.append((int(i), int(j))) # Append the integer indices as a tuple to the list
    return pairs # Return the list of same-artist pairs


def build_random_pairs(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[tuple[int, int]]: # Define function to generate pairs of songs by different artists
    pairs = [] # Initialize an empty list to store the generated random pairs
    while len(pairs) < n: # Continue generating until we reach the requested number of pairs
        i, j = rng.integers(0, len(df), size=2) # Pick two completely random indices from the dataset
        if df.iloc[i]['track_artist'] != df.iloc[j]['track_artist']: # Ensure the two randomly selected songs have different artists
            pairs.append((int(i), int(j))) # If they are by different artists, add them to the list
    return pairs # Return the list of cross-artist random pairs


def pair_score(meta: np.ndarray, emb: np.ndarray, i: int, j: int, mode: str, lam: float) -> float: # Define function to compute distance between two specific songs
    """Distance between two songs under the chosen scoring mode."""
    if mode == 'mse_only': # Check if we are running the metadata-only ablation test
        return float(np.mean((meta[i] - meta[j]) ** 2)) # Return the purely Euclidean (MSE) distance on the metadata features
    if mode == 'cos_only': # Check if we are running the audio-only ablation test
        cos = float(cosine_similarity(emb[i:i+1], emb[j])[0]) # Calculate the cosine similarity for the audio embeddings
        return 1.0 - cos # Return the audio cosine distance
    m = float(np.mean((meta[i] - meta[j]) ** 2)) # For hybrid mode, compute the metadata distance
    cos = float(cosine_similarity(emb[i:i+1], emb[j])[0]) # For hybrid mode, compute the audio similarity
    return m + lam * (1.0 - cos) # Return the final fused hybrid score using the given lambda weight


def pass_rate(meta, emb, same_pairs, rand_pairs, mode: str, lam: float) -> float: # Define function to calculate the discrimination pass rate
    same = np.array([pair_score(meta, emb, i, j, mode, lam) for i, j in same_pairs]) # Compute distances for all same-artist pairs
    rand = np.array([pair_score(meta, emb, i, j, mode, lam) for i, j in rand_pairs]) # Compute distances for all random cross-artist pairs
    return float(np.mean(same[:, None] < rand[None, :])) # Return the fraction of times a same-artist pair was scored closer than a random pair


def precision_at_k(df: pd.DataFrame, meta: np.ndarray, emb: np.ndarray, # Define function to evaluate the real-world recommendation quality
                   mode: str, lam: float, k: int, n_seeds: int, # Pass in the evaluation parameters and state
                   rng: np.random.Generator) -> float: # Require a seeded generator for reproducibility
    """For each seed, fraction of top-k neighbors sharing the seed's artist. Averaged across seeds."""
    artist_counts = df['track_artist'].value_counts() # Count how many tracks each artist has in total
    multi_artist = set(artist_counts[artist_counts >= 2].index) # Identify artists with at least two tracks (needed to have a possible 'hit')
    eligible = df[df['track_artist'].isin(multi_artist)].index.tolist() # Filter dataset indices to only include those eligible tracks
    n_seeds = min(n_seeds, len(eligible)) # Cap the requested number of seeds if there aren't enough eligible tracks

    seeds = rng.choice(eligible, size=n_seeds, replace=False) # Randomly select the seed tracks for evaluation

    # Precompute embedding norms (needed only for cos-based modes)
    emb_norms = np.linalg.norm(emb, axis=1) # Calculate the magnitude of all audio embeddings in advance
    emb_norms = np.where(emb_norms == 0, 1.0, emb_norms) # Prevent division by zero errors for zero-vectors

    total_hits = 0 # Initialize a counter for successful artist matches
    for seed_idx in seeds: # Iterate over each randomly selected seed track
        seed_artist = df.iloc[seed_idx]['track_artist'] # Look up the target artist we want to match

        if mode == 'mse_only': # Calculate scores using only metadata
            scores = np.mean((meta - meta[seed_idx]) ** 2, axis=1) # Broadcast MSE calculation across the whole dataset
        elif mode == 'cos_only': # Calculate scores using only audio embeddings
            q_norm = np.linalg.norm(emb[seed_idx]) or 1.0 # Get the magnitude of the seed embedding
            cos = (emb @ emb[seed_idx]) / (emb_norms * q_norm) # Vectorized cosine similarity across the whole dataset
            scores = 1.0 - cos # Convert similarity into distance
        else: # Calculate hybrid scores
            meta_d = np.mean((meta - meta[seed_idx]) ** 2, axis=1) # Get the metadata distance component
            q_norm = np.linalg.norm(emb[seed_idx]) or 1.0 # Get the seed audio magnitude
            cos = (emb @ emb[seed_idx]) / (emb_norms * q_norm) # Get the audio similarity component
            scores = meta_d + lam * (1.0 - cos) # Combine components using the lambda weight

        scores[seed_idx] = np.inf # Prevent the seed track from recommending itself by setting its distance to infinity
        top_k = np.argsort(scores)[:k] # Get the indices of the 'K' tracks with the smallest distance scores
        hits = sum(1 for idx in top_k if df.iloc[idx]['track_artist'] == seed_artist) # Count how many of those top-K tracks share the seed's artist
        total_hits += hits # Add the hits to the running total

    return total_hits / (n_seeds * k) # Compute the average precision ratio across all evaluated seeds


def main(): # Define the main execution script
    data_path = PROJECT_ROOT / 'data/dataset/master_music_data.parquet' # Set the path to the fully merged master dataset
    if not data_path.exists(): # Ensure the dataset has been built before running the tuner
        print(f"Missing {data_path}. Run build_master_dataset.py first.") # Print error if data is missing
        return # Exit the script early

    df = pd.read_parquet(data_path) # Load the full master dataset into memory
    meta, emb = build_matrices(df) # Normalize the dataset into the math-ready metadata and audio matrices
    rng = np.random.default_rng(RANDOM_SEED) # Initialize the random generator with our fixed seed

    same_pairs = build_same_artist_pairs(df, N_PAIRS, rng) # Generate the evaluation set of same-artist pairs
    rand_pairs = build_random_pairs(df, N_PAIRS, rng) # Generate the evaluation set of random cross-artist pairs

    # ── (1) λ sweep ──────────────────────────────────────────────────────────
    print("=" * 70) # Print decorative header
    print("(1) λ Sweep — same-artist vs random-pair discrimination") # Print section title
    print("=" * 70) # Print decorative header
    print(f"{'λ':>6} | {'same_avg':>10} | {'rand_avg':>10} | {'pass_rate':>10}") # Print column headers for the results table
    print("-" * 55) # Print table separator line

    lambdas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] # Define the grid search space for the lambda weight
    results = [] # Initialize list to store the results of the sweep
    for lam in lambdas: # Iterate through each candidate lambda value
        same_scores = np.array([pair_score(meta, emb, i, j, 'hybrid', lam) for i, j in same_pairs]) # Get distances for all similar pairs using this lambda
        rand_scores = np.array([pair_score(meta, emb, i, j, 'hybrid', lam) for i, j in rand_pairs]) # Get distances for all random pairs using this lambda
        pr = float(np.mean(same_scores[:, None] < rand_scores[None, :])) # Compute the pass rate (how often similar pairs scored closer than random pairs)
        results.append({'lambda': lam, 'pass_rate': pr}) # Save the lambda and its pass rate to the results list
        print(f"{lam:>6.2f} | {same_scores.mean():>10.6f} | {rand_scores.mean():>10.6f} | {pr:>10.4f}") # Print the metrics to the console table

    best_lam = max(results, key=lambda r: r['pass_rate'])['lambda'] # Find the lambda that achieved the absolute highest pass rate
    print(f"\nBest λ: {best_lam} (random-chance baseline = 0.5)") # Announce the winning lambda weight

    # ── (2) Ablation: hybrid vs each modality alone ──────────────────────────
    print("\n" + "=" * 70) # Print decorative header
    print(f"(2) Ablation Study — Pass rate at the chosen λ={best_lam}") # Print section title
    print("=" * 70) # Print decorative header
    print(f"{'Variant':<25} | {'pass_rate':>10}") # Print column headers
    print("-" * 45) # Print table separator line
    pr_mse = pass_rate(meta, emb, same_pairs, rand_pairs, 'mse_only', 0.0) # Run the discrimination test using ONLY Spotify metadata
    pr_cos = pass_rate(meta, emb, same_pairs, rand_pairs, 'cos_only', 0.0) # Run the discrimination test using ONLY Librosa audio
    pr_hybrid = pass_rate(meta, emb, same_pairs, rand_pairs, 'hybrid', best_lam) # Run the test using the best tuned hybrid setting
    print(f"{'MSE only (metadata)':<25} | {pr_mse:>10.4f}") # Display the metadata-only performance
    print(f"{'Cosine only (audio)':<25} | {pr_cos:>10.4f}") # Display the audio-only performance
    print(f"{f'Hybrid (λ={best_lam})':<25} | {pr_hybrid:>10.4f}") # Display the hybrid performance (should be higher than the individual parts)

    # ── (3) Precision@K ──────────────────────────────────────────────────────
    print("\n" + "=" * 70) # Print decorative header
    print(f"(3) Precision@{K} — fraction of top-{K} sharing seed's artist ({N_SEEDS} seeds)") # Print section title
    print("=" * 70) # Print decorative header
    print(f"{'Variant':<25} | {f'P@{K}':>10}") # Print column headers
    print("-" * 45) # Print table separator line
    rng = np.random.default_rng(RANDOM_SEED)  # Reset the RNG seed to ensure all three variants evaluate the exact same set of seed tracks
    p_mse = precision_at_k(df, meta, emb, 'mse_only', 0.0, K, N_SEEDS, rng) # Test recommendation precision using metadata only
    rng = np.random.default_rng(RANDOM_SEED) # Reset RNG again
    p_cos = precision_at_k(df, meta, emb, 'cos_only', 0.0, K, N_SEEDS, rng) # Test recommendation precision using audio only
    rng = np.random.default_rng(RANDOM_SEED) # Reset RNG again
    p_hybrid = precision_at_k(df, meta, emb, 'hybrid', best_lam, K, N_SEEDS, rng) # Test recommendation precision using the hybrid system
    print(f"{'MSE only (metadata)':<25} | {p_mse:>10.4f}") # Display metadata precision
    print(f"{'Cosine only (audio)':<25} | {p_cos:>10.4f}") # Display audio precision
    print(f"{f'Hybrid (λ={best_lam})':<25} | {p_hybrid:>10.4f}") # Display hybrid precision

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70) # Print decorative header
    print("Summary") # Print section title
    print("=" * 70) # Print decorative header
    print(f"Chosen λ           : {best_lam}") # Output the final decided parameter
    print(f"Pass rate (hybrid) : {pr_hybrid:.4f}  (random baseline 0.5)") # Summarize the discrimination performance
    print(f"Precision@{K}       : {p_hybrid:.4f}  (random baseline ≈ {1/len(df):.4f})") # Summarize the recommendation performance
    print(f"Hybrid beats MSE-only by     : {(pr_hybrid - pr_mse) * 100:+.2f} pp (pass_rate)") # Show the quantitative gain over metadata alone
    print(f"Hybrid beats cosine-only by  : {(pr_hybrid - pr_cos) * 100:+.2f} pp (pass_rate)") # Show the quantitative gain over audio alone


if __name__ == "__main__": # Check if run from terminal
    main() # Run the suite