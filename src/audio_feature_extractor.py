import os
import glob
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent # Resolve the absolute path to the root directory of the project

def extract_audio_features(audio_path: str) -> np.ndarray: # Define a function that takes an audio file path and returns a numpy array
    """
    Extracts a 56-dimensional feature vector from audio using Librosa.
    
    Dimensions:
    - MFCC (26D): mean + std. Captures timbre (the 'texture' of the sound).
    - Chroma (12D): mean. Captures harmonic/musical key information.
    - Spectral Centroid/BW/Rolloff (6D): Captures 'brightness' and 'richness'.
    - Zero Crossing Rate/RMS (4D): Captures percussiveness and loudness.
    - Tempo (1D): Beats per minute (BPM).
    - Spectral Contrast (7D): foreground vs background energy separation.
    """
    # Load audio (mono, 22.05kHz) - limited to first 60s for consistency
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60) # Load up to 60 seconds of audio, downsampled to 22050Hz and converted to mono

    features = [] # Initialize an empty list to accumulate the extracted audio features

    # 1. MFCCs - The most common feature in speech/music recognition (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # Compute 13 Mel-frequency cepstral coefficients (MFCCs)
    features.extend(np.mean(mfcc, axis=1).tolist()) # Append the mean of each MFCC over time (13 values)
    features.extend(np.std(mfcc, axis=1).tolist()) # Append the standard deviation of each MFCC over time (13 values)

    # 2. Chroma - Maps energy into 12 semitones of the musical octave
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # Compute a chromagram from a waveform or power spectrogram
    features.extend(np.mean(chroma, axis=1).tolist()) # Append the mean of the 12 chroma bins (12 values)

    # 3. Spectral Features - Shape of the frequency distribution
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr) # Compute the spectral centroid (center of mass of the spectrum)
    features.append(float(np.mean(spectral_centroid))) # Append the mean spectral centroid (1 value)
    features.append(float(np.std(spectral_centroid))) # Append the standard deviation of the spectral centroid (1 value)

    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) # Compute p'th-order spectral bandwidth
    features.append(float(np.mean(spectral_bw))) # Append the mean spectral bandwidth (1 value)
    features.append(float(np.std(spectral_bw))) # Append the standard deviation of spectral bandwidth (1 value)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) # Compute roll-off frequency (frequency below which a certain percentage of energy lies)
    features.append(float(np.mean(rolloff))) # Append the mean spectral rolloff (1 value)
    features.append(float(np.std(rolloff))) # Append the standard deviation of spectral rolloff (1 value)

    # 4. Temporal Features - Percussiveness and Loudness
    zcr = librosa.feature.zero_crossing_rate(y=y) # Compute the zero-crossing rate of an audio time series
    features.append(float(np.mean(zcr))) # Append the mean zero-crossing rate (1 value)
    features.append(float(np.std(zcr))) # Append the standard deviation of zero-crossing rate (1 value)

    rms = librosa.feature.rms(y=y) # Compute root-mean-square (RMS) energy for each frame
    features.append(float(np.mean(rms))) # Append the mean RMS energy (1 value)
    features.append(float(np.std(rms))) # Append the standard deviation of RMS energy (1 value)

    # 5. Rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr) # Compute the onset strength envelope
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr) # Estimate the tempo (BPM)
    features.append(float(tempo[0])) # Append the estimated tempo (1 value)

    # 6. Spectral Contrast - Separation of frequency peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6) # Compute spectral contrast across 6 frequency bands (returns 7 bands)
    features.extend(np.mean(contrast, axis=1).tolist()) # Append the mean spectral contrast for each band (7 values)

    return np.array(features, dtype=np.float32) # Convert the accumulated feature list into a 32-bit float numpy array and return


def main(): # Define the main execution function
    input_dir = PROJECT_ROOT / "data/audio_files" # Define the directory containing the downloaded raw audio files
    embed_dir = PROJECT_ROOT / "data/embeddings" # Define the output directory for the extracted embeddings
    embed_dir.mkdir(parents=True, exist_ok=True) # Create the embeddings directory and any necessary parent directories

    # Gather all audio files
    audio_files = ( # Create a list of all matching audio file paths
        glob.glob(str(input_dir / "*.m4a")) + # Find all files with the .m4a extension
        glob.glob(str(input_dir / "*.mp3")) # Find all files with the legacy .mp3 extension
    )

    if not audio_files: # Check if the audio_files list is empty
        print(f"No audio files found in '{input_dir}'. Please run fetch_youtube_audio.py first.") # Print an error message prompting data download
        return # Exit the main function early

    # Skip files already in parquet
    out_path = embed_dir / "audio_features.parquet" # Define the path for the output parquet database
    existing_ids = set() # Initialize an empty set to store IDs of already processed tracks
    if out_path.exists(): # Check if the output parquet file already exists
        existing_df = pd.read_parquet(out_path) # Load the existing parquet file into a pandas DataFrame
        existing_ids = set(existing_df["track_id"].tolist()) # Extract the existing track IDs into a fast-lookup set
        print(f"Existing parquet: {len(existing_ids)} tracks. Skipping these.") # Print the number of tracks that will be skipped

    todo_files = [p for p in audio_files if os.path.splitext(os.path.basename(p))[0] not in existing_ids] # Filter the file list to keep only unprocessed tracks
    print(f"Found {len(audio_files)} audio files total, {len(todo_files)} new to process.") # Print the summary of total vs new tracks

    if not todo_files: # Check if there are no new files left to process
        print("Nothing new to extract.") # Print a message indicating no feature extraction is needed
        # Still update processed_songs.csv below so newly downloaded tracks are registered
        all_embeddings = [] # Initialize empty embeddings list since no processing happens
        track_ids = list(existing_ids) # Set track_ids to the existing set for downstream CSV updates
    else:
        all_embeddings = [] # Initialize a list to hold all newly extracted embeddings
        track_ids = [] # Initialize a list to hold the corresponding track IDs

        for i, audio_path in enumerate(todo_files, 1): # Iterate over each unprocessed file, keeping a count
            filename = os.path.basename(audio_path) # Extract just the filename from the full path
            track_id = os.path.splitext(filename)[0] # Strip the extension to get the unique track ID (safe_name)

            try:
                print(f"[{i}/{len(todo_files)}] Extracting: {filename}") # Print progress indicator
                features = extract_audio_features(audio_path) # Call the extraction function to get the 56D vector
                all_embeddings.append(features.tolist()) # Append the feature vector to the master list
                track_ids.append(track_id) # Append the track ID to the tracking list
            except Exception as e: # Catch any exceptions that occur during feature extraction (e.g. corrupt audio)
                print(f"    -> SKIPPED {filename}: {e}") # Print the error and skip this track

        if all_embeddings: # Check if any new embeddings were successfully extracted
            feature_dim = len(all_embeddings[0]) # Get the dimensionality of the extracted vectors (should be 56)
            print(f"\nExtracted {feature_dim}D features for {len(all_embeddings)} new songs.") # Print a summary of the extraction process

            new_df = pd.DataFrame({"track_id": track_ids, "embedding": all_embeddings}) # Create a DataFrame with the new IDs and embeddings

            # Append to existing parquet
            if out_path.exists(): # Re-check if the parquet file exists to merge with existing data
                existing_df = pd.read_parquet(out_path) # Read the existing parquet data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["track_id"]) # Concatenate and drop any accidental duplicates
            else:
                combined_df = new_df # If no existing file, the combined DataFrame is just the new data

            combined_df.to_parquet(out_path) # Save the full combined dataset back to the parquet file
            print(f"Saved {len(combined_df)} total tracks to {out_path}") # Print the final count of saved tracks

            # For downstream CSV update, use all track_ids including existing ones
            track_ids = combined_df["track_id"].tolist() # Update the track_ids variable to include all historical and new IDs
        else:
            track_ids = list(existing_ids) # Fallback to existing IDs if the extraction loop yielded no valid data

    # Update spotify_songs.csv with newly downloaded tracks
    existing_csv = PROJECT_ROOT / "data/dataset/processed_songs.csv" # Define path to the processed metadata CSV
    manifest_path = PROJECT_ROOT / "data/dataset/download_manifest.csv" # Define path to the download manifest linking Spotify IDs to safe_names

    if not manifest_path.exists(): # Check if the manifest file exists
        print("No manifest found, skipping processed_songs.csv update.") # Print warning and skip updating metadata
        return # Exit the main function early

    existing_df = pd.read_csv(existing_csv) if existing_csv.exists() else pd.DataFrame() # Load the existing processed metadata if available, else empty DataFrame
    existing_ids = set(existing_df["track_id"].tolist()) if not existing_df.empty else set() # Extract existing Spotify track IDs to avoid duplicating rows

    manifest = pd.read_csv(manifest_path) # Load the download manifest containing mapping information
    full_csv = PROJECT_ROOT / "data/dataset/spotify_songs_full.csv" # Define the path to the original raw Kaggle dataset
    df_full = pd.read_csv(full_csv) # Load the entire raw dataset

    extracted_safe_names = set(track_ids) # Convert the list of all processed safe_names into a fast-lookup set
    matched = manifest[manifest["safe_name"].isin(extracted_safe_names)] # Filter the manifest to only include tracks we have features for
    matched_spotify_ids = set(matched["track_id"].tolist()) # Extract the corresponding Spotify track IDs for those matched rows

    new_tracks = df_full[df_full["track_id"].isin(matched_spotify_ids) & ~df_full["track_id"].isin(existing_ids)] # Find rows in the raw dataset that are now processed but aren't in the processed CSV
    if not new_tracks.empty: # Check if there are actually new metadata rows to add
        updated_df = pd.concat([existing_df, new_tracks], ignore_index=True).drop_duplicates(subset=["track_id"]) # Concatenate and drop duplicates by Spotify ID
        updated_df.to_csv(existing_csv, index=False) # Save the updated metadata back to the processed CSV without the index column
        print(f"Updated {existing_csv} with {len(new_tracks)} new tracks.") # Print a summary of how many new metadata rows were added
    else:
        print(f"No new tracks to add to {existing_csv}") # Print a message if the processed CSV was already up-to-date


if __name__ == "__main__": # Standard Python boilerplate to execute main() only if script is run directly
    main() # Call the main execution function

