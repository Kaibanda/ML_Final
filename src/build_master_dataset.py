import pandas as pd
import numpy as np
from pathlib import Path

try: # Start a try block to handle potential import path issues depending on where the script is run from
    from src.audio_utils import get_safe_name # Attempt to import the safe string generator from the src package module
except ImportError: # Catch the ImportError if the script is run from inside the src directory directly
    from audio_utils import get_safe_name # Fallback to a relative import if the src prefix fails

PROJECT_ROOT = Path(__file__).parent.parent # Define the project root by resolving two directories up from the current file


def main(): # Define the main execution function for building the dataset
    """
    Fuses textual metadata with high-dimensional audio embeddings.
    
    This process is critical for Multimodal Retrieval:
    1. It aligns Spotify's subjective features (danceability) with Librosa's objective features (MFCC).
    2. It uses 'safe_name' (Artist_Title slug) as a robust join key since raw track_ids 
       might differ between the CSV and the YouTube crawler results.
    3. It enforces data integrity by dropping duplicates, ensuring every vector in the
       master dataset corresponds to a unique acoustic fingerprint.
    """
    processed_csv = PROJECT_ROOT / 'data/dataset/processed_songs.csv' # Define the path to the CSV file containing Spotify metadata
    embedding_parquet = PROJECT_ROOT / 'data/embeddings/audio_features.parquet' # Define the path to the Parquet file containing Librosa embeddings
    output_master = PROJECT_ROOT / 'data/dataset/master_music_data.parquet' # Define the target path for the final merged dataset

    print("Building Master Dataset...") # Print a status message indicating the start of the build process

    if not processed_csv.exists(): # Check if the required metadata CSV file exists
        print(f"Error: Missing {processed_csv}") # Print an error message if the metadata file is missing
        return # Exit the function early because a critical file is missing
    if not embedding_parquet.exists(): # Check if the required audio embeddings Parquet file exists
        print(f"Error: Missing {embedding_parquet}") # Print an error message if the embeddings file is missing
        return # Exit the function early because a critical file is missing

    # Load both modalities
    df_meta = pd.read_csv(processed_csv) # Load the Spotify metadata into a pandas DataFrame
    df_emb = pd.read_parquet(embedding_parquet) # Load the audio feature embeddings into a pandas DataFrame

    # Sanitize keys for joining
    df_meta['safe_name'] = df_meta.apply( # Create a new column in the metadata DataFrame for the join key
        lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1 # Apply the safe name generation logic to each row's track and artist
    )
    df_emb = df_emb.rename(columns={'track_id': 'safe_name'}) # Rename the 'track_id' column in the embeddings DataFrame to 'safe_name' to match

    # Inner join captures only songs for which we have both Meta and Audio
    master_df = pd.merge(df_meta, df_emb, on='safe_name', how='inner') # Perform an inner join on the 'safe_name' key to fuse the modalities
    
    # Drop duplicates to prevent 'Multiple Candidates' errors during retrieval
    master_df = master_df.drop_duplicates(subset=['safe_name']).reset_index(drop=True) # Remove any duplicate rows based on the join key and reset the index

    if master_df.empty: # Check if the resulting merged DataFrame has zero rows
        print("Error: Join produced 0 rows. Check safe_name keys.") # Print an error indicating the join failed to match any records
        return # Exit the function early because the resulting dataset is empty

    print(f"Master Dataset: {len(master_df)} songs, {master_df.shape[1]} columns.") # Print a summary of the successfully built master dataset
    master_df.to_parquet(output_master) # Save the final merged DataFrame to disk as a highly efficient Parquet file
    print(f"Saved to {output_master}") # Print a confirmation message showing where the file was saved


if __name__ == "__main__": # Check if the script is being executed as the main program
    main() # Call the main function to execute the dataset building logic
