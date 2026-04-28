import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import yt_dlp

PROJECT_ROOT = Path(__file__).parent.parent # Define the absolute path to the project's root directory

try: # Start a try block to handle relative imports depending on execution context
    from src.audio_utils import get_safe_name # Import the standard function to sanitize song filenames
except ImportError: # Catch import errors if script is run directly from the src directory
    from audio_utils import get_safe_name # Fallback to a local relative import

BATCH_SIZE = 1000 # Define the number of songs to attempt downloading in a single batch execution
SLEEP_BETWEEN_MIN = 5.0 # Define the minimum number of seconds to sleep between successful downloads
SLEEP_BETWEEN_MAX = 10.0 # Define the maximum number of seconds to sleep between successful downloads (randomized)
SLEEP_ON_ERROR = 15.0 # Define a longer static sleep penalty if YouTube blocks or fails a request
OUTPUT_DIR = PROJECT_ROOT / "data/audio_files" # Set the target directory where the raw .m4a files will be saved
MANIFEST_PATH = PROJECT_ROOT / "data/dataset/download_manifest.csv" # Set the path to the tracking file that manages what needs to be downloaded
FAIL_LOG_PATH = PROJECT_ROOT / "data/dataset/download_failures.jsonl" # Set the path to log permanent failures (like region blocks or missing songs)


def build_or_load_manifest(full_csv: Path, existing_csv: Path) -> pd.DataFrame: # Define function to initialize or retrieve the download queue
    """Create a stable shuffled manifest once, then reuse it across batches."""
    if MANIFEST_PATH.exists(): # Check if the manifest already exists from a previous run
        return pd.read_csv(MANIFEST_PATH) # Load and return the existing manifest to resume where we left off

    df_full = pd.read_csv(full_csv) # Load the complete list of 30,000+ Spotify songs
    existing_ids = set(pd.read_csv(existing_csv)["track_id"].tolist()) # Load the set of track IDs we've already successfully processed in the past

    remaining = ( # Build a new dataframe containing only the songs that still need to be downloaded
        df_full[~df_full["track_id"].isin(existing_ids)] # Filter out the already processed IDs
        .sample(frac=1, random_state=42) # Shuffle the remaining rows with a fixed seed to randomize the download order without losing track
        .reset_index(drop=True) # Reset the pandas index after shuffling
        .copy() # Ensure we have an independent copy of the dataframe
    )
    remaining["safe_name"] = remaining.apply( # Create a new column with the filesystem-safe name for each track
        lambda r: get_safe_name(r["track_name"], r["track_artist"]), axis=1 # Apply the sanitization logic row by row
    )
    remaining.to_csv(MANIFEST_PATH, index=False) # Save the newly created manifest to disk for future runs
    return remaining # Return the manifest dataframe


def already_downloaded(manifest: pd.DataFrame, processed_csv: Path) -> set[str]: # Define function to aggregate all tracks that shouldn't be downloaded again
    """Return safe_names already in processed_songs.csv or present as files or permanently failed."""
    done = set() # Initialize an empty set to store the safe_names of completed or failed tracks

    # Check processed_songs.csv via manifest mapping
    if processed_csv.exists(): # Verify if the master processed CSV exists
        processed_ids = set(pd.read_csv(processed_csv)["track_id"].tolist()) # Get all track IDs that have already been fully feature-extracted
        done |= set(manifest.loc[manifest["track_id"].isin(processed_ids), "safe_name"].tolist()) # Map those IDs to safe_names and add them to the done set

    # Also check actual files (current batch in progress)
    if OUTPUT_DIR.exists(): # Verify if the audio output directory exists
        for p in OUTPUT_DIR.iterdir(): # Iterate over all files currently sitting in the output directory
            if p.is_file(): # Ensure the path is actually a file, not a subdirectory
                done.add(p.stem) # Add the filename (without extension) to the done set

    # Skip previously failed tracks (video unavailable, not found, etc.)
    if FAIL_LOG_PATH.exists(): # Verify if the failure log file exists
        try: # Use a try block to handle potential JSON parsing errors
            with open(FAIL_LOG_PATH, "r", encoding="utf-8") as f: # Open the JSONL file in read mode
                for line in f: # Iterate through each line (each line is a JSON object)
                    try: # Inner try block for individual line parsing
                        entry = json.loads(line) # Parse the JSON string into a Python dictionary
                        safe_name = get_safe_name(entry["track_name"], entry["artist_name"]) # Reconstruct the safe_name from the logged track and artist
                        done.add(safe_name) # Add the permanently failed track to the done set so we don't retry it
                    except (json.JSONDecodeError, KeyError): # Catch formatting errors on specific lines
                        continue # Skip corrupted log lines and continue reading
        except Exception: # Catch broader filesystem errors
            pass # Ignore the error and proceed without the failure cache

    return done # Return the complete set of safe_names to skip


def log_failure(track_name: str, artist_name: str, query: str, error: str) -> None: # Define function to record permanent download errors
    FAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory for the log file exists
    with open(FAIL_LOG_PATH, "a", encoding="utf-8") as f: # Open the failure log file in append mode
        f.write( # Write the serialized JSON string to the file
            json.dumps( # Convert the error details into a JSON formatted string
                { # Construct the dictionary to log
                    "track_name": track_name, # Log the raw track name
                    "artist_name": artist_name, # Log the raw artist name
                    "query": query, # Log the exact YouTube search query that failed
                    "error": error, # Log the explicit error message returned by yt-dlp
                },
                ensure_ascii=False, # Allow unicode characters to be written natively
            )
            + "\n" # Append a newline character to maintain the JSONL format
        )


def find_downloaded_file(video_id: str) -> Path | None: # Define function to locate the file downloaded by yt-dlp
    """Find the actual file yt-dlp/ffmpeg produced for a given video_id."""
    candidates = list(OUTPUT_DIR.glob(f"{video_id}.*")) # Search the output directory for any files starting with the YouTube video ID
    if not candidates: # If no files matched the video ID
        return None # Return None indicating the file wasn't saved properly

    preferred_suffixes = [".m4a", ".mp3", ".webm", ".opus"] # Define a hierarchy of preferred audio extensions
    candidates.sort(key=lambda p: preferred_suffixes.index(p.suffix) if p.suffix in preferred_suffixes else 999) # Sort the found files to prioritize m4a/mp3 over obscure formats
    return candidates[0] # Return the path to the most desirable audio file


def download_one(ydl: yt_dlp.YoutubeDL, track_name: str, artist_name: str) -> bool: # Define the core function that attempts to download a single track
    safe_name = get_safe_name(track_name, artist_name) # Generate the standardized filesystem name for the target track

    for existing in OUTPUT_DIR.glob(f"{safe_name}.*"): # Check if a file with this safe_name already exists with any extension
        if existing.is_file(): # Verify it's a file
            return True # Return True early because the track is already safely downloaded

    query = f"ytsearch1:{track_name} {artist_name} audio" # Construct the YouTube search query to find the first matching audio video
    try: # Start try block to handle yt-dlp exceptions (like bot blocks or unavailable videos)
        info = ydl.extract_info(query, download=True) # Execute the search and force the download of the best result
        entries = info.get("entries", []) # Retrieve the list of video entries from the search results
        if not entries: # Check if the search yielded zero results
            log_failure(track_name, artist_name, query, "No search entries returned") # Log the failure permanently
            return False # Return False indicating the download failed

        video_id = entries[0]["id"] # Extract the YouTube video ID of the top search result
        downloaded = find_downloaded_file(video_id) # Locate the actual audio file that yt-dlp saved to disk
        if downloaded is None: # Check if yt-dlp failed to save the file despite not throwing an error
            log_failure(track_name, artist_name, query, f"Downloaded file for video_id={video_id} not found") # Log the missing file anomaly
            return False # Return False indicating failure

        final_path = OUTPUT_DIR / f"{safe_name}{downloaded.suffix}" # Construct the final desired path using our standard safe_name naming convention

        if final_path.exists(): # Handle the edge case where the target file name already exists
            downloaded.unlink(missing_ok=True) # Delete the newly downloaded file to prevent duplicates
        else: # If the target filename is available
            downloaded.rename(final_path) # Rename the downloaded file from the YouTube ID to our standardized safe_name

        return True # Return True indicating successful download and renaming

    except Exception as e: # Catch any exceptions thrown by the yt-dlp extraction process
        error_msg = repr(e).lower() # Convert the error to a lowercase string for easier keyword matching
        
        # Check if it's a transient error (rate limit, bot ban, network failure)
        is_transient = False # Initialize a flag to track if the error is temporary
        transient_keywords = [ # Define a list of error keywords that indicate YouTube is temporarily blocking us
            "bot", # Bot detection trigger
            "429", # HTTP status code for Too Many Requests
            "too many requests", # Explicit text for rate limiting
            "rate-limit", # Explicit text for rate limiting
            "ssl", # SSL handshake failure
            "timed out", # Network timeout
            "connection reset", # Network connection dropped
            "unavailable" # Temporary video unavailability
        ]
        
        for k in transient_keywords: # Iterate through our list of known temporary errors
            if k in error_msg: # If the error message contains the keyword
                is_transient = True # Flag the error as temporary
                break # Stop searching keywords
                
        # Handle "Sign in to confirm" carefully: "bot" is transient, "age" is permanent
        if "sign in to confirm" in error_msg and "age" not in error_msg: # YouTube sometimes asks for sign-in as a bot check, unless it's explicitly age-restricted
            is_transient = True # Flag this specific non-age sign-in request as a temporary block
            
        if not is_transient: # If the error was not flagged as temporary (meaning the song is fundamentally unavailable)
            # Only permanently log legitimate failures (Age restricted, video not found, etc.)
            log_failure(track_name, artist_name, query, repr(e)) # Write the error to the failure log so we never retry it
            
        return False # Return False indicating the download attempt failed


def main() -> None: # Define the main execution block for the batch downloading script
    parser = argparse.ArgumentParser(description="Batch YouTube audio downloader") # Initialize the argument parser
    parser.add_argument("--batch", type=int, required=True, help="Batch index (0-based, 1000 songs each)") # Require a --batch integer argument to partition the workload
    args = parser.parse_args() # Parse the provided command-line arguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the audio output directory exists before starting

    full_csv = PROJECT_ROOT / "data/dataset/spotify_songs_full.csv" # Define the path to the original full Spotify dataset
    existing_csv = PROJECT_ROOT / "data/dataset/processed_songs.csv" # Define the path to the dataset of already processed songs

    manifest = build_or_load_manifest(full_csv, existing_csv) # Initialize or load the stateful download queue

    start = args.batch * BATCH_SIZE # Calculate the starting index for this specific batch partition
    batch = manifest.iloc[start:start + BATCH_SIZE].copy() # Slice the manifest dataframe to get the assigned chunk of songs

    if batch.empty: # Check if this batch partition falls completely outside the remaining rows
        print(f"Batch {args.batch}: nothing left to download.") # Inform the user there is no work to do
        return # Exit the script early

    done = already_downloaded(manifest, existing_csv) # Dynamically compile the set of safe_names we shouldn't attempt
    todo = batch[~batch["safe_name"].isin(done)].copy() # Filter the current batch down to only those songs that aren't in the 'done' list

    print(f"Batch {args.batch}: {len(batch)} songs | {len(todo)} to download | {len(batch) - len(todo)} already done") # Print batch statistics
    print(f"Output: {OUTPUT_DIR}\n") # Print the target output directory

    success, failed = 0, 0 # Initialize counters for tracking the pass/fail rate of the current run

    ydl_opts = { # Configure the yt-dlp downloader options dictionary
        "format": "bestaudio[ext=m4a]", # Request the best quality audio stream that natively has an .m4a extension
        "outtmpl": str(OUTPUT_DIR / "%(id)s.%(ext)s"), # Temporarily save the file using its YouTube ID to avoid special character issues during download
        "noplaylist": True, # Ensure only single videos are downloaded, avoiding entire playlist rips
        "quiet": True, # Suppress the standard noisy yt-dlp console output
        "no_warnings": True, # Suppress yt-dlp warning messages
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # Initialize the yt-dlp downloader context manager
        for i, (_, row) in enumerate(todo.iterrows(), 1): # Iterate sequentially over every song that needs to be downloaded
            ok = download_one(ydl, row["track_name"], row["track_artist"]) # Attempt to find and download the target track
            print(f"[{i:4d}/{len(todo)}] {'✓' if ok else '✗'}  {row['track_name']} — {row['track_artist']}") # Print a progress update with a success/fail icon

            if ok: # If the download was successful
                success += 1 # Increment the success counter
                time.sleep(random.uniform(SLEEP_BETWEEN_MIN, SLEEP_BETWEEN_MAX)) # Sleep for a randomized duration to mimic human browsing and avoid bot detection
            else: # If the download failed (either transiently or permanently)
                failed += 1 # Increment the failure counter
                time.sleep(SLEEP_ON_ERROR) # Sleep for a fixed, longer duration to let potential rate limits cool down

    print(f"\nBatch {args.batch} done: {success} success / {failed} failed") # Print final batch statistics
    print(f"Total audio files now: {len([p for p in OUTPUT_DIR.iterdir() if p.is_file()])}") # Print the total count of accumulated audio files
    print("\nNext steps:") # Print helpful instructions for what to do after downloading finishes
    print("  python src/audio_feature_extractor.py") # Suggest running the feature extractor
    print("  python src/pca.py") # Suggest generating the PCA visuals
    print("  python src/clustering.py") # Suggest running the K-Means clustering
    print("  python src/build_master_dataset.py") # Suggest fusing the final dataset


if __name__ == "__main__": # Check if the script is being executed directly from the terminal
    main() # Run the main batch processing pipeline
