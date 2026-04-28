import yt_dlp # Import yt-dlp for downloading YouTube audio
import os # Import os for operating system and path manipulation
import glob # Import glob for Unix style pathname pattern expansion
import time # Import time for file timestamp operations

def get_safe_name(track_name, artist_name): # Define a function to sanitize strings for filesystem use
    """
    Standardizes 'Song - Artist' into a filesystem-safe string.
    Ensures that downloaded files and dataset join-keys are identical.
    """
    return "".join([c for c in f"{track_name} - {artist_name}" if c.isalpha() or c.isdigit() or c==' ']).rstrip() # Strip out all special characters except alphanumerics and spaces, then trim trailing whitespace

def fetch_youtube_audio(query_or_track, artist_name=None, cache_dir="data/playback_cache"): # Define function to download audio given a search query
    """
    On-demand fetcher that finds a song on YouTube.
    Supports both "Track - Artist" metadata and direct search queries.
    Downloads as .m4a to ensure Safari/iOS compatibility and librosa support.
    """
    os.makedirs(cache_dir, exist_ok=True) # Create the cache directory if it doesn't already exist
    
    if artist_name: # Check if a specific artist name was provided (Spotify track mode)
        safe_name = get_safe_name(query_or_track, artist_name) # Generate a safe filename using track and artist
        query = f"ytsearch1:{query_or_track} {artist_name} audio" # Construct a YouTube search query looking for the audio version
    else: # If no artist name is provided (raw search mode)
        safe_name = "".join([c for c in query_or_track if c.isalnum() or c==' ']).rstrip() # Sanitize the raw query string for the filename
        if len(safe_name) > 50: # Check if the sanitized query is too long for a filesystem name
            safe_name = safe_name[:50] # Truncate the filename to 50 characters to prevent OS errors
        query = f"ytsearch1:{query_or_track} official audio" # Construct a YouTube search query emphasizing official audio
            
    file_path = os.path.join(cache_dir, f"{safe_name}.m4a") # Construct the full absolute path for the target .m4a file
    
    if os.path.exists(file_path): # Check if the file has already been downloaded and cached
        return file_path # Return the cached file path immediately to save time and bandwidth
        
    ydl_opts = { # Configure the yt-dlp options dictionary
        'format': 'bestaudio[ext=m4a]', # Force download of the best quality audio stream that natively has an .m4a extension
        'outtmpl': file_path, # Set the output template to save exactly to our constructed file_path
        'noplaylist': True, # Ensure only a single track is downloaded, not an entire playlist
        'quiet': True, # Suppress standard output logs from yt-dlp
        'no_warnings': True, # Suppress warning messages from yt-dlp
        'extract_flat': False, # Ensure full extraction occurs so the actual file is downloaded
        'extractor_args': {'youtube': ['player_client=android']} # Masquerade as an Android client to bypass some YouTube bot protections
    }
    
    try: # Start a try-except block to handle potential download failures gracefully
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: # Initialize the yt-dlp object with our options
            ydl.extract_info(query, download=True) # Execute the search query and trigger the download
            if os.path.exists(file_path): # Verify that the expected file was successfully created on disk
                manage_cache_size(cache_dir) # Trigger the cache cleanup function to prevent infinite disk usage
                return file_path # Return the path to the newly downloaded file
    except Exception as e: # Catch any errors that occurred during the yt-dlp process
        print(f"Error fetching audio: {e}") # Print the error message for debugging purposes
        return None # Return None to indicate the download failed
    
    return None # Fallback return None if the file wasn't created despite no exceptions

def manage_cache_size(cache_dir, max_files=20): # Define function to limit the number of cached files
    """Keep the playback cache lean by deleting oldest files."""
    # Delete both .m4a and legacy .mp3 files if they exist
    files = glob.glob(os.path.join(cache_dir, "*.m4a")) + glob.glob(os.path.join(cache_dir, "*.mp3")) # Aggregate a list of all audio files currently in the cache directory
    if len(files) > max_files: # Check if the number of cached files exceeds the allowed maximum
        files.sort(key=os.path.getmtime) # Sort the files chronologically by modification time (oldest first)
        for i in range(len(files) - max_files): # Loop over the oldest files that exceed the limit
            try: # Use a try block in case the file is locked or already deleted
                os.remove(files[i]) # Delete the old file from the filesystem
            except: # Catch any deletion errors
                pass # Silently ignore deletion errors to keep the application running
