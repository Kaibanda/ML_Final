import yt_dlp
import os
import glob
import time

def get_safe_name(track_name, artist_name):
    """
    Standardizes 'Song - Artist' into a filesystem-safe string.
    Ensures that downloaded files and dataset join-keys are identical.
    """
    return "".join([c for c in f"{track_name} - {artist_name}" if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def fetch_youtube_audio(track_name, artist_name, cache_dir="data/playback_cache"):
    """
    On-demand fetcher that finds a song on YouTube and downloads the .m4a 
    for real-time playback in the Streamlit app.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 1. Generate consistent filename
    safe_name = get_safe_name(track_name, artist_name)
    file_path = os.path.join(cache_dir, f"{safe_name}.m4a")
    
    # 2. Check Cache: Avoid redundant downloads
    if os.path.exists(file_path):
        return file_path
        
    query = f"ytsearch1:{track_name} {artist_name} audio"
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': file_path,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # We use extract_info directly with download=True as per Kai's PR
            ydl.extract_info(query, download=True)
            if os.path.exists(file_path):
                # Clean up if cache is getting too large (keep last 20)
                manage_cache_size(cache_dir)
                return file_path
    except Exception as e:
        print(f"Error fetching audio for {track_name}: {e}")
        return None
    
    return None

def manage_cache_size(cache_dir, max_files=20):
    """Keep the playback cache lean by deleting oldest files."""
    files = glob.glob(os.path.join(cache_dir, "*.m4a"))
    if len(files) > max_files:
        # Sort by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        for i in range(len(files) - max_files):
            try:
                os.remove(files[i])
            except:
                pass
