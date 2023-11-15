import os
import spotipy.util as util

def check_cached_token():
    # Get the cache path from spotipy's util module
    cache_path = util._get_cache_path()

    # Check if the cache file exists
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as cache_file:
            # Read and print the contents of the cache file
            cache_contents = cache_file.read()
            print(f"Contents of the cache file at {cache_path}:\n{cache_contents}")
    else:
        print(f"No cache file found at {cache_path}")

# Call the function to check for cached tokens
check_cached_token()