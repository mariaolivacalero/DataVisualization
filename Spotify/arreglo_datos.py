import history
from config import *

token = history.get_token(username, client_id, 
                              client_secret, redirect_uri, scope)
    
streamings = history.get_streamings()
print(f'Recovered {len(streamings)} streamings.')

#getting a list of unique tracks in our history
# Add artist names too as multiple artist can have same song name
tracks=[]
    
track_ids = history.get_saved_ids(tracks)
track_features = history.get_saved_features(track_ids.keys()) 
tracks_without_features = [track for track in track_ids.keys() if track_features.get(track) is None]
print(f"There are still {len(tracks_without_features)} tracks without features.")
path = 'MyData/features.csv'
print(len(tracks_without_features))
#connecting to spotify API to retrieve missing features

total_rows = len(track_ids)
chunk_size = total_rows // 10

# Process each chunk
for i in range(10):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size if i != 9 else total_rows

    # Extract a chunk of the original dictionary
    chunk_dict = dict(list(track_ids.items())[start_index:end_index])

print('Connecting to Spotify to extract features...')
acquired = 0
for track, idd in chunk_dict.items():  # le puedo pasar hasta 100 ids en la misma peticion 
                                       # mañana cambio el codigo 
    if idd is not None and track in tracks_without_features:
        try:
            features = history.get_api_features(idd, token)
            track_features[track] = features
            features['albumName'], features['albumID'] = history.get_album(idd, token)
            if features:
                acquired += 1
                print(f"Acquired features: {', '.join(track.split('___'))}. Total: {acquired}")
        except:
            features = None

tracks_without_features = [track for track in track_ids.keys() if track_features.get(track) is None]
print(f'Successfully recovered features of {acquired} tracks.')
if len(tracks_without_features):
    print(f'Failed to identify {len(tracks_without_features)} items. Some of these may not be tracks.')
#saving features 
features_dataframe = pd.DataFrame(track_features).T
features_dataframe.to_csv(path)
print(f'Saved features to {path}.')
