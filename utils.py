import fma.utils
import pandas as pd
import os
import tqdm


def get_dataloader(audio_dir, size, split, batch_size):
    """
    Args:
        audio_dir: The path of the folder where fma_metadata and fma_medium are
        size: small, medium, large as string
        split: train, test or split as string
        batch_size: batch size as integer

    Returns:

    """
    # Get raw audio dataset
    # Get metadata
    tracks = fma.utils.load(os.path.join(audio_dir, 'fma_metadata/tracks.csv'))
    tracks = _remove_nonexistent_tracks(tracks, os.path.join(audio_dir, f"fma_{size}"))

    subset = tracks.index[tracks['set', 'subset'] <= size]
    tracks = tracks.loc[subset]

    subset = tracks.index[tracks['set', 'split'] == split]
    tracks = tracks.loc[subset]

    if tracks.index.size == 0:
        raise ValueError(f"No tracks found for size {size} and split {split}")

    # Get labels
    labels = tracks['track', 'genre_top'].astype('category').cat.remove_unused_categories()
    labels = labels.cat.codes
    labels = pd.DataFrame(labels, index=tracks.index)
    Y = labels

    loader = fma.utils.LibrosaLoader(sampling_rate=44100)
    SampleLoader = fma.utils.build_sample_loader(os.path.join(audio_dir, f"fma_{size}"), Y, loader)
    print('Dimensionality: {}'.format(loader.shape))
    loader = SampleLoader(tracks.index, batch_size=batch_size)
    return Y, loader


def _remove_nonexistent_tracks(tracks: pd.DataFrame, audio_dir: str):
    """
    Remove tracks that do not exist in audio directory respecting the data format specified in self.file_ext
    :param tracks: DataFrame to remove non-existent tracks from
    :param audio_dir: audio directory
    """
    cache_path = os.path.expanduser("~/.cache/fma/fma_metadata/tracks_cleaned.csv")
    if os.path.isfile(cache_path):
        tracks = fma.utils.load(cache_path)
        return tracks
    tracks.reset_index(inplace=True)
    iterator = tqdm.tqdm(tracks.iterrows())
    iterator.set_description("Removing nonexistent tracks")
    counter = 0
    for idx, track in iterator:
        filename = f"{os.path.splitext(fma.utils.get_audio_path(audio_dir, track['track_id'].values[0]))[0]}.mp3"
        if not os.path.isfile(filename):
            tracks.drop(idx, inplace=True)
            counter += 1
    tracks.set_index('track_id', inplace=True)
    print(f"INFO: Removed {counter} tracks without corresponding .mp3 file.")
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tracks.to_csv(cache_path)
    return tracks
