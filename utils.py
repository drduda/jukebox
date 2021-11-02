import fma.utils
import pandas as pd



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
    tracks = fma.utils.load(audio_dir + '/fma_metadata/tracks.csv')

    subset = tracks.index[tracks['set', 'subset'] <= size]
    tracks = tracks.loc[subset]


    subset = tracks.index[tracks['set', 'split'] == split]
    tracks = tracks.loc[subset]

    # Get labels
    labels = tracks['track', 'genre_top'].astype('category').cat.remove_unused_categories()
    labels = labels.cat.codes
    labels = pd.DataFrame(labels, index=tracks.index)
    Y = labels

    loader = fma.utils.LibrosaLoader(sampling_rate=44100)
    SampleLoader = fma.utils.build_sample_loader(audio_dir + '/fma_' + size, Y, loader)
    print('Dimensionality: {}'.format(loader.shape))
    loader = SampleLoader(tracks.index, batch_size=batch_size)
    return Y, loader