import fire
import torch
import tqdm

import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
import fma.utils as utils
import pandas as pd

# The maximum token length of a 30s snippet
SIZE = 11000

def run(target, size, audio_dir):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Get models
    model = "1b_lyrics"
    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = 3 if model == '5b_lyrics' else 8
    hps.name = 'samples'
    chunk_size = 16 if model == "5b_lyrics" else 32
    max_batch_size = 3 if model == "5b_lyrics" else 16
    hps.levels = 3
    hps.hop_fraction = [.5, .5, .125]
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)

    # Get raw audio dataset
    # Get metadata
    tracks = utils.load('/home/marko/data/fma_metadata/tracks.csv')
    subset = tracks.index[tracks['set', 'subset'] <= 'small']
    tracks = tracks.loc[subset]
    if target is "genre":
        # Get labels
        labels_onehot = tracks['track', 'genre_top'].astype('category').cat.remove_unused_categories()
        labels_onehot = labels_onehot.cat.codes
        labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)
        Y = labels_onehot[0]
    else:
        raise ValueError("Target unknown")

    # Make the arrays
    tracks_as_tokens = torch.zeros((len(tracks), SIZE), dtype=torch.int16)
    tracks_length = torch.zeros(len(tracks), dtype=torch.int16)

    for idx, (track_idx, row) in tqdm.tqdm(enumerate(tracks.iterrows())):

        #Get actual track
        audio_path = utils.get_audio_path(audio_dir+'/fma_'+size, track_idx)
        track, sr = librosa.load(audio_path, sr=44100)

        with t.no_grad():
            # Technical adjustments of the input
            track = torch.from_numpy(track).to(device)
            track = t.unsqueeze(track, 0)
            track = t.unsqueeze(track, -1)

            # Feed in Jukebox + technical adjustments
            tokens = vqvae.encode(track, start_level=2, end_level=3, bs_chunks=track.shape[0])[0]
            tokens = torch.squeeze(tokens)

            # Put into the array
            tracks_as_tokens[idx, :len(tokens)] = tokens
            tracks_length[idx] = len(tokens)


    saving_path = "tokens_ds_target_" + target + "_size_" + size + ".pt"
    torch.save((tracks_as_tokens, tracks_length, Y), saving_path)
    print("Saved as " + saving_path)


if __name__ == '__main__':
    fire.Fire(run)
