import fire
import torch
import tqdm
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
import jukebox
import torch as t
import librosa
import os
from IPython.display import Audio
import utils
import fma.utils
import pandas as pd

# The maximum token length of a 30s snippet
SIZE = 11000

def get_vq_vae(device):
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
    return make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), device)

def run(size, audio_dir, batch_size):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    vqvae = get_vq_vae(device)

    for split in ['training', 'validation', 'test']:
        Y, loader = utils.get_dataloader(audio_dir, size, split, batch_size)

        # Make the arrays
        tracks_as_tokens = torch.zeros((len(Y), SIZE), dtype=torch.int16)
        tracks_length = torch.zeros(len(Y), dtype=torch.int16)

        idx = 0
        for x, _ in tqdm.tqdm(loader):
            with torch.no_grad():
                x = t.from_numpy(x).to(device)
                x = t.unsqueeze(x, -1)

                # Feed in Jukebox + technical adjustments
                tokens = vqvae.encode(x, start_level=2, end_level=3, bs_chunks=x.shape[0])[0]
                tokens = torch.squeeze(tokens)

                tracks_as_tokens[[range(idx, idx+batch_size)], :tokens.shape[1]] = tokens.short()
                tracks_length[range(idx, idx+batch_size)] = tokens.shape[1]
                idx = idx + batch_size

                # Save every 4th batch
                if (idx/batch_size) % 4 == 0:
                    tracks_as_tokens.length = idx
                    saving_path = "tokens_ds_size_%s_split_%s.pt" % (size, split)
                    torch.save((tracks_as_tokens, tracks_length, Y[0]), saving_path)

        tracks_as_tokens.length = idx
        saving_path = "tokens_ds_size_%s_split_%s.pt" % (size, split)
        torch.save((tracks_as_tokens, tracks_length, Y[0]), saving_path)


if __name__ == '__main__':
    fire.Fire(run)
