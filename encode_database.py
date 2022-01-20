import os
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.utils.dist_utils import setup_dist_from_mpi
import torch as t
import numpy as np
from jukebox.utils.io import load_audio
from time import time
from datetime import datetime
from tqdm import tqdm


class JukeboxEncoder:
    def __init__(self):
        rank, local_rank, self.device = setup_dist_from_mpi(backend='nccl')

        model = "1b_lyrics"  # "5b_lyrics" # or "1b_lyrics"
        hps = Hyperparams()
        hps.sr = 44100
        hps.n_samples = 3 if model == '5b_lyrics' else 8
        hps.name = 'samples'
        hps.levels = 3
        hps.hop_fraction = [.5, .5, .125]
        hps.use_bottleneck = False

        self.hps = hps

        vqvae, *priors = MODELS[model]

        self.vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=1048576)), self.device)

    def encode_directory(self, source, destination, duration=30):
        # files = ["../test_music/000iSoC2f28zZJH4i12pSQ.mp3"]

        assert not os.path.exists(destination), 'Destination directory already exists.'

        os.makedirs(destination)
        os.makedirs(os.path.join(destination, 'zs'))
        os.makedirs(os.path.join(destination, 'xs_quantised'))

        files = [os.path.join(source, f) for f in os.listdir(source) if f.endswith('.mp3')]

        for f in tqdm(files):

            try:

                zs, xs_quantised = self.encode(f, duration)

                fout = os.path.basename(f).split('.')[0]
                levels = ['bottom_level', 'middle_level', 'top_level']

                xs_quantised = dict(zip(levels, xs_quantised))

                np.save(os.path.join(
                    destination, 'xs_quantised', fout),
                    xs_quantised['top_level']
                )

                zs = dict(zip(levels, zs))
                np.save(
                    os.path.join(destination, 'zs', fout),
                    zs['top_level']
                )
                """
                xs_quantised = dict(zip(levels, xs_quantised))
                np.savez(
                    os.path.join(destination, 'xs_quantised', fout),
                    top_level=xs_quantised['top_level']  # **xs_quantised
                )
                """
            except Exception as e:
                print(f"Exception for '{f}': {str(e)}")

    def load_audio(self, filepath, duration):
        s, _ = load_audio(filepath, self.hps.sr, 0, duration, resample=True, approx=False, time_base='sec', check_duration=True)
        s = s.T
        s = np.mean(s, axis=1)  # make mono
        s = t.Tensor(np.expand_dims(s, axis=[0, 2]))  # batch rep
        s = t.Tensor(s).to(self.device)
        return s

    def _encode(self, x_in, start_level=0, end_level=None):
        end_level = self.vqvae.levels if end_level is None else end_level
        xs = []
        xs_quantised = []
        zs = []
        for level in range(start_level, end_level):
            # encode
            level_encoder = self.vqvae.encoders[level]
            x_out = level_encoder(x_in)
            xs.append(x_out[-1])

            # quantise encoding
            z, x_quantised, _, _ = self.vqvae.bottleneck.level_blocks[level](x_out[-1], update_k=False)
            x_quantised = x_quantised.detach()
            zs.append(z)
            xs_quantised.append(x_quantised)

        return zs, xs_quantised

    def encode(self, filepath, duration):
        s = self.load_audio(filepath, duration)
        return self.encode_sample(s)

    def encode_sample(self, s, start_level=0, end_level=None):
        x = self.vqvae.preprocess(s)
        zs, xs_quantised = self._encode(x, start_level, end_level)
        zs = [zs_i.detach().cpu() for zs_i in zs]
        xs_quantised = [xs_q_i.detach().cpu() for xs_q_i in xs_quantised]
        return zs, xs_quantised

    def get_emb_width(self):
        return self.vqvae.bottleneck.level_blocks[0].emb_width


if __name__ == '__main__':
    source = "../temp_datasets/preview_audio_survey_playlist"
    destination = f"../temp_datasets/preview_audio_survey_playlist_encoded/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    encoder = JukeboxEncoder()
    encoder.encode_directory(source, destination, duration=30)
