import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch.nn.functional as F

from fma import utils as fma_utils

from spectrograms.spectrogram_utils import gen_spec


def build_dataset(split):
    pass


def build_dataloader(dataset):
    pass


class SpectrogramDataset(Dataset):

    def __init__(self, fma_dir, subset, n_fft, hop_length, sr, labels, num_classes):
        super(SpectrogramDataset).__init__()
        self.fma_dir = fma_dir
        self.subset = subset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.labels = labels
        self.num_classes = num_classes

        self.tracks = fma_utils.load(os.path.join(self.fma_dir, 'fma_metadata/tracks.csv'))
        self.tracks = self.tracks[self.tracks['set', 'subset'] <= self.subset]
        self.tracks = self.tracks.loc[labels.index]
        self.tracks.reset_index(inplace=True)

        self.audio_dir = os.path.join(self.fma_dir, f"fma_{self.subset}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index) -> T_co:
        track = self.tracks.loc[index]
        filename = f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track['track_id'].values[0]))[0]}.wav"
        mel, sr = gen_spec(filename, self.n_fft, self.hop_length, self.sr)
        if sr != self.sr:
            raise ValueError("The output sampling rate does not match the requested one.")
        mel = torch.from_numpy(mel)

        y = self.labels[track['track_id']]
        y = torch.from_numpy(y.values)
        y = F.one_hot(y, self.num_classes)

        return mel, y
