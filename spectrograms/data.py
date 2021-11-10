import os
from typing import Optional, Union, List, Dict

import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch.nn.functional as F

from fma import utils as fma_utils

from spectrograms.spectrogram_utils import gen_spec

import pytorch_lightning as pl


def build_dataset(split):
    pass


def build_dataloader(dataset):
    pass


class SpectrogramDataset(Dataset):

    def __init__(self, fma_dir, subset, n_fft, hop_length, sr, tracks, num_classes, spec_len=2582, file_ext=".wav"):
        super(SpectrogramDataset).__init__()
        self.fma_dir = fma_dir
        self.subset = subset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.tracks = tracks
        self.labels = tracks.set_index('track_id')['track', 'genre_top_id']
        self.num_classes = num_classes
        self.spec_len = spec_len
        self.file_ext = file_ext

        self.duration_max = self.tracks['track', 'duration'].max()

        self.audio_dir = os.path.join(self.fma_dir, f"fma_{self.subset}")

        self.remove_nonexistent_tracks()

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index) -> T_co:
        track = self.tracks.loc[index]
        filename = f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track['track_id'].values[0]))[0]}{self.file_ext}"
        mel, sr = gen_spec(filename, self.n_fft, self.hop_length, self.sr)

        if sr != self.sr:
            raise ValueError("The output sampling rate does not match the requested one.")
        mel = torch.from_numpy(mel)

        # TODO: solve dynamically
        # ugly workaround
        mel = mel[:, :self.spec_len]
        if mel.shape[1] < self.spec_len:
            # pad with zeros
            mel = torch.cat((mel, torch.zeros(mel.shape[0], self.spec_len - mel.shape[1])), dim=1)

        assert mel.shape == (128, self.spec_len)

        y = self.labels[track['track_id']]
        y = torch.from_numpy(y.values)
        y = F.one_hot(y, self.num_classes)

        return mel, y

    def remove_nonexistent_tracks(self):
        """
        iterate over self.tracks and remove entries where the corresponding file does not exist
        """
        iterator = tqdm.tqdm(self.tracks.iterrows())
        iterator.set_description("Removing nonexistent tracks")
        for idx, track in iterator:
            filename = f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track['track_id'].values[0]))[0]}{self.file_ext}"
            if not os.path.isfile(filename):
                self.tracks.drop(idx, inplace=True)


class FmaSpectrogramGenreDataModule(pl.LightningDataModule):
    """
    class lightning datamodule that converts fma data into spectrograms and provides the spectrograms as a dataset
    """
    def __init__(self, fma_dir: str, subset: str, n_fft: int, hop_length: int, sr: int, batch_size: int = 32, file_ext: str = ".wav"):
        super(FmaSpectrogramGenreDataModule, self).__init__()
        self.data_dir = fma_dir
        self.subset = subset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self. sr = sr
        self.batch_size = batch_size
        self.file_ext = file_ext

        # load metadata
        self.tracks = fma_utils.load(os.path.join(self.data_dir, 'fma_metadata/tracks.csv'))
        self.tracks = self.tracks[self.tracks['set', 'subset'] <= self.subset]

        # generate label IDs in new column genre_top_id
        self.tracks = self.tracks.dropna(subset=[('track', 'genre_top')])
        labels = self.tracks['track', 'genre_top']
        label_id_mapping = {label: label_id for label_id, label in enumerate(labels.cat.categories)}
        self.tracks['track', 'genre_top_id'] = self.tracks['track', 'genre_top'].apply(
            lambda x: label_id_mapping[x]
        ).astype('int64')
        self.num_classes = len(labels.cat.categories)

        self.spec_ds_train = None
        self.spec_ds_val = None
        self.spec_ds_test = None

    def setup(self, stage: Optional[str] = None) -> None:

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            train_tracks = self.tracks[self.tracks['set', 'split'] == 'training'].copy().reset_index()

            self.spec_ds_train = SpectrogramDataset(
                self.data_dir,
                self.subset,
                self.n_fft,
                self.hop_length,
                self.sr,
                train_tracks,
                self.num_classes,
                2582
            )

            val_tracks = self.tracks[self.tracks['set', 'split'] == 'validation'].copy().reset_index()

            self.spec_ds_train = SpectrogramDataset(
                self.data_dir,
                self.subset,
                self.n_fft,
                self.hop_length,
                self.sr,
                val_tracks,
                self.num_classes,
                2582
            )

        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:

            test_tracks = self.tracks[self.tracks['set', 'split'] == 'test'].copy().reset_index()

            self.spec_ds_test = SpectrogramDataset(
                self.data_dir,
                self.subset,
                self.n_fft,
                self.hop_length,
                self.sr,
                test_tracks,
                self.num_classes,
                2582
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_test, batch_size=self.batch_size, shuffle=False)
