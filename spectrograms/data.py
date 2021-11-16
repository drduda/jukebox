import os
from typing import Optional, Union, List, Dict

import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch.nn.functional as F
import librosa
import numpy as np
import pytorch_lightning as pl

from fma import utils as fma_utils

from spectrograms.spectrogram_utils import gen_spec


class SpectrogramDataset(Dataset):

    def __init__(self, fma_dir, subset, n_fft, hop_length, sr, tracks, num_classes, n_frames=2582, n_mels=128, file_ext=".wav"):
        super(SpectrogramDataset).__init__()
        self.fma_dir = fma_dir
        self.subset = subset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.tracks = tracks
        self.labels = tracks.set_index('track_id')['track', 'genre_top_id']
        self.num_classes = num_classes
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.file_ext = file_ext

        self.duration_max = self.tracks['track', 'duration'].max()

        self.audio_dir = os.path.join(self.fma_dir, f"fma_{self.subset}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index) -> T_co:
        track = self.tracks.loc[index]
        filename = self.get_filename_for_track_id(track['track_id'].values[0])
        mel, sr = gen_spec(filename, self.n_fft, self.hop_length, self.sr, self.n_mels)

        if sr != self.sr:
            raise ValueError("The output sampling rate does not match the requested one.")
        mel = torch.from_numpy(mel)

        # ensure the spectrogram has the correct shape
        mel = mel[:, :self.n_frames]
        if mel.shape[1] < self.n_frames:
            # pad with zeros
            mel = torch.cat((mel, torch.zeros(mel.shape[0], self.n_frames - mel.shape[1])), dim=1)

        assert mel.shape == (self.n_mels, self.n_frames)

        y = self.labels[track['track_id']]
        y = torch.from_numpy(y.values)
        y = F.one_hot(y, self.num_classes)

        return mel, y

    def get_filename_for_track_id(self, track_id):
        return f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track_id))[0]}{self.file_ext}"


class FmaSpectrogramGenreDataModule(pl.LightningDataModule):
    """
    class lightning datamodule that converts fma data into spectrograms and provides the spectrograms as a dataset
    """
    def __init__(self, fma_dir: str, subset: str, n_fft: int, hop_length: int, sr: int, n_mels: int = 128, batch_size: int = 32, file_ext: str = ".wav"):
        super(FmaSpectrogramGenreDataModule, self).__init__()
        self.data_dir = fma_dir
        self.audio_dir = os.path.join(self.data_dir, f"fma_{subset}")
        self.subset = subset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self. sr = sr
        self.batch_size = batch_size
        self.file_ext = file_ext
        self.n_mels = n_mels

        # load metadata
        extended_metadata_path = os.path.join(self.data_dir, f"fma_metadata/tracks_ext_{self.subset}_genre-top.csv")
        if os.path.isfile(extended_metadata_path):
            print(f"INFO: Loading extended metadata from {extended_metadata_path}")
            self.tracks = fma_utils.load(extended_metadata_path)
        else:
            print(f"INFO: No extended metadata found in path {extended_metadata_path}. Generating now...")
            self.tracks = self.build_extended_metadata(extended_metadata_path)

        self.num_classes = len(self.tracks['track', 'genre_top'].cat.categories)
        self.max_frames = self.tracks['track', 'expected_frames'].max()

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
                self.max_frames,
                self.n_mels
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
                self.max_frames,
                self.n_mels
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
                self.max_frames,
                self.n_mels
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_test, batch_size=self.batch_size, shuffle=False)

    def get_clip_duration(self, x):
        fn = f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, x['track_id'].values[0]))[0]}{self.file_ext}"
        try:
            y, r = librosa.load(fn, sr=self.sr)
            x['track', 'clip_duration'] = librosa.get_duration(y, sr=r)
        except FileNotFoundError:
            x['track', 'clip_duraiton'] = np.nan
        return x

    def get_expected_frames(self, x):
        x['track', 'expected_frames'] = np.ceil(
            x['track', 'clip_duration'] * self.sr / self.hop_length
        ).astype('int64')
        return x

    def build_extended_metadata(self, path):
        tqdm.tqdm.pandas()

        tracks = fma_utils.load(os.path.join(self.data_dir, 'fma_metadata/tracks.csv'))
        tracks = tracks[tracks['set', 'subset'] <= self.subset]

        tracks = tracks.dropna(subset=[('track', 'genre_top')])
        self.remove_nonexistent_tracks(tracks)

        # generate label IDs in new column genre_top_id
        labels = tracks['track', 'genre_top']
        label_id_mapping = {label: label_id for label_id, label in enumerate(labels.cat.categories)}
        tracks['track', 'genre_top_id'] = tracks['track', 'genre_top'].progress_apply(
            lambda x: label_id_mapping[x]
        ).astype('int64')

        # generate clip durations in new column clip_duration
        tracks.reset_index(inplace=True)
        tracks['track', 'clip_duration'] = tracks.progress_apply(
            self.get_clip_duration,
            axis=1
        )['track', 'clip_duration']
        tracks.set_index('track_id', inplace=True)

        # generate expected number of spectrogram frames in new column expected_frames
        tracks['track', 'expected_frames'] = tracks.progress_apply(
            self.get_expected_frames,
            axis=1
        )['track', 'expected_frames']

        print("INFO: Generated new columns ('track', 'genre_top_id'), "
              "('track', 'clip_duration'), ('track', 'expected_frames').")

        if os.access(path, os.W_OK):
            # save extended metadata
            print(f"INFO: Saving extended metadata to {path}.")
            tracks.to_csv(path, index=True)
        else:
            # print warning
            print(f"WARNING: could not save extended metadata to {path}. No write permissions.")

        return tracks

    def remove_nonexistent_tracks(self, tracks):
        """
        iterate over self.tracks and remove entries where the corresponding file does not exist
        """
        tracks.reset_index(inplace=True)
        iterator = tqdm.tqdm(tracks.iterrows())
        iterator.set_description("Removing nonexistent tracks")
        for idx, track in iterator:
            filename = self.get_filename_for_track_id(track['track_id'].values[0])
            if not os.path.isfile(filename):
                tracks.drop(idx, inplace=True)
        tracks.set_index('track_id', inplace=True)

    def get_filename_for_track_id(self, track_id):
        return f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track_id))[0]}{self.file_ext}"
