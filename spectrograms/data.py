import os
from typing import Optional, Union, List, Dict

import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import librosa
import numpy as np
import pytorch_lightning as pl
from scipy.stats import zscore
import pandas as pd

from fma import utils as fma_utils

from spectrograms.spectrogram_utils import gen_spec


class SpectrogramDataset(Dataset):
    """
    Dataset converting fma audio files to spectrograms.
    """

    def __init__(self, fma_dir, subset, n_fft, hop_length, sr, tracks, num_classes, n_frames=2582, n_mels=128, file_ext=".wav", save_specs=False, save_specs_dir="", from_scratch=False):
        """
        Constructor
        :param fma_dir: Path to the fma directory.
        :param subset: Subset of the fma dataset to use.
        :param n_fft: Number of FFT bins.
        :param hop_length: Number of samples between each FFT.
        :param sr: Sampling rate.
        :param tracks: Tracks DataFrame.
        :param num_classes: Number of classes.
        :param n_frames: Spectrogram width.
        :param n_mels: Number of mel bands.
        :param file_ext: File extension specifying audio type to be used.
        :param save_specs: Whether to save spectrograms to disk.
        :param save_specs_dir: Directory to save spectrograms to.
        :param from_scratch: Whether to generate spectrograms from scratch.
        """
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
        self.save_specs = save_specs
        self.save_specs_dir = save_specs_dir
        self.from_scratch = from_scratch

        if self.save_specs_dir == "":
            self.save_specs_dir = os.path.join(self.fma_dir, "spectrograms")
        if not os.path.isdir(self.save_specs_dir):
            os.makedirs(self.save_specs_dir)

        self.audio_dir = os.path.join(self.fma_dir, f"fma_{self.subset}")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index) -> T_co:
        track = self.tracks.loc[index]

        id_str = str(track['track_id'].values[0]).zfill(6)
        spec_dir = os.path.join(self.save_specs_dir, id_str[:3])
        if not os.path.isdir(spec_dir):
            os.makedirs(spec_dir)
        spec_path = os.path.join(spec_dir, f"{id_str}.spec")

        # if possible and allowed use pre-generated spectrograms
        if not self.from_scratch and os.path.isfile(spec_path) and os.access(spec_path, os.R_OK):
            spec_data = torch.load(spec_path)
            spec = spec_data['spec']
            y = spec_data['y']

            assert spec.shape == (self.n_frames, self.n_mels)

            return spec, y

        # otherwise generate spectrograms

        filename = self.get_filename_for_track_id(track['track_id'].values[0])
        spec, sr = gen_spec(filename, self.n_fft, self.hop_length, self.sr, self.n_mels)

        if sr != self.sr:
            raise ValueError("The output sampling rate does not match the requested one.")
        spec = torch.from_numpy(spec)

        # ensure the spectrogram has the correct shape
        spec = spec[:, :self.n_frames]
        if spec.shape[1] < self.n_frames:
            # pad with -80 dB
            spec = torch.cat((spec, torch.full((spec.shape[0], self.n_frames - spec.shape[1]), -80.)), dim=1)

        spec = torch.swapaxes(spec, 0, 1)

        y = self.labels[track['track_id']]
        y = torch.from_numpy(y.values)

        if self.save_specs:
            if os.access(spec_dir, os.W_OK):
                torch.save({"spec": spec, "y": y}, spec_path)
            else:
                print(f"WARNING: Could not save spectrogram to {spec_dir}. No write permissions.")

        assert spec.shape == (self.n_frames, self.n_mels)

        return spec, y

    def get_filename_for_track_id(self, track_id: int) -> str:
        """
        Returns the filename of the audio file for the given track ID respecting the data type specified in
        self.file_ext.
        :param track_id: Track ID.
        """
        return f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track_id))[0]}{self.file_ext}"


class FmaSpectrogramGenreDataModule(pl.LightningDataModule):
    """
    class lightning datamodule that converts fma data into spectrograms and provides the spectrograms as a dataset
    """

    def __init__(self, fma_dir: str, subset: str, n_fft: int, hop_length: int, sr: int, n_mels: int = 128, batch_size: int = 32, file_ext: str = ".wav", save_specs: bool = False, save_specs_dir: str = "", from_scratch: bool = False):
        """
        Constructor
        :param fma_dir: Path to the FMA dataset.
        :param subset: Subset of the FMA dataset to use.
        :param n_fft: Number of FFT bins.
        :param hop_length: Number of samples between successive frames.
        :param sr: Sampling rate.
        :param n_mels: Number of mel bands.
        :param batch_size: Batch size.
        :param file_ext: File extension specifying audio type to be used.
        :param save_specs: Whether to save spectrograms to disk.
        :param save_specs_dir: Directory to save spectrograms to.
        :param from_scratch: Whether to generate spectrograms from scratch.
        """
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
        self.save_specs = save_specs
        self.save_specs_dir = save_specs_dir
        self.from_scratch = from_scratch

        if self.save_specs_dir == "":
            self.save_specs_dir = os.path.join(self.data_dir, "spectrograms")

        # load metadata
        extended_metadata_path = os.path.join(self.data_dir, f"fma_metadata/tracks_ext_{self.subset}_genre-top.csv")
        if os.path.isfile(extended_metadata_path):
            print(f"INFO: Loading extended metadata from {extended_metadata_path}")
            self.tracks = fma_utils.load(extended_metadata_path)
        else:
            print(f"INFO: No extended metadata found in path {extended_metadata_path}. Generating now...")
            self.tracks = self.build_extended_metadata(extended_metadata_path)

        self.num_classes = len(self.tracks['track', 'genre_top'].cat.categories)

        # spectrograms will padded to maximum length of all spectrograms
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
                self.n_mels,
                save_specs=self.save_specs,
                save_specs_dir=self.save_specs_dir,
                from_scratch=self.from_scratch
            )

            val_tracks = self.tracks[self.tracks['set', 'split'] == 'validation'].copy().reset_index()

            self.spec_ds_val = SpectrogramDataset(
                self.data_dir,
                self.subset,
                self.n_fft,
                self.hop_length,
                self.sr,
                val_tracks,
                self.num_classes,
                self.max_frames,
                self.n_mels,
                save_specs=self.save_specs,
                save_specs_dir=self.save_specs_dir,
                from_scratch=self.from_scratch
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
                self.n_mels,
                save_specs=self.save_specs,
                save_specs_dir=self.save_specs_dir,
                from_scratch=self.from_scratch
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.spec_ds_test, batch_size=self.batch_size, shuffle=False)

    def get_clip_duration(self, x: pd.Series) -> pd.Series:
        """
        Helper function for adding clip duration to metadata
        :param x: sample to add clip duration to
        """
        fn = f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, x['track_id'].values[0]))[0]}{self.file_ext}"
        try:
            y, r = librosa.load(fn, sr=self.sr)
            x['track', 'clip_duration'] = librosa.get_duration(y, sr=r)
        except FileNotFoundError:
            x['track', 'clip_duraiton'] = np.nan
        return x

    def get_expected_frames(self, x: pd.Series) -> pd.Series:
        """
        Helper function for adding expected frames to metadata
        :param x: sample to add expected frames to
        """
        x['track', 'expected_frames'] = np.ceil(
            x['track', 'clip_duration'] * self.sr / self.hop_length
        ).astype('int64')
        return x

    def build_extended_metadata(self, path: str) -> pd.DataFrame:
        """
        Extend metadata with clip duration and expected frames
        :param path: path to save extended metadata to
        """
        tqdm.tqdm.pandas()

        tracks = fma_utils.load(os.path.join(self.data_dir, 'fma_metadata/tracks.csv'))
        tracks = tracks[tracks['set', 'subset'] <= self.subset]

        # clean from non-usable samples
        tracks_new = tracks.dropna(subset=[('track', 'genre_top')])
        print(f"INFO: Removed {len(tracks) - len(tracks_new)} tracks without genre.")
        tracks = tracks_new
        del tracks_new
        self.remove_nonexistent_tracks(tracks)

        # generate label IDs in new column genre_top_id
        labels = tracks['track', 'genre_top'].astype('category').cat.remove_unused_categories()
        labels = labels.cat.codes
        tracks['track', 'genre_top_id'] = labels

        # generate clip durations in new column clip_duration
        tracks.reset_index(inplace=True)
        tracks['track', 'clip_duration'] = tracks.progress_apply(
            self.get_clip_duration,
            axis=1
        )['track', 'clip_duration']
        tracks.set_index('track_id', inplace=True)

        # remove rows with clip duration zscore > 3
        tracks['track', 'cd_zscore'] = np.abs(zscore(tracks['track', 'clip_duration']))
        tracks_new = tracks[tracks['track', 'cd_zscore'] <= 3]
        print(f"INFO: Removed {len(tracks) - len(tracks_new)} tracks with clip duration zscore > 3.")
        tracks = tracks_new
        del tracks_new
        tracks = tracks.drop([('track', 'cd_zscore')], axis=1)

        # generate expected number of spectrogram frames in new column expected_frames
        tracks['track', 'expected_frames'] = tracks.progress_apply(
            self.get_expected_frames,
            axis=1
        )['track', 'expected_frames']

        print("INFO: Generated new columns ('track', 'genre_top_id'), "
              "('track', 'clip_duration'), ('track', 'expected_frames').")

        dirname = os.path.dirname(path)
        if os.access(dirname, os.W_OK):
            print(f"INFO: Saving extended metadata to {path}.")
            tracks.to_csv(path, index=True)
        else:
            print(f"WARNING: Could not save extended metadata to {dirname}. No write permissions.")

        return tracks

    def remove_nonexistent_tracks(self, tracks: pd.DataFrame):
        """
        Remove tracks that do not exist in audio directory respecting the data format specified in self.file_ext
        :param tracks: DataFrame to remove non-existent tracks from
        """
        tracks.reset_index(inplace=True)
        iterator = tqdm.tqdm(tracks.iterrows())
        iterator.set_description("Removing nonexistent tracks")
        counter = 0
        for idx, track in iterator:
            filename = self.get_filename_for_track_id(track['track_id'].values[0])
            if not os.path.isfile(filename):
                tracks.drop(idx, inplace=True)
                counter += 1
        tracks.set_index('track_id', inplace=True)
        print(f"INFO: Removed {counter} tracks without corresponding {self.file_ext} file.")

    def get_filename_for_track_id(self, track_id: int) -> str:
        """
        Returns the filename of the audio file for the given track ID respecting the data type specified in
        self.file_ext.
        :param track_id: Track ID.
        """
        return f"{os.path.splitext(fma_utils.get_audio_path(self.audio_dir, track_id))[0]}{self.file_ext}"
