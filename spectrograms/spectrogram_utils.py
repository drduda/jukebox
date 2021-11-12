import numpy as np
import torch
import librosa


def save_spec(filename, spec, meta_data={}):
    if type(spec) == np.ndarray:
        spec = torch.from_numpy(spec)
    elif type(spec) != torch.Tensor:
        raise TypeError(f"Spectrogram has to be numpy.ndarray or torch.Tensor. Is {type(spec)}.")

    meta_data.update({"data": spec})
    torch.save(meta_data, filename)


def load_spec(filename, logger=None):
    try:
        spec_dict = torch.load(filename)
        return spec_dict
    except FileNotFoundError as e:
        if logger is not None:
            logger.error(f"File not found: \n\n{e}")
    return None


def gen_spec(filename, n_fft, hop_length, sr=None, n_mels=None):
    x, sr = librosa.load(filename, sr=sr, mono=True)
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2, n_mels=n_mels)
    return mel, sr
