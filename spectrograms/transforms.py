# source: https://github.com/ChristianBergler/ORCA-CLEAN
import torch


class Spectrogram(object):
    """Converts a given audio to a spectrogram."""

    def __init__(self, n_fft, hop_length, center=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        S = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            onesided=True,
            return_complex=False
        ).transpose(1, 2)
        Sp = S/(self.window.pow(2).sum().sqrt())
        Sp = Sp.pow(2).sum(-1)
        return Sp, S
