import os
from fma import utils as fma_utils
import tqdm
from spectrograms.spectrogram_utils import save_spec, gen_spec


def generate_spectrograms(fma_dir, output_dir, subset="small", n_fft=2048, hop_length=512, logger=None):
    audio_dir = os.path.join(fma_dir, f"fma_{subset}")
    metadata_path = os.path.join(fma_dir, 'fma_metadata/tracks.csv')
    tracks = fma_utils.load(metadata_path)
    tracks = tracks[tracks['set', 'subset'] <= subset]

    if logger is not None:
        logger.info(f"Generating spectrograms from directory {audio_dir} for {subset} subset.")

    for track_id, track in tqdm.tqdm(tracks.iterrows()):
        filename = f"{os.path.splitext(fma_utils.get_audio_path(audio_dir, track_id))[0]}.wav"
        logger.debug('File to load: {}'.format(filename))

        try:
            mel, sr = gen_spec(filename, n_fft, hop_length)
            output_filename = os.path.join(output_dir, os.path.basename(f"{os.path.splitext(filename)[0]}.pt"))
            if logger is not None:
                logger.debug('File to save: {}'.format(output_filename))
            save_spec(output_filename, mel, {"sr": sr, "hop_length": hop_length, "n_fft": n_fft})
        except FileNotFoundError as e:
            if logger is not None:
                logger.error(f"File not found: \n\n{e}")

