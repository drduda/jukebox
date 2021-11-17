import fire
from spectrograms import data
import os
import tqdm
import json
import logging
from datetime import datetime


def run(fma_dir, output_dir, subset, n_fft, hop_length, sr, n_mels=128, batch_size=1, file_ext=".wav",
                       save_specs=False, from_scratch=False):
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(output_dir, f"{time_str}_gen_specs.log"))
    logger = logging.getLogger("gen_specs")

    hyper_params = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "sr": sr,
        "n_mels": n_mels,
        "batch_size": batch_size,
        "file_ext": file_ext
    }
    with open(os.path.join(output_dir, f"{time_str}_hyper_params.json"), "w") as f:
        json.dump(hyper_params, f)

    data_module = data.FmaSpectrogramGenreDataModule(
        fma_dir,
        subset,
        n_fft,
        hop_length,
        sr,
        n_mels,
        batch_size,
        file_ext,
        save_specs=save_specs,
        from_scratch=from_scratch
    )
    data_module.setup()

    train_dl = data_module.train_dataloader()
    iterator = tqdm.tqdm(enumerate(iter(train_dl)))
    iterator.set_description("Processing train")
    for idx, (spec, y) in iterator:
        logger.debug(f"train {idx}: {spec.shape}, {y}")

    val_dl = data_module.val_dataloader()
    iterator = tqdm.tqdm(enumerate(iter(val_dl)))
    iterator.set_description("Processing val")
    for idx, (spec, y) in iterator:
        logger.debug(f"val {idx}: {spec.shape}, {y}")

    test_dl = data_module.test_dataloader()
    iterator = tqdm.tqdm(enumerate(iter(test_dl)))
    iterator.set_description("Processing test")
    for idx, (spec, y) in iterator:
        logger.debug(f"test {idx}: {spec.shape}, {y}")


if __name__ == "__main__":
    fire.Fire()
