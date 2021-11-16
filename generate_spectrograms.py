import logging
import os
from spectrograms.generate import generate_spectrograms
import fire


def run(fma_dir, output_dir, subset="small", n_fft=2048, hop_length=512, debug=False):
    # TODO: parallel processing
    logging.basicConfig(filename=os.path.join(output_dir, "test.log"), level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger("spec_gen")
    try:
        generate_spectrograms(fma_dir, output_dir, subset, n_fft, hop_length, logger)
    except TypeError as e:
        logging.error(f"A TypeError occurred!\n\n{e}")


if __name__ == "__main__":
    fire.Fire(run)
