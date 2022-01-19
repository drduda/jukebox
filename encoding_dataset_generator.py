import fire
import torch
import tqdm
import torch as t
import utils
import os

from encode_database import JukeboxEncoder


def run(size, audio_dir, batch_size, output_dir='.'):
    if audio_dir.startswith('~'):
        audio_dir = os.path.expanduser(audio_dir)
    if output_dir.startswith('~'):
        output_dir = os.path.expanduser(output_dir)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    encoder = JukeboxEncoder()

    for split in ['training', 'validation', 'test']:
        output_path = os.path.join(output_dir, f"embedding_dataset_{size}_{split}.pt")
        Y, loader = utils.get_dataloader(audio_dir, size, split, batch_size)

        # Make the arrays
        tracks_embedded = []
        tracks_length = []

        for idx, (x, _) in tqdm.tqdm(enumerate(loader)):
            with torch.no_grad():
                x = t.from_numpy(x).to(device)
                x = t.unsqueeze(x, -1)
                if x.nelement() == 0:
                    print("INFO: Skipping empty tensor")
                    continue

                # Feed in Jukebox + technical adjustments
                zs, encodings_quantized = encoder.encode_sample(x, start_level=2, end_level=3)
                es = torch.squeeze(encodings_quantized[0], 0).t().unsqueeze(0)

                tracks_embedded.append(es)
                tracks_length.append(es.shape[1])

                # Save as a backup on every 4th iteration
                if idx % 4 == 0:
                    torch.save((torch.cat(tracks_embedded, dim=0), tracks_length, Y[:len(tracks_embedded)]), output_path)

        torch.save((torch.cat(tracks_embedded, dim=0), tracks_length, Y[:len(tracks_embedded)]), output_path)


if __name__ == '__main__':
    fire.Fire(run)
