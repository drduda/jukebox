import time
import os

#import IPython.display as ipd
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
#import keras
#from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier

import fma.utils as utils

# Jukebox imports
from jukebox.hparams import Hyperparams
from jukebox.make_models import make_model
import fire
import torch

from jukebox.prior.conditioners import SimpleEmbedding
from jukebox.transformer.transformer import Transformer
from jukebox.prior.autoregressive import PositionEmbedding

AUDIO_DIR = "data/fma_small"

def run(model, **kwargs):
    # Load metadata
    tracks = utils.load('data/fma_metadata/tracks.csv')
    subset = tracks.index[tracks['set', 'subset'] <= 'small']
    tracks = tracks.loc[subset]

    assert subset.isin(tracks.index).all()

    # Get split
    train = tracks.index[tracks['set', 'split'] == 'training']
    val = tracks.index[tracks['set', 'split'] == 'validation']
    test = tracks.index[tracks['set', 'split'] == 'test']
    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))


    # Get labels
    labels_onehot = tracks['track', 'genre_top'].astype('category')
    labels_onehot = labels_onehot.cat.codes
    labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)

    # Load raw audio
    #todo what about sampling rate
    loader = utils.LibrosaLoader(sampling_rate=22100)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)
    print('Dimensionality: {}'.format(loader.shape))
    train_loader = SampleLoader(train, batch_size=5)

    optimizer = torch.optim.Adam

    input, labels = train_loader.__next__()

    # Get models
    hps = Hyperparams(**kwargs)
    #sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))
    device = torch.device("cuda")
    vqvae, priors = make_model(model, device, hps)

    del priors

    # Top raw to tokens is the compressing rate
    # 8192 context codebooks/(44100 sample rate/128 compression_rate(raw_to_tokens) = 24sec

    # Reshape input
    input = torch.Tensor(np.expand_dims(input, axis=-1)).cuda()
    # Get codebooks
    zs = vqvae.encode(input, start_level=0, end_level=3, bs_chunks=input.shape[0])

    codebook_amount = 2048
    transformer_size = 2048
    context_size = 8192


    top_level_codebooks = zs[2]
    if context_size < top_level_codebooks.shape[1]:
        # Get only so many codebooks which fit in transformer
        top_level_codebooks[:,:-context_size]

    codebook_amount = top_level_codebooks.shape[1]

    # Get embeddings
    #todo what is init_scale
    embedding_layer = SimpleEmbedding(context_size, transformer_size, init_scale=1).cuda()
    embeddings = embedding_layer(top_level_codebooks)

    # Get positional embeddings
    #todo again init_scale
    pos_emb = PositionEmbedding(input_shape=context_size, width=transformer_size, init_scale=1).cuda()
    embeddings += pos_emb()[:codebook_amount, :]

    # Define model
    #todo what is n_depth
    transformer = Transformer(n_in=transformer_size, n_ctx=context_size, n_head=2, blocks=32, n_depth=5).cuda()
    classifier = torch.nn.Sequential(
        torch.nn.Linear(transformer_size, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 8)).cuda()


    optimizer = torch.optim.Adam(transformer.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    output = transformer.forward(embeddings, None, fp16_out=True, fp16=True)
    output = classifier(output[:, 0, :])

    loss = loss_fn(output, labels)

    optimizer.zero_grad()
    loss.backwards()
    optimizer.step()
    print("done")


if __name__ == '__main__':
    fire.Fire(run)
