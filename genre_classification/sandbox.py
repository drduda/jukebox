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
codebook_amount = 2048
transformer_size = 2048
context_size = 8192
batch_size = 2

class GenreClassifier(torch.nn.Module):
    def __init__(self, embedding_layer, pos_embedding, transformer, classifier):
        super(GenreClassifier, self).__init__()

        self.embedding_layer = embedding_layer
        #todo pos_embedding requires_grad = False
        self.pos_embedding = pos_embedding

        self.transformer = transformer
        self.classifier = classifier


    def forward(self, x):

        embeddings = self.embedding_layer(x)
        actual_length = embeddings.shape[1]
        embeddings += self.pos_embedding()[:actual_length, :]
        # Trai
        output = self.transformer.forward(embeddings, None, fp16_out=True, fp16=True)
        output = self.classifier(output[:, 0, :])
        return output

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
    train_loader = SampleLoader(train, batch_size=batch_size)

    # Get models
    hps = Hyperparams(**kwargs)
    #sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))
    device = torch.device("cuda")
    vqvae, priors = make_model(model, device, hps)

    top_prior = priors.pop(-1)
    del priors

    transformer = top_prior.prior.transformer
    pos_emb = top_prior.prior.pos_emb
    embedding_layer = top_prior.prior.x_emb
    # Top raw to tokens is the compressing rate
    # 8192 context codebooks/(44100 sample rate/128 compression_rate(raw_to_tokens) = 24sec

    classifier = torch.nn.Sequential(
        torch.nn.Linear(transformer_size, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 8)).half().cuda()

    model = GenreClassifier(embedding_layer, pos_emb, transformer, classifier).cuda()

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for input, labels in train_loader:
        # Reshape input
        input = torch.Tensor(np.expand_dims(input, axis=-1)).cuda()
        # Get codebooks
        zs = vqvae.encode(input, start_level=0, end_level=3, bs_chunks=input.shape[0])

        top_level_codebooks = zs[2]
        if context_size < top_level_codebooks.shape[1]:
            raise NotImplementedError("Cannot handle different size so far")


        labels = torch.tensor(labels, dtype=torch.float16).cuda()
        output = model(top_level_codebooks)

        loss = loss_fn(output, labels)

        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()

        print("done")

if __name__ == '__main__':
    fire.Fire(run)
