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
import tqdm

from jukebox.prior.conditioners import SimpleEmbedding
from jukebox.transformer.transformer import Transformer
from jukebox.prior.autoregressive import PositionEmbedding

# MIGHT MAKE TRAINING FASTER ACCORDING TO: https://github.com/pytorch/pytorch/issues/46377
#torch.backends.cudnn.benchmark = True

AUDIO_DIR = "data/fma_small"
batch_size = 12

class GenreClassifier(torch.nn.Module):
    def __init__(self, embedding_layer, pos_embedding, transformer, classifier, unfreeze_from_block=71):
        super(GenreClassifier, self).__init__()

        self.embedding_layer = embedding_layer
        #todo pos_embedding requires_grad = False
        self.pos_embedding = pos_embedding

        unfreeze = False
        # Each block is mode of 12 layers
        for name, parameters in transformer.named_parameters():
            if str(unfreeze_from_block) in name or unfreeze:
                unfreeze = True
                parameters.requires_grad = True

        self.transformer = transformer
        self.classifier = classifier


    def forward(self, x):

        embeddings = self.embedding_layer(x)
        actual_length = embeddings.shape[1]
        embeddings += self.pos_embedding()
        # Trai
        output = self.transformer.forward(embeddings, None)
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
    #todo fma github page says there are only 8 top genres, we have 16 why??
    labels_onehot = tracks['track', 'genre_top'].astype('category').cat.remove_unused_categories()
    labels_onehot = labels_onehot.cat.codes
    labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)

    # Get models
    hps = Hyperparams(**kwargs)
    # sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))
    device = torch.device("cuda")
    vqvae, priors = make_model(model, device, hps)

    top_prior = priors.pop(-1)
    del priors

    # Load raw audio
    loader = utils.LibrosaLoader(sampling_rate=hps.sr)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)
    print('Dimensionality: {}'.format(loader.shape))
    train_loader = SampleLoader(train, batch_size=batch_size)
    val_loader = SampleLoader(val, batch_size=batch_size)

    transformer = top_prior.prior.transformer
    pos_emb = top_prior.prior.pos_emb
    embedding_layer = top_prior.prior.x_emb
    # Top raw to tokens is the compressing rate
    # 8192 context codebooks/(44100 sample rate/128 compression_rate(raw_to_tokens) = 24sec

    classifier = torch.nn.Sequential(
        torch.nn.Linear(transformer.n_in, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 8))

    model = GenreClassifier(embedding_layer, pos_emb, transformer, classifier).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Metrics
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    print("Number of epochs {}".format(hps.epochs))
    for e in range(hps.epochs):
        print("Epoch {} of {}".format(e, hps.epochs))

        epoch_loss = []
        epoch_accuracy = []

        for input, labels in train_loader:
            print(".", end="")

            # Reshape input
            input = torch.Tensor(np.expand_dims(input, axis=-1)).to(device)
            # Get codebooks
            with torch.no_grad():
                zs = vqvae.encode(input, start_level=2, end_level=3, bs_chunks=input.shape[0])

            # Take only top level
            top_level_codebooks = zs[0]
            if transformer.n_ctx > top_level_codebooks.shape[1]:
                raise NotImplementedError("Cannot handle shorter song length so far")

            # Take cutout of song so that it can fit inside the transformer at once
            top_level_codebooks = top_level_codebooks[:, :transformer.n_ctx]

            labels = torch.squeeze(torch.tensor(labels, dtype=torch.long, device=device))

            optimizer.zero_grad()
            #todo half precision training
            output = model(top_level_codebooks)
            loss = loss_fn(output, labels)

            # Metrics
            epoch_loss.append(loss.item())
            epoch_accuracy.append(accuracy(output, labels))

            loss.backward()
            optimizer.step()

        print("")
        print("Train accuracy is {}".format(np.mean(epoch_accuracy)))
        train_loss.append(np.mean(epoch_loss))
        train_accuracy.append(np.mean(epoch_accuracy))

        epoch_loss = []
        epoch_accuracy = []

        for input, labels in val_loader:
            print(".", end="")
            with torch.no_grad():

                # Reshape input
                input = torch.Tensor(np.expand_dims(input, axis=-1)).to(device)
                # Get codebooks
                zs = vqvae.encode(input, start_level=2, end_level=3, bs_chunks=input.shape[0])

                # Take only top level
                top_level_codebooks = zs[0]
                if transformer.n_ctx > top_level_codebooks.shape[1]:
                    raise NotImplementedError("Cannot handle shorter song length so far")

                # Take cutout of song so that it can fit inside the transformer at once
                top_level_codebooks = top_level_codebooks[:, :transformer.n_ctx]

                labels = torch.squeeze(torch.tensor(labels, dtype=torch.long, device=device))

                output = model(top_level_codebooks)
                loss = loss_fn(output, labels)

                # Metrics
                epoch_loss.append(loss.item())
                epoch_accuracy.append(accuracy(output, labels))

        print("")
        print("Validation accuracy is {}".format(np.mean(epoch_accuracy)))
        val_loss.append(np.mean(epoch_loss))
        val_accuracy.append(np.mean(epoch_accuracy))


def accuracy(output, labels):
    correct = 0
    total = 0

    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    return correct/total

if __name__ == '__main__':
    fire.Fire(run)
