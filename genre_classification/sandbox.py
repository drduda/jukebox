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
    labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
    labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)

    # Load raw audio
    #todo what about sampling rate
    loader = utils.LibrosaLoader(sampling_rate=22100)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)
    print('Dimensionality: {}'.format(loader.shape))
    train_loader = SampleLoader(train, batch_size=5)
    input, labels = train_loader.__next__()

    # Get models
    hps = Hyperparams(**kwargs)
    #sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))
    device = torch.device("cuda")
    vqvae, priors = make_model(model, device, hps)

    # Top raw to tokens is the compressing rate
    top_raw_to_tokens = priors[-1].raw_to_tokens
    # 8192 context codebooks/(44100 sample rate/128 top_raw_tokens) = 24sec

    # Reshape input
    input = np.expand_dims(input, axis=-1)
    # Get codebooks
    zs = priors[-1].encode(input, start_level=0, end_level=len(priors), bs_chunks=input.shape[0])

    # finetune the top prior (priors[-1]) for classification

    priors[-1].train()

    print("done")


if __name__ == '__main__':
    fire.Fire(run)
