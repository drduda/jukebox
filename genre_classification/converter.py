import librosa
import os

def convert(filename):
    y, sr = librosa.load(filename, sr=SAMPLE_RATE)
    os.remove(filename)
    librosa.output.write_wav(filename.replace('mp3', 'wav'), y, sr)

DATA_PATH = "C:/Users/Marko/PycharmProjects/jukebox/data/fma_small"
SAMPLE_RATE = 44100


for root, subdirs, files in os.walk(DATA_PATH):
    for subdir in subdirs:
        print(".", end="")
        for file in os.listdir(os.path.join(root, subdir)):
            if file.endswith(".mp3"):
                convert(os.path.join(root, subdir, file))
