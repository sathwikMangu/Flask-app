from typing import Any
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os



def convert(path):
    # os.remove("static/chroma.jpg")
    # os.remove("static/mel.jpg")
    # os.remove("static/mfccs.jpg")

    filename = path
    y, sr = librosa.load(filename)
    
    
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    fig, ax = plt.subplots(figsize=(10,6))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr,fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.savefig("static/mel.jpg", dpi = 300)

    data,sample_rate1 = librosa.load(filename, sr=22050, mono=True, offset=0.0, duration=50, res_type='kaiser_best')

    C = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=C, sr=sr)
    fig, ax = plt.subplots(figsize=(10,6))
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='s', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Chromagram')
    plt.savefig("static/chroma.jpg", dpi = 300)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
    fig, ax = plt.subplots(figsize=(10,6))
    img = librosa.display.specshow(mfccs, y_axis='mel', x_axis='s', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel Frequency Cepstral Coeffecients')
    plt.savefig("static/mfccs.jpg", dpi = 300)

    pass

