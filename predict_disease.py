from pathlib import Path
from typing import Any
import IPython.display as ipd
import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from mlxtend.plotting import heatmap




def getFeatures(path):
    soundArr,sample_rate=lb.load(path)
    mfcc=lb.feature.mfcc(y=soundArr,sr=sample_rate)
    cstft=lb.feature.chroma_stft(y=soundArr,sr=sample_rate)
    mSpec=lb.feature.melspectrogram(y=soundArr,sr=sample_rate)

    return mfcc,cstft,mSpec

def find_disease(path):

    # os.remove("static/heatmap.jpg")

    temp = path
    a,b,c = getFeatures(temp)
    mfcc,cstft,mSpec=[],[],[]


    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)

    mfcc_test=np.array(mfcc)
    cstft_test=np.array(cstft)
    mSpec_test=np.array(mSpec)

    mfcc_test.resize(1,20,259)
    cstft_test.resize(1,12,259)
    mSpec_test.resize(1,128,259)

    model = keras.models.load_model("static/model/finalmodel.h5")

    diseases = ['COPD', 'Healthy', 'URTI', 'Pneumonia', 'Bronchiectasis', 'Bronchiolitis', 'LRTI', 'Asthma']

    f = model.predict({"mfcc":mfcc_test,"croma":cstft_test,"mspec":mSpec_test})
    # print(type(f))

    heatmap(f, figsize=(20, 10), cell_values=False)
    plt.xlabel("diseases")
    plt.title("Probability of Diseases")
    plt.savefig("static/heatmap.jpg")



# find_disease("static/input.wav")