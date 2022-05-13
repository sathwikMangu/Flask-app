#########################################################################
#Importing python Libraries
#########################################################################
from pathlib import Path
from typing import Any
import IPython.display as ipd
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from mlxtend.plotting import heatmap
#########################################################################

#########################################################################
def getFeatures(path):
    soundArr,sample_rate=lb.load(path)
    mfcc=lb.feature.mfcc(y=soundArr,sr=sample_rate,fmax = 8000)
    cstft=lb.feature.chroma_stft(y=soundArr,sr=sample_rate)
    mSpec=lb.feature.melspectrogram(y=soundArr,sr=sample_rate)

    return mfcc,cstft,mSpec
#########################################################################

#########################################################################
def find_disease(path):
    temp = path
    a,b,c = getFeatures(temp)
    mfcc,cstft,mSpec=[],[],[]
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)
    mfcc_test=np.array(mfcc)
    cstft_test=np.array(cstft)
    mSpec_test=np.array(mSpec)
    #mSpec_test = mSpec_test.reshape(20,259,1)
    print(mfcc_test.shape,cstft_test.shape,mSpec_test.shape)
    mfcc_test.resize(1,20,259)
    cstft_test.resize(1,12,259)
    mSpec_test.resize(1,128,259)
    print(mfcc_test.shape)

    model = keras.models.load_model('../input/finalmodel/finalmodel.h5')

    diseases = ['COPD', 'Healthy', 'URTI', 'Pneumonia', 'Bronchiectasis', 'Bronchiolitis', 'LRTI', 'Asthma']
    d = diseases.sort()
    print(diseases)

    f = model.predict({"mfcc":mfcc_test,"croma":cstft_test,"mspec":mSpec_test})

    

    heatmap(f, figsize=(20, 10), cell_values=False)
    plt.xlabel("diseases")
    plt.title("Probability of Diseases")
    plt.savefig("heatmap.jpg")

#########################################################################

#########################################################################

find_disease("../static/input.wav")