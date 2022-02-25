from string import printable
from tkinter import S
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa
import librosa
from models import models

# RAVDESS native sample rate is 48k
sample_rate = 24000

# Mel Spectrograms are not directly used as a feature in this model
# Mel Spectrograms are used in calculating MFCCs, which are a higher-level representation of pitch transition
# MFCCs work better - left the mel spectrogram function here in case anyone wants to experiment
def feature_melspectrogram(
    waveform, 
    sample_rate,
    fft = 1024,
    winlen = 512,
    window='hamming',
    hop=256,
    mels=128,
    ):
    
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2)
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients

def get_features(waveforms, features, samplerate):

    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress 
        #print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    
    # return all features from list of waveforms
    return features

def get_waveforms(file):

    duration=6
    
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=6, offset=0.5, sr=sample_rate)
    
    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*duration,)))
    waveform_homo[:len(waveform)] = waveform
    
    # return a single file's waveform                                      
    return waveform_homo
    
# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}

# Additional attributes from RAVDESS to play with
emotion_attributes = {
    '01': 'normal',
    '02': 'strong'
}
# define loss function; CrossEntropyLoss() fairly standard for multiclass problems 
def criterion(predictions, targets): 
    return nn.CrossEntropyLoss()(input=predictions, target=targets)
def load_checkpoint(optimizer, model, filename):
    device = torch.device('cuda')
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch

def analysis(file_1):
    wave_F=[None]
    wave_F[0]= get_waveforms(file_1)
    wave_F = np.array(wave_F)
    features_test =[]
    features_test = get_features(wave_F, features_test, sample_rate)
    X_test = np.expand_dims(features_test,1)
    for i in range(len(X_test[0][0])):
        for j in range(len(X_test[0][0][0])):
            X_test[0][0][0][j] = (X_test[0][0][0][j] - np.mean(X_test[0][0][i]))/np.std(X_test[0][0][i])

    model = models.parallel_all_you_want(len(emotions_dict))
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)
    epoch = '428'
    model_name = f'./models/parallel_all_you_wantFINAL-{epoch}.pkl'
    load_checkpoint(optimizer, model, model_name)
    X_test_tensor = torch.tensor(X_test,device=torch.device('cuda')).float()
    output_logits, output_softmax = model(X_test_tensor)
    predictions = torch.argmax(output_softmax,dim=1)
    if predictions== int(list(emotions_dict.keys())[0]):
        out= 'Surprised'
    elif predictions== int(list(emotions_dict.keys())[1]):
        out= 'Neutral'
    elif predictions== int(list(emotions_dict.keys())[2]):
        out= 'Calm'
    elif predictions== int(list(emotions_dict.keys())[3]):
        out= 'Happy'
    elif predictions== int(list(emotions_dict.keys())[4]):
        out= 'Sad'
    elif predictions== int(list(emotions_dict.keys())[5]):
        out= 'Angry'
    elif predictions== int(list(emotions_dict.keys())[6]):
        out= 'fearful'
    elif predictions== int(list(emotions_dict.keys())[7]):
        out= 'disgust'

    return out
