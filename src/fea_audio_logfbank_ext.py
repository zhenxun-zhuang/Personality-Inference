# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:17:21 2016

@author: oldbo
"""

from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.io as sio
import numpy as np
import glob
import os

path='C:/Study/Research/Computer Vision/Introduction to Computer Vision/Project/Data/train/audios'

os.chdir(path)

num = 3059*26#default feature size
fea = np.zeros((num,1));#initialization
i   = 1;

for filename in glob.iglob('./*.wav'):
    if(i%100 == 0):
        print(i)
    i+=1;
    
    (rate,sig)  = wav.read(filename)#read file
    fbank_feat  = logfbank(sig,rate)#compute logfbank feature
    num_fea     = fbank_feat.size#obtain current feature size
    fbank_fea_c = fbank_feat.reshape((num_fea,1),order='C')#reshape to be a column

     
    if(num_fea!=num):#make the current feature size to be the same as default
        num_repeat  = num//num_fea
        num_fill    = num % num_fea
        fbank_fea_c = np.tile(fbank_fea_c,(num_repeat,1))
        fbank_fea_c = np.append(fbank_fea_c,fbank_fea_c[0:num_fill],axis=0)
    
    fea = np.column_stack((fea,fbank_fea_c));#concatenation

sio.savemat('fea_audio.mat',{'fea_audio':fea[:,1:]})#write file