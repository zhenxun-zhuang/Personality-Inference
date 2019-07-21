# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:17:21 2016

@author: oldbo
"""

from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import scipy.io as sio
import numpy as np
import glob
import os

path='C:/Study/Research/Computer Vision/Introduction to Computer Vision/Project/Data/val/audios'

os.chdir(path)

num = 3059*13#default feature size
fea_mfcc   = np.zeros((num,1));#initialization
fea_d_mfcc = np.zeros((num,1))

i = 1;

for filename in glob.iglob('./*.wav'):
    if(i%100 == 0):
        print(i)
    i+=1;
    
    (rate,sig)  = wav.read(filename)#read file
    mfcc_feat   = mfcc(sig,rate)#compute mfcc
    d_mfcc_feat = delta(mfcc_feat, 2)#compute delta mfcc
    d_mfcc_feat = np.asarray(d_mfcc_feat, order = 'C')
    num_fea     = mfcc_feat.size#obtain current feature size
    mfcc_fea_c  = mfcc_feat.reshape((num_fea,1),order='C')#reshape to be a column
    d_mfcc_fea_c= d_mfcc_feat.reshape((num_fea,1),order='C')

     
    if(num_fea!=num):#make the current feature size to be the same as default
        num_repeat   = num//num_fea
        num_fill     = num % num_fea
        mfcc_fea_c   = np.tile(mfcc_fea_c,(num_repeat,1))
        mfcc_fea_c   = np.append(mfcc_fea_c,mfcc_fea_c[0:num_fill],axis=0)
        d_mfcc_fea_c = np.tile(d_mfcc_fea_c,(num_repeat,1))
        d_mfcc_fea_c = np.append(d_mfcc_fea_c,d_mfcc_fea_c[0:num_fill],axis=0)
        
    fea_mfcc   = np.column_stack((fea_mfcc,mfcc_fea_c));#concatenation
    fea_d_mfcc = np.column_stack((fea_d_mfcc,d_mfcc_fea_c))
    

sio.savemat('fea_audio_mfcc.mat',{'fea_audio_mfcc':fea_mfcc[:,1:]})#write file

fea_comb_mfcc = np.concatenate((fea_mfcc, fea_d_mfcc), axis=0)
sio.savemat('fea_audio_comb_mfcc.mat',{'fea_audio_comb_mfcc':fea_comb_mfcc[:,1:]})#write file