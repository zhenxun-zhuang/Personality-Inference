# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:57:20 2016

@author: oldbo
"""
import glob
import os

path='C:/Study/Research/Computer Vision/Introduction to Computer Vision/Project/Data/test/audios'

os.chdir(path)

with open('order.txt', 'w') as the_file:
    for filename in glob.iglob('*.wav'):
        the_file.write(filename+'\n')