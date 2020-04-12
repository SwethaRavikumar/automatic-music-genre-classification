#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 20:06:25 2020
@author: swetharavikumar
"""

import tensorflow as tf
import sklearn.metrics as sm
import read_preprocess_data as rd

LR = 0.003
num = 50

dta = 10
# 80% training
train_size = int(dta*0.8)
test_size = dta - train_size

# computing the average
def average(input):
    return sum(input) / len(input)

def norm(x,op, p, s='bn'):
    a = tf.Variable(tf.constant(0.0,shape=[op]),name='a')
    b = tf.Variable(tf.constant(1.0,shape=[op]),name='b')
    bm, bv = tf.nn.moments(x,[0,1,2],name='moments')
    mean, var = tf.cond(p,lambda : (average(bm), average(bv)))
    result = tf.nn.batch_normalization(x, mean, var, a, b, 1e-3)

# functionality for CNN model
def cnn_model(mel_spect, wt, p):
    tf.nn.conv2d()

if __name__=='main':
    # get data split from module form read_preprocess_data
    X_train, Y_train, X_test, Y_test = rd.getSplitData()
    music_train = (X_train, Y_train)
    music_test = (X_test, Y_test)
