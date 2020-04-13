#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Framework : Tensorflow 

import tensorflow as tf
import sklearn.metrics as sm
import read_preprocess_data as rd

LR = 0.003
num = 50
data = 10
train_size = int(data*0.8)          # 80% training data and 20% validation
test_size = data - train_size

# computing the average
def average(input):
    return sum(input) / len(input)

# We have used batch normalization on the data. Batch normalizing each activation to have zero mean and unit variance
# Advantages of Batch normalization :
# Fast learning, Improved accuracy, Normalization
# Solves the problem of internal covariate shifts

def norm(x,op, p, s='bn'):
    a = tf.Variable(tf.constant(0.0,shape=[op]),name='a')
    b = tf.Variable(tf.constant(1.0,shape=[op]),name='b')
    bm, bv = tf.nn.moments(x,[0,1,2],name='moments')
    mean, var = tf.cond(p,lambda : (average(bm), average(bv)))
    result = tf.nn.batch_normalization(x, mean, var, a, b, 1e-3)
    return result

# functionality for CNN model
# CNN model has 4 layers
# 4 convolutional layers: 2 Convolutional + max pooling + Dropouts

def cnn_model(mel_spect, wt, p):
    #tf.nn.conv2d()
    a = tf.reshape(mel_spect,[-1,1,96,1366])
    a = norm(a, 1366, p_t)
    a = tf.reshape(a,[-1,96,1366,1])   
    convolution_1 = tf.add(tf.nn.conv2d(a, wt['w1'], strides=[1, 1, 1, 1], padding='SAME'), wt['b1'])       # Convolution layer 1
    convolution_1 = tf.nn.relu(norm(convolution_1, 32, p_t))                                                # Convolution layer 2
    max_pool_1 = tf.nn.max_pool(convolution_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')   # Maxpool 
    dp_1 = tf.nn.dp(max_pool_1, 0.5)                                                                        # Dropout 
    
    convolution_2 = tf.add(tf.nn.conv2d(dp_1, wt['w2'], strides=[1, 1, 1, 1], padding='SAME'), wt['b2'])    # Convolution layer 1
    convolution_2 = tf.nn.relu(norm(convolution_2, 128, p_t))                                               # Convolution layer 2
    max_pool_2 = tf.nn.max_pool(convolution_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')   # Maxpool
    dp_2 = tf.nn.dp(max_pool_2, 0.5)                                                                        # Dropout 
    
    f = tf.reshape(dp_2, [-1, wt['wop'].get_shape().as_list()[0]])
    result = tf.nn.sigmoid(tf.add(tf.matmul(f,wt['wop']),wt['bop']))
    return result


if __name__=='main':
    # get data split from module form read_preprocess_data
    X_train, Y_train, X_test, Y_test = rd.getSplitData()
    music_train = (X_train, Y_train)
    music_test = (X_test, Y_test)
    wt = {
        'w1':tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)),
        'w2':tf.Variable(tf.random_normal([3, 3, 32, 128], stddev=0.01)),
        'w3':tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01)),
        'b1':tf.Variable(tf.zeros(32)),
        'b2':tf.Variable(tf.zeros(128)),
        'b3':tf.Variable(tf.zeros(128)),
        'wop':tf.Variable(tf.random_normal([256,6], stddev=0.01)),
        'bop':tf.Variable(tf.zeros(6))}
    X = tf.placeholder("float", [None, 96, 1366, 1])
    p_t = tf.placeholder(tf.bool, name='p_t')
    Y = cnn(X, wt, p_t)
    
# When only the simple convolutional neural network is used, an accuracy of 64% is obtained after convergence.
