#!/bin/env python3                                                         
#-*- coding=utf-8 -*- 
#testing for sentiment classification

import os, sys
import csv
import string
import random
import pickle
import math
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
#from cnn_model import cnn_model                                                                  
from cnn_layermodel import cnn_model


random_state=None
batch_size = 64
epochs = 500
pheight = 48
pwidth = 48
num_channels = 1

np.random.seed(random_state)

def load(testf):
    #label_list = []
    feature_list = []

    with open(testf, 'r', encoding='utf-8') as ftest:
        reader = csv.reader(ftest, delimiter=',', )
        for it,row in enumerate(reader):
            if it==0: #header
                continue

            feature_onesample = list(map(int, row[1].split(' ')))
            feature_list.append(feature_onesample)

    test_X = np.array(feature_list, dtype=np.float32)
    #reshape to 2d array                                                                          
    test_X = np.reshape(test_X, (it,pwidth,pheight,1))

    return test_X

def main():

    if len(sys.argv)<3:
        print("usage:ml2017springhw3_test.py test_file out_file")
        sys.exit()

    intestf  = sys.argv[1]
    outtestf = sys.argv[2]

    if os.path.exists('test.pkl')==True:
        with open("test.pkl", "rb") as pkf:
            test_X=pickle.load(pkf)
    else:
        test_X = load(intestf)
        with open("test.pkl", "wb") as pkf:
            pickle.dump(test_X, pkf)

    mean_pixel = [125.0]
    dev_pixel = [63.0]

    #preprocess, to standardizer the pixel value                                                                           
    for ch in range(num_channels):
        print("Normalize picture ch#{}".format(ch))
        #test_X[:,:,:,ch]=(test_X[:,:,:,ch]-mean_pixel[ch])/dev_pixel[ch]
        test_X[:,:,:,ch]=test_X[:,:,:,ch]/255.0
            
    # Create batches                                       
    num_batches = int(len(test_X)/batch_size) + 1
    # Split up text indices into subarrays, of equal size                                         
    batches = np.array_split(np.arange(test_X.shape[0]), num_batches)
    # Reshape each split into [batch_size, training_seq_len]                                      
    batches = [np.resize(x, [batch_size]) for x in batches]

    tf.reset_default_graph()
    #training variable for batch norm                                                             
    training = tf.placeholder_with_default(False,shape=(),name='training')
    
    #initialize place holder for input and output of model                                        
    X = tf.placeholder(tf.float32, shape=(batch_size, pheight, pwidth, num_channels), name="X")
    y = tf.placeholder(tf.int32, shape=(batch_size,7), name="y")
    #CNNmodel object                                           
    cnn = cnn_model(target_size=7)
    
    y_predict = cnn.model(X, batch_size, training=training)

    saver = tf.train.Saver()
    predict_index=[]
    DIR = "model/tf/"
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(DIR,"model_final"))
        for iv,tbatch in enumerate(batches):
            X_batch = test_X[tbatch]
            test_dict = {X:X_batch, training:False}
            predict_batch = y_predict.eval(feed_dict=test_dict)
            predict_index.extend(np.argmax(predict_batch,axis=-1))


    with open(outtestf, 'w') as fout:
        fout.write('id,label\n')
        for id, value in enumerate(predict_index):
            fout.write('%d,%d\n' %(id, value))

    return
    
    
if __name__=="__main__":

    main()
