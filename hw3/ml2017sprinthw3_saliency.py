#!/bin/env python3                                                         
#-*- coding=utf-8 -*- 
#plot saliency maps for sentiment classification

import os, sys
import csv
import string
import random
import pickle
import math
import itertools
import numpy as np
import scipy as sp
import scipy.ndimage
import pandas as pd
import tensorflow as tf

#from cnn_model import cnn_model                                                                  
from cnn_layermodel import cnn_model


random_state=None
batch_size = 16 #input only one picture every time
epochs = 500
pheight = 48
pwidth = 48
num_channels = 1

np.random.seed(random_state)

def load(trainf):
    label_list = []
    feature_list = []

    with open(trainf, 'r', encoding='utf-8') as ftrain:
        reader = csv.reader(ftrain, delimiter=',', )
        for ir,row in enumerate(reader):
            if ir==0: #header         
                continue

            label_list.append(int(row[0]))
            feature_onesample = list(map(int, row[1].split(' ')))
            feature_list.append(feature_onesample)

    train_X = np.array(feature_list, dtype=np.float32)
    #reshape feature to 2d array
    train_X = np.reshape(train_X, (ir,pwidth,pheight,1))
    train_y = np.array(label_list,   dtype=np.int32)

    return train_X, train_y

def process_heat_map(heat_img):
    """
    Make normalization and smoothening on gradient
    """
    #print("Shape of heat map is {}".format(heat_img.shape))
    sigma_y = 2.0
    sigma_x = 2.0
    # Apply gaussian filter to make smoothening
    sigma = [sigma_y, sigma_x]
    smooth_img = sp.ndimage.filters.gaussian_filter(heat_img, sigma, mode='constant')
    # Normalization
    max_v = np.max(smooth_img)
    min_v = np.min(smooth_img)
    norm_img = (smooth_img[:,:]-min_v)/(max_v-min_v)
    
    return norm_img

def get_saliency_map(X, y, img_X, label_y, predict_y, sess):
    """
    Compute a saliency map using the stored model for img_X and label_y
    referenced from: 
    https://github.com/nick6918/MyDeepLearning/blob/master/SaliencyMap.py
    """
    saliency = None
    label = tf.cast(tf.argmax(y,1), tf.int32)
    #require the predicted probabilities of label index
    #the prob. are correct_scores
    correct_scores = tf.gather_nd(predict_y,
                                  tf.stack((tf.range(img_X.shape[0]),label),axis=1))
    #loss function used here is square of (1-correct_scores)
    losses = tf.square(1-correct_scores)
    #calculate gradient of loss function for each img pixel
    grad_img = tf.gradients(losses, X)
    img_dict = {X:img_X, y:label_y}
    grad_img_val = sess.run(grad_img, feed_dict=img_dict)[0]
    #the saliency pixel value should be larger than 0, and remove the fourth axis (num_channels)
    saliency = np.squeeze(np.maximum(grad_img_val,0), axis=(3,))
    return saliency

def main():

    if len(sys.argv)<2:
        print("usage:ml2017springhw3_saliency.py train_file")
        sys.exit()

    intrainf  = sys.argv[1]
    #outtestf = sys.argv[2]
    if os.path.exists('train.pkl')==True:
        with open("train.pkl", "rb") as pkf:
            train_X, train_y_label=pickle.load(pkf)
    else:
        #train_X, train_y, test_X = load(intrainf, intestf) 
        train_X, train_y_label  = load(intrainf)
        with open("train.pkl", "wb") as pkf:
            pickle.dump((train_X, train_y_label), pkf)

    #transfer the train_y to one hot encoding         
    train_y = np.zeros((len(train_y_label),7))
    train_y[np.arange(len(train_y_label)), train_y_label] = 1

    train_ix = np.random.permutation(len(train_X))
    
    train_X=train_X[train_ix]
    train_y=train_y[train_ix]
    
    mean_pixel = [125.0]
    dev_pixel = [63.0]
    
    #plotint a sample for checking                   
    sample_X = np.squeeze(train_X[:16].copy())
    sample_X.astype(np.int32)
    print("shape of image X[:16] is {}".format(sample_X.shape))

    sample_labels = np.argmax(train_y[:16], axis=-1)
    import matplotlib.pyplot as plt
    ip=0
    for label, plot in zip(sample_labels,sample_X):

        plt.subplot(4,4, ip+1)
        plt.imshow(plot, cmap='gray')
#        title = 'Actual index:'+str(label)
        plt.title('Actual index:'+str(label))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        #plt.imshow(sample_plot, cmap='gray')
        ip+=1

    plt.show()
    
    #preprocess, to standardizer the pixel value                                 
    for ch in range(num_channels):
        print("Normalize picture ch#{}".format(ch))
        #train_X[:,:,:,ch]=(train_X[:,:,:,ch]-mean_pixel[ch])/dev_pixel[ch]
        train_X[:,:,:,ch]=train_X[:,:,:,ch]/255.0

    #pick up random 16 pictures to draw their saliency map
    input_img = train_X[:16]
    input_label = train_y[:16]
    
    tf.reset_default_graph()
    #training variable for batch norm
    training = tf.placeholder_with_default(False,shape=(),name='training')
    
    #initialize place holder for input and output of model    
    X = tf.placeholder(tf.float32, shape=(batch_size, pheight, pwidth, num_channels), name="X")
    y = tf.placeholder(tf.int32, shape=(batch_size,7), name="y")
    #CNNmodel object                                           
    cnn = cnn_model(target_size=7)
    
    y_predict = cnn.model(X, batch_size, training=training)

    #loss and accuracy declaration 
    loss = cnn.loss_ce(y_predict, y)
    
    saver = tf.train.Saver()
    predict_index=[]
    DIR = "model/tf/"
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(DIR,"model_final"))        
        saliency_maps = get_saliency_map(X, y, input_img, input_label, y_predict, sess)
    
    #heat map threshold
    thres = 0.4
    
    for iv in range(len(saliency_maps)):
        sal_map = saliency_maps[iv]
        heat_map = process_heat_map(sal_map)
        label = sample_labels[iv]
        #plt.figure(iv+4)
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
        #first plot: the original plot
        
        #plt.subplot(1,3,1)
        ori_plot = np.squeeze(sample_X[iv])
        
        ca1 = ax1.imshow(ori_plot, cmap='gray')
        #plt.tight_layout()
        frame = plt.gcf()  
        title = 'Actual index:'+str(label)
        ax1.set_title(title)

        #second plot: the saliency heat map
        #plt.subplot(1,3,2)
        ca2 = ax2.imshow(heat_map, cmap=plt.cm.jet)
        cbar2 = fig.colorbar(ca2,ax=ax2)
        #plt.colorbar()
        plt.tight_layout()
        frame = plt.gcf()
        ax2.set_title('Saliency map')

        #third plot: mask plot of original plot
        see = sample_X[iv]
        see[np.where(heat_map <= thres)] = np.mean(see)
        #plt.subplot(1,3,3)
        ca3 = ax3.imshow(see, cmap='gray')
        #plt.colorbar()
        cbar3 = fig.colorbar(ca3, ax=ax3)
        plt.tight_layout()
        frame = plt.gcf()
        ax3.set_title('See map')

        #plt.savefig('saliency_map/saliplot_'+str(iv)+'.png')
        plt.show()

    return
    
if __name__=="__main__":

    main()
