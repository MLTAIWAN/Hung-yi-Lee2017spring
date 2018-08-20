#!/bin/env python3
#-*- coding=utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import _pickle as pk
import jieba
import argparse
from skimage import io, transform
from collections import Counter
from nltk.corpus import stopwords
from glove import Corpus, Glove

#from gensim.models import word2vec

def build_dict( words, vocab_size):
    print ('build new vocabulary dictionary')
    # collection of words
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
            
    #data = list()
    unk_count = 0
        
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
            #data.append(index)
    
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #return  count, self.dictionary, self.reversed_dictionary
    print('Most common words (+UNK)', count[:5])
    return count, dictionary, reversed_dictionary

#resize picture to smaller size                            
def resizepic(large_img, args):
    new_shape = (args.newsize, args.newsize, 3)
    small_img = transform.resize(large_img, new_shape)

    return small_img

#minus avg of each channel                              
def minusavg(ori_img):
    img_shape = ori_img.shape
    img_chflat = np.reshape(ori_img, (img_shape[0]*img_shape[1], img_shape[2]))
    chn_avg = np.mean(img_chflat, axis=1)
    minus_maplist =[]
    chs = ["r","g","b"]
    for ich, ch in enumerate(chs):
        chmap = chn_avg[ich]*np.ones((img_shape[0], img_shape[1]))
        minus_maplist.append(chmap)
    minus_map = np.stack((minus_maplist[0],minus_maplist[1],minus_maplist[2]),axis=2)
    new_img = ori_img - minus_map

    return new_img, minus_map

    

    
