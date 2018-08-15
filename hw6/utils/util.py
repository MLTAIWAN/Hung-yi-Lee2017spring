#!/bin/env python3
#-*- coding=utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import _pickle as pk
import jieba
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

class DataManager(object):
    def __init__(self):
        self.data = {}
        self.stop = stopwords.words('chinese')
        # Read data from data_path
        #  name       : string, name of data, usually 'train' 'test' 'semi'
        #  with_label : bool, read data with label or without label
    def add_data(self,name, data_path, with_label=False):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r') as f:
            for line in f:
                if with_label:
                    lines = line.strip().split(' +++$+++ ')
                    X.append(lines[1])
                    Y.append(int(lines[0]))
                else:
                    X.append(line)

        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]

    # Build dictionary
    #  vocab_size : maximum number of word in your dictionary
    def build_dict(self, keys,vocab_size, jiebacut=True):
        print ('build new vocabulary dictionary')
        # collection of words
        jieba.initialize()
        jieba.set_dictionary('../big5dict/dict.txt.big')
        words = []
        for key in keys:
            for sentence in self.data[key][0]:
                #words.extend(jieta.cut(sentence, cut_all=False))
                ws_insen = jieta.cut(sentence, cut_all=jiebacut)
                ws_remove = [ws for ws in ws_insen
                             if ws not in self.stop]
                words.extend(ws_remove)


        count = [['UNK', -1]]
        count.extend(Counter(words).most_common(vocab_size - 1))
        self.dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(self.dictionary)
            
        #data = list()
        unk_count = 0
        
        for word in words:
            index = self.dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
                #data.append(index)
    
        count[0][1] = unk_count
        self.reversed_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        #return  count, self.dictionary, self.reversed_dictionary
        print('Most common words (+UNK)', count[:5])
        return count


    # Save dictionary and reversed dictionary to specified path
    def save_dictionary(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump((self.dictionary, self.reversed_dictionary), open(path, 'wb'))
    

    # Load dictionary and reversed dictionary from specified path
    def load_dictionary(self,path):
        print ('Load dictionary from %s'%path)
        self.dictionary, self.reversed_dictionary = pk.load(open(path, 'rb'))
        
        return
    

    
