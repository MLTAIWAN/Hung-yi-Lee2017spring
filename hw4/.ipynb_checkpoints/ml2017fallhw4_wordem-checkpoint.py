#!/bin/env python3
#-*-coding=utf-8 -*-


import os, sys
import readline
import numpy as np
import tensorflow as tf
import pickle as pk
from utils.util import DataManager
from RNNmodel import simpleRNN

def main():
    parser = argparse.ArgumentParser(description='Sentiment classification')
    parser.add_argument('model')
    parser.add_argument('action', choices=['train','test','semi'])

    # training argument
    parser.add_argument('--batch_size', default=128, type=float)
    parser.add_argument('--nb_epoch', default=20, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--gpu_fraction', default=0.4, type=float)
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--max_length', default=40,type=int)
    
    # model parameter
    parser.add_argument('--loss_function', default='binary_crossentropy')
    parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
    parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
    parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
#    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--keep_prob', default=1.0, type=float)
    parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
    parser.add_argument('--threshold', default=0.1,type=float)
    # output path for your prediction
    parser.add_argument('--result_path', default='result.csv',)

    # put model in the same directory
    parser.add_argument('--load_model', default = None)
    parser.add_argument('--save_dir', default = 'model/')
    args = parser.parse_args()

    train_path = 'data/training_label.txt'
    test_path = 'data/testing_data.txt'
    semi_path = 'data/training_nolabel.txt'

    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    sess = get_session(args.gpu_fraction)
    
    

    
    return

if  __name__ == "__main__":
    main()

