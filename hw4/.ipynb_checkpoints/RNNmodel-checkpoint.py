#!/bin/env python3
#-*- coding=utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf

# class that builds model 
class simpleRNN(object):

    def __init__(args):
        # Embedding layer
        # Define Embedding
        with tf.name_scope("embeddings"):
            self.embedding_mat = tf.get_variable('embedding_mat', [args.vocab_size, args.embedding_dim],
                                                 tf.float32, tf.random_normal_initializer())
            self.RNNcells=None
            


    def model(self, args, x_data):

        embedding_output = tf.nn.embedding_lookup(self.embedding_mat, x_data)
        #build RNN from here
        return_sequence = False
        dropout_rate = args.dropout_rate

        with tf.name_scope("RNNlayers"):
            if args.cell == 'GRU':
                grucell = tf.contrib.rnn.GRUCell(args.embedding_dim)
                drop = tf.contrib.rnn.DropoutWrapper(grucell, output_keep_prob=args.keep_prob)
                self.RNNcells = tf.contrib.rnn.MultiRNNCell([drop for _ in range(args.num_layers)])
            
                #RNN_cell = GRU(args.hidden_size, 
                #               return_sequences=return_sequence, 
                #               dropout=dropout_rate)
            elif args.cell == 'LSTM':
                #usage of peephole
                lstmcell = tf.contrib.rnn.LSTMCell(args.embedding_dim, use_peepholes=True)
                drop = tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=args.keep_prob)
                self.RNNcells = tf.contrib.rnn.MultiRNNCell([drop for _ in range(args.num_layers)])
                #RNN_cell = LSTM(args.hidden_size, 
                #                return_sequences=return_sequence, 
                #dropout=dropout_rate)
                
            #initial state for RNN cell 
            self.initial_state = self.RNNcells.zero_state(args.batch_size, tf.float32)

        with tf.name_scope("RNN_forward"):
            outputs, final_state = tf.nn.dynamic_rnn(self.RNNcells, embedding_output, initial_state=initial_state)
        #prediction use last output of each sample sentence in batch
        with tf.name_scope('predictions'):
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            tf.summary.histogram('predictions', predictions)

        return predictions


    def loss(self, labels,predictions):
        #loss function: MSE 
        cost = tf.losses.mean_squared_error(labels_, predictions)
        #record for tensorboard
        tf.summary.scalar('cost', cost)

    def optimizer(self, args, loss):
        #optim = tf.train.AdamOptimizer(args.learning_rate)
        optim = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate, rho=0.95, epsilon=1e-08)
        training_op = optim.minimize(loss)
        return training_op

    
