#!/bin/env python3
#-*-coding=utf-8 -*-


import os, sys
import readline
import argparse
import time
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import _pickle as pk
from utils.util import DataManager
from utils.batch_index import batch_index, get_batches, get_batches_nolabel
from RNNmodel import simpleRNN

def main():
    parser = argparse.ArgumentParser(description='Sentiment classification')
    parser.add_argument('model')
    parser.add_argument('action', choices=['train','test','semi'])

    # training argument
    parser.add_argument('--batch_size', default=64, type=float)
    parser.add_argument('--nb_epoch', default=30, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--gpu_fraction', default=0.4, type=float)
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('--max_length', default=40,type=int)
    parser.add_argument('--patience', default = 4, type=int)
    
    # model parameter
    parser.add_argument('--loss_function', default='binary_crossentropy')
    parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU'])
    parser.add_argument('-num_lay', '--num_layers', default=3, type=int)
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
    # log dir for tensorboard
    parser.add_argument('--log_dir', default='log_dir/')
    args = parser.parse_args()

    train_path = 'data/training_label.txt'
    test_path = 'data/testing_data.txt'
    semi_path = 'data/training_nolabel.txt'

    save_path = 'token/'
    #load token path
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)
        
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    sess = get_session(args.gpu_fraction)
    
    #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    else:
        dm.add_data('test_data', test_path, False)
        #raise Exception ('Implement your testing parser')
            
    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)
                            
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk')) 

    # convert to sequences
    dm.to_sequence(args.max_length)

    # Create the graph object
    tf.reset_default_graph() 
    # initial model
    print ('initial model...')
    rnnmodel = simpleRNN(args)    
    #print (model.summary())
    
    with tf.name_scope('inputs'):
        #create placeholder for training (testing) data 
        X_ = tf.placeholder(tf.int32, [None, args.max_length], name='X')
        y_ = tf.placeholder(tf.int32, [args.batch_size, ], name='y_')
        keep_prob = tf.placeholder_with_default(1.0, shape=(),name="keep_prob")
        
    y_predict = rnnmodel.model(args,X_, keep_prob)

    #prepare for saving model to evaluate
    train_var = [X_, y_, keep_prob, y_predict]
    tf.add_to_collection('train_var', train_var[0])
    tf.add_to_collection('train_var', train_var[1])
    tf.add_to_collection('train_var', train_var[2])
    tf.add_to_collection('train_var', train_var[3])
    
    #loss (MSE)
    mse = rnnmodel.loss(y_, y_predict)
    
    #optimizers
    train_op = rnnmodel.optimizer(args, mse)

    #accuracy for validation
    accuracy = rnnmodel.accuracy(y_, y_predict)

    #initial state
    init_state = rnnmodel.initial_state
    
    # merge the write out histogram plots (tensorboard)
    merged = tf.summary.merge_all()

    #check outputs of LSTM
    routputs = rnnmodel.outputs
    
    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model variables and keep training')
        path = os.path.join(load_path,'Sentimen_rnn_final.ckpt')
        if os.path.exists(path+".meta"):
            print ('load model from %s' % path)
            #model.load_weights(path) change to tensorflow model
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test' or args.action == 'semi':
        print ('Warning : testing or semi-training without loading any model')
        raise Exception ('Not loading model for test and semi-training...')        
        
    # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        print("Shape of X is {}, and y is {}".format(np.array(X).shape, np.array(Y).shape))

    elif args.action == 'test' :
        X = dm.get_data('test_data')
        print("Load test data (shape {})".format(X.shape))
        #raise Exception ('Implement your testing function')

    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        [semi_all_X] = dm.get_data('semi_data')
        
    init = tf.global_variables_initializer()
    
    #prepare to save model
    save_vars = tf.trainable_variables()
    saver = tf.train.Saver(save_vars, max_to_keep=7, keep_checkpoint_every_n_hours=1)

    last_loss = 1000000.0
    
    with tf.Session() as sess:
        init.run()
        train_writer = tf.summary.FileWriter(args.log_dir+'train', sess.graph)
        valid_writer = tf.summary.FileWriter(args.log_dir+'valid', sess.graph)
        # load variables in graphs if assigned
        if args.load_model is not None:
            saver.restore(sess, path)

        #if semi-learning, first apply model to semi-learning data
        if (args.action == 'semi' or args.action=='train'):
            #training 
            early_stop_counter = 0
            generation_num = 0
            # repeat nb_epoch times
            for e in range(args.nb_epoch):
                state = sess.run([init_state])
                semi_preds = []
                # add semi-data for the training
                if (args.action == 'semi' and e%2==0):
                    # label the semi-data
                    for ise, X_batch in enumerate(get_batches_nolabel(semi_all_X, args.batch_size)):
                        semi_dict = {X_:X_batch, init_state: state}
                        semi_pred = sess.run([y_predict], feed_dict=semi_dict)
                        #print("shape of semi_pred for each batch is {}".format(semi_pred.shape))
                        semi_preds.extend(semi_pred)

                    semi_X, semi_Y = dm.get_semi_data('semi_data', semi_preds, args.threshold, args.loss_function)
                    #combine labeled data with semi-data to training data
                    X_train = np.concatenate((semi_X, X))
                    Y_train = np.concatenate((semi_Y, Y))
                    print("shape of semi+train for each batch is X: {} y: {}".format(semi_X.shape, semi_Y.shape))
                    
                elif (e==0):
                    # hard copy
                    X_train = X.copy()
                    Y_train = Y.copy()
                
                #elif ( args.action='train'):
                #reset initial LSTM state every epochs
                n_batches = len(X)//args.batch_size
                for ix, (X_batch,y_batch) in enumerate(get_batches(X_train,Y_train, args.batch_size),1):
                        
                    generation_num += 1
                    train_dict = {X_:X_batch, y_:y_batch, keep_prob:args.keep_prob, init_state: state}
                    #for each traing generation, reload zero initial states
                    
                    _, summary, mse_train = sess.run([train_op, merged, mse], feed_dict=train_dict)
                    
                    train_writer.add_summary(summary, generation_num)
                    outputs_ = routputs.eval( feed_dict=train_dict)
                    #if (ix==1):
                        #print(X_batch[:10,:])
                        #print("shape of outputs is {}".format(outputs_[:,-1].shape))
                    
                    if (generation_num %10 ==0):
                        print("Epoch: {}/{}".format(e, args.nb_epoch),
                          "Iteration: {}".format(generation_num),
                          "Train loss: {:.3f}".format(mse_train))

                    #validation for each 50 generations or end of each epoch
                    if (generation_num %5 ==0 or ix==n_batches):
                        val_acc = []
                        val_loss = []
                        val_state = sess.run([init_state])
                        for iv, (X_batch, y_batch) in enumerate(get_batches(X_val, Y_val, args.batch_size),1):
                            val_dict = {X_: X_batch,
                                        y_: y_batch,
                                        keep_prob: 1,
                                        init_state: val_state}

                            summary, batch_acc, batch_loss = sess.run([merged, accuracy, mse], feed_dict=val_dict)
                            #print out some answer for checking
                            val_predict=sess.run(y_predict, feed_dict=val_dict)
                            #print("shape of val_predict is {}".format(np.array(val_predict).shape))
                            #last ten elements of each batch
                            
                            #for y_true, y_pre in zip(y_batch[-9:],val_predict[-9:]):
                            #    print("y_true: {}, y_predict: {}".format(y_true, y_pre))
                                                        
                            val_loss.append(batch_loss)
                            val_acc.append(batch_acc)
                            
                        print("Iteration: {}".format(generation_num),
                              "Val acc: {:.3f}".format(np.mean(val_acc)))
                        valid_writer.add_summary(summary, generation_num)
                        loss_val_avg = np.mean(val_loss)
                        #save variables every 50 generations
                        saver.save(sess, os.path.join(args.save_dir, "Sentimen_rnn"), global_step=generation_num)
                        
                        if(ix==n_batches): 
                            #early stop count here        
                            if (last_loss > loss_val_avg):
                                last_loss = loss_val_avg
                                early_stop_counter = 0
                            else:
                                early_stop_counter += 1
                        
                if (early_stop_counter>=args.patience or e==(args.nb_epoch-1)):
                    #save model                                                                   
                    saver.save(sess, os.path.join(args.save_dir, "Sentimen_rnn_final"))
                    saver.export_meta_graph(os.path.join(args.save_dir, "Sentimen_rnn_final.meta"), collection_list=['train_var'])
                    break
                
            print("End of training.....")
            
            
        #testing
        elif (args.action=='test'):
            
            raise Exception ('Implement your testing function')        
        
    return

if  __name__ == "__main__":
    main()

