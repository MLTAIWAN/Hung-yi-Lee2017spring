#!/bin/env python3
#-*- coding=utf-8 -*-
#training for sentiment classification

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
#A =  1.7159 #Amplitude of activation function
PATIENCE = 30
keep_drop = 0.5
classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
#for random number seeds of numpy
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
    
    """
    featuretest_list = []
    with open(testf, 'r', encoding='utf-8') as ftest:
        reader = csv.reader(ftest, delimiter=',', )
        for it,row in enumerate(reader):
            if it==0: #header
                continue
        
            feature_onesample = list(map(int, row[1].split(' ')))
            featuretest_list.append(feature_onesample)
            
    test_X = np.array(featuretest_list, dtype=np.float32)
    #reshape to 2d array
    test_X = np.reshape(test_X, (it,pwidth,pheight,1))
    """

    print("shape of training X is {}.".format(train_X.shape))
    print("shape of training Y is {}.".format(train_y.shape))
    #print("shape of test X is {}.".format(test_X.shape))
    
    return train_X, train_y #, test_X

def plot_confusion_matrix(cm, cmap, plt, classes=classes, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix
    
    #cmap = plt.cm.jet
    """
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

    
def main():
    if len(sys.argv)<2:
        #print("usage:ml2017springhw3.py train_file test_file out_file")
        print("usage:ml2017springhw3.py train_file")
        sys.exit()

    intrainf = sys.argv[1]
    #intestf  = sys.argv[2]
    #outtestf = sys.argv[3]

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
    #train_y = train_y_label
    
    #train_ix = range(len(train_X))
    train_ix = np.random.permutation(len(train_X))
    
    train_X=train_X[train_ix]
    train_y=train_y[train_ix]
    
    #plotint a sample for checking
    sample_plot = np.squeeze(train_X[40:48])
    sample_plot.astype(np.int32)
    print("shape of image X[40:48] is {}".format(sample_plot.shape))

    sample_labels = np.argmax(train_y[40:48], axis=-1)
    import matplotlib.pyplot as plt
    ip=0
    for label, plot in zip(sample_labels,sample_plot):
        
        plt.subplot(2,4, ip+1)
        plt.imshow(plot, cmap='gray')
        plt.title('Actual index:'+str(label))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)    
        #plt.imshow(sample_plot, cmap='gray')
        ip+=1
        
    plt.show()
            
    mean_pixel = [125.0]
    dev_pixel = [63.0]

    #preprocess, to standardizer the pixel value
    for ch in range(num_channels):
        print("normalize the picture ch{}".format(ch))
        #train_X[:,:,:,ch]=(train_X[:,:,:,ch]-mean_pixel[ch])/dev_pixel[ch]
        train_X[:,:,:,ch]=train_X[:,:,:,ch]/255.0
    
    train_ix = np.random.permutation(len(train_X))
        
    #first 3 thousand samples are used as validation
    valid_X = train_X[train_ix[:3000]]
    valid_y = train_y[train_ix[:3000]]
    #others is used as training samples
    train_X = train_X[train_ix[3000:]]
    train_y = train_y[train_ix[3000:]]
    #permutation again
    train_ix = np.random.permutation(len(train_X))

    
    # Create batches for each epoch
    num_batches = int(len(train_X)/batch_size) + 1
    # Split up text indices into subarrays, of equal size
    batches = np.array_split(train_ix, num_batches)
    # Reshape each split into [batch_size, training_seq_len]
    batches = [np.resize(x, [batch_size]) for x in batches]

    print("batch 0 is {}".format(batches[0]))
    valid_size = batch_size
    valid_num = int(3000.0/valid_size)
    iteration_count = 1

    valid_batches = np.array_split(range(3000), valid_num)
    valid_batches = [np.resize(x, [valid_size]) for x in valid_batches]

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
    accuracy = cnn.accuracy(y_predict, y)
        
    #generation_num = tf.Variable(0, trainable=False)
    generation_num = 0
    global_step = tf.Variable(0, trainable=False)
    train_op = cnn.optimizer(loss, global_step)
    
    #prepare for saving model to evaluate
    train_var = [X, y]
    tf.add_to_collection('train_var', train_var[0])
    tf.add_to_collection('train_var', train_var[1])
    #prepare the running session
    init = tf.global_variables_initializer()

    step_record = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    predict_index = []
    true_index = []
    output_every = 100
    last_loss = 100000.0
    # model plot
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y%m%d%H%M")
    tflog_dir = 'tf_logs'
    logdir = "{}/run_{}".format(tflog_dir, now)

    loss_summary = tf.summary.scalar('CE_Ein', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    #prepare to save model
    saver = tf.train.Saver(max_to_keep=7, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:
        init.run()
        early_stop_counter = 0
        #training stage
        for i in range(epochs):
            random.shuffle(batches)
            train_loss_epoch = []
            train_accu_epoch = []
            for ix, batch in enumerate(batches):
                X_batch = train_X[batch]
                y_batch = train_y[batch]
                generation_num += 1
                train_dict = {X:X_batch, y:y_batch, training:True}
                _, loss_train, accu_train= sess.run([train_op, loss, accuracy], feed_dict=train_dict)
                #accu_train = sess.run([accuracy_train], feed_dict=train_dict)
                loss_str=loss_summary.eval(feed_dict=train_dict)
                file_writer.add_summary(loss_str,generation_num)
                train_loss_epoch.append(loss_train)
                train_accu_epoch.append(accu_train)
                
                if (ix%50==0):
                    print('Generation {}: epoch {}: Iteraction {}: Loss E_in: {:.4f} Accuracy (train): {:.3f}'.format(generation_num, i, ix,loss_train,accu_train))
            
            step_record.append(i)
            train_loss_avg = np.mean(np.array(train_loss_epoch))
            train_accu_avg = np.mean(np.array(train_accu_epoch))
            train_loss.append(train_loss_avg)
            train_accuracy.append(train_accu_avg)
            #after every epoch, check the accuracy
            loss_val_batch=[]
            accu_val_batch=[]
            for iv, vbatch in enumerate(valid_batches):
                valid_X_batch = valid_X[vbatch]
                valid_y_batch = valid_y[vbatch]
                valid_dict = {X:valid_X_batch, y: valid_y_batch, training: False}
                #val_soft = y_predict.eval(feed_dict=valid_dict)
                loss_val, accu_val = sess.run([loss, accuracy], feed_dict=valid_dict)
                loss_val_batch.append(loss_val)
                accu_val_batch.append(accu_val)
                        
            loss_val_avg = np.mean(np.array(loss_val_batch))
            accu_val_avg = np.mean(np.array(accu_val_batch))
            print('\t\t Valid loss {:.4f}: Valid Accuracy {:.3f}%.'.format(loss_val_avg, accu_val_avg*100.0))
            valid_loss.append(loss_val_avg)
            valid_accuracy.append(accu_val_avg)
            #early stop count here
            if (last_loss > loss_val_avg):
                last_loss = loss_val_avg
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            """
            if (i % 20==0):
                print("print out first 20 validation prediction and true value to check")
                val_soft = y_est_valid.eval(feed_dict=valid_dict)
                print("shape of val_soft is {}".format(val_soft.shape))
                #val_head = np.argmax(val_soft,axis=-1)[0:20]
                val_heads = np.squeeze(val_soft)[0:20]
                for j, val_head in enumerate(val_heads):
                    print("predict {}: {}".format(j+1,val_head))
            """
            if (i%20==0):
                saver.save(sess, os.path.join(logdir, "model_cnn"), global_step=i)
            if (early_stop_counter>=PATIENCE or i==(epochs-1)):
                #save model
                saver.save(sess, os.path.join(logdir, "model_final"))
                saver.export_meta_graph(os.path.join(logdir, "model_final.meta"), collection_list=['train_var'])
                break
            #else:
            #    continue
                    
            #break
        
        print("End of training.....")
        #Get prediction of each batch
        for ix, batch in enumerate(batches):
            X_batch = train_X[batch]
            y_batch = train_y[batch]
            
            true_index.extend(np.argmax(y_batch,axis=-1))
            
            train_dict = {X:X_batch, y:y_batch, training:False}
            predict_batch = y_predict.eval(feed_dict=train_dict)            
            predict_index.extend(np.argmax(predict_batch,axis=-1))
            
    with open("predict.pkl", "wb") as pkf:
        pickle.dump((true_index, predict_index), pkf)


    file_writer.close()
    #import matplotlib.pyplot as plt
    
    plt.figure(4)
    #plot loss over generations
    plt.subplot(121)
    plt.plot(step_record, train_loss, 'k-', label='train batch loss(ce)')
    plt.plot(step_record, valid_loss, 'r--', label='valid loss(ce)')
    plt.title('Softmax loss per generations')
    plt.title('Epochs')
    plt.ylabel('Softmax loss + regular loss')
    plt.legend(loc='upper right')
    #plt.show()

    #plot accuracy over generations
    plt.subplot(122)
    plt.plot(step_record, train_accuracy, 'k', label='train accuracy')
    plt.plot(step_record, valid_accuracy, 'r', label='valid accuracy')

    #plt.plot(step_record, valid_loss, 'r', label='valid loss')
    plt.title('Accuracy of Train and Valid')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig('result/accuracy_loss.png')

    #confusion matrix
    print("Plot confusion matrix using training data....")
    true_index = np.array(true_index, dtype=np.int32)
    predict_index = np.array(predict_index, dtype=np.int32)

    conf_mat = confusion_matrix(true_index,predict_index)
    plt.figure(6)
    plot_confusion_matrix(conf_mat, cmap=plt.cm.jet, plt=plt, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    #plt.show()
    plt.savefig('result/confu.png')
    
    return


if __name__=="__main__":
    
    main()
