#!/bin/env python3
#-*- coding=utf-8 -*-
#training for sentiment classification

import os, sys
import csv
import string
import random
import pickle 
import itertools
import numpy as np
import pandas as pd
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

nb_class = 7
from sklearn.metrics import confusion_matrix


random_state=None
batch_size = 400
epochs = 100
pwidth = 48
pheight = 48
num_channels = 1
A =  1.7159 #Amplitude of activation function
PATIENCE = 8
classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
#for random number seeds of numpy
np.random.seed(random_state)

def load(trainf, testf):
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
    

    print("shape of training X is {}.".format(train_X.shape))
    print("shape of training Y is {}.".format(train_y.shape))
    
    return train_X, train_y

def plot_confusionmatrix(cm, plt, cmap, classes=classes, title='Confusion matrix'):
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

def build_model():

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    input_img = Input(shape=(48, 48, 1))
    '''
    先來看一下keras document 的Conv2D
    keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
        padding='valid', data_format=None, dilation_rate=(1, 1),
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)
    '''
    
    block1 = Conv2D(64, (5, 5), padding='valid', kernel_initializer='he_normal')(input_img)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)
    
    block2 = Conv2D(64, (3, 3), kernel_initializer='he_normal')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), kernel_initializer='he_normal')(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), kernel_initializer='he_normal')(block3)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), kernel_initializer='he_normal')(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)
    
    fc1 = Dense(1024, kernel_initializer='he_normal')(block5)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu', kernel_initializer='he_normal')(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model



def train(batch_size, num_epoch, pretrain, save_every, train_pixels, train_labels, val_pixels, val_labels, model_name=None):

    if pretrain == False:
        model = build_model()
    else:
        model = load_model(model_name)

    '''
    "1 Epoch" means you have been looked all of the training data once already.
    Batch size B means you look B instances at once when updating your parameter.
    Thus, given 320 instances, batch size 32, you need 10 iterations in 1 epoch.
    '''

    num_instances = len(train_labels)
    iter_per_epoch = int(num_instances / batch_size) + 1
    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start_t = time.time()
    best_metrics = 0.0
    early_stop_counter = 0
    for e in range(num_epoch):
        #shuffle data in every epoch
        rand_idxs = np.random.permutation(num_instances)
        print ('#######')
        print ('Epoch ' + str(e+1))
        print ('#######')
        start_t = time.time()

        for i in range(iter_per_epoch):
            if i % 10 == 0:
                print ('Iteration ' + str(i+1))
            X_batch = []
            Y_batch = []
            ''' fill data into each batch '''
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                X_batch.append(train_pixels[rand_idxs[n]])
                Y_batch.append(train_labels[rand_idxs[n]])
                #X_batch[-1] = np.fromstring(X_batch[-1], dtype=float, sep=' ').reshape((48, 48, 1))
                #Y_batch[-1][int(train_labels[rand_idxs[n]])] = 1.

            ''' use these batch data to train your model '''
            model.train_on_batch(np.asarray(X_batch),np.asarray(Y_batch))

        '''
        The above process is one epoch, and then we can check the performance now.
        '''
        loss_and_metrics = model.evaluate(val_pixels, val_labels, batch_size)
        print ('\nloss & metrics:')
        print (loss_and_metrics)

        '''
        early stop is a mechanism to prevent your model from overfitting
        '''
        if loss_and_metrics[1] >= best_metrics:
            best_metrics = loss_and_metrics[1]
            print ("save best score!! "+str(loss_and_metrics[1]))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        
        #Sample code to write result :
        if e%2 == 0:
            val_proba = model.predict(val_pixels)
            val_classes = val_proba.argmax(axis=-1)


            with open('result/simple%s.csv' % str(e), 'w') as f:
                f.write('acc = %s\n' % str(loss_and_metrics[1]))
                f.write('id,label')
                for i in range(len(val_classes)):
                    f.write('\n' + str(i) + ',' + str(val_classes[i]))
        

        print ('Elapsed time in epoch ' + str(e+1) + ': ' + str(time.time() - start_t))

        if (e+1) % save_every == 0:
            model.save('model/model-%d.h5' %(e+1))
            print ('Saved model %s!' %str(e+1))

        if early_stop_counter >= PATIENCE:
            print ('Stop by early stopping')
            print ('Best score: '+str(best_metrics))
            break

    print ('Elapsed time in total: ' + str(time.time() - total_start_t))
    return model
    
def main():
    if len(sys.argv)<2:
        #print("usage:ml2017springhw3.py train_file test_file out_file")
        print("usage:ml2017springhw3.py train_file test_file out_file")
        sys.exit()

    intrainf = sys.argv[1]
    intestf  = sys.argv[2]
    outtestf = sys.argv[3]

    if os.path.exists('train.pkl')==True:
        with open("train.pkl", "rb") as pkf:
            train_X, train_y_label=pickle.load(pkf)
    else:
        #train_X, train_y, test_X = load(intrainf, intestf)
        train_X, train_y_label  = load(intrainf, intestf)
        with open("train.pkl", "wb") as pkf:
            pickle.dump((train_X, train_y_label), pkf)
            
    
    #transfer the train_y to one hot encoding
    train_y = np.zeros((len(train_y_label),7))
    train_y[np.arange(len(train_y_label)), train_y_label] = 1
    #train_y = train_y_label

    #train_ix = range(len(train_X))
    train_ix = np.random.permutation(len(train_X))
    #first 3 thousand samples are used as validation
    valid_X = train_X[train_ix[:3000]]
    valid_y = train_y[train_ix[:3000]]
    #others is used as training samples
    train_X = train_X[train_ix[3000:]]
    train_y = train_y[train_ix[3000:]]
    #permutation again
    train_ix = np.random.permutation(len(train_X))

    #plotint a sample for checking
    sample_plot = np.squeeze(train_X[40])
    sample_plot.astype(np.int32)
    print("shape of image X[40] is {}".format(sample_plot.shape))

    import matplotlib.pyplot as plt
    #plt.imshow(sample_plot, cmap='gray')
    #plt.show()

    # Create batches for each epoch
    num_batches = int(len(train_X)/batch_size) + 1
    # Split up text indices into subarrays, of equal size
    batches = np.array_split(train_ix, num_batches)
    # Reshape each split into [batch_size, training_seq_len]
    batches = [np.resize(x, [batch_size]) for x in batches]
    
    valid_size = 3000
    iteration_count = 1
    

    step_record = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    predict_index = []
    true_index = []
    
    # model plot
    from datetime import datetime
    #now = datetime.utcnow().strftime("%Y%m%d%H%M")

    # start training
    model_=train(batch_size, epochs, pretrain=False, save_every=10,
          train_pixels=train_X, train_labels=train_y,
          val_pixels=valid_X, val_labels=valid_y,
          model_name=None)

    predictions = model_.predict(train_X)
    predictions = predictions.argmax(axis=-1)
    print("Plot confusion matrix using training data....")
    true_index = np.squeeze(train_y)
    predict_index = np.squeeze(np.ndarray.astype(predictions, dtype=np.int32))
    print("true shape is {}".format(true_index))
    print("predict shape is {}".format(predict_index.shape))


    conf_mat = confusion_matrix(true_index,predict_index)
    plt.figure(6)
    plot_confusion_matrix(conf_mat, plt=plt,cmap=plt.cm.jet, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.show()
    
    return


if __name__=="__main__":
    
    main()
