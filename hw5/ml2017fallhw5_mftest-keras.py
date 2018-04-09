#!/bin/env python3
#*-*code=utf-8*-*
#using MatrixFactorization to predict the test rate

import os, sys
import argparse
import chardet
import pandas as pd
import numpy as np
import time
import itertools
import pickle
#import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, Embedding, Dot, Add
from keras.optimizers import SGD, Adam, Adadelta

random_state=None
np.random.seed(random_state)

def predict( test_vec, model_f="model/mf_model-final.h5", is_norm=False, rmean=3.0, rdev=1.0):

    """
    #prepare for output model filename
    model_f ='model/mf_model-final.h5'
    if (is_norm==True):
        model_f ='model/mf_model_norm-final.h5'
    """    
    model = load_model(model_f)
    
    #normalized or not
    predict_y = np.squeeze(model.predict([test_vec[:,0],test_vec[:,1]], verbose=1))
    if (is_norm==True):
        predict_y = (predict_y*rdev) - rmean

    #using around to predict rate
    predict_y = np.around(predict_y)
    print("shape: predict_y:{} ".format(predict_y.shape))
    
    return predict_y

def main():
    
    parser = argparse.ArgumentParser(prog='ml2017fallhw5_mftest-keras.py')
    parser.add_argument('--test_file',type=str,dest="testf",default='test.csv')
    parser.add_argument('--isnorm',dest='is_norm',action='store_true') #if add this option, True. otherwise False.
    parser.add_argument('--modelf',type=str,dest='modelf',default='model/mf_model-final.h5')
    parser.add_argument('--test_submit',type=str,dest="testsub",default='testsubmit.csv')
    args = parser.parse_args()

    print("is_norm:{}\nmodel file:{}".format(args.is_norm, args.modelf))
    
    #retrieve user dict and movie dict for userid and movieid
    with open("id_dict.pkl", "rb") as idf:
        user_dict, movie_dict = pickle.load(idf)
        

    df_test = pd.read_csv(args.testf, sep=',', encoding='utf-8', engine='python')
    #print(df_test.tail())

    #using map to make userid and movieid corresponding to new ids
    df_test['userid']  = df_test['UserID'].map(user_dict)
    df_test['movieid'] = df_test['MovieID'].map(movie_dict)

    
    with open('./meandev_train.txt', 'r') as fmean:
        linemean = fmean.readline()
        linedev = fmean.readline()

    meanr = float(linemean.split(" ")[1])
    devr  = float(linedev.split(" ")[1])
    print('Rate mean {}; deviation {}'.format(meanr, devr))


    X_columns = ['userid','movieid']
    test_X = df_test.loc[:,X_columns].values


    #start to predict
    predict_y = predict(test_vec=test_X, model_f=args.modelf,
                      is_norm=args.is_norm, rmean=meanr, rdev=devr)
        
        
    
    with open(args.testsub, 'w') as pf:
        pf.write('TestDataID,Rating\n')
        for ir, pr in enumerate(predict_y):
            pf.write('%d,%d\n' %(ir,pr))
            
    print('Save prediction result of testing data at {}'.format(args.testsub))
        
    return


if __name__=="__main__":

    main()
