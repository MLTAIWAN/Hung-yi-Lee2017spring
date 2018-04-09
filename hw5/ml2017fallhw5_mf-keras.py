#!/bin/env python3
#*-*code=utf-8*-*
#using MatrixFactorization to complete the matrix elements prediction

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

batch_size = 64
epochs = 500
PATIENCE = 8

def get_model(n_users, n_items, latent_dim=6666):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec,item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = Model([user_input, item_input], r_hat)
    opt = SGD(lr=0.13,momentum=0.001,decay=0.001,nesterov=True)
    #opt = Adadelta(lr=0.13, rho=0.95, epsilon=1e-08)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model

def train(batch_size, num_epoch, pretrain, save_every, train_vec, train_rate, val_ratio=0.1, model_name=None, n_user=100, n_movie=100, latent_dim = 666, is_norm=False, rmean=3.0, rdev=1.0):

    if pretrain == False:
        model = get_model(n_user,n_movie,latent_dim)
    else:
        model = load_model(model_name)


    #prepare for output model filename
    final_f ='model/mf_model-final.h5'
    if (is_norm==True):
        final_f ='model/mf_model_norm-final.h5'
    
        
    #print(train_vec.shape)
    train_ix = np.random.permutation(train_vec.shape[0])
    num_val = int(val_ratio*1.0*train_vec.shape[0])
    print("numval is {}".format(num_val))

    
    #separate to validation set
    valid_vec  = train_vec[train_ix[:num_val]]
    valid_rate = train_rate[train_ix[:num_val]]
    #no-normalized rate
    if (is_norm==True):
        #no-normalized rate
        valid_TrueRate = np.around((valid_rate.copy())*rdev - rmean)
        #print("Shape of valid_TrueRate is {}".format(valid_TrueRate.shape))
    else:
        valid_TrueRate = valid_rate.copy()
        
    
    #separate to training set
    train_vec = train_vec[train_ix[num_val:]]
    train_rate = train_rate[train_ix[num_val:]]
    #print("shape of train_vec is {}".format(train_vec.shape))
    
    #from here, I borrowed from HW3 solution
    num_instances = len(train_rate)
    iter_per_epoch = int(num_instances / batch_size) + 1
    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start_t = time.time()
    best_metrics = 3000.0
    early_stop_counter = 0
    #for plot training loss
    loss_val = []
    epo_step = []
    for e in range(num_epoch):
        epo_step.append(e+1)
        #shuffle data in every epoch                                                      
        rand_idxs = np.random.permutation(num_instances)
        print ('#######')
        print ('Epoch ' + str(e+1))
        print ('#######')
        start_t = time.time()

        
        for i in range(iter_per_epoch):
            if i % 1000 == 0:
                print ('Iteration ' + str(i+1))
            user_batch = []
            movie_batch = []
            Y_batch = []
            ''' fill data into each batch '''
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                #if n==0:
                #    print("shape of user batch {}".format(train_vec[rand_idxs[n]].shape))
                user_batch.append(train_vec[rand_idxs[n],0])
                movie_batch.append(train_vec[rand_idxs[n],1])
                Y_batch.append(train_rate[rand_idxs[n]])

            #print("shape of user_batch is {}".format(train_vec.shape))
            #batch data to train model
            model.train_on_batch([ np.asarray(user_batch), np.asarray(movie_batch)], np.asarray(Y_batch))
        

        '''                                                                               
        The above process is one epoch, and then we can check the performance now.        
        '''
        #loss_and_metrics = model.evaluate([valid_vec[:,0],valid_vec[:,1]], valid_rate, batch_size)        
        #normalized or not
        predict_y = np.squeeze(model.predict([valid_vec[:,0],valid_vec[:,1]], verbose=1))
        if (is_norm==True):
            predict_y = (predict_y*rdev) - rmean

        #using around to predict rate
        #predict_y = np.around(predict_y)
        #print("shape: predict_y:{} valid_TrueRate {}".format(predict_y.shape,valid_TrueRate.shape))
        loss_rmse = np.sqrt(np.mean(np.square(valid_TrueRate - predict_y)))
        print ('\nloss :')
        print (loss_rmse)
        loss_val.append(loss_rmse)
        
        
        '''                                                                               
        early stop is a mechanism to prevent your model from overfitting                  
        '''
        if loss_rmse <= best_metrics:
            best_metrics = loss_rmse
            print ("save least loss!! "+str(loss_rmse))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        print ('Elapsed time in epoch ' + str(e+1) + ': ' + str(time.time() - start_t))
        if (e+1) % save_every == 0:
            model.save('model/mf_model-%d.h5' %(e+1))
            print ('Saved model %s!' %str(e+1))

        if early_stop_counter >= PATIENCE:
            print ('Stop by early stopping')
            print ('Least loss: '+str(best_metrics))
            print('Save model to file {}.'.format(final_f))
            model.save(final_f)
            break

        #last epoch, saving model
        if (e==(num_epoch-1)):
            print('Save model to file {}.'.format(final_f))
            model.save(final_f)

    print ('Elapsed time in total: ' + str(time.time() - total_start_t))
    
    import matplotlib.pyplot as plt
    plt.figure(4)
    #plot loss over epochs
    plt.plot(epo_step, loss_val, 'r-', label='valid loss(rmse)')
    plt.title('RMSE loss ')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    #plt.show()
    if (is_norm==True):
        figfn = "result/mfloss_rmse_norm_d%d.png" %(latent_dim)
        plt.savefig(figfn)
    else:
        figfn = "result/mfloss_rmse_d%d.png" %(latent_dim)
        plt.savefig(figfn)
    
    return model

def main():
    
    parser = argparse.ArgumentParser(prog='ml2017fallhw5_mf-keras.py')
    parser.add_argument('--user_file',type=str,dest="userf",default='users.csv')
    parser.add_argument('--movie_file',type=str,dest="movief",default='movies.csv')
    parser.add_argument('--train_file',type=str,dest="trainf",default='train.csv')
    #parser.add_argument('--isnorm',type=bool,dest='is_norm',default=False)
    parser.add_argument('--isnorm',dest='is_norm',action='store_true') #if add this option, True. otherwise False.
    parser.add_argument('--latent_dim',type=int,dest='latent_dim',default=30)
    parser.add_argument('--savetrain',dest='savetrain',action='store_true') #saving training prediction
    args = parser.parse_args()

    #check encoding
    #with open(args.movief,'rb') as rawdata_try:
    #    result_try = chardet.detect(rawdata_try.read(50000)) #look more to find the true encoding

    print("is_norm:{}\nlatent_dim:{}".format(args.is_norm, args.latent_dim))
    
    #print(result_try)
    df_movie = pd.read_csv(args.movief, sep='::', encoding='ISO-8859-1', engine='python')
    print(df_movie.tail(10))
    #check movie id unique
    movie_uid = df_movie['movieID'].unique()
    print("Number of movie id is {}".format(len(movie_uid)))
    n_movie = len(movie_uid)
    
    #Checking encoding
    #with open(args.userf,'rb') as rawdata_try:
    #    result_try = chardet.detect(rawdata_try.read(50000)) #look more to find the true encoding
    #print(result_try)    
    movie_dict = {}
    for mid in range(df_movie.shape[0]):
        #print('row {}, content {}'.format(mid, movie_r))
        movie_id = df_movie.loc[mid, 'movieID']
        movie_dict[movie_id] = mid

    #print(movie_dict.keys())
    
    df_user = pd.read_csv(args.userf, sep='::', encoding='ascii', engine='python')
    print(df_user.tail())
    
    #check user id unique
    user_uid = df_user['UserID'].unique()
    print("Number of user id is {}".format(len(user_uid)))
    n_user = len(user_uid)
    
    user_dict = {}
    for uid in range(df_user.shape[0]):
        user_id = df_user.loc[uid,'UserID']
        user_dict[user_id] = uid


    #saving user dict and movie dict for using at later tsne plot
    with open("id_dict.pkl", "wb") as idf:
        pickle.dump((user_dict, movie_dict), idf)
        
    #print(user_dict.keys())

    #Checking encoding
    #with open(args.trainf,'rb') as rawdata_try:
    #    result_try = chardet.detect(rawdata_try.read(50000)) #look more to find the true encoding    
    #print(result_try)

    df_rate = pd.read_csv(args.trainf, sep=',', encoding='utf-8', engine='python')    
    #print(df_rate.tail())

    #using map to make userid and movieid corresponding to new ids
    df_rate['userid']  = df_rate['UserID'].map(user_dict)
    df_rate['movieid'] = df_rate['MovieID'].map(movie_dict)

    train_meanr = df_rate['Rating'].mean()
    train_devr = df_rate['Rating'].std()
    print('Rate mean {}; deviation {}'.format(train_meanr, train_devr))

    with open('./meandev_train.txt', 'w') as fmean:
        write_md = 'mean: %f\ndev: %f' %(train_meanr, train_devr)
        fmean.write(write_md)
    
    #normalized rate
    df_rate['Rate_norm'] = df_rate['Rating'].map(lambda x: (x-train_meanr)/train_devr)
    print(df_rate.head())

    X_columns = ['userid','movieid']
    train_X = df_rate.loc[:,X_columns].values
    if (args.is_norm):
        #normalized rate
        train_y = df_rate.loc[:,'Rate_norm'].values
    else:
        #original rate
        train_y = df_rate.loc[:,'Rating'].values

    #start to train
    mf_model = train(batch_size=batch_size, num_epoch=epochs, pretrain=False, save_every=20,
                     train_vec=train_X, train_rate=train_y, val_ratio=0.1,
                     model_name=None, n_user=n_user, n_movie=n_movie,
                     latent_dim=args.latent_dim,
                     is_norm=args.is_norm, rmean=train_meanr, rdev=train_devr)

    user_emb = np.array(mf_model.layers[2].get_weights()).squeeze()
    print('user embedding shape:{}'.format(user_emb.shape))
    movie_emb = np.array(mf_model.layers[3].get_weights()).squeeze()
    print('movie embedding shape:{}'.format(movie_emb.shape))
    np.save('user_mf_emb.npy', user_emb)
    np.save('movie_mf_emb.npy', movie_emb)
    
    train_X = df_rate.loc[:,X_columns].values
    #prediction of training data and calculate loss value
    predict_y = mf_model.predict([train_X[:,0], train_X[:,1]], verbose=1)
    if (args.is_norm==True):
        predict_y = predict_y*train_devr + train_meanr

        
    #round to numbers
    predict_y = np.squeeze(np.around(predict_y))
    train_y = df_rate.loc[:,'Rating'].values
    loss_tot = np.sqrt(np.mean((train_y - predict_y)**2))

    print("\nRMSE of training data set is {} for MF mathod.".format(loss_tot))

    if (args.savetrain==True):
        if (args.is_norm==True):
            predictf = "result/predict_mf_norm_d%d.csv" %(args.latent_dim)
        else:
            predictf = "result/predict_mf_d%d.csv" %(args.latent_dim)

        with open(predictf, 'w') as pf:
            pf.write('true_rate, predict_rate\n')
            for tr, pr in zip(train_y,predict_y):
                pf.write('%d,%d\n' %(tr,pr))

        print('Save prediction result of training data at {}'.format(predictf))
        
    return


if __name__=="__main__":

    main()
