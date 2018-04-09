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
#from keras.models import Model, load_model
#from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, Embedding, Dot, Add
#from keras.optimizers import SGD, Adam, Adadelta

random_state=None
np.random.seed(random_state)

#using tsne in sklearn package
def draw(x,y):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    x = np.array(x,dtype=np.float64)
    y = np.array(y)
    # preform t-SNE embedding
    vis_data = TSNE(n_components=2).fit_transform(x)
    # plot the result
    vis_x = vis_data[:,0]
    vis_y = vis_data[:,1]

    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(vis_x, vis_y, c=y,cmap=cm)
    plt.colorbar(sc)
    plt.show()
    

def main():
    
    parser = argparse.ArgumentParser(prog='ml2017fallhw5_mftest-keras.py')
    parser.add_argument('--movie_file',type=str,dest="movief",default='movies.csv')
    parser.add_argument('--movie_emb',type=str,dest="m_emb",default='movie_mf_emb.npy')
    args = parser.parse_args()

    
    #retrieve user dict and movie dict for userid and movieid
    with open("id_dict.pkl", "rb") as idf:
        user_dict, movie_dict = pickle.load(idf)
        
    df_movie = pd.read_csv(args.movief, sep='::', encoding='ISO-8859-1', engine='python')


    df_movie['movieid'] = df_movie['movieID'].map(movie_dict)
    print(df_movie.tail())
    
    #X_columns = ['movieid']
    #test_X = df_movie.loc[:,X_columns].values

    genres_all = df_movie.loc[:,'Genres'].values
    genres_list = []
    for genre in genres_all:
        genres_list.extend(genre.split("|"))
        
    #print(genres_all)    
    genre_uni = np.unique(np.array(genres_list))
    print("Genres number is {}".format(len(genre_uni)))
    print(genre_uni)
    
    #decide label here
    genre_dict = dict()
    
    genre_dict["Action"] = 7
    genre_dict["Adventure"] = 1
    genre_dict["Animation"] = 1
    genre_dict["Children's"] = 1
    genre_dict["Comedy"] = 3
    genre_dict["Crime"] = 10
    genre_dict["Documentary"] = 6
    genre_dict["Drama"] = 5
    genre_dict["Fantasy"] = 4
    genre_dict["Film-Noir"] = 9
    genre_dict["Horror"] = 10
    genre_dict["Musical"] = 5
    genre_dict["Mystery"] = 8
    genre_dict["Romance"] = 4
    genre_dict["Sci-Fi"] = 2
    genre_dict["Thriller"] = 10
    genre_dict["War"] = 9
    genre_dict["Western"] = 9

    movie_ohvec = np.zeros((df_movie.shape[0], df_movie.shape[0]),dtype=np.float64)
    color_label = []
    for ir, row in df_movie.iterrows():
        #assign one-hot vector
        m_id = row["movieid"]
        movie_ohvec[ir,m_id] = 1
        #assign color label
        r_genre = np.array(row['Genres'].split("|"))
        #print(r_genre)
        #transfer string to number by genere_dict
        #vec_genre = [genre_dict[x] for x in r_genre]
        #vec_genre = np.array(vec_genre)
        vec_genre = np.vectorize(genre_dict.get)(r_genre)
        #print(vec_genre)
        #find max counts elements
        counts = np.bincount(vec_genre)
        if (max(counts)>1):
            label =np.argmax(counts)
        else:
            #if no count larger than 1, the first element is used
            label = vec_genre[0]
            
        color_label.append(label/10.0) #10 kinds of labels

    arr_gnenre = np.array(color_label)
    print("Shape of label array is {}".format(arr_gnenre.shape))

    if (os.path.exists(args.m_emb)==True):
        print("Load embedding matrix of movie from file {}".format(args.m_emb))
        m_emb = np.load(args.m_emb)
        print("Embedding matrix shape is {}".format(m_emb.shape))
    else:
        print("File does not exist, quit....")
        exit(1)

    embedded_movie = np.dot(movie_ohvec, m_emb)
    draw(embedded_movie, arr_gnenre)
    return


if __name__=="__main__":

    main()
