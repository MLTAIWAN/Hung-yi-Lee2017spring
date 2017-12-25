#!/bin/env python3
#-*- coding=utf-8 -*-

import csv
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ProbGenerative import ProbGenerative
#only import kFold for cross-validation
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss


#load and clean train and test arrays
def load(inftrain, inftest):
    df_train = pd.read_csv(inftrain, sep=',',header=0, skipinitialspace=True)
    income_map = {'<=50K': 0, '>50K': 1}
    sex_map = {'Female': 0, 'Male': 1}
    df_train['income']=df_train['income'].map(income_map)
    df_train['sex']=df_train['sex'].map(sex_map)

    objlist = list(df_train.select_dtypes(include=['object']).columns)

    #one-hot encoding for non-numerical data column
    df_train=pd.get_dummies(df_train,prefix=None, prefix_sep='_', drop_first=False,columns=objlist,sparse=True)

    #retrieve the income column
    label_train = df_train['income'].values
    #Then, we can drop the income column
    df_train=df_train.drop('income', axis=1)
    df_test = pd.read_csv(inftest, sep=',',header=0, skipinitialspace=True)
    
    df_test['sex']=df_test['sex'].map(sex_map)
    df_test=pd.get_dummies(df_test,prefix=None, prefix_sep='_', drop_first=False,columns=objlist,sparse=True)
    #print(df_test.info())

    lost_element=[x for x in df_train.columns if (x in df_test.columns)==False]
    #print ("lost elements: ",lost_element)
    testLength = len(df_test.index)
    #print("length {}".format(testLength))
    for elem in lost_element:
        df_test = df_test.assign(newcolumn=pd.Series(np.zeros(testLength, dtype=np.int64)).values)
        df_test.rename(columns={'newcolumn':elem}, inplace=True)

    #print(df_test.head(5))
    
    extra_element=[x for x in df_test.columns if (x in df_train.columns)==False]
    #print ("extra elements: ", extra_element)
    trainLength = len(df_train.index)
    for elem in extra_element:
        df_train = df_train.assign(newcolumn=pd.Series(np.zeros(trainLength, dtype=np.int64)).values)
        df_train.rename(columns={'newcolumn':elem}, inplace=True)
    
    #print(df_train.head(5))
    train_columns = list(df_train.columns.values)
    #print(train_columns)
    train_X = df_train[train_columns].values
    test_X = df_test[train_columns].values
    return train_X, label_train, test_X, train_columns

def main():
    if len(sys.argv)<4:
        print("usage:ml2017springhw2_probgene.py trainfile testfile outfile")
        sys.exit(0)

    intrainf = sys.argv[1]
    intestf = sys.argv[2]
    outtestf = sys.argv[3]

    train_X, train_y, test_X, columns = load(intrainf, intestf)
    print("Shape of train_X is {}".format(train_X.shape))
    print("Shape of train_y is {}".format(train_y.shape))
    print("Shape of test_X is {}".format(test_X.shape))

    print(columns)

    std_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    std_index=[]
    #there should be only one element that is corresponding to the element in std_columns, so we can use index approach
    for std_column in std_columns:
        cinde = columns.index(std_column)
        std_index.append(cinde)

    #for decision trees, we don't have to standarize
    params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

    gbc = GradientBoostingClassifier(**params)
    #K-fold validation (10)
    kf = KFold( n_splits=10, shuffle=True)
    vali_error = []
    for train_index, vali_index in kf.split(train_X):
        trainKF_X, valiKF_X = train_X[train_index], train_X[vali_index]
        trainKF_y, valiKF_y = train_y[train_index], train_y[vali_index]

        gbc.fit(trainKF_X, trainKF_y)
        pred_y = np.array(gbc.predict(X=valiKF_X))
        Evali_p = np.absolute(np.subtract(valiKF_y, pred_y))
        Evali = (1.0/valiKF_X.shape[0])*np.sum(Evali_p)
        vali_error.append(Evali)
        
    vali_errarr = np.array(vali_error)
    #print the validation error of KFold validation
    print('KFold validation error of GradientBDT is mean {}, deviation {}.'.format(vali_errarr.mean(), vali_errarr.std()))
    
    #use all training data to fit
    gbc.fit(train_X,train_y)

    #Ein using all training data
    y_pred = gbc.predict(train_X)
    Evali_p = np.absolute(np.subtract(valiKF_y, pred_y))
    Evali = (1.0/valiKF_X.shape[0])*np.sum(Evali_p)
    print("Ein for GredientBDT is {}".format(Evali))
    
    test_ypred = gbc.predict(test_X)
    print("Shape of test predicted y is {}".format(test_ypred.shape))
    #test_ypred = np.reshape(test_ypred,(test_ypred.shape[1],test_ypred.shape[0]))
    print('write output to file {}'.format(outtestf))
    with open(outtestf, 'w') as f:
        f.write('id,label\n')
        for id,value in enumerate(test_ypred):
            f.write('%d,%d\n' %(id, value))

    return
    
if __name__=="__main__":
    
    main()
