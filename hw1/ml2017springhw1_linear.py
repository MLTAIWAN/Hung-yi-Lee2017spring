#!/bin/env python3
#-*- coding=Big5 -*-

import csv
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
#only import kFold for cross-validation
#from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold

def trainclean(inf):
    trow = []
    with open(inf, 'r',encoding='Big5') as ftrain:
        reader = csv.reader(ftrain)
        for row in reader:
            trow.append(row)

    #prepare a list of 18 empty list elements
    Data=list()
    for _, i in enumerate(range(18)):
        Data.append(list())
        for _, j in enumerate(range(12)):
            Data[i].append(list())
            
    #preparing the header for the out train_X csv file    
    for i,row in enumerate(trow[1:]):
        for j, num in enumerate(row):
            if j>2:
                imonth = int(i/(18*20))
                Data[i%18][imonth].append(num)            
                #print(len(Data[0][0]))
                
    train_x = []
    train_y = []
    feature_hours = 9
    feature_num = 18
    for imonth in range(12):
        hours_9 = len(Data[0][imonth])-10+1
        for ihour in range(hours_9):
            sample_x = []
            sample_y = 0
            for ifea in range(feature_num):
                for ih in range(9):
                    i_row = ihour+ih
                    element = 0.0 if Data[ifea][imonth][i_row]=='NR'else Data[ifea][imonth][i_row]
                    try:
                        sample_x.append(float(element))
                    except ValueError:
                        sample_x.append(0.0)
                    
            #ph2.5 value in 10th hour
            try:
                sample_y = float(Data[9][imonth][ihour+9])
            except IndexError:
                print("imonth {}, ihour+9 {}".format(imonth,ihour+9))
        
            #now, fill the cleaned train data
            train_x.append(sample_x)
            train_y.append(sample_y)

    NPtrain_X = np.array(train_x, dtype=np.float64)
    NPtrain_Y = np.array(train_y, dtype=np.float64)

    print("shape of training X is {}.".format(NPtrain_X.shape))
    print("shape of training Y is {}.".format(NPtrain_Y.shape))

    return (NPtrain_X,NPtrain_Y)

def testclean(inf):
    trow = []
    with open(inf, 'r',encoding='Big5') as ftest:
        reader = csv.reader(ftest)
        for row in reader:
            trow.append(row)

    #prepare a list of 18 empty list elements
    Data=list()
    for _, i in enumerate(range(18)):
        Data.append(list())
            
    #no header for testing data
    for i,row in enumerate(trow):
        for j, num in enumerate(row):
            if j>1:                
                Data[i%18].append(num)

    print(len(Data))
    print(len(Data[17]))
                
    test_x = []
    feature_hours = 9
    feature_num = 18

    sample_num = len(Data[0])
    for isample in np.arange(0, sample_num, 9):
        sample_x = []
        for ifea in range(feature_num):
            for ih in range(9):
                i_row = isample+ih
                element = 0.0 if Data[ifea][i_row]=='NR'else Data[ifea][i_row]
                try:
                    sample_x.append(float(element))
                except ValueError:
                    sample_x.append(0.0)
                     
        test_x.append(sample_x)

    NPtest_X = np.array(test_x, dtype=np.float64)
    print("shape of testing X is {}.".format(NPtest_X.shape))

    return NPtest_X
    
def main():
    if len(sys.argv)<4:
        print("usage:dataclean.py trainfile testfile outfile")
        sys.exit(0)

    intrainf = sys.argv[1]
    intestf = sys.argv[2]
    outtestf = sys.argv[3]
    
    train_X,train_y = trainclean(intrainf)
    test_X = testclean(intestf)

    train_Xstd = np.zeros(train_X.shape)
    test_Xstd  = np.zeros(test_X.shape)
    #standardize the training data and the testing data
    for j in range(train_X.shape[1]):
        train_Xstd[:,j] = (train_X[:,j]-train_X[:,j].mean())/(train_X[:,j].std()+1e-6)
        #1e-6 to avoid the zero division
        test_Xstd[:,j] = (test_X[:,j]-train_X[:,j].mean())/(train_X[:,j].std()+1e-6)
    
    #linear regression
    LRmodel = LinearRegression(epochs=20000, eta=0.017, shuffle=True, random_state=None, adagrad=False,earlystop=50)
    #K-fold validation (10)
    #kf = KFold(train_Xstd.shape[0], n_fold=10)
    kf = KFold( n_splits=10, shuffle=True)
    vali_error = []
    for train_index, vali_index in kf.split(train_Xstd):
        trainKF_X, valiKF_X = train_Xstd[train_index], train_Xstd[vali_index]
        trainKF_y, valiKF_y = train_y[train_index], train_y[vali_index]
        
        LRmodel.fit(trainKF_X, trainKF_y)
        vali_error.append(LRmodel._get_error(valiKF_X, valiKF_y))
        
    vali_errarr = np.array(vali_error)
    #print the validation error of KFold validation
    print('KFold validation error of Linear Regression is mean {}, deviation {}.'.format(vali_errarr.mean(), vali_errarr.std()))
    
    #use all training data to fit
    err_hist = LRmodel.fit(train_Xstd,train_y)
    
    plt.plot(err_hist,color='red',linestyle='-',label="LinearRegression")
    plt.xlabel('epoch')
    plt.ylabel('Ein')
    plt.legend(loc='upper right')
    #plt.ylim(0.3, 0.71)                                                                              
    plt.show()

    test_ypred = np.ceil(LRmodel.predict(test_Xstd))
    print("Shape of test predicted y is {}".format(test_ypred.shape))
    #test_ypred = np.reshape(test_ypred,(test_ypred.shape[1],test_ypred.shape[0]))
    print('write output to file {}'.format(outtestf))
    with open(outtestf, 'w') as	f:
        f.write('id,value\n')
        for id,value in	enumerate(test_ypred):
            f.write('id_%d,%d\n' %(id, value))
    #np.savetxt(outtestf,test_ypred,fmt="%4d",delimiter='')    

    
if __name__=="__main__":
    
    main()
