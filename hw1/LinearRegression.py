#!/bin/env python3
#-*- coding=utf-8 -*-

import numpy as np



#Linear Regression class
class LinearRegression(object):
    #initialize
    def __init__(self,epochs=500,eta=0.001, shuffle=True,random_state=None, adagrad=False,earlystop=150, L2=0.001):
        np.random.seed(random_state)
        self.epochs=epochs
        self.eta=eta
        self.shuffle=shuffle
        self.w_initialized=False
        self.adagrad=adagrad
        self.earlystop=earlystop
        self.w_=np.zeros((100,))
        self.L2=L2

    def fit(self, X, y, weightinit=False):
        #X:array-like shape=[n_samples, n_features], trainging vectors
        #y:array-like shape=[n_samples], means true label value
        self.n_features=X.shape[1]
        self.w_initialized=weightinit
        if (self.w_initialized==False):
            self.w_=self._initialize_weights()
        
        self.error_=[]
        self.prev_gra = 0
        self.prev_grab = 0
        self.stopstep = 0
        #loop for training
        for i_trai in range(self.epochs):
            if self.stopstep >=self.earlystop:
                print("Stop at Epoch {}, E_in is {}".format(i_trai+1,self.error_[-1]))
                break
            #data shuffle
            if self.shuffle:
                X,y = self._shuffle(X,y)

            self.error_.append(self._get_error(X,y))


            #set early stop here
            if i_trai>50:
                if (self.error_[-1]>self.error_[-2]):
                    self.stopstep+=1
                    
            if i_trai % 500 ==0:
                print("Epoch {}, E_in is {}".format(i_trai+1,self.error_[-1]))

            self._update_weights(X,y)
        return self.error_

    #initialize the weighting vector with random number (-1,1)
    #dimension of weighting vector is n_features+1 (w0....wd)
    def _initialize_weights(self):
        return np.random.uniform(-1.0,1.0, size=self.n_features+1)

    #shuffle the data
    def _shuffle(self, X, y):
        r=np.random.permutation(len(y))
        return X[r],y[r]
    #update the weighting vector
    def _update_weights(self,X,y):
        n_samples=X.shape[0]
        wx_=self.net_input(X)
        
        net_sig = np.subtract(y,wx_)
        
        gra = -2.0*np.dot(np.transpose(X),net_sig)+(2.0*self.L2*self.w_[1:])
        grab = -2.0*np.sum(net_sig)

        self.prev_gra += gra**2 
        self.prev_grab += grab**2
        if self.adagrad==True:
            ada = np.sqrt(self.prev_gra)
            adab = np.sqrt(self.prev_grab)
        else:
            ada=1.0
            adab=1.0

        self.w_[1:] += self.eta*(-1.0/n_samples)*gra/ada
        self.w_[0] += self.eta*(-1.0/n_samples)*grab/adab

    
    #obtain the Ein for each step (MSE)
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        wx_=self.net_input(X)
        Ein_p=np.square(np.subtract(wx_,y))
            
        return (1.0/n_samples)*np.sum(np.array(Ein_p))

    #return wx
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    #return the prediction
    def predict(self, X):
        #return sigmoid function of wx
        z=self.net_input(X)
        return z
