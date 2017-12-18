#!/bin/env python3
#-*- coding=utf-8 -*-

import numpy as np
from scipy.special import expit


#Logistic Regression class
class LogisticRegression(object):
    #initialize
    def __init__(self,epochs=500,eta=0.001, shuffle=True,random_state=None,earlystop=150, L2=0.001):
        np.random.seed(random_state)
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.w_initialized=False
        self.earlystop=earlystop
        self.L2=L2
        
    def fit(self, X, y, weightinit=False):
        #X:array-like shape=[n_samples, n_features], trainging vectors
        #y:array-like shape=[n_samples], means true label value
        self.n_features=X.shape[1]
        self.w_initialized=weightinit
        if (self.w_initialized==False):
            self.w_=self._initialize_weights()
        self.error_=[]
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
            #if i_trai % 500 ==0:
            #    print("Epoch {}, E_in is {}".format(i_trai+1,self.error_[-1]))
            #set early stop here                                                              
            if i_trai>50:
                if (self.error_[-1]>self.error_[-2]):
                    self.stopstep+=1
            
            self._update_weights(X,y)

        return self.error_
            
    #initialize the weighting vector with random number (-1,1)
    #dimension of weighting vector is n_features+1 (w0....wd)
    def _initialize_weights(self):
        return np.random.uniform(-1.0,1.0, size=self.n_features+1)

    def _sigmoid(self,z):
        return expit(z) #sigmoid function

    #shuffle the data
    def _shuffle(self, X, y):
        r=np.random.permutation(len(y))
        return X[r],y[r]
    #update the weighting vector
    def _update_weights(self,X,y):
        n_samples=X.shape[0]
        #put in sigmoid function result
        wx_=self._sigmoid(self.net_input(X))
            
        net_sig = np.subtract(y,wx_)
        
        gra = -1.0*np.dot(np.transpose(X),net_sig)+(2.0*self.L2*self.w_[1:])
        grab = -1.0*np.sum(net_sig)
        
        self.w_[1:] += self.eta*(-1.0/n_samples)*gra
        self.w_[0] += self.eta*(-1.0/n_samples)*grab
        
        """
    #obtain the Ein for each step (MSE)
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        pre_=self.predict(X)        
        Ein_p=np.square(np.subtract(pre_,y))
        return (1.0/n_samples)*np.sum(np.array(Ein_p))

    """
    #obtain the Ein for each step (cross-entropy)
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        wx_=self._sigmoid(self.net_input(X))
        #wx_=self.net_input(X)
        #print("shape of wx_ is {}".format(wx_.shape))
        #Ein_p = np.log(1+np.exp(-1*np.multiply(wx_, y))) #element wise multiply
        Ein_p = -1*(np.multiply(y,np.log(wx_))+np.multiply(1-y, np.log(1-wx_)))
        return (1.0/n_samples)*np.sum(Ein_p)
    
    #return wx
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    #return the prediction
    def predict(self, X):
        #return sigmoid function of wx
        z=self.net_input(X)
        return np.where(self._sigmoid(z)>=0.5,1,0)
