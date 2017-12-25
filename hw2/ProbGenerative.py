#!/bin/env python3
#-*- coding=utf-8 -*-

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss

#ProbGenerative class for impletement of probpabilstic generative model
class ProbGenerative(object):
    #initialize
    def __init__(self, random_state=False, n_features=100):
        np.random.seed(random_state)
        self.n_features=n_features
        self.w_=self._initialize_weights
        
    def fit(self, X, y, weightinit=False):
        #X:array-like shape=[n_samples, n_features], trainging vectors
        #y:array-like shape=[n_samples], means true label value
        self.n_features=X.shape[1]
        self.w_initialized=weightinit
        if (self.w_initialized==False):
            self.w_=self._initialize_weights()

        #class 1: > 50K, the label=1
        y_C1 = y[np.where(y==1)]
        X_C1 = X[np.where(y==1)]
        num_C1 = y_C1.shape[0]
        #print("shape of X_c1 is {}".format(X_C1.shape))
        #class 2: <=50K, the label=0
        y_C2 = y[np.where(y==0)]
        X_C2 = X[np.where(y==0)]
        num_C2 = y_C2.shape[0]
        #print("shape of X_C2 is {}".format(X_C2.shape))
        
        self.C1mu_ = (1.0/X_C1.shape[0])*np.sum(X_C1, axis=0)
        self.C2mu_ = (1.0/X_C2.shape[0])*np.sum(X_C2, axis=0)
        #print("C1 mu is {}".format(self.C1mu_))
        #print(self.C2mu_)
        #covariance matrix
        X1_mu = np.subtract(X_C1, self.C1mu_)
        C1cov = (1.0/num_C1)*np.dot(np.transpose(X1_mu), X1_mu)
        #print("C1conv shape is {}".format(C1cov.shape))
        X2_mu = np.subtract(X_C2, self.C2mu_)
        C2cov = (1.0/num_C2)*np.dot(np.transpose(X2_mu), X2_mu)
        
        self.CovM_ = (1.0*num_C1/(num_C1+num_C2))*C1cov + (1.0*num_C2/(num_C1+num_C2))*C2cov
        print("shape of CovM_ is {}".format(self.CovM_.shape))
        
        #fill the coefficient self.w_
        try:
            CovMInv = np.linalg.inv(self.CovM_)
        except np.linalg.LinAlgError:
            print("Can not obtain invere matrix of covariance matrix")
            return False
            
        
        #mu1-mu2
        mu1_mu2 = np.reshape(self.C1mu_ - self.C2mu_, (self.C1mu_.shape[0],1))
        self.w_[1:] = np.dot(np.transpose(mu1_mu2), CovMInv)

        u1 = np.reshape(self.C1mu_,(self.C1mu_.shape[0],1))
        u2 = np.reshape(self.C2mu_,(self.C2mu_.shape[0],1))
        u1_CovMInv_u1 = (-1.0/2)*np.dot(np.dot(np.transpose(u1),CovMInv),u1)
        u2_CovMinv_u2 = ( 1.0/2)*np.dot(np.dot(np.transpose(u2),CovMInv),u2)
        lnN1N2 = np.log(1.0*num_C1/num_C2)
        self.w_[0] = u1_CovMInv_u1 + u2_CovMinv_u2 + lnN1N2
        print("weight is {}".format(self.w_))
        self.error_ = self._get_error(X,y)
        return self.error_
            
    #initialize the weighting vector with random number (-1,1)
    #dimension of weighting vector is n_features+1 (w0....wd)
    def _initialize_weights(self):
        return np.random.uniform(-1.0,1.0, size=self.n_features+1)

    def _sigmoid(self,z):
        #return expit(z) #sigmoid function
        res = 1.0/(1.0+np.exp(-z))
        return np.clip(res, 0.1e-12, 0.999999999)
    
    #obtain the Ein for each step (error rate)
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        pre_=self.predict(X)        
        Ein_p=np.absolute(np.subtract(pre_,y))
        return (1.0/n_samples)*np.sum(Ein_p)

    """
    #obtain the Ein for each step (cross-entropy)
    def _get_error(self, X, y):
        n_samples=X.shape[0]
        wx_=self._sigmoid(self.net_input(X))
        #pred_y = self.predict(X)
        #print("wx_ shape is {}".format(wx_.shape))
        #print(" ")
        #print("shape of wx_ is {}".format(wx_.shape))
        
        #Ein_p = -1*(np.multiply(y,np.log(wx_))+np.multiply(1-y, np.log(1-wx_)))
        print("mean of y is {}".format(y.mean()))
        Ein_p = log_loss(y, wx_)
        print("Ein {}".format(Ein_p))
        return (1.0/n_samples)*np.sum(Ein_p)
    
"""
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    #return the prediction
    def predict(self, X):
        #return sigmoid function of wx
        z=self.net_input(X)
        return np.where(self._sigmoid(z)>=0.5,1,0)
