#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 13:22:50 2020

@author: michaelmclaren
"""

import support_code as sc
import numpy as np


def sigmoid(a):
    return 1/(1 + np.exp(-a))

keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
with np.load('ct_data.npz') as data:
    X_train = data[keys[0]]
    X_val = data[keys[1]]
    X_test = data[keys[2]]
    y_train = data[keys[3]]
    y_val = data[keys[4]]
    y_test = data[keys[5]]



alpha = 10
'''
# Root mean square error for training set
ww, bb = sc.fit_linreg_gradopt(X_train, y_train, alpha)
vector_pred = (y_train - (ww @ X_train.T + bb))**2
sum_pred = np.sum(vector_pred)
rmse_train = ((sum_pred) / len(y_train))**(1/2)
print(rmse_train)

# Root mean square error for validation set
vector_pred = (y_val - (ww @ X_val.T + bb))**2
sum_pred = np.sum(vector_pred)
rmse_val = ((sum_pred) / len(y_val))**(1/2)
print(rmse_val)

#Checking answer using horizontal baseline of the training set mean
base = np.mean(y_train)
vector_pred = (y_train - base)**2
sum_pred = np.sum(vector_pred)
rmse_base_train = ((sum_pred) / len(y_train))**(1/2)
print(rmse_base_train)

#RMSE on the Validation set for baseline
vector_pred = (y_val - base)**2
sum_pred = np.sum(vector_pred)
rmse_base_val = ((sum_pred) / len(y_val))**(1/2)
print(rmse_base_val)



#arbitrary classification
'''
alpha = 10
K = 10 # number of thresholded classification problems to fit
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
prediction_matrix_train = np.zeros((10, np.shape(y_train)[0])) #tranposed so its easier to slice
prediction_matrix_val = np.zeros((10, np.shape(y_val)[0]))

for kk in range(K):
    labels = y_train > thresholds[kk]
    ww, bb = sc.fit_logreg_gradopt(X_train, labels, alpha) #Same as linreg_gradopt except with log cost
    prediction_matrix_train[kk] = sigmoid(X_train @ ww + bb)
    prediction_matrix_val[kk] = sigmoid(X_val @ ww + bb)
    
prediction_matrix_train = prediction_matrix_train.T
prediction_matrix_val = prediction_matrix_val.T

#Fit linear regression with training prediction matrix and then RMSE
ww, bb = sc.fit_linreg_gradopt(prediction_matrix_train, y_train, alpha)
vector_pred = (y_train - (ww @ prediction_matrix_train.T + bb))**2
sum_pred = np.sum(vector_pred)
rmse_train = ((sum_pred) / len(y_train))**(1/2)
print(rmse_train)

#Fit linear regression with validation prediction matrix and then RMSE
vector_pred = (y_val - (ww @ prediction_matrix_val.T + bb))**2
sum_pred = np.sum(vector_pred)
rmse_val = ((sum_pred) / len(y_val))**(1/2)
print(rmse_val)