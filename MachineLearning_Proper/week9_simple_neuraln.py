#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:14:07 2020

@author: michaelmclaren
"""

import numpy as np
from scipy.optimize import minimize
import support_code_w9 as sc


def sigmoid(a):
    return 1/(1 + np.exp(-a))

def report_rmse(yy, pred):
    vector_pred = (yy - pred)**2
    sum_pred = np.sum(vector_pred)
    rmse = ((sum_pred) / len(yy))**(1/2)
    print(rmse)

keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
with np.load('ct_data.npz') as data:
    X_train = data[keys[0]]
    X_val = data[keys[1]]
    X_test = data[keys[2]]
    y_train = data[keys[3]]
    y_val = data[keys[4]]
    y_test = data[keys[5]]
 
K = 10
D = X_train.shape[1]
alpha = 10
args = (X_train, y_train, alpha)
#init parameters
ww_init = 0.1*np.random.randn(K)/np.sqrt(K)
bb_init = 0.1*np.random.randn()/np.sqrt(K)
V_init = 0.1*np.random.randn(K,384)/np.sqrt(K)
bk_init = 0.1*np.random.randn(K)/np.sqrt(K)
init = (ww_init, bb_init, V_init, bk_init)
#train neural net and report rmse
ww, bb, V, bk = sc.minimize_list(sc.nn_cost,init, args)
pred = sc.nn_cost([ww, bb, V, bk], X_train)
pred_val = sc.nn_cost([ww, bb, V, bk], X_val)
pred_test = sc.nn_cost([ww, bb, V, bk], X_test)
report_rmse(y_train, pred)
report_rmse(y_test, pred_test)
report_rmse(y_val, pred_val)

#Use previous method to find the initialisation of the parameters
mx = np.max(y_train)
mn = np.min(y_train)
hh = (mx-mn)/(K+1)
thresholds = np.linspace(mn+hh, mx-hh, num=K, endpoint=True)
V2_init = np.zeros((D, K))
bk2_init = np.zeros(K)
for kk in range(K):
    labels = y_train > thresholds[kk]
    V2_init[:,kk], bk2_init[kk] = sc.fit_logreg_gradopt(X_train, labels, alpha=10)

a = X_train @ V2_init + bk2_init
Z = sigmoid(a)
ww2_init, bb2_init = sc.fit_linreg_gradopt(Z, y_train, alpha=10)
#init parameters
V2_init = V2_init.T #transpose so it works
init2 = (ww2_init, bb2_init, V2_init, bk2_init)
#train and report rmse
ww2, bb2, V2, bk2 = sc.minimize_list(sc.nn_cost,init2, args)
pred2 = sc.nn_cost([ww2, bb2, V2, bk2], X_train)
pred2_val = sc.nn_cost([ww2, bb2, V2, bk2], X_val)
pred2_test = sc.nn_cost([ww2, bb2, V2, bk2], X_test)
report_rmse(y_train, pred2)
report_rmse(y_test, pred2_test)
report_rmse(y_val, pred2_val)