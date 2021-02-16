#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:32:28 2020

@author: michaelmclaren
"""

import numpy as np
import matplotlib.pyplot as plt

def phi(C,K):
    times = np.arange(20 - C, 20, 1)/20
    r_times = times[::-1] #reverse it to start from lowest time
    features = np.array([r_times**i for i in range(K)])
    phi = features.T #change it into correct dimensions
    return phi

def make_vv(C, K):
    times = np.array([0.95- i/20 for i in range(C)])
    r_times = times[::-1] #reverse it to start from lowest time
    features = np.array([r_times**i for i in range(K)])
    phi = features.T #change it into correct dimensions
    phi_final = np.ones((K, 1)) #As t=1 here, this works
    v = phi @ (np.linalg.inv(phi.T @ phi)) @ phi_final
    return v

def square_error(x_row, y_row, c = 2, k = 2):
    #Reduce x row to correct dimensions
    x_row = np.flip(x_row)
    x_row = x_row[:c]
    x_row = np.flip(x_row)
    #Find the predicted value
    v_loop = make_vv(c,k)
    prediction = v_loop.T @ x_row
    return (y_row - prediction)**2

# Make splits reproducible
np.random.seed(seed=4131112)

# Create dataset of consecutive amplitudes
amp_data = np.load('amp_data.npz')['amp_data']
n_data = amp_data.size
chunk_width = 21
n_chunks = n_data // chunk_width  # Floor division
X = np.reshape(amp_data[:chunk_width*n_chunks], (n_chunks, chunk_width))

# Split into training, validation and test sets
X_shuf = np.random.permutation(X)
train_frac, val_frac = 0.7, 0.15
train_idx = int(np.floor(train_frac * n_chunks))
val_idx = train_idx + int(np.floor(val_frac * n_chunks))
X_shuf_train = X_shuf[:train_idx, :-1]
y_shuf_train = X_shuf[:train_idx, -1]
X_shuf_val = X_shuf[train_idx:val_idx, :-1]
y_shuf_val = X_shuf[train_idx:val_idx, -1]
X_shuf_test = X_shuf[val_idx:, :-1]
y_shuf_test = X_shuf[val_idx:, -1]

#Loop to find average weights for basis and least square methods


def two_method_error(c, k, x_row, y_row):
        #basis method for v
        #Reduce x row to correct dimensions
        x_row = np.flip(x_row)
        x_row = x_row[:c]
        x_row = np.flip(x_row)
        x_row = x_row[None,:]
        #Find v for basis and append
        v_basis = make_vv(c,k)
        v_fit = np.linalg.lstsq(x_row, np.array([[y_row]]), rcond=None)[0]
        V = np.hstack((v_basis, v_fit))
        v_average = np.mean(V, axis = 1)[:,None]
        prediction = v_average.T @ x_row.T #Wrong way round, creates c,c matrix otherwise
        error =  (y_row - prediction)**2
        return error


'''
#Find most effective c and k
N = 10**4
c_list = np.arange(1, 21)
k_list = np.arange(1, 11)
error_array = np.zeros((len(c_list), len(k_list)))
i = -1 #i,j used for indexing
for c in c_list:
    i = i + 1
    j = -1
    for k in k_list:
        j = j+ 1
        average_error = np.zeros((N,))
        #average error over N chunks
        for n in range(N):
            x_row = X_shuf_train[n]
            y_row = y_shuf_train[n]
            average_error[n] = two_method_error(c, k, x_row, y_row)
            
        error_array[i][j] = np.mean(average_error)

#Find min then index for it
min_ = np.amin(error_array)
c_best,k_best = np.where(error_array == min_)
'''

N = 10**4
mean_test = np.zeros((N,1))
for n in range(N):
    x_row_test = X_shuf_test[n]
    y_row_test = y_shuf_test[n]
    
    mean_test[n] = two_method_error(2, 2, x_row_test, y_row_test)
    

test_error = np.mean(mean_test)

