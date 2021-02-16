#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 18:05:42 2020

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

def find_ck(x_row, y_row):
    c_list = np.arange(1, 21)
    k_list = np.arange(1, 11)
    error_array = np.zeros((len(c_list), len(k_list)))
    i = -1
    for c in c_list:
        i = i + 1
        j = -1
        for k in k_list:
            j = j+ 1
            x_row_copy = np.copy(x_row)
            #Reduce x row to correct dimensions
            x_row_copy = np.flip(x_row_copy)
            x_row_copy = x_row_copy[:c]
            x_row_copy = np.flip(x_row_copy)
            #Find the predicted value
            v_loop = make_vv(c,k)
            prediction = v_loop.T @ x_row_copy
            error_array[i][j]=  (y_row - prediction)**2
    return error_array

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
'''

#Find error on least square method
c = 20
N = 10**4
x_row = np.zeros((N,c))
y_row = np.zeros((N,))
for n in range(N):
    x_row_full_e = X_shuf_val[n]
    y_row_e = y_shuf_val[n]
    #reduce x_row to c dimensions
    x_row_e = np.flip(x_row_full_e)
    x_row_e = x_row_e[:c]
    x_row_e = (np.flip(x_row_e))
    #append to x matrix and y vector
    y_row[n] = y_row_e
    x_row[n] = x_row_e
#Find the fit
v_fit = np.linalg.lstsq(x_row, y_row, rcond=None)[0]
v_fit = np.reshape(v_fit, (1,c))
#Find mean square error
sqr_error = (y_row - v_fit @ x_row.T)**2
least_sqr_error = np.mean(sqr_error)

#Find error on basis method
mean_test = np.zeros((N,1))
for n in range(N):
    x_row_test = X_shuf_test[n]
    y_row_test = y_shuf_test[n]
    mean_test[n] = square_error(x_row_test, y_row_test)
    
basis_error = np.mean(mean_test)


'''

C = 21
mean_list_train = np.zeros((C-1,1))
mean_list_val = np.zeros((C-1,1))
i = -1
for c in range(1,C):
    i += 1
    N = 10**4
    x_row_train = np.zeros((N,c))
    y_row_train = np.zeros((N,))
    x_row_val = np.zeros((N,c))
    y_row_val = np.zeros((N,))
    for n in range(N):
        x_row_full1 = X_shuf_train[n]
        y_row1 = y_shuf_train[n]
        #reduce x_row to c dimensions
        x_row1 = np.flip(x_row_full1)
        x_row1 = x_row1[:c]
        x_row1 = np.flip(x_row1)
        #append to x matrix and y vector
        y_row_train[n] = y_row1
        x_row_train[n] = x_row1
        
        #same for validation
        x_row_full2 = X_shuf_val[n]
        y_row2 = y_shuf_val[n]
        x_row2 = np.flip(x_row_full2)
        x_row2 = x_row2[:c]
        x_row2 = np.flip(x_row2)
        y_row_val[n] = y_row2
        x_row_val[n] = x_row2
    #Find the fit with training set
    v_fit = np.linalg.lstsq(x_row_train, y_row_train, rcond=None)[0]
    v_fit = np.reshape(v_fit, (1,c))
    #Find mean square error for training set
    sqr_error_train = (y_row_train - v_fit @ x_row_train.T)**2
    mean_sqr_error_train = np.mean(sqr_error_train)
    mean_list_train[i] = mean_sqr_error_train
    #Find mean square error for validation set
    sqr_error_val = (y_row_val - v_fit @ x_row_val.T)**2
    mean_sqr_error_val = np.mean(sqr_error_val)
    mean_list_val[i] = mean_sqr_error_val
    
#find miniumum of both training and validation sets
min_c_train = np.argmin(mean_list_train)
min_c_val = np.argmin(mean_list_val)

'''

N = 10**4
mean_train = np.zeros((N,1))
mean_val = np.zeros((N,1))
mean_test = np.zeros((N,1))
for n in range(N):
    x_row_train = X_shuf_train[n]
    y_row_train = y_shuf_train[n]
    x_row_val = X_shuf_val[n]
    y_row_val = y_shuf_val[n]
    x_row_test = X_shuf_test[n]
    y_row_test = y_shuf_test[n]
    
    mean_train[n] = square_error(x_row_train, y_row_train)
    mean_val[n] = square_error(x_row_val, y_row_val)
    mean_test[n] = square_error(x_row_test, y_row_test)
    
train = np.mean(mean_train)
val = np.mean(mean_val)
test = np.mean(mean_test)
'''

# Loop over square error
#Finding most effective c and k, basis method
'''
row_id = 0
N = 10**4
x_row_train = X_shuf_train[row_id]
y_row_train = y_shuf_train[row_id]
array = square_error(x_row, y_row)
for n in range(N):
    x_row = X_shuf_train[row_id + n + 1] #Change row used
    y_row = y_shuf_train[row_id + n + 1]
    array_dstack = find_ck(x_row, y_row)
    array = np.dstack((array, array_dstack)) #stack into 3d square error array
    
            
#convert 3d array into 2d mean array, find min of array, index for best (c,k)
mean_array = np.mean(array, axis =2)
min_ = np.amin(mean_array)
i,j = np.where(mean_array == min_)
print('c' , np.arange(1, 21)[i])
print('k' , np.arange(1, 11)[j])'''