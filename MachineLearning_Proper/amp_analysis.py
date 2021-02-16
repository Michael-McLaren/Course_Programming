#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:16:44 2020

@author: michaelmclaren
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(111)
data = np.load('amp_data.npz')['amp_data']

f = plt.figure(1)
plt.plot(data)
f.show()

g = plt.figure(2)
plt.hist(data, bins = 10**4)
g.show()

data = np.delete(data, [-1, -2, -3, -4, -5, -6])
data = data.reshape(1605394, 21)
data = np.random.permutation(data)
shuf_train, shuf_val, shuf_test = np.split(data, [1123000, 1363000], axis=0)

X_shuf_train = shuf_train[:,:20]
y_shuf_train = shuf_train[:,20:]
X_shuf_val = shuf_val[:,:20]
y_shuf_val = shuf_val[:,20:]
X_shuf_test = shuf_test[:,:20]
y_shuf_test = shuf_test[:,20:]

ab = 16
X = X_shuf_train[ab]
yy = y_shuf_train[ab][0]
t_grid = np.array([x/20 for x in range(20)])
t_grid = np.reshape(t_grid, (20,1))
plt.plot(t_grid, X, '.', markersize=5, mew=1)
plt.plot(1, yy, '.r', markersize = 7, mew = 1)

def phi_linear(tt):
    return np.hstack([np.ones((tt.shape[0],1)), tt])

def phi_quartic(tt):
    return np.hstack([np.ones((tt.shape[0],1)), tt, tt**2, tt**3, tt**4])




def fit_and_plot(phi, t_grid, X):
    w_fit = np.linalg.lstsq(phi(t_grid), X, rcond=None)[0]
    f_grid = phi(t_grid) @ w_fit
    plt.plot(t_grid, f_grid)
    
    
fit_and_plot(phi_quartic, t_grid, X)
fit_and_plot(phi_linear, t_grid, X)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.show()

