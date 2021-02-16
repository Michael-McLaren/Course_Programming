#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:37:42 2020

@author: michaelmclaren
"""

import numpy as np
from matplotlib import pyplot as plt

def sigmoid(X):
  return 1 / (1 + np.exp(-X))

def neural_net(X):
    h1 = sigmoid(X @ np.random.randn(1,100))
    h2 = sigmoid(h1 @ np.random.randn(100,50))
    ff = sigmoid(h2 @ np.random.randn(50,1))
    return ff

N = 100
X = np.linspace(-2, 2, num=N)[:, None]  # N,1

plt.clf()

for i in range(12):
    ff = neural_net(X)
    plt.plot(X, ff);