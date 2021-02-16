#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:44:03 2020

@author: michaelmclaren
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10**3
D = 2
x = np.random.rand(N,D)

mat = np.random.rand(D,D)
A = (mat + mat.T)/2
c = np.random.rand(D,1)
precision = 2*A
sigma = np.linalg.inv(precision)

mu = -(1/2)*np.linalg.inv(A) @ c


def first_gaussian(xi, A, c, D):
    xi = np.reshape(xi, (D,1))
    return np.exp(-xi.T @ (A @ xi) - xi.T @ c)

def standard_gaussian(xi, precision, mu, D):
    xi = np.reshape(xi, (D,1))
    y = xi - mu
    exponent = y.T @ (precision  @ y) 
    return np.exp( (-1/2) * exponent)



f = np.array([standard_gaussian(xi, precision, mu, D) for xi in x])
g = np.array([first_gaussian(xi, A, c, D) for xi in x])
diff_array = np.divide(f,g)

