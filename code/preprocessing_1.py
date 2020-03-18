# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:36:21 2020

@author: Dr. Salha Alzahrani
"""

from sklearn import preprocessing
import numpy as np

#scaling or mean removal
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

X_scaled = preprocessing.scale(X_train)

X_scaled.mean(axis=0)
X_scaled.std(axis=0)


