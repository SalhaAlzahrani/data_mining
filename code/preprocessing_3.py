# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:53:57 2020

@author: Dr. Salha Alzahrani
"""

from sklearn import preprocessing
import numpy as np

#normalization
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

X_normalized = preprocessing.normalize(X_train, norm='l2')
