# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:50:10 2020

@author: Dr. Salha Alzahrani
"""
from sklearn import preprocessing
import numpy as np

#scaling to a range [0,1]
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)

