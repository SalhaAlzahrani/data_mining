# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:47:21 2020

@author: Dr. Salha Alzahrani
"""

from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_sel = sel.fit_transform(X)
