# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:57:35 2020

@author: Dr. Salha Alzahrani
"""

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
