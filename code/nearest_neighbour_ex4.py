# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:24:03 2020

@author: Dr. Salha Alzahrani
"""

from sklearn.neighbors import KNeighborsRegressor

# dataset (X=m^2, y=rental price)
X = [[40], [45], [60], [70]]
y = [1000, 1200, 2000, 2500]

# fit
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y)

# predict
print('Monthly Rental Price for 65m^2 in $'),
print(neigh.predict([[65]]))