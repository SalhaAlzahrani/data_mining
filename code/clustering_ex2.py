# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:59:41 2020

@author: Dr. Salha Alzahrani
"""
#Libs for Visualziation
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)

#Libs for Clustering
from sklearn.cluster import KMeans
from sklearn import datasets

#Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Build and Train your kmeans clustering
estimator = kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
#Labels of data points after clsutering
labels = estimator.labels_

#This part ofr 3D visulzation of resulted clusters
fignum = 1
title = '3 clusters'
fig = plt.figure(fignum, figsize=(4, 3))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title(title)
ax.dist = 12

fig.show()
