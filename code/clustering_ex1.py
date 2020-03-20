# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:49:14 2020

@author: Dr. Salha Alzahrani
"""
#import K-Means Clustering algorithm from SciKit-Learn
from sklearn.cluster import KMeans
#Numbers package
import numpy as np

#Define the dataset as an array of points
X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])

#Build and Train your kmeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

#Results
#Labels of data points after clsutering
print("Clusters' Labels: ")
print(kmeans.labels_)

#Predict the cluster of unclassified/unseen instance
print("Precit cluster of (0,0) and (12,3): ")
print(kmeans.predict([[0, 0], [12, 3]]))

#Show the clustering centers
print("Clusters' Centers: ")
print(kmeans.cluster_centers_)
