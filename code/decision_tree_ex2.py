# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:54:21 2020

@author: Dr. Salha Alzahrani
"""
#Load dataset
from sklearn.datasets import load_iris
#import decition tree from sklearn
from sklearn import tree

#Load features X and targets Y
X, Y = load_iris(return_X_y=True)

#Build and train your classifier
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X, Y)

#Predict unclassifed instance
print(classifier.predict([[5, 5, 5, 2]]))

