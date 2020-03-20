# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:40:25 2020

@author: Dr. Salha Alzahrani
"""
#import decition tree from sklearn
from sklearn import tree

#Suppose that your data is very simple
X = [[0, 0], [1, 1]]  #features
Y = [0, 1]            #labels or target

#Build your classifier
classifier = tree.DecisionTreeClassifier()
#Train your classifier
classifier = classifier.fit(X, Y)
#Finally use your classifier to predict unclassified / unseen instances
print(classifier.predict([[0., 0.5]]))
print(classifier.predict([[1., 1.5]]))

