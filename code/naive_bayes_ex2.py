# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 04:50:04 2020

@author: Dr. Salha Alzahrani
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#load features X, and class y
X, y = load_iris(return_X_y=True)

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

#create Naive Bayes Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train,y_train)

#Make predictions for the test dataset
y_pred = model.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
