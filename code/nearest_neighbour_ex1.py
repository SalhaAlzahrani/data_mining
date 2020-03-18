# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:11:18 2020

@author: Dr. Salha Alzahrani
"""


# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Weather:", weather_encoded)
print("Temp:", temp_encoded)
print("Play:", label)

#Combinig weather and temp into single listof tuples
features=list(zip(weather_encoded,temp_encoded))
print("features:", features)

###############################################################################
# import Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7)

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)

#Predict Output
predicted= model.predict([[1,1]]) # 1:Rainy, 1:Hot
print("Predicted Value:", predicted)
