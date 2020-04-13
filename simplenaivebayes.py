# -*- coding: utf-8 -*-
"""naivebayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13hyxUhTeCjYI7yPnUwwkQsoG_NLBG9qq
"""

from sklearn.datasets import load_iris as LoadData
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn import metrics as met

iris = LoadData()
X = iris.data
y = iris.target

"""Splitting into training and testing sets"""
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.4, random_state=1)
  
# training set is loaded to fit
gnb = GNB()
gnb.fit(X_train, y_train) 
  
# make prediction
predicted_y = gnb.predict(X_test)
print("Gaussian Naive Bayes model accuracy(in %):", met.accuracy_score(y_test, predicted_y)*100)