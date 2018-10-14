#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def isTasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0

data = pd.read_csv("/Users/Rohan/Desktop/3rdAug/udemy_ml/MLA/Datasets/wine.csv", sep=";")

#print(data['quality'].describe())
#print(data['quality'].value_counts())

features = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar","chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
data['tasty'] = data["quality"].apply(isTasty)
targets = data['tasty']

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)

param_dist = {
 'n_estimators': [50,100,200],
 'learning_rate' : [0.01,0.05,0.1,0.3,1],
}

grid_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_dist, cv=10)
grid_search.fit(feature_train, target_train)

preds = grid_search.predict(feature_test)

print(confusion_matrix(target_test, preds))
print(accuracy_score(target_test, preds))