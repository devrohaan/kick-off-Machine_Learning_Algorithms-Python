#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan

If the user can pay back the loan
"""



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

creditData = pd.read_csv("/Users/Rohan/Desktop/3rdAug/udemy_ml/MLA/Datasets/credit_data.csv")

print(creditData.head())
print(creditData.describe())
print(creditData.corr())

features = creditData[["income","age","loan"]]
target = creditData.default

feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
predictions = model.fit.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))