#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

creditData = pd.read_csv("/Users/Rohan/Desktop/3rdAug/udemy_ml/MLA/Datasets/credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 97.5% !!!!!!!!
# 84%

#print(creditData.head())
#print(creditData.describe())
#print(creditData.corr())

features = creditData[["income","age","loan"]]
targetVariables = creditData.default


featureTrain, featureTest, targetTrain, targetTest = train_test_split(features,targetVariables, test_size=0.3)

model = GaussianNB()  
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)

print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest,predictions))