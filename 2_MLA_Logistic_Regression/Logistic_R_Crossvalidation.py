#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

creditData = pd.read_csv("/Users/Rohan/Desktop/3rdAug/udemy_ml/MLA/Datasets/credit_data.csv")

features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
predicted = cross_validation.cross_val_predict(model,features,target, cv=10)

print(accuracy_score(target,predicted))