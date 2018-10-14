#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""



import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:\\Users\\User\\Desktop\\linreg.csv")

npMatrix = np.matrix(dataset)
X,Y = npMatrix[:,0], npMatrix[:,1]

regression = linear_model.LinearRegression()
regression.fit(X,Y)

a = regression.intercept_
b = regression.coef_[0]

print("Coefficients: this is the slope 'b' \n ", b)
print("Coefficients: this is the intercept 'a' \n ", a)
print("H(x) = ",a," + ",b,"*x")

plt.scatter(X,Y, color='black')
plt.plot(X, regression.predict(X),color='blue',linewidth=3)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)

#making new prediction with the model
print("Prediction: X=500 Y=", regression.predict(500))

plt.show()
