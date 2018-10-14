#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""


from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
y = dataset.target.reshape(-1,1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y)

train_features, test_features, train_targets, test_targets = train_test_split(features,targets, test_size=0.2)

model = Sequential()
# first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#we can define the loss function MSE or negative log lokelihood
#optimizer will find the right adjustements for the weights: SGD, Adagrad, ADAM ...
optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, batch_size=20, verbose=2)

results = model.evaluate(test_features, test_targets)

print("Accuracy on the test dataset: %.2f" % results[1])