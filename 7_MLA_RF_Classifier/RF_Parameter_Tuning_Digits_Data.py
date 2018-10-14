#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

dataset = datasets.load_digits()

image_features = dataset.images.reshape((len(dataset.images), -1))
image_targets = dataset.target

random_forest_model = RandomForestClassifier(n_jobs=-1,max_features='sqrt')

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=.2)

param_grid = {
    "n_estimators" : [10,100,500,1000],
    "max_depth" : [1,5,10,15],
    "min_samples_leaf" : [1,2,3,4,5,10,15,20,30,40,50]              
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
grid_search.fit(feature_train, target_train)
print(grid_search.best_params_)

optimal_estimators = grid_search.best_params_.get("n_estimators")
optimal_depth = grid_search.best_params_.get("max_depth")
optimal_leaf = grid_search.best_params_.get("min_samples_leaf")

best_model = RandomForestClassifier(n_estimators=optimal_estimators, max_depth=optimal_depth, max_features='sqrt', min_samples_leaf = optimal_leaf)
k_fold = model_selection.KFold(n_splits=10, random_state=123)

predictions = model_selection.cross_val_predict(best_model, feature_test, target_test, cv=k_fold)
print("Accuracy of the tuned model: ", accuracy_score(target_test, predictions))