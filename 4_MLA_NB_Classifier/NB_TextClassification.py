#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

trainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#print("\n".join(trainingData.data[1].split("\n")[:10]))
#print("Target is:", trainingData.target_names[trainingData.target[1]])

# we just count the word occurances
countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(trainingData.data)
#print countVectorizer.vocabulary_.get(u'software')

# we transform the word occurances into tfidf 
tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

model = MultinomialNB().fit(xTrainTfidf, trainingData.target)

new = ['This has nothing to do with church or religion', 'Software engineering is getting hotter and hotter nowadays']
xNewCounts = countVectorizer.transform(new)
xNewTfidf = tfidTransformer.transform(xNewCounts)

predicted = model.predict(xNewTfidf)

for doc, category in zip(new,predicted):
	print('%r --------> %s' % (doc, trainingData.target_names[category]))


