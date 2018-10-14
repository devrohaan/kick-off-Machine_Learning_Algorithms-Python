#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x,y = make_blobs(n_samples=100,centers=5,random_state=0,cluster_std=0.6)
plt.scatter(x[:,0],x[:,1],s=50)

plt.show()

est = KMeans(5)
est.fit(x)
y_kmeans = est.predict(x)

plt.scatter(x[:,0],x[:,1],c=y_kmeans, s=50,cmap='rainbow')
plt.show()