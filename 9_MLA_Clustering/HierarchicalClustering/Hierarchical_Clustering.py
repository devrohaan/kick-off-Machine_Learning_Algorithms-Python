#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:40:00 2018

@author: Rohan
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

x = np.array([[1,1],[1.5,1],[3,3],[4,4],[3,3.5],[3.5,4]])
plt.scatter(x[:,0],x[:,1],s=50)

plt.show()

linkage_matrix = linkage(x, "single")

dendrogram = dendrogram(linkage_matrix,truncate_mode='none')
			   
plt.title("Hierarchical clustering")

plt.show()