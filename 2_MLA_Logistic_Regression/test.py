import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# p i = 1 / 1 + exp[ - ( b0 + b1 * x1 + b2 * x2 + b3 * x3 )]

	#	balance  |        income          |  age     |           class
	#	 10000$		      80 000$            35       1 ( can pay back the debt )
	# 	 7000$		      120 000$           57       1 ( can pay back the debt )
	# 	 100$		      23 000$             22       0 ( can NOT pay back the debt )
	# 	 223$		      18 000$             26       0 ( can NOT pay back the debt )
	# ----------------------------------------------------------------------
	# 	 5500$		      50 000$             25       ? make a prediction

X = np.array([[10000,80000,35],[7000,120000,57],[100,23000,22],[223,18000,26]])
Y = np.array([1,1,0,0])

classifier = LogisticRegression()
classifier.fit(X,Y)

print(classifier.predict_proba([1000,30000,24]))