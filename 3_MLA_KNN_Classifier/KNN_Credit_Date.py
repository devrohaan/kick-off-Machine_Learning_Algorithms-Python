import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("/Users/Rohan/Desktop/3rdAug/udemy_ml/MLA/Datasets/credit_data.csv")

# Logistic regression accuracy: 93%
# we do better with knn: 98.5% !!!!!!!!
# 84%

#print(creditData.head())
#print(creditData.describe())
print(data.corr())

data.features = data[["income","age","loan"]]
data.target = data.default

data.features = preprocessing.MinMaxScaler().fit_transform(data.features) #HUGE DIFFERENCE !!!

feature_train, feature_test, target_train, target_test = train_test_split(data.features,data.target, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=28)  # k value !!!
fittedModel = model.fit(feature_train, target_train) # training
predictions = fittedModel.predict(feature_test) # testing

cross_valid_scores = []

for k in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,data.features,data.target,cv=10,scoring='accuracy')
    cross_valid_scores.append(scores.mean())
    

print("Optimal k with cross-validation: ", np.argmax(cross_valid_scores))    
    
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))