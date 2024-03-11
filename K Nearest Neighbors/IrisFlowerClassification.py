import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn import model_selection
from collections import Counter

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=1234)

class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        distances = [np.sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
knn = KNN(k=3)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
acc = np.sum(predictions == Y_test) / len(Y_test)
print(acc)
