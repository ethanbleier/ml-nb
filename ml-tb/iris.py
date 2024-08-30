#!/usr/bin/env python3
# Chapter 1 notebook
# pip install numpy scipy matplotlib ipython scikit-learn pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


iris_dataset = load_iris()
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print('****')

X_train, X_test, y_train, y_test = train_test_split(
  iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# View your data
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# scatter_matrix(iris_dataframe, figsize=(15, 15))
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Evaluating model performance
import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# Two methods for scoring 
# TODO: Confirm these are equivalent concepts
print("Test set score: {:.2f}%".format(np.mean(y_pred == y_test)))
print("Test set accuracy: {:.5f}".format(knn.score(X_test, y_test)))

