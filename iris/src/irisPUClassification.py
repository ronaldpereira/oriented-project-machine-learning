#!/usr/bin/python3

import pandas as pd
import random
from sklearn.model_selection import train_test_split
# Imports multi-layer perceptron classifier
from sklearn.neural_network import MLPClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Loads the iris csv dataset
irisDataset = pd.read_csv('../input/iris_dataset.csv')

# Transform specie string value to positive (1) and unlabeled (-1)
irisDataset.loc[irisDataset['Species'] == 'Iris-setosa', 'Species'] = 1
irisDataset.loc[irisDataset['Species'] == 'Iris-versicolor', 'Species'] = -1
irisDataset.loc[irisDataset['Species'] == 'Iris-virginica', 'Species'] = -1
# Converts all values to numeric
irisDataset = irisDataset.apply(pd.to_numeric)

# Convert dataframe to matrix
irisDataset = irisDataset.values

irisDataset = irisDataset.astype('int')

print(irisDataset)

# Splits x and y (features and target)
positiveQuantity = 60
x_train, _, y_train, _ = train_test_split(
    irisDataset[:positiveQuantity, :4], irisDataset[:positiveQuantity, 4], train_size=0.99)

# Splits x and y (features and target)
_, x_test, _, y_test = train_test_split(
    irisDataset[positiveQuantity:, :4], irisDataset[positiveQuantity:, 4], test_size=0.99)

'''
Multilayer perceptron model, with one hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 3 neurons, represents the class of Iris
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
max iterations = 500
'''

mlp = MLPClassifier(hidden_layer_sizes=10, solver='sgd', learning_rate_init=0.01, max_iter=500)

# Train the model
mlp.fit(x_train, y_train)

# Test the model
print(mlp.score(x_test, y_test))

sepalLength = 0.8
SepalWidth = 0.1
petalLength = 0.3
petalWidth = 0.8

data = [[sepalLength, SepalWidth, petalLength, petalWidth]]

print(mlp.predict(x_test))
