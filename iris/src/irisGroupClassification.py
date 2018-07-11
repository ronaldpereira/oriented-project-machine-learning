#!/usr/bin/python3

import pandas as pd
import random
from sklearn.model_selection import train_test_split
# Imports multi-layer perceptron classifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Loads the iris csv dataset
irisDataset = pd.read_csv('../input/iris_dataset.csv')

# Transform specie string value to int (0 = 'Iris-setosa', 1 = 'Iris-versicolor', 2 = 'Iris-virginica')
irisDataset.loc[irisDataset['Species'] == 'Iris-setosa', 'Species'] = 0
irisDataset.loc[irisDataset['Species'] == 'Iris-versicolor', 'Species'] = 1
irisDataset.loc[irisDataset['Species'] == 'Iris-virginica', 'Species'] = 2

# Converts all values to numeric
irisDataset = irisDataset.apply(pd.to_numeric)

# Convert dataframe to matrix
irisDataset = irisDataset.values

# Splits x and y (features and target)
x_train, x_test, y_train, y_test = train_test_split(
    irisDataset[:, 1:5], irisDataset[:, 5].astype('int'), train_size=0.8)

'''
Multilayer perceptron model, with one hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU (default)
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
y_predict = mlp.predict(x_test)

# Test the model
score = f1_score(y_test, y_predict, average='micro')

print('f1-score:', score)

print(mlp.predict(x_test), y_test, sep='\n')

# Creates a custom data and predicts it's group
sepalLength = 2.4
SepalWidth = 1.8
petalLength = 1.8
petalWidth = 1.8

customData = [[sepalLength, SepalWidth, petalLength, petalWidth]]

print(mlp.predict(customData))
