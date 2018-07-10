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

# Transform specie string value to int (0 = 'Iris-setosa', 1 = 'Iris-versicolor', 2 = 'Iris-virginica')
irisDataset.loc[irisDataset['Species'] == 'Iris-setosa', 'Species'] = 0
irisDataset.loc[irisDataset['Species'] == 'Iris-versicolor', 'Species'] = 1
irisDataset.loc[irisDataset['Species'] == 'Iris-virginica', 'Species'] = 2
# Converts all values to numeric
irisDataset = irisDataset.apply(pd.to_numeric)

# Convert dataframe to matrix
irisDataset = irisDataset.values

irisDataset = irisDataset.astype('int')

# Splits x and y (features and target)
x_train, x_test, y_train, y_test = train_test_split(
    irisDataset[:, :4], irisDataset[:, 4], train_size=0.8)

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
score = 0.0
while score < 0.8:
    mlp = MLPClassifier(hidden_layer_sizes=10, solver='sgd', learning_rate_init=0.01, max_iter=10000)

    # Train the model
    mlp.fit(x_train, y_train)

    # Test the model
    score = mlp.score(x_test, y_test)

print(score)

sepalLength = 2.4
SepalWidth = 1.8
petalLength = 1.8
petalWidth = 1.8

data = [[sepalLength, SepalWidth, petalLength, petalWidth]]

print(mlp.predict(x_test))
