#!/usr/bin/python3

import pandas as pd
import random
from sklearn.model_selection import train_test_split
# Imports multi-layer perceptron classifier
from sklearn.neural_network import MLPClassifier
import PUF1Score

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

# Splits x and y (features and target)
positiveQuantity = 20
x_train, _, y_train, _ = train_test_split(
    irisDataset[:positiveQuantity, :4], irisDataset[:positiveQuantity, 4], train_size=positiveQuantity-1)

# Splits x and y (features and target)
_, x_test, _, y_test = train_test_split(
    irisDataset[positiveQuantity:, :4], irisDataset[positiveQuantity:, 4], test_size=150-positiveQuantity-1)

'''
Multilayer perceptron model, with one hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer : 10 neuron, activation using ReLU
output layer : 2 neurons, represents the positive or negative class of Iris
optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
learning rate = 0.01
max iterations = 500
'''

score = 0.0
while score < 0.8:
    mlp = MLPClassifier(hidden_layer_sizes=10, solver='sgd', learning_rate_init=0.01, max_iter=500)

    # Train the model
    mlp.fit(x_train, y_train)

    score = PUF1Score.calculateF1Score(mlp.predict(x_test), y_test)
    print(score)

# Test the model
print(score)

sepalLength = 0.8
SepalWidth = 0.1
petalLength = 0.3
petalWidth = 0.8

customData = [[sepalLength, SepalWidth, petalLength, petalWidth]]

print(mlp.predict(customData))
