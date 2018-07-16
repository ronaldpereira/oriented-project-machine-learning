#!/usr/bin/python3
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import train_test_split
# Imports logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Loads the iris csv dataset
irisDataset = pd.read_csv('../input/iris_dataset.csv')

# Transform specie string value to positive (1) and negative (0)
for species in ['Iris-virginica','Iris-setosa','Iris-versicolor']:
    irisDataset['y'] = irisDataset['Species'] == species
    print(np.array(irisDataset['y']))

    # Converts all values to numeric
    #irisDataset = irisDataset.apply(pd.to_numeric)

    # Convert dataframe to matrix
    #irisDataset = irisDataset.values

    # Splits x and y (features and target)
    train_size = int(sys.argv[1])

    x_train, x_test, y_train, y_test = train_test_split(
        irisDataset.drop(['Id','y','Species'],axis=1), irisDataset['y'].astype('int'), train_size=train_size, stratify=irisDataset['Species'])

    logReg = LogisticRegression()

    # Train the model
    logReg.fit(x_train, y_train)

    # Test the model
    y_predict = logReg.predict(x_test)

    score = f1_score(y_test, y_predict)

    print('f1-score:', score)

    print(logReg.predict(x_test), y_test, sep=' ')

    sepalLength = 2.4
    SepalWidth = 1.8
    petalLength = 1.8
    petalWidth = 1.8

    customData = [[sepalLength, SepalWidth, petalLength, petalWidth]]

    print(logReg.predict(customData))
