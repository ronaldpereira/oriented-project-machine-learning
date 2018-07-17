#!/usr/bin/python3
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import train_test_split
# Imports logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Loads the iris csv dataset
irisDataset = pd.read_csv('../input/iris_dataset.csv')

# Transform specie string value to positive (1) and negative (0)
for species in ['Iris-virginica','Iris-setosa','Iris-versicolor']:
    irisDataset['y'] = irisDataset['Species'] == species

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

    print('f1-score:\n', score)

    print('Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
