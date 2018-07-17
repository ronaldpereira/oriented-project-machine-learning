#!/usr/bin/python3
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from pywsl.pul import pu_mr
from pywsl.utils.comcalc import bin_clf_err

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Transform specie string value to positive (1) and negative (0)
for specie in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    # Loads the iris csv dataset
    irisDataset = pd.read_csv('../input/iris_dataset.csv')

    print('###', specie)

    numberOfPositives = 0
    for index, sp in enumerate(irisDataset.loc[:, 'Species'].sample(frac=1)):
        if sp == specie and numberOfPositives < int(sys.argv[2]):
            irisDataset.loc[index, 'y'] = 1
            numberOfPositives += 1
        else:
            irisDataset.loc[index, 'y'] = 0
            
    # Splits x and y (features and target)
    train_size = int(sys.argv[1])

    x_train, x_test, y_train, y_test = train_test_split(
        irisDataset.drop(['Id','Species', 'y'],axis=1), irisDataset['y'], train_size=train_size, stratify=irisDataset['Species'])

    pu_sl = pu_mr.PU_SL()

    # Train the model
    pu_sl.fit(x_train, y_train)

    # Test the model
    y_predict = pu_sl.predict(x_test)

    score = bin_clf_err(y_test, y_predict, prior=.5)

    print('\nbin_clf_err:\n\n', score)

    print('\nConfusion Matrix:\n\n', confusion_matrix(y_test, y_predict))
