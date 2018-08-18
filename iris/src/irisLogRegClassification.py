#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import sys

from sklearn.model_selection import train_test_split
# Imports logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

from pulib.pu_data import pn_from_dataframe, pos_from_y_train

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

for specie in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    # Loads the iris csv dataset
    irisDataset = pd.read_csv('../input/iris_dataset.csv')

    print('###', specie)
    irisDataset = pn_from_dataframe(irisDataset, 'Species', specie)
    irisDataset['y'].loc[irisDataset['y'] == -1] = 0

    # Splits x and y (features and target)
    train_size = int(sys.argv[1])

    x_train, x_test, y_train, y_test = train_test_split(
        irisDataset.drop(['Id','y','Species'],axis=1), irisDataset['y'].astype('int'), train_size=train_size, stratify=irisDataset['Species'])

    y_train = pos_from_y_train(y_train, sys.argv[2])

    logReg = LogisticRegression()

    # Train the model
    logReg.fit(x_train, y_train)

    # Test the model
    y_predict = logReg.predict(x_test)

    # F1 score calculation
    score = f1_score(y_test, y_predict, average='macro')

    print('\nf1-score:\n\n', score)

    # Confusion Matrix calculation
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('\nConfusion Matrix:\n\n', conf_matrix)

    # Cross Validation score calculation with stratified k-fold=5
    cross_val = cross_val_score(logReg, x_train, y_train, cv=5)
    print('\nCross-validation:\n\nAccuracy: %0.2f (+/- %0.2f)\n\n' %(cross_val.mean(), cross_val.std()))

    # ROC Curve calculation
    fpr = {}
    tpr = {}
    roc_auc = {}

    fpr[0], tpr[0], _ = roc_curve(y_test, y_predict)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_predict)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + specie)
    plt.legend(loc="lower right")
    plt.savefig('../output/rocCurves/logReg/' + specie + '.png')
