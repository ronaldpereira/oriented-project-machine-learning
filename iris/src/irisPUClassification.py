#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

from pywsl.pul import pu_mr
from pywsl.utils.comcalc import bin_clf_err

from pulib.pu_data import pn_from_dataframe, pu_from_y_train

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

for specie in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    # Loads the iris csv dataset
    irisDataset = pd.read_csv('../input/iris_dataset.csv')

    print('###', specie)

    irisDataset = pn_from_dataframe(irisDataset, 'Species', specie)
            
    # Splits x and y (features and target)
    train_size = int(sys.argv[1])

    x_train, x_test, y_train, y_test = train_test_split(
        irisDataset.drop(['Id','Species', 'y'],axis=1), irisDataset['y'].astype('int'), train_size=train_size, stratify=irisDataset['y'])

    pu_sl = pu_mr.PU_SL()

    y_train = pu_from_y_train(y_train, int(sys.argv[2]))

    # Train the model
    pu_sl.fit(x_train, y_train.values)

    # Test the model
    y_predict = pu_sl.predict(x_test)

    # Binary clf error calculation
    score = bin_clf_err(y_test, y_predict, prior=.5)
    print('\nbin_clf_err:\n\n', score)

    # Confusion Matrix calculation
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('\nConfusion Matrix:\n\n', conf_matrix)

    # ROC Curve calculation
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_test_values = y_test.values

    fpr[0], tpr[0], _ = roc_curve(y_test_values, y_predict)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_values, y_predict)
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('../output/rocCurves/pu/' + specie + '.png')

