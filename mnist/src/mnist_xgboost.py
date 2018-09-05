#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class dataframe:
    def __init__(self, inputfolder='../input/'):
        self.train = pd.read_csv(inputfolder+'train.csv')
        self.test = pd.read_csv(inputfolder+'test.csv')

    def split_train_xy(self):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.train.drop('label', axis=1), self.train['label'].astype('int'), train_size=float(sys.argv[1]))


df = dataframe()

df.split_train_xy()

xgb_classifier = xgb.XGBClassifier()

xgb_classifier.fit(df.x_train, df.y_train)

y_predict = xgb_classifier.predict(df.x_val)

# F1 score calculation
score = f1_score(df.y_val, y_predict, average='macro')

print('\nf1-score:\n\n', score)

# Confusion Matrix calculation
conf_matrix = confusion_matrix(df.y_val, y_predict)
print('\nConfusion Matrix:\n\n', conf_matrix)

# Cross Validation score calculation with stratified k-fold=5
cross_val = cross_val_score(xgb_classifier, df.x_train, df.y_train, cv=5)
print('\nCross-validation:\n\nAccuracy: %0.2f (+/- %0.2f)\n\n' %(cross_val.mean(), cross_val.std()))
