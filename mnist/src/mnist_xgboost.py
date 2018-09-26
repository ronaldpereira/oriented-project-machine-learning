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

# ROC Curve calculation
fpr = {}
tpr = {}
roc_auc = {}

fpr[0], tpr[0], _ = roc_curve(df.y_val, y_predict)
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(df.y_val, y_predict)
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
plt.title('Receiver operating characteristic for Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('../output/titanic_logreg.png')
