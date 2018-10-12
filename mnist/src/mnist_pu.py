import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score

from pywsl.pul import pu_mr
from pywsl.utils.comcalc import bin_clf_err
from pywsl.cpe.cpe_ene import cpe

import warnings
warnings.filterwarnings('ignore')


def mountTitanicSet(setArray, type='train'):
    pclassArray = setArray.loc[:, 'Pclass']

    sexArray = list(map(lambda sex: 0 if str.lower(sex) ==
                        'male' else 1, setArray.loc[:, 'Sex']))

    ageArray = list(map(lambda age: age/100 if not np.isnan(age)
                        else np.nanmean(setArray.loc[:, 'Age'] / 100), setArray.loc[:, 'Age']))

    sibSpParchArray = setArray.loc[:, 'SibSp'] + setArray.loc[:, 'Parch']

    fareArray = list(map(lambda fare: fare if str.lower(
        str(fare)) != 'nan' else 0, setArray.loc[:, 'Fare']))

    hasCabinArray = list(map(lambda cabin: 1 if str.lower(
        str(cabin)) != 'nan' else 0, setArray.loc[:, 'Cabin']))

    portsArray = ['nan', 'c', 'q', 's']

    embarkGroupArray = list(map(lambda port: portsArray.index(
        str.lower(str(port))), setArray.loc[:, 'Embarked']))

    if type == 'train':
        survivedArray = setArray.loc[:, 'Survived']

        trainingMatrix = np.array([pclassArray, sexArray, ageArray,
                                   sibSpParchArray, fareArray, hasCabinArray, embarkGroupArray])

        return np.transpose(trainingMatrix), survivedArray

    elif type == 'test':
        trainingMatrix = np.array([pclassArray, sexArray, ageArray,
                                   sibSpParchArray, fareArray, hasCabinArray, embarkGroupArray])

        return np.transpose(trainingMatrix)


np.set_printoptions(threshold=np.inf)
trainSet = pd.read_csv('../input/train.csv', sep=',')
testSet = pd.read_csv('../input/test.csv', sep=',')

x_train, y_train = mountTitanicSet(trainSet, 'train')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train.astype('int'), train_size=float(sys.argv[1]))

prior = cpe(x_train.loc[x_train==1], y_train, x_train.loc[x_train==0])

pu_sl = pu_mr.PU_SL(basis=sys.argv[3])

pu_sl.fit(x_train, y_train)

y_predict = pu_sl.predict(x_val)

score = bin_clf_err(y_val, y_predict)
print('\nbin_clf_err:\n\n', score)

# Confusion Matrix calculation
conf_matrix = confusion_matrix(y_val, y_predict)
print('\nConfusion Matrix:\n\n', conf_matrix)

# ROC Curve calculation
fpr = {}
tpr = {}
roc_auc = {}

fpr[0], tpr[0], _ = roc_curve(y_val, y_predict)
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_val, y_predict)
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
plt.title('Receiver operating characteristic for PU ' + sys.argv[3])
plt.legend(loc="lower right")
plt.savefig('../output/pu_' + sys.argv[3] + '.png')
