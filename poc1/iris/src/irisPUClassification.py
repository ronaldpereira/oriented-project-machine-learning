import pandas as pd
import matplotlib.pyplot as plt

import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

from pywsl.pul import pu_mr
from pywsl.utils.comcalc import bin_clf_err
from pywsl.cpe.cpe_ene import cpe

from pulib.pu_data import pnu_from_dataframe, pn_from_dataframe

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

for specie in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
    # Loads the iris csv dataset
    irisDataset = pd.read_csv('../input/iris_dataset.csv')

    print('###', specie)

    # Splits x and y (features and target)
    train_size = float(sys.argv[1]) if float(sys.argv[1]) != int(sys.argv[1]) else int(sys.argv[1])

    irisDataset = pn_from_dataframe(irisDataset, 'Species', specie)

    _, x_test, _, y_test = train_test_split(
        irisDataset.drop(['Id','Species', 'y'], axis=1), irisDataset['y'].astype('int'), train_size=train_size, stratify=irisDataset['Species'], random_state=1212)

    # Loads the iris csv dataset
    irisDataset = pd.read_csv('../input/iris_dataset.csv')

    irisDataset = pnu_from_dataframe(irisDataset, 'Species', specie, float(sys.argv[2]))

    x_train, _, y_train, _ = train_test_split(
        irisDataset.drop(['Id','Species', 'y'], axis=1), irisDataset['y'].astype('int'), train_size=train_size, stratify=irisDataset['Species'], random_state=1212)

    x_l = irisDataset.copy().loc[irisDataset['y']!=0].drop(['Id', 'Species', 'y'], axis=1).as_matrix()
    y_l = irisDataset.copy().loc[irisDataset['y']!=0]['y'].as_matrix()
    x_u = irisDataset.copy().loc[irisDataset['y']==0].drop(['Id', 'Species', 'y'], axis=1).as_matrix()

    prior = cpe(x_l, y_l, x_u)

    # Using CPE to estimate prior
    # pu_sl = pu_mr.PU_SL(prior=prior, basis=sys.argv[3])

    # Using default prior (0.5)
    # pu_sl = pu_mr.PU_SL(basis=sys.argv[3])

    # Using real prior (0.3333)
    pu_sl = pu_mr.PU_SL(prior=0.3333, basis=sys.argv[3])

    # Train the model
    pu_sl.fit(x_train, y_train)

    # Test the model
    y_predict = pu_sl.predict(x_test)

    # Binary clf error calculation
    score = bin_clf_err(y_test, y_predict)
    print('\nbin_clf_err:\n\n', score)

    # Confusion Matrix calculation
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('\nConfusion Matrix:\n\n', conf_matrix)

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
    plt.title('Receiver operating characteristic for ' + specie + 'using positive size = ' + sys.argv[2])
    plt.legend(loc="lower right")
    plt.savefig('../output/pu_'+ sys.argv[3] + '/' + sys.argv[2] + '/' + specie + '.png')
