# Machine Learning Results:
## Multi-layer Perceptron with training size = 120:

f1-score:

 0.9666666666666667

Confusion Matrix:

 [[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]

Cross-validation:

Accuracy: 0.72 (+/- 0.31)


## Logistic Regression with training size = 120:
### Iris-setosa

f1-score:

 1.0

Confusion Matrix:

 [[20  0]
 [ 0 10]]

Cross-validation:

Accuracy: 1.00 (+/- 0.00)


### Iris-versicolor

f1-score:

 0.28571428571428575

Confusion Matrix:

 [[18  2]
 [ 8  2]]

Cross-validation:

Accuracy: 0.67 (+/- 0.07)


### Iris-virginica

f1-score:

 0.9523809523809523

Confusion Matrix:

 [[19  1]
 [ 0 10]]

Cross-validation:

Accuracy: 0.98 (+/- 0.02)


## Positive-Unlabeled Classification with training size = 120 and positive size = 10:
### Iris-setosa

bin_clf_err:

 1.0

Confusion Matrix:

 [[ 0  0  0]
 [25  0  5]
 [ 0  0  0]]
### Iris-versicolor

bin_clf_err:

 0.9166666666666667

Confusion Matrix:

 [[ 0  0  0]
 [23  0  5]
 [ 1  0  1]]
### Iris-virginica

bin_clf_err:

 0.8333333333333333

Confusion Matrix:

 [[ 0  0  0]
 [22  0  4]
 [ 2  0  2]]
