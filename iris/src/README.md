# Machine Learning Results:
## Multi-layer Perceptron:

f1-score:

 0.9666666666666667

Confusion Matrix:

 [[ 8  0  0]
 [ 0 10  1]
 [ 0  0 11]]

Cross-validation:

Accuracy: 0.91 (+/- 0.13)


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

 0.30769230769230765

Confusion Matrix:

 [[19  1]
 [ 8  2]]

Cross-validation:

Accuracy: 0.65 (+/- 0.06)


### Iris-virginica

f1-score:

 1.0

Confusion Matrix:

 [[20  0]
 [ 0 10]]

Cross-validation:

Accuracy: 0.97 (+/- 0.02)


## Positive-Unlabeled Classification with training size = 120 and positive size = 10:
### Iris-setosa

bin_clf_err:

 1.0

Confusion Matrix:

 [[ 0  0  0]
 [27  0  1]
 [ 2  0  0]]
### Iris-versicolor

bin_clf_err:

 1.0

Confusion Matrix:

 [[ 0  0  0]
 [23  0  5]
 [ 2  0  0]]
### Iris-virginica

bin_clf_err:

 0.875

Confusion Matrix:

 [[ 0  0  0]
 [26  0  3]
 [ 0  0  1]]
