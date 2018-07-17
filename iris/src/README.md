# Machine Learning Results:
## Multi-layer Perceptron with training size = 120:

f1-score:

 0.9666666666666667

Confusion Matrix:

 [[11  0  0]
 [ 0  9  1]
 [ 0  0  9]]

Cross-validation:

Accuracy: 0.91 (+/- 0.12)


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

Accuracy: 0.65 (+/- 0.09)


### Iris-virginica

f1-score:

 1.0

Confusion Matrix:

 [[20  0]
 [ 0 10]]

Cross-validation:

Accuracy: 0.97 (+/- 0.03)


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

 0.9

Confusion Matrix:

 [[ 0  0  0]
 [24  0  4]
 [ 1  0  1]]
### Iris-virginica

bin_clf_err:

 0.8888888888888888

Confusion Matrix:

 [[ 0  0  0]
 [21  0  7]
 [ 0  0  2]]
