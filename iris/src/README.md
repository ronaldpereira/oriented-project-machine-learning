# Machine Learning Results:
## Multi-layer Perceptron with training size = 75:

f1-score:

 0.6666666666666666

Confusion Matrix:

 [[25  0  0]
 [ 0 25  0]
 [ 0 25  0]]

Cross-validation:

Accuracy: 0.69 (+/- 0.26)


## Logistic Regression with training size = 75 and positive size = 30:
### Iris-setosa

f1-score:

 0.3902439024390244

Confusion Matrix:

 [[48  2]
 [25  0]]

Cross-validation:

Accuracy: 0.63 (+/- 0.05)


### Iris-versicolor

f1-score:

 0.3902439024390244

Confusion Matrix:

 [[48  2]
 [25  0]]

Cross-validation:

Accuracy: 0.63 (+/- 0.09)


### Iris-virginica

f1-score:

 0.4

Confusion Matrix:

 [[50  0]
 [25  0]]

Cross-validation:

Accuracy: 0.67 (+/- 0.00)


## Positive-Unlabeled Classification with training size = 75 and positive size = 30:
### Iris-setosa

bin_clf_err:

 0.22666666666666666

Confusion Matrix:

 [[50  0]
 [17  8]]
### Iris-versicolor

bin_clf_err:

 0.37333333333333335

Confusion Matrix:

 [[47  3]
 [25  0]]
### Iris-virginica

bin_clf_err:

 0.4

Confusion Matrix:

 [[45  5]
 [25  0]]
