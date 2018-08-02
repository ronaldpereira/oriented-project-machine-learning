OUTPUT = README.md
TRAIN_SIZE = 30
POSITIVE_SIZE = 10

generateResults: PrintHeader MLP LogReg PU

PrintHeader:
	echo '# Machine Learning Results:' > $(OUTPUT)

MLP:
	echo '## Multi-layer Perceptron with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	python3 irisMLPClassification.py $(TRAIN_SIZE) >> $(OUTPUT)

LogReg:
	echo '## Logistic Regression with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	python3 irisLogRegClassification.py $(TRAIN_SIZE) >> $(OUTPUT)

PU:
	echo '## Positive-Unlabeled Classification with training size = $(TRAIN_SIZE) and positive size = $(POSITIVE_SIZE):' >> $(OUTPUT)
	python3 irisPUClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) >> $(OUTPUT)
