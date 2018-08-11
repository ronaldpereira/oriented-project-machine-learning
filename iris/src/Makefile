OUTPUT = README.md
TRAIN_SIZE = 120
POSITIVE_SIZE = 10

generateResults: PrintHeader MLP LogReg PU

PrintHeader:
	@echo '# Machine Learning Results:' > $(OUTPUT)

MLP:
	@echo 'Generating Multi-layer Perceptron results...'
	@echo '## Multi-layer Perceptron with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 irisMLPClassification.py $(TRAIN_SIZE) >> $(OUTPUT)
	@echo 'Multi-layer Perceptron results generated successfully'

LogReg:
	@echo 'Generating Logistic Regression results...'
	@echo '## Logistic Regression with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 irisLogRegClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) >> $(OUTPUT)
	@echo 'Logistic Regression results generated successfully'

PU:
	@echo 'Generating Positive Unlabeled results...'
	@echo '## Positive-Unlabeled Classification with training size = $(TRAIN_SIZE) and positive size = $(POSITIVE_SIZE):' >> $(OUTPUT)
	@python3 irisPUClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) >> $(OUTPUT)
	@echo 'Positive Unlabeled results generated successfully'
