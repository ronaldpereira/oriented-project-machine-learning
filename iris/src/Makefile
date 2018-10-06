OUTPUT = README.md
TRAIN_SIZE = 120
POSITIVE_SIZE = 30

generateResults: PrintHeader MLP LogReg PU_LM PU_GAUSS

PrintHeader:
	@echo '# Machine Learning Results:' > $(OUTPUT)

MLP:
	@echo 'Generating Multi-layer Perceptron results...'
	@echo '## Multi-layer Perceptron with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 irisMLPClassification.py $(TRAIN_SIZE) >> $(OUTPUT)
	@echo 'Multi-layer Perceptron results generated successfully'

LogReg:
	@echo 'Generating Logistic Regression results...'
	@echo '## Logistic Regression with training size = $(TRAIN_SIZE) and positive size = $(POSITIVE_SIZE):' >> $(OUTPUT)
	@python3 irisLogRegClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) >> $(OUTPUT)
	@echo 'Logistic Regression results generated successfully'

PU_LM:
	@echo 'Generating Positive Unlabeled with Linear Model results...'
	@echo '## Positive-Unlabeled Classification with training size = $(TRAIN_SIZE) and positive size = $(POSITIVE_SIZE):' >> $(OUTPUT)
	@python3 irisPUClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) lm >> $(OUTPUT)
	@echo 'Positive Unlabeled with Linear Model results generated successfully'

PU_GAUSS:
	@echo 'Generating Positive Unlabeled with Gaussian Kernel results...'
	@echo '## Positive-Unlabeled Classification with training size = $(TRAIN_SIZE) and positive size = $(POSITIVE_SIZE):' >> $(OUTPUT)
	@python3 irisPUClassification.py $(TRAIN_SIZE) $(POSITIVE_SIZE) gauss >> $(OUTPUT)
	@echo 'Positive Unlabeled with Gaussian Kernel results generated successfully'
