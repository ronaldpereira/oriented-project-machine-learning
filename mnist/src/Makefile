OUTPUT = README.md
TRAIN_SIZE = 0.8
POSITIVE_SIZE = 0.5

run: printheader xgboost pu_lm pu_gauss

printheader:
	@echo '# Machine Learning Results:' > $(OUTPUT)

xgboost:
	@echo 'Generating XGBoost results...'
	@echo '## XGBoost with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 mnist_xgboost.py $(TRAIN_SIZE) >> $(OUTPUT)
	@echo 'XGBoost results generated successfully'

pu_lm:
	@echo 'Generating PU with Linear Model results...'
	@echo '## PU with Linear Model with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 mnist_pu.py $(TRAIN_SIZE)  $(POSITIVE_SIZE) lm >> $(OUTPUT)
	@echo 'PU with Linear Model results generated successfully'

pu_gauss:
	@echo 'Generating PU with Gaussian Kernel results...'
	@echo '## PU with Gaussian Kernel with training size = $(TRAIN_SIZE):' >> $(OUTPUT)
	@python3 mnist_pu.py $(TRAIN_SIZE) $(POSITIVE_SIZE) gauss >> $(OUTPUT)
	@echo 'PU with Gaussian Kernel results generated successfully'
