run: tests distTests

tests: test2 test3 test4

distTests: distTest2 distTest3

test2:
	@echo 'Executing test2...'
	python3 cpeTest.py < ../input/input2.txt > ../output/output2.txt
	@echo 'Finished test2'

test3:
	@echo 'Executing test3...'
	python3 cpeTest.py < ../input/input3.txt > ../output/output3.txt
	@echo 'Finished test3'

test4:
	@echo 'Executing test4...'
	python3 cpeTest.py < ../input/input4.txt > ../output/output4.txt
	@echo 'Finished test4'

distTest2:
	@echo 'Executing distTest2...'
	python3 cpeDistanceTest2.py > ../output/dist2Output.txt
	@echo 'Finished distTest2'

distTest3:
	@echo 'Executing distTest3...'
	python3 cpeDistanceTest3.py > ../output/dist3Output.txt
	@echo 'Finished distTest3'
