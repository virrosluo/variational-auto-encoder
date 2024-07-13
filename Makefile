TRAINING_LOG="./training_log"
TESTING_LOG="./lightning_logs"

run-expr1:
	[ -d "$(TRAINING_LOG)" ] && rm -r "$(TRAINING_LOG)" && echo "$(TRAINING_LOG) removed." || echo "$(TRAINING_LOG) does not exist."
	python main.py Training_Config/experiment1.json
test-expr1:
	python inference.py Training_Config/experiment1.json
	[ -d "$(TESTING_LOG)" ] && rm -r "$(TESTING_LOG)" && echo "$(TESTING_LOG) removed." || echo "$(TESTING_LOG) does not exist."