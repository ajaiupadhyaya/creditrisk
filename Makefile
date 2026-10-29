PYTHON ?= python3
.PHONY: analysis test private-credit
analysis:
	$(PYTHON) run_analysis.py
test:
	$(PYTHON) -m pytest tests -q
# Legacy experimental track. Not the validated portfolio result.
private-credit:
	$(PYTHON) private_credit/run_pipeline.py
