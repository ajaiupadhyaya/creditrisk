.PHONY: private-credit

# Run synthetic private-credit fragility pipeline (requires .venv with requirements.txt)
private-credit:
	.venv/bin/python private_credit/run_pipeline.py
