.PHONY: install test validate real-download real-analyze

PY := python3
export PYTHONPATH := .

install:
	$(PY) -m pip install -r requirements.txt --break-system-packages

test:
	$(PY) -m pytest tests/ -v

validate:
	$(PY) scripts/run_synthetic_validation.py

# Real-data pipeline: requires network access to huggingface.co (dataset +
# CLIP weights) that this project's authoring environment did not have.
# Run these two targets yourself once you have that access.
real-download:
	$(PY) -m causal_price_lab.real_pipeline.build_dataset

real-analyze:
	$(PY) -m causal_price_lab.real_pipeline.run_analysis
