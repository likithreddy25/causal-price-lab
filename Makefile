.PHONY: install test validate real-download real-combine real-analyze real-analyze-combined

PY := python3
export PYTHONPATH := .

install:
	$(PY) -m pip install -r requirements.txt --break-system-packages

test:
	$(PY) -m pytest tests/ -v

validate:
	$(PY) scripts/run_synthetic_validation.py

# Real-data pipeline: requires network access to huggingface.co (dataset +
# CLIP weights). Actually run and verified locally -- see README's
# "Results (real data)" section. Each category is downloaded and embedded
# as its own run (safer on a laptop than one long multi-category job; see
# scripts/merge_real_categories.py's docstring), then merged for a combined
# multi-category PSM analysis.
real-download:
	$(PY) -m causal_price_lab.real_pipeline.build_dataset --categories Office_Products --out data/processed/real_dataset.parquet
	$(PY) -m causal_price_lab.real_pipeline.build_dataset --categories Home_and_Kitchen --out data/processed/real_dataset_hk.parquet

real-combine:
	$(PY) scripts/merge_real_categories.py --inputs data/processed/real_dataset.parquet data/processed/real_dataset_hk.parquet --out data/processed/real_dataset_combined.parquet

real-analyze:
	$(PY) -m causal_price_lab.real_pipeline.run_analysis

real-analyze-combined:
	$(PY) -m causal_price_lab.real_pipeline.run_analysis --data data/processed/real_dataset_combined.parquet --out data/processed/real_analysis_report_combined.json
