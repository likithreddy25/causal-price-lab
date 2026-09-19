"""Runs the same PSM + diagnostics pipeline validated in
scripts/run_synthetic_validation.py, against the real dataset produced by
build_dataset.py. Not executed in this sandbox -- see that module's
docstring for why. This script has no synthetic-data special-casing: it
takes any DataFrame matching causal_price_lab/schema.py and reports on it,
which is the entire point of keeping the two pipelines schema-compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_price_lab import diagnostics, psm, schema


def run(df: pd.DataFrame) -> dict:
    embed_dim = sum(1 for c in df.columns if c.startswith(schema.IMAGE_EMBED_PREFIX))
    covariates = [schema.BASELINE_RATING, schema.BASELINE_REVIEW_COUNT] + schema.image_embed_columns(embed_dim)

    # Pre-compute PCA reduction on embeddings so both match() and balance_table()
    # work with the same df that has PCA columns. This ensures balance metrics are
    # computed on the actual covariates used for propensity fitting, not raw embeddings.
    embed_cols = [c for c in covariates if c.startswith(schema.IMAGE_EMBED_PREFIX)]
    if embed_cols:
        df, _ = psm._reduce_embeddings(df, embed_cols, n_components=8, random_state=0)

    naive = psm.naive_ate(df, schema.OUTCOME_REVIEW_GROWTH)
    match_result = psm.match(df, covariate_cols=covariates)
    matched = psm.estimate_ate(df, match_result, schema.OUTCOME_REVIEW_GROWTH)
    balance = diagnostics.balance_table(df, match_result, match_result.propensity_covariates, schema.TREATMENT)
    placebo = diagnostics.placebo_test(df, match_result, schema.PLACEBO_OUTCOME, schema.TREATMENT)

    return {
        "n_products": len(df),
        "n_treated": match_result.n_treated,
        "n_matched_pairs": match_result.n_matched,
        "caliper": match_result.caliper,
        "naive_ate": naive,
        "matched_ate": matched.estimate,
        "matched_ate_ci": [matched.ci_low, matched.ci_high],
        "max_abs_smd_before": balance.max_abs_smd_before,
        "max_abs_smd_after": balance.max_abs_smd_after,
        "placebo_naive_ate": placebo.naive_ate,
        "placebo_matched_ate": placebo.matched_ate.estimate,
        "placebo_matched_ci": [placebo.matched_ate.ci_low, placebo.matched_ate.ci_high],
        "placebo_ci_contains_zero": placebo.passes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/real_dataset.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/real_analysis_report.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    report = run(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
