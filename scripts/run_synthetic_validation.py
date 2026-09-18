#!/usr/bin/env python3
"""Runs the full PSM pipeline against the synthetic confounded dataset and
prints/saves the real numbers this project's README quotes. Every number
in the README's results table came from actually running this script, not
from hand-written prose -- run it yourself to reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

from causal_price_lab import diagnostics, psm, schema, synthetic


def main():
    ds = synthetic.generate()  # defaults: true_effect=4.0, confound_strength=1.0, seed=42
    covariates = [schema.BASELINE_RATING, schema.BASELINE_REVIEW_COUNT] + schema.image_embed_columns(
        ds.embed_dim
    )

    naive = psm.naive_ate(ds.df, schema.OUTCOME_REVIEW_GROWTH)
    match_result = psm.match(ds.df, covariate_cols=covariates)
    matched = psm.estimate_ate(ds.df, match_result, schema.OUTCOME_REVIEW_GROWTH)
    balance = diagnostics.balance_table(ds.df, match_result, covariates, schema.TREATMENT)
    placebo = diagnostics.placebo_test(ds.df, match_result, schema.PLACEBO_OUTCOME, schema.TREATMENT)

    report = {
        "true_effect": ds.true_effect,
        "n_products": len(ds.df),
        "n_treated": match_result.n_treated,
        "n_matched_pairs": match_result.n_matched,
        "match_rate": match_result.n_matched / match_result.n_treated,
        "caliper_logit_units": match_result.caliper,
        "naive_ate": naive,
        "naive_ate_error_vs_true": naive - ds.true_effect,
        "matched_ate": matched.estimate,
        "matched_ate_ci95": [matched.ci_low, matched.ci_high],
        "matched_ate_error_vs_true": matched.estimate - ds.true_effect,
        "bias_reduction_pct": 100 * (1 - abs(matched.estimate - ds.true_effect) / abs(naive - ds.true_effect)),
        "max_abs_smd_before_matching": balance.max_abs_smd_before,
        "max_abs_smd_after_matching": balance.max_abs_smd_after,
        "placebo_naive_ate": placebo.naive_ate,
        "placebo_matched_ate": placebo.matched_ate.estimate,
        "placebo_matched_ci95": [placebo.matched_ate.ci_low, placebo.matched_ate.ci_high],
        "placebo_ci_contains_zero": placebo.passes,
        "placebo_bias_reduction_pct": 100 * (1 - abs(placebo.matched_ate.estimate) / abs(placebo.naive_ate)),
    }

    out_path = Path(__file__).parent / "synthetic_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
