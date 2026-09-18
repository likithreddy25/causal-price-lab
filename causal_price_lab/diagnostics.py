"""Diagnostics for judging whether a match is trustworthy *before* looking
at the outcome — the standard PSM discipline of checking covariate balance
first, since an analyst who tunes the match to get the outcome they want is
no longer doing causal inference. Two checks:

1. Standardized mean difference (SMD) per covariate, before vs. after
   matching. Rule of thumb (Austin 2011): |SMD| < 0.1 is well-balanced,
   0.1-0.25 is borderline, > 0.25 is a real imbalance worth worrying about.
2. A placebo test: re-run the whole estimation on an outcome the treatment
   could not possibly have caused. If the matched estimate on that outcome
   is not near zero, the match isn't doing its job on the real outcome
   either — this is what actually catches an under-powered or mis-specified
   matching procedure, since covariate balance alone can look fine while
   residual confounding on unmeasured (or noisily-measured) factors remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_price_lab import psm


def _smd(treated: np.ndarray, control: np.ndarray) -> float:
    """Standardized mean difference: (mean_t - mean_c) / pooled_std.
    Unitless, so covariates on wildly different scales (a rating out of 5
    vs. a review count in the hundreds) are directly comparable.
    """
    pooled_std = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_std)


@dataclass
class BalanceTable:
    table: pd.DataFrame  # columns: covariate, smd_before, smd_after
    max_abs_smd_before: float
    max_abs_smd_after: float


def balance_table(
    df: pd.DataFrame,
    match_result: psm.MatchResult,
    covariate_cols: list[str],
    treatment_col: str,
) -> BalanceTable:
    pairs = match_result.matched_pairs
    rows = []
    for col in covariate_cols:
        before = _smd(
            df.loc[df[treatment_col] == 1, col].to_numpy(),
            df.loc[df[treatment_col] == 0, col].to_numpy(),
        )
        after = _smd(
            df.loc[pairs["treated_index"], col].to_numpy(),
            df.loc[pairs["control_index"], col].to_numpy(),
        )
        rows.append({"covariate": col, "smd_before": before, "smd_after": after})

    table = pd.DataFrame(rows)
    return BalanceTable(
        table=table,
        max_abs_smd_before=float(table["smd_before"].abs().max()),
        max_abs_smd_after=float(table["smd_after"].abs().max()),
    )


@dataclass
class PlaceboResult:
    naive_ate: float
    matched_ate: psm.ATEResult
    passes: bool  # True if the matched 95% CI contains 0


def placebo_test(
    df: pd.DataFrame,
    match_result: psm.MatchResult,
    placebo_outcome_col: str,
    treatment_col: str,
    n_boot: int = 1000,
) -> PlaceboResult:
    naive = psm.naive_ate(df, placebo_outcome_col, treatment_col)
    matched = psm.estimate_ate(df, match_result, placebo_outcome_col, n_boot=n_boot)
    passes = matched.ci_low <= 0 <= matched.ci_high
    return PlaceboResult(naive_ate=naive, matched_ate=matched, passes=passes)
