"""Propensity score matching: estimate the causal effect of a binary
treatment by pairing each treated unit with the untreated unit(s) most
similar to it in covariate space (via their propensity score, the
predicted probability of treatment given covariates), so the comparison
approximates a randomized experiment within each matched pair instead of
comparing two groups that differ systematically before treatment.

Matching is done within category (never across, since price medians and
what "similar" means are category-specific) and with a caliper (a max
allowed propensity-score distance) so no match is accepted just because
it was the closest available — an unmatched treated unit with no good
control nearby is dropped rather than paired with a bad match, which is
the standard, defensible choice over forcing every unit into some match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from causal_price_lab import schema


@dataclass
class MatchResult:
    matched_pairs: pd.DataFrame  # one row per treated unit, with control_index + distance
    n_treated: int
    n_matched: int
    caliper: float
    propensity_covariates: list[str]  # actual covariates used for propensity model (post-PCA-reduction)


def _reduce_embeddings(df: pd.DataFrame, embed_cols: list[str], n_components: int = 8) -> tuple[pd.DataFrame, list[str]]:
    if not embed_cols or len(embed_cols) <= n_components:
        return df, embed_cols
    embeddings = df[embed_cols].to_numpy(dtype=float)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    component_cols = [f"_pca_embed_{i}" for i in range(n_components)]
    df_with_pca = df.copy()
    for i, col in enumerate(component_cols):
        df_with_pca[col] = reduced[:, i]
    return df_with_pca, component_cols


def _fit_propensity(df: pd.DataFrame, covariate_cols: list[str], treatment_col: str, random_state: int):
    X = StandardScaler().fit_transform(df[covariate_cols].to_numpy(dtype=float))
    y = df[treatment_col].to_numpy()
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    # Clip away from 0/1 before taking a logit — avoids -inf/inf for a
    # perfectly separated prediction, which a caliper on raw probability
    # would otherwise treat as infinitely far from everything.
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    logit = np.log(proba / (1 - proba))
    return proba, logit


def match(
    df: pd.DataFrame,
    covariate_cols: list[str],
    treatment_col: str = schema.TREATMENT,
    category_col: str = schema.CATEGORY,
    caliper_in_logit_sds: float = 0.2,
    random_state: int = 0,
) -> MatchResult:
    """1:1 nearest-neighbor matching on the propensity-score logit, without
    replacement, within category, with a caliper of `caliper_in_logit_sds`
    standard deviations of the logit score — Austin (2011)'s commonly-cited
    rule of thumb for how tight a caliper to use.

    High-dimensional image embeddings are reduced via PCA before propensity
    model fitting to avoid overfitting in p > n regimes (more covariates than
    samples), which causes near-perfect separation regardless of true
    separability.
    """
    df = df.reset_index(drop=True).copy()

    # Separate embeddings from other covariates and apply PCA (if not already reduced)
    embed_cols = [c for c in covariate_cols if c.startswith(schema.IMAGE_EMBED_PREFIX)]
    non_embed_cols = [c for c in covariate_cols if not c.startswith(schema.IMAGE_EMBED_PREFIX)]
    pca_cols = [c for c in covariate_cols if c.startswith("_pca_embed_")]

    # If PCA columns already exist in data, use them directly; otherwise reduce raw embeddings
    if pca_cols:
        propensity_covariates = non_embed_cols + pca_cols
    elif embed_cols:
        df, reduced_embed_cols = _reduce_embeddings(df, embed_cols, n_components=8)
        propensity_covariates = non_embed_cols + reduced_embed_cols
    else:
        propensity_covariates = non_embed_cols

    proba, logit = _fit_propensity(df, propensity_covariates, treatment_col, random_state)
    df[schema.PROPENSITY_SCORE] = proba
    df["_logit"] = logit

    caliper = caliper_in_logit_sds * np.std(logit)

    pairs = []
    for category, group in df.groupby(category_col):
        treated = group[group[treatment_col] == 1]
        control = group[group[treatment_col] == 0]
        if len(control) == 0 or len(treated) == 0:
            continue

        control_logits = control["_logit"].to_numpy().reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=min(len(control), 5))
        nn.fit(control_logits)

        available_control = set(control.index)
        # Match treated units in order of how confidently treated they are
        # (highest propensity first) so the hardest-to-match units get
        # first pick of the closest controls, a common greedy-matching
        # convention that reduces worst-case match quality.
        treated_sorted = treated.sort_values("_logit", ascending=False)

        for treated_idx, treated_row in treated_sorted.iterrows():
            distances, neighbor_positions = nn.kneighbors(
                [[treated_row["_logit"]]], n_neighbors=min(len(control), 20)
            )
            matched_control_idx = None
            matched_distance = None
            for dist, pos in zip(distances[0], neighbor_positions[0]):
                candidate_idx = control.index[pos]
                if candidate_idx in available_control and dist <= caliper:
                    matched_control_idx = candidate_idx
                    matched_distance = dist
                    break
            if matched_control_idx is not None:
                available_control.discard(matched_control_idx)
                pairs.append(
                    {
                        "treated_index": treated_idx,
                        "control_index": matched_control_idx,
                        "category": category,
                        "distance": matched_distance,
                    }
                )

    matched_pairs = pd.DataFrame(pairs)
    n_treated = int((df[treatment_col] == 1).sum())
    return MatchResult(
        matched_pairs=matched_pairs,
        n_treated=n_treated,
        n_matched=len(matched_pairs),
        caliper=caliper,
        propensity_covariates=propensity_covariates,
    )


@dataclass
class ATEResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_pairs: int
    per_pair_diffs: np.ndarray = field(repr=False)


def estimate_ate(
    df: pd.DataFrame,
    match_result: MatchResult,
    outcome_col: str,
    n_boot: int = 1000,
    random_state: int = 0,
) -> ATEResult:
    """Average treatment effect on the matched sample: mean(treated outcome
    - matched control outcome) across pairs, with a percentile bootstrap
    CI computed by resampling *pairs* (not individual rows) with
    replacement, which respects that each pair is the actual unit of
    comparison here, not each row independently.
    """
    pairs = match_result.matched_pairs
    treated_outcome = df.loc[pairs["treated_index"], outcome_col].to_numpy()
    control_outcome = df.loc[pairs["control_index"], outcome_col].to_numpy()
    diffs = treated_outcome - control_outcome

    rng = np.random.default_rng(random_state)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = sample.mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return ATEResult(
        estimate=float(diffs.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_pairs=n,
        per_pair_diffs=diffs,
    )


def naive_ate(df: pd.DataFrame, outcome_col: str, treatment_col: str = schema.TREATMENT) -> float:
    """The comparison PSM exists to improve on: an unmatched mean
    difference between treated and control groups, still confounded by
    whatever drove treatment assignment in the first place."""
    treated_mean = df.loc[df[treatment_col] == 1, outcome_col].mean()
    control_mean = df.loc[df[treatment_col] == 0, outcome_col].mean()
    return float(treated_mean - control_mean)
