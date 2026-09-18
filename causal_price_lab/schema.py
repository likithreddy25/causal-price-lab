"""Canonical column names shared by every stage of the pipeline (synthetic
and real). Keeping these as named constants, not string literals scattered
across files, is what lets synthetic.py and real_pipeline/ produce
DataFrames that psm.py and diagnostics.py can consume identically without
either side knowing which one produced the data.
"""

from __future__ import annotations

# Identity
PRODUCT_ID = "product_id"
CATEGORY = "category"

# Treatment definition (computed from raw price + category median, not a
# raw input column — see synthetic.py / real_pipeline for how it's derived)
PRICE = "price"
TREATMENT = "above_category_median_price"  # 1 = treated, 0 = control

# Pre-treatment covariates (must be measured before/at the price snapshot,
# never after — using a post-treatment variable as a covariate would
# introduce collider bias, not remove confounding)
BASELINE_RATING = "baseline_avg_rating"
BASELINE_REVIEW_COUNT = "baseline_review_count"
IMAGE_EMBED_PREFIX = "img_emb_"  # columns img_emb_0 .. img_emb_{d-1}

# Outcome (post-treatment, the thing we're estimating the causal effect on)
OUTCOME_REVIEW_GROWTH = "review_growth"

# Placebo outcome (also pre-treatment — used only to sanity-check the
# matching procedure: a well-matched design should show ~zero "effect" of
# treatment on something treatment couldn't possibly have caused)
PLACEBO_OUTCOME = "placebo_prior_review_growth"

PROPENSITY_SCORE = "propensity_score"
MATCH_ID = "match_id"


def image_embed_columns(dim: int) -> list[str]:
    return [f"{IMAGE_EMBED_PREFIX}{i}" for i in range(dim)]
