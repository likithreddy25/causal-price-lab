"""Generates a synthetic-but-realistically-confounded dataset with a known
ground-truth causal effect, so the PSM pipeline can be validated against a
number we actually know is correct — the same role a hand-crafted test
fixture plays for ordinary unit tests, just for a causal-inference claim
instead of a function's return value.

The confounding structure is deliberate, not just noise: a single latent
"product quality" factor U drives price (so treatment assignment is not
random), the baseline rating/review-count/image-embedding covariates (so
those covariates are legitimate confound proxies to match on), AND the
outcome directly (so a naive, unmatched comparison is provably biased).
Only after conditioning on U's proxies should the estimated effect
recover the true one. That's the property tests/test_psm_recovers_true_effect.py
checks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_price_lab import schema
from causal_price_lab.embeddings import SyntheticEmbedder

CATEGORIES = ["home_kitchen", "electronics_accessories", "outdoor_gear", "office_supplies"]
CATEGORY_BASE_PRICE = {
    "home_kitchen": 28.0,
    "electronics_accessories": 45.0,
    "outdoor_gear": 60.0,
    "office_supplies": 15.0,
}


@dataclass
class SyntheticDataset:
    df: pd.DataFrame
    true_effect: float
    embed_dim: int
    latent_quality: np.ndarray  # U, kept around only for diagnostic plots/tests


def generate(
    n_per_category: int = 400,
    true_effect: float = 4.0,
    confound_strength: float = 1.0,
    embed_dim: int = 16,
    seed: int = 42,
) -> SyntheticDataset:
    """Builds `len(CATEGORIES) * n_per_category` synthetic products.

    true_effect: the ground-truth causal effect of being priced above the
    category median on review_growth. This is the number the PSM pipeline
    is being asked to recover without being told it.

    confound_strength: how strongly the latent quality factor U drives
    both price and outcome. 0 = no confounding (naive and matched
    estimates should roughly agree); larger values make the naive
    estimate's bias more pronounced, which is the whole point of the
    validation test.
    """
    rng = np.random.default_rng(seed)
    embedder = SyntheticEmbedder(dim=embed_dim, noise_std=0.15, seed=seed + 1)

    rows = []
    latent_all = []
    for category in CATEGORIES:
        n = n_per_category
        base_price = CATEGORY_BASE_PRICE[category]

        u = rng.normal(size=n)  # latent product quality
        latent_all.append(u)

        log_price = np.log(base_price) + confound_strength * 0.35 * u + rng.normal(scale=0.20, size=n)
        price = np.exp(log_price)

        baseline_rating = np.clip(3.6 + confound_strength * 0.45 * u + rng.normal(scale=0.25, size=n), 1.0, 5.0)
        baseline_review_count = rng.poisson(
            lam=np.exp(3.0 + confound_strength * 0.5 * u + rng.normal(scale=0.3, size=n))
        )

        median_price = np.median(price)
        treatment = (price > median_price).astype(int)

        # Outcome: true causal effect of treatment, plus the confound's
        # direct (non-causal, via U) influence, plus noise. A naive
        # treated-vs-control mean difference conflates the true_effect
        # term with the confound term below.
        review_growth = (
            true_effect * treatment
            + confound_strength * 3.0 * u
            + rng.normal(scale=2.5, size=n)
        )

        # Placebo outcome: driven by the same U, but by construction has
        # no treatment term at all (it represents something measured
        # *before* the price snapshot). A matching procedure that's
        # actually removing confounding should estimate ~0 effect here;
        # if it doesn't, the matching isn't working.
        placebo = confound_strength * 3.0 * u + rng.normal(scale=2.5, size=n)

        image_embeddings = embedder.embed_images(u.tolist())

        cat_df = pd.DataFrame(
            {
                schema.PRODUCT_ID: [f"{category}_{i}" for i in range(n)],
                schema.CATEGORY: category,
                schema.PRICE: price,
                schema.TREATMENT: treatment,
                schema.BASELINE_RATING: baseline_rating,
                schema.BASELINE_REVIEW_COUNT: baseline_review_count,
                schema.OUTCOME_REVIEW_GROWTH: review_growth,
                schema.PLACEBO_OUTCOME: placebo,
            }
        )
        for i, col in enumerate(schema.image_embed_columns(embed_dim)):
            cat_df[col] = image_embeddings[:, i]

        rows.append(cat_df)

    df = pd.concat(rows, ignore_index=True)
    return SyntheticDataset(
        df=df,
        true_effect=true_effect,
        embed_dim=embed_dim,
        latent_quality=np.concatenate(latent_all),
    )
