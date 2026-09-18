"""The one test in this repo that actually matters: does the PSM pipeline
recover a causal effect we already know, from data where we deliberately
built in confounding? Everything else (schema, embeddings, matching
mechanics) only matters in service of this claim.

Assertions are written against relative comparisons (matched vs. naive,
after vs. before) rather than exact CI containment, because a single
95%-CI-contains-the-true-value check is itself a ~1-in-20 coin flip and
would make this test flaky by design. What must hold every run, for any
seed, is: matching reduces bias, it doesn't just relabel it.
"""

from __future__ import annotations

from causal_price_lab import diagnostics, psm, schema, synthetic


def _run(seed: int, confound_strength: float = 1.0, true_effect: float = 4.0):
    ds = synthetic.generate(seed=seed, confound_strength=confound_strength, true_effect=true_effect)
    covariates = [schema.BASELINE_RATING, schema.BASELINE_REVIEW_COUNT] + schema.image_embed_columns(
        ds.embed_dim
    )
    match_result = psm.match(ds.df, covariate_cols=covariates)
    return ds, match_result, covariates


def test_naive_estimate_is_biased():
    ds, match_result, _ = _run(seed=1)
    naive = psm.naive_ate(ds.df, schema.OUTCOME_REVIEW_GROWTH)
    # Confounding was built to inflate, not deflate, the naive estimate.
    assert naive - ds.true_effect > 1.5, (
        f"expected naive ATE ({naive:.2f}) to overstate true_effect ({ds.true_effect}) "
        "by a wide margin -- if it doesn't, the synthetic confound isn't strong enough "
        "to exercise the matching logic"
    )


def test_matched_estimate_recovers_true_effect():
    ds, match_result, _ = _run(seed=1)
    naive = psm.naive_ate(ds.df, schema.OUTCOME_REVIEW_GROWTH)
    matched = psm.estimate_ate(ds.df, match_result, schema.OUTCOME_REVIEW_GROWTH)

    naive_error = abs(naive - ds.true_effect)
    matched_error = abs(matched.estimate - ds.true_effect)

    assert matched_error < naive_error, (
        f"matching should reduce bias: naive error={naive_error:.2f}, "
        f"matched error={matched_error:.2f}"
    )
    # Matched estimate should land close to the true effect in absolute terms,
    # not just "closer than naive."
    assert matched_error < 1.0, f"matched ATE {matched.estimate:.2f} too far from true {ds.true_effect}"
    assert match_result.n_matched > 0.2 * match_result.n_treated, "too few units survived the caliper to trust the estimate"


def test_covariate_balance_improves_after_matching():
    ds, match_result, covariates = _run(seed=1)
    bt = diagnostics.balance_table(ds.df, match_result, covariates, schema.TREATMENT)

    assert bt.max_abs_smd_after < bt.max_abs_smd_before, "matching should improve, not worsen, covariate balance"
    # Austin (2011)'s commonly-cited threshold for "reasonably well balanced".
    assert bt.max_abs_smd_after < 0.25, f"worst post-match SMD {bt.max_abs_smd_after:.3f} exceeds the 0.25 balance threshold"


def test_placebo_effect_shrinks_toward_zero_after_matching():
    """The placebo outcome has no treatment term by construction -- any
    "effect" measured on it is pure confounding. Matching should shrink
    that spurious effect sharply, even if it doesn't drive it to exactly
    zero (the covariates are noisy proxies for the true confound, not the
    confound itself, so some residual bias from measurement error is
    expected and is itself an honest finding, not a bug).
    """
    ds, match_result, _ = _run(seed=1)
    pt = diagnostics.placebo_test(ds.df, match_result, schema.PLACEBO_OUTCOME, schema.TREATMENT)

    assert abs(pt.matched_ate.estimate) < abs(pt.naive_ate) * 0.5, (
        f"expected matching to cut the spurious placebo effect by at least half: "
        f"naive={pt.naive_ate:.2f}, matched={pt.matched_ate.estimate:.2f}"
    )
    assert abs(pt.matched_ate.estimate) < 1.5, "residual placebo effect after matching is too large to trust the design"


def test_zero_confounding_naive_and_matched_agree():
    """Sanity check on the synthetic generator itself: turn confounding off
    (confound_strength=0) and the naive and matched estimates should both
    land near the true effect, since there's nothing to correct for.
    """
    ds, match_result, _ = _run(seed=2, confound_strength=0.0)
    naive = psm.naive_ate(ds.df, schema.OUTCOME_REVIEW_GROWTH)
    matched = psm.estimate_ate(ds.df, match_result, schema.OUTCOME_REVIEW_GROWTH)

    assert abs(naive - ds.true_effect) < 1.0
    assert abs(matched.estimate - ds.true_effect) < 1.0
