# causal-price-lab

Does a product's price tier causally affect how fast its reviews grow —
or does that correlation just reflect that higher-quality products get
priced higher *and* reviewed more, regardless of price itself?

This project answers that with propensity score matching (PSM), using
real product-image embeddings (CLIP) as one of the confound-control
covariates alongside rating and review-count history — not as a
"multimodal" decoration, but because product photography quality
plausibly correlates with both price and review growth, making it a
genuine confounder if left uncontrolled.

## The causal question, precisely

- **Treatment:** a product is priced above its category's median price.
- **Outcome:** review growth (count of new reviews) after that price is
  observed.
- **Confound:** an unobserved "product quality" factor that drives price,
  prior rating, prior review volume, *and* review growth — the reason a
  naive treated-vs-control comparison is biased.
- **Method:** logistic-regression propensity model → nearest-neighbor
  matching on the propensity logit, within category, with a caliper →
  paired-difference ATE estimate with a bootstrap confidence interval.
- **Validation:** a placebo outcome (also driven by the confound, but by
  construction unaffected by treatment) — if the matched design still
  shows a large "effect" on the placebo, the matching isn't actually
  removing confounding.

## What's real vs. what's written-but-unexecuted

This project has two pipelines that share one schema
(`causal_price_lab/schema.py`) and consume it through the exact same
`psm.py` / `diagnostics.py` code:

| Pipeline | Data | Status |
|---|---|---|
| `causal_price_lab/synthetic.py` | Synthetic, confounded-by-construction, known ground-truth effect | **Actually run in this sandbox.** All numbers below are real output, not estimates. |
| `causal_price_lab/real_pipeline/` | McAuley-Lab/Amazon-Reviews-2023 (real Amazon product + review data) + real CLIP image embeddings | **Written, not executed.** This sandbox's network is allowlisted to git+pypi only; huggingface.co (dataset + model weights) and the product-image CDN both return 403 through the proxy. Confirmed via direct request, not assumed. |

The synthetic pipeline exists specifically so the causal-inference
*methodology* — does this matching procedure actually recover a known
effect from confounded data — can be verified without needing external
data access at all. It's a test fixture with a known right answer, the
same role a hand-crafted fixture plays for an ordinary unit test.

The real-data pipeline (`real_pipeline/build_dataset.py`,
`real_pipeline/run_analysis.py`) is real, complete code — model selection,
image downloading, CLIP embedding, the pre/post review-window split — that
you can run yourself once you have network access to huggingface.co:

```bash
make real-download   # downloads Amazon-Reviews-2023 + embeds product images with CLIP
make real-analyze    # runs the same PSM pipeline validated below, on real data
```

One honest caveat specific to the real pipeline: Amazon-Reviews-2023
doesn't carry price *history*, only a scrape-time price snapshot. So the
real-data run answers a slightly softer question than the synthetic
validation does — "is current price tier associated with recent review
growth after matching," not "did setting this price cause these reviews"
— because price isn't a timestamped intervention in that dataset. This
limitation is documented in `build_dataset.py`'s docstring, not hidden.

## Results (synthetic validation, actually run — reproduce with `make validate`)

Generated 1,600 synthetic products across 4 categories with a known
ground-truth causal effect of **4.0** review-growth units, confounded by a
latent quality factor that drives price, prior rating, prior review
count, *and* the outcome directly.

| Estimator | Estimate | vs. true effect (4.0) |
|---|---|---|
| Naive (unmatched treated − control) | 7.85 | overstates by 3.85 (96%) |
| PSM-matched (328 pairs, 95% CI) | 4.42, CI [4.04, 4.79] | overstates by 0.42 (10%) |

Matching cut the bias by **89%**. Covariate balance (max standardized
mean difference across rating, review count, and 16 image-embedding
dimensions) improved from **1.69 before matching to 0.21 after** —
inside the commonly-cited 0.25 "reasonably balanced" threshold (Austin,
2011).

**Placebo check**, run on an outcome the treatment couldn't possibly have
caused: the naive comparison shows a spurious 3.87-unit "effect" (pure
confounding); after matching, that drops to **0.52** — an 87% reduction,
but its 95% CI ([0.13, 0.89]) does not quite contain zero.

That residual is disclosed, not smoothed over: `baseline_avg_rating`,
`baseline_review_count`, and the image embedding are noisy *proxies* for
the true latent confound, not the confound itself, and propensity-score
matching only guarantees balance on the covariates you actually match on.
Measurement error in those proxies is expected to leave some residual
bias — this is a textbook limitation of PSM (it assumes no *unmeasured or
mismeasured* confounding), not a bug in this implementation. The 89%/87%
bias reductions are the honest headline number; "reduces confounding
sharply" is the accurate claim, "eliminates it" is not one this design
supports.

Full numbers: [`scripts/synthetic_validation_report.json`](scripts/synthetic_validation_report.json),
regenerated by `make validate`.

## Tests

`tests/test_psm_recovers_true_effect.py` — 5 tests, actually run (`make
test`), checking properties that must hold for any seed rather than a
single lucky run: the naive estimate is biased, matching reduces that
bias (not just relabels it), covariate balance improves and clears the
0.25 threshold, the placebo effect shrinks by at least half after
matching, and a zero-confounding sanity check where naive and matched
estimates should already agree.

## Project layout

```
causal_price_lab/
  schema.py          canonical column names shared by both pipelines
  embeddings.py       Embedder interface: SyntheticEmbedder (deterministic,
                       network-free) / CLIPEmbedder (real, needs network)
  synthetic.py        confounded synthetic dataset generator, known ground truth
  psm.py              propensity model, caliper matching, bootstrap ATE
  diagnostics.py       SMD balance table, placebo test
  real_pipeline/
    build_dataset.py   real Amazon-Reviews-2023 + CLIP pipeline (unexecuted here)
    run_analysis.py     runs psm.py/diagnostics.py against the real dataset
tests/
  test_psm_recovers_true_effect.py
scripts/
  run_synthetic_validation.py   regenerates the results table above
```

The `Embedder` interface (real implementation + deterministic mock behind
one interface) mirrors the same pattern used for LLM calls in a companion
project, [AgentGate](../agentgate)'s `internal/backend/{mock,anthropic}.go`
— real infrastructure and a network-free stand-in behind one interface, so
the rest of the pipeline never has to know which one produced its inputs.

## Setup

```bash
make install
make test        # 5/5 real tests, no network needed
make validate     # regenerate the results table above
```
