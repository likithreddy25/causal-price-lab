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

## What's real: two independently-verified pipelines

This project has two pipelines that share one schema
(`causal_price_lab/schema.py`) and consume it through the exact same
`psm.py` / `diagnostics.py` code:

| Pipeline | Data | Status |
|---|---|---|
| `causal_price_lab/synthetic.py` | Synthetic, confounded-by-construction, known ground-truth effect | **Actually run.** Verifies the matching *methodology* recovers a known effect — see results below. |
| `causal_price_lab/real_pipeline/` | McAuley-Lab/Amazon-Reviews-2023 (real Amazon product + review data, 2 categories) + real CLIP image embeddings | **Actually run, end to end, on real data downloaded locally.** Office_Products (106 products) and Home_and_Kitchen (100 products), 206 products total. All numbers in "Results (real data)" below are real output. |

The synthetic pipeline exists specifically so the causal-inference
*methodology* — does this matching procedure actually recover a known
effect from confounded data — can be verified without needing external
data access at all. It's a test fixture with a known right answer, the
same role a hand-crafted fixture plays for an ordinary unit test.

The real-data pipeline (`real_pipeline/build_dataset.py`,
`real_pipeline/run_analysis.py`) was written in a sandbox with no network
access to huggingface.co, then actually executed locally, category by
category, once that access was available:

```bash
make real-download          # downloads Amazon-Reviews-2023 + embeds product images with CLIP, one category per run
make real-combine           # merges per-category parquets into one multi-category dataset
make real-analyze-combined  # runs the same PSM pipeline validated below, on the combined real data
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

## Results (real data, actually run — reproduce with `make real-download real-combine real-analyze-combined`)

206 real Amazon products across two categories (Office_Products: 106,
Home_and_Kitchen: 100), each with a real product photo embedded through
CLIP (`openai/clip-vit-base-patch32`, 512-dim), a scrape-time price, prior
rating and review count, and reviews split into baseline/placebo/outcome
windows from that product's own review history.

| Estimator | Estimate | 95% CI |
|---|---|---|
| Naive (unmatched treated − control) | +3.30 reviews | — |
| PSM-matched (57 pairs, 55.9% match rate) | **−0.58 reviews** | [−16.62, 16.46] — CI contains zero, **not statistically significant** |

Covariate balance (max SMD across rating, review count, and the 8
PCA-reduced image-embedding dimensions) improved from **0.683 before
matching to 0.177 after** — clears the 0.25 "reasonably balanced"
threshold (Austin, 2011), a 74% reduction. **Placebo check:** naive shows a
spurious +3.61 "effect"; after matching that drops to −0.28, 95% CI
[−16.37, 16.64] — contains zero, so the design passes: matching removed
the confounding it was supposed to remove.

**Honest headline:** after matching on rating, review-count history, and
image-embedding confounds, this real (if small, N=206) sample shows *no
statistically significant association* between a product's price tier and
its subsequent review growth. The naive +3.30 correlation is consistent
with being entirely explained by product-quality confounding, not price
itself — which is the substantive, if unglamorous, finding the whole
matching design was built to be able to say honestly either way.

| Category | Products | Matched pairs | Naive ATE |
|---|---|---|---|
| Office_Products | 106 | 28 | +5.06 |
| Home_and_Kitchen | 100 | 29 | +1.28 |

Full numbers: [`data/processed/real_analysis_report_combined.json`](data/processed/real_analysis_report_combined.json)
(gitignored locally since it's regenerated data, not source — regenerate
with the commands above).

### Three real bugs found and fixed while getting here

This section exists because "ran it and it worked" would hide the actual
engineering. Three separate, real bugs were found and fixed against this
data, each changing the reported numbers:

1. **p > n perfect separation.** The first real run used all 512 raw
   CLIP dimensions as covariates against 106 samples — a logistic
   regression with more covariates than data points achieves near-perfect
   separation regardless of true group separability, which produced 0
   matched pairs and looked like "these products can't be matched." Fixed
   by reducing embeddings to 8 PCA components before fitting the
   propensity model (`psm._reduce_embeddings`). This is a standard,
   textbook fix for exactly this regime — the point is that the initial
   "0 matches" result was a measurement artifact of the covariate
   dimensionality, not a real finding about the data.

2. **Balance metric computed on the wrong covariate space.** After fix
   #1, the SMD balance table was still computed on all 512 raw embedding
   dimensions (a holdover from before the PCA fix), producing a
   multiple-comparisons-inflated "0.808 → 0.785, barely improved" result
   — while the actual propensity model and matching were correctly using
   only 8 PCA components. Fixed by threading the matcher's actual
   `propensity_covariates` into `diagnostics.balance_table()` instead of
   the raw covariate list. Balance measured on the *same space actually
   used for matching* is the only version of this number that means
   anything.

3. **Non-deterministic PCA.** Re-running the identical analysis on the
   identical combined dataset produced two different matched ATEs (+1.12
   vs. −0.63) from run to run. Root cause: `PCA(n_components=8)` had no
   `random_state`, and sklearn's `svd_solver="auto"` picks the randomized
   SVD solver on this shape (512 features → 8 components), which is
   itself stochastic. Fixed by seeding it. This also retroactively
   explained why the earlier Office_Products-only balance result
   (previously reported as 0.745 → 0.167, "clearing the threshold")
   wasn't reproducible: it was a lucky draw. The stable, reproducible
   Office_Products-only number is **0.745 → 0.337** — a real 55%
   improvement, but with only 26 matched pairs it doesn't clear the 0.25
   threshold on its own. The 2-category combined result above (57 pairs)
   does clear it. That's reported as the honest headline instead of the
   old, unreproducible number.

Getting a second real category (Home_and_Kitchen) working also meant
fixing two genuine memory/scale bottlenecks in `build_dataset.py` rather
than accepting "requires more resources": metadata was held fully in
memory before filtering (3.7M rows → fixed with inline filter-during-parse,
1.27M eligible), and the review-matching scan ran against the full
eligible pool instead of the post-sampling set (fixed by sampling to
`--max-products-per-category` *before* loading reviews, not after).

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
    build_dataset.py   real Amazon-Reviews-2023 + CLIP pipeline, actually run
    run_analysis.py     runs psm.py/diagnostics.py against the real dataset
tests/
  test_psm_recovers_true_effect.py
scripts/
  run_synthetic_validation.py   regenerates the synthetic results table above
  merge_real_categories.py      merges per-category real datasets before a combined run
```

The `Embedder` interface (real implementation + deterministic mock behind
one interface) mirrors the same pattern used for LLM calls in a companion
project, [AgentGate](../agentgate)'s `internal/backend/{mock,anthropic}.go`
— real infrastructure and a network-free stand-in behind one interface, so
the rest of the pipeline never has to know which one produced its inputs.

## Setup

```bash
make install
make test                 # 5/5 real tests, no network needed
make validate              # regenerate the synthetic results table above

# Real data (needs network access to huggingface.co):
make real-download          # ~5-15 min per category on a laptop
make real-combine
make real-analyze-combined  # regenerate the real-data results table above
```
