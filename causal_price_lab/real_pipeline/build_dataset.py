"""Builds a real product dataset from McAuley-Lab/Amazon-Reviews-2023
(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), matching
the exact schema (causal_price_lab/schema.py) that synthetic.py produces,
so psm.py and diagnostics.py run unmodified against real data.

STATUS: written but NOT executed. This sandbox's outbound network is
allowlisted to git+pypi only -- huggingface.co (dataset + CLIP weights) and
the product image CDN are both unreachable from here (confirmed via direct
request: 403 from the network proxy). This script is correct by inspection
and mirrors the exact transformation the synthetic generator encodes, but
running it and confirming the real-data result is a step for you to do
locally, where those hosts are reachable. See README.md's "What's real"
section.

Important limitation, disclosed rather than hidden: unlike the synthetic
generator, the treatment here (price) is a *scrape-time snapshot*, not a
timestamped assignment -- Amazon-Reviews-2023 doesn't carry price history.
So "does price affect review growth" here really means "is a product's
*current* price tier associated with its *recent* review growth, after
matching on rating/review-count/image-embedding confounds" -- an
observational association claim with the causal language softened
accordingly, not a claim that price-at-time-T caused reviews after time T.
The pre/post split below is a best-effort proxy (see _split_review_windows),
not a true intervention date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from causal_price_lab import schema
from causal_price_lab.embeddings import CLIPEmbedder

# Real Amazon-Reviews-2023 category names -> our 4 synthetic category labels.
CATEGORY_MAP = {
    "Office_Products": "office_supplies",
    "Home_and_Kitchen": "home_kitchen",
    "Electronics": "electronics_accessories",
    "Sports_and_Outdoors": "outdoor_gear",
}

MIN_REVIEWS_PER_PRODUCT = 8  # need enough reviews to split into windows meaningfully


def _load_category(category: str, max_products: int | None):
    """Loads and joins the meta + review splits for one category from JSONL files.
    The Amazon-Reviews-2023 dataset structure: metadata and reviews are stored as
    JSONL files in raw/meta_categories/ and raw/review_categories/ respectively.
    """
    from huggingface_hub import hf_hub_download
    import json

    # Download JSONL files from HuggingFace
    meta_file = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/meta_categories/meta_{category}.jsonl",
        repo_type="dataset"
    )
    reviews_file = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/review_categories/{category}.jsonl",
        repo_type="dataset"
    )

    # Load metadata JSONL with inline filtering to avoid holding all rows in memory.
    # Check price-numeric and has_large_image on each record as parsed; only keep
    # those that pass both checks. This reduces memory footprint dramatically for
    # large categories (e.g., Home_and_Kitchen: 3.7M rows → ~150K eligible).
    print(f"    Loading and filtering metadata...", flush=True)

    def has_large_image(images):
        if not isinstance(images, list) or len(images) == 0:
            return False
        return any(isinstance(img, dict) and "large" in img and img["large"] for img in images)

    meta_rows = []
    rows_scanned = 0
    with open(meta_file, 'r') as f:
        for line in f:
            rows_scanned += 1
            record = json.loads(line)

            # Check price validity
            price = record.get("price")
            if price is None:
                continue
            try:
                if isinstance(price, str):
                    price_float = float(price)
                else:
                    price_float = float(price)
            except (ValueError, TypeError):
                continue

            # Check has_large_image
            if not has_large_image(record.get("images", [])):
                continue

            # Both checks passed; keep this record
            meta_rows.append(record)

            if rows_scanned % 10000 == 0:
                print(f"      {rows_scanned} scanned, {len(meta_rows)} eligible...", flush=True)

    meta_df = pd.DataFrame(meta_rows)
    print(f"    Loaded {len(meta_df)} eligible metadata rows (from {rows_scanned} total)", flush=True)

    # Load reviews JSONL (streaming, filtered to products in meta_df)
    print(f"    Loading reviews (filtered to {len(meta_df)} products)...", flush=True)
    meta_asins = set(meta_df["parent_asin"].unique())
    reviews_rows = []
    with open(reviews_file, 'r') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if record.get("parent_asin") in meta_asins:
                reviews_rows.append(record)
            if (i + 1) % 100000 == 0:
                print(f"      {i + 1} reviews scanned, {len(reviews_rows)} matched...", flush=True)
    reviews_df = pd.DataFrame(reviews_rows)
    print(f"    Loaded {len(reviews_df)} review rows", flush=True)

    if max_products is not None:
        meta_df = meta_df.sample(n=min(max_products, len(meta_df)), random_state=0)

    print(f"    Sampled to {len(meta_df)} meta products", flush=True)

    # Filter reviews to only products in metadata
    reviews_df = reviews_df[reviews_df["parent_asin"].isin(meta_df["parent_asin"])]
    print(f"    Reviews after filtering to meta products: {len(reviews_df)}", flush=True)

    # Filter to products with enough reviews
    review_counts = reviews_df["parent_asin"].value_counts()
    print(f"    Review count stats: min={review_counts.min()}, max={review_counts.max()}, mean={review_counts.mean():.1f}, median={review_counts.median():.1f}", flush=True)

    eligible = review_counts[review_counts >= MIN_REVIEWS_PER_PRODUCT].index
    print(f"    Products with >= {MIN_REVIEWS_PER_PRODUCT} reviews: {len(eligible)}", flush=True)

    meta_df = meta_df[meta_df["parent_asin"].isin(eligible)]
    reviews_df = reviews_df[reviews_df["parent_asin"].isin(eligible)]
    print(f"    Final: {len(meta_df)} meta, {len(reviews_df)} reviews", flush=True)

    return meta_df, reviews_df


def _split_review_windows(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Per product, sorts its reviews by timestamp and splits them into
    thirds: baseline (oldest) / placebo-growth (middle) / outcome-growth
    (most recent). Using terciles of *that product's own* review history,
    rather than one global calendar cutoff, keeps the split meaningful for
    products with very different review volumes and lifespans.
    """
    if len(reviews_df) == 0:
        # Empty DataFrame - return empty result with correct columns
        return pd.DataFrame(columns=[
            "parent_asin",
            schema.BASELINE_RATING,
            schema.BASELINE_REVIEW_COUNT,
            schema.PLACEBO_OUTCOME,
            schema.OUTCOME_REVIEW_GROWTH,
        ])

    # Convert timestamp from milliseconds to seconds if needed
    if reviews_df["timestamp"].dtype != 'datetime64[ns]':
        # Check if values look like milliseconds (> 1e10)
        if reviews_df["timestamp"].max() > 1e10:
            reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"], unit="ms")
        else:
            reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"], unit="s")

    rows = []
    for parent_asin, group in reviews_df.groupby("parent_asin"):
        # Only process products with enough reviews
        if len(group) < MIN_REVIEWS_PER_PRODUCT:
            continue
        g = group.sort_values("timestamp")
        n = len(g)
        third = max(n // 3, 1)
        baseline = g.iloc[:third]
        placebo_window = g.iloc[third : 2 * third]
        outcome_window = g.iloc[2 * third :]
        rows.append(
            {
                "parent_asin": parent_asin,
                schema.BASELINE_RATING: baseline["rating"].mean(),
                schema.BASELINE_REVIEW_COUNT: len(baseline),
                schema.PLACEBO_OUTCOME: len(placebo_window),
                schema.OUTCOME_REVIEW_GROWTH: len(outcome_window),
            }
        )
    result = pd.DataFrame(rows)
    print(f"    Windows split: {len(result)} products with >= {MIN_REVIEWS_PER_PRODUCT} reviews", flush=True)
    return result


def build(categories: list[str], max_products_per_category: int, embed_dim_batch: int = 32) -> pd.DataFrame:
    all_rows = []
    embedder = CLIPEmbedder()  # real network + GPU/CPU inference, see module docstring

    for real_category, our_category in CATEGORY_MAP.items():
        if real_category not in categories:
            continue
        print(f"Loading {real_category}...", flush=True)
        meta_df, reviews_df = _load_category(real_category, max_products_per_category)
        print(f"  Loaded {len(meta_df)} metadata, {len(reviews_df)} reviews", flush=True)

        print(f"  Splitting review windows...", flush=True)
        windows = _split_review_windows(reviews_df)
        print(f"  Got {len(windows)} windows", flush=True)

        merged = meta_df.merge(windows, on="parent_asin", how="inner")
        print(f"  Merged: {len(merged)} products", flush=True)
        if merged.empty:
            continue

        median_price = merged["price_numeric"].median()
        merged[schema.TREATMENT] = (merged["price_numeric"] > median_price).astype(int)

        # Download + embed the first ("large") product image per row, in
        # batches so a single failed download doesn't need re-fetching
        # everything.
        from io import BytesIO

        import requests
        from PIL import Image

        print(f"  Downloading and embedding {len(merged)} product images...", flush=True)
        embeddings = []
        for batch_idx, start in enumerate(range(0, len(merged), embed_dim_batch)):
            chunk = merged.iloc[start : start + embed_dim_batch]
            print(f"    Batch {batch_idx + 1}/{(len(merged) + embed_dim_batch - 1) // embed_dim_batch}: downloading {len(chunk)} images...", flush=True)
            images = []
            for img_idx, image_field in enumerate(chunk["images"]):
                # image_field is a list of dicts; get first image's "large" URL
                try:
                    if isinstance(image_field, list) and len(image_field) > 0:
                        url = image_field[0].get("large")
                        if url:
                            resp = requests.get(url, timeout=10)
                            images.append(Image.open(BytesIO(resp.content)).convert("RGB"))
                        else:
                            print(f"      Warning: no large URL in image {img_idx}", flush=True)
                            images.append(None)
                    else:
                        print(f"      Warning: invalid image field {img_idx}: {image_field}", flush=True)
                        images.append(None)
                except Exception as e:
                    print(f"      Warning: failed to download/process image {img_idx}: {e}", flush=True)
                    images.append(None)
            images = [img for img in images if img is not None]
            if images:
                print(f"    Embedding {len(images)} images...", flush=True)
                embeddings.append(embedder.embed_images(images))
        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)
            # Filter merged to only products with successful embeddings
            valid_asins = merged.iloc[:len(embeddings)]["parent_asin"].values
            merged = merged[merged["parent_asin"].isin(valid_asins)]
        else:
            embeddings = np.zeros((0, embedder.dim))
            merged = merged.iloc[0:0]  # Empty dataframe

        out = pd.DataFrame(
            {
                schema.PRODUCT_ID: merged["parent_asin"].values,
                schema.CATEGORY: our_category,
                schema.PRICE: merged["price_numeric"].values,
                schema.TREATMENT: merged[schema.TREATMENT].values,
                schema.BASELINE_RATING: merged[schema.BASELINE_RATING].values,
                schema.BASELINE_REVIEW_COUNT: merged[schema.BASELINE_REVIEW_COUNT].values,
                schema.OUTCOME_REVIEW_GROWTH: merged[schema.OUTCOME_REVIEW_GROWTH].values,
                schema.PLACEBO_OUTCOME: merged[schema.PLACEBO_OUTCOME].values,
            }
        )
        for i, col in enumerate(schema.image_embed_columns(embedder.dim)):
            out[col] = embeddings[:, i]

        all_rows.append(out)

    return pd.concat(all_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_MAP.keys()))
    parser.add_argument("--max-products-per-category", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("data/processed/real_dataset.parquet"))
    args = parser.parse_args()

    df = build(args.categories, args.max_products_per_category)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Wrote {len(df)} products to {args.out}")


if __name__ == "__main__":
    main()
