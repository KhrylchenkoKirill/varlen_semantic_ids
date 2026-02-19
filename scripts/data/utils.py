import os

import polars as pl


def preprocess_data(
        train: pl.DataFrame,
        test: pl.DataFrame,
        item_embeddings: pl.DataFrame,
        dst_dir: str,
        core_threshold: int = 16,
        holdout_frac: float = 0.1,
        seed: int = 42,
        verbose: bool = False,
        topk_head: int = 30_000
):
    os.makedirs(dst_dir, exist_ok=True)
    
    # Keep only items with embeddings
    train = train.join(item_embeddings.select("item_id"), on="item_id", how="semi")
    test = test.join(item_embeddings.select("item_id"), on="item_id", how="semi")

    # Item frequencies in *training* (this defines "head")
    train_items = (
        train.select("item_id")
        .group_by("item_id")
        .len()
        .rename({"len": "train_count"})
        .sort(["train_count", "item_id"], descending=True)
    )

    head_items = (
        train_items
        .head(topk_head)
        .select("item_id")
        .with_columns(head=pl.lit(True))
    )

    # cold items: present in test, absent in train
    cold_items = (
        test.select("item_id")
        .group_by("item_id")
        .len()
        .rename({"len": "test_count"})
        .sort(["test_count", "item_id"], descending=True)
        .join(train_items.select("item_id"), on="item_id", how="anti")
        .join(head_items, on="item_id", how="left")
        .with_columns(head=pl.col("head").fill_null(False))
    )

    if verbose:
        print(f"Train items: {train_items.height:,}")

    core_items = train_items.filter(pl.col("train_count") >= core_threshold)

    if verbose:
        print(f"Core items (count >= {core_threshold}): {core_items.height:,}")

    # seqrec interactions: only core items
    seqrec_train = train.join(core_items.select("item_id"), on="item_id", how="semi")
    seqrec_test = test.join(core_items.select("item_id"), on="item_id", how="semi")

    # holdout split over core items
    holdout_items = core_items.sample(fraction=holdout_frac, shuffle=True, seed=seed)
    dvae_train_items = core_items.join(holdout_items.select("item_id"), on="item_id", how="anti")

    # dvae training interactions: train interactions restricted to dvae_train_items
    dvae_training_data = train.join(dvae_train_items.select("item_id"), on="item_id", how="semi")
    dvae_holdout_training_data = train.join(holdout_items.select("item_id"), on="item_id", how="semi")

    # mark positives in seqrec_test (one positive per user)
    seqrec_test_positives = seqrec_test \
        .group_by("user_id") \
        .agg(pl.all().sample(n=1, shuffle=True, seed=seed)) \
        .explode(["timestamp", "item_id"]) \
        .with_columns(target=pl.lit(True))
    
    seqrec_test = (
        seqrec_test
        .join(
            seqrec_test_positives.select(["user_id", "timestamp", "item_id", "target"]),
            on=["user_id", "timestamp", "item_id"], 
            how="left"
        )
        .with_columns(target=pl.col("target").fill_null(False))
        .with_columns(
            target = (
                pl.when(pl.col("target"))
                .then(1).otherwise(0)
                .cum_sum()
                .over("user_id")
                .eq(1) & pl.col("target")
            )
        )
    )

    # Helper: attach embeddings + head flag
    def enrich_items(items_df: pl.DataFrame) -> pl.DataFrame:
        return (
            items_df
            .join(head_items, on="item_id", how="left")
            .with_columns(head=pl.col("head").fill_null(False))
            .join(item_embeddings, on="item_id", how="left")
        )

    # Write item tables (now with `head`)
    enrich_items(dvae_train_items).write_parquet(os.path.join(dst_dir, "dvae_train_items.parquet"))
    enrich_items(holdout_items).write_parquet(os.path.join(dst_dir, "dvae_holdout_items.parquet"))
    enrich_items(cold_items).write_parquet(os.path.join(dst_dir, "dvae_cold_items.parquet"))

    # Write interaction tables
    dvae_training_data.write_parquet(os.path.join(dst_dir, "dvae_train_interactions.parquet"))
    dvae_holdout_training_data.write_parquet(os.path.join(dst_dir, "dvae_holdout_interactions.parquet"))
    seqrec_train.write_parquet(os.path.join(dst_dir, "seqrec_train_interactions.parquet"))
    seqrec_test.write_parquet(os.path.join(dst_dir, "seqrec_test_interactions.parquet"))

    if verbose:
        print(f"Dvae train interactions: {dvae_training_data.height:,}")
        print(f"Dvae holdout interactions: {dvae_holdout_training_data.height:,}")
        print(f"Dvae train items:        {dvae_train_items.height:,}")
        print(f"Dvae holdout items:      {holdout_items.height:,}")
        print(f"Dvae cold items:         {cold_items.height:,}")
        print(f"Seqrec train interactions:{seqrec_train.height:,}")
        print(f"Seqrec test interactions: {seqrec_test.height:,}")
