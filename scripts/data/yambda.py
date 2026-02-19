import os
import tqdm

import numpy as np
import polars as pl

from .utils import preprocess_data


TEST_INTERVAL = 7 * 24 * 60 * 60 # one week


def download(dst_dir):
    from datasets import load_dataset

    ds = load_dataset("yandex/yambda", data_dir="flat/5b", data_files="likes.parquet")
    ds = ds['train'].to_polars()
    ds.write_parquet(os.path.join(dst_dir, 'interactions.parquet'))

    embeddings = load_dataset("yandex/yambda", data_dir="", data_files="embeddings.parquet")
    embeddings = embeddings['train'].to_polars()

    liked_items = ds.select('item_id').unique()
    filtered_embeddings = embeddings \
        .join(liked_items, on='item_id', how='semi') \
        .select('item_id', pl.col('embed').cast(pl.Array(pl.Float32, (128,)))) \
        .write_parquet(os.path.join(dst_dir, 'embeddings.parquet'))
    
    return ds, filtered_embeddings


def main(data_dir, dst_dir, core_threshold=16, holdout_frac=0.1, seed=42, topk_head=30000):
    item_embeddings = pl.read_parquet(os.path.join(data_dir, 'embeddings.parquet'))
    
    interactions = pl.read_parquet(os.path.join(data_dir, 'interactions.parquet')) \
        .join(item_embeddings, on='item_id', how='semi') \
        .rename({'uid': 'user_id'})

    max_timestamp = interactions['timestamp'].max()
    train = interactions.filter(pl.col('timestamp') < max_timestamp - TEST_INTERVAL)
    test = interactions.filter(pl.col('timestamp') >= max_timestamp - TEST_INTERVAL)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed, topk_head=topk_head, verbose=True)


if __name__ == '__main__':
    main(
        data_dir='../data/yambda/', 
        dst_dir='./data/yambda', 
        core_threshold=16,
        holdout_frac=0.1,
        seed=42,
        topk_head=30000
    )
    
    seqrec_test_interactions = pl.read_parquet('./data/yambda/seqrec_test_interactions.parquet')
    sampled_users = seqrec_test_interactions.select('user_id') \
        .unique().sample(fraction=0.05, shuffle=True, seed=42)\

    seqrec_test_interactions \
        .join(sampled_users, on='user_id', how='semi') \
        .write_parquet('./data/yambda/seqrec_test_sample_interactions.parquet')
