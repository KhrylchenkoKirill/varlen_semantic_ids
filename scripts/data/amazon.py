import os
import tqdm
import json

import numpy as np
import polars as pl

from varlen_sids.scripts.data.utils import preprocess_data


TEST_INTERVAL = 7 * 24 * 60 * 60 * 4 # 4 weeks


# actual data should be downloaded through https://amazon-reviews-2023.github.io/
def process(reviews_path, meta_path, hf_token=None):
    from huggingface_hub import login
    from sentence_transformers import SentenceTransformer
    import torch

    if hf_token is None:
        login(token=os.environ["HF_TOKEN"])
    else:
        login(token=hf_token)

    def generator(path):
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line.strip())

    interactions = pl.DataFrame(generator(reviews_path)) \
        .filter(pl.col('rating') >= 4.)
    items = interactions['parent_asin'].unique()

    meta = pl.DataFrame(generator(meta_path))
    meta = meta.join(items, on='parent_asin', how='semi')
    texts = meta.map_rows(lambda x: f'Title: {x[1]} | Store: {x[9]} | Main category: {x[0]}')

    model = SentenceTransformer("google/embeddinggemma-300m")
    with torch.autocast('cuda', torch.bfloat16):
        item_embeddings = model.encode_document(
            texts['map'].to_list(), 
            batch_size=128, 
            show_progress_bar=True, 
            truncate_dim=128, 
            normalize_embeddings=True,
            device='cuda'
        )
    item_embeddings = pl.DataFrame({'parent_asin': meta['parent_asin'], 'embed': item_embeddings})

    return interactions, item_embeddings


def main(data_dir, dst_dir, core_threshold=16, holdout_frac=0.1, seed=42):
    item_embeddings = pl.read_parquet(os.path.join(data_dir, 'embeddings.parquet'))

    interactions = pl.read_parquet(os.path.join(data_dir, 'interactions.parquet')) \
        .filter(pl.col('rating') >= 4.) \
        .select([
            pl.col('user_id'),
            pl.col('parent_asin').alias('item_id'),
            pl.col('timestamp') // 1000
        ])

    max_timestamp = interactions['timestamp'].max()
    train = interactions.filter(pl.col('timestamp') < max_timestamp - TEST_INTERVAL)
    test = interactions.filter(pl.col('timestamp') >= max_timestamp - TEST_INTERVAL)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed)


if __name__ == '__main__':
    main(
        data_dir='../data/amazon/toys_and_games', 
        dst_dir='./data/amazon_toys_and_games', 
        core_threshold=5,
        holdout_frac=0.1,
        seed=42
    )
