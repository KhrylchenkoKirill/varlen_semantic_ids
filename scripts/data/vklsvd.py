import os
import tqdm

import numpy as np
import polars as pl

from varlen_sids.scripts.data.utils import preprocess_data

NUM_TRAIN_VKLSVD_WEEKS = 25


def download(dst_dir):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import numpy as np

    for week in range(25):
        ds = load_dataset('deepvk/VK-LSVD', data_dir='subsamples/ur0.1/train', data_files=f'week_{week:02}.parquet')
        ds = ds['train'].to_polars()
        ds = ds.filter(pl.col('like'))
        ds.write_parquet(os.path.join(dst_dir, 'week_{week:02}.parquet'))
    ds = load_dataset('deepvk/VK-LSVD', data_dir='subsamples/ur0.1/validation', data_files='week_25.parquet')
    ds = ds['train'].to_polars()
    ds = ds.filter(pl.col('like'))
    ds.write_parquet(os.path.join(dst_dir, 'week_25.parquet'))

    hf_hub_download(
        repo_id='deepvk/VK-LSVD', repo_type='dataset',
        filename='metadata/item_embeddings.npz', local_dir=dst_dir
    )

    item_embeddings = np.load(os.path.join(dst_dir, 'item_embeddings.npz'))
    item_embeddings = pl.DataFrame({'item_id': item_embeddings['item_id'], 'embed': item_embeddings['embedding']})
    item_embeddings.write_parquet(os.path.join(dst_dir, 'embeddings.parquet'))


def main(data_dir, dst_dir, core_threshold=16, holdout_frac=0.1, seed=42):
    item_embeddings = pl.read_parquet(os.path.join(data_dir, 'embeddings.parquet'))

    train = []
    for week in tqdm.tqdm(range(NUM_TRAIN_VKLSVD_WEEKS), total=NUM_TRAIN_VKLSVD_WEEKS):
        train.append(
            pl.scan_parquet(os.path.join(data_dir, f'week_{week:02}.parquet')) \
                .filter(pl.col('like')) \
                .select('user_id', 'item_id') \
                .join(item_embeddings.lazy().select('item_id'), on='item_id', how='semi') \
                .collect(engine='streaming')
        )
    train = pl.concat(train).with_row_index('timestamp')
    
    test = pl.scan_parquet(os.path.join(data_dir, f'week_25.parquet')) \
        .filter(pl.col('like')) \
        .select('user_id', 'item_id') \
        .join(item_embeddings.lazy().select('item_id'), on='item_id', how='semi') \
        .collect(engine='streaming') \
        .with_row_index('timestamp') \
        .with_columns(timestamp=pl.col('timestamp') + train['timestamp'].max() + 1)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed)


if __name__ == '__main__':
    main(
        data_dir='../data/vklsvd/', 
        dst_dir='./data/vklsvd', 
        core_threshold=16,
        holdout_frac=0.1,
        seed=42
    )

    seqrec_test_interactions = pl.read_parquet('./data/vklsvd/seqrec_test_interactions.parquet')
    sampled_users = seqrec_test_interactions.select('user_id') \
        .unique().sample(fraction=0.1, shuffle=True, seed=42)\

    seqrec_test_interactions \
        .join(sampled_users, on='user_id', how='semi') \
        .write_parquet('./data/vklsvd/seqrec_test_sample_interactions.parquet')
