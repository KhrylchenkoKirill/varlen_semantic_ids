#!/usr/bin/env python3
import argparse
import json
import os

import polars as pl
import torch
import torch.nn.functional as F
import yaml

from lib.rkmeans import RKMeans, RKMeansEncoder, RKMeansDecoder
import lib.evaluate as ev
from lib.utils import configure_torch, encode_to_sids_df


def _fit_rkmeans(items: pl.DataFrame, device: str, cfg: dict):
    emb = items["embed"].to_torch().to(torch.float32).to(device, non_blocking=True)
    emb = F.normalize(emb, dim=-1).to(torch.bfloat16).contiguous()

    q = RKMeans(
        num_levels=cfg["num_levels"],
        num_clusters=cfg["num_clusters"],
        num_iters=cfg["num_iters"],
        batch_size=cfg["fit_batch_size"],
    )
    q.fit(emb)

    encoder = RKMeansEncoder(q).inference()
    decoder = RKMeansDecoder(q).inference()
    return encoder, decoder


def main(cfg):
    configure_torch()

    data_dir = cfg["data_dir"]
    device = cfg["device"]
    mode = cfg.get("mode", "train")  # "train" | "train_holdout"
    id_col = cfg.get("id_col", "item_id")

    train_items = pl.read_parquet(os.path.join(data_dir, "dvae_train_items.parquet"))
    holdout_items = pl.read_parquet(os.path.join(data_dir, "dvae_holdout_items.parquet"))
    cold_items = pl.read_parquet(os.path.join(data_dir, "dvae_cold_items.parquet"))

    if mode == "train":
        encoder, decoder = _fit_rkmeans(train_items, device, cfg)

        metrics = ev.evaluate_all(
            train_items=train_items,
            holdout_items=holdout_items,
            cold_items=cold_items,
            encoder=encoder,
            decoder=decoder,
            batch_size=cfg["eval_batch_size"],
            device=device,
            per_step=cfg["per_step"],
            expected_holdout_frac=cfg["expected_holdout_frac"],
        )

        os.makedirs(os.path.dirname(cfg["metrics_json_path"]) or ".", exist_ok=True)
        with open(cfg["metrics_json_path"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    elif mode == "train_holdout":
        # ---- fit on train+holdout
        train_holdout = pl.concat([train_items, holdout_items], how="vertical", rechunk=True)
        encoder, decoder = _fit_rkmeans(train_holdout, device, cfg)

        metrics = ev.evaluate_all(
            splits={"train_holdout": train_holdout},
            encoder=encoder,
            decoder=decoder,
            batch_size=cfg["eval_batch_size"],
            device=device,
            per_step=cfg["per_step"],
        )

        out_dir = cfg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        sids_path = os.path.join(out_dir, "sids.parquet")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        sids = encode_to_sids_df(
            train_holdout,
            encoder,
            id_col=id_col,
            batch_size=cfg["eval_batch_size"],
            device=device,
        )
        sids.write_parquet(sids_path)

    else:
        raise ValueError(f"Unknown mode={mode!r}. Expected 'train' or 'train_holdout'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
