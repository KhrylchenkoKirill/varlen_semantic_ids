#!/usr/bin/env python3
import argparse
import json
import os

import yaml
import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from lib.data import EventsDataset
from lib.utils import get_cosine_scheduler, configure_torch, encode_to_sids_df
from lib.layers import Encoder, Decoder
from lib.game import Game, VarlenGame
from lib.evaluate import evaluate_all, evaluate_for_tb
import lib.evaluate as ev


def main(cfg):
    configure_torch()

    data_dir = cfg["data_dir"]
    device = cfg["device"]
    mode = cfg.get("mode", "train")  # "train" | "train_holdout"
    id_col = cfg.get("id_col", "item_id")

    torch.manual_seed(cfg["seed"])
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])

    train_items = pl.read_parquet(os.path.join(data_dir, "dvae_train_items.parquet"))
    holdout_items = pl.read_parquet(os.path.join(data_dir, "dvae_holdout_items.parquet"))
    cold_items = pl.read_parquet(os.path.join(data_dir, "dvae_cold_items.parquet"))

    if mode == "train":
        fit_items = train_items
    elif mode == "train_holdout":
        fit_items = pl.concat([train_items, holdout_items], how="vertical", rechunk=True)
    else:
        raise ValueError(f"Unknown mode={mode!r}. Expected 'train' or 'train_holdout'.")

    fit_embeddings = fit_items["embed"].to_torch().to(torch.float32).to(device, non_blocking=True)
    fit_embeddings = F.normalize(fit_embeddings, dim=-1).to(torch.bfloat16).contiguous()

    interactions = pl.read_parquet(os.path.join(data_dir, cfg["train_interactions_file"]))
    if mode == "train_holdout":
        assert cfg.get("holdout_interactions_file") is not None
        interactions_hold = pl.read_parquet(os.path.join(data_dir, cfg["holdout_interactions_file"]))
        interactions = pl.concat([interactions, interactions_hold], how="vertical", rechunk=True)

    mapping = fit_items.select(id_col).with_row_index("token_id")
    interactions = (
        interactions
        .join(mapping, on=id_col, how="left")
        .filter(pl.col("token_id").is_not_null())
    )

    dataloader = EventsDataset(
        interactions,
        batch_size=cfg["batch_num_tokens"],
        shuffle=cfg["shuffle"],
        drop_last=cfg["drop_last"],
        embedding_table=fit_embeddings,
    )

    encoder = Encoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        maxlen=cfg["maxlen"],
        codebook_dropout=cfg["codebook_dropout"],
        dropout=cfg["dropout"],
        varlen=cfg["varlen"],
        shared_codebooks=cfg["shared_codebooks"],
        init_logit_scale=cfg["init_logit_scale"],
        init_gamma=cfg["init_gamma"]
    )
    decoder = Decoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"] * cfg["decoder_hidden_mul"],
        maxlen=cfg["maxlen"],
        dropout=cfg["dropout"],
        num_layers=cfg["decoder_num_layers"],
    )
    graph = (VarlenGame(encoder, decoder) if cfg["varlen"] else Game(encoder, decoder)).to(device)
    graph = torch.compile(graph, dynamic=False)
    graph.train()

    enc_inf = torch.compile(encoder.inference(mode="argmax"), dynamic=False)
    dec_inf = torch.compile(decoder.inference(), dynamic=False)

    optimizer = torch.optim.AdamW(
        graph.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)

    writer = None
    if cfg.get("tensorboard_logdir") is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg["tensorboard_logdir"])

    length_cost_t = None
    if cfg["varlen"]:
        length_cost_t = torch.tensor(cfg["length_cost"], device=device)

    sampled_train_items = train_items.sample(
        fraction=cfg["train_sample_frac_for_eval"],
        seed=cfg["seed"],
        shuffle=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = cfg["num_epochs"] * steps_per_epoch
    tokens_passed = 0
    global_step = 0

    pbar = tqdm(
        total=total_steps,
        desc=f"train({mode})",
        disable=not cfg["enable_progress_bar"],
        dynamic_ncols=True,
    )

    for epoch in range(cfg["num_epochs"]):
        graph.train()
        for batch in dataloader:
            tau = get_cosine_scheduler(
                global_step,
                start=cfg["tau_start"],
                end=cfg["tau_end"],
                total_steps=steps_per_epoch,
            )

            beta_t = None
            if cfg.get("beta_end") is not None:
                b = get_cosine_scheduler(
                    global_step,
                    start=cfg["beta_start"],
                    end=cfg["beta_end"],
                    total_steps=steps_per_epoch,
                )
                beta_t = torch.tensor(b, device=device)

            free_bits_t = None
            if cfg.get("free_bits") is not None:
                free_bits_t = torch.tensor(cfg["free_bits"], device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                kwargs = dict(
                    x=batch.embeddings,
                    tau=torch.tensor(tau, device=device),
                    beta=beta_t,
                    free_bits=free_bits_t,
                )
                if cfg["varlen"]:
                    kwargs["length_cost"] = length_cost_t

                loss, metrics = graph(**kwargs)
                loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if writer is not None and (global_step % cfg["log_every_steps"] == 0):
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", float(value), tokens_passed)
                writer.add_scalar("params/tau", float(tau), tokens_passed)
                if beta_t is not None:
                    writer.add_scalar("params/beta", float(beta_t.item()), tokens_passed)

            tokens_passed += int(getattr(batch, "size", 0))
            global_step += 1
            pbar.update(1)

            if writer is not None and cfg["eval_every_steps"] > 0 and (global_step % cfg["eval_every_steps"] == 0):
                graph.eval()
                m = evaluate_for_tb(
                    encoder=enc_inf,
                    decoder=dec_inf,
                    train_items=sampled_train_items,
                    holdout_items=holdout_items,
                    cold_items=None,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    detail=cfg["eval_detail"],
                )
                if writer is not None:
                    for k, v in m.items():
                        writer.add_scalar(k, float(v), tokens_passed)
                graph.train()

            if global_step % cfg["progress_bar_every"] == 0:
                pbar.set_postfix(epoch=epoch, loss=float(loss))

    pbar.close()

    graph.eval()
    with torch.inference_mode():
        if mode == "train":
            metrics = evaluate_all(
                train_items=train_items,
                holdout_items=holdout_items,
                cold_items=cold_items,
                encoder=enc_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
                expected_holdout_frac=cfg["expected_holdout_frac"],
            )
        else:
            metrics = evaluate_all(
                splits={
                    "train_holdout": fit_items,
                    "cold": cold_items,
                },
                encoder=enc_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
            )

    # ---- Saving: unified under out_dir in train_holdout mode ----
    if mode == "train_holdout":
        out_dir = cfg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        sids_path = os.path.join(out_dir, "sids.parquet")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        sids = encode_to_sids_df(
            fit_items,
            enc_inf,
            id_col=id_col,
            batch_size=cfg["eval_batch_size"],
            device=device,
        )
        sids.write_parquet(sids_path)
    else:
        # Backward-compatible behavior for plain train mode
        os.makedirs(os.path.dirname(cfg["metrics_json_path"]) or ".", exist_ok=True)
        with open(cfg["metrics_json_path"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if cfg.get("save_weights_path") is not None:
        os.makedirs(os.path.dirname(cfg["save_weights_path"]) or ".", exist_ok=True)
        torch.save(graph.state_dict(), cfg["save_weights_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main()
