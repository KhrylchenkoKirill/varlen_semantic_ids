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
from lib.utils import configure_torch, encode_to_sids_df
from lib.layers import Decoder
from lib.reinforce import ReinforceEncoder, ReinforceFixedGame, ReinforceVarlenGame
from lib.evaluate import evaluate_all, evaluate_for_tb


import math

def _cosine_interp(start: float, end: float, t01: float) -> float:
    t01 = max(0.0, min(1.0, float(t01)))
    # cosine from start -> end
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t01)))


def _schedule_progress(
    *,
    cfg: dict,
    global_step: int,
    step_in_epoch: int,
    epoch: int,
    steps_per_epoch: int,
) -> float:
    """
    Returns progress in [0,1] for the schedule, depending on cfg["schedule_mode"].

    schedule_mode:
      - "steps": progress = global_step / (schedule_steps-1)
      - "epochs": progress = (epoch + step_in_epoch/steps_per_epoch) / schedule_epochs
    """
    mode = str(cfg.get("schedule_mode", "epochs")).lower()

    if mode == "steps":
        total = int(cfg.get("schedule_steps", 0) or 0)
        if total <= 1:
            return 1.0
        return max(0.0, min(1.0, global_step / float(total - 1)))

    if mode == "epochs":
        total_epochs = float(cfg.get("schedule_epochs", 1) or 1)
        if total_epochs <= 0:
            return 1.0
        frac_epoch = 0.0 if steps_per_epoch <= 0 else (step_in_epoch / float(steps_per_epoch))
        return max(0.0, min(1.0, (epoch + frac_epoch) / total_epochs))

    raise ValueError(f"Unknown schedule_mode={mode!r}. Use 'steps' or 'epochs'.")


def _scheduled_value(
    cfg: dict,
    name: str,
    *,
    global_step: int,
    step_in_epoch: int,
    epoch: int,
    steps_per_epoch: int,
) -> float:
    """
    Supports:
      - constant: <name> or <name>_start
      - cosine schedule: <name>_start -> <name>_end, with progress controlled by schedule_mode

    If <name>_end is absent -> constant.
    """
    end_key = f"{name}_end"
    start_key = f"{name}_start"

    # no schedule => constant
    if end_key not in cfg or cfg[end_key] is None:
        if name in cfg and cfg[name] is not None:
            return float(cfg[name])
        if start_key in cfg and cfg[start_key] is not None:
            return float(cfg[start_key])
        return 0.0

    # schedule enabled
    start = float(cfg.get(start_key, cfg.get(name, 0.0)))
    end = float(cfg[end_key])

    p = _schedule_progress(
        cfg=cfg,
        global_step=global_step,
        step_in_epoch=step_in_epoch,
        epoch=epoch,
        steps_per_epoch=steps_per_epoch,
    )
    return _cosine_interp(start, end, p)


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

    sender = ReinforceEncoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        maxlen=cfg["maxlen"],
        codebook_dropout=cfg["codebook_dropout"],
        dropout=cfg["dropout"],
        shared_codebooks=cfg["shared_codebooks"],
        init_logit_scale=cfg["init_logit_scale"],
        init_gamma=cfg["init_gamma"],
        varlen=cfg["varlen"],
    )

    decoder = Decoder(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"] * cfg["decoder_hidden_mul"],
        maxlen=cfg["maxlen"],
        dropout=cfg["dropout"],
        num_layers=cfg["decoder_num_layers"],
    )

    if cfg["varlen"]:
        graph = ReinforceVarlenGame(sender, decoder, baseline_momentum=cfg.get("baseline_momentum", 0.99))
    else:
        graph = ReinforceFixedGame(sender, decoder, baseline_momentum=cfg.get("baseline_momentum", 0.99))

    graph = graph.to(device)
    graph = torch.compile(graph, dynamic=False)
    graph.train()

    sender_inf = torch.compile(sender.inference(), dynamic=False)
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
        for step_in_epoch, batch in enumerate(dataloader):
            # schedules: reach *_end by end of epoch (if provided)
            sender_entropy_coeff = _scheduled_value(
                cfg, "sender_entropy_coeff",
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
            )

            if cfg["varlen"]:
                length_cost = _scheduled_value(
                    cfg, "length_cost",
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sec_t = torch.tensor(sender_entropy_coeff, device=device)
                if cfg["varlen"]:
                    lc_t = torch.tensor(length_cost, device=device)
                    loss, metrics = graph(x=batch.embeddings, sender_entropy_coeff=sec_t, length_cost=lc_t)
                else:
                    loss, metrics = graph(x=batch.embeddings, sender_entropy_coeff=sec_t)

                loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if writer is not None and (global_step % cfg["log_every_steps"] == 0):
                for name, value in metrics.items():
                    writer.add_scalar(f"train/{name}", float(value), tokens_passed)

            tokens_passed += int(getattr(batch, "size", 0))
            global_step += 1
            pbar.update(1)

            if writer is not None and cfg["eval_every_steps"] > 0 and (global_step % cfg["eval_every_steps"] == 0):
                graph.eval()
                m = evaluate_for_tb(
                    encoder=sender_inf,
                    decoder=dec_inf,
                    train_items=sampled_train_items,
                    holdout_items=holdout_items,
                    cold_items=None,
                    device=device,
                    batch_size=cfg["eval_batch_size"],
                    detail=cfg["eval_detail"],
                )
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
                encoder=sender_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
                expected_holdout_frac=cfg["expected_holdout_frac"],
            )
        else:
            metrics = evaluate_all(
                splits={"train_holdout": fit_items, "cold": cold_items},
                encoder=sender_inf,
                decoder=dec_inf,
                batch_size=cfg["eval_batch_size"],
                device=device,
                per_step=cfg["per_step"],
            )

    if mode == "train_holdout":
        out_dir = cfg["out_dir"]
        os.makedirs(out_dir, exist_ok=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        sids_path = os.path.join(out_dir, "sids.parquet")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        sids = encode_to_sids_df(
            fit_items,
            sender_inf,
            id_col=id_col,
            batch_size=cfg["eval_batch_size"],
            device=device,
        )
        sids.write_parquet(sids_path)
    else:
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
    main(cfg)
