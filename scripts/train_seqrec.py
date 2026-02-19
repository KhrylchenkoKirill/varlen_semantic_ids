#!/usr/bin/env python3
import os
import json
from os.path import join
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

from lib.gpt import GPT
from lib.seqrec.muon import NorMuon
from lib.seqrec.utils import train_loop, step_optimizers
from lib.seqrec.beam_search import BeamSearchVarLen, KVCacheGPT, iter_length_buckets
from lib.seqrec.evaluate import calculate_metrics
from lib.seqrec.data import SeqrecDataset, SeqrecBatch
from lib.utils import configure_torch



def require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[p]
    return cur


def ceil_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _mean_scalar(df: pl.DataFrame) -> float:
    if df.height == 0:
        return 0.0
    return float(df.item())


@torch.no_grad()
def cast_modules_to_bf16(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.bfloat16()


class Graph(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, depth: int):
        super().__init__()
        self.depth = depth
        model_dim = depth * 64

        self.gpt = GPT(
            model_dim=model_dim,
            num_layers=depth,
            num_heads=model_dim // 128,
            max_seq_len=max_seq_len,
            embeddings_cls=lambda: nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim),
        )

        self.head = nn.Linear(model_dim, vocab_size, bias=False)
        self.head.weight.detach().zero_()

    def setup_optimizers(self, adam_lr: float, muon_lr: float, weight_decay: float):
        hidden_matrix_params = [
            p for n, p in self.gpt.blocks.named_parameters()
            if p.ndim >= 2 and "embed" not in n and "gate" not in n
        ]
        embed_params = [p for n, p in self.gpt.named_parameters() if "embed" in n]
        scalar_params = [p for p in self.parameters() if p.ndim < 2]
        head_params = list(self.head.parameters())
        gate_params = [p for n, p in self.named_parameters() if "gate" in n]

        opt1 = torch.optim.AdamW(
            [
                {"params": scalar_params, "lr": 5.0 * adam_lr},
                {"params": embed_params, "lr": 75.0 * adam_lr},
                {"params": head_params, "lr": 1.0 * adam_lr},
            ],
            lr=adam_lr,
            betas=(0.65, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        opt2 = NorMuon(
            hidden_matrix_params + gate_params,
            lr=muon_lr,
            momentum=0.95,
            weight_decay=weight_decay,
            beta2=0.95,
        )

        for opt in (opt1, opt2):
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]

        return [opt1, opt2]

    def forward(self, batch: SeqrecBatch, with_metrics: bool = False):
        with torch.autocast("cuda", torch.bfloat16):
            x = self.gpt(batch.inputs)
            logits = self.head(x)
            logits = 30.0 * torch.sigmoid(logits / 7.5)

            flat_targets = batch.targets.view(-1)
            logits = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits, flat_targets, ignore_index=-1)

        if with_metrics:
            return loss, {"loss": loss}
        return loss


def compute_grad_accum(cfg: Dict[str, Any], max_seq_len: int) -> int:
    ideal = require(cfg, "train.ideal_batch_size_tokens")
    device_batch_size = require(cfg, "train.device_batch_size")
    tokens_per_batch = device_batch_size * max_seq_len
    return ideal // tokens_per_batch


def run_train(cfg: Dict[str, Any], stats: Dict[str, Any], pretrain_df: pl.DataFrame) -> Dict[str, Any]:
    max_seq_len = int(stats["pretrain_max_len_observed"])
    vocab_size = int(stats["vocab_size"])

    dataloader = SeqrecDataset(
        pretrain_df,
        batch_size=require(cfg, "train.device_batch_size"),
        seq_len=max_seq_len,
        device=require(cfg, "device"),
        parquet_bs=int(cfg.get("train", {}).get("dataloader_chunk_rows", 1024)),
    )

    grad_accum_steps = compute_grad_accum(cfg, max_seq_len)
    num_iterations = len(dataloader) // grad_accum_steps

    print("Num training tokens:", getattr(dataloader, "total_num_tokens", "unknown"))
    print("Grad accum steps:", grad_accum_steps)
    print("Total num steps:", num_iterations)

    graph = Graph(vocab_size=vocab_size, max_seq_len=max_seq_len, depth=require(cfg, "train.depth")).cuda()
    cast_modules_to_bf16(graph)

    if require(cfg, "train.compile"):
        graph = torch.compile(graph, dynamic=require(cfg, "train.compile_dynamic"))

    optimizers = graph.setup_optimizers(
        adam_lr=require(cfg, "train.adam_lr"),
        muon_lr=require(cfg, "train.muon_lr"),
        weight_decay=require(cfg, "train.weight_decay"),
    )

    def step_optimizers_func(step: int):
        step_optimizers(
            step,
            optimizers,
            graph,
            num_iterations,
            warmdown_ratio=require(cfg, "train.warmdown_ratio"),
            final_lr_frac=require(cfg, "train.final_lr_frac"),
        )

    last_step, tokens_passed = train_loop(
        graph=graph,
        train_dataloader=dataloader,
        log_dir=cfg.get("log_dir"),
        step_optimizers_func=step_optimizers_func,
        num_iterations=num_iterations,
        grad_accum_steps=grad_accum_steps,
        grad_clip=require(cfg, "train.grad_clip"),
        valid_dataloaders=None,
        eval_every=-1,
        custom_logging=None,
        checkpoint_dir=None,
        checkpoint_every=None,
        custom_validation=None,
    )

    return graph


@torch.no_grad()
def run_eval(
        cfg: Dict[str, Any],
        stats: Dict[str, Any],
        graph: Graph,
        processed_semantic_ids: pl.DataFrame,
        test_df: pl.DataFrame,
        targets_df: pl.DataFrame,     # columns: user_id, item_id(list[int])
        head_items_df: pl.DataFrame,  # column: item_id
) -> Dict[str, float]:
    max_seq_len = int(stats["pretrain_max_len_observed"])
    vocab_size = int(stats["vocab_size"])

    depth = graph.depth
    model_dim = depth * 64
    graph.eval()

    gpt_inf = KVCacheGPT(
        model_dim=model_dim,
        num_layers=depth,
        num_heads=model_dim // 128,
        max_seq_len=max_seq_len,
        embeddings_cls=lambda: nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim),
    ).cuda()
    gpt_inf.load_state_dict(graph.gpt.state_dict())
    gpt_inf.eval()

    semantic_ids_v = processed_semantic_ids["sid"].to_list()
    beam_search = BeamSearchVarLen(
        gpt_inf,
        graph.head,
        semantic_ids_v,
        constrain_only_at_end=False,
        end_beam_mul=1,
        beam_size=require(cfg, "eval.beam_size"),
    )

    code2items = defaultdict(list)
    for item_id, code in processed_semantic_ids.select("item_id", "sid").iter_rows():
        code2items[tuple(code)].append(int(item_id))

    max_candidates = require(cfg, "eval.max_candidates")
    per_code_max_items = require(cfg, "eval.per_code_max_items")

    mean_sid_len = 0.0
    mean_sid_len_cnt = 0
    candidates: Dict[int, List[int]] = {}

    for uids, token_lists in tqdm.tqdm(
        iter_length_buckets(
            test_df,
            uid_col="uid",
            token_col="token_id",
            max_batch_tokens=require(cfg, "eval.max_batch_tokens"),
            max_batch_users=require(cfg, "eval.max_batch_users"),
        ),
        desc="beam_search_batched",
        dynamic_ncols=True,
    ):
        tokens_BT = torch.tensor(token_lists, dtype=torch.long, device="cuda")

        # codes_per_user: List[U][<=beam_size][code_tokens]
        codes_per_user = beam_search(tokens_BT)

        for uid, codes in zip(uids, codes_per_user):
            uid_candidates: List[int] = []

            for c in codes:
                key = tuple(c)
                mean_sid_len_cnt += 1
                mean_sid_len += (float(len(key)) - mean_sid_len) / mean_sid_len_cnt

                if key not in code2items:
                    continue
                for cand in code2items[key][:per_code_max_items]:
                    uid_candidates.append(cand)
                    if len(uid_candidates) >= max_candidates:
                        break
                if len(uid_candidates) >= max_candidates:
                    break

            candidates[int(uid)] = uid_candidates

    head_items_set = set(head_items_df["item_id"].to_list())

    catalog_size = float(require(cfg, "eval.catalog_size"))
    k_list = require(cfg, "eval.k_list")

    metrics: Dict[str, float] = {}
    metrics["mean_candidate_sid_length"] = float(mean_sid_len) if mean_sid_len_cnt else float("nan")
    metrics["unique_sids_per_user"] = float(mean_sid_len_cnt) / len(candidates)

    for k in k_list:
        unique_candidates = set()
        for cand in candidates.values():
            unique_candidates.update(cand[:k])
        metrics[f"coverage@{k}"] = len(unique_candidates) / catalog_size

    for k in k_list:
        for key, value in calculate_metrics(candidates, targets_df, k=k).items():
            metrics[f"{key}@{k}"] = float(value)

    # long_tail_share: 1 - mean(head_fraction_in_candidates)
    long_tail_share = 0.0
    cnt = 0
    for cand in candidates.values():
        if not cand:
            continue
        head_frac = sum(el in head_items_set for el in cand) / len(cand)
        cnt += 1
        long_tail_share += (head_frac - long_tail_share) / cnt
    metrics["long_tail_share"] = 1.0 - float(long_tail_share)

    # head/tail targets split
    pop_test = targets_df.explode("item_id").join(head_items_df, on="item_id", how="semi")
    pop_targets = pop_test.group_by("user_id").agg(pl.col("item_id"))
    for k in k_list:
        for key, value in calculate_metrics(candidates, pop_targets, k=k).items():
            metrics[f"head_{key}@{k}"] = float(value)

    tail_test = targets_df.explode("item_id").join(head_items_df, on="item_id", how="anti")
    tail_targets = tail_test.group_by("user_id").agg(pl.col("item_id"))
    for k in k_list:
        for key, value in calculate_metrics(candidates, tail_targets, k=k).items():
            metrics[f"tail_{key}@{k}"] = float(value)

    return metrics


def build_dataframes_and_stats(cfg: Dict[str, Any]) -> Tuple[
        pl.DataFrame,  # pretrain_df (uid, token_id:list)
        pl.DataFrame,  # test_df (uid, token_id:list)
        pl.DataFrame,  # processed_semantic_ids (item_id, sid:list[int], length)
        pl.DataFrame,  # head_items_df (item_id)
        pl.DataFrame,  # targets_df (user_id, item_id:list[int])
        Dict[str, Any],  # stats
]:
    data_dir = require(cfg, "data_dir")

    train_path = join(data_dir, cfg.get("train_interactions_path", "seqrec_train_interactions.parquet"))
    test_path = join(data_dir, cfg.get("test_interactions_path", "seqrec_test_interactions.parquet"))
    sids_path = require(cfg, "semantic_ids_path")

    user_col = cfg.get("user_col", "uid")
    item_col = cfg.get("item_col", "item_id")
    time_col = require(cfg, "time_col")
    target_col = cfg.get("target_col", "target")

    min_item_count = int(cfg.get("min_item_count", 16))
    pop_k = int(cfg.get("pop_k", 30000))

    budget_mode = str(cfg.get("budget_mode", "events")).lower()  # "events" | "tokens"
    total_budget = int(require(cfg, "history_budget"))  # includes USER_START; test reserves next event

    USER_START = int(cfg.get("user_start_token", 0))
    SEP_TOKEN = int(cfg.get("sep_token", 1))
    NUM_SPECIAL_TOKENS = int(cfg.get("num_special_tokens", 2))

    flatten_codebooks = bool(cfg.get("flatten_codebooks", False))
    append_unique_token = bool(cfg.get("append_unique_token", True))
    seed = int(cfg.get("seed", 42))

    TOKEN_DTYPE = pl.UInt32

    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)
    semantic_ids = pl.read_parquet(sids_path)

    need_train = {user_col, item_col, time_col}
    need_test = {user_col, item_col, time_col, target_col}
    need_sids = {item_col, "sid", "length"}
    if not need_train.issubset(train.columns):
        raise ValueError(f"train must contain columns: {sorted(need_train)}")
    if not need_test.issubset(test.columns):
        raise ValueError(f"test must contain columns: {sorted(need_test)}")
    if not need_sids.issubset(semantic_ids.columns):
        raise ValueError(f"semantic_ids must contain columns: {sorted(need_sids)}")

    train_counts = train[item_col].value_counts().sort(["count", item_col], descending=True)
    core_items = train_counts.filter(pl.col("count") >= min_item_count).select(item_col)

    train = train.join(core_items, on=item_col, how="semi")
    test = test.join(core_items, on=item_col, how="semi")

    head_items_df = train_counts.select(item_col).head(pop_k).rename({item_col: "item_id"})

    if flatten_codebooks:
        sid_np = semantic_ids["sid"].to_numpy(writable=True)
        if sid_np.ndim != 2:
            raise ValueError("flatten_codebooks=true expects rectangular sid (same length for all rows)")
        for i in range(sid_np.shape[1] - 1):
            sid_np[:, i + 1] = sid_np[:, i + 1] + sid_np[:, i].max() + 1
        semantic_ids = semantic_ids.with_columns(sid=sid_np)
    
    semantic_ids = semantic_ids.with_columns(
        sid=pl.struct(['sid', 'length']) \
            .map_elements(lambda s: [int(x) + NUM_SPECIAL_TOKENS for x in s['sid'][: int(s['length'])]], return_dtype=pl.List(pl.Int64))
    )

    
    max_sid_token_id = int(semantic_ids["sid"].list.max().max())

    if append_unique_token:
        item_pop = train.group_by(item_col).len()
        cluster_counts = {}
        item_to_unique = {}

        for it, sid_list in (
            semantic_ids.join(item_pop, on=item_col, how="left")
            .fill_null(0)
            .sort(["len", item_col], descending=True)
            .select([item_col, "sid"])
            .iter_rows()
        ):
            key = tuple(sid_list)
            cnt = cluster_counts.get(key, 0)
            item_to_unique[it] = cnt
            cluster_counts[key] = cnt + 1

        unique_token_offset = max_sid_token_id + 1
        item_to_unique_df = pl.DataFrame(
            {item_col: list(item_to_unique.keys()), "unique_value": list(item_to_unique.values())}
        )
        processed_semantic_ids = (
            semantic_ids.join(item_to_unique_df, on=item_col, how="left")
            .with_columns(sid=pl.concat_list([pl.col("sid"), (pl.col("unique_value") + unique_token_offset)]))
            .drop("unique_value")
        )
        vocab_size = int(processed_semantic_ids["sid"].list.max().max()) + 1
        unique_token_offset_saved = unique_token_offset
        sid_len_max = int(processed_semantic_ids["sid"].list.len().max())
        sid_stream_len_max = sid_len_max
    else:
        processed_semantic_ids = semantic_ids.with_columns(sid=pl.concat_list([pl.col("sid"), pl.lit([SEP_TOKEN])]))
        vocab_size = max_sid_token_id + 1
        unique_token_offset_saved = None
        sid_len_max = int(processed_semantic_ids["sid"].list.len().max())
        sid_stream_len_max = sid_len_max

    processed_semantic_ids = processed_semantic_ids.rename({item_col: "item_id"})

    # keep only items that exist in processed_semantic_ids
    train = train.join(processed_semantic_ids.select(["item_id"]), left_on=item_col, right_on="item_id", how="semi")
    test = test.join(processed_semantic_ids.select(["item_id"]), left_on=item_col, right_on="item_id", how="semi")

    def tail_events(df: pl.DataFrame, n_events: int) -> pl.DataFrame:
        if n_events <= 0:
            return df.head(0)
        return df.group_by(user_col, maintain_order=True).tail(n=n_events)

    def tail_tokens(df: pl.DataFrame, token_budget: int) -> pl.DataFrame:
        if token_budget <= 0:
            return df.head(0)
        base = df.join(processed_semantic_ids.select(["item_id", "sid"]), on='item_id', how="left")
        base = base.with_columns(token_len=pl.col("sid").list.len())
        base = base.with_columns(idx=(pl.col(item_col).cum_count().over(user_col) - 1))
        return (
            base.sort([user_col, "idx"], descending=[False, True])
            .with_columns(cum_tokens=pl.col("token_len").cum_sum().over(user_col))
            .filter(pl.col("cum_tokens") <= token_budget)
            .sort([user_col, "idx"])
            .drop(["idx", "cum_tokens"])
        )

    train_sorted = train.sort([user_col, time_col])

    if budget_mode == "events":
        train_hist = tail_events(train_sorted, total_budget)
        test_events_budget = total_budget - 1
    elif budget_mode == "tokens":
        train_tokens_budget = total_budget - 1
        test_tokens_budget = total_budget - 1 - sid_stream_len_max
        train_hist = tail_tokens(train_sorted, train_tokens_budget)
    else:
        raise ValueError("budget_mode must be 'events' or 'tokens'")

    users_train = train_hist.select(user_col).unique()

    if budget_mode == "tokens":
        train_hist_sids = train_hist
    else:
        train_hist_sids = train_hist.join(
            processed_semantic_ids.select(["item_id", "sid"]), on="item_id", how="left"
        )

    pretrain_stream = (
        train_hist_sids.explode("sid")
        .select([user_col, pl.col("sid").alias("token_id")])
        .with_columns(pl.col("token_id").cast(TOKEN_DTYPE))
    )

    pretrain_df = (
        pl.concat(
            [
                users_train.select([user_col, pl.lit(USER_START).cast(TOKEN_DTYPE).alias("token_id")]),
                pretrain_stream,
            ]
        )
        .group_by(user_col, maintain_order=True)
        .agg(pl.col("token_id"))
        .sample(fraction=1, shuffle=True, seed=seed)
    )

    # targets: exactly one positive per user
    targets_time = (
        test.filter(pl.col(target_col).cast(pl.Int8) == 1)
        .select([user_col, pl.col(time_col).alias("target_time"), pl.col(item_col).alias("target_item_id")])
    )
    if targets_time.select(user_col).n_unique() != targets_time.height:
        raise ValueError("expected exactly one target per user")

    users_test = targets_time.select(user_col).unique()

    prefix = (
        pl.concat(
            [
                train.select([user_col, item_col, pl.col(time_col).alias(time_col)]),
                test.filter(pl.col(target_col).cast(pl.Int8) == 0).select([user_col, item_col, pl.col(time_col).alias(time_col)]),
            ]
        )
        .join(users_test, on=user_col, how="semi")
        .join(targets_time.select([user_col, "target_time"]), on=user_col, how="left")
        .filter(pl.col(time_col) < pl.col("target_time"))
        .sort([user_col, time_col])
        .drop("target_time")
    )

    if budget_mode == "events":
        test_hist = tail_events(prefix, test_events_budget)
        test_hist_sids = test_hist.join(
            processed_semantic_ids.select(["item_id", "sid"]), left_on=item_col, right_on="item_id", how="left"
        )
    else:
        test_hist_sids = tail_tokens(prefix, test_tokens_budget)
        test_hist = test_hist_sids

    test_stream = (
        test_hist_sids.explode("sid")
        .select([user_col, pl.col("sid").alias("token_id")])
        .with_columns(pl.col("token_id").cast(TOKEN_DTYPE))
    )

    test_df = (
        pl.concat(
            [
                users_test.select([user_col, pl.lit(USER_START).cast(TOKEN_DTYPE).alias("token_id")]),
                test_stream,
            ]
        )
        .group_by(user_col, maintain_order=True)
        .agg(pl.col("token_id"))
        .sort(pl.col("token_id").list.len(), descending=True)
    )

    targets_path = os.path.join(cfg["data_dir"], require(cfg, "test_interactions_path"))
    targets_raw = pl.read_parquet(targets_path)

    need_targets = {user_col, item_col, target_col}
    if not need_targets.issubset(targets_raw.columns):
        raise ValueError(f"targets parquet must contain columns: {sorted(need_targets)}")

    targets_df = (
        targets_raw
        .filter(pl.col(target_col).cast(pl.Int8) == 1)
        .select([pl.col(user_col).alias("user_id"), pl.col(item_col).alias("item_id")])
        .with_columns(item_id=pl.concat_list([pl.col("item_id")]))
    )


    pretrain_avg_len_tokens = _mean_scalar(pretrain_df.select(pl.col("token_id").list.len().mean()))
    test_avg_len_tokens = _mean_scalar(test_df.select(pl.col("token_id").list.len().mean()))

    train_avg_len_events = _mean_scalar(train_hist.group_by(user_col).len().select(pl.col("len").mean()))
    test_avg_len_events = _mean_scalar(test_hist.group_by(user_col).len().select(pl.col("len").mean()))

    stats = {
        "data_dir": data_dir,
        "train_path": train_path,
        "test_path": test_path,
        "semantic_ids_path": sids_path,
        "user_col": user_col,
        "item_col": item_col,
        "time_col": time_col,
        "target_col": target_col,
        "min_item_count": min_item_count,
        "pop_k": pop_k,
        "budget_mode": budget_mode,
        "history_budget_total": total_budget,
        "train_history_budget": (total_budget if budget_mode == "events" else (total_budget - 1)),
        "test_history_budget": ((total_budget - 1) if budget_mode == "events" else (total_budget - 1 - sid_stream_len_max)),
        "append_unique_token": append_unique_token,
        "flatten_codebooks": flatten_codebooks,
        "user_start_token": USER_START,
        "sep_token": SEP_TOKEN,
        "num_special_tokens": NUM_SPECIAL_TOKENS,
        "token_offset": NUM_SPECIAL_TOKENS,
        "unique_token_offset": unique_token_offset_saved,
        "sid_len_max": sid_len_max,
        "sid_stream_len_max": sid_stream_len_max,
        "vocab_size": int(vocab_size),
        "n_train_interactions_after_core": int(train.height),
        "n_test_interactions_after_core": int(test.height),
        "n_users_pretrain": int(pretrain_df.height),
        "n_users_test": int(test_df.height),
        "pretrain_max_len_observed": int(pretrain_df["token_id"].list.len().max()) if pretrain_df.height else 0,
        "test_max_len_observed": int(test_df["token_id"].list.len().max()) if test_df.height else 0,
        "seed": seed,
        "token_dtype": "UInt32",
        "pretrain_avg_len_tokens": pretrain_avg_len_tokens,
        "test_avg_len_tokens": test_avg_len_tokens,
        "train_avg_len_events": train_avg_len_events,
        "test_avg_len_events": test_avg_len_events,
    }

    pretrain_df = pretrain_df.rename({user_col: "uid"})
    test_df = test_df.rename({user_col: "uid"})

    return (
        pretrain_df, 
        test_df.select(["uid", "token_id"]), 
        processed_semantic_ids, 
        head_items_df, 
        targets_df, 
        stats
    )


def main(cfg):
    configure_torch()

    # 1) build dfs + stats in-memory
    pretrain_df, test_df, processed_semantic_ids, head_items_df, targets_df, stats = build_dataframes_and_stats(cfg)

    # 2) train
    graph = run_train(cfg, stats, pretrain_df)

    # 3) eval
    metrics = run_eval(
        cfg,
        stats,
        graph,
        processed_semantic_ids=processed_semantic_ids,
        test_df=test_df.rename({"uid": "uid"}),  # keep same
        targets_df=targets_df,
        head_items_df=head_items_df,
    )

    result = {
        "metrics": metrics,
        "stats": stats,
        "cfg": cfg
    }

    summary_path = cfg["summary_json"]
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        print("Saved summary:", summary_path)

    print("Done. Metrics keys:", sorted(metrics.keys()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
