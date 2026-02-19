import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from lib.utils import df_embeddings_to_torch, encode_all


def _df_popularity_to_numpy(df: pl.DataFrame) -> np.ndarray:
    # Popularity for correlation & unigram weighting
    if "train_count" in df.columns:
        pop = df["train_count"].to_numpy()
    elif "test_count" in df.columns:
        pop = df["test_count"].to_numpy()
    elif "raw_weight" in df.columns:
        pop = df["raw_weight"].to_numpy()
    else:
        pop = np.ones((df.height,), dtype=np.float64)
    pop = pop.astype(np.float64, copy=False)
    return pop


def _make_unigram_weights(pop: np.ndarray) -> torch.Tensor:
    # stable scale: mean weight ~ 1
    w = pop.copy()
    w[w < 0] = 0.0
    mean = w.mean() if w.size > 0 else 1.0
    if mean <= 0:
        w[:] = 1.0
        mean = 1.0
    w = w / mean
    return torch.from_numpy(w.astype(np.float32, copy=False))


@dataclass
class MeanMetric:
    num: torch.Tensor
    den: torch.Tensor

    @staticmethod
    def create(device="cpu", dtype=torch.float64) -> "MeanMetric":
        return MeanMetric(
            num=torch.zeros((), dtype=dtype, device=device),
            den=torch.zeros((), dtype=dtype, device=device),
        )

    def update_sum(self, val_sum: torch.Tensor, weight_sum: torch.Tensor):
        self.num += val_sum.to(self.num.device)
        self.den += weight_sum.to(self.den.device)

    def value(self) -> float:
        return float((self.num / (self.den + 1e-12)).item())


def _mse_on_unit_vectors_sumD(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a, b: [B, D] float32/float16 on CUDA
    Returns per-item loss: [B] float32
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    # per-item MSE summed over D
    return F.mse_loss(a, b, reduction="none").sum(dim=-1)


@torch.inference_mode()
def _recon_metrics_streaming(
    decoder,
    codes_cpu: torch.Tensor,                 # [N, T] on CPU
    emb_cpu: torch.Tensor,                   # [N, D] on CPU
    weights_unigram_cpu: torch.Tensor,       # [N] on CPU float32 (mean ~ 1)
    lengths_cpu: Optional[torch.Tensor],     # [N] on CPU or None
    batch_size: int,
    device: str = "cuda",
    per_step: bool = True,
    head_mask_cpu: Optional[torch.Tensor] = None,  # [N] bool on CPU
) -> Dict[str, Any]:
    """
    Recon computed on GPU per batch.
    If head_mask_cpu is provided and contains BOTH groups, also computes
    recon_head_longtail with only @last (+ @varlen if present) WITHOUT extra decoder pass.
    """
    decoder.eval()

    N = codes_cpu.size(0)
    T = codes_cpu.size(1)

    # which steps
    steps = list(range(T)) if per_step else [T - 1]
    last_t = T - 1

    # accumulators: sums for each scheme/step
    acc = {
        "uniform": {t: {"num": 0.0, "den": 0.0} for t in steps},
        "unigram": {t: {"num": 0.0, "den": 0.0} for t in steps},
    }
    if lengths_cpu is not None:
        acc["uniform"]["varlen"] = {"num": 0.0, "den": 0.0}
        acc["unigram"]["varlen"] = {"num": 0.0, "den": 0.0}

    # --- optional: head/longtail accumulators (@last + @varlen only)
    do_hl = False
    hl_acc = None
    if head_mask_cpu is not None:
        if head_mask_cpu.dtype != torch.bool:
            head_mask_cpu = head_mask_cpu.to(torch.bool)
        n_head = int(head_mask_cpu.sum().item())
        n_tail = int(head_mask_cpu.numel() - n_head)
        do_hl = (n_head > 0) and (n_tail > 0)
        if do_hl:
            hl_acc = {
                "head": {
                    "uniform": {"@last": {"num": 0.0, "den": 0.0}},
                    "unigram": {"@last": {"num": 0.0, "den": 0.0}},
                },
                "longtail": {
                    "uniform": {"@last": {"num": 0.0, "den": 0.0}},
                    "unigram": {"@last": {"num": 0.0, "den": 0.0}},
                },
            }
            if lengths_cpu is not None:
                for g in ("head", "longtail"):
                    hl_acc[g]["uniform"]["@varlen"] = {"num": 0.0, "den": 0.0}
                    hl_acc[g]["unigram"]["@varlen"] = {"num": 0.0, "den": 0.0}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cur = end - start

        codes = codes_cpu[start:end].to(device, non_blocking=True)
        emb = emb_cpu[start:end].to(device, non_blocking=True).to(torch.float32)

        w_uni = torch.ones((cur,), device=device, dtype=torch.float32)
        w_w = weights_unigram_cpu[start:end].to(device, non_blocking=True).to(torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            recon_seq = decoder(codes)  # [B, T, D] bf16

        recon_seq = recon_seq.to(torch.float32)

        for t in steps:
            loss_t = _mse_on_unit_vectors_sumD(emb, recon_seq[:, t])  # [B]
            acc["uniform"][t]["num"] += float((loss_t * w_uni).sum().item())
            acc["uniform"][t]["den"] += float(w_uni.sum().item())
            acc["unigram"][t]["num"] += float((loss_t * w_w).sum().item())
            acc["unigram"][t]["den"] += float(w_w.sum().item())

        if lengths_cpu is not None:
            L = lengths_cpu[start:end].to(device, non_blocking=True).to(torch.int64)  # [B] in {1..T}
            idx = torch.arange(cur, device=device)
            loss_var = _mse_on_unit_vectors_sumD(emb, recon_seq[idx, L - 1])  # [B]
            acc["uniform"]["varlen"]["num"] += float((loss_var * w_uni).sum().item())
            acc["uniform"]["varlen"]["den"] += float(w_uni.sum().item())
            acc["unigram"]["varlen"]["num"] += float((loss_var * w_w).sum().item())
            acc["unigram"]["varlen"]["den"] += float(w_w.sum().item())

        # --- head/longtail recon (@last + @varlen only)
        if do_hl:
            m = head_mask_cpu[start:end]  # [cur] bool CPU
            m = m.to(device, non_blocking=True)  # [cur] bool CUDA

            # @last
            loss_last = _mse_on_unit_vectors_sumD(emb, recon_seq[:, last_t])  # [cur]

            # head
            if bool(m.any()):
                lh = loss_last[m]
                wh_uni = w_uni[m]
                wh_w = w_w[m]
                hl_acc["head"]["uniform"]["@last"]["num"] += float((lh * wh_uni).sum().item())
                hl_acc["head"]["uniform"]["@last"]["den"] += float(wh_uni.sum().item())
                hl_acc["head"]["unigram"]["@last"]["num"] += float((lh * wh_w).sum().item())
                hl_acc["head"]["unigram"]["@last"]["den"] += float(wh_w.sum().item())

            # longtail
            mt = ~m
            if bool(mt.any()):
                lt = loss_last[mt]
                wt_uni = w_uni[mt]
                wt_w = w_w[mt]
                hl_acc["longtail"]["uniform"]["@last"]["num"] += float((lt * wt_uni).sum().item())
                hl_acc["longtail"]["uniform"]["@last"]["den"] += float(wt_uni.sum().item())
                hl_acc["longtail"]["unigram"]["@last"]["num"] += float((lt * wt_w).sum().item())
                hl_acc["longtail"]["unigram"]["@last"]["den"] += float(wt_w.sum().item())

            # @varlen
            if lengths_cpu is not None:
                L = lengths_cpu[start:end].to(device, non_blocking=True).to(torch.int64)
                idx = torch.arange(cur, device=device)
                loss_var = _mse_on_unit_vectors_sumD(emb, recon_seq[idx, L - 1])

                if bool(m.any()):
                    lv_h = loss_var[m]
                    wh_uni = w_uni[m]
                    wh_w = w_w[m]
                    hl_acc["head"]["uniform"]["@varlen"]["num"] += float((lv_h * wh_uni).sum().item())
                    hl_acc["head"]["uniform"]["@varlen"]["den"] += float(wh_uni.sum().item())
                    hl_acc["head"]["unigram"]["@varlen"]["num"] += float((lv_h * wh_w).sum().item())
                    hl_acc["head"]["unigram"]["@varlen"]["den"] += float(wh_w.sum().item())

                if bool(mt.any()):
                    lv_t = loss_var[mt]
                    wt_uni = w_uni[mt]
                    wt_w = w_w[mt]
                    hl_acc["longtail"]["uniform"]["@varlen"]["num"] += float((lv_t * wt_uni).sum().item())
                    hl_acc["longtail"]["uniform"]["@varlen"]["den"] += float(wt_uni.sum().item())
                    hl_acc["longtail"]["unigram"]["@varlen"]["num"] += float((lv_t * wt_w).sum().item())
                    hl_acc["longtail"]["unigram"]["@varlen"]["den"] += float(wt_w.sum().item())

    # --- format main output
    out: Dict[str, Any] = {"uniform": {}, "unigram": {}}
    for t in steps:
        key = f"@{t+1}" if per_step else "@last"
        out["uniform"][key] = acc["uniform"][t]["num"] / (acc["uniform"][t]["den"] + 1e-12)
        out["unigram"][key] = acc["unigram"][t]["num"] / (acc["unigram"][t]["den"] + 1e-12)

    if lengths_cpu is not None:
        out["uniform"]["@varlen"] = acc["uniform"]["varlen"]["num"] / (acc["uniform"]["varlen"]["den"] + 1e-12)
        out["unigram"]["@varlen"] = acc["unigram"]["varlen"]["num"] / (acc["unigram"]["varlen"]["den"] + 1e-12)

    # --- attach head/longtail if computed
    if do_hl and hl_acc is not None:
        def _finalize(group: str) -> Dict[str, Any]:
            res = {"uniform": {}, "unigram": {}}
            for scheme in ("uniform", "unigram"):
                for k, vv in hl_acc[group][scheme].items():
                    res[scheme][k] = vv["num"] / (vv["den"] + 1e-12)
            return res

        out["recon_head_longtail"] = {
            "head": _finalize("head"),
            "longtail": _finalize("longtail"),
        }

    return out


def _code_key_strings(
    codes_cpu: torch.Tensor,                 # [N, T]
    lengths_cpu: Optional[torch.Tensor],     # [N] or None
) -> list[str]:
    codes_np = codes_cpu.to(torch.int64).numpy()
    if lengths_cpu is None:
        return [",".join(map(str, row.tolist())) for row in codes_np]
    lens = lengths_cpu.to(torch.int64).numpy()
    keys = []
    for row, L in zip(codes_np, lens, strict=True):
        keys.append(",".join(map(str, row[: int(L)].tolist())))
    return keys


def _codebook_stats(
    codes_cpu: torch.Tensor,                 # [N, T]
    vocab_size: int,
    lengths_cpu: Optional[torch.Tensor],
    per_step: bool,
) -> Dict[str, Any]:
    # Varlen-aware: step t considers only items with length > t
    N, T = codes_cpu.shape
    codes_i64 = codes_cpu.to(torch.int64)

    # Per-step: macro averages
    ppl_sum = 0.0
    usage_sum = 0.0
    steps_used = 0

    out: Dict[str, Any] = {}

    # Micro (all tokens concatenated)
    micro_counts = torch.zeros((vocab_size,), dtype=torch.int64)
    micro_total = 0

    for t in range(T):
        if lengths_cpu is None:
            alive = None
            x = codes_i64[:, t]
        else:
            alive = (lengths_cpu > t)
            if not bool(alive.any()):
                continue
            x = codes_i64[alive, t]

        counts = torch.bincount(x, minlength=vocab_size).to(torch.float64)
        total = float(counts.sum().clamp_min(1.0).item())
        p = counts / total
        H = -(p * p.clamp_min(1e-12).log()).sum()
        ppl = float(torch.exp(H).item())

        used = int((counts > 0).sum().item())
        usage = used / float(vocab_size)

        ppl_sum += ppl
        usage_sum += usage
        steps_used += 1

        if per_step:
            out[f"perplexity@{t+1}"] = ppl
            out[f"codebook_usage@{t+1}"] = usage

        # micro update
        micro_counts += torch.bincount(x, minlength=vocab_size)
        micro_total += x.numel()

    denom = max(steps_used, 1)
    out["perplexity_macro"] = ppl_sum / denom
    out["codebook_usage_macro"] = usage_sum / denom

    # micro entropy/perplexity/usage
    if micro_total > 0:
        mc = micro_counts.to(torch.float64)
        p = mc / float(mc.sum().clamp_min(1.0).item())
        H = -(p * p.clamp_min(1e-12).log()).sum()
        out["perplexity_micro"] = float(torch.exp(H).item())
        out["codebook_usage_micro"] = float((mc > 0).sum().item()) / float(vocab_size)
        out["total_tokens_micro"] = int(micro_total)
    else:
        out["perplexity_micro"] = float("nan")
        out["codebook_usage_micro"] = float("nan")
        out["total_tokens_micro"] = 0

    return out


def _length_stats(
    lengths_cpu: torch.Tensor,                 # [N] int
    weights_unigram_cpu: torch.Tensor,         # [N] float
) -> Dict[str, Any]:
    lengths = lengths_cpu.to(torch.float64)
    w = weights_unigram_cpu.to(torch.float64)

    out = {
        "mean_uniform": float(lengths.mean().item()),
        "mean_unigram": float((lengths * w).sum().item() / (w.sum().item() + 1e-12)),
    }
    return out


def _pop_stats_by_length(
    lengths_cpu: Optional[torch.Tensor],
    popularity: np.ndarray,
) -> Dict[str, Any]:
    if lengths_cpu is None:
        return {}

    L = lengths_cpu.to(torch.int64).cpu().numpy()
    pop = np.asarray(popularity, dtype=np.float64)

    df = pl.DataFrame({"length": L, "pop": pop})
    stats = (
        df.group_by("length")
        .agg([
            pl.len().alias("count"),
            pl.col("pop").mean().alias("pop_mean"),
            pl.col("pop").median().alias("pop_median"),
            pl.col("pop").std().alias("pop_std"),
            pl.col("pop").min().alias("pop_min"),
            pl.col("pop").max().alias("pop_max"),
        ])
        .sort("length")
    )

    per_len = {}
    for row in stats.iter_rows(named=True):
        l = int(row["length"])
        per_len[str(l)] = {
            "count": int(row["count"]),
            "pop_mean": float(row["pop_mean"]),
            "pop_median": float(row["pop_median"]),
            "pop_std": float(row["pop_std"]) if row["pop_std"] is not None else float("nan"),
            "pop_min": float(row["pop_min"]),
            "pop_max": float(row["pop_max"]),
        }

    return {"pop_stats_by_length": per_len}


def _head_longtail_length_stats(
    lengths_cpu: Optional[torch.Tensor],
    head_mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    if lengths_cpu is None or head_mask is None:
        return {}

    L = lengths_cpu.to(torch.float64).numpy()
    head_mask = head_mask.astype(bool, copy=False)

    if head_mask.size != L.size:
        return {}

    def _mean_or_nan(x: np.ndarray) -> float:
        return float(x.mean()) if x.size > 0 else float("nan")

    return {
        "mean_length_head": _mean_or_nan(L[head_mask]),
        "mean_length_longtail": _mean_or_nan(L[~head_mask]),
        "count_head": int(head_mask.sum()),
        "count_longtail": int((~head_mask).sum()),
    }


def _distinctness(
    codes_cpu: torch.Tensor,
    lengths_cpu: Optional[torch.Tensor],
) -> Dict[str, Any]:
    keys = _code_key_strings(codes_cpu, lengths_cpu)
    n_items = len(keys)
    n_unique = len(set(keys))
    return {
        "n_items": int(n_items),
        "n_unique_codes": int(n_unique),
        "unique_codes_per_item": float(n_unique / max(n_items, 1)),
    }


@torch.inference_mode()
def evaluate_split(
        name: str,
        items_df: pl.DataFrame,
        encoder,
        decoder,
        batch_size: int = 512,
        device: str = "cuda",
        per_step: bool = True,
) -> Dict[str, Any]:
    # embeddings
    emb_cpu = df_embeddings_to_torch(items_df, col="embed")  # CPU [N, D]
    N = emb_cpu.size(0)

    # popularity -> unigram weights
    pop = _df_popularity_to_numpy(items_df)           # numpy float64
    w_unigram_cpu = _make_unigram_weights(pop)        # torch float32 CPU, mean~1

    # optional head mask (CPU bool), but we will compute HL metrics only if both groups exist
    head_mask_cpu: Optional[torch.Tensor] = None
    if "head" in items_df.columns:
        hm = items_df["head"].to_numpy()
        hm = hm.astype(np.bool_, copy=False)
        n_head = int(hm.sum())
        n_tail = int(hm.size - n_head)
        if (n_head > 0) and (n_tail > 0):
            head_mask_cpu = torch.from_numpy(hm)

    # encode
    codes_cpu, lengths_cpu = encode_all(encoder, emb_cpu, batch_size=batch_size, device=device)
    maxlen = int(getattr(encoder, "maxlen", codes_cpu.size(1)))
    codes_cpu = codes_cpu[:, :maxlen]

    # varlen lengths
    if getattr(encoder, "varlen", False):
        lengths_cpu = lengths_cpu.to(torch.int64)
    else:
        lengths_cpu = None

    # metrics
    out: Dict[str, Any] = {"n_items": int(N)}

    # recon (full split) + (optional) recon_head_longtail computed inside (no extra decoder pass)
    recon_out = _recon_metrics_streaming(
        decoder=decoder,
        codes_cpu=codes_cpu,
        emb_cpu=emb_cpu,
        weights_unigram_cpu=w_unigram_cpu,
        lengths_cpu=lengths_cpu,
        batch_size=batch_size,
        device=device,
        per_step=per_step,
        head_mask_cpu=head_mask_cpu,   # None -> no HL; valid mask -> HL only @last/@varlen
    )

    # if HL was computed, pop it into a separate top-level field for clarity
    if isinstance(recon_out, dict) and "recon_head_longtail" in recon_out:
        out["recon_head_longtail"] = recon_out.pop("recon_head_longtail")

    out["recon"] = recon_out

    # codebook stats
    out["codebook"] = _codebook_stats(
        codes_cpu=codes_cpu.to(torch.int64),
        vocab_size=int(getattr(encoder, "vocab_size")),
        lengths_cpu=lengths_cpu,
        per_step=per_step,
    )

    # length-related stats
    if getattr(encoder, "varlen", False) and lengths_cpu is not None:
        out["length"] = _length_stats(lengths_cpu, w_unigram_cpu)

        out["popularity_length"] = _pop_stats_by_length(
            lengths_cpu=lengths_cpu,
            popularity=pop,
        )

        # head/long-tail length stats only if both groups exist
        if head_mask_cpu is not None:
            out["head_longtail"] = _head_longtail_length_stats(
                lengths_cpu=lengths_cpu,
                head_mask=head_mask_cpu.cpu().numpy(),
            )
    else:
        out["length"] = {"fixed": int(maxlen)}

    # distinctness
    out["distinctness"] = _distinctness(
        codes_cpu=codes_cpu.to(torch.int64),
        lengths_cpu=lengths_cpu,
    )

    return {name: out}


@torch.inference_mode()
def evaluate_all(
        encoder,
        decoder,
        batch_size: int = 512,
        device: str = "cuda",
        per_step: bool = True,
        expected_holdout_frac: float = 0.01,
        # old-style explicit splits (optional)
        train_items: Optional[pl.DataFrame] = None,
        holdout_items: Optional[pl.DataFrame] = None,
        cold_items: Optional[pl.DataFrame] = None,
        # new-style: any named splits
        splits: Optional[Dict[str, pl.DataFrame]] = None,
) -> Dict[str, Any]:
    encoder.eval()
    decoder.eval()

    # pick evaluation set
    if splits is None:
        splits = {}
        if train_items is not None:
            splits["train"] = train_items
        if holdout_items is not None:
            splits["holdout"] = holdout_items
        if cold_items is not None:
            splits["cold"] = cold_items

    if not splits:
        raise ValueError("evaluate_all: no splits provided (splits is empty and all *_items are None).")

    metrics: Dict[str, Any] = {
        "meta": {
            "varlen": bool(getattr(encoder, "varlen", False)),
            "maxlen": int(getattr(encoder, "maxlen", -1)),
            "vocab_size": int(getattr(encoder, "vocab_size", -1)),
            "batch_size": int(batch_size),
            "per_step": bool(per_step),
            "expected_holdout_frac": float(expected_holdout_frac),
        }
    }

    # sanity sizes: always record per-split sizes; plus some common deltas if available
    sanity: Dict[str, Any] = {f"{name}_items": int(df.height) for name, df in splits.items()}

    if ("train" in splits) and ("holdout" in splits):
        tr = splits["train"].height
        ho = splits["holdout"].height
        sanity["holdout_frac_of_train"] = float(ho / max(tr, 1))

    if ("cold" in splits) and ("holdout" in splits):
        sanity["cold_minus_holdout"] = int(splits["cold"].height - splits["holdout"].height)

    metrics["sanity"] = sanity

    # evaluations (single loop instead of three hardcoded calls)
    for name, df in splits.items():
        metrics.update(evaluate_split(name, df, encoder, decoder, batch_size, device, per_step))

    return metrics


@torch.inference_mode()
def evaluate_split_for_tb(
    prefix: str,
    items_df: pl.DataFrame,
    encoder,
    decoder,
    *,
    device: str = "cuda",
    batch_size: int = 512,
    detail: bool = False,
    pop_lite_maxlen: int = 3,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # --- embeddings + pop weights
    emb_cpu = df_embeddings_to_torch(items_df, col="embed")  # [N,D] CPU
    pop = _df_popularity_to_numpy(items_df)
    w_unigram_cpu = _make_unigram_weights(pop)               # [N] CPU float32

    # --- encode
    codes_cpu, lengths_cpu = encode_all(encoder, emb_cpu, batch_size=batch_size, device=device)
    maxlen = int(getattr(encoder, "maxlen", codes_cpu.size(1)))
    codes_cpu = codes_cpu[:, :maxlen]
    if getattr(encoder, "varlen", False):
        lengths_cpu = lengths_cpu.to(torch.int64)

    recon = _recon_metrics_streaming(
        decoder=decoder,
        codes_cpu=codes_cpu,
        emb_cpu=emb_cpu,
        weights_unigram_cpu=w_unigram_cpu,
        lengths_cpu=lengths_cpu if getattr(encoder, "varlen", False) else None,
        batch_size=batch_size,
        device=device,
        per_step=detail,   # detail=False -> "@last"; detail=True -> "@1..@T"
    )

    # weighted == unigram, unweighted == uniform
    # lite:
    if not detail:
        v = recon.get("unigram", {}).get("@last", None)
        if v is None:
            ks = [int(k[1:]) for k in recon.get("unigram", {}) if isinstance(k,str) and k.startswith("@") and k[1:].isdigit()]
            if ks:
                v = recon["unigram"][f"@{max(ks)}"]
        if v is not None:
            out[f"recon/{prefix}_@last"] = float(v)

        if getattr(encoder, "varlen", False) and "@varlen" in recon.get("unigram", {}):
            out[f"recon/{prefix}_@varlen"] = float(recon["unigram"]["@varlen"])
    else:
        for k, v in recon.get("unigram", {}).items():
            if k.startswith("@"):
                out[f"recon/{prefix}_{k}"] = float(v)
        for k, v in recon.get("uniform", {}).items():
            if k.startswith("@"):
                out[f"recon/{prefix}_unweighted{k}"] = float(v)

    cb = _codebook_stats(
        codes_cpu=codes_cpu.to(torch.int64),
        vocab_size=int(getattr(encoder, "vocab_size")),
        lengths_cpu=lengths_cpu if getattr(encoder, "varlen", False) else None,
        per_step=detail,
    )
    out[f"codebook/{prefix}_perplexity"] = float(cb["perplexity_macro"])

    if detail:
        for k, v in cb.items():
            if k.startswith("perplexity@") or k.startswith("codebook_usage@"):
                out[f"codebook/{prefix}_{k}"] = float(v)
        out[f"codebook/{prefix}_codebook_usage"] = float(cb["codebook_usage_macro"])

    # --- msg/length
    if getattr(encoder, "varlen", False):
        ls = _length_stats(lengths_cpu, w_unigram_cpu)
        out[f"msg/{prefix}_uniform_length"] = float(ls["mean_uniform"])
        out[f"msg/{prefix}_weighted_length"] = float(ls["mean_unigram"])

        pop_stats = _pop_stats_by_length(lengths_cpu=lengths_cpu, popularity=pop).get("pop_stats_by_length", {})
        for l in range(1, pop_lite_maxlen + 1):
            row = pop_stats.get(str(l))
            if row and "pop_mean" in row:
                out[f"msg/{prefix}_avg_pop@{l}"] = float(row["pop_mean"])
            if detail and row:
                if "pop_max" in row:
                    out[f"msg/{prefix}_max_pop@{l}"] = float(row["pop_max"])
                if "count" in row:
                    out[f"msg/{prefix}_count@{l}"] = float(row["count"])
    else:
        pass

    return out


@torch.inference_mode()
def evaluate_for_tb(
        encoder,
        decoder,
        train_items: pl.DataFrame,
        holdout_items: pl.DataFrame,
        cold_items: Optional[pl.DataFrame] = None,
        device: str = "cuda",
        batch_size: int = 512,
        detail: bool = False,
) -> Dict[str, float]:
    out = {}
    out.update(evaluate_split_for_tb("train", train_items, encoder, decoder, device=device, batch_size=batch_size, detail=detail))
    out.update(evaluate_split_for_tb("holdout", holdout_items, encoder, decoder, device=device, batch_size=batch_size, detail=detail))
    if cold_items is not None:
        out.update(evaluate_split_for_tb("cold", cold_items, encoder, decoder, device=device, batch_size=batch_size, detail=detail))
    return out
