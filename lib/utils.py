import math
import os
import polars as pl
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def configure_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    os.environ.setdefault("TORCH_LOGS", "recompiles")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "1")


def df_embeddings_to_torch(df: pl.DataFrame, col: str = "embed") -> torch.Tensor:
    x = df[col].to_torch()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected embeddings [N, D], got {tuple(x.shape)}")
    return x


@torch.inference_mode()
def encode_all(
        encoder,
        emb_cpu: torch.Tensor,
        batch_size: int,
        device: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    encoder.eval()

    N = emb_cpu.size(0)
    codes_list = []
    lengths_list = [] if getattr(encoder, "varlen", False) else None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = emb_cpu[start:end].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = F.normalize(x, dim=-1)
            out = encoder(x)

        if getattr(encoder, "varlen", False):
            codes, lengths = out
            # allow logits over length
            if lengths.ndim == 2:
                lengths = lengths.argmax(dim=-1) + 1  # [B] in {1..T}
            lengths_list.append(lengths.to("cpu", non_blocking=False))
        else:
            codes = out

        codes_list.append(codes.to("cpu", non_blocking=False))

    codes_cpu = torch.cat(codes_list, dim=0)
    lengths_cpu = torch.cat(lengths_list, dim=0) if lengths_list is not None else None
    return codes_cpu, lengths_cpu


@torch.inference_mode()
def encode_to_sids_df(
        items_df: pl.DataFrame,
        encoder,
        id_col: str,
        batch_size: int,
        device: str,
) -> pl.DataFrame:
    emb_cpu = df_embeddings_to_torch(items_df, col="embed")
    codes_cpu, lengths_cpu = encode_all(encoder, emb_cpu, batch_size=batch_size, device=device)

    maxlen = int(getattr(encoder, "maxlen", codes_cpu.size(1)))
    codes_cpu = codes_cpu[:, :maxlen].to(torch.int64)

    varlen = bool(getattr(encoder, "varlen", False))
    if varlen:
        lengths_cpu = lengths_cpu.to(torch.int64)
    else:
        lengths_cpu = torch.full((codes_cpu.size(0),), maxlen, dtype=torch.int64)

    return pl.DataFrame({id_col: items_df[id_col], "sid": codes_cpu.numpy(), "length": lengths_cpu.numpy()})


def get_cosine_scheduler(step, start, end, total_steps):
    if step >= total_steps:
        return end
    progress = step / total_steps
    if end < start:
        return end + 0.5 * (start - end) * (1. + math.cos(math.pi * progress))
    else:
        return start + 0.5 * (end - start) * (1. - math.cos(math.pi * progress))
