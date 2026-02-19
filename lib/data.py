from dataclasses import dataclass
from typing import Iterator, Optional, Union

import numpy as np
import polars as pl

import torch


@dataclass
class EventsBatch:
    tokens: torch.Tensor
    embeddings: Optional[torch.Tensor]
    size: int


@dataclass
class EmbeddingsBatch:
    embeddings: torch.Tensor
    weights: torch.Tensor
    size: int


class EventsDataset:
    def __init__(
            self,
            df: Union[pl.DataFrame, pl.LazyFrame],
            batch_size: int,
            token_col: str = "token_id",
            device: str = "cuda",
            slice_rows: int = 262_144,
            pin_memory: bool = True,
            shuffle: bool = False,
            seed: int = 42,
            drop_last: bool = True,
            embedding_table: Optional[torch.Tensor] = None,  # [V, D]
            move_embedding_table_to_device: bool = True,
    ):
        assert batch_size > 0
        self.batch_size = int(batch_size)
        self.token_col = token_col
        self.device = device
        self.slice_rows = int(slice_rows)
        self.pin_memory = bool(pin_memory)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.df = df.collect() if isinstance(df, pl.LazyFrame) else df
        if self.token_col not in self.df.columns:
            raise KeyError(f"Column '{self.token_col}' not found; df.columns={self.df.columns}")
        if self.df[self.token_col].null_count() != 0:
            raise ValueError(f"Column '{self.token_col}' contains nulls")

        self.num_rows = int(self.df.height)

        # Optional embedding table
        if embedding_table is None:
            self.embedding_table = None
        else:
            if embedding_table.ndim != 2:
                raise ValueError(f"embedding_table must be 2D [V, D], got shape={tuple(embedding_table.shape)}")
            self.embedding_table = (
                embedding_table.to(self.device, non_blocking=True)
                if move_embedding_table_to_device and str(self.device) != "cpu"
                else embedding_table
            )

    @property
    def total_num_tokens(self):
        return self.num_rows
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.num_rows // self.batch_size
        return (self.num_rows + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[EventsBatch]:
        df = self.df
        if self.shuffle:
            df = df.sample(fraction=1, shuffle=True, seed=self.seed)

        # double-buffer pinned CPU
        cpu0 = torch.empty(self.batch_size, dtype=torch.long, pin_memory=self.pin_memory)
        cpu1 = torch.empty(self.batch_size, dtype=torch.long, pin_memory=self.pin_memory)
        np0 = cpu0.numpy()
        np1 = cpu1.numpy()

        buf_idx = 0
        buf = np0
        fill = 0

        emb_table = self.embedding_table

        for sl in df.iter_slices(self.slice_rows):
            vals = sl.get_column(self.token_col).to_numpy()
            if vals.dtype != np.int64:
                vals = vals.astype(np.int64, copy=False)

            i = 0
            n = int(vals.shape[0])
            while i < n:
                take = min(self.batch_size - fill, n - i)
                buf[fill:fill + take] = vals[i:i + take]
                fill += take
                i += take

                if fill == self.batch_size:
                    cpu = cpu0 if buf_idx == 0 else cpu1
                    tokens = cpu.to(self.device, non_blocking=True)

                    embeddings = None
                    if emb_table is not None:
                        embeddings = emb_table.index_select(0, tokens)

                    yield EventsBatch(tokens=tokens, embeddings=embeddings, size=self.batch_size)

                    buf_idx ^= 1
                    buf = np0 if buf_idx == 0 else np1
                    fill = 0

        if not self.drop_last and fill:
            tail_cpu = torch.as_tensor(buf[:fill].copy(), dtype=torch.long)
            if self.pin_memory:
                tail_cpu = tail_cpu.pin_memory()
            tokens = tail_cpu.to(self.device, non_blocking=True)

            embeddings = None
            if emb_table is not None:
                embeddings = emb_table.index_select(0, tokens)

            yield EventsBatch(tokens=tokens, embeddings=embeddings, size=int(fill))


class EmbeddingsDataset:
    def __init__(
            self,
            embeddings: torch.Tensor,          # [N, D]
            batch_size: int,
            device: Union[str, torch.device] = "cuda",
            move_to_device: bool = True,
            drop_last: bool = True,
            weights: torch.Tensor = None
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [N, D], got {tuple(embeddings.shape)}")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.embeddings = embeddings
        self.raw_weights = weights
        if weights is None:
            self.raw_weights = torch.ones(embeddings.size(0), device=embeddings.device, dtype=torch.float32)
        self.weights = self.raw_weights.float() / self.raw_weights.float().mean()
        assert self.embeddings.size(0) == self.weights.size(0)

        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.move_to_device = bool(move_to_device)
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = self.embeddings.size(0)
        b = self.batch_size
        return n // b if self.drop_last else (n + b - 1) // b

    def __iter__(self) -> Iterator[torch.Tensor]:
        E = self.embeddings
        B = self.batch_size
        N = E.size(0)

        for start in range(0, N, B):
            end = min(start + B, N)
            cur = end - start
            if self.drop_last and cur < B:
                break

            batch = E[start:end]
            if self.move_to_device and batch.device != self.device:
                batch = batch.to(self.device)

            weights = self.weights[start:end]
            if self.move_to_device and weights.device != self.device:
                weights = weights.to(self.device)

            yield EmbeddingsBatch(embeddings=batch, weights=weights, size=cur)

