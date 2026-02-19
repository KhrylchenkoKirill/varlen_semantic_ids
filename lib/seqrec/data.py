from dataclasses import dataclass

import numpy as np

import polars as pl
import pyarrow.parquet as pq

import torch


@dataclass
class SeqrecBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    size: int


class SeqrecDataset:
    COLS = ["token_id"]

    def __init__(
        self,
        data,
        batch_size,
        seq_len,
        device="cuda",
        parquet_bs=64,
    ):
        self.data = data
        self.datapath = data if isinstance(data, str) else None

        self.batch_num_tokens = batch_size * seq_len + 1
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.device = device
        self._parquet_bs = parquet_bs

        self.total_num_tokens = self._count_tokens(data)

    def _count_tokens(self, data) -> int:
        def count_lazy(lf: pl.LazyFrame) -> int:
            schema = lf.schema
            dt = schema.get("token_id")
            if dt is not None and dt == pl.List(pl.Int64) or dt is not None and isinstance(dt, pl.List):
                return lf.select(pl.col("token_id").list.len().sum()).collect().item()
            return lf.select(pl.len()).collect().item()

        if isinstance(data, str):
            lf = pl.scan_parquet(data)
            return int(count_lazy(lf))

        if isinstance(data, pl.LazyFrame):
            return int(count_lazy(data))

        if isinstance(data, pl.DataFrame):
            s = data.get_column("token_id")
            if s.dtype == pl.List(pl.Int64) or isinstance(s.dtype, pl.List):
                return int(s.list.len().sum())
            return int(data.height)

        raise TypeError(f"Unsupported data type: {type(data)}")

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def __iter__(self):
        buf_tokens = np.empty(self.batch_num_tokens, dtype=np.int64)
        fill = 0

        if self.datapath is not None:
            pf = pq.ParquetFile(self.datapath)
            for rb in pf.iter_batches(batch_size=self._parquet_bs, columns=self.COLS):
                tokens = rb.column("token_id").values.to_numpy(zero_copy_only=False).astype(np.int64)
                n = tokens.shape[0]

                i = 0
                while i < n:
                    need = self.batch_num_tokens - fill
                    take = need if (i + need) <= n else (n - i)

                    buf_tokens[fill : fill + take] = tokens[i : i + take]
                    fill += take
                    i += take

                    if fill == self.batch_num_tokens:
                        t = torch.as_tensor(buf_tokens).pin_memory().to(self.device, non_blocking=True)

                        inputs = t[:-1].view(self.batch_size, self.seq_len)
                        targets = t[1:].view(self.batch_size, self.seq_len)

                        yield SeqrecBatch(inputs=inputs, targets=targets, size=inputs.numel())
                        fill = 0

            return

        df = self.data.collect() if isinstance(self.data, pl.LazyFrame) else self.data
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame/pl.LazyFrame when not path, got {type(df)}")

        total_rows = df.height
        offset = 0

        is_list = isinstance(df.get_column("token_id").dtype, pl.List)

        while offset < total_rows:
            take_rows = self._parquet_bs if (offset + self._parquet_bs) <= total_rows else (total_rows - offset)
            chunk_df = df.slice(offset, take_rows)
            offset += take_rows

            if is_list:
                chunk_s = chunk_df.explode("token_id").get_column("token_id")
            else:
                chunk_s = chunk_df.get_column("token_id")

            tokens = chunk_s.to_numpy(zero_copy_only=False).astype(np.int64)
            n = tokens.shape[0]

            i = 0
            while i < n:
                need = self.batch_num_tokens - fill
                take = need if (i + need) <= n else (n - i)

                buf_tokens[fill : fill + take] = tokens[i : i + take]
                fill += take
                i += take

                if fill == self.batch_num_tokens:
                    t = torch.as_tensor(buf_tokens).pin_memory().to(self.device, non_blocking=True)
                    inputs = t[:-1].view(self.batch_size, self.seq_len)
                    targets = t[1:].view(self.batch_size, self.seq_len)

                    yield SeqrecBatch(inputs=inputs, targets=targets, size=inputs.numel())
                    fill = 0
