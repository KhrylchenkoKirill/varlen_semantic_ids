from typing import List, Optional, Union
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class RKMeans:
    def __init__(
            self,
            num_levels: int,
            num_clusters: Union[int, List[int]],
            num_iters: int = 20,
            tol: float = 1e-4,
            batch_size: Optional[int] = None,
            verbose: bool = False,
    ):
        self.num_levels = num_levels
        if isinstance(num_clusters, int):
            self.clusters_per_level = [num_clusters] * num_levels
        else:
            assert len(num_clusters) == num_levels
            self.clusters_per_level = list(num_clusters)

        self.num_iters = num_iters
        self.tol = tol
        self.batch_size = batch_size
        self.verbose = verbose

        self.codebooks: Optional[List[torch.Tensor]] = None
        self.codebooks_norms: Optional[List[torch.Tensor]] = None
        self.dim: Optional[int] = None

    @torch.no_grad()
    def _kmeans(
            self,
            X: torch.Tensor,
            num_clusters: int,
    ):
        assert X.dim() == 2
        N, D = X.shape
        device = X.device

        perm = torch.randperm(N, device=device)
        centroids = X[perm[:num_clusters]].clone()  # (K, D)

        x_norm = X.pow(2).sum(dim=1, keepdim=True)  # (N, 1)

        for it in range(self.num_iters):
            # ----- E-step -----
            if self.batch_size is None:
                # full-matrix
                c_norm = centroids.pow(2).sum(dim=1)[None, :]  # (1, K)
                dists = x_norm + c_norm - 2.0 * X @ centroids.T  # (N, K)
                labels = dists.argmin(dim=1)
            else:
                labels = torch.empty(N, dtype=torch.long, device=device)
                c_norm = centroids.pow(2).sum(dim=1)[None, :]  # (1, K)

                for start in range(0, N, self.batch_size):
                    end = min(start + self.batch_size, N)
                    Xb = X[start:end]             # (B, D)
                    Xb_norm = x_norm[start:end]   # (B, 1)
                    dists = Xb_norm + c_norm - 2.0 * Xb @ centroids.T
                    labels[start:end] = dists.argmin(dim=1)

            # ----- M-step -----
            new_centroids = torch.zeros_like(centroids)  # (K, D)
            counts = torch.bincount(labels, minlength=num_clusters).to(device=device)

            new_centroids.index_add_(0, labels, X)

            empty = (counts == 0)
            if empty.any():
                num_empty = empty.sum()
                rand_idx = torch.randint(0, N, (num_empty,), device=device)
                new_centroids[empty] = X[rand_idx]
                counts = counts.clone()
                counts[empty] = 1

            new_centroids /= counts.unsqueeze(1)

            shift = (centroids - new_centroids).norm(dim=1).max()
            centroids = new_centroids

            if self.verbose:
                print(f"[kmeans] iter {it}, max shift = {shift.item():.6f}")
            if shift < self.tol:
                if self.verbose:
                    print("[kmeans] converged")
                break

        return centroids, labels

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        assert X.dim() == 2
        N, D = X.shape
        self.dim = D
        device = X.device

        residuals = X.clone()
        codebooks: List[torch.Tensor] = []
        codebooks_norms: List[torch.Tensor] = []

        for level in range(self.num_levels):
            K_l = self.clusters_per_level[level]
            if self.verbose:
                print(f"\n[RQ] Level {level+1}/{self.num_levels}, K={K_l}")

            centroids_l, labels_l = self._kmeans(residuals, K_l)
            codebooks.append(centroids_l)
            codebooks_norms.append(centroids_l.pow(2).sum(dim=1))  # (K_l,)

            residuals = residuals - centroids_l[labels_l]

        self.codebooks = codebooks
        self.codebooks_norms = codebooks_norms

        if self.verbose:
            total_params = sum(c.numel() for c in codebooks)
            print(f"[RQ] fit done. Total params in codebooks: {total_params}")

        return self

    @torch.no_grad()
    def encode(
            self, 
            X: torch.Tensor, 
            batch_size: Optional[int] = None, 
            verbose: Optional[bool] = False
    ) -> torch.Tensor:
        assert self.codebooks is not None, "Call fit() or load() first."
        assert X.dim() == 2
        N, D = X.shape
        assert D == self.dim, f"Dim mismatch: got {D}, expected {self.dim}"

        device = X.device
        L = self.num_levels
        codes = torch.empty(N, L, dtype=torch.long, device=device)

        bs = batch_size or self.batch_size
        if bs is None:
            bs = N

        for start in tqdm.tqdm(range(0, N, bs), total=N // bs, disable=not verbose):
            end = min(start + bs, N)
            Xb = X[start:end].clone()  # (B, D)
            B = Xb.shape[0]

            residuals = Xb
            for l in range(L):
                C_l = self.codebooks[l].to(device)              # (K_l, D)
                c_norm = self.codebooks_norms[l].to(device)     # (K_l,)

                x_norm = residuals.pow(2).sum(dim=1, keepdim=True)  # (B, 1)
                # d(x,c)^2 = ||x||^2 + ||c||^2 - 2 x c^T
                dists = x_norm + c_norm.view(1, -1) - 2.0 * residuals @ C_l.T  # (B, K_l)
                labels = dists.argmin(dim=1)  # (B,)

                codes[start:end, l] = labels
                residuals = residuals - C_l[labels]

        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert self.codebooks is not None, "Call fit() or load() first."
        assert codes.dim() == 2
        N, L = codes.shape
        assert L == self.num_levels

        device = codes.device
        D = self.dim
        recon = torch.zeros(N, D, device=device, dtype=self.codebooks[0].dtype)

        for l in range(L):
            C_l = self.codebooks[l].to(device)  # (K_l, D)
            recon += C_l[codes[:, l]]

        return recon

    def save(self, path: str):
        state = {
            "version": 1,
            "num_levels": self.num_levels,
            "clusters_per_level": self.clusters_per_level,
            "num_iters": self.num_iters,
            "tol": self.tol,
            "batch_size": self.batch_size,
            "dim": self.dim,
            "codebooks": self.codebooks,
            "codebooks_norms": self.codebooks_norms,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None):
        state = torch.load(path, map_location=map_location)
        rq = cls(
            num_levels=state["num_levels"],
            num_clusters=state["clusters_per_level"],
            num_iters=state["num_iters"],
            tol=state["tol"],
            batch_size=state["batch_size"],
            verbose=False,
        )
        rq.dim = state["dim"]
        rq.codebooks = state["codebooks"]
        rq.codebooks_norms = state["codebooks_norms"]
        return rq


class RKMeansEncoder(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer
        self.max_len = quantizer.num_levels
        self.vocab_size = quantizer.clusters_per_level[0]

    @property
    def maxlen(self):
        return self.max_len
    
    def forward(self, x, tau=None):
        codes = self.quantizer.encode(x.bfloat16())
        if codes.dim() != 2 or codes.size(1) != self.max_len:
            raise RuntimeError(f"Expected codes shape [B, {self.max_len}], got {tuple(codes.shape)}")

        V = self.vocab_size
        probs = F.one_hot(codes, num_classes=V).to(dtype=torch.float32)  # [B, T, V]

        logits = probs.clone()
        logits[~probs.bool()] = -float("inf")  # [B, T, V]

        return logits, probs
        
    def inference(self):
        return RKMeansInferenceEncoder(self)


class RKMeansInferenceEncoder(nn.Module):
    def __init__(self, sender: RKMeansEncoder):
        super().__init__()
        self.sender = sender

    @property
    def vocab_size(self):
        return self.sender.vocab_size

    @property
    def maxlen(self):
        return self.sender.max_len
    
    @property
    def varlen(self):
        return False
    
    def forward(self, x: torch.Tensor):
        codes = self.sender.quantizer.encode(x.bfloat16())  # [B, T]
        if codes.dim() != 2 or codes.size(1) != self.maxlen:
            raise RuntimeError(f"Expected codes shape [B, {self.maxlen}], got {tuple(codes.shape)}")
        return codes.to(torch.long)


class RKMeansDecoder(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer

        self.maxlen = quantizer.num_levels
        self.vocab_size = quantizer.clusters_per_level[0]

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        # message: [B, T, V]
        if message.dim() != 3:
            raise RuntimeError(f"Expected message [B,T,V], got {tuple(message.shape)}")

        B, T, V = message.shape
        if T != self.maxlen:
            raise RuntimeError(f"Expected T=={self.maxlen}, got {T}")
        if V != self.vocab_size:
            raise RuntimeError(f"Expected V=={self.vocab_size}, got {V}")

        # training-time: probs are one-hot -> get discrete ids
        codes = message.argmax(dim=-1)  # [B, T], long

        device = codes.device
        # We follow ResidualQuantizer.decode dtype choice (dtype of codebooks[0])
        cb0 = self.quantizer.codebooks[0].to(device)
        D = cb0.size(1)
        out = torch.empty((B, T, D), device=device, dtype=cb0.dtype)

        running = torch.zeros((B, D), device=device, dtype=cb0.dtype)
        for l in range(T):
            C_l = self.quantizer.codebooks[l].to(device)  # [V, D]
            running = running + C_l[codes[:, l]]          # [B, D]
            out[:, l] = running

        out = F.normalize(out, dim=-1)
        return out

    def inference(self):
        return RKMeansInferenceDecoder(self)
    

class RKMeansInferenceDecoder(nn.Module):
    def __init__(self, receiver: RKMeansDecoder):
        super().__init__()
        self.receiver = receiver
        self.quantizer = receiver.quantizer

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 2:
            raise RuntimeError(f"Expected codes [B,T], got {tuple(codes.shape)}")

        B, T = codes.shape
        if T != self.receiver.maxlen:
            raise RuntimeError(f"Expected T=={self.receiver.maxlen}, got {T}")

        device = codes.device
        cb0 = self.quantizer.codebooks[0].to(device)
        D = cb0.size(1)

        out = torch.empty((B, T, D), device=device, dtype=cb0.dtype)
        running = torch.zeros((B, D), device=device, dtype=cb0.dtype)

        for l in range(T):
            C_l = self.quantizer.codebooks[l].to(device)
            running = running + C_l[codes[:, l].to(torch.long)]
            out[:, l] = running

        out = F.normalize(out, dim=-1)
        return out
