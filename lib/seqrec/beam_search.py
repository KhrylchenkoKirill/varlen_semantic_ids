from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Sequence, Set, Iterator, Any
import polars as pl
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class KVCacheFast:
    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        self.reset()

    def reset(self):
        self.k_buf = [None] * self.num_layers
        self.v_buf = [None] * self.num_layers
        self.prefix_len = 0
        self.suffix_len = 0
        self.beam_size = 1
        self.max_total_len = 0
        self.cur_pos = None

    def store_prefix(self, layer_idx: int, k: Tensor, v: Tensor, max_decode_len: int):
        # k,v: [1,nH,T0,Hd]
        self.prefix_len = int(k.size(2))
        self.max_total_len = self.prefix_len + int(max_decode_len)
        self.k_buf[layer_idx] = k
        self.v_buf[layer_idx] = v

    def init_beams(self, beam_size: int):
        B = int(beam_size)
        self.beam_size = B
        self.suffix_len = 0
        self.cur_pos = None

        for l in range(self.num_layers):
            kp = self.k_buf[l]
            vp = self.v_buf[l]
            if kp is None:
                continue
            nH, T0, Hd = int(kp.size(1)), int(kp.size(2)), int(kp.size(3))
            k_full = torch.empty((B, nH, self.max_total_len, Hd), device=kp.device, dtype=kp.dtype)
            v_full = torch.empty((B, nH, self.max_total_len, Hd), device=vp.device, dtype=vp.dtype)
            k_full[:, :, :T0, :].copy_(kp.expand(B, -1, -1, -1))
            v_full[:, :, :T0, :].copy_(vp.expand(B, -1, -1, -1))
            self.k_buf[l] = k_full
            self.v_buf[l] = v_full

    def reorder_beams(self, new_beam_idx: Tensor):
        for l in range(self.num_layers):
            kb = self.k_buf[l]
            if kb is None:
                continue
            self.k_buf[l] = kb.index_select(0, new_beam_idx)
            self.v_buf[l] = self.v_buf[l].index_select(0, new_beam_idx)
        self.beam_size = int(new_beam_idx.numel())

    def begin_token(self):
        self.cur_pos = self.prefix_len + self.suffix_len

    def end_token(self):
        self.suffix_len += 1
        self.cur_pos = None

    def append_suffix(self, layer_idx: int, k_new: Tensor, v_new: Tensor):
        # k_new,v_new: [B,nH,1,Hd]
        pos = int(self.cur_pos)
        self.k_buf[layer_idx][:, :, pos:pos + 1, :].copy_(k_new)
        self.v_buf[layer_idx][:, :, pos:pos + 1, :].copy_(v_new)

    def get_kv_slice_upto(self, layer_idx: int, pos_inclusive: int):
        T = int(pos_inclusive) + 1
        k = self.k_buf[layer_idx][:, :, :T, :]
        v = self.v_buf[layer_idx][:, :, :T, :]
        return k, v


class Yarn(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.head_dim // 4, dtype=torch.float32, device="cuda"
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device="cuda")
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.sin = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = 128 * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    # x: [B,T,nH,Hd], cos/sin: [>=T, Hd/2]
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int, causal: bool = True):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.causal = bool(causal)

        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dim = int(dim)
        self.hdim = self.num_heads * self.head_dim
        assert self.hdim == self.dim

        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim * 4))
        self.qkvo_w.label = "attn"
        with torch.no_grad():
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].uniform_(-bound, bound)
            self.qkvo_w.view(4, self.hdim, self.dim)[3].zero_()

    def _qkv(self, x: Tensor, cos_sin):
        B, T, D = x.shape
        q, k, v = F.linear(
            x,
            self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x),
        ).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)

        q, k = norm(q), norm(k)
        cos, sin = cos_sin
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        return q, k, v

    def forward(self, x: Tensor, cos_sin, scale: float):
        B, T, D = x.shape
        q, k, v = self._qkv(x, cos_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.hdim)
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y

    @torch.no_grad()
    def prefill(self, x: Tensor, cos_sin, scale: float, need_kv: bool = True):
        B, T, D = x.shape
        q, k, v = self._qkv(x, cos_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)
        y = y.transpose(1, 2).contiguous().view(B, T, self.hdim)
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        kv = (k, v) if need_kv else None
        return y, kv

    @torch.no_grad()
    def decode(self, x: Tensor, cos_sin, scale: float, kv_cache: KVCacheFast):
        B, T, D = x.shape
        assert T == 1

        q, k, v = self._qkv(x, cos_sin)  # [B,1,nH,Hd]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B,nH,1,Hd]

        kv_cache.append_suffix(self.layer_idx, k, v)
        pos = int(kv_cache.cur_pos)
        k_total, v_total = kv_cache.get_kv_slice_upto(self.layer_idx, pos_inclusive=pos)

        y = F.scaled_dot_product_attention(
            q, k_total, v_total, attn_mask=None, is_causal=False, scale=scale
        )
        y = y.transpose(1, 2).contiguous().view(B, 1, self.hdim)
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        dim = int(dim)
        hdim = 4 * dim
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        self.c_fc.label = "mlp"
        self.c_proj.label = "mlp"
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_()

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square()
        x = F.linear(x, self.c_proj.type_as(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int, causal=True, dropout=0.0):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads, layer_idx=layer_idx, causal=causal)
        self.mlp = MLP(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cos_sin, scale: float):
        x = x + self.dropout(self.attn(norm(x), cos_sin, scale))
        x = x + self.dropout(self.mlp(norm(x)))
        return x

    @torch.no_grad()
    def prefill_with_cache(self, x: Tensor, cos_sin, scale: float, kv_cache: KVCacheFast, layer_idx: int, max_decode_len: int):
        y, kv = self.attn.prefill(norm(x), cos_sin, scale, need_kv=True)
        x = x + self.dropout(y)
        if kv is not None:
            k_pref, v_pref = kv
            kv_cache.store_prefix(layer_idx, k_pref, v_pref, max_decode_len=max_decode_len)
        x = x + self.dropout(self.mlp(norm(x)))
        return x

    @torch.no_grad()
    def decode(self, x: Tensor, cos_sin, scale: float, kv_cache: KVCacheFast):
        x = x + self.dropout(self.attn.decode(norm(x), cos_sin, scale, kv_cache))
        x = x + self.dropout(self.mlp(norm(x)))
        return x


class KVCacheGPT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        max_seq_len: int,
        embeddings_cls=None,
        causal=True,
        dropout=0.0,
    ):
        super().__init__()
        head_dim = model_dim // num_heads

        self.embed = None
        if embeddings_cls is not None:
            self.embed = embeddings_cls()

        self.yarn = Yarn(head_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [Block(model_dim, head_dim, num_heads, layer_idx=i, causal=causal, dropout=dropout)
             for i in range(num_layers)]
        )
        self.L = int(num_layers)

    @torch.no_grad()
    def prefill_prefix(self, token_ids_or_x: Tensor, kv_cache: KVCacheFast, max_decode_len: int):
        if self.embed is not None:
            x = self.embed(token_ids_or_x)
        else:
            x = token_ids_or_x

        x = norm(x)
        cos_sin = (self.yarn.cos, self.yarn.sin)
        scale = self.yarn.attn_scale

        kv_cache.suffix_len = 0
        kv_cache.cur_pos = None

        for i in range(self.L):
            x = self.blocks[i].prefill_with_cache(x, cos_sin, scale, kv_cache=kv_cache, layer_idx=i, max_decode_len=max_decode_len)

        return norm(x), kv_cache

    @torch.no_grad()
    def decode(self, token_id_or_x: Tensor, kv_cache: KVCacheFast) -> Tensor:
        if self.embed is not None:
            if token_id_or_x.dim() == 1:
                token_id_or_x = token_id_or_x[:, None]
            x = self.embed(token_id_or_x)
        else:
            x = token_id_or_x
            if x.dim() == 2:
                x = x[:, None, :]

        x = norm(x)

        kv_cache.begin_token()
        pos = int(kv_cache.cur_pos)

        cos = self.yarn.cos[pos:pos + 1]
        sin = self.yarn.sin[pos:pos + 1]
        cos_sin = (cos, sin)
        scale = self.yarn.attn_scale

        for i in range(self.L):
            x = self.blocks[i].decode(x, cos_sin, scale, kv_cache)

        kv_cache.end_token()
        return norm(x)


class SemanticTrieVarLen:
    def __init__(self, sequences: Sequence[Sequence[int]], vocab_size: Optional[int] = None):
        self.sequences = [list(map(int, s)) for s in sequences]
        assert len(self.sequences) > 0

        self.max_len = max(len(s) for s in self.sequences)
        self.levels: List[Dict[Tuple[int, ...], List[int]]] = [defaultdict(list) for _ in range(self.max_len)]
        self.terminal_prefixes: List[Set[Tuple[int, ...]]] = [set() for _ in range(self.max_len + 1)]

        max_token = 0
        for seq in self.sequences:
            max_token = max(max_token, max(seq) if seq else 0)
            self.terminal_prefixes[len(seq)].add(tuple(seq))
            for d in range(len(seq)):
                prefix = tuple(seq[:d])
                tok = seq[d]
                self.levels[d][prefix].append(tok)

        for d in range(self.max_len):
            for prefix, children in self.levels[d].items():
                self.levels[d][prefix] = sorted(set(children))

        self.first_tokens: List[int] = sorted(set(seq[0] for seq in self.sequences if len(seq) > 0))
        self.vocab_size = int(max_token) + 1 if vocab_size is None else int(vocab_size)

    def is_terminal(self, prefix_tokens: Sequence[int]) -> bool:
        d = len(prefix_tokens)
        if d > self.max_len:
            return False
        return tuple(prefix_tokens) in self.terminal_prefixes[d]


class BeamSearchVarLen(nn.Module):
    def __init__(
        self,
        gpt,
        head,
        semantic_ids: Sequence[Sequence[int]],
        beam_size: int = 100,
        max_sem_len: Optional[int] = None,     # default: trie.max_len

        constrain_only_at_end: bool = False,
        end_beam_mul: int = 4,
        keep_first_token_constraint: bool = True,
    ):
        super().__init__()
        self.gpt = gpt
        self.head = head

        self.trie = SemanticTrieVarLen(semantic_ids)
        self.vocab_size = int(head.out_features)

        self.beam_size = int(beam_size)
        self.max_sem_len = int(max_sem_len or self.trie.max_len)

        self.constrain_only_at_end = bool(constrain_only_at_end)
        self.end_beam_mul = int(end_beam_mul)
        self.keep_first_token_constraint = bool(keep_first_token_constraint)

        self.search_beam = self.beam_size * self.end_beam_mul if self.constrain_only_at_end else self.beam_size

        allowed_first = self.trie.first_tokens
        self.first_mask = torch.zeros((self.vocab_size,), dtype=torch.bool, device="cuda")
        self.first_mask[allowed_first] = True

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor):
        """
        tokens:
          - [T0] -> returns List[List[int]] (codes)
          - [U,T0] (same T0 inside batch) -> returns List[U][<=beam_size][code]
        """
        if tokens.dim() == 1:
            codes = self._beam_search_with_cache(tokens[None, :])  # [1,T0]
            return codes[0]
        elif tokens.dim() == 2:
            return self._beam_search_with_cache(tokens)            # [U,T0]
        else:
            raise ValueError("tokens must be [T0] or [U,T0]")

    def _beam_search_with_cache(self, tokens_UT: torch.Tensor) -> List[List[List[int]]]:
        device = tokens_UT.device
        U, T0 = tokens_UT.shape

        kv_cache = KVCacheFast(num_layers=self.gpt.L)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            hidden_pref, kv_cache = self.gpt.prefill_prefix(tokens_UT, kv_cache, max_decode_len=self.max_sem_len)
            last_hidden_pref = hidden_pref[:, -1]                 # [U,D]
            logits0 = self.head(last_hidden_pref)                 # [U,V]
            logits0 = 30.0 * torch.sigmoid(logits0 / 7.5)

        if self.keep_first_token_constraint:
            logits0 = logits0.masked_fill(~self.first_mask[None, :], float("-inf"))

        logprobs0 = logits0.log_softmax(dim=-1)                   # [U,V]
        topk_scores, topk_tokens = torch.topk(logprobs0, k=self.search_beam, dim=-1)  # [U,B]

        B = self.search_beam
        G = U * B

        # sem tokens per user/beam
        sem_tokens = torch.full((U, B, self.max_sem_len), -1, dtype=torch.long, device=device)
        sem_tokens[:, :, 0] = topk_tokens
        beam_scores = topk_scores.to(torch.float32)               # [U,B]

        # step scores for end-only selection
        step_scores = torch.full((U, B, self.max_sem_len), float("-inf"), device=device, dtype=torch.float32)
        step_scores[:, :, 0] = beam_scores

        # expand KV from batch U -> beams U*B
        kv_cache = _kv_expand_prefill_batch_to_beams(kv_cache, U=U, B=B)

        # decode first semantic token
        tok0_flat = topk_tokens.reshape(G)                        # [G]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            hidden0 = self.gpt.decode(tok0_flat, kv_cache)         # [G,1,D]
            last_hidden = hidden0[:, -1]                           # [G,D]

        # memoization for terminal checks (used in constrained mode + finalization)
        term_cache: Dict[Tuple[int, ...], bool] = {}
        def is_term(code: Tuple[int, ...]) -> bool:
            v = term_cache.get(code)
            if v is None:
                v = self.trie.is_terminal(code)
                term_cache[code] = v
            return v

        # finished lists per user (only used in constrained-throughout mode)
        finished_codes: List[List[List[int]]] = [[] for _ in range(U)]
        finished_scores: List[List[float]] = [[] for _ in range(U)]

        for step in range(1, self.max_sem_len):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = self.head(last_hidden)                   # [G,V]
                logits = 30.0 * torch.sigmoid(logits / 7.5)
            logprobs = logits.log_softmax(dim=-1)                 # [G,V]
            logprobs_uv = logprobs.view(U, B, self.vocab_size)     # [U,B,V]

            if not self.constrain_only_at_end:
                # build mask per user/beam
                mask = torch.zeros((U, B, self.vocab_size), dtype=torch.bool, device=device)

                for u in range(U):
                    for b in range(B):
                        pref = tuple(int(x) for x in sem_tokens[u, b, :step].tolist())
                        if is_term(pref):
                            continue
                        if step >= self.trie.max_len:
                            continue
                        kids = self.trie.levels[step].get(pref, [])
                        if kids:
                            mask[u, b, kids] = True

                logprobs_uv = logprobs_uv.masked_fill(~mask, float("-inf"))

            # expand+topk per user (keep B beams)
            cand = beam_scores[:, :, None] + logprobs_uv           # [U,B,V]
            flat = cand.view(U, -1)                                # [U, B*V]
            top_scores, top_idx = torch.topk(flat, k=B, dim=-1)    # [U,B]
            parent = top_idx // self.vocab_size                    # [U,B]
            token  = top_idx %  self.vocab_size                    # [U,B]

            # update beams
            sem_tokens = sem_tokens.gather(1, parent[:, :, None].expand(-1, -1, self.max_sem_len))
            sem_tokens[:, :, step] = token
            beam_scores = top_scores
            step_scores[:, :, step] = beam_scores

            # reorder KV cache to selected parents within each user
            new_global = (torch.arange(U, device=device)[:, None] * B + parent).reshape(-1)  # [G]
            kv_cache.reorder_beams(new_global)

            # decode appended tokens
            tok_flat = token.reshape(-1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                hidden = self.gpt.decode(tok_flat, kv_cache)
                last_hidden = hidden[:, -1]

            if not self.constrain_only_at_end:
                # move terminal beams to finished (per user), but keep them in beam set (we'll just let score handle it)
                for u in range(U):
                    # if already have enough finished, we could early stop per user
                    for b in range(B):
                        code = tuple(int(x) for x in sem_tokens[u, b, :step+1].tolist())
                        if is_term(code):
                            finished_codes[u].append(list(code))
                            finished_scores[u].append(float(beam_scores[u, b].item()))

        out: List[List[List[int]]] = []
        max_t = self.max_sem_len

        if self.constrain_only_at_end:
            # pick best terminal prefixes across all beams & all t
            for u in range(U):
                cand_codes: List[List[int]] = []
                cand_scores: List[float] = []
                for b in range(B):
                    for t in range(1, max_t + 1):
                        code = tuple(int(x) for x in sem_tokens[u, b, :t].tolist())
                        if is_term(code):
                            cand_codes.append(list(code))
                            cand_scores.append(float(step_scores[u, b, t-1].item()))

                if len(cand_codes) == 0:
                    # fallback: return best raw beams
                    order = torch.argsort(beam_scores[u], descending=True)
                    out.append([sem_tokens[u, i, :max_t].tolist() for i in order[: self.beam_size].tolist()])
                else:
                    scores_t = torch.tensor(cand_scores, device=device)
                    k = min(self.beam_size, int(scores_t.numel()))
                    top = torch.topk(scores_t, k=k).indices.tolist()
                    out.append([cand_codes[i] for i in top])
            return out

        # constrained-throughout mode: use finished if any, else fallback to best beams
        for u in range(U):
            if len(finished_codes[u]) > 0:
                # sort by score desc (keep top beam_size)
                scores_t = torch.tensor(finished_scores[u], device=device)
                k = min(self.beam_size, int(scores_t.numel()))
                top = torch.topk(scores_t, k=k).indices.tolist()
                out.append([finished_codes[u][i] for i in top])
            else:
                order = torch.argsort(beam_scores[u], descending=True)
                out.append([sem_tokens[u, i, :max_t].tolist() for i in order[: self.beam_size].tolist()])

        return out


def _kv_expand_prefill_batch_to_beams(kv_cache: "KVCacheFast", U: int, B: int):
    """
    Assumes after prefill, kv_cache.k_buf[l]/v_buf[l] are prefix tensors [U,nH,T0,Hd] (not expanded yet).
    Expands them to full buffers [U*B, nH, max_total_len, Hd] by repeat_interleave.
    """
    G = U * B
    for l in range(kv_cache.num_layers):
        kp = kv_cache.k_buf[l]
        vp = kv_cache.v_buf[l]
        if kp is None:
            continue

        # if already expanded, just ensure correct shape
        if kp.dim() == 4 and int(kp.size(0)) == G and int(kp.size(2)) == kv_cache.max_total_len:
            continue

        # expected raw prefix: [U,nH,T0,Hd]
        if not (kp.dim() == 4 and int(kp.size(0)) == U and int(kp.size(2)) == kv_cache.prefix_len):
            raise RuntimeError(f"Unexpected KV prefix shape for layer {l}: {tuple(kp.shape)}")

        nH, T0, Hd = int(kp.size(1)), int(kp.size(2)), int(kp.size(3))
        k_full = torch.empty((G, nH, kv_cache.max_total_len, Hd), device=kp.device, dtype=kp.dtype)
        v_full = torch.empty((G, nH, kv_cache.max_total_len, Hd), device=vp.device, dtype=vp.dtype)
        k_full[:, :, :T0, :].copy_(kp.repeat_interleave(B, dim=0))
        v_full[:, :, :T0, :].copy_(vp.repeat_interleave(B, dim=0))

        kv_cache.k_buf[l] = k_full
        kv_cache.v_buf[l] = v_full

    kv_cache.beam_size = G
    kv_cache.suffix_len = 0
    kv_cache.cur_pos = None
    return kv_cache


def iter_length_buckets(
    test_df: pl.DataFrame,
    uid_col: str = "uid",
    token_col: str = "token_id",
    max_batch_tokens: int = 8192,
    max_batch_users: int = 256,
) -> Iterator[Tuple[List[int], List[List[int]]]]:
    """
    test_df must be sorted by length(token_col).
    Yields (uids, token_lists) where all token_lists have same len.
    """
    cur_uids: List[int] = []
    cur_tokens: List[List[int]] = []
    cur_len: int = -1
    cur_tok_budget: int = 0

    for uid, toks in test_df.select([uid_col, token_col]).iter_rows():
        toks = list(toks)
        L = len(toks)

        if cur_len == -1:
            cur_len = L

        # if length changes or budget exceeded -> flush
        next_budget = cur_tok_budget + L
        if (L != cur_len) or (len(cur_uids) >= max_batch_users) or (next_budget > max_batch_tokens and len(cur_uids) > 0):
            yield cur_uids, cur_tokens
            cur_uids = []
            cur_tokens = []
            cur_len = L
            cur_tok_budget = 0

        cur_uids.append(int(uid))
        cur_tokens.append(toks)
        cur_tok_budget += L

    if len(cur_uids) > 0:
        yield cur_uids, cur_tokens
