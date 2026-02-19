import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from lib.layers import Encoder


def find_lengths_eos(message_ids: torch.Tensor) -> torch.Tensor:
    """
    message_ids: [B, T] long, EOS token is 0
    returns lengths: [B] long in 1..T
    length = first EOS position + 1, or T if EOS never appears
    """
    B, T = message_ids.shape
    eos = (message_ids == 0)
    has_eos = eos.any(dim=1)
    first = eos.float().argmax(dim=1)  # 0 if no eos
    lengths = torch.where(has_eos, first + 1, torch.full_like(first, T))
    return lengths

    
class ReinforceEncoder(Encoder):
    """
    Fixed:
      - samples tokens for all steps, lengths=T

    Varlen:
      - EOS is token 0 in the same categorical at each step
      - after EOS: force future tokens to 0 and zero-out logp/entropy
      - lengths = first EOS + 1 (or T if no EOS)
      - also force ids to 0 from (length-1) onward for a clean receiver input
    """
    def __init__(self, *args, varlen: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.varlen = bool(varlen)

    def _residual_update(self, h: torch.Tensor, step: int, tok: torch.Tensor, done: torch.Tensor | None):
        if step >= self.maxlen - 1:
            return h
        C_b = self._get_codebook(step)               # [V,H]
        codeword_b = C_b.index_select(0, tok)        # [B,H]
        delta = self.gamma[step].exp() * codeword_b  # [B,H]
        if done is None:
            h = h - delta
        else:
            h = torch.where(done[:, None], h, h - delta)
        h = self.rmsnorm(h)
        return h

    def forward(self, x: torch.Tensor):
        """
        Returns:
          ids:     [B,T] long
          msg:     [B,T,V] float one-hot
          logp:    [B,T] float (0 after EOS if varlen)
          entropy: [B,T] float (0 after EOS if varlen)
          lengths: [B] long in 1..T
        """
        B = x.size(0)
        T = self.maxlen
        V = self.vocab_size
        device = x.device

        h = self.backbone(self.dropout(x))

        ids = torch.empty((B, T), device=device, dtype=torch.long)
        logp = torch.empty((B, T), device=device, dtype=torch.float32)
        ent = torch.empty((B, T), device=device, dtype=torch.float32)

        done = torch.zeros((B,), device=device, dtype=torch.bool) if self.varlen else None

        for step in range(T):
            step_logits = self._get_logits(h, step)  # [B,V]
            distr = Categorical(logits=step_logits)

            tok = distr.sample()
            if self.varlen:
                tok = torch.where(done, torch.zeros_like(tok), tok)

            ids[:, step] = tok

            if self.varlen:
                logp[:, step] = torch.where(done, torch.zeros_like(logp[:, step]), distr.log_prob(tok))
                ent[:, step] = torch.where(done, torch.zeros_like(ent[:, step]), distr.entropy())
                done = done | (tok == 0)
            else:
                logp[:, step] = distr.log_prob(tok)
                ent[:, step] = distr.entropy()

            h = self._residual_update(h, step, tok, done=done)

        if self.varlen:
            lengths = find_lengths_eos(ids)
            t = torch.arange(T, device=device)[None, :]
            after = (t >= (lengths - 1)[:, None])
            ids = torch.where(after, torch.zeros_like(ids), ids)
        else:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        msg = F.one_hot(ids, num_classes=V).to(torch.float32)
        return ids, msg, logp, ent, lengths

    def inference(self):
        return InferenceReinforceEncoder(self)


class InferenceReinforceEncoder(nn.Module):
    def __init__(self, sender: ReinforceEncoder):
        super().__init__()
        self.sender = sender

    @property
    def vocab_size(self):
        return self.sender.vocab_size
    
    @property
    def maxlen(self):
        return self.sender.maxlen
    
    @property
    def varlen(self):
        return self.sender.varlen
    
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        T = self.sender.maxlen
        device = x.device

        h = self.sender.backbone(x)
        ids = torch.empty((B, T), device=device, dtype=torch.long)

        done = torch.zeros((B,), device=device, dtype=torch.bool) if self.sender.varlen else None

        with torch.autocast("cuda", torch.float32):
            for step in range(T):
                step_logits = self.sender._get_logits(h, step)
                tok = step_logits.argmax(dim=-1)

                if self.sender.varlen:
                    tok = torch.where(done, torch.zeros_like(tok), tok)
                    done = done | (tok == 0)

                ids[:, step] = tok
                h = self.sender._residual_update(h, step, tok, done=done)

        if self.sender.varlen:
            lengths = find_lengths_eos(ids)
            t = torch.arange(T, device=device)[None, :]
            after = (t >= (lengths - 1)[:, None])
            ids = torch.where(after, torch.zeros_like(ids), ids)
            return ids, lengths

        return ids


def find_lengths_eos(message_ids: torch.Tensor) -> torch.Tensor:
    B, T = message_ids.shape
    eos = (message_ids == 0)
    has_eos = eos.any(dim=1)
    first = eos.float().argmax(dim=1)  # 0 if no eos
    return torch.where(has_eos, first + 1, torch.full_like(first, T))


class LSTMSenderReinforce(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        maxlen: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        varlen: bool = False,
    ):
        super().__init__()
        assert maxlen >= 1
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.maxlen = int(maxlen)
        self.num_layers = int(num_layers)
        self.varlen = bool(varlen)

        self.dropout = nn.Dropout(dropout)
        self.backbone = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
        )

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(self.embed_dim))

        self.cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size=self.embed_dim, hidden_size=self.hidden_size)
                if i == 0
                else nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
                for i in range(self.num_layers)
            ]
        )
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.vocab_size)

        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x: torch.Tensor):
        """
        Returns:
          ids:     [B,T] long
          msg:     [B,T,V] float one-hot
          logp:    [B,T] float (0 after EOS if varlen)
          entropy: [B,T] float (0 after EOS if varlen)
          lengths: [B] long in 1..T
        """
        B = x.size(0)
        T = self.maxlen
        V = self.vocab_size
        device = x.device

        h0 = self.backbone(self.dropout(x))  # [B,H]
        prev_h = [h0] + [torch.zeros_like(h0) for _ in range(self.num_layers - 1)]
        prev_c = [torch.zeros_like(h0) for _ in range(self.num_layers)]

        inp = self.sos_embedding[None, :].expand(B, -1).to(device)

        ids_list = []
        logp_list = []
        ent_list = []

        done = torch.zeros((B,), device=device, dtype=torch.bool) if self.varlen else None
        zero_tok = torch.zeros((B,), device=device, dtype=torch.long)
        zero_f = torch.zeros((B,), device=device, dtype=torch.float32)

        for step in range(T):
            # unroll stacked LSTMCells
            h_in = inp
            for i, cell in enumerate(self.cells):
                h_i, c_i = cell(h_in, (prev_h[i], prev_c[i]))
                prev_h[i], prev_c[i] = h_i, c_i
                h_in = h_i

            step_logits = self.hidden_to_logits(h_in).to(torch.float32)  # [B,V]
            distr = Categorical(logits=step_logits)

            tok = distr.sample()  # [B]
            step_logp = distr.log_prob(tok).to(torch.float32)  # [B]
            step_ent = distr.entropy().to(torch.float32)        # [B]

            if self.varlen:
                tok = torch.where(done, zero_tok, tok)
                step_logp = torch.where(done, zero_f, step_logp)
                step_ent = torch.where(done, zero_f, step_ent)
                done = done | (tok == 0)

            ids_list.append(tok)
            logp_list.append(step_logp)
            ent_list.append(step_ent)

            inp = self.embedding(tok)

        ids = torch.stack(ids_list, dim=1)          # [B,T]
        logp = torch.stack(logp_list, dim=1)        # [B,T]
        ent = torch.stack(ent_list, dim=1)          # [B,T]

        if self.varlen:
            lengths = find_lengths_eos(ids)
            t = torch.arange(T, device=device)[None, :]
            after = (t >= (lengths - 1)[:, None])
            ids = torch.where(after, torch.zeros_like(ids), ids)
        else:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)

        msg = F.one_hot(ids, num_classes=V).to(torch.float32)
        return ids, msg, logp, ent, lengths

    def inference(self):
        return InferenceLSTMSenderReinforce(self)


class InferenceLSTMSenderReinforce(nn.Module):
    def __init__(self, sender: LSTMSenderReinforce):
        super().__init__()
        self.sender = sender

    @property
    def vocab_size(self):
        return self.sender.vocab_size
    
    @property
    def maxlen(self):
        return self.sender.maxlen
    
    @property
    def varlen(self):
        return self.sender.varlen
    
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        T = self.sender.maxlen
        device = x.device

        h0 = self.sender.backbone(self.sender.dropout(x))
        prev_h = [h0] + [torch.zeros_like(h0) for _ in range(self.sender.num_layers - 1)]
        prev_c = [torch.zeros_like(h0) for _ in range(self.sender.num_layers)]

        inp = self.sender.sos_embedding[None, :].expand(B, -1).to(device)
        ids = []

        done = torch.zeros((B,), device=device, dtype=torch.bool) if self.sender.varlen else None
        zero_tok = torch.zeros((B,), device=device, dtype=torch.long)

        for step in range(T):
            h_in = inp
            for i, cell in enumerate(self.sender.cells):
                h_i, c_i = cell(h_in, (prev_h[i], prev_c[i]))
                prev_h[i], prev_c[i] = h_i, c_i
                h_in = h_i

            step_logits = self.sender.hidden_to_logits(h_in)
            tok = step_logits.argmax(dim=-1)

            if self.sender.varlen:
                tok = torch.where(done, zero_tok, tok)
                done = done | (tok == 0)

            ids.append(tok)
            inp = self.sender.embedding(tok)

        ids = torch.stack(ids, dim=1)

        if self.sender.varlen:
            lengths = find_lengths_eos(ids)
            t = torch.arange(T, device=device)[None, :]
            after = (t >= (lengths - 1)[:, None])
            ids = torch.where(after, torch.zeros_like(ids), ids)
            return ids, lengths

        return ids
