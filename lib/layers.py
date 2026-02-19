import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt import GPT


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_size: int,
            maxlen: int,
            dropout: float = 0.0,
            codebook_dropout: float = 0.0,
            init_logit_scale: float = 10.0,
            init_gamma: float = 0.2,
            varlen=True,
            shared_codebooks=True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.codebook_dropout = nn.Dropout(codebook_dropout)

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        self.stop_backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        num_codebooks = 1 if shared_codebooks else maxlen
        self.codebook = nn.Parameter(torch.empty(num_codebooks, vocab_size, hidden_size))
        nn.init.normal_(self.codebook, mean=0.0, std=0.02)

        self.logit_scale = nn.Parameter(torch.full((maxlen,), math.log(float(init_logit_scale))))
        self.gamma = nn.Parameter(torch.zeros(maxlen - 1) + math.log(float(init_gamma)))
        self.rmsnorm = nn.RMSNorm(hidden_size)

        self.shared_codebooks = shared_codebooks
        self.varlen = varlen
        if varlen:
            self.stop_transformer = GPT(
                num_layers=1,
                num_heads=hidden_size // 64,
                model_dim=hidden_size,
                max_seq_len=maxlen,
                causal=True,
                dropout=dropout
            )
            self.stop_predictor = nn.Linear(hidden_size, 1)
            self.stop_bias = nn.Parameter(torch.zeros(maxlen - 1))

    def _get_codebook(self, step):
        if self.shared_codebooks:
            return self.codebook[0]
        return self.codebook[step]
    
    def _get_logits(self, h: torch.Tensor, step: int) -> torch.Tensor:
        z = self.codebook_dropout(h)
        C = self._get_codebook(step)

        logit_scale = torch.exp(self.logit_scale[step])
        z_n = F.normalize(z, dim=-1)
        C_n = F.normalize(C, dim=-1)
        return (z_n @ C_n.t()) * logit_scale
        
    @staticmethod
    def _process_stop_logits(stop_logits, return_survival_logits=False):
        log_cont = F.logsigmoid(-stop_logits)
        S = torch.cumsum(log_cont, dim=-1)
        length_logits = torch.cat([S - log_cont + F.logsigmoid(stop_logits), S[..., -1:]], dim=-1)
        if return_survival_logits:
            survival_logits = torch.cat([S.new_zeros(S.shape[:-1] + (1,)), S], dim=-1)
            return length_logits, survival_logits
        return length_logits
    
    def forward(self, x: torch.Tensor, tau: float):
        B = x.size(0)
        T = self.maxlen
        V = self.vocab_size
        dtype = x.dtype
        device = x.device

        h = self.backbone(self.dropout(x))

        with torch.autocast('cuda', torch.float32):
            if self.varlen:
                stop_hidden_states = torch.empty((B, T, self.hidden_size), device=device, dtype=dtype)
                stop_hidden_states[:, 0] = self.stop_backbone(self.dropout(x))

            logits = torch.empty((B, T, V), device=device, dtype=dtype)
            gs_probs = torch.empty((B, T, V), device=device, dtype=dtype)

            for step in range(T):
                step_logits = self._get_logits(h, step)
                step_gs_probs = F.gumbel_softmax(step_logits, tau=tau, hard=False, dim=-1)

                if step < T - 1:
                    C_b = self._get_codebook(step)
                    expected_b = step_gs_probs @ C_b
                    h = h - self.gamma[step].exp() * expected_b
                    h = self.rmsnorm(h)
                    if self.varlen:
                        stop_hidden_states[:, step + 1] = h

                logits[:, step] = step_logits
                gs_probs[:, step] = step_gs_probs
            
            if self.varlen:
                with torch.autocast('cuda', torch.bfloat16):
                    stop_hidden_states = self.stop_transformer(stop_hidden_states)
                stop_logit = self.stop_predictor(self.dropout(stop_hidden_states[:, 1:])).squeeze(dim=-1) + self.stop_bias[None]
                length_logits, survival_logits = self._process_stop_logits(stop_logit.to(torch.float32), return_survival_logits=True)

                return logits, gs_probs, length_logits, survival_logits
            
        return logits, gs_probs
    
    def inference(self, mode):
        return InferenceEncoder(self, mode)
    

class InferenceEncoder(nn.Module):
    def __init__(self, sender: Encoder, mode='argmax'):
        super().__init__()
        self.sender = sender
        self.mode = mode

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
        with torch.autocast('cuda', torch.float32):
            if self.sender.varlen:
                stop_hidden_states = torch.empty((B, T, self.sender.hidden_size), device=device, dtype=h.dtype)
                stop_hidden_states[:, 0] = self.sender.stop_backbone(x)
            codes = torch.empty((B, T), device=device, dtype=torch.long)

            for step in range(T):
                step_logits = self.sender._get_logits(h, step)
                step_code = step_logits.argmax(dim=-1)
                codes[:, step] = step_code

                if step < T - 1:
                    C_b = self.sender._get_codebook(step)
                    codeword_b = C_b.index_select(0, step_code)
                    h = h - self.sender.gamma[step].exp() * codeword_b
                    h = self.sender.rmsnorm(h)
                    if self.sender.varlen:
                        stop_hidden_states[:, step + 1] = h

            if self.sender.varlen:
                with torch.autocast('cuda', torch.bfloat16):
                    stop_hidden_states = self.sender.stop_transformer(stop_hidden_states)
                stop_logit = self.sender.stop_predictor(stop_hidden_states[:, 1:]).squeeze(dim=-1) + self.sender.stop_bias[None]
                length_logits = self.sender._process_stop_logits(stop_logit.to(torch.float32))

                if self.mode == 'argmax':
                    lengths = length_logits.argmax(dim=-1) + 1  # [B] in 1..T
                elif self.mode == 'mean':
                    with torch.autocast("cuda", torch.float32):
                        probs = length_logits.softmax(dim=-1)  # [B, T]
                        t = torch.arange(1, probs.size(-1) + 1, device=probs.device, dtype=probs.dtype)
                        E_L = (probs * t[None, :]).sum(dim=-1)  # [B]
                        lengths = E_L.round().clamp(1, probs.size(-1)).to(torch.long)
                elif self.mode == 'median':
                    probs = length_logits.softmax(dim=-1)  # [B,T]
                    cdf = probs.cumsum(dim=-1)
                    lengths = (cdf >= 0.5).to(torch.long).argmax(dim=-1) + 1
                else:
                    raise ValueError(self.mode)
                    
                return codes, lengths
            
            return codes
    

class Decoder(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            embed_dim, 
            hidden_size, 
            maxlen, 
            dropout=0., 
            num_layers=2
    ):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, hidden_size, bias=False)
        self.decoder = nn.Linear(hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer = GPT(
            num_layers=num_layers,
            num_heads=hidden_size // 64,
            model_dim=hidden_size,
            max_seq_len=maxlen,
            causal=True,
            dropout=dropout
        )

        self.vocab_size, self.maxlen = vocab_size, maxlen

    def forward(self, message):
        emb = self.dropout(self.embedding(message))
        emb = self.transformer(emb)
        x = self.decoder(emb)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x
    
    def inference(self):
        return InferenceDecoder(self)
    

class InferenceDecoder(nn.Module):
    def __init__(self, receiver : Decoder):
        super(InferenceDecoder, self).__init__()
        self.receiver = receiver
        self.transposed_embedding = self.receiver.embedding.weight.T

    def forward(self, message):
        emb = F.embedding(message, self.transposed_embedding)
        emb = self.receiver.transformer(emb)
        x = self.receiver.decoder(emb)
        x = F.normalize(x, dim=-1)
        return x
