import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ReinforceEncoder
from lib.layers import Decoder


def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)  # [B]


class MeanBaseline(nn.Module):
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("value", torch.zeros(()), persistent=True)
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool), persistent=True)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        m = x.mean()
        if not bool(self.inited):
            self.value.copy_(m)
            self.inited.fill_(True)
        else:
            self.value.lerp_(m, 1.0 - self.momentum)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_full(x.shape, self.value)



class ReinforceFixedGame(nn.Module):
    """
    optimized_loss = policy_loss - sender_entropy_coeff * entropy + loss.mean()
    policy_loss = mean( (loss.detach() - baseline(loss)) * sum_t logp_t )
    """
    def __init__(self, sender: ReinforceEncoder, decoder: Decoder, baseline_momentum: float = 0.99):
        super().__init__()
        self.sender = sender
        self.decoder = decoder
        self.baseline_loss = MeanBaseline(momentum=baseline_momentum)

    def forward(self, x: torch.Tensor, sender_entropy_coeff: torch.Tensor):
        with torch.autocast("cuda", torch.bfloat16):
            _ids, msg, logp_tok, ent_tok, lengths = self.sender(x)
            out = self.decoder(msg)

        with torch.autocast("cuda", torch.float32):
            # per-sample loss [B]
            loss_vec = cosine_loss(x.float(), out[:, -1].float())

            eff_logp = logp_tok.sum(dim=1)  # [B]
            eff_ent = ent_tok.sum(dim=1) / lengths.to(torch.float32).clamp(min=1.0)  # [B]

            b = self.baseline_loss.predict(loss_vec.detach())
            policy_loss = ((loss_vec.detach() - b) * eff_logp).mean()

            entropy_term = -(eff_ent.mean() * sender_entropy_coeff.to(torch.float32))

            optimized_loss = policy_loss + entropy_term + loss_vec.mean()

            if self.training:
                self.baseline_loss.update(loss_vec.detach())

            metrics = {
                "loss": optimized_loss.detach(),
                "recon_loss": loss_vec.mean().detach(),
                "policy_loss": policy_loss.detach(),
                "sender_entropy": eff_ent.mean().detach(),
                "baseline_loss": self.baseline_loss.value.detach(),
                "sender_entropy_coeff": sender_entropy_coeff.detach(),
            }

        return optimized_loss, metrics


class ReinforceVarlenGame(nn.Module):
    """
    optimized_loss = policy_length_loss + policy_loss - entropy + loss.mean()
    with EGG mask i < message_length (EOS included automatically).
    """
    def __init__(self, sender: ReinforceEncoder, decoder: Decoder, baseline_momentum: float = 0.99):
        super().__init__()
        self.sender = sender
        self.decoder = decoder
        self.baseline_loss = MeanBaseline(momentum=baseline_momentum)
        self.baseline_len = MeanBaseline(momentum=baseline_momentum)

    def forward(
        self,
        x: torch.Tensor,
        sender_entropy_coeff: torch.Tensor,
        length_cost: torch.Tensor,
    ):
        with torch.autocast("cuda", torch.bfloat16):
            _ids, msg, logp_tok, ent_tok, lengths = self.sender(x)
            out = self.decoder(msg)

        with torch.autocast("cuda", torch.float32):
            B, T, _ = out.shape
            device = x.device

            idx = (lengths - 1).clamp(0, T - 1)
            chosen = out[torch.arange(B, device=device), idx]
            loss_vec = cosine_loss(x.float(), chosen.float())  # [B]

            t = torch.arange(T, device=device)[None, :]
            not_eosed = (t < lengths[:, None]).to(torch.float32)  # [B,T] (EOS included)

            eff_logp = (not_eosed * logp_tok).sum(dim=1)  # [B]
            eff_ent = (not_eosed * ent_tok).sum(dim=1) / lengths.to(torch.float32).clamp(min=1.0)  # [B]

            b_loss = self.baseline_loss.predict(loss_vec.detach())
            policy_loss = ((loss_vec.detach() - b_loss) * eff_logp).mean()

            length_loss = lengths.to(torch.float32) * length_cost.to(torch.float32)  # [B]
            b_len = self.baseline_len.predict(length_loss.detach())
            policy_length_loss = ((length_loss.detach() - b_len) * eff_logp).mean()

            entropy_term = -(eff_ent.mean() * sender_entropy_coeff.to(torch.float32))

            optimized_loss = policy_length_loss + policy_loss + entropy_term + loss_vec.mean()

            if self.training:
                self.baseline_loss.update(loss_vec.detach())
                self.baseline_len.update(length_loss.detach())

            metrics = {
                "loss": optimized_loss.detach(),
                "recon_loss": loss_vec.mean().detach(),
                "policy_loss": policy_loss.detach(),
                "policy_length_loss": policy_length_loss.detach(),
                "sender_entropy": eff_ent.mean().detach(),
                "E_L": lengths.float().mean().detach(),
                "p_L_max": (lengths == T).float().mean().detach(),
                "baseline_loss": self.baseline_loss.value.detach(),
                "baseline_len": self.baseline_len.value.detach(),
                "sender_entropy_coeff": sender_entropy_coeff.detach(),
                "length_cost": length_cost.detach(),
            }

        return optimized_loss, metrics