"""DiT-class AdaLN-Zero hidden denoiser.

Drop-in replacement for the plain `nn.TransformerDecoder`-based hidden denoiser
in `diffusion.scene_model.SingleViewSceneDiffusion`. Conditioning (time +
pooled observation context) flows through AdaLN-Zero, matching the original
DiT (Peebles & Xie, 2023) with cross-attention to the full observation memory.

Forward API (intended to be called from the scene model):
    hidden_ctx = DiTHiddenDenoiser(d_model=..., ...)(
        queries  = [B, K, D],     # noisy latent tokens for the K hidden slots
        memory   = [B, M, D],     # observation + visible anchor tokens
        cond     = [B, D],        # sum of time embedding + pooled global token
    )

Shape contract is identical to the baseline hidden_decoder so the surrounding
state heads (hidden_state_head, hidden_exist_head, ...) can reuse hidden_ctx.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _AdaLNZeroMLP(nn.Module):
    """Produce (shift, scale, gate) triples from a conditioning vector.

    `num_triples` lets one MLP produce all modulation parameters a block needs
    in a single matmul, following the original DiT implementation.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, num_triples: int) -> None:
        super().__init__()
        self.num_triples = num_triples
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * num_triples * hidden_dim, bias=True),
        )
        # zero-init so blocks are identity at step 0 (DiT trick)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        chunks = self.proj(cond).chunk(3 * self.num_triples, dim=-1)
        out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(self.num_triples):
            shift = chunks[3 * i]
            scale = chunks[3 * i + 1]
            gate = chunks[3 * i + 2]
            out.append((shift, scale, gate))
        return out


class _DiTBlock(nn.Module):
    """One DiT-style decoder block: AdaLN-Zero → self-attn → cross-attn → MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_ratio: float,
        dropout: float,
        cond_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = _RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = _RMSNorm(d_model)
        inner = int(d_model * ffn_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, inner),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
        )
        self.ada_ln = _AdaLNZeroMLP(cond_dim, d_model, num_triples=3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (sa_shift, sa_scale, sa_gate), (ca_shift, ca_scale, ca_gate), (mlp_shift, mlp_scale, mlp_gate) = self.ada_ln(cond)

        h = _modulate(self.norm1(x), sa_shift, sa_scale)
        sa_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_gate.unsqueeze(1) * sa_out

        h = _modulate(self.norm2(x), ca_shift, ca_scale)
        ca_out, _ = self.cross_attn(
            h, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + ca_gate.unsqueeze(1) * ca_out

        h = _modulate(self.norm3(x), mlp_shift, mlp_scale)
        x = x + mlp_gate.unsqueeze(1) * self.mlp(h)
        return x


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiTHiddenDenoiser(nn.Module):
    """DiT-class hidden denoiser with AdaLN-Zero conditioning + cross-attention.

    Presets (from the design spec):
      - base : d_model=768,  num_blocks=8,  num_heads=12
      - large: d_model=1024, num_blocks=12, num_heads=16
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_blocks: int = 12,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        num_slots: int = 8,
        input_dim: int | None = None,
        output_dim: int | None = None,
        cond_extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.input_proj = nn.Linear(input_dim or d_model, d_model)
        self.output_proj = nn.Linear(d_model, output_dim or d_model)
        # zero-init output projection so the model starts as identity on x_t
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.slot_embedding = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.time_embedding = _SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        cond_dim = d_model + cond_extra_dim
        self.cond_extra_dim = cond_extra_dim

        self.blocks = nn.ModuleList(
            [
                _DiTBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = _RMSNorm(d_model)

    def forward(
        self,
        hidden_xt: torch.Tensor,                # [B, K, input_dim]
        memory: torch.Tensor,                   # [B, M, d_model]
        timesteps: torch.Tensor,                # [B]
        cond_extra: torch.Tensor | None = None, # [B, cond_extra_dim]
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_xt.shape[1] != self.num_slots:
            raise ValueError(
                f"DiT was configured for {self.num_slots} slots but got {hidden_xt.shape[1]}"
            )
        x = self.input_proj(hidden_xt) + self.slot_embedding.unsqueeze(0)
        t_emb = self.time_mlp(self.time_embedding(timesteps))
        if self.cond_extra_dim > 0:
            if cond_extra is None or cond_extra.shape[-1] != self.cond_extra_dim:
                raise ValueError("cond_extra missing or wrong shape for configured cond_extra_dim")
            cond = torch.cat([t_emb, cond_extra], dim=-1)
        else:
            cond = t_emb
        for block in self.blocks:
            x = block(x, memory, cond, memory_key_padding_mask=memory_key_padding_mask)
        x = self.final_norm(x)
        return self.output_proj(x)

    @classmethod
    def from_preset(cls, preset: str, num_slots: int, input_dim: int, output_dim: int, cond_extra_dim: int = 0) -> "DiTHiddenDenoiser":
        if preset == "base":
            return cls(
                d_model=768, num_blocks=8, num_heads=12,
                num_slots=num_slots, input_dim=input_dim, output_dim=output_dim,
                cond_extra_dim=cond_extra_dim,
            )
        if preset == "large":
            return cls(
                d_model=1024, num_blocks=12, num_heads=16,
                num_slots=num_slots, input_dim=input_dim, output_dim=output_dim,
                cond_extra_dim=cond_extra_dim,
            )
        raise ValueError(f"unknown DiT preset {preset!r}; expected 'base' or 'large'")


__all__ = ["DiTHiddenDenoiser"]
