from __future__ import annotations

import math

import torch


def _make_beta_schedule(train_timesteps: int, schedule: str) -> torch.Tensor:
    if train_timesteps <= 0:
        raise ValueError("train_timesteps must be positive")
    name = schedule.lower()
    if name == "linear":
        return torch.linspace(1.0e-4, 2.0e-2, train_timesteps, dtype=torch.float32)
    if name != "cosine":
        raise ValueError(f"unsupported beta schedule: {schedule}")

    steps = train_timesteps + 1
    x = torch.linspace(0, train_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1.0e-5, 0.999)


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    while values.ndim < len(target_shape):
        values = values.unsqueeze(-1)
    return values


class NoiseScheduler:
    """Cosine/linear beta-schedule bookkeeping shared by training and sampling.

    Supports `epsilon` and `v_prediction` parameterizations.
    """

    def __init__(
        self,
        *,
        train_timesteps: int,
        schedule: str,
        prediction_type: str,
    ) -> None:
        self.train_timesteps = int(train_timesteps)
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in {"epsilon", "eps", "v", "v_prediction"}:
            raise ValueError(f"unsupported prediction_type: {prediction_type}")
        betas = _make_beta_schedule(self.train_timesteps, schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def register_buffers(self, module: torch.nn.Module) -> None:
        module.register_buffer("betas", self.betas, persistent=False)
        module.register_buffer("alphas_cumprod", self.alphas_cumprod, persistent=False)
        module.register_buffer("sqrt_alphas_cumprod", self.sqrt_alphas_cumprod, persistent=False)
        module.register_buffer(
            "sqrt_one_minus_alphas_cumprod", self.sqrt_one_minus_alphas_cumprod, persistent=False
        )

    @staticmethod
    def q_sample(
        *,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        alpha = _extract(sqrt_alphas_cumprod, timesteps, tuple(x0.shape))
        sigma = _extract(sqrt_one_minus_alphas_cumprod, timesteps, tuple(x0.shape))
        return alpha * x0 + sigma * noise

    def prediction_target(
        self,
        *,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type in {"epsilon", "eps"}:
            return noise
        alpha = _extract(sqrt_alphas_cumprod, timesteps, tuple(x0.shape))
        sigma = _extract(sqrt_one_minus_alphas_cumprod, timesteps, tuple(x0.shape))
        return alpha * noise - sigma * x0

    def prediction_to_x0_and_eps(
        self,
        *,
        prediction: torch.Tensor,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = _extract(sqrt_alphas_cumprod, timesteps, tuple(xt.shape))
        sigma = _extract(sqrt_one_minus_alphas_cumprod, timesteps, tuple(xt.shape))
        if self.prediction_type in {"epsilon", "eps"}:
            eps = prediction
            x0 = (xt - sigma * eps) / alpha.clamp_min(1.0e-6)
            return x0, eps
        v = prediction
        x0 = alpha * xt - sigma * v
        eps = sigma * xt + alpha * v
        return x0, eps
