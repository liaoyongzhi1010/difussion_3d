from __future__ import annotations

from typing import Callable

import torch


def _sampling_schedule(train_timesteps: int, sampling_steps: int, device: torch.device) -> torch.Tensor:
    steps = max(1, min(int(sampling_steps), int(train_timesteps)))
    schedule = torch.linspace(train_timesteps - 1, 0, steps, device=device)
    schedule = schedule.round().long()
    schedule = torch.unique_consecutive(schedule)
    if int(schedule[-1].item()) != 0:
        schedule = torch.cat([schedule, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
    return schedule


def sample_ddim_posterior(
    *,
    x_shape: tuple[int, ...],
    device: torch.device,
    train_timesteps: int,
    sampling_steps: int,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    denoiser_step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prediction_to_x0_and_eps: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic DDIM posterior sampling.

    - `denoiser_step(xt, t)` returns the prediction (eps or v) for a given noisy state and integer step.
    - `prediction_to_x0_and_eps(prediction, xt, t)` decomposes the prediction back to (x0, eps).
    """
    xt = initial_noise if initial_noise is not None else torch.randn(x_shape, device=device)
    batch_size = x_shape[0]
    schedule = _sampling_schedule(train_timesteps, sampling_steps, device)

    for index, timestep in enumerate(schedule):
        t = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
        prediction = denoiser_step(xt, t)
        x0, eps = prediction_to_x0_and_eps(prediction, xt, t)

        if index == len(schedule) - 1:
            return x0

        next_timestep = int(schedule[index + 1].item())
        alpha_prev = sqrt_alphas_cumprod[next_timestep]
        sigma_prev = sqrt_one_minus_alphas_cumprod[next_timestep]
        xt = alpha_prev * x0 + sigma_prev * eps

    return xt
