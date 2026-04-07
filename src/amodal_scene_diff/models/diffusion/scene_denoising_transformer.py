from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from amodal_scene_diff.structures import C_OBJ, D_MODEL, D_POSE, K_HID, K_VIS, N_OBJ_MAX, Z_DIM, SceneDiffusionBatch


_CONTINUOUS_DIM = D_POSE + Z_DIM


def _make_beta_schedule(train_timesteps: int, schedule: str) -> torch.Tensor:
    if train_timesteps <= 0:
        raise ValueError("train_timesteps must be positive")
    schedule_name = schedule.lower()
    if schedule_name == "linear":
        return torch.linspace(1.0e-4, 2.0e-2, train_timesteps, dtype=torch.float32)
    if schedule_name != "cosine":
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


def _sampling_schedule(train_timesteps: int, sampling_steps: int, device: torch.device) -> torch.Tensor:
    steps = max(1, min(int(sampling_steps), int(train_timesteps)))
    schedule = torch.linspace(train_timesteps - 1, 0, steps, device=device)
    schedule = schedule.round().long()
    schedule = torch.unique_consecutive(schedule)
    if int(schedule[-1].item()) != 0:
        schedule = torch.cat([schedule, torch.zeros(1, device=device, dtype=torch.long)], dim=0)
    return schedule


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class SceneDenoisingTransformer(nn.Module):
    """Structured denoiser over layout/object geometry states.

    This model keeps the fixed thesis boundary:
    - single-view conditioned
    - scene-level, not object-only
    - no explicit 2D multiview bridge
    - denoising over structured 3D-native scene variables

    Continuous denoising targets:
    - layout state: room proxy parameters
    - visible state: amodal residual + geometry latent
    - hidden state: pose + geometry latent

    Auxiliary supervised heads:
    - hidden existence
    - hidden class
    - support/floor/wall relations
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        layout_weight: float = 1.0,
        visible_weight: float = 1.0,
        hidden_weight: float = 1.0,
        hidden_exist_weight: float = 1.0,
        hidden_cls_weight: float = 1.0,
        support_weight: float = 0.5,
        floor_weight: float = 0.25,
        wall_weight: float = 0.25,
        visible_mode: str = "diffusion",
        attention_mode: str = "full",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.train_timesteps = int(train_timesteps)
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in {"epsilon", "eps", "v", "v_prediction"}:
            raise ValueError(f"unsupported prediction_type: {prediction_type}")

        self.layout_weight = float(layout_weight)
        self.visible_weight = float(visible_weight)
        self.hidden_weight = float(hidden_weight)
        self.hidden_exist_weight = float(hidden_exist_weight)
        self.hidden_cls_weight = float(hidden_cls_weight)
        self.support_weight = float(support_weight)
        self.floor_weight = float(floor_weight)
        self.wall_weight = float(wall_weight)
        self.visible_mode = str(visible_mode).lower()
        if self.visible_mode not in {"diffusion", "deterministic"}:
            raise ValueError(f"unsupported visible_mode: {visible_mode}")
        self.attention_mode = str(attention_mode).lower()
        if self.attention_mode not in {"full", "occlusion_biased"}:
            raise ValueError(f"unsupported attention_mode: {attention_mode}")

        betas = _make_beta_schedule(self.train_timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)

        self.source_embedding = nn.Embedding(3, d_model)
        self.token_type_embedding = nn.Embedding(4, d_model)
        self.time_embedding = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.global_proj = nn.Linear(D_MODEL, d_model)
        self.layout0_proj = nn.Linear(D_POSE, d_model)
        self.pose_proj = nn.Linear(D_POSE, d_model)
        self.conf_proj = nn.Linear(3, d_model)

        self.layout_cond_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.uncertainty_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.visible_cond_in = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        self.layout_state_in = nn.Sequential(nn.LayerNorm(D_POSE), nn.Linear(D_POSE, d_model))
        self.visible_state_in = nn.Sequential(nn.LayerNorm(_CONTINUOUS_DIM), nn.Linear(_CONTINUOUS_DIM, d_model))
        self.hidden_state_in = nn.Sequential(nn.LayerNorm(_CONTINUOUS_DIM), nn.Linear(_CONTINUOUS_DIM, d_model))
        self.hidden_queries = nn.Parameter(torch.randn(K_HID, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scene_norm = nn.LayerNorm(d_model)
        self.object_norm = nn.LayerNorm(d_model)
        self.layout_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, D_POSE),
        )
        self.visible_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, _CONTINUOUS_DIM),
        )
        self.hidden_state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, _CONTINUOUS_DIM),
        )
        self.hidden_exist_head = nn.Linear(d_model, 1)
        self.hidden_cls_head = nn.Linear(d_model, C_OBJ)
        self.relation_q = nn.Linear(d_model, d_model)
        self.relation_k = nn.Linear(d_model, d_model)
        self.floor_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.wall_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _attention_mask(self, device: torch.device) -> torch.Tensor | None:
        if self.attention_mode == "full":
            return None

        token_count = 2 + K_VIS + K_HID
        hidden_start = 2 + K_VIS
        mask = torch.zeros((token_count, token_count), dtype=torch.bool, device=device)
        # Keep visible/layout tokens clean: hidden noisy states can read visible context,
        # but visible/layout tokens do not attend back into hidden slots.
        mask[:hidden_start, hidden_start:] = True
        return mask

    def continuous_state_targets(self, batch: SceneDiffusionBatch) -> dict[str, torch.Tensor]:
        target = batch.target
        return {
            "layout": target.layout_gt,
            "visible": torch.cat([target.visible_amodal_res_gt, target.visible_z_gt], dim=-1),
            "hidden": torch.cat([target.hidden_pose_gt, target.hidden_z_gt], dim=-1),
        }

    def q_sample(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = _extract(self.sqrt_alphas_cumprod, timesteps, tuple(x0.shape))
        sigma = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, tuple(x0.shape))
        return alpha * x0 + sigma * noise

    def prediction_target(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.prediction_type in {"epsilon", "eps"}:
            return noise
        alpha = _extract(self.sqrt_alphas_cumprod, timesteps, tuple(x0.shape))
        sigma = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, tuple(x0.shape))
        return alpha * noise - sigma * x0

    def _prediction_to_x0_and_eps(
        self,
        prediction: torch.Tensor,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = _extract(self.sqrt_alphas_cumprod, timesteps, tuple(xt.shape))
        sigma = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, tuple(xt.shape))
        if self.prediction_type in {"epsilon", "eps"}:
            eps = prediction
            x0 = (xt - sigma * eps) / alpha.clamp_min(1.0e-6)
            return x0, eps

        v = prediction
        x0 = alpha * xt - sigma * v
        eps = sigma * xt + alpha * v
        return x0, eps

    def forward_denoiser(
        self,
        batch: SceneDiffusionBatch,
        timesteps: torch.Tensor,
        layout_xt: torch.Tensor,
        visible_xt: torch.Tensor,
        hidden_xt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = batch.cond
        source = self.source_embedding(cond.source_id.long())
        global_bias = self.global_proj(cond.f_global)
        time_bias = self.time_mlp(self.time_embedding(timesteps))
        context_bias = source + global_bias + time_bias

        layout_token = (
            self.layout_cond_in(cond.layout_token_cond.squeeze(1) + self.layout0_proj(cond.layout0_calib))
            + self.layout_state_in(layout_xt)
            + context_bias
            + self.token_type_embedding.weight[0].unsqueeze(0)
        )
        uncertainty_token = (
            self.uncertainty_in(cond.uncertainty_token.squeeze(1))
            + context_bias
            + self.token_type_embedding.weight[1].unsqueeze(0)
        )

        visible_aux = self.pose_proj(cond.pose0_calib) + self.conf_proj(
            torch.cat(
                [
                    cond.slot_confidence,
                    cond.lock_gate,
                    cond.visible_valid_mask.float().unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        visible_tokens = (
            self.visible_cond_in(cond.visible_tokens_cond + visible_aux)
            + self.visible_state_in(visible_xt)
            + context_bias.unsqueeze(1)
            + self.token_type_embedding.weight[2].view(1, 1, -1)
        )
        hidden_tokens = (
            self.hidden_queries.unsqueeze(0)
            + self.hidden_state_in(hidden_xt)
            + context_bias.unsqueeze(1)
            + self.token_type_embedding.weight[3].view(1, 1, -1)
        )

        tokens = torch.cat(
            [
                layout_token.unsqueeze(1),
                uncertainty_token.unsqueeze(1),
                visible_tokens,
                hidden_tokens,
            ],
            dim=1,
        )
        src_key_padding_mask = torch.zeros(
            (tokens.shape[0], tokens.shape[1]),
            dtype=torch.bool,
            device=tokens.device,
        )
        src_key_padding_mask[:, 2 : 2 + K_VIS] = ~cond.visible_valid_mask.bool()
        encoded = self.transformer(
            tokens,
            mask=self._attention_mask(tokens.device),
            src_key_padding_mask=src_key_padding_mask,
        )

        layout_ctx = self.scene_norm(encoded[:, 0])
        visible_ctx = self.object_norm(encoded[:, 2 : 2 + K_VIS])
        hidden_ctx = self.object_norm(encoded[:, 2 + K_VIS :])
        object_tokens = torch.cat([visible_ctx, hidden_ctx], dim=1)

        relation_q = self.relation_q(object_tokens)
        relation_k = self.relation_k(object_tokens)
        support_logits = torch.einsum("bid,bjd->bij", relation_q, relation_k) / math.sqrt(self.d_model)

        return {
            "layout_pred": self.layout_head(layout_ctx),
            "visible_pred": self.visible_head(visible_ctx),
            "hidden_pred": self.hidden_state_head(hidden_ctx),
            "hidden_exist_logits": self.hidden_exist_head(hidden_ctx),
            "hidden_cls_logits": self.hidden_cls_head(hidden_ctx),
            "support_logits": support_logits,
            "floor_logits": self.floor_head(object_tokens).squeeze(-1),
            "wall_logits": self.wall_head(object_tokens).squeeze(-1),
        }

    @torch.no_grad()
    def sample_posterior(
        self,
        batch: SceneDiffusionBatch,
        num_sampling_steps: int = 50,
    ) -> dict[str, torch.Tensor]:
        device = batch.cond.f_global.device
        batch_size = batch.batch_size
        visible_mask = batch.cond.visible_valid_mask.float().unsqueeze(-1)

        layout_xt = torch.randn((batch_size, D_POSE), device=device)
        if self.visible_mode == "diffusion":
            visible_xt = torch.randn((batch_size, K_VIS, _CONTINUOUS_DIM), device=device) * visible_mask
        else:
            visible_xt = torch.zeros((batch_size, K_VIS, _CONTINUOUS_DIM), device=device)
        hidden_xt = torch.randn((batch_size, K_HID, _CONTINUOUS_DIM), device=device)

        schedule = _sampling_schedule(self.train_timesteps, num_sampling_steps, device)
        latest_pred: dict[str, torch.Tensor] | None = None

        for index, timestep in enumerate(schedule):
            t = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            pred = self.forward_denoiser(
                batch=batch,
                timesteps=t,
                layout_xt=layout_xt,
                visible_xt=visible_xt,
                hidden_xt=hidden_xt,
            )
            latest_pred = pred

            layout_x0, layout_eps = self._prediction_to_x0_and_eps(pred["layout_pred"], layout_xt, t)
            if self.visible_mode == "diffusion":
                visible_x0, visible_eps = self._prediction_to_x0_and_eps(pred["visible_pred"], visible_xt, t)
                visible_x0 = visible_x0 * visible_mask
            else:
                visible_x0 = pred["visible_pred"] * visible_mask
                visible_eps = torch.zeros_like(visible_x0)
            hidden_x0, hidden_eps = self._prediction_to_x0_and_eps(pred["hidden_pred"], hidden_xt, t)

            if index == len(schedule) - 1:
                layout_xt = layout_x0
                visible_xt = visible_x0
                hidden_xt = hidden_x0
                break

            prev_timestep = schedule[index + 1]
            alpha_prev = self.sqrt_alphas_cumprod[int(prev_timestep.item())]
            sigma_prev = self.sqrt_one_minus_alphas_cumprod[int(prev_timestep.item())]

            layout_xt = alpha_prev * layout_x0 + sigma_prev * layout_eps
            if self.visible_mode == "diffusion":
                visible_xt = (alpha_prev * visible_x0 + sigma_prev * visible_eps) * visible_mask
            else:
                visible_xt = visible_x0
            hidden_xt = alpha_prev * hidden_x0 + sigma_prev * hidden_eps

        final_t = torch.zeros((batch_size,), device=device, dtype=torch.long)
        final_pred = self.forward_denoiser(
            batch=batch,
            timesteps=final_t,
            layout_xt=layout_xt,
            visible_xt=visible_xt,
            hidden_xt=hidden_xt,
        )
        latest_pred = final_pred if latest_pred is None else final_pred

        return {
            "layout": layout_xt,
            "visible": visible_xt,
            "hidden": hidden_xt,
            "hidden_exist_logits": latest_pred["hidden_exist_logits"],
            "hidden_exist_probs": torch.sigmoid(latest_pred["hidden_exist_logits"].squeeze(-1)),
            "hidden_cls_logits": latest_pred["hidden_cls_logits"],
            "hidden_cls_probs": torch.softmax(latest_pred["hidden_cls_logits"], dim=-1),
            "support_logits": latest_pred["support_logits"],
            "floor_logits": latest_pred["floor_logits"],
            "wall_logits": latest_pred["wall_logits"],
        }

    def compute_losses(self, batch: SceneDiffusionBatch) -> dict[str, torch.Tensor]:
        target = batch.target
        device = batch.cond.f_global.device
        batch_size = batch.batch_size
        timesteps = torch.randint(0, self.train_timesteps, (batch_size,), device=device, dtype=torch.long)

        continuous_targets = self.continuous_state_targets(batch)
        layout_x0 = continuous_targets["layout"]
        visible_x0 = continuous_targets["visible"]
        hidden_x0 = continuous_targets["hidden"]

        layout_noise = torch.randn_like(layout_x0)
        if self.visible_mode == "diffusion":
            visible_noise = torch.randn_like(visible_x0)
            visible_xt = self.q_sample(visible_x0, visible_noise, timesteps)
        else:
            visible_noise = torch.zeros_like(visible_x0)
            visible_xt = torch.zeros_like(visible_x0)
        hidden_noise = torch.randn_like(hidden_x0)

        layout_xt = self.q_sample(layout_x0, layout_noise, timesteps)
        hidden_xt = self.q_sample(hidden_x0, hidden_noise, timesteps)

        visible_mask = target.visible_loss_mask.float().unsqueeze(-1)
        hidden_mask = target.hidden_gt_mask.float().unsqueeze(-1)
        visible_xt = visible_xt * visible_mask
        hidden_xt = hidden_xt * hidden_mask

        pred = self.forward_denoiser(
            batch=batch,
            timesteps=timesteps,
            layout_xt=layout_xt,
            visible_xt=visible_xt,
            hidden_xt=hidden_xt,
        )

        layout_target = self.prediction_target(layout_x0, layout_noise, timesteps)
        if self.visible_mode == "diffusion":
            visible_target = self.prediction_target(visible_x0, visible_noise, timesteps)
        else:
            visible_target = visible_x0
        hidden_target = self.prediction_target(hidden_x0, hidden_noise, timesteps)

        layout_loss = F.mse_loss(pred["layout_pred"], layout_target)
        visible_loss = ((pred["visible_pred"] - visible_target) ** 2 * visible_mask).sum() / visible_mask.sum().clamp_min(1.0)
        hidden_loss = ((pred["hidden_pred"] - hidden_target) ** 2 * hidden_mask).sum() / hidden_mask.sum().clamp_min(1.0)

        hidden_mask_2d = target.hidden_gt_mask.float()
        hidden_exist_loss = F.binary_cross_entropy_with_logits(
            pred["hidden_exist_logits"].squeeze(-1),
            hidden_mask_2d,
        )
        if hidden_mask_2d.bool().any():
            hidden_cls_loss = F.cross_entropy(
                pred["hidden_cls_logits"][hidden_mask_2d.bool()],
                target.hidden_cls_gt[hidden_mask_2d.bool()],
            )
        else:
            hidden_cls_loss = torch.zeros((), device=device)

        relation_valid = target.relation_valid_mask.float()
        pair_mask = relation_valid.unsqueeze(1) * relation_valid.unsqueeze(2)
        support_loss = F.binary_cross_entropy_with_logits(
            pred["support_logits"],
            target.support_gt.float(),
            weight=pair_mask,
            reduction="sum",
        ) / pair_mask.sum().clamp_min(1.0)
        floor_loss = F.binary_cross_entropy_with_logits(
            pred["floor_logits"],
            target.floor_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)
        wall_loss = F.binary_cross_entropy_with_logits(
            pred["wall_logits"],
            target.wall_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)

        total = (
            self.layout_weight * layout_loss
            + self.visible_weight * visible_loss
            + self.hidden_weight * hidden_loss
            + self.hidden_exist_weight * hidden_exist_loss
            + self.hidden_cls_weight * hidden_cls_loss
            + self.support_weight * support_loss
            + self.floor_weight * floor_loss
            + self.wall_weight * wall_loss
        )
        return {
            "loss_total": total,
            "loss_layout_diff": layout_loss,
            "loss_visible_diff": visible_loss,
            "loss_hidden_diff": hidden_loss,
            "loss_hidden_exist": hidden_exist_loss,
            "loss_hidden_cls": hidden_cls_loss,
            "loss_support": support_loss,
            "loss_floor": floor_loss,
            "loss_wall": wall_loss,
            "metric_t_mean": timesteps.float().mean(),
        }

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
