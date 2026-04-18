from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from amodal_scene_diff.backbones import build_observation_backbone
from amodal_scene_diff.structures import (
    C_OBJ,
    D_MODEL,
    D_POSE,
    K_HID,
    K_VIS,
    Z_DIM,
    SingleViewSceneBatch,
)

from .sampler import sample_ddim_posterior
from .scheduler import NoiseScheduler

_CONTINUOUS_DIM = D_POSE + Z_DIM


class _SinusoidalTimeEmbedding(nn.Module):
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


class SingleViewSceneDiffusion(nn.Module):
    """Paper mainline: direct visible reconstruction + conditional hidden diffusion.

    The module is the composition of:
    - an observation backbone (patch ViT / DINOv2 / DINOv2-hybrid),
    - a deterministic transformer-decoder head over visible slots,
    - a conditional transformer-decoder denoiser over hidden slots,
    - a relation head for floor/wall/support predictions,
    - a noise scheduler + DDIM sampler.

    The head architectures here mirror the v3/v4 checkpoint layout so existing
    weights are load-compatible. Newer v5-class heads (DiT, DETR) live under
    `amodal_scene_diff.heads` and are swapped in via config.
    """

    def __init__(
        self,
        *,
        obs_channels: int = 4,
        image_size: int = 512,
        patch_size: int = 32,
        d_model: int = D_MODEL,
        encoder_layers: int = 8,
        decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        observation_backbone_cfg: dict[str, Any] | None = None,
        layout_weight: float = 1.0,
        visible_weight: float = 1.0,
        visible_exist_weight: float = 0.5,
        visible_cls_weight: float = 0.5,
        hidden_weight: float = 1.0,
        hidden_exist_weight: float = 1.0,
        hidden_cls_weight: float = 1.0,
        support_weight: float = 0.5,
        floor_weight: float = 0.25,
        wall_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.scheduler = NoiseScheduler(
            train_timesteps=train_timesteps,
            schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.scheduler.register_buffers(self)

        self.layout_weight = float(layout_weight)
        self.visible_weight = float(visible_weight)
        self.visible_exist_weight = float(visible_exist_weight)
        self.visible_cls_weight = float(visible_cls_weight)
        self.hidden_weight = float(hidden_weight)
        self.hidden_exist_weight = float(hidden_exist_weight)
        self.hidden_cls_weight = float(hidden_cls_weight)
        self.support_weight = float(support_weight)
        self.floor_weight = float(floor_weight)
        self.wall_weight = float(wall_weight)

        self.observation_encoder = build_observation_backbone(
            obs_channels=obs_channels,
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            backbone_cfg=observation_backbone_cfg,
        )
        self.source_embedding = nn.Embedding(3, d_model)
        self.time_embedding = _SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        visible_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visible_decoder = nn.TransformerDecoder(visible_decoder_layer, num_layers=decoder_layers)
        self.visible_queries = nn.Parameter(torch.randn(K_VIS, d_model) * 0.02)

        hidden_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hidden_decoder = nn.TransformerDecoder(hidden_decoder_layer, num_layers=decoder_layers)
        self.hidden_queries = nn.Parameter(torch.randn(K_HID, d_model) * 0.02)

        self.layout_token_proj = nn.Linear(D_POSE, d_model)
        self.visible_state_in = nn.Sequential(nn.LayerNorm(_CONTINUOUS_DIM), nn.Linear(_CONTINUOUS_DIM, d_model))
        self.hidden_state_in = nn.Sequential(nn.LayerNorm(_CONTINUOUS_DIM), nn.Linear(_CONTINUOUS_DIM, d_model))
        self.visible_presence_proj = nn.Linear(1, d_model)

        self.global_norm = nn.LayerNorm(d_model)
        self.object_norm = nn.LayerNorm(d_model)

        self.layout_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, D_POSE),
        )
        self.visible_state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, _CONTINUOUS_DIM),
        )
        self.visible_exist_head = nn.Linear(d_model, 1)
        self.visible_cls_head = nn.Linear(d_model, C_OBJ)

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

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SingleViewSceneDiffusion":
        model_cfg = cfg["model"]
        noise_cfg = cfg.get("noise", {})
        loss_cfg = cfg.get("loss", {})
        return cls(
            obs_channels=int(model_cfg.get("obs_channels", 4)),
            image_size=int(cfg["data"].get("image_size", model_cfg.get("image_size", 512))),
            patch_size=int(model_cfg.get("patch_size", 32)),
            d_model=int(model_cfg.get("d_model", 512)),
            encoder_layers=int(model_cfg.get("encoder_layers", 8)),
            decoder_layers=int(model_cfg.get("decoder_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_ratio=float(model_cfg.get("ffn_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            train_timesteps=int(noise_cfg.get("train_timesteps", 1000)),
            beta_schedule=str(noise_cfg.get("beta_schedule", "cosine")),
            prediction_type=str(noise_cfg.get("prediction_type", "v_prediction")),
            observation_backbone_cfg=model_cfg.get("observation_backbone"),
            layout_weight=float(loss_cfg.get("lambda_layout", 1.0)),
            visible_weight=float(loss_cfg.get("lambda_visible_state", 1.0)),
            visible_exist_weight=float(loss_cfg.get("lambda_visible_exist", 0.5)),
            visible_cls_weight=float(loss_cfg.get("lambda_visible_cls", 0.5)),
            hidden_weight=float(loss_cfg.get("lambda_hidden_state", 1.0)),
            hidden_exist_weight=float(loss_cfg.get("lambda_hidden_exist", 1.0)),
            hidden_cls_weight=float(loss_cfg.get("lambda_hidden_cls", 1.0)),
            support_weight=float(loss_cfg.get("lambda_support", 0.5)),
            floor_weight=float(loss_cfg.get("lambda_floor", 0.25)),
            wall_weight=float(loss_cfg.get("lambda_wall", 0.25)),
        )

    def continuous_state_targets(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        target = batch.target
        return {
            "layout": target.layout_gt,
            "visible": torch.cat([target.visible_amodal_pose_gt, target.visible_z_gt], dim=-1),
            "hidden": torch.cat([target.hidden_pose_gt, target.hidden_z_gt], dim=-1),
        }

    def encode_observation(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        encoded = self.observation_encoder(batch)
        source_bias = self.source_embedding(batch.cond.source_id.long())
        global_token = self.global_norm(encoded["global_token"] + source_bias)
        patch_tokens = encoded["patch_tokens"] + source_bias.unsqueeze(1)
        return {"global_token": global_token, "patch_tokens": patch_tokens}

    def decode_visible(
        self,
        batch: SingleViewSceneBatch,
        observation_ctx: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        global_token = observation_ctx["global_token"]
        patch_tokens = observation_ctx["patch_tokens"]
        memory = torch.cat([global_token.unsqueeze(1), patch_tokens], dim=1)
        queries = self.visible_queries.unsqueeze(0).expand(batch.batch_size, -1, -1)
        visible_ctx = self.visible_decoder(queries, memory)
        layout_pred = self.layout_head(global_token)
        visible_state = self.visible_state_head(visible_ctx)
        visible_exist_logits = self.visible_exist_head(visible_ctx).squeeze(-1)
        visible_cls_logits = self.visible_cls_head(visible_ctx)
        visible_presence = torch.sigmoid(visible_exist_logits).unsqueeze(-1)
        visible_anchor_tokens = (
            visible_ctx
            + self.visible_state_in(visible_state)
            + self.visible_presence_proj(visible_presence)
        )
        return {
            "layout_pred": layout_pred,
            "visible_ctx": visible_ctx,
            "visible_anchor_tokens": visible_anchor_tokens,
            "visible_state": visible_state,
            "visible_exist_logits": visible_exist_logits,
            "visible_cls_logits": visible_cls_logits,
        }

    def forward_hidden_denoiser(
        self,
        batch: SingleViewSceneBatch,
        *,
        observation_ctx: dict[str, torch.Tensor],
        visible_ctx: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        hidden_xt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_token = observation_ctx["global_token"]
        patch_tokens = observation_ctx["patch_tokens"]
        layout_token = self.layout_token_proj(visible_ctx["layout_pred"]).unsqueeze(1)
        memory = torch.cat(
            [
                global_token.unsqueeze(1),
                layout_token,
                visible_ctx["visible_anchor_tokens"],
                patch_tokens,
            ],
            dim=1,
        )
        time_bias = self.time_mlp(self.time_embedding(timesteps)).unsqueeze(1)
        hidden_queries = self.hidden_queries.unsqueeze(0) + self.hidden_state_in(hidden_xt) + time_bias
        hidden_ctx = self.hidden_decoder(hidden_queries, memory)
        object_tokens = torch.cat([visible_ctx["visible_anchor_tokens"], hidden_ctx], dim=1)
        relation_q = self.relation_q(object_tokens)
        relation_k = self.relation_k(object_tokens)
        support_logits = torch.einsum("bid,bjd->bij", relation_q, relation_k) / math.sqrt(self.d_model)
        return {
            "hidden_ctx": hidden_ctx,
            "hidden_pred": self.hidden_state_head(hidden_ctx),
            "hidden_exist_logits": self.hidden_exist_head(hidden_ctx).squeeze(-1),
            "hidden_cls_logits": self.hidden_cls_head(hidden_ctx),
            "support_logits": support_logits,
            "floor_logits": self.floor_head(object_tokens).squeeze(-1),
            "wall_logits": self.wall_head(object_tokens).squeeze(-1),
        }

    def _class_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.bool()
        if not bool(valid.any()):
            return torch.zeros((), device=logits.device)
        return F.cross_entropy(logits[valid], labels[valid])

    def compute_losses(self, batch: SingleViewSceneBatch) -> dict[str, torch.Tensor]:
        device = batch.cond.obs_image.device
        batch_size = batch.batch_size
        target = batch.target
        states = self.continuous_state_targets(batch)

        observation_ctx = self.encode_observation(batch)
        visible_ctx = self.decode_visible(batch, observation_ctx)

        layout_loss = F.mse_loss(visible_ctx["layout_pred"], states["layout"])

        visible_mask = target.visible_loss_mask.float().unsqueeze(-1)
        visible_mask_2d = target.visible_loss_mask.float()
        visible_state_loss = (
            ((visible_ctx["visible_state"] - states["visible"]) ** 2 * visible_mask).sum()
            / visible_mask.sum().clamp_min(1.0)
        )
        visible_exist_loss = F.binary_cross_entropy_with_logits(visible_ctx["visible_exist_logits"], visible_mask_2d)
        visible_cls_loss = self._class_loss(
            visible_ctx["visible_cls_logits"], target.visible_cls_gt, target.visible_loss_mask
        )

        hidden_x0 = states["hidden"]
        hidden_mask = target.hidden_gt_mask.float().unsqueeze(-1)
        hidden_mask_2d = target.hidden_gt_mask.float()
        timesteps = torch.randint(0, self.scheduler.train_timesteps, (batch_size,), device=device, dtype=torch.long)
        hidden_noise = torch.randn_like(hidden_x0)
        hidden_xt = self.scheduler.q_sample(
            x0=hidden_x0,
            noise=hidden_noise,
            timesteps=timesteps,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        ) * hidden_mask

        hidden_pred = self.forward_hidden_denoiser(
            batch,
            observation_ctx=observation_ctx,
            visible_ctx=visible_ctx,
            timesteps=timesteps,
            hidden_xt=hidden_xt,
        )
        hidden_target = self.scheduler.prediction_target(
            x0=hidden_x0,
            noise=hidden_noise,
            timesteps=timesteps,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
        )
        hidden_loss = (
            ((hidden_pred["hidden_pred"] - hidden_target) ** 2 * hidden_mask).sum()
            / hidden_mask.sum().clamp_min(1.0)
        )
        hidden_exist_loss = F.binary_cross_entropy_with_logits(hidden_pred["hidden_exist_logits"], hidden_mask_2d)
        hidden_cls_loss = self._class_loss(hidden_pred["hidden_cls_logits"], target.hidden_cls_gt, target.hidden_gt_mask)

        relation_valid = target.relation_valid_mask.float()
        pair_mask = relation_valid.unsqueeze(1) * relation_valid.unsqueeze(2)
        support_loss = F.binary_cross_entropy_with_logits(
            hidden_pred["support_logits"],
            target.support_gt.float(),
            weight=pair_mask,
            reduction="sum",
        ) / pair_mask.sum().clamp_min(1.0)
        floor_loss = F.binary_cross_entropy_with_logits(
            hidden_pred["floor_logits"],
            target.floor_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)
        wall_loss = F.binary_cross_entropy_with_logits(
            hidden_pred["wall_logits"],
            target.wall_gt.float(),
            weight=relation_valid,
            reduction="sum",
        ) / relation_valid.sum().clamp_min(1.0)

        total = (
            self.layout_weight * layout_loss
            + self.visible_weight * visible_state_loss
            + self.visible_exist_weight * visible_exist_loss
            + self.visible_cls_weight * visible_cls_loss
            + self.hidden_weight * hidden_loss
            + self.hidden_exist_weight * hidden_exist_loss
            + self.hidden_cls_weight * hidden_cls_loss
            + self.support_weight * support_loss
            + self.floor_weight * floor_loss
            + self.wall_weight * wall_loss
        )
        return {
            "loss_total": total,
            "loss_layout_direct": layout_loss,
            "loss_visible_direct": visible_state_loss,
            "loss_visible_exist": visible_exist_loss,
            "loss_visible_cls": visible_cls_loss,
            "loss_hidden_diff": hidden_loss,
            "loss_hidden_exist": hidden_exist_loss,
            "loss_hidden_cls": hidden_cls_loss,
            "loss_support": support_loss,
            "loss_floor": floor_loss,
            "loss_wall": wall_loss,
            "metric_t_mean": timesteps.float().mean(),
        }

    @torch.no_grad()
    def sample_posterior(self, batch: SingleViewSceneBatch, num_sampling_steps: int = 50) -> dict[str, torch.Tensor]:
        device = batch.cond.obs_image.device
        batch_size = batch.batch_size
        observation_ctx = self.encode_observation(batch)
        visible_ctx = self.decode_visible(batch, observation_ctx)

        def denoiser_step(xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.forward_hidden_denoiser(
                batch,
                observation_ctx=observation_ctx,
                visible_ctx=visible_ctx,
                timesteps=t,
                hidden_xt=xt,
            )["hidden_pred"]

        def prediction_to_x0_and_eps(prediction: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.scheduler.prediction_to_x0_and_eps(
                prediction=prediction,
                xt=xt,
                timesteps=t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            )

        hidden_x0 = sample_ddim_posterior(
            x_shape=(batch_size, K_HID, _CONTINUOUS_DIM),
            device=device,
            train_timesteps=self.scheduler.train_timesteps,
            sampling_steps=num_sampling_steps,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            denoiser_step=denoiser_step,
            prediction_to_x0_and_eps=prediction_to_x0_and_eps,
        )

        final_t = torch.zeros((batch_size,), device=device, dtype=torch.long)
        final_pred = self.forward_hidden_denoiser(
            batch,
            observation_ctx=observation_ctx,
            visible_ctx=visible_ctx,
            timesteps=final_t,
            hidden_xt=hidden_x0,
        )

        visible_exist_probs = torch.sigmoid(visible_ctx["visible_exist_logits"])
        hidden_exist_probs = torch.sigmoid(final_pred["hidden_exist_logits"])
        return {
            "layout": visible_ctx["layout_pred"],
            "visible": visible_ctx["visible_state"],
            "visible_exist_logits": visible_ctx["visible_exist_logits"],
            "visible_exist_probs": visible_exist_probs,
            "visible_cls_logits": visible_ctx["visible_cls_logits"],
            "visible_cls_probs": torch.softmax(visible_ctx["visible_cls_logits"], dim=-1),
            "hidden": hidden_x0,
            "hidden_exist_logits": final_pred["hidden_exist_logits"],
            "hidden_exist_probs": hidden_exist_probs,
            "hidden_cls_logits": final_pred["hidden_cls_logits"],
            "hidden_cls_probs": torch.softmax(final_pred["hidden_cls_logits"], dim=-1),
            "support_logits": final_pred["support_logits"],
            "floor_logits": final_pred["floor_logits"],
            "wall_logits": final_pred["wall_logits"],
        }
