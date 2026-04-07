from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GeometryVAEOutput:
    z_mu: torch.Tensor
    z_logvar: torch.Tensor
    z_sample: torch.Tensor
    triplanes: torch.Tensor
    sdf_pred: torch.Tensor
    occ_logits: torch.Tensor


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, surface_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, points, channels = surface_inputs.shape
        feats = self.backbone(surface_inputs.reshape(batch * points, channels)).reshape(batch, points, -1)
        pooled = feats.max(dim=1).values
        return self.mu_head(pooled), self.logvar_head(pooled)


class TriPlaneDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        triplane_feat_dim: int,
        triplane_res_xy: int,
        query_hidden_dims: list[int],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.triplane_feat_dim = triplane_feat_dim
        self.triplane_res_xy = triplane_res_xy
        self.num_planes = 3
        triplane_dim = self.num_planes * triplane_feat_dim * triplane_res_xy * triplane_res_xy
        self.triplane_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, triplane_dim),
        )
        mlp_dims = [self.num_planes * triplane_feat_dim + 3] + query_hidden_dims
        mlp = []
        for in_dim, out_dim in zip(mlp_dims[:-1], mlp_dims[1:]):
            mlp.append(nn.Linear(in_dim, out_dim))
            mlp.append(nn.SiLU())
        self.query_mlp = nn.Sequential(*mlp)
        self.sdf_head = nn.Linear(query_hidden_dims[-1], 1)
        self.occ_head = nn.Linear(query_hidden_dims[-1], 1)

    def _sample_plane(self, plane_feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grid = coords.unsqueeze(2)
        sampled = F.grid_sample(
            plane_feats,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def _sample_triplanes(self, triplanes: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        xy = query_points[..., [0, 1]].clamp(-1.0, 1.0)
        xz = query_points[..., [0, 2]].clamp(-1.0, 1.0)
        yz = query_points[..., [1, 2]].clamp(-1.0, 1.0)
        plane_xy = self._sample_plane(triplanes[:, 0], xy)
        plane_xz = self._sample_plane(triplanes[:, 1], xz)
        plane_yz = self._sample_plane(triplanes[:, 2], yz)
        return torch.cat([plane_xy, plane_xz, plane_yz], dim=-1)

    def forward(self, z_sample: torch.Tensor, query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, queries, _ = query_points.shape
        triplanes = self.triplane_head(z_sample).view(
            batch,
            self.num_planes,
            self.triplane_feat_dim,
            self.triplane_res_xy,
            self.triplane_res_xy,
        )
        query_feats = self._sample_triplanes(triplanes, query_points)
        query_inputs = torch.cat([query_feats, query_points], dim=-1)
        hidden = self.query_mlp(query_inputs)
        sdf_pred = self.sdf_head(hidden)
        occ_logits = self.occ_head(hidden)
        return triplanes, sdf_pred, occ_logits


class GeometryVAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,
        latent_dim: int = 256,
        encoder_hidden_dims: list[int] | None = None,
        triplane_feat_dim: int = 16,
        triplane_res_xy: int = 32,
        query_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        encoder_hidden_dims = encoder_hidden_dims or [64, 128, 256, 512]
        query_hidden_dims = query_hidden_dims or [256, 256, 128]
        self.encoder = PointNetEncoder(input_dim=input_dim, hidden_dims=encoder_hidden_dims, latent_dim=latent_dim)
        self.decoder = TriPlaneDecoder(
            latent_dim=latent_dim,
            triplane_feat_dim=triplane_feat_dim,
            triplane_res_xy=triplane_res_xy,
            query_hidden_dims=query_hidden_dims,
        )

    def reparameterize(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, surface_points: torch.Tensor, surface_normals: torch.Tensor, query_points: torch.Tensor) -> GeometryVAEOutput:
        surface_inputs = torch.cat([surface_points, surface_normals], dim=-1)
        z_mu, z_logvar = self.encoder(surface_inputs)
        z_sample = self.reparameterize(z_mu, z_logvar)
        triplanes, sdf_pred, occ_logits = self.decoder(z_sample, query_points)
        return GeometryVAEOutput(
            z_mu=z_mu,
            z_logvar=z_logvar,
            z_sample=z_sample,
            triplanes=triplanes,
            sdf_pred=sdf_pred,
            occ_logits=occ_logits,
        )

    def compute_losses(
        self,
        *,
        surface_points: torch.Tensor,
        surface_normals: torch.Tensor,
        query_points: torch.Tensor,
        query_sdf: torch.Tensor,
        query_occ: torch.Tensor,
        lambda_kl: float,
        lambda_sdf: float,
        lambda_occ: float,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(surface_points, surface_normals, query_points)
        sdf_loss = F.l1_loss(out.sdf_pred, query_sdf)
        occ_loss = F.binary_cross_entropy_with_logits(out.occ_logits, query_occ)
        kl_loss = 0.5 * torch.mean(torch.exp(out.z_logvar) + out.z_mu ** 2 - 1.0 - out.z_logvar)
        total = lambda_sdf * sdf_loss + lambda_occ * occ_loss + lambda_kl * kl_loss
        return {
            "loss_total": total,
            "loss_sdf": sdf_loss,
            "loss_occ": occ_loss,
            "loss_kl": kl_loss,
        }

    @property
    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())
