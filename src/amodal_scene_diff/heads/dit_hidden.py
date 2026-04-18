"""Placeholder for the DiT-class AdaLN-Zero hidden denoiser.

Populated in the step-11 commit; import is wired now so the tree is
installable and the v5 scaffolding type-checks.
"""

from __future__ import annotations

from torch import nn


class DiTHiddenDenoiser(nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - placeholder
        super().__init__()
        raise NotImplementedError("DiTHiddenDenoiser is added in a follow-up commit")
