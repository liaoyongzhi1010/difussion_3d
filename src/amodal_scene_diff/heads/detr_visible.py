"""Placeholder for the DETR-style variable-slot visible head.

Populated in the step-11 commit.
"""

from __future__ import annotations

from torch import nn


class DetrVisibleHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - placeholder
        super().__init__()
        raise NotImplementedError("DetrVisibleHead is added in a follow-up commit")
