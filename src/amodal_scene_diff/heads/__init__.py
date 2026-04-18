"""Set-decoding heads.

- `detr_visible`: variable-slot visible reconstruction head with a Hungarian
  matcher. Replaces the fixed K_VIS=12 transformer-decoder head when enabled.
- `dit_hidden`: DiT-class (AdaLN-Zero) hidden denoiser, drop-in for the
  baseline transformer-decoder denoiser.
- `layout`, `relation`: shared scene-level heads used by both the v3/v4
  baselines and the v5 stack.
"""

from .dit_hidden import DiTHiddenDenoiser
from .detr_visible import DetrVisibleHead

__all__ = ["DetrVisibleHead", "DiTHiddenDenoiser"]
