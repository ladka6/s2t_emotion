# src/model/para_encoder.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .config import ParalinguisticConfig


class ParalinguisticEncoder(nn.Module):
    """
    Projects your precomputed paralinguistic vector (emotion+speaker) into a stable,
    model-friendly latent z_p used by fusion/adapter.

    Inputs
    ------
    - combined_embedding: float32 tensor [B, d_in] (your dataset: d_in = 960)
      * typically concatenation of emotion (≈768) + speaker (≈192), but agnostic here.

    Outputs
    -------
    - z_p: float32 tensor [B, d_proj]   (use cfg.d_proj, e.g., 256)

    Responsibilities
    ----------------
    1) Normalize input scale (LayerNorm recommended).
    2) Small projection MLP to d_proj with a nonlinearity (GELU/Tanh).
    3) Optional dropout for regularization.
    4) Keep a simple API and utility methods to freeze/unfreeze during stages.
    """

    def __init__(self, cfg: ParalinguisticConfig):
        """
        Define:
        - self.norm: normalization over d_in (LayerNorm)
        - self.mlp: sequence of Linear(d_in->d_proj), activation, Dropout, Linear(d_proj->d_proj)
        - store cfg for logging
        """
        super().__init__()
        self.cfg = cfg
        # TODO: define self.norm, self.mlp

    def forward(self, combined_embedding: torch.Tensor) -> torch.Tensor:
        """
        Steps:
        1) Validate shape: combined_embedding.ndim == 2 and size(1) == cfg.d_in
        2) Apply self.norm
        3) Apply self.mlp → z_p [B, d_proj]
        4) Return z_p
        """
        # TODO: implement forward pass
        raise NotImplementedError

    def freeze(self) -> None:
        """
        Set requires_grad=False for all parameters (Stage-2 if you want to keep it fixed).
        """
        # TODO: loop and freeze
        raise NotImplementedError

    def unfreeze(self) -> None:
        """
        Set requires_grad=True for all parameters.
        """
        # TODO: loop and unfreeze
        raise NotImplementedError

    def output_shape(self, batch_size: int = 1) -> Tuple[int, int]:
        """
        Return (B, d_proj) given batch_size (for logging/tests).
        """
        # TODO: return (batch_size, self.cfg.d_proj)
        raise NotImplementedError
