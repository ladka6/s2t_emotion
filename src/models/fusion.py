# src/model/fusion.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .config import FusionConfig


class CrossAttentionFusion(nn.Module):
    """
    Injects paralinguistic context z_p into every semantic timestep of H_sem
    using a lightweight multi-head cross-attention.

    Inputs
    ------
    - H_sem: float32 tensor [B, T, d_sem]       (from Whisper encoder; d_sem must equal cfg.d_semantic)
    - z_p:   float32 tensor [B, d_para]         (from ParalinguisticEncoder; d_para == cfg.d_para)

    Outputs
    -------
    - H_tilde: float32 tensor [B, T, d_sem]     (same time length as H_sem)

    Design
    ------
    - Queries (Q) are derived from H_sem (per-timestep).
    - Keys/Values (K,V) are derived from z_p, then broadcast across time.
      * You can treat z_p as a single token per batch, or expand to N_p pseudo-tokens
        via a small learned expansion (optional).
    - Multi-head attention → residual connection → dropout → LayerNorm.

    Notes
    -----
    - Keep d_sem % n_heads == 0 (config validator enforces this).
    - Apply dropout to attention outputs, not to inputs.
    - Keep this module agnostic to any LLM specifics; it only outputs Ĥ.
    """

    def __init__(self, cfg: FusionConfig, num_pseudo_tokens: int = 1):
        """
        Define:
        - Linear projections:
            W_q: d_sem -> d_sem
            W_k: d_para -> d_sem  (then optionally reshape to [B, n_heads, N_p, d_head])
            W_v: d_para -> d_sem
            out: d_sem -> d_sem
        - LayerNorm over d_sem
        - Dropout (cfg.dropout)
        - Optional learned expansion from z_p -> Z_pseudo [B, N_p, d_para] if num_pseudo_tokens > 1

        Store shapes:
        - n_heads: cfg.n_heads
        - d_head:  cfg.d_semantic // cfg.n_heads
        - d_sem:   cfg.d_semantic
        - d_para:  cfg.d_para
        """
        super().__init__()
        self.cfg = cfg
        self.num_pseudo_tokens = int(num_pseudo_tokens)
        # TODO: define projections, norm, drop, optional expansion

    def forward(self, H_sem: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """
        Compute Ĥ = H_sem + MHA( Q=H_sem , K=expand(z_p), V=expand(z_p) )

        Steps
        -----
        1) Validate shapes:
           - H_sem.ndim == 3, H_sem.size(-1) == d_sem
           - z_p.ndim == 2, z_p.size(-1) == d_para
        2) Optionally expand z_p to N_p pseudo tokens:
           - If num_pseudo_tokens == 1: K,V from a single token per batch.
           - Else: pass through a tiny expansion module to [B, N_p, d_para].
        3) Project:
           - Q = W_q(H_sem)         → [B, T, d_sem]
           - K = W_k(Z_pseudo)      → [B, N_p, d_sem]
           - V = W_v(Z_pseudo)      → [B, N_p, d_sem]
        4) Split heads & compute attention:
           - Reshape Q,K,V to [B, nH, T_or_N, d_head]
           - scores = (Q ⋅ K^T) / sqrt(d_head)
           - attn = softmax(scores, dim=-1)
           - ctx = attn ⋅ V         → [B, nH, T, d_head]
           - merge heads back → [B, T, d_sem]
        5) Output:
           - O = out(ctx)
           - Ĥ = LayerNorm( H_sem + Dropout(O) )
        6) Return Ĥ
        """
        # TODO: implement attention math and residual
        raise NotImplementedError

    # ----------------------------
    # Utilities / diagnostics
    # ----------------------------

    def extra_loss_terms(self, H_tilde: torch.Tensor) -> torch.Tensor:
        """
        (Optional) Add tiny regularizers to keep Ĥ stable:
        - L2 penalty on output magnitude
        - Temporal smoothness on consecutive steps
        Return a scalar tensor (or 0.0 * H_tilde.sum()) for trainer to add with a lambda.
        """
        # TODO: implement or return zero
        raise NotImplementedError

    def output_shape(self, T_in: int, B: int = 1) -> tuple[int, int, int]:
        """
        Predict output shape given input time length T_in and batch size B.
        Returns (B, T_in, d_sem).
        """
        # TODO: return (B, T_in, self.cfg.d_semantic)
        raise NotImplementedError
