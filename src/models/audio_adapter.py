# src/model/audio_adapter.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn

from .config import AudioAdapterConfig


class AudioConditioningAdapter(nn.Module):
    """
    Converts fused acoustic features into a form the decoder-only LLM can consume.

    Modes
    -----
    1) "soft_tokens"  (D1)  → produce N learned audio token *embeddings* A_tok ∈ ℝ[B, N, d_llm]
       - Concatenate with text token embeddings along time dimension.
       - Requires building `inputs_embeds` for the LLM (instead of only `input_ids`).
       - You MUST extend attention_mask and labels: prepend N positions with 1 in mask and -100 in labels.

    2) "prefix_kv"    (D2)  → produce per-layer KV cache from z_p / pooled Ĥ
       - No extra tokens; supply `past_key_values` to the LLM call.
       - Attention mask and labels are unchanged (pure text).
       - You must build tensors shaped like L × (k,v) each of [B, nH, N_pref, d_head].

    3) "side_attn"    (D3)  → expose audio as cross-attention keys/values (requires LLM surgery)
       - Out of scope for a first pass; leave stubs that return NotImplemented.

    Inputs
    ------
    - H_tilde: [B, T_sem, d_sem]  (fused features from fusion step)
    - sem_mask: [B, T_sem]        (validity)
    - z_p:      [B, d_para]       (optional, may be concatenated with pooled H)

    Outputs
    -------
    - For "soft_tokens":
        A_tok: [B, N_tok, d_llm]
      plus helper to build LLM kwargs given tokenizer outputs and labels.

    - For "prefix_kv":
        past_key_values: List[Tuple[K,V] per layer]
      plus passthrough of original text tensors.

    - For "side_attn":
        placeholder stubs (document how you'd integrate).
    """

    def __init__(
        self,
        cfg: AudioAdapterConfig,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_head: Optional[int] = None,
    ):
        """
        Define members based on cfg.mode:

        Common:
        - store cfg, n_layers, n_heads, d_head (needed for prefix_kv).

        Soft tokens:
        - pooler: reduces variable T_sem to fixed N_tok
            * "attn": additive attention pooling to N_tok learnable queries
            * "mean": masked mean, then a small upsampler to N_tok (e.g., MLP to N_tok×d_sem reshaped)
            * "conv": depthwise 1D conv stride to compress to ≈N_tok, then crop/pad
        - proj_sem_to_llm: Linear(d_sem -> d_llm) (if d_sem != d_llm)
        - optional fusion with z_p: concat([pooled_H, z_p]) followed by Linear to d_llm

        Prefix-KV:
        - pooler: to a fixed conditioning vector C ∈ ℝ[B, d_cond] (e.g., mean/attn over H_tilde with mask)
        - per-layer generators: map C → K,V for each layer with shapes [B, nH, N_pref, d_head]
          (either shared across layers or layer-specific; start with shared)
        - N_pref: small (e.g., 8)

        Side-attn:
        - stubs only; document that you'd project H_tilde to K,V and register in the LLM blocks.
        """
        super().__init__()
        self.cfg = cfg
        # TODO: define modules per mode (poolers, projections, kv generators)

    # ---------------------------
    # Mode dispatch (public API)
    # ---------------------------

    def forward_soft_tokens(
        self,
        H_tilde: torch.Tensor,
        sem_mask: torch.Tensor,
        z_p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build N audio tokens for LLM input_embeds.

        Steps to implement:
        1) Mask-aware pooling over time to fixed N_tok (B, N_tok, d_sem).
        2) Optional conditioning fusion with z_p (concat + Linear) or FiLM-like scaling.
        3) Project to d_llm (if needed) → A_tok [B, N_tok, d_llm].
        4) Return A_tok.
        """
        # TODO
        raise NotImplementedError

    def forward_prefix_kv(
        self,
        H_tilde: torch.Tensor,
        sem_mask: torch.Tensor,
        z_p: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build per-layer KV cache.

        Steps:
        1) Pool H_tilde (mask-aware) to conditioning vector C [B, d_cond] (e.g., d_cond=d_sem or d_cond=d_sem+d_para).
        2) Map C -> K_l, V_l for each layer l:
           - K_l, V_l shapes: [B, nH, N_pref, d_head]
        3) Return list length n_layers: [(K_0, V_0), ..., (K_L-1, V_L-1)]
        """
        # TODO
        raise NotImplementedError

    def forward_side_attn(
        self, H_tilde: torch.Tensor, sem_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Placeholder describing how to attach audio as cross-attn:
        - Project H_tilde to K,V per layer and insert via custom LLM wrapper.
        - Return dict of buffers keyed by layer index.
        """
        # TODO
        raise NotImplementedError

    # ------------------------------------------
    # Helpers for integrating with HF LLM calls
    # ------------------------------------------

    def build_llm_inputs_with_soft_tokens(
        self,
        llm: nn.Module,
        text_input_ids: torch.Tensor,  # [B, T_txt]
        text_attention_mask: torch.Tensor,  # [B, T_txt]
        labels: Optional[torch.Tensor],  # [B, T_txt]
        A_tok: torch.Tensor,  # [B, N_tok, d_llm]
    ) -> Dict[str, Any]:
        """
        Prepare kwargs for LLM.forward when using soft tokens.

        You will:
        - Lookup text embeddings using llm.get_input_embeddings()(text_input_ids) → [B, T_txt, d_llm].
        - Concatenate: inputs_embeds = cat([A_tok, text_embeds], dim=1).
        - Extend attention_mask by prepending ones for A_tok: [B, N_tok + T_txt].
        - Extend labels by prepending -100 for A_tok positions (if labels is provided).
        - Do NOT pass input_ids when passing inputs_embeds.

        Return a dict suitable for: llm(**kwargs)
          keys: {"inputs_embeds", "attention_mask", ("labels" if training)}
        """
        # TODO
        raise NotImplementedError

    def build_llm_inputs_with_prefix_kv(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, Any]:
        """
        Prepare kwargs for LLM.forward when using prefix-KV conditioning.

        You will:
        - Keep text_input_ids, attention_mask, labels unchanged.
        - Add "past_key_values": the list[(K,V)*L].
        - Most HF models accept this and will prepend the KV before attending to tokens.

        Return dict with keys: {"input_ids", "attention_mask", "past_key_values", ("labels")}
        """
        # TODO
        raise NotImplementedError

    # -----------------
    # Utility methods
    # -----------------

    def freeze(self) -> None:
        """Set requires_grad=False (e.g., after Stage-1)."""
        # TODO
        raise NotImplementedError

    def unfreeze(self) -> None:
        """Set requires_grad=True."""
        # TODO
        raise NotImplementedError

    def output_shapes_soft_tokens(
        self, B: int, N_tok: Optional[int] = None, d_llm: Optional[int] = None
    ) -> tuple[int, int, int]:
        """
        Return (B, N_tok, d_llm) for logging/tests.
        """
        # TODO
        raise NotImplementedError
