# src/model/semantic_encoder.py
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn

from .config import SemanticEncoderConfig


class WhisperSemanticEncoder(nn.Module):
    """
    Thin wrapper around Hugging Face Whisper (encoder-only) used as a *frozen* semantic encoder.

    Goals
    -----
    - Accept collator-produced log-mel features [B, T_mel, 80].
    - Reformat/normalize to the shape Whisper expects (usually [B, 80, T_mel]).
    - Run the official Whisper *encoder* to obtain hidden states.
    - Return final (or pooled) encoder states as H_sem [B, T_sem, d_sem] and
      an aligned downsampled mask sem_mask [B, T_sem].
    - Keep the option to:
        * select which encoder layer(s) to extract,
        * average/concat multiple layers,
        * enable gradient checkpointing if later fine-tuning (off by default).

    Important Notes
    ---------------
    - Whisper's encoder downsamples time by x4 (two conv layers, stride=2 each), so:
        T_sem ≈ floor(T_mel / 4)
      (Exact length depends on padding; you should compute it from the model output.)
    - d_sem must match cfg.d_hidden (for Whisper-base, it's 768).
    - You usually **freeze** this module for Stage-1; only unfreeze in Stage-2 if needed.
    """

    def __init__(
        self,
        cfg: SemanticEncoderConfig,
        model_name_or_path: str = "openai/whisper-base",
        output_layers: Optional[List[int]] = None,
        layer_aggregation: str = "last",  # {"last", "mean", "concat"} (document behavior)
        use_feature_extractor_norm: bool = True,
        trust_remote_code: bool = False,
    ):
        """
        Construct and hold a WhisperModel (or WhisperEncoder) from 🤗 Transformers.

        You will:
        - import and instantiate transformers.WhisperModel.from_pretrained(...)
        - keep only the encoder (model.model.encoder) or call model.forward and grab encoder states
        - assert model.config.d_model == cfg.d_hidden (expected 768 for base)
        - set eval() and requires_grad=False initially (frozen backbone)
        - optionally enable gradient checkpointing if later unfreezing
        - store output layer indices and aggregation strategy

        Args
        ----
        cfg:              SemanticEncoderConfig with in_mels=80, d_hidden=768 (Whisper-base), stride≈4 (informational)
        model_name_or_path: HF id or local path for Whisper-base
        output_layers:    Which encoder layer indices to extract (e.g., [-1] for last).
                          If None, default to [-1].
        layer_aggregation: How to aggregate multiple layers:
                           - "last": take last selected layer
                           - "mean": average across selected layers
                           - "concat": concat across feature dim (then project back to d_hidden if needed)
        use_feature_extractor_norm: Whether to apply Whisper feature-extractor’s per-channel mean/var normalization
                                    to collator log-mels (recommended for compatibility).
        trust_remote_code: passed through to HF if you use custom forks
        """
        super().__init__()
        self.cfg = cfg
        # TODO:
        # - Load Whisper model
        # - Keep pointers to encoder, config, proj (if needed for "concat")
        # - Save args (output_layers, layer_aggregation, use_feature_extractor_norm)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def forward(
        self,
        mel: torch.Tensor,  # [B, T_mel, 80] from collator
        mel_mask: Optional[torch.Tensor] = None,  # [B, T_mel] bool/int or None
        return_all_layers: bool = False,  # if True, return list of layer states + mask
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Run Whisper encoder on collator features and return semantic sequence + mask.

        Steps to implement
        ------------------
        1) Validate input:
           - dtype float32/bfloat16
           - mel.ndim == 3 and mel.size(-1) == 80
        2) Reformat for Whisper:
           - Whisper expects [B, 80, T_mel]
           - Optionally apply the same normalization used by WhisperFeatureExtractor
             (mean/var per channel; *not* dB conversion—collator already did log-mel)
        3) Forward pass through Whisper encoder:
           - obtain hidden states per layer if needed (set output_hidden_states=True)
           - determine T_sem from the output instead of guessing
        4) Select/aggregate layers:
           - If return_all_layers: produce a list [B, T_sem, d_sem] for each chosen layer
           - Else: produce a single [B, T_sem, d_sem] tensor per 'layer_aggregation'
             ("last" | "mean" | "concat"); for "concat", add a linear projection back to d_sem if needed
        5) Build sem_mask aligned to T_sem:
           - Use `_downsample_mask_whisper(mel_mask, T_mel, T_sem)` (defined below)
           - If mel_mask is None, make a full-True mask of length T_sem
        6) Return:
           - (H_sem, sem_mask) or ([H_l0, H_l1, ...], sem_mask) if return_all_layers=True
        """
        # TODO: implement steps above
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        """
        Set requires_grad=False on all Whisper params (default for Stage-1).
        Keep the module in eval() mode for deterministic behavior.
        """
        # TODO: loop over parameters and freeze; call eval()
        raise NotImplementedError

    def unfreeze_backbone(self, gradient_checkpointing: bool = False) -> None:
        """
        Allow fine-tuning in Stage-2. Optionally enable gradient checkpointing on the encoder.
        """
        # TODO: set requires_grad=True; if requested, enable gc on encoder
        raise NotImplementedError

    def output_shapes(
        self,
        T_mel: int,
        batch_size: int = 1,
    ) -> Tuple[int, int]:
        """
        Predict output time steps and hidden width without a forward pass.

        Logic
        -----
        - Whisper-base has 2 conv layers, stride=2 each -> ~T_mel / 4.
        - The exact formula depends on conv padding; treat as floor division for estimates.
        - Hidden width d_sem is from model.config.d_model (should equal cfg.d_hidden).

        Returns
        -------
        T_sem_est: int  (≈ floor(T_mel / 4))
        d_sem:     int  (= model.config.d_model)
        """
        # TODO: compute and return (T_sem_est, d_sem)
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Internals / helpers
    # -------------------------------------------------------------------------

    def _maybe_normalize_for_whisper(self, mel_bt80: torch.Tensor) -> torch.Tensor:
        """
        Apply Whisper feature-extractor style normalization if `use_feature_extractor_norm` is True.

        Input:
            mel_bt80: [B, T_mel, 80] or [B, 80, T_mel] depending on where you call it.

        Guidance:
            - WhisperFeatureExtractor does per-channel mean/var normalization on log-mels.
            - If your collator already approximates Whisper preprocessing closely, keep this ON
              to reduce train/test skew when using official checkpoints.
            - Implement as an in-graph operation (no CPU numpy).

        Output:
            mel normalized identically to Whisper’s expected scale.
        """
        # TODO: implement normalization (optional)
        raise NotImplementedError

    def _downsample_mask_whisper(
        self,
        mel_mask: torch.Tensor,  # [B, T_mel]
        T_mel: int,
        T_sem: int,
    ) -> torch.Tensor:
        """
        Compute the semantic mask after Whisper's x4 time reduction.

        Strategy:
            - Prefer pooling-by-window: for each semantic index t,
              derive its receptive range on the original T_mel, then set sem_mask[t] = any(valid).
            - A simpler approximation is chunking T_mel into T_sem windows and OR-reducing
              (max-pool on a boolean tensor). Implement whichever is simpler and document it.

        Edge cases:
            - If T_sem * 4 != T_mel due to padding, last window may be smaller: handle safely.
            - Ensure bool dtype on output.
        """
        # TODO: implement mask downsampling
        raise NotImplementedError

    def _select_and_aggregate_layers(
        self,
        hidden_states: List[torch.Tensor],  # each [B, T_sem, d_sem]
    ) -> torch.Tensor:
        """
        Select indices in self.output_layers and aggregate per self.layer_aggregation.

        - "last": return the last selected layer.
        - "mean": average across selected layers (dim=0 of the selected list).
        - "concat": concat along feature dim -> [B, T_sem, k*d_sem]; if k>1, project
          back to d_sem using a learned linear defined in __init__.

        Returns:
            H_sem: [B, T_sem, d_sem]  (after optional projection)
        """
        # TODO: implement selection + aggregation
        raise NotImplementedError

    def _ensure_format(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Ensure input features are formatted for Whisper:
        - Collator provides [B, T_mel, 80].
        - Whisper expects [B, 80, T_mel].
        - Perform transpose if necessary; keep dtype/device.
        """
        # TODO: transpose as needed
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # (Optional) API for exporting intermediate features
    # -------------------------------------------------------------------------

    def extract_layerwise(
        self,
        mel: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Return a list of per-layer semantic sequences for analysis/ablation.

        Behavior:
            - Same preprocessing as forward().
            - Force output_hidden_states=True in the HF model call.
            - Select and return the 'layers' indices you request.
        """
        # TODO: implement using HF outputs.hidden_states
        raise NotImplementedError
