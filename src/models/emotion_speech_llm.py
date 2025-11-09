# src/model/emotion_speech_llm.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from .config import EmotionSpeechLLMConfig
from .semantic_encoder import WhisperSemanticEncoder
from .para_encoder import ParalinguisticEncoder
from .fusion import CrossAttentionFusion
from .audio_adapter import AudioConditioningAdapter


class EmotionSpeechLLM(nn.Module):
    """
    End-to-end model:
      collator batch ──> Whisper encoder ──> paralinguistic encoder ──> fusion ──> audio adapter ──> LLM

    Expected batch keys (from your collator):
      - audio_features:      [B, T_mel, 80]
      - audio_attention_mask:[B, T_mel]
      - combined_embedding:  [B, 960]
      - input_ids:           [B, T_txt]
      - attention_mask:      [B, T_txt]
      - labels:              [B, T_txt]  (-100 for prompt & pad)

    Forward returns:
      - dict with "loss" (if labels present), "logits" (optional), and any diagnostics you add.
    """

    def __init__(self, cfg: EmotionSpeechLLMConfig, llm: nn.Module, tokenizer=None):
        """
        Instantiate submodules from cfg:
          self.sem = WhisperSemanticEncoder(cfg.semantic, model_name_or_path=cfg.llm.model_name_or_path or "openai/whisper-base")
            (Note: you may pass whisper path explicitly, not cfg.llm.)
          self.para = ParalinguisticEncoder(cfg.paralinguistic)
          self.fuse = CrossAttentionFusion(cfg.fusion)
          self.adapter = AudioConditioningAdapter(cfg.adapter, n_layers=<llm layers>, n_heads=<llm heads>, d_head=<llm head dim>)
          self.llm = llm
          self.tokenizer = tokenizer (optional; mainly for special handling if needed)

        Freezing defaults (Stage-1):
          - self.sem.freeze_backbone()
          - keep LLM frozen or LoRA-only per cfg.train
        """
        super().__init__()
        self.cfg = cfg
        # TODO: instantiate modules, infer n_layers/n_heads/d_head from llm.config if available

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        1) Unpack batch tensors.
        2) Semantic encoding:
             H_sem, sem_mask = self.sem(batch["audio_features"], batch["audio_attention_mask"])
        3) Paralinguistic:
             z_p = self.para(batch["combined_embedding"])
        4) Fusion:
             H_tilde = self.fuse(H_sem, z_p)
        5) Adapter dispatch by cfg.adapter.mode:
           - "soft_tokens":
               A_tok = self.adapter.forward_soft_tokens(H_tilde, sem_mask, z_p)
               kwargs = self.adapter.build_llm_inputs_with_soft_tokens(
                           llm=self.llm,
                           text_input_ids=batch["input_ids"],
                           text_attention_mask=batch["attention_mask"],
                           labels=batch.get("labels"),
                           A_tok=A_tok,
                        )
               outputs = self.llm(**kwargs)

           - "prefix_kv":
               pkv = self.adapter.forward_prefix_kv(H_tilde, sem_mask, z_p)
               kwargs = self.adapter.build_llm_inputs_with_prefix_kv(
                           text_input_ids=batch["input_ids"],
                           text_attention_mask=batch["attention_mask"],
                           labels=batch.get("labels"),
                           past_key_values=pkv,
                        )
               outputs = self.llm(**kwargs)

           - "side_attn":
               raise NotImplementedError (until you wrap the LLM)

        6) Return dict:
             {"loss": outputs.loss, "logits": outputs.logits, "extras": {...}} as needed.

        Notes:
        - Ensure the loss only applies to text tokens (soft-tokens are -100 via the adapter helper).
        - Mixed precision: respect the LLM's autocast dtype (bf16).
        - Device: move adapter products to the same device as LLM params.
        """
        # TODO
        raise NotImplementedError

    # -------------
    # Convenience
    # -------------

    def generate(
        self, batch: Dict[str, torch.Tensor], max_new_tokens: int = 64, **gen_kwargs
    ) -> Dict[str, Any]:
        """
        Inference path (no labels):
          - Same steps 2..5 but call llm.generate(...)
          - For "soft_tokens": prebuild inputs_embeds and attention_mask, then pass to generate
            (HF supports generation with inputs_embeds for many models; verify your LLM does).
          - For "prefix_kv": call generate with past_key_values set, plus input_ids+attention_mask.
        Return dict with "sequences" and/or "texts" if you decode.
        """
        # TODO
        raise NotImplementedError

    def freeze_backbones_stage1(self) -> None:
        """
        Freeze semantic encoder and (optionally) LLM per cfg.train.
        Keep fusion/adapter trainable for alignment.
        """
        # TODO
        raise NotImplementedError

    def unfreeze_for_stage2(self) -> None:
        """
        Unfreeze components per your Stage-2 recipe (e.g., enable LoRA on LLM, keep paralinguistic fixed).
        """
        # TODO
        raise NotImplementedError
