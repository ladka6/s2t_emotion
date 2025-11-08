# src/data/collator.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio

from audio_processor import load_audio, rms_normalize, chunk_or_pad


class EmotionDialogueCollator:
    """
    Decoder-only batching:
      Returns
        - audio_features: float32 [B, 3000, 80]       # log-Mel
        - audio_attention_mask: bool/int64 [B, 3000]
        - combined_embedding: float32 [B, 960]
        - input_ids: int64 [B, T]                     # prompt + answer
        - attention_mask: bool/int64 [B, T]
        - labels: int64 [B, T]                        # -100 for prompt & pad
    """

    def __init__(
        self,
        tokenizer,
        max_audio_seconds: int = 30,
        max_input_tokens: int = 256,
        max_output_tokens: int = 128,
        combined_dim: int = 960,
        apply_specaug: bool = False,
        training: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_audio_seconds = int(max_audio_seconds)
        self.max_input_tokens = int(max_input_tokens)
        self.max_output_tokens = int(max_output_tokens)
        self.combined_dim = int(combined_dim)
        self.apply_specaug = bool(apply_specaug)
        self.training = bool(training)

        self.sr = 16000
        self.n_fft = 400
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 80
        self.fmin = 0.0
        self.fmax = 8000.0

        self.T_audio = self.max_audio_seconds * self.sr
        self.T_mel = self.max_audio_seconds * (self.sr // self.hop_length)

        if (
            getattr(self.tokenizer, "pad_token", None) is None
            and getattr(self.tokenizer, "eos_token", None) is not None
        ):
            # Make padding well-defined; keep consistent in training/eval.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Mel + log
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.fmin,
            f_max=self.fmax,
            n_mels=self.n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)

        # Simple SpecAug (off by default)
        self.time_mask_param = 40
        self.freq_mask_param = 12
        self.num_time_masks = 2
        self.num_freq_masks = 2

    # --- internal helpers ---

    def _wav_to_logmel(self, wav_np: np.ndarray) -> torch.Tensor:
        """np.float32 [T] -> torch.float32 [frames, 80]"""
        x = torch.from_numpy(wav_np).float().unsqueeze(0)  # [1, T]
        mel = self.mel_transform(x)  # [1, 80, frames]
        mel_db = self.amp_to_db(mel)  # [1, 80, frames]
        return mel_db.squeeze(0).transpose(0, 1).contiguous()  # [frames, 80]

    def _specaugment_inplace(self, mel: torch.Tensor) -> None:
        # mel: [T, 80]
        if not (self.apply_specaug and self.training):
            return
        # time masks
        for _ in range(self.num_time_masks):
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            if t <= 0 or mel.size(0) <= t:
                continue
            t0 = int(torch.randint(0, mel.size(0) - t, (1,)).item())
            mel[t0 : t0 + t, :] = 0.0
        # freq masks
        for _ in range(self.num_freq_masks):
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if f <= 0 or mel.size(1) <= f:
                continue
            f0 = int(torch.randint(0, mel.size(1) - f, (1,)).item())
            mel[:, f0 : f0 + f] = 0.0

    def _build_prompt(self, transcript: str) -> str:
        # Minimal, decoder-only friendly. No context for now.
        # You can prepend a one-liner instruction if you want stricter style control.
        return f"User (spoken): {transcript}\nAssistant:"

    # --- main call ---

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # A) AUDIO → log-Mel w/ fixed length + mask
        mels, mel_masks = [], []
        for ex in examples:
            wav = load_audio(ex["audio_path"], target_sr=self.sr, mono=True)
            wav = rms_normalize(wav, target_dbfs=-23.0)
            wav = chunk_or_pad(
                wav, max_seconds=self.max_audio_seconds, sr=self.sr, strategy="center"
            )

            mel = self._wav_to_logmel(wav)  # [Ti, 80]
            self._specaugment_inplace(mel)

            Ti = mel.size(0)
            if Ti >= self.T_mel:
                mel = mel[: self.T_mel, :]
                mask = torch.ones(self.T_mel, dtype=torch.bool)
            else:
                pad = self.T_mel - Ti
                mel = torch.cat(
                    [mel, torch.zeros((pad, self.n_mels), dtype=mel.dtype)], dim=0
                )
                mask = torch.cat(
                    [
                        torch.ones(Ti, dtype=torch.bool),
                        torch.zeros(pad, dtype=torch.bool),
                    ],
                    dim=0,
                )

            mels.append(mel)  # [3000, 80]
            mel_masks.append(mask)  # [3000]

        batch = {
            "audio_features": torch.stack(mels, dim=0),  # [B, 3000, 80]
            "audio_attention_mask": torch.stack(mel_masks, dim=0),  # [B, 3000]
        }

        # B) COMBINED EMBEDDINGS → [B, 960]
        cmb_list = []
        for ex in examples:
            vec = ex.get("combined_embedding")
            if vec is None:
                arr = np.zeros((self.combined_dim,), dtype=np.float32)
            else:
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape[0] != self.combined_dim:
                    fixed = np.zeros((self.combined_dim,), dtype=np.float32)
                    n = min(self.combined_dim, arr.shape[0])
                    fixed[:n] = arr[:n]
                    arr = fixed
            cmb_list.append(torch.from_numpy(arr))
        batch["combined_embedding"] = torch.stack(cmb_list, dim=0)  # [B, 960]

        # C) PROMPT (no context) + ANSWER → single sequence (decoder-only)
        prompts = [
            self._build_prompt(ex.get("transcript", "") or "") for ex in examples
        ]
        answers = [ex.get("assistant_reply", "") or "" for ex in examples]

        # tokenize separately to control truncation per part
        tok_prompt = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=True,
        )
        tok_answer = self.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=self.max_output_tokens,
            add_special_tokens=False,
        )

        # concat per example, then pad as a batch
        input_ids_list, labels_list, attn_mask_list = [], [], []
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        for i in range(len(examples)):
            p_ids = tok_prompt["input_ids"][i]
            a_ids = tok_answer["input_ids"][i]

            # decoder-only scheme: loss on answer tokens only
            ids = p_ids + a_ids
            lbl = [-100] * len(p_ids) + a_ids  # ignore prompt

            input_ids_list.append(ids)
            labels_list.append(lbl)
            attn_mask_list.append([1] * len(ids))

        # pad to batch max
        max_len = max(len(x) for x in input_ids_list)
        input_ids = []
        labels = []
        attention_mask = []
        for ids, lbl, mask in zip(input_ids_list, labels_list, attn_mask_list):
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append(mask + [0] * pad_n)
            # pad labels with -100
            labels.append(lbl + [-100] * pad_n)

        batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)  # [B, T]
        batch["attention_mask"] = torch.tensor(
            attention_mask, dtype=torch.long
        )  # [B, T]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)  # [B, T]

        return batch
