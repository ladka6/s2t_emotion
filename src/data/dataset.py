# dataset.py
# Purpose: expose HuggingFace rows as a PyTorch Dataset that returns canonical fields
# without decoding audio (collator will load audio by path).

import os
import numpy as np
from torch.utils.data import Dataset


class EmotionDialogueDataset(Dataset):
    """
    Contract per item (dict):
      - "audio_path": str (absolute path to .wav in the HF snapshot)
      - "transcript": str
      - "assistant_reply": str
      - "context": list[str]
      - "paralinguistic": np.float32 array, shape [960]
      - "emotion_label": str (optional)
    """

    def __init__(
        self,
        repo_id_or_ds,
        split="train",
        data_root=None,
        sr=16000,
        include_embeddings=True,
    ):
        """
        Steps to implement:
        1) If `repo_id_or_ds` is a string, call datasets.load_dataset(...) to get splits.
           Else assume it's already a DatasetDict/Dataset.
        2) Select the split (`train`/`validation`/`test`).
        3) Determine the local snapshot root where files are stored:
           - If the dataset provides an absolute path, you may not need `data_root`.
           - If rows hold relative paths, set `self.data_root` to the repo snapshot dir or given `data_root`.
        4) Build an internal list of lightweight records (e.g., dicts with the fields you need):
           - Resolve each `audio_path` to an ABSOLUTE path (join root + relative).
           - Copy text fields: transcript, assistant_reply, context (ensure list[str], convert None -> []).
           - Handle embeddings:
             * If `include_embeddings=True`, convert `combined_embedding` (list of floats) to np.float32 array, shape [960].
             * Else, store a zero vector of shape [960].
           - Optionally keep emotion_label for metrics.
        5) Store `self.sr = sr` and `self.include_embeddings`.
        6) Consider filtering out rows whose audio file is missing (raise or skip).
        """
        pass

    def __len__(self):
        """
        Steps to implement:
        1) Return the length of your internal records list.
        """
        pass

    def __getitem__(self, idx):
        """
        Steps to implement:
        1) Fetch the idx-th record from your internal list.
        2) Return a dict with the exact keys described in the class docstring.
           - Do NOT load audio here (leave that to the collator).
           - Ensure 'paralinguistic' is np.float32 and has shape (960,).
           - Ensure 'context' is a list[str].
        3) You may assert invariants (path existence, embedding length) for early error detection.
        """
        pass
