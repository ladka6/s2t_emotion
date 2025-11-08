# dataset.py
# Purpose: expose HuggingFace rows as a PyTorch Dataset that returns canonical fields
# without decoding audio (collator will load audio by path).
from __future__ import annotations

import os
import warnings
from typing import List, cast

import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset

# Make sure this import path matches your actual project structure
from data_types import DatasetInputRow, DatasetOutputRow


class EmotionDialogueDataset(Dataset):
    """
    Dataset that processes Hugging Face rows into canonical internal records.
    Does NOT load audio; that is deferred to the DataCollator.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        sr: int = 16000,
        embedding_dim: int = 960,
        data_root: str | None = None,
    ):
        """
        Args:
            repo_id: Hugging Face dataset repository ID.
            split: 'train', 'validation', or 'test'.
            sr: Expected sampling rate (informational only here, used in collator/preprocessing).
            embedding_dim: Dimension of the pre-computed embeddings.
            data_root: Optional override for the root directory of audio files.
                       If None, tries to auto-detect based on this file's location.
        """
        print(f"Loading dataset '{repo_id}' split '{split}'...")
        ds = load_dataset(repo_id, split=split)

        self.sr = sr
        self.embedding_dim = embedding_dim

        if data_root is None:
            self.root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        else:
            self.root = os.path.abspath(data_root)

        self._records: List[DatasetOutputRow] = []

        print(f"Processing {len(ds)} rows...")
        for idx, raw_row in enumerate(ds):
            row = cast(DatasetInputRow, raw_row)

            rel_path = row.get("audio_path", "").lstrip("/")
            abs_path = os.path.join(self.root, rel_path)

            # 2. Basic validation (Optional: skip if file doesn't exist)
            if not os.path.exists(abs_path):
                pass

            combined_embedding = np.array(row["combined_embedding"], dtype=np.float32)

            context_list = row.get("context")
            if context_list is None:
                context_list = []

            record: DatasetOutputRow = {
                "audio_path": abs_path,
                "transcript": row.get("transcript", ""),
                "emotion_label": row.get("emotion_label", "neutral"),
                "combined_embedding": combined_embedding,
                "assistant_reply": row.get("assistant_reply", ""),
                "context": context_list,
            }

            self._records.append(record)

        print(f"Dataset loaded. {len(self._records)} valid records prepared.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DatasetOutputRow:
        return self._records[idx]
