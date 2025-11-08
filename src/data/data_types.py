from __future__ import annotations
from typing import List, TypedDict

import numpy as np


class DatasetInputRow(TypedDict):
    audio_path: str
    transcript: str
    emotion_label: str
    combined_embedding: List[float]
    assistant_reply: str
    context: List[str]
    dataset_source: str


class DatasetOutputRow(TypedDict):
    audio_path: str
    transcript: str
    emotion_label: str
    combined_embedding: np.ndarray
    assistant_reply: str
    context: List[str]
