# src/data/audio_processor.py
# Purpose: load, normalize, and length-control waveforms for 16 kHz speech.
# All functions should be pure and operate on/return numpy float32 arrays.

import numpy as np
import torchaudio


def load_audio(path, target_sr=16000, mono=True):
    wav, sr = torchaudio.load(path)

    if mono and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.squeeze(0)

    wav = wav.detach().cpu().numpy().astype(np.float32)

    max_abs = np.max(np.abs(wav)) + 1e-9
    if max_abs > 1.0:
        wav = wav / max_abs

    return wav


def trim_silence(wav, top_db=30, frame_length=2048, hop_length=512):
    """
    Steps to implement:
    1) Compute an energy (or use librosa.effects.split) to detect non-silent intervals.
    2) Keep only leading/trailing active region; DO NOT remove internal pauses.
    3) Return the trimmed waveform (np.float32). If no speech detected, return original.
    """
    pass


def rms_normalize(wav, target_dbfs=-23.0, eps=1e-9, peak_cap=0.99):
    """
    RMS-normalize an audio waveform to the target dBFS level.
    wav: 1D np.ndarray float32 in [-1, 1]
    target_dbfs: desired loudness (dB full-scale), typical speech ≈ -23 dBFS
    peak_cap: optional safety clamp to avoid clipping.
    """

    rms = np.sqrt(np.mean(wav**2) + eps)

    if rms < eps:
        return wav.astype(np.float32)

    target_rms = 10 ** (target_dbfs / 20.0)

    scale = target_rms / rms

    wav = wav * scale

    max_abs = np.max(np.abs(wav)) + eps
    if max_abs > peak_cap:
        wav = wav * (peak_cap / max_abs)

    return wav.astype(np.float32)


def chunk_or_pad(wav, max_seconds=30.0, sr=16000, strategy="center", pad_value=0.0):
    wav = wav.astype(np.float32, copy=False)
    max_len = int(max_seconds * sr)

    if len(wav) == max_len:
        return wav

    if len(wav) > max_len:
        if strategy == "center":
            start = (len(wav) - max_len) // 2
        elif strategy == "front":
            start = 0
        elif strategy == "random":
            start = np.random.randint(0, len(wav) - max_len + 1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return wav[start : start + max_len]

    pad_width = max_len - len(wav)
    return np.pad(wav, (0, pad_width), mode="constant", constant_values=pad_value)
