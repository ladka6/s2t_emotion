# src/model/config.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import warnings
import yaml


# -----------------------------
# Section dataclasses
# -----------------------------


@dataclass
class SemanticEncoderConfig:
    in_mels: int = 80
    d_hidden: int = 768
    stride: int = 4
    dropout: float = 0.10


@dataclass
class ParalinguisticConfig:
    d_in: int = 960
    d_proj: int = 256
    dropout: float = 0.10


@dataclass
class FusionConfig:
    d_semantic: int = 768
    d_para: int = 256
    n_heads: int = 8
    dropout: float = 0.10


@dataclass
class AudioAdapterConfig:
    # mode: "soft_tokens" | "prefix_kv" | "side_attn"
    mode: str = "soft_tokens"
    d_llm: int = 3072  # 3072 (Phi-3 mini) or 4096 (Mistral 7B)
    n_audio_tokens: int = 8
    pool: str = "attn"  # "attn" | "mean" | "conv"
    dropout: float = 0.10


@dataclass
class LLMConfig:
    model_name_or_path: str = "microsoft/phi-3-mini-4k-instruct"
    pad_to_left: bool = False


@dataclass
class TrainConfig:
    freeze_semantic: bool = True
    freeze_llm: bool = True
    use_lora: bool = True
    lora_rank: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    bf16: bool = True
    grad_accum_steps: int = 32
    seed: int = 1337


@dataclass
class PathsConfig:
    output_dir: str = "outputs/exp01"
    ckpt_dir: str = "checkpoints/exp01"
    log_dir: str = "runs/exp01"


@dataclass
class EmotionSpeechLLMConfig:
    name: str = "emotion-speech-llm-base"
    semantic: SemanticEncoderConfig = field(default_factory=SemanticEncoderConfig)
    paralinguistic: ParalinguisticConfig = field(default_factory=ParalinguisticConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    adapter: AudioAdapterConfig = field(default_factory=AudioAdapterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # --------- Derived helpers (read-only) ---------

    @property
    def d_head(self) -> int:
        return self.fusion.d_semantic // self.fusion.n_heads

    @property
    def t_mel_nominal(self) -> int:
        # 30s * 100 fps (hop=160 at 16kHz) ≈ 3000 frames (matches your collator)
        return 3000

    @property
    def t_sem_est(self) -> int:
        s = max(1, int(self.semantic.stride))
        # simple ceil division
        return (self.t_mel_nominal + s - 1) // s

    # --------- Summary ---------

    def summary_str(self) -> str:
        lines = [
            f"[Config: {self.name}]",
            f"SEM: in_mels={self.semantic.in_mels}, d_hidden={self.semantic.d_hidden}, stride={self.semantic.stride}, dropout={self.semantic.dropout:.2f}  → T_sem≈{self.t_sem_est}",
            f"PARA: d_in={self.paralinguistic.d_in} → d_proj={self.paralinguistic.d_proj}, dropout={self.paralinguistic.dropout:.2f}",
            f"FUSION: d_sem={self.fusion.d_semantic}, n_heads={self.fusion.n_heads} → d_head={self.d_head}, d_para={self.fusion.d_para}, dropout={self.fusion.dropout:.2f}",
            f"ADAPTER: mode={self.adapter.mode}, d_llm={self.adapter.d_llm}, n_audio_tokens={self.adapter.n_audio_tokens}, pool={self.adapter.pool}, dropout={self.adapter.dropout:.2f}",
            f"LLM: {self.llm.model_name_or_path} (pad_to_left={self.llm.pad_to_left})",
            f"TRAIN: lr={self.train.lr}, wd={self.train.weight_decay}, bf16={self.train.bf16}, lora_rank={self.train.lora_rank}, freeze_semantic={self.train.freeze_semantic}, freeze_llm={self.train.freeze_llm}",
            f"PATHS: {self.paths.output_dir} | {self.paths.ckpt_dir} | {self.paths.log_dir}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary_str())

    # --------- Validation ---------

    def validate(self) -> None:
        # Heads divisibility
        if self.fusion.d_semantic % self.fusion.n_heads != 0:
            raise ValueError(
                f"fusion.n_heads ({self.fusion.n_heads}) must divide fusion.d_semantic ({self.fusion.d_semantic})."
            )
        # Adapter checks
        if self.adapter.n_audio_tokens < 1:
            raise ValueError("adapter.n_audio_tokens must be ≥ 1.")
        if self.adapter.d_llm <= 0:
            raise ValueError("adapter.d_llm must be > 0.")
        if self.adapter.mode not in {"soft_tokens", "prefix_kv", "side_attn"}:
            raise ValueError(
                "adapter.mode must be one of {'soft_tokens','prefix_kv','side_attn'}."
            )
        if self.adapter.pool not in {"attn", "mean", "conv"}:
            raise ValueError("adapter.pool must be one of {'attn','mean','conv'}.")

        # Cross-module consistency
        if self.paralinguistic.d_proj != self.fusion.d_para:
            raise ValueError(
                f"paralinguistic.d_proj ({self.paralinguistic.d_proj}) must equal fusion.d_para ({self.fusion.d_para})."
            )
        if self.semantic.d_hidden != self.fusion.d_semantic:
            raise ValueError(
                f"semantic.d_hidden ({self.semantic.d_hidden}) must equal fusion.d_semantic ({self.fusion.d_semantic})."
            )

        # Train sanity
        if not (0.0 < self.train.lr < 1.0):
            raise ValueError("train.lr must be between 0 and 1.")
        if (
            not self.paths.output_dir
            or not self.paths.ckpt_dir
            or not self.paths.log_dir
        ):
            raise ValueError("paths.* must be non-empty strings.")

        # LLM width hints (warnings, not hard errors)
        lower = self.llm.model_name_or_path.lower()
        if "mistral" in lower and self.adapter.d_llm != 4096:
            warnings.warn(
                "Detected 'mistral' in LLM path but adapter.d_llm != 4096; "
                "verify d_llm matches the model's embedding size."
            )
        if ("phi" in lower or "phi-3" in lower) and self.adapter.d_llm != 3072:
            warnings.warn(
                "Detected 'phi' in LLM path but adapter.d_llm != 3072; "
                "verify d_llm matches the model's embedding size."
            )

    # --------- Construction from dict/YAML ---------

    @staticmethod
    def _deep_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @classmethod
    def from_dict(
        cls, cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
    ) -> "EmotionSpeechLLMConfig":
        cfg = copy.deepcopy(cfg) if cfg is not None else {}

        def _get(section: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
            user = cfg.get(section, {}) or {}
            # unknown key detection inside section
            for k in user.keys():
                if k not in defaults:
                    raise KeyError(f"Unknown config key: {section}.{k}")
            merged = {**defaults, **user}
            return merged

        # Section defaults to validate unknown keys
        sem_def = asdict(SemanticEncoderConfig())
        para_def = asdict(ParalinguisticConfig())
        fus_def = asdict(FusionConfig())
        adp_def = asdict(AudioAdapterConfig())
        llm_def = asdict(LLMConfig())
        trn_def = asdict(TrainConfig())
        pth_def = asdict(PathsConfig())

        name = cfg.get("name", "emotion-speech-llm-base")

        semantic = SemanticEncoderConfig(**_get("semantic", sem_def))
        paralinguistic = ParalinguisticConfig(**_get("paralinguistic", para_def))
        fusion = FusionConfig(**_get("fusion", fus_def))
        adapter = AudioAdapterConfig(**_get("adapter", adp_def))
        llm = LLMConfig(**_get("llm", llm_def))
        train = TrainConfig(**_get("train", trn_def))
        paths = PathsConfig(**_get("paths", pth_def))

        obj = cls(
            name=name,
            semantic=semantic,
            paralinguistic=paralinguistic,
            fusion=fusion,
            adapter=adapter,
            llm=llm,
            train=train,
            paths=paths,
        )

        # Apply dotted-key overrides after construction (e.g., {"adapter.d_llm": 4096})
        if overrides:
            for dk, val in overrides.items():
                _apply_dotted_override(obj, dk, val)

        obj.validate()
        return obj

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None
    ) -> "EmotionSpeechLLMConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not available. Install with `pip install pyyaml`."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, overrides=overrides)


# -----------------------------
# Dotted-key override utility
# -----------------------------


def _apply_dotted_override(obj: Any, dotted_key: str, value: Any) -> None:
    """
    Sets obj.section.subfield = value for 'section.subfield' syntax.
    Raises AttributeError if path is invalid.
    """
    parts = dotted_key.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise AttributeError(f"Unknown override path: '{dotted_key}' (at '{p}')")
        cur = getattr(cur, p)
    leaf = parts[-1]
    if not hasattr(cur, leaf):
        raise AttributeError(f"Unknown override key: '{dotted_key}' (missing '{leaf}')")
    setattr(cur, leaf, value)


# -----------------------------
# CLI/demo helper
# -----------------------------


def demo_print(path: Optional[str] = None) -> EmotionSpeechLLMConfig:
    """
    Convenience for quick manual checks in a REPL:
        >>> from model.config import demo_print
        >>> cfg = demo_print("configs/base.yaml")
    """
    if path is None:
        cfg = EmotionSpeechLLMConfig()  # defaults
    else:
        cfg = EmotionSpeechLLMConfig.from_yaml(path)
    cfg.print_summary()
    return cfg


if __name__ == "__main__":
    # Example usage:
    #   python -m model.config
    # or
    #   python src/model/config.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default=None, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted overrides, e.g. adapter.d_llm=4096",
    )
    args = parser.parse_args()

    overrides_dict: Dict[str, Any] = {}
    for ov in args.override:
        if "=" not in ov:
            raise ValueError(f"Invalid --override '{ov}', expected key=value.")
        k, v = ov.split("=", 1)
        # naive typing: try int -> float -> bool -> str
        vv: Any
        if v.lower() in {"true", "false"}:
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        overrides_dict[k] = vv

    cfg = demo_print(args.yaml)
    if overrides_dict:
        # Rebuild with overrides (so validation runs again)
        if args.yaml:
            cfg = EmotionSpeechLLMConfig.from_yaml(args.yaml, overrides=overrides_dict)
        else:
            cfg = EmotionSpeechLLMConfig.from_dict(
                asdict(cfg), overrides=overrides_dict
            )
        print("\n[After overrides]")
        cfg.print_summary()
