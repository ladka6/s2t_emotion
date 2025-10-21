# 🎧 Emotion-Aware Speech-Language Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)

An open-source model that converts **raw speech → emotionally-aware textual response**, jointly understanding **linguistic, paralinguistic, and contextual** cues (emotion, tone, timbre, speaker style) without hand-feeding categorical labels.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python examples/inference.py --audio samples/happy.wav

# Or use the pipeline
from src.pipeline import AudioToTextPipeline
pipe = AudioToTextPipeline("EgeErdal/emotion-speech-llm")
text = pipe("samples/happy.wav", max_new_tokens=60)
print(text[0]["generated_text"])
```

## 📁 Project Structure

```
s2t_e/
├── src/                          # Core model implementation
│   ├── models/                   # Model architectures
│   ├── data/                     # Data loaders and preprocessing
│   ├── training/                 # Training utilities
│   └── pipeline.py               # Inference pipeline
├── data/                         # Dataset storage
│   ├── raw/                      # Raw audio files
│   ├── processed/                # Preprocessed features
│   └── metadata/                 # JSONL metadata files
├── configs/                      # Training and model configs
├── scripts/                      # Data prep and training scripts
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit tests
├── examples/                     # Usage examples
└── huggingface/                  # HF model and dataset cards
```

## 🏗️ Architecture

The model uses a hybrid approach combining:

1. **Whisper-base** for semantic speech embeddings
2. **Wav2Vec2 + ECAPA-TDNN** for emotion and speaker embeddings
3. **Cross-Attention Fusion Layer** to integrate paralinguistic features
4. **Phi-3 or Mistral** LLM for empathetic text generation

## 📊 Dataset

- **Size**: 70-100k audio-dialogue pairs
- **Sources**: RAVDESS, CREMA-D, IEMOCAP, MELD, Emo-WavCaps
- **Emotions**: 10 categories (angry, happy, sad, neutral, etc.)

## 🎯 Training

Two-stage training process:

**Stage 1: Component Alignment**

```bash
python scripts/train_stage1.py --config configs/stage1_config.yaml
```

**Stage 2: End-to-End Fine-Tuning**

```bash
python scripts/train_stage2.py --config configs/stage2_config.yaml
```

## 📈 Evaluation

```bash
python scripts/evaluate.py --model_path checkpoints/best_model --test_data data/test.jsonl
```
