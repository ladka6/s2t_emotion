# 🎧 Emotion-Aware Speech-Language Model

### Technical Design Document

**Author:** Ege Erdal  
**Version:** 2.0 (Hybrid Research Prototype)  
**Date:** October 2025

---

## 1. Objective

Create an open-source model that converts **raw speech → emotionally-aware textual response**, jointly understanding **linguistic, paralinguistic, and contextual** cues (emotion, tone, timbre, speaker style) without hand-feeding categorical labels.

---

## 2. System Overview

| Stage | Module                           | Description                                                             | Example Model                                                                                  |
| ----- | -------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| ①     | **Semantic Encoder**             | Contextualized speech embeddings retaining linguistic & prosodic cues   | `openai/whisper-base`                                                                          |
| ②     | **Paralinguistic Encoders**      | Emotion + speaker conditioning                                          | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`, `speechbrain/spkrec-ecapa-voxceleb`        |
| ③     | **Cross-Attention Fusion Layer** | Inject paralinguistic information into every timestep of Whisper output | Custom Transformer cross-attention                                                             |
| ④     | **Projection Layer**             | MLP mapping fused features to LLM hidden dimension                      | Linear + GELU                                                                                  |
| ⑤     | **LLM Decoder**                  | Generates text response conditioned on fused features                   | `microsoft/phi-3-mini-4k-instruct` (efficient) or `mistralai/Mistral-7B-Instruct-v0.3` (large) |

Output: **Emotion-aware, contextually grounded text reply**

---

## 3. Data Specification

### 3.1 Unified Schema (`train.jsonl`)

```json
{
  "audio_path": "clips/ANG_0001.wav",
  "transcript": "Don't do that!",
  "emotion_label": "angry",
  "speaker_embedding": [0.12, -0.08, 0.05, ...],
  "assistant_reply": "I'm sorry, I didn't mean to upset you.",
  "context": ["Hi, how can I help?", "It keeps crashing."]
}
```

### 3.2 Datasets

| Source      | Size        | Labels      | License  |
| ----------- | ----------- | ----------- | -------- |
| RAVDESS     | 1.4 k       | 8 emotions  | CC BY-NC |
| CREMA-D     | 7 k         | 6 emotions  | CC BY-NC |
| IEMOCAP     | 10 k        | 10 emotions | Research |
| MELD        | 13 k        | 7 emotions  | CC BY-NC |
| Emo-WavCaps | 50 k subset | 10 emotions | Open     |

**Total target ≈ 70–100 k** audio-dialogue pairs after augmentation.

Optional: augment with synthetic empathy data using an LLM rewriter.

---

## 4. Model Architecture

### 4.1 Diagram

```
                ┌──────────────────────────────┐
                │        Whisper-base          │
                │ (semantic audio embeddings)  │
                └──────────────┬───────────────┘
                               │ [B,T,768]
                     ┌─────────┴─────────┐
                     │                   │
       ┌─────────────▼────────────┐  ┌───▼──────────────────┐
       │ Emotion Encoder (wav2vec)│  │ Speaker Encoder (ECAPA)│
       └─────────────┬────────────┘  └───┬──────────────────┘
                     │                   │
                     └──────────┬────────┘
                                ↓
                  Paralinguistic vector z_p ∈ ℝ^(De+Ds)
                                ↓
            Cross-Attention Fusion (Whisper → z_p)
                                ↓
                     Projection MLP → ℝ^D_LLM
                                ↓
                    LLM Decoder (Φ-3 or Mistral)
                                ↓
                    Emotion-Aware Text Output
```

### 4.2 Cross-Attention Formulation

Let

- H ∈ ℝ^(T×d_h): Whisper hidden sequence
- z_p ∈ ℝ^(d_p): concatenated emotion + speaker vector

We create key/value matrices K,V = W_k z_p, W_v z_p broadcasted across time.  
The fused representation:
ilde{H} = softmax((H W_q)(K)^T / sqrt(d_h))V + H

Then project ilde{H} → LLM embedding dimension.

This ensures **every timestep “attends” to speaker & emotion context**, unlike the token-prefix trick.

---

## 5. Training Strategy

### 5.1 Stage 1 – Component Alignment

- Freeze Whisper, emotion, speaker, and LLM weights.
- Train only cross-attention + projection MLP.
- Losses:
  L = λ*emo * L*emo + λ_align * L_contrast
  - L_emo: predict emotion label from fused embedding
  - L_contrast: contrastive alignment between fused and LLM hidden states

### 5.2 Stage 2 – End-to-End Fine-Tuning

- Freeze paralinguistic encoders.
- Apply **LoRA (rank = 32–64)** to LLM and fusion layer.
- Optimize next-token loss:
  L*text = -Σ_t log P(y_t | y*<t, ilde{H})
- LR = 1e-4, AdamW, batch = 8 (grad accum = 32), bfloat16.

Optional: add **spec-augment** and **audio mixup** for robustness.

---

## 6. Evaluation (can be changed)

### 6.1 Automated

| Metric                   | Purpose                                                   |
| ------------------------ | --------------------------------------------------------- |
| **BLEU / ROUGE-L**       | Text quality                                              |
| **Emotion Alignment F1** | Classifier match between audio emotion and generated text |
| **Empathy Score**        | Fine-tuned LLM classifier rating                          |
| **Sensitivity**          | Change in output when tone changes                        |
| **WER (for analysis)**   | Ensure linguistic comprehension                           |

### 6.2 Human

5-point Likert: **Appropriateness, Empathy, Relevance, Naturalness**  
Compare:

1. Transcript-only baseline
2. Audio-informed hybrid (ours)
3. Ablations: −emotion, −speaker, −cross-attention

---

## 7. Implementation Plan

| Phase | Task                                   | Deliverable                             |
| ----- | -------------------------------------- | --------------------------------------- |
| P0    | Data curation & preprocessing          | `datasets/prepare_dataset.py`           |
| P1    | Encoder extraction                     | cached Whisper/Emotion/Speaker features |
| P2    | Fusion & Projection training (Stage 1) | `fusion_adapter.pt`                     |
| P3    | End-to-end fine-tuning (Stage 2)       | `pytorch_model.bin`                     |
| P4    | Evaluation + Gradio demo               | interactive app                         |
| P5    | Release on Hugging Face                | model + dataset repos                   |

**Example demo snippet**

```python
from pipeline import AudioToTextPipeline
pipe = AudioToTextPipeline("EgeErdal/emotion-speech-llm")
text = pipe("samples/happy.wav", max_new_tokens=60)
print(text[0]["generated_text"])
```

---

## 8. Deployment & Ethics

- Embeddings, not raw voice, stored → privacy-safe.
- No categorical gender outputs.
- Audit fairness across gender, accent, and emotion intensity.
- Open license: CC BY-NC-SA / Apache 2.0 hybrid for code + model.
- Optional on-device distilled variant using 8-bit quantization.

---

## 9. Future Directions

1. **Continuous emotion regression** (valence-arousal space).
2. **Multilingual extension** via Whisper large-v3 + mT5 decoder.
3. **Audio-text co-training** with _speech prompting_.
4. **Emotion-guided TTS** for full empathetic conversation loop.

---

### 🔍 Summary of Improvements

| Aspect     | Old Design             | Improved Hybrid                      |
| ---------- | ---------------------- | ------------------------------------ |
| Fusion     | Simple MLP token       | Cross-attention across all timesteps |
| Learning   | Independent stages     | Two-phase alignment + end-to-end     |
| Encoder    | Whisper-small          | Whisper-base                         |
| LLM        | Mistral or Phi-3       | Configurable small/large             |
| Dataset    | Multi-emotion mix      | Same + dialogue-context              |
| Evaluation | Text + emotion metrics | Unified automatic + human + ablation |

---
