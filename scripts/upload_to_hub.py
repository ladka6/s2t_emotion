"""
Upload processed emotion speech dataset to Hugging Face Hub.
This script uploads JSONL files containing audio embeddings and metadata.
"""

import os
import argparse
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import login
import json


def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_dataset_from_jsonl(ravdess_path, cremad_path, iemocap_path, meld_path):
    """Create a Hugging Face dataset from JSONL files."""
    print("Loading JSONL files...")
    ravdess_data = load_jsonl(ravdess_path)
    cremad_data = load_jsonl(cremad_path)
    iemocap_data = load_jsonl(iemocap_path)
    meld_data = load_jsonl(meld_path)

    print(f"   RAVDESS: {len(ravdess_data)} samples")
    print(f"   CREMA-D: {len(cremad_data)} samples")
    print(f"   IEMOCAP: {len(iemocap_data)} samples")
    print(f"   MELD: {len(meld_data)} samples")

    # Combine datasets and add source labels
    all_data = []
    for item in ravdess_data:
        item["dataset_source"] = "RAVDESS"
        all_data.append(item)

    for item in cremad_data:
        item["dataset_source"] = "CREMA-D"
        all_data.append(item)

    for item in iemocap_data:
        item["dataset_source"] = "IEMOCAP"
        all_data.append(item)

    for item in meld_data:
        item["dataset_source"] = "MELD"
        all_data.append(item)

    print(f"   Total: {len(all_data)} samples")

    # Define features schema
    features = Features(
        {
            "audio_path": Value("string"),
            "transcript": Value("string"),
            "emotion_label": Value("string"),
            "combined_embedding": Sequence(Value("float32")),
            "assistant_reply": Value("string"),
            "context": Sequence(Value("string")),
            "dataset_source": Value("string"),
        }
    )

    # Create dataset
    dataset = Dataset.from_list(all_data, features=features)

    # Split into train/validation/test (80/10/10)
    print("\n📊 Splitting dataset...")
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    )

    print(f"   Train: {len(dataset_dict['train'])} samples")
    print(f"   Validation: {len(dataset_dict['validation'])} samples")
    print(f"   Test: {len(dataset_dict['test'])} samples")

    return dataset_dict


def upload_to_hub(dataset_dict, repo_name, token=None, private=False):
    """Upload dataset to Hugging Face Hub."""

    print(f"\n🚀 Uploading to Hugging Face Hub: {repo_name}")

    # Login if token provided
    if token:
        login(token=token)
    else:
        print("   No token provided, using cached credentials...")

    # Upload dataset
    dataset_dict.push_to_hub(
        repo_name,
        private=private,
        commit_message="Upload emotion speech dataset with embeddings",
    )

    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")


def create_readme(dataset_dict, output_path="README.md"):
    """Create a README.md for the dataset."""

    train_size = len(dataset_dict["train"])
    val_size = len(dataset_dict["validation"])
    test_size = len(dataset_dict["test"])
    total_size = train_size + val_size + test_size

    # Get embedding dimension from first sample
    emb_dim = len(dataset_dict["train"][0]["combined_embedding"])

    readme_content = f"""---
license: mit
task_categories:
- audio-classification
- text-generation
language:
- en
tags:
- emotion-recognition
- speech
- empathy
- conversational-ai
size_categories:
- 1K<n<10K
---

# Emotion-Aware Speech Dataset with Embeddings

This dataset contains processed audio samples from RAVDESS and CREMA-D datasets, enriched with:
- **Pre-computed embeddings** (emotion + speaker)
- **Emotion labels** (automatically detected)
- **Transcripts** of spoken utterances
- **GPT-generated empathetic responses**

## Dataset Overview

- **Total samples**: {total_size}
  - Train: {train_size}
  - Validation: {val_size}
  - Test: {test_size}
- **Embedding dimension**: {emb_dim} (concatenated emotion + speaker embeddings)
- **Source datasets**: RAVDESS, CREMA-D

## Dataset Structure

Each sample contains:

```python
{{
    'audio_path': str,              # Path to original audio file
    'transcript': str,              # Transcribed text
    'emotion_label': str,           # Detected emotion (e.g., 'angry', 'happy', 'sad')
    'combined_embedding': List[float],  # Concatenated emotion + speaker embeddings
    'assistant_reply': str,         # GPT-generated empathetic response
    'context': List[str],           # Conversation context (currently empty)
    'dataset_source': str,          # 'RAVDESS' or 'CREMA-D'
}}
```

## Features

### Embeddings
- **Emotion embeddings**: Extracted using Wav2Vec2 model trained on IEMOCAP
- **Speaker embeddings**: Extracted using ECAPA-TDNN model trained on VoxCeleb
- **Combined**: Concatenated into a single feature vector

### Emotion Labels
Automatically detected emotions include:
- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise

### Assistant Replies
Empathetic responses generated by GPT-3.5-turbo based on:
- The spoken transcript
- The detected emotion

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("YOUR_USERNAME/emotion-speech-embeddings")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example: Access first sample
sample = train_data[0]
print(f"Transcript: {{sample['transcript']}}")
print(f"Emotion: {{sample['emotion_label']}}")
print(f"Assistant reply: {{sample['assistant_reply']}}")
print(f"Embedding shape: {{len(sample['combined_embedding'])}}")
```

## Source Datasets

### RAVDESS
The Ryerson Audio-Visual Database of Emotional Speech and Song contains 1440 audio files from 24 actors speaking two sentences with different emotions.

### CREMA-D
The Crowd-sourced Emotional Multimodal Actors Dataset contains 7442 audio clips from 91 actors speaking various sentences with different emotions.

## Citation

If you use this dataset, please cite the original datasets:

```bibtex
@misc{{livingstone2018ryerson,
  title={{Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)}},
  author={{Livingstone, Steven R and Russo, Frank A}},
  year={{2018}},
  publisher={{Zenodo}}
}}

@inproceedings{{cao2014crema,
  title={{CREMA-D: Crowd-sourced emotional multimodal actors dataset}},
  author={{Cao, Houwei and Cooper, David G and Keutmann, Michael K and Gur, Raquel C and Nenkova, Ani and Verma, Ragini}},
  booktitle={{IEEE transactions on affective computing}},
  year={{2014}}
}}
```

## License

MIT License - See LICENSE file for details.

Note: Please ensure you comply with the licenses of the original RAVDESS and CREMA-D datasets.
"""

    with open(output_path, "w") as f:
        f.write(readme_content)

    print(f"\n📝 Created README at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Upload emotion speech dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--ravdess",
        type=str,
        default="data/processed/ravdess_emb.jsonl",
        help="Path to RAVDESS JSONL file",
    )
    parser.add_argument(
        "--cremad",
        type=str,
        default="data/processed/cremad_emb.jsonl",
        help="Path to CREMA-D JSONL file",
    )
    parser.add_argument(
        "--iemocap",
        type=str,
        default="data/processed/iemocap_emb.jsonl",
        help="Path to IEMOCAP JSONL file",
    )
    parser.add_argument(
        "--meld",
        type=str,
        default="data/processed/meld.jsonl",
        help="Path to MELD JSONL file",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repository name (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the dataset private"
    )
    parser.add_argument(
        "--create-readme", action="store_true", help="Create README.md file"
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.ravdess):
        print(f"❌ Error: RAVDESS file not found: {args.ravdess}")
        return

    if not os.path.exists(args.cremad):
        print(f"❌ Error: CREMA-D file not found: {args.cremad}")
        return

    # Create dataset
    dataset_dict = create_dataset_from_jsonl(
        args.ravdess, args.cremad, args.iemocap, args.meld
    )

    # Create README if requested
    if args.create_readme:
        create_readme(dataset_dict)

    # Get token from args or environment
    token = args.token or os.getenv("HF_TOKEN")

    # Upload to Hub
    upload_to_hub(dataset_dict, args.repo_name, token=token, private=args.private)


if __name__ == "__main__":
    main()
