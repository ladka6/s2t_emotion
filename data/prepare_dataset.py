import os
import json
import torch
from glob import glob
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
import sys

repo_source = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
local_module = os.path.abspath("custom_interface.py")
sys.path.append(os.path.dirname(local_module))
from custom_interface import CustomEncoderWav2vec2Classifier


device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔧 Loading models...")
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
)

emo_model = CustomEncoderWav2vec2Classifier.from_hparams(
    source=repo_source,
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
    run_opts={"device": device},
)


def get_embeddings(wav_path):
    """Extract emotion + speaker embeddings and concatenate them."""
    try:
        print(f"🎧 Loading file: {wav_path}", flush=True)

        signal = spk_model.load_audio(wav_path).to(device)

        with torch.no_grad():
            spk_emb = spk_model.encode_batch(signal).squeeze().cpu()

            out_prob, score, index, text_lab, emb = emo_model.classify_file(wav_path)
            emo_emb = emb.squeeze().cpu()

            if isinstance(text_lab, (list, tuple)):
                if len(text_lab) == 1 and isinstance(text_lab[0], (list, tuple)):
                    text_lab = text_lab[0]
                if len(text_lab) == 1:
                    text_lab = text_lab[0]

            combined_emb = torch.cat((emo_emb, spk_emb), dim=0)
            combined_emb_list = combined_emb.tolist()

            return combined_emb_list, text_lab

    except Exception as e:
        print(f"Failed to process {wav_path}: {e}")
        return [], "unknown"


# --------------------------------------------------------
# 4. Dataset builders
# --------------------------------------------------------
def prepare_ravd(data_dir="data/raw/ravd", output_jsonl="ravdess_emb.jsonl"):
    """Prepare RAVDESS dataset with emotion + speaker embeddings."""
    statement_map = {
        "01": "Kids are talking by the door",
        "02": "Dogs are sitting by the door",
    }

    entries = []
    wav_files = sorted(glob(os.path.join(data_dir, "Actor_*", "*.wav")))
    print(f"Found {len(wav_files)} RAVDESS files.\n")

    for wav_path in tqdm(wav_files, desc="Processing RAVDESS"):
        fname = os.path.basename(wav_path).replace(".wav", "")
        parts = fname.split("-")
        if len(parts) != 7:
            continue

        _, _, _, _, stmt, _, _ = parts
        transcript = statement_map.get(stmt, "")

        combined_emb, emo_pred = get_embeddings(wav_path)

        entry = {
            "audio_path": wav_path,
            "transcript": transcript,
            "emotion_label": emo_pred,
            "combined_embedding": combined_emb,
            "assistant_reply": "",
            "context": [],
        }
        entries.append(entry)

    with open(output_jsonl, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")

    print(f"Saved {len(entries)} RAVDESS entries → {output_jsonl}")


def prepare_crema(data_dir="data/raw/crema-d", output_jsonl="cremad_emb.jsonl"):
    """Prepare CREMA-D dataset with emotion + speaker embeddings."""
    utterance_map = {
        "DFA": "Dogs are sitting by the door",
        "DFB": "Dogs are sitting by the door",
        "IEO": "It's eleven o'clock",
        "IEM": "It's eleven o'clock",
        "SAE": "Somebody is talking by the elevator",
        "TIE": "There is someone at the elevator",
    }

    entries = []
    wav_files = sorted(glob(os.path.join(data_dir, "*.wav")))
    print(f"Found {len(wav_files)} CREMA-D files.\n")

    for wav_path in tqdm(wav_files, desc="Processing CREMA-D"):
        fname = os.path.basename(wav_path).replace(".wav", "")
        parts = fname.split("_")
        if len(parts) != 4:
            continue

        _, utt, _, _ = parts
        transcript = utterance_map.get(utt, "")

        combined_emb, emo_pred = get_embeddings(wav_path)

        entry = {
            "audio_path": wav_path,
            "transcript": transcript,
            "emotion_label": emo_pred,
            "combined_embedding": combined_emb,
            "assistant_reply": "",
            "context": [],
        }
        entries.append(entry)

    with open(output_jsonl, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")

    print(f"Saved {len(entries)} CREMA-D entries → {output_jsonl}")


# --------------------------------------------------------
# 5. Main
# --------------------------------------------------------
if __name__ == "__main__":
    prepare_ravd()
    prepare_crema()
