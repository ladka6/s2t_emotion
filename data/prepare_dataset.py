import os
import json
import re
from networkx import display
import torch
from glob import glob
from tqdm import tqdm
from speechbrain.inference import EncoderClassifier
import sys
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()

repo_source = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
local_module = os.path.abspath("custom_interface.py")
sys.path.append(os.path.dirname(local_module))
from custom_interface import CustomEncoderWav2vec2Classifier
from utils import clean_text


EMO_MAP = {
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "ang": "angry",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Loading models...")
spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
)

emo_model = CustomEncoderWav2vec2Classifier.from_hparams(
    source=repo_source,
    savedir="pretrained_models/emotion-recognition-wav2vec2-IEMOCAP",
    run_opts={"device": device},
)


def generate_assistant_reply(transcript, emotion_label):
    """Generate an empathetic assistant reply using GPT based on transcript and emotion."""
    try:
        prompt = f"""You are an empathetic assistant. Given the following transcript and detected emotion, 
generate a short, natural, and empathetic response (1-2 sentences).

Transcript: "{transcript}"
Detected Emotion: {emotion_label}

Generate an appropriate empathetic response:"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are an empathetic assistant that provides supportive and contextually appropriate responses.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        print(f"Failed to generate GPT reply: {e}")
        return ""


def get_embeddings(wav_path):
    """Extract emotion + speaker embeddings and concatenate them."""
    try:
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

        # Generate assistant reply using GPT
        assistant_reply = generate_assistant_reply(transcript, emo_pred)

        entry = {
            "audio_path": wav_path,
            "transcript": transcript,
            "emotion_label": emo_pred,
            "combined_embedding": combined_emb,
            "assistant_reply": assistant_reply,
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
        "DFA": "Don't forget a jacket",
        "DFB": "Dogs are sitting by the door",
        "IEO": "It's eleven o'clock",
        "IEM": "It's eleven o'clock",
        "SAE": "Somebody is talking by the elevator",
        "TIE": "That is exactly what happened",
        "IOM": "I'm on my way to the meeting",
        "IWW": "I wonder what this is about",
        "TAI": "The airplane is almost full",
        "MTI": "Maybe tomorrow it will be cold",
        "IWL": "I would like a new alarm clock",
        "ITH": "I think I have a doctor's appointment",
        "ITS": "I think I've seen this before",
        "TSI": "The surface is slick",
        "WSI": "We'll stop in a couple of minutes",
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

        # Generate assistant reply using GPT
        assistant_reply = generate_assistant_reply(transcript, emo_pred)

        entry = {
            "audio_path": wav_path,
            "transcript": transcript,
            "emotion_label": emo_pred,
            "combined_embedding": combined_emb,
            "assistant_reply": assistant_reply,
            "context": [],
        }
        entries.append(entry)

    with open(output_jsonl, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")

    print(f"Saved {len(entries)} CREMA-D entries → {output_jsonl}")


# --- Canonical emotion mapping (IEMOCAP → simplified 4 labels)
EMO_MAP = {
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
    "ang": "angry",
}


def prepare_iemocap(
    data_dir="data/raw/iemocap",
    output_jsonl="iemocap_emb.jsonl",
    context_window=3,
):
    """
    Prepare IEMOCAP dataset with emotion + speaker embeddings, GPT replies, and dialogue context.

    Output schema:
    {
        "audio_path": "...",
        "transcript": "...",
        "emotion_label": "...",
        "speaker_embedding": [...],
        "assistant_reply": "...",
        "context": [...]
    }
    """

    entries = []

    # Regex for emotion & transcription parsing
    emo_pattern = re.compile(
        r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(\S+)\s+(\w+)\s+\[(.*?)\]"
    )
    trans_pattern = re.compile(r"(\S+)\s+\[([\d\.]+)-([\d\.]+)\]:\s+(.*)")

    # Iterate over all sessions
    for sess in range(1, 6):
        sess_dir = os.path.join(data_dir, f"Session{sess}")
        emo_dir = os.path.join(sess_dir, "dialog", "EmoEvaluation")
        trans_dir = os.path.join(sess_dir, "dialog", "transcriptions")
        wav_root = os.path.join(sess_dir, "sentences", "wav")

        print(f"\n📂 Processing {sess_dir}...")
        wav_dialogs = sorted(glob(os.path.join(wav_root, "Ses*")))
        print(f"🎧 Found {len(wav_dialogs)} dialogue folders with wavs")

        # Loop over dialogue folders
        for wav_dir in tqdm(wav_dialogs, desc=f"Session {sess}"):
            dialog_id = os.path.basename(wav_dir)
            emo_path = os.path.join(emo_dir, f"{dialog_id}.txt")
            trans_path = os.path.join(trans_dir, f"{dialog_id}.txt")

            # --- Parse emotion file ---
            emo_map = {}
            if os.path.exists(emo_path):
                with open(emo_path) as ef:
                    for line in ef:
                        m = emo_pattern.match(line.strip())
                        if not m:
                            continue
                        start, end, utt_id, emo, vad = m.groups()
                        emo_map[utt_id] = {"emotion": emo}

            # --- Parse transcription file ---
            trans_map = {}
            if os.path.exists(trans_path):
                with open(trans_path) as tf:
                    for line in tf:
                        m = trans_pattern.match(line.strip())
                        if not m:
                            continue
                        utt_id, start, end, text = m.groups()
                        trans_map[utt_id] = text.strip()

            # --- Iterate through wavs ---
            wav_files = sorted(glob(os.path.join(wav_dir, "*.wav")))
            context_buffer = []
            processed_count = 0

            if not wav_files:
                print(f"⚠️ No wav files found in {dialog_id}")
                continue

            for wav_path in tqdm(wav_files, leave=False, desc=f"{dialog_id}"):
                utt_id = os.path.splitext(os.path.basename(wav_path))[0]
                transcript = trans_map.get(utt_id, "").strip()
                if not transcript:
                    continue  # skip missing transcripts

                emo_info = emo_map.get(utt_id, {})
                raw_emo = emo_info.get("emotion", "unknown")
                emo_label = EMO_MAP.get(raw_emo, "unknown")

                # --- Safe embedding extraction ---
                try:
                    combined_emb, emo_pred = get_embeddings(wav_path)
                except Exception as e:
                    print(f"⚠️ Skipping {utt_id} (embedding error: {e})")
                    continue

                if emo_label == "unknown":
                    emo_label = emo_pred

                # --- Generate GPT empathy reply ---
                assistant_reply = generate_assistant_reply(transcript, emo_label)

                # --- Context = previous N utterances
                context = [c["transcript"] for c in context_buffer[-context_window:]]

                # --- Build final entry
                entry = {
                    "audio_path": wav_path,
                    "transcript": transcript,
                    "emotion_label": emo_label,
                    "combined_embedding": combined_emb,
                    "assistant_reply": assistant_reply,
                    "context": context,
                }

                entries.append(entry)
                context_buffer.append(entry)
                processed_count += 1

            print(f"✅ {dialog_id}: processed {processed_count} utterances")

    # --- Save JSONL ---
    with open(output_jsonl, "w") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")

    print(f"\n✅ Saved {len(entries)} utterances → {output_jsonl}")


def prepare_meld():
    output_jsonl = "meld.jsonl"
    path = os.path.join("data/raw/meld/train")
    csv_dir = os.path.join(path, "train_sent_emo.csv")
    data_dir = os.path.join(path, "train_splits")

    df = pd.read_csv(csv_dir)
    df = df[["Dialogue_ID", "Utterance_ID", "Utterance", "Emotion"]]
    df["Utterance"] = df["Utterance"].apply(clean_text)

    processed_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r") as f:
            for line in f:
                try:
                    sample = json.loads(line)

                    key = os.path.basename(sample["audio_path"])
                    processed_ids.add(key)
                except json.JSONDecodeError:
                    continue
        print(
            f"Resuming from checkpoint: {len(processed_ids)} samples already processed."
        )

    with open(output_jsonl, "a") as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing MELD"):
            audio_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
            audio_path = os.path.join(data_dir, audio_filename)

            if audio_filename in processed_ids:
                continue

            if not os.path.exists(audio_path):
                print(f"Missing file: {audio_path}")
                continue

            try:
                combined_emb, emo_pred = get_embeddings(audio_path)
            except Exception as e:
                print(f"Skipping {audio_filename} ({e})")
                continue

            assistant_reply = generate_assistant_reply(row["Utterance"], emo_pred)

            sample = {
                "audio_path": audio_path,
                "transcript": row["Utterance"],
                "emotion_label": emo_pred,
                "combined_embedding": combined_emb,
                "assistant_reply": assistant_reply,
                "context": [],
            }
            json.dump(sample, f_out)
            f_out.write("\n")
            f_out.flush()

    print("✅ MELD preparation complete (checkpoint-safe).")


# --------------------------------------------------------
# 5. Main
# --------------------------------------------------------
if __name__ == "__main__":
    # prepare_ravd()
    # prepare_crema()
    # prepare_iemocap()
    prepare_meld()
