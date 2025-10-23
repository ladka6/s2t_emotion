# custom_interface.py
import torch
from speechbrain.inference.interfaces import Pretrained


class CustomEncoderWav2vec2Classifier(Pretrained):
    """Official interface for emotion-recognition-wav2vec2-IEMOCAP
    + added embedding extraction (encode_file()).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # -----------------------------------------------------------
    # === EMBEDDING EXTRACTION ===
    # -----------------------------------------------------------
    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding."""
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        outputs = self.mods.wav2vec2(wavs)
        outputs = self.mods.avg_pool(outputs, wav_lens)
        outputs = outputs.view(outputs.shape[0], -1)
        return outputs

    # -----------------------------------------------------------
    # === CLASSIFICATION ===
    # -----------------------------------------------------------
    def classify_batch(self, wavs, wav_lens=None):
        outputs = self.encode_batch(wavs, wav_lens)
        outputs = self.mods.output_mlp(outputs)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path):
        """Classifies and returns (out_prob, score, index, text_lab, embedding)."""
        waveform = self.load_audio(path)
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        embedding = self.encode_batch(batch, rel_length)
        outputs = self.mods.output_mlp(embedding).squeeze(1)
        out_prob = self.hparams.softmax(outputs)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab, embedding

    # -----------------------------------------------------------
    # === EMBEDDING-ONLY SHORTCUT ===
    # -----------------------------------------------------------
    def encode_file(self, path):
        """Returns embedding, index, and label (no classification head)."""
        waveform = self.load_audio(path)
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        embedding = self.encode_batch(batch, rel_length)
        return embedding
