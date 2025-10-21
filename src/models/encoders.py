# """
# Encoders for extracting semantic, emotion, and speaker embeddings from audio.
# """

# import torch
# import torch.nn as nn
# from transformers import WhisperModel, Wav2Vec2Model
# from speechbrain.pretrained import EncoderClassifier
# from typing import Optional


# class WhisperEncoder(nn.Module):
#     """
#     Extracts contextualized speech embeddings using Whisper encoder.
#     """
    
#     def __init__(self, model_name: str = "openai/whisper-base", freeze: bool = True):
#         super().__init__()
#         self.model = WhisperModel.from_pretrained(model_name)
#         self.d_model = self.model.config.d_model
        
#         if freeze:
#             for param in self.model.parameters():
#                 param.requires_grad = False
                
#     def forward(self, input_features: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             input_features: [batch_size, n_mels, seq_len] - Mel spectrogram
            
#         Returns:
#             hidden_states: [batch_size, seq_len, d_model]
#         """
#         outputs = self.model.encoder(input_features)
#         return outputs.last_hidden_state


# class EmotionEncoder(nn.Module):
#     """
#     Extracts emotion embeddings using pre-trained Wav2Vec2 emotion recognition model.
#     """
    
#     def __init__(
#         self,
#         model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#         freeze: bool = True
#     ):
#         super().__init__()
#         # Note: SpeechBrain models need special handling
#         # This is a placeholder structure - actual implementation depends on SpeechBrain API
#         self.model_name = model_name
#         self.freeze = freeze
        
#         # Load SpeechBrain model
#         try:
#             self.classifier = EncoderClassifier.from_hparams(
#                 source=model_name,
#                 savedir=f"pretrained_models/{model_name.split('/')[-1]}"
#             )
#             if freeze:
#                 for param in self.classifier.mods.parameters():
#                     param.requires_grad = False
#         except Exception as e:
#             print(f"Warning: Could not load SpeechBrain model. Using placeholder. Error: {e}")
#             self.classifier = None
            
#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             waveform: [batch_size, samples] - Raw audio waveform
            
#         Returns:
#             emotion_embedding: [batch_size, d_emotion]
#         """
#         if self.classifier is None:
#             # Placeholder: return random embeddings
#             return torch.randn(waveform.shape[0], 256, device=waveform.device)
        
#         with torch.set_grad_enabled(not self.freeze):
#             embeddings = self.classifier.encode_batch(waveform)
#             return embeddings.squeeze(1)


# class SpeakerEncoder(nn.Module):
#     """
#     Extracts speaker embeddings using ECAPA-TDNN.
#     """
    
#     def __init__(
#         self,
#         model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
#         freeze: bool = True
#     ):
#         super().__init__()
#         self.model_name = model_name
#         self.freeze = freeze
        
#         # Load SpeechBrain model
#         try:
#             self.encoder = EncoderClassifier.from_hparams(
#                 source=model_name,
#                 savedir=f"pretrained_models/{model_name.split('/')[-1]}"
#             )
#             if freeze:
#                 for param in self.encoder.mods.parameters():
#                     param.requires_grad = False
#         except Exception as e:
#             print(f"Warning: Could not load SpeechBrain model. Using placeholder. Error: {e}")
#             self.encoder = None
            
#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             waveform: [batch_size, samples] - Raw audio waveform
            
#         Returns:
#             speaker_embedding: [batch_size, d_speaker]
#         """
#         if self.encoder is None:
#             # Placeholder: return random embeddings
#             return torch.randn(waveform.shape[0], 192, device=waveform.device)
        
#         with torch.set_grad_enabled(not self.freeze):
#             embeddings = self.encoder.encode_batch(waveform)
#             return embeddings.squeeze(1)


# class ParalinguisticEncoder(nn.Module):
#     """
#     Combines emotion and speaker encoders into a single paralinguistic representation.
#     """
    
#     def __init__(
#         self,
#         emotion_model: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#         speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb",
#         freeze: bool = True
#     ):
#         super().__init__()
        
#         self.emotion_encoder = EmotionEncoder(emotion_model, freeze)
#         self.speaker_encoder = SpeakerEncoder(speaker_model, freeze)
        
#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             waveform: [batch_size, samples] - Raw audio waveform
            
#         Returns:
#             paralinguistic: [batch_size, d_emotion + d_speaker]
#         """
#         emotion_emb = self.emotion_encoder(waveform)
#         speaker_emb = self.speaker_encoder(waveform)
        
#         # Concatenate emotion and speaker embeddings
#         paralinguistic = torch.cat([emotion_emb, speaker_emb], dim=-1)
        
#         return paralinguistic
