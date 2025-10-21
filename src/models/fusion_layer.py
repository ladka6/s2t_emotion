# """
# Cross-Attention Fusion Layer for integrating paralinguistic features
# into Whisper embeddings at every timestep.
# """

# import torch
# import torch.nn as nn
# import math


# class CrossAttentionFusion(nn.Module):
#     """
#     Fuses Whisper hidden states with paralinguistic (emotion + speaker) embeddings
#     using cross-attention mechanism.
    
#     Args:
#         d_whisper: Dimension of Whisper hidden states (default: 768 for base)
#         d_paralinguistic: Dimension of concatenated emotion + speaker vector
#         d_hidden: Hidden dimension for attention computation
#         num_heads: Number of attention heads
#         dropout: Dropout probability
#     """
    
#     def __init__(
#         self,
#         d_whisper: int = 768,
#         d_paralinguistic: int = 512,
#         d_hidden: int = 768,
#         num_heads: int = 8,
#         dropout: float = 0.1
#     ):
#         super().__init__()
        
#         self.d_whisper = d_whisper
#         self.d_paralinguistic = d_paralinguistic
#         self.d_hidden = d_hidden
#         self.num_heads = num_heads
#         self.head_dim = d_hidden // num_heads
        
#         assert self.head_dim * num_heads == d_hidden, "d_hidden must be divisible by num_heads"
        
#         # Query projection from Whisper embeddings
#         self.W_q = nn.Linear(d_whisper, d_hidden)
        
#         # Key and Value projections from paralinguistic embeddings
#         self.W_k = nn.Linear(d_paralinguistic, d_hidden)
#         self.W_v = nn.Linear(d_paralinguistic, d_hidden)
        
#         # Output projection
#         self.W_o = nn.Linear(d_hidden, d_whisper)
        
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_whisper)
        
#         self.scale = math.sqrt(self.head_dim)
        
#     def forward(self, whisper_hidden: torch.Tensor, paralinguistic: torch.Tensor):
#         """
#         Args:
#             whisper_hidden: [batch_size, seq_len, d_whisper]
#             paralinguistic: [batch_size, d_paralinguistic]
            
#         Returns:
#             fused_hidden: [batch_size, seq_len, d_whisper]
#         """
#         batch_size, seq_len, _ = whisper_hidden.shape
        
#         # Compute queries from Whisper hidden states
#         Q = self.W_q(whisper_hidden)  # [B, T, d_hidden]
#         Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         # [B, num_heads, T, head_dim]
        
#         # Compute keys and values from paralinguistic embeddings
#         K = self.W_k(paralinguistic).unsqueeze(1)  # [B, 1, d_hidden]
#         V = self.W_v(paralinguistic).unsqueeze(1)  # [B, 1, d_hidden]
        
#         K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
#         V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
#         # [B, num_heads, 1, head_dim]
        
#         # Compute attention scores
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         # [B, num_heads, T, 1]
        
#         attn_weights = torch.softmax(attn_scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
        
#         # Apply attention to values
#         attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, T, head_dim]
        
#         # Reshape and project
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(batch_size, seq_len, self.d_hidden)
        
#         output = self.W_o(attn_output)  # [B, T, d_whisper]
        
#         # Residual connection and layer norm
#         fused_hidden = self.layer_norm(whisper_hidden + self.dropout(output))
        
#         return fused_hidden


# class ProjectionMLP(nn.Module):
#     """
#     Projects fused embeddings to LLM hidden dimension.
#     """
    
#     def __init__(
#         self,
#         d_input: int = 768,
#         d_llm: int = 3072,
#         hidden_dim: int = 2048,
#         dropout: float = 0.1
#     ):
#         super().__init__()
        
#         self.projection = nn.Sequential(
#             nn.Linear(d_input, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, d_llm),
#             nn.LayerNorm(d_llm)
#         )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: [batch_size, seq_len, d_input]
#         Returns:
#             projected: [batch_size, seq_len, d_llm]
#         """
#         return self.projection(x)
