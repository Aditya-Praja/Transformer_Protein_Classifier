import torch
from torch import nn

from src.model_attention import MultiHeadSelfAttention

class AttentionSublayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_head: int,
        dropout: float = 0.1,
    ) -> None:
        
        super().__init__()
        
        self.MHAttention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_head
        )
        
        self.norm_layer = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        attention_output, attention_weights = self.MHAttention(
            x=x,
            padding_mask=padding_mask
        )
        
        attention_output = self.dropout(attention_output)
        
        output = self.norm_layer(x + attention_output)
        
        return output, attention_weights