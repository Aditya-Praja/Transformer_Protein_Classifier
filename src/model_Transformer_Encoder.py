import torch
from torch import nn

from src.model_attention_sublayer import AttentionSublayer
from src.model_ff_sublayer import FeedForwardSublayer

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.attention_sublayer = AttentionSublayer(
            embedding_dim=embedding_dim,
            num_head=num_heads,
            dropout=dropout
        )
        
        self.feedforward_sublayer = FeedForwardSublayer(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        attention_output, attention_weights = self.attention_sublayer(
            x,
            padding_mask,
        )
        
        final_output = self.feedforward_sublayer(attention_output)
        
        return final_output, attention_weights