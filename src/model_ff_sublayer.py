import torch
from torch import nn

from src.model_feed_forward import FeedForwardNetwork

class FeedForwardSublayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        
        super().__init__() 
        
        self.FFN = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm_layer = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        ffn_output = self.FFN(x)
        
        ffn_output = self.dropout(ffn_output)
        
        output = self.norm_layer(x + ffn_output)
        
        return output