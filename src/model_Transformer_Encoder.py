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
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            AttentionSublayer(
                embedding_dim=embedding_dim,
                num_head=num_heads,
                dropout=dropout
            ),
            FeedForwardSublayer(
                embedding_dim=embedding_dim,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        attention_weights_list = []
        
        for layer in self.layers:
            x, attention_weights = layer(
                x=x,
                padding_mask=padding_mask
            )
            attention_weights_list.append(attention_weights)
        
        return x, attention_weights_list