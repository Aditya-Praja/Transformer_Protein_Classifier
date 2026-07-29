import torch
from torch import nn

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
       
        self.ffn = nn.Sequential(
           nn.Linear(embedding_dim, feedforward_dim),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(feedforward_dim, embedding_dim),
       )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        x = self.ffn(x)
        return x