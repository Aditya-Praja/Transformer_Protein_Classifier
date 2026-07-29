from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        max_seq_length: int
    ) -> None:
        super().__init__()
        
        
        
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        