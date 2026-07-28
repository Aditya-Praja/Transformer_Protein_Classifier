import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )
        
        attention_scores = attention_scores / (self.embedding_dim ** 0.5)