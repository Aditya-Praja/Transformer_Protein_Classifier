import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.output = nn.Linear(embedding_dim, embedding_dim)
        
    def split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, sequence_length, _ = x.size()
        
        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        
        x = x.transpose(1, 2)
        
        return x
    
    def combine_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        batch_size, num_heads, sequence_length, head_dim = x.size()
        
        x = x.transpose(1, 2)
        
        x = x.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim
        )
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )
        
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        
        if padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                padding_mask[:, None, None, :],
                float('-inf')
            )
        
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )
        
        attention_output = torch.matmul(
            attention_weights,
            V,
        )
        
        attention_output = self.combine_heads(attention_output)
        
        attention_output = self.output(attention_output)
        
        return attention_output, attention_weights