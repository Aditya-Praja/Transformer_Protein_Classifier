from torch import nn
import torch

class InputEncoding(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int, 
        padding_idx: int,
        max_seq_length: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.max_seq_length = max_seq_length
        
        self.positional_encoding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        _, seq_length, _ = x.size()
        
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum "
                f"sequence length {self.max_seq_length}."
            )
        
        positions = torch.arange(
            seq_length, 
            device=x.device
        )
        
        token_embeddings = self.embedding(x)
        
        positional_encodings = self.positional_encoding(positions)
        
        x = token_embeddings + positional_encodings
        
        x = self.dropout(x)
        
        return x 
    
    
    