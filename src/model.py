import torch
from torch import nn

class ProteinTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        max_seq_length: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of heads."
            )
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )
        
        self.max_seq_length = max_seq_length
        
        self.positional_encoding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.classifier = nn.Linear(
            in_features=embedding_dim, 
            out_features=num_classes)

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size, seq_length = sequences.size()
        
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds the maximum allowed length {self.max_seq_length}."
            )
            
        padding_mask = sequences == 0
        
        positions = torch.arange(
            seq_length,
            device=sequences.device
        )
        
        token_embeddings = self.embedding(sequences)
        position_embeddings = self.positional_encoding(positions)
        
        embedding = token_embeddings + position_embeddings
        
        x = self.dropout(embedding)
        encoded_output = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        
        valid_token_mask = ~padding_mask
        valid_token_mask = valid_token_mask.unsqueeze(-1)
        
        masked_encoded_output = encoded_output * valid_token_mask
        sum_embeddings = masked_encoded_output.sum(dim=1)
        valid_token_counts = valid_token_mask.sum(dim=1).clamp(min=1)
        
        pooled = sum_embeddings / valid_token_counts
        
        logits = self.classifier(pooled)
        
        return logits
        
        
