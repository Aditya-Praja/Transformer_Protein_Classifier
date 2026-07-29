import torch
from torch import nn

from src.model_input_encoding import InputEncoding
from src.model_Transformer_Encoder import TransformerEncoder

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
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of heads."
            )
        
        self.padding_idx = padding_idx
        
        self.input_embedding = InputEncoding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )
        

    def forward(
        self,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        
        padding_mask = (sequences == self.padding_idx)
        
        x = self.input_embedding(sequences)
        
        x, attention_weights_list = self.encoder(
            x=x,
            padding_mask=padding_mask,
        )
        
        valid_token_mask = ~padding_mask
        valid_token_mask = valid_token_mask.unsqueeze(-1)
        
        masked_x = x * valid_token_mask
        pooled_x = masked_x.sum(dim=1) / valid_token_mask.sum(dim=1).clamp(min=1)
        
        logits = self.classifier(pooled_x)
        
        return logits, attention_weights_list