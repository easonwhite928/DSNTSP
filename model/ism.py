from typing import List

import torch
import torch.nn as nn

from data.temporal_sets_data_loader import TemporalSetsInput
from model.functions import itm, compute_scores
from model.attention import Attention
from model.time_encoding import PositionalEncoding, TimestampEncoding


class ISM(nn.Module):
    """
    (ISM) Item-level Sequential Model
    """

    def __init__(self,
                 items_total: int,
                 embed_dim: int=256,
                 time_encoding='none',
                 transformer_num_heads: int=8,
                 transformer_num_layers: int=2,
                 temperature: float=1.0,
                 dropout: float=0.1):
        """
        Args:
            items_total:
            embed_dim:
            time_encoding: 'none'/'positional'/'timestamp'
            transformer_num_heads:
            transformer_num_layers:
            dropout:
        """
        super(ISM, self).__init__()
        self.items_total = items_total
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(num_embeddings=items_total, embedding_dim=embed_dim)
        self.time_encoding_method = time_encoding
        if time_encoding == 'none': self.time_encoding = None
        elif time_encoding == 'positional': self.time_encoding = PositionalEncoding(embed_dim=embed_dim)
        elif time_encoding == 'timestamp': self.time_encoding = TimestampEncoding(embed_dim=embed_dim)
        else: raise NotImplementedError()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                                                  nhead=transformer_num_heads,
                                                                                                  dropout=dropout),
                                                         num_layers=transformer_num_layers)
        self.attn = Attention(embed_dim=embed_dim, temperature=temperature)
        self.items_bias = nn.Parameter(torch.zeros(items_total))

    def forward(self, input_batch: TemporalSetsInput) -> torch.Tensor:
        """
        Args:
            input_batch (TemporalSetsInput):
        Returns:
            output (Tensor): (batch_size, items_total)
        """
        user_embeds = itm(input_batch.get_items(),
                          self.item_embed,
                          input_batch.get_item_times(),
                          self.time_encoding_method,
                          self.time_encoding,
                          self.transformer_encoder,
                          self.attn)
        scores = compute_scores(user_embeds, self.item_embed.weight, self.items_bias)
        return scores


