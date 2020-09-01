import torch
import torch.nn as nn

from data.temporal_sets_data_loader import TemporalSetsInput
from model.attention import Attention, AttentionPool
from model.time_encoding import PositionalEncoding, TimestampEncoding
from model.functions import stm, compute_scores
from model.pool import Pool


class SSM(nn.Module):
    """
    SSM (Set-level Sequential Model)
    """

    def __init__(self,
                 items_total: int,
                 embed_dim: int,
                 num_attn_queries,
                 time_encoding='none',
                 transformer_num_heads=8,
                 transformer_num_layers=2,
                 temperature: float=1.0,
                 dropout=0.1,
                 set_embed_dropout=0.1):
        """
        Args:
            pool(str): sum/mean/max
        """
        super(SSM, self).__init__()
        self.items_total = items_total
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(num_embeddings=items_total, embedding_dim=embed_dim)
        self.num_attn_queries = num_attn_queries
        if num_attn_queries == -1:
            self.attention_pool = Pool(pool='mean')
        else:
            self.attention_pool = AttentionPool(embed_dim, num_attn_queries, dropout=set_embed_dropout)
        self.time_encoding_method = time_encoding
        if time_encoding == 'none':
            self.time_encoding = None
        elif time_encoding == 'positional':
            self.time_encoding = PositionalEncoding(embed_dim=embed_dim)
        elif time_encoding == 'timestamp':
            self.time_encoding = TimestampEncoding(embed_dim=embed_dim)
        else:
            raise NotImplementedError()
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                                                  nhead=transformer_num_heads,
                                                                                                  dropout=dropout),
                                                         num_layers=transformer_num_layers)
        self.attn = Attention(embed_dim=embed_dim, temperature=temperature)
        self.items_bias = nn.Parameter(torch.zeros(items_total))

    def forward(self, input_batch: TemporalSetsInput):
        """
        Args:
            input_batch (TemporalSetsInput):
        Returns:
            output (Tensor): (batch_size, items_total)
        """
        user_embeds = stm(sets_seqs=input_batch.get_sets(),
                          item_embed=self.item_embed,
                          set_embed_module=self.attention_pool,
                          times_seqs=input_batch.get_set_times(),
                          time_encoding_method=self.time_encoding_method,
                          time_encoding=self.time_encoding,
                          transformer_encoder=self.transformer_encoder,
                          attn=self.attn)
        scores = compute_scores(user_embeds, self.item_embed.weight, self.items_bias)
        return scores