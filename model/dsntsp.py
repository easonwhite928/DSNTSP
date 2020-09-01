import torch
import torch.nn as nn
import torch.nn.functional as F

from data.temporal_sets_data_loader import TemporalSetsInput
from model.attention import Attention, AttentionPool
from model.time_encoding import PositionalEncoding, TimestampEncoding
from model.functions import itm, stm, pad_sequence, time_encode, set_embedding, din
from model.co_transformer import CDTE, PDTE, DualTransformer, CoTransformerLayer, CoTransformerLayer2, CoTransformerLayer3, CoTransformer


class DSNTSP(nn.Module):
    """
    DSNTSP(Dual Sequential Network for Temporal Sets Prediction)
    """

    def __init__(self,
                 items_total: int,
                 embed_dim: int,
                 time_encoding: str,
                 set_embed_method: str,
                 num_set_embeds: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 itm_temperature: float=1.0,
                 stm_temperature: float=1.0,
                 dropout: float=0.1,
                 set_embed_dropout: float = 0.1,
                 attn_output: bool=True):
        """

        Args:
            items_total:
            embed_dim:
            time_encoding:
            set_embed_method:
            num_transformer_heads:
            num_set_embeds:
            dropout:
        """
        super(DSNTSP, self).__init__()
        self.items_total = items_total
        self.embed_dim = embed_dim
        self.item_embed = nn.Embedding(num_embeddings=items_total, embedding_dim=embed_dim)

        # time encoding
        self.time_encoding_method = time_encoding
        if time_encoding == 'none':
            self.time_encoding = None
        elif time_encoding == 'positional':
            self.time_encoding = PositionalEncoding(embed_dim=embed_dim)
        elif time_encoding == 'timestamp':
            self.time_encoding = TimestampEncoding(embed_dim=embed_dim)
        else:
            raise NotImplementedError()

        # set embedding
        self.set_embed_method = set_embed_method
        if set_embed_method == 'attn_pool':
            self.set_embed = AttentionPool(embed_dim, num_queries=num_set_embeds, dropout=set_embed_dropout)
        else:
            raise NotImplementedError()

        # Dual Transformer
        # self.dual_transformer = DualTransformer(layers=[
        #     PDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     CDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     PDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     CDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     PDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     CDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     PDTE(embed_dim, num_transformer_heads, dropout=dropout),
        #     CDTE(embed_dim, num_transformer_heads, dropout=dropout)
        # ])

        # co-transformer
        self.co_transformer = CoTransformer(layer=CoTransformerLayer(embed_dim, num_transformer_heads, dropout=dropout), num_layers=num_transformer_layers)

        # attention-based prediction
        self.item_attn = Attention(embed_dim=embed_dim, temperature=itm_temperature)
        self.set_attn = Attention(embed_dim=embed_dim, temperature=stm_temperature)

        self.items_bias = nn.Parameter(torch.zeros(items_total))

        # gate network
        if attn_output:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.Sigmoid()
            )
        else:
            self.gate_net = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

        self.attn_output = attn_output

    def forward(self, input_batch: TemporalSetsInput, return_fusion_weights: bool=False):
        """
        Args:
            input_batch (TemporalSetsInput):
        Returns:
            output (Tensor): (batch_size, items_total)
        """
        items_seqs = input_batch.get_items()
        sets_seqs = input_batch.get_sets()
        item_times_seqs = input_batch.get_item_times()
        set_times_seqs = input_batch.get_set_times()

        # get padded items sequences with time encoding
        items_seqs = [self.item_embed(items) for items in items_seqs]
        padded_items_seqs, items_padding_mask = pad_sequence(items_seqs)
        padded_item_times_seqs, _ = pad_sequence(item_times_seqs)
        padded_items_seqs = time_encode(self.time_encoding_method, self.time_encoding, padded_items_seqs, padded_item_times_seqs)

        # get padded set embeddings sequences with time encoding
        sets_seqs = [[self.item_embed(user_set) for user_set in sets] for sets in sets_seqs]
        set_embed_seqs = set_embedding(sets_seqs, self.set_embed)
        padded_set_embed_seqs, sets_padding_mask = pad_sequence(set_embed_seqs)
        padded_set_times_seqs, _ = pad_sequence(set_times_seqs)
        padded_set_embed_seqs = time_encode(self.time_encoding_method, self.time_encoding, padded_set_embed_seqs, padded_set_times_seqs)

        # items_output, sets_output = self.dual_transformer(padded_items_seqs, padded_set_embed_seqs, items_padding_mask, sets_padding_mask)
        padded_items_seqs = torch.transpose(padded_items_seqs, 0, 1)
        padded_set_embed_seqs = torch.transpose(padded_set_embed_seqs, 0, 1)
        items_output, sets_output = self.co_transformer(padded_items_seqs, padded_set_embed_seqs, items_padding_mask, sets_padding_mask)
        items_output = torch.transpose(items_output, 0, 1)
        sets_output = torch.transpose(sets_output, 0, 1)

        if self.attn_output:
            # get item-level and set-level user embeddings
            item_user_embed = din(items_output, items_padding_mask, self.item_embed.weight, self.item_attn)
            set_user_embed = din(sets_output, sets_padding_mask, self.item_embed.weight, self.set_attn)

            assert item_user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            assert set_user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])

            # fusion item-level and set-level user embeddings and predict scores
            item_weight = self.item_embed.weight.unsqueeze(0).expand(input_batch.batch_size, -1, -1)
            assert item_weight.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            tmp = torch.cat([item_user_embed, set_user_embed, item_weight], dim=-1)
            assert tmp.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim * 3])
            gate = self.gate_net(tmp)
            assert gate.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            ones = gate.new_ones((input_batch.batch_size, self.items_total, self.embed_dim), dtype=torch.float)

            if return_fusion_weights:
                return ones - gate, gate

            user_embed = (ones - gate) * item_user_embed + gate * set_user_embed
            assert user_embed.shape == torch.Size([input_batch.batch_size, self.items_total, self.embed_dim])
            scores = (user_embed * self.item_embed.weight).sum(-1) + self.items_bias
        else:
            # items_output.shape -> (batch_size, sequence_length, embed_dim)
            # sets_output.shape -> (batch_size, sequence_length, embed_dim)
            item_user_embed = items_output.mean(dim=-2)
            set_user_embed = sets_output.mean(dim=-2)

            assert item_user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])
            assert set_user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])

            tmp = torch.cat([item_user_embed, set_user_embed], dim=-1)
            assert tmp.shape == torch.Size([input_batch.batch_size, self.embed_dim * 2])
            gate = self.gate_net(tmp)
            assert gate.shape == torch.Size([input_batch.batch_size, self.embed_dim])
            ones = gate.new_ones((input_batch.batch_size, self.embed_dim), dtype=torch.float)
            user_embed = (ones - gate) * item_user_embed + gate * set_user_embed
            assert user_embed.shape == torch.Size([input_batch.batch_size, self.embed_dim])

            scores = F.linear(user_embed, self.item_embed.weight, bias=self.items_bias)
            assert scores.shape == torch.Size([input_batch.batch_size, self.items_total])

        return scores
