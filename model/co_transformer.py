from typing import List
import copy

import torch
import torch.nn as nn


class CDTE(nn.Module):
    """
    Cross Dual-Transformer Encoder Layer
    """

    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CDTE, self).__init__()
        self.embed_dim = embed_dim

        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm11 = nn.LayerNorm(embed_dim)
        self.layer_norm21 = nn.LayerNorm(embed_dim)

        self.ffw1 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffw2 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm12 = nn.LayerNorm(embed_dim)
        self.layer_norm22 = nn.LayerNorm(embed_dim)

    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (batch_size, sequence_length1, embed_dim)
            seqs2: (batch_size, sequence_length2, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (batch_size, sequence_length1, embed_dim)
            output2: (batch_size, sequence_length2, embed_dim)
        """
        seqs1 = torch.transpose(seqs1, 0, 1)
        seqs2 = torch.transpose(seqs2, 0, 1)

        attn_output1, _ = self.attn1(seqs1, seqs2, seqs2, key_padding_mask=padding_mask2)
        output1 = self.layer_norm11(seqs1 + self.dropout1(attn_output1))
        output1 = self.layer_norm12(output1 + self.ffw1(output1))

        attn_output2, _ = self.attn2(seqs2, seqs1, seqs1, key_padding_mask=padding_mask1)
        output2 = self.layer_norm21(seqs2 + self.dropout2(attn_output2))
        output2 = self.layer_norm22(output2 + self.ffw2(output2))

        output1 = torch.transpose(output1, 0, 1)
        output2 = torch.transpose(output2, 0, 1)

        return output1, output2


import torch
import torch.nn as nn

class PDTE(nn.Module):
    """
    Parallel Dual-Transformer Encoder Layer
    """

    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(PDTE, self).__init__()
        self.transformer1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)

    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (batch_size, sequence_length1, embed_dim)
            seqs2: (batch_size, sequence_length2, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (batch_size, sequence_length1, embed_dim)
            output2: (batch_size, sequence_length2, embed_dim)
        """
        seqs1 = torch.transpose(seqs1, 0, 1)
        seqs2 = torch.transpose(seqs2, 0, 1)

        output1 = self.transformer1(seqs1, src_key_padding_mask=padding_mask1)
        output2 = self.transformer2(seqs2, src_key_padding_mask=padding_mask2)

        output1 = torch.transpose(output1, 0, 1)
        output2 = torch.transpose(output2, 0, 1)

        return output1, output2


class DualTransformer(nn.Module):
    """
    Dual Transformer

    Examples:
        dual_transformer = DualTransformer([
                PDTE(512, 8),
                CDTE(512, 8),
                PDTE(512, 8),
                CDTE(512, 8),
            ])
        output1, output2 = dual_transformer(seqs1, seqs2, padding_mask1, padding_mask2)
    """

    def __init__(self, layers: List):
        super(DualTransformer, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (batch_size, sequence_length1, embed_dim)
            seqs2: (batch_size, sequence_length2, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (batch_size, sequence_length1, embed_dim)
            output2: (batch_size, sequence_length2, embed_dim)
        """
        output = (seqs1, seqs2)
        for layer in self.layers:
            output = layer(output[0], output[1], padding_mask1, padding_mask2)

        return output


class CoTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CoTransformerLayer, self).__init__()
        self.embed_dim = embed_dim

        # cross transformer
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm11 = nn.LayerNorm(embed_dim)
        self.layer_norm21 = nn.LayerNorm(embed_dim)

        self.ffw1 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffw2 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm12 = nn.LayerNorm(embed_dim)
        self.layer_norm22 = nn.LayerNorm(embed_dim)

        # self.fusion_net1 = nn.Sequential(
        #     nn.Linear(embed_dim, dim_feedforward),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, embed_dim),
        #     nn.Dropout(dropout)
        # )
        #
        # self.fusion_net2 = nn.Sequential(
        #     nn.Linear(embed_dim, dim_feedforward),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, embed_dim),
        #     nn.Dropout(dropout)
        # )

        # self.fusion_net1 = nn.Sequential(
        #     nn.Linear(2 * embed_dim, embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout)
        # )
        #
        # self.fusion_net2 = nn.Sequential(
        #     nn.Linear(2 * embed_dim, embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout)
        # )

        # self.fusion_norm1 = nn.LayerNorm(embed_dim)
        # self.fusion_norm2 = nn.LayerNorm(embed_dim)

        # parallel transformer
        self.transformer1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)

        # fusion gate
        # self.gate_net = nn.Linear(2 * embed_dim, embed_dim)

        # fusion gate
        self.fusion_gate_1 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.fusion_gate_2 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (sequence_length1, batch_size, embed_dim)
            seqs2: (sequence_length2, batch_size, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (sequence_length1, batch_size, embed_dim)
            output2: (sequence_length2, batch_size, embed_dim)
        """
        batch_size, sequence_length1 = seqs1.shape[1], seqs1.shape[0]
        sequence_length2 = seqs2.shape[0]

        # cross transformer
        attn_output1, _ = self.attn1(seqs1, seqs2, seqs2, key_padding_mask=padding_mask2)
        cross_output1 = self.layer_norm11(seqs1 + self.dropout1(attn_output1))
        cross_output1 = self.layer_norm12(cross_output1 + self.ffw1(cross_output1))

        attn_output2, _ = self.attn2(seqs2, seqs1, seqs1, key_padding_mask=padding_mask1)
        cross_output2 = self.layer_norm21(seqs2 + self.dropout2(attn_output2))
        cross_output2 = self.layer_norm22(cross_output2 + self.ffw2(cross_output2))

        # parallel transformer
        parallel_output1 = self.transformer1(seqs1, src_key_padding_mask=padding_mask1)
        parallel_output2 = self.transformer2(seqs2, src_key_padding_mask=padding_mask2)

        assert cross_output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert parallel_output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert cross_output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])
        assert parallel_output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        # fusion cross & parallel
        # output1 = self.fusion_norm1(parallel_output1 + self.fusion_net1(torch.cat([parallel_output1, cross_output1], dim=-1)))
        # output2 = self.fusion_norm2(parallel_output2 + self.fusion_net2(torch.cat([parallel_output2, cross_output2], dim=-1)))

        # output1 = self.fusion_norm1(parallel_output1 + self.fusion_net1(torch.cat([parallel_output1, cross_output1], dim=-1)))
        # output2 = self.fusion_norm2(parallel_output2 + self.fusion_net2(torch.cat([parallel_output2, cross_output2], dim=-1)))

        # fusion cross & parallel
        gate1 = self.fusion_gate_1(torch.cat([cross_output1, parallel_output1], dim=-1))
        gate2 = self.fusion_gate_2(torch.cat([cross_output2, parallel_output2], dim=-1))

        output1 = (-gate1 + 1.0) * cross_output1 + gate1 * parallel_output1
        output2 = (-gate2 + 1.0) * cross_output2 + gate2 * parallel_output2

        # fusion ffw
        # output1 = self.fusion_norm1(output1 + self.fusion_net1(output1))
        # output2 = self.fusion_norm2(output2 + self.fusion_net2(output2))

        assert output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        return output1, output2


class CoTransformerLayer2(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CoTransformerLayer2, self).__init__()
        self.embed_dim = embed_dim

        # parallel attention
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)

        # cross attention
        self.co_attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        self.co_attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.co_dropout1 = nn.Dropout(dropout)
        self.co_dropout2 = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.co_layer_norm1 = nn.LayerNorm(embed_dim)
        self.co_layer_norm2 = nn.LayerNorm(embed_dim)

        # self.ffw1 = nn.Sequential(
        #     nn.Linear(2 * embed_dim, dim_feedforward),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, embed_dim),
        #     nn.Dropout(dropout)
        # )
        #
        # self.ffw2 = nn.Sequential(
        #     nn.Linear(2 * embed_dim, dim_feedforward),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, embed_dim),
        #     nn.Dropout(dropout)
        # )

        # fusion
        self.fusion_1 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.fusion_2 = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.fusion_ffw_1 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.fusion_ffw_2 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.output_layer_norm1 = nn.LayerNorm(embed_dim)
        self.output_layer_norm2 = nn.LayerNorm(embed_dim)


    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (sequence_length1, batch_size, embed_dim)
            seqs2: (sequence_length2, batch_size, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (sequence_length1, batch_size, embed_dim)
            output2: (sequence_length2, batch_size, embed_dim)
        """
        batch_size, sequence_length1 = seqs1.shape[1], seqs1.shape[0]
        sequence_length2 = seqs2.shape[0]

        # parallel attention
        attn_output1, _ = self.attn1(seqs1, seqs1, seqs1, key_padding_mask=padding_mask1)
        attn_output1 = self.layer_norm1(seqs1 + self.dropout1(attn_output1))

        attn_output2, _ = self.attn2(seqs2, seqs2, seqs2, key_padding_mask=padding_mask2)
        attn_output2 = self.layer_norm2(seqs2 + self.dropout2(attn_output2))

        # cross attention
        co_attn_output1, _ = self.co_attn1(seqs1, seqs2, seqs2, key_padding_mask=padding_mask2)
        co_attn_output1 = self.co_layer_norm1(seqs1 + self.co_dropout1(co_attn_output1))

        co_attn_output2, _ = self.co_attn2(seqs2, seqs1, seqs1, key_padding_mask=padding_mask1)
        co_attn_output2 = self.co_layer_norm2(seqs2 + self.co_dropout2(co_attn_output2))

        assert attn_output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert attn_output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])
        assert co_attn_output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert co_attn_output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        # fusion
        gate_1 = self.fusion_1(torch.cat([attn_output1, co_attn_output1], dim=-1))
        gate_2 = self.fusion_2(torch.cat([attn_output2, co_attn_output2], dim=-1))

        output1 = gate_1 * attn_output1 + (1 - gate_1) * co_attn_output1
        output2 = gate_2 * attn_output2 + (1 - gate_2) * co_attn_output2

        output1 = self.output_layer_norm1(output1 + self.fusion_ffw_1(output1))
        output2 = self.output_layer_norm2(output2 + self.fusion_ffw_2(output2))

        # output1 = self.output_layer_norm1(attn_output1 + self.ffw1(torch.cat([attn_output1, co_attn_output1], dim=-1)))
        # output2 = self.output_layer_norm2(attn_output2 + self.ffw2(torch.cat([attn_output2, co_attn_output2], dim=-1)))

        # output1 = self.output_layer_norm1(attn_output1 + co_attn_output1 + self.ffw1(torch.cat([attn_output1, co_attn_output1], dim=-1)))
        # output2 = self.output_layer_norm2(attn_output2 + co_attn_output2 + self.ffw2(torch.cat([attn_output2, co_attn_output2], dim=-1)))

        assert output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        return output1, output2


class CoTransformerLayer3(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super(CoTransformerLayer3, self).__init__()
        self.embed_dim = embed_dim

        # parallel attention
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           dropout=dropout)

        # cross attention
        self.co_attn1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)
        self.co_attn2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ffw_1 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.ffw_2 = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

        self.output_layer_norm1 = nn.LayerNorm(embed_dim)
        self.output_layer_norm2 = nn.LayerNorm(embed_dim)


    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (sequence_length1, batch_size, embed_dim)
            seqs2: (sequence_length2, batch_size, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (sequence_length1, batch_size, embed_dim)
            output2: (sequence_length2, batch_size, embed_dim)
        """
        batch_size, sequence_length1 = seqs1.shape[1], seqs1.shape[0]
        sequence_length2 = seqs2.shape[0]

        attn_output1, _ = self.attn1(seqs1, seqs1, seqs1, key_padding_mask=padding_mask1)
        attn_output2, _ = self.attn2(seqs2, seqs2, seqs2, key_padding_mask=padding_mask2)
        co_attn_output1, _ = self.co_attn1(seqs1, seqs2, seqs2, key_padding_mask=padding_mask2)
        co_attn_output2, _ = self.co_attn2(seqs2, seqs1, seqs1, key_padding_mask=padding_mask1)

        attn_output1 = self.layer_norm1(seqs1 + self.dropout1(attn_output1 + co_attn_output1))
        attn_output2 = self.layer_norm2(seqs2 + self.dropout2(attn_output2 + co_attn_output2))

        assert attn_output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert attn_output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        output1 = self.output_layer_norm1(attn_output1 + self.ffw_1(attn_output1))
        output2 = self.output_layer_norm2(attn_output2 + self.ffw_2(attn_output2))

        # output1 = self.output_layer_norm1(attn_output1 + self.ffw1(torch.cat([attn_output1, co_attn_output1], dim=-1)))
        # output2 = self.output_layer_norm2(attn_output2 + self.ffw2(torch.cat([attn_output2, co_attn_output2], dim=-1)))

        # output1 = self.output_layer_norm1(attn_output1 + co_attn_output1 + self.ffw1(torch.cat([attn_output1, co_attn_output1], dim=-1)))
        # output2 = self.output_layer_norm2(attn_output2 + co_attn_output2 + self.ffw2(torch.cat([attn_output2, co_attn_output2], dim=-1)))

        assert output1.shape == torch.Size([sequence_length1, batch_size, self.embed_dim])
        assert output2.shape == torch.Size([sequence_length2, batch_size, self.embed_dim])

        return output1, output2


class CoTransformer(nn.Module):
    """
    Co-Transformer

    Examples:
        co_transformer = CoTransformer(CoTransformerLayer(512, 8), num_layers=4)
        output1, output2 = co_transformer(seqs1, seqs2, padding_mask1, padding_mask2)
    """

    def __init__(self, layer, num_layers=1):
        super(CoTransformer, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, seqs1: torch.Tensor, seqs2: torch.Tensor, padding_mask1: torch.Tensor, padding_mask2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            seqs1: (sequence_length1, batch_size, embed_dim)
            seqs2: (sequence_length2, batch_size, embed_dim)
            padding_mask1: (batch_size, sequence_length1)
            padding_mask2: (batch_size, sequence_length2)

        Returns:
            output1: (sequence_length1, batch_size, embed_dim)
            output2: (sequence_length2, batch_size, embed_dim)
        """
        output = (seqs1, seqs2)
        for layer in self.layers:
            output = layer(output[0], output[1], padding_mask1, padding_mask2)

        return output
