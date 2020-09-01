import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, seqs: torch.Tensor):
        """
        Args:
            seqs(nn.Tensor): shape (batch_size, sequence_length, embed_dim)
        Returns:
            output(nn.Tensor): shape (batch_size, sequence_length, embed_dim)
        """
        sequence_length = seqs.shape[1]
        position_embed = seqs.new_zeros((sequence_length, self.embed_dim))
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)  # position -> (sequence_length, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))  # div_term -> (embed_dim / 2,)
        position_embed[:, 0::2] = torch.sin(position * div_term)  # position * div_term -> (sequence_length, embed_dim / 2)
        position_embed[:, 1::2] = torch.cos(position * div_term)
        position_embed = position_embed.unsqueeze(0)
        output = seqs + position_embed
        return output


class TimestampEncoding(nn.Module):

    def __init__(self, embed_dim: int):
        super(TimestampEncoding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, seqs: torch.Tensor, times: torch.Tensor):
        """
        Args:
            seqs(nn.Tensor): shape (batch_size, sequence_length, embed_dim)
            times(nn.Tensor): shape (batch_size, sequence_length)
        Returns:
            output(nn.Tensor): shape (batch_size, sequence_length, embed_dim)
        """
        batch_size, sequence_length = seqs.shape[0], seqs.shape[1]
        timestamp_embed = seqs.new_zeros((batch_size, sequence_length, self.embed_dim))
        timestamp = times.unsqueeze(-1).float()  # position -> (batch_size, sequence_length, 1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim))  # div_term -> (embed_dim / 2,)
        div_term = div_term.to(times.device)
        # print(timestamp.device)
        # print(div_term.device)
        timestamp_embed[:, :, 0::2] = torch.sin(timestamp * div_term)  # timestamp * div_term -> (batch_size, sequence_length, embed_dim / 2)
        timestamp_embed[:, :, 1::2] = torch.cos(timestamp * div_term)
        # timestamp_embed = timestamp_embed.unsqueeze(0)
        output = seqs + timestamp_embed
        return output


if __name__ == '__main__':
    batch_size, sequence_length, embed_dim = 3, 4, 6
    seqs = torch.zeros(batch_size, sequence_length, embed_dim)
    pe = PositionalEncoding(embed_dim=embed_dim)
    print(pe(seqs).shape)
    times = torch.tensor([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
        [0, 2, 1, 3]
    ])
    te = TimestampEncoding(embed_dim=embed_dim)
    print(te(seqs, times).shape)
