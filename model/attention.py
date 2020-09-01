import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    a = softmax(q^TWk)
    """

    def __init__(self, embed_dim, temperature: float=1.0, dropout: float=0.0):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        # fully-connected network to process `key`
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.fc2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_weight_dropout = nn.Dropout(dropout)
        self.attn_output_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, padding_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            query(torch.Tensor): shape (batch_size, num_queries, embed_dim)
            key(torch.Tensor): shape (batch_size, num_keys, embed_dim)
            padding_mask(torch.Tensor): shape (batch_size, num_keys)

        Returns:
            attn_output(torch.Tensor): shape (batch_size, num_queries, embed_dim)
            attn_weight(torch.Tensor): shape (batch_size, num_queries, num_keys)
        """
        # compute attention weights
        batch_size, num_queries, num_keys = query.shape[0], query.shape[1], key.shape[1]
        attn_weight = torch.matmul(query, torch.transpose(self.fc(key), 1, 2))
        # apply temperature
        attn_weight /= self.temperature
        # scale attention weights
        # if self.scaled:
        #     attn_weight *= self.embed_dim ** -0.5
        assert attn_weight.shape == torch.Size([batch_size, num_queries, num_keys])
        # mask attention weights
        attn_weight = attn_weight.masked_fill_(padding_mask.unsqueeze(1), float('-inf'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_weight_dropout(attn_weight)

        # attention pool
        attn_output = torch.matmul(attn_weight, key)
        # attn_output = torch.matmul(attn_weight, self.fc2(key))
        assert attn_output.shape == torch.Size([batch_size, num_queries, self.embed_dim])
        attn_output = self.attn_output_dropout(attn_output)
        return attn_output, attn_weight


def dot_attention(query, key, padding_mask):
    """
    Args:
        query(torch.Tensor): shape (batch_size, num_queries, embed_dim)
        key(torch.Tensor): shape (batch_size, num_keys, embed_dim)
        padding_mask(torch.Tensor): shape (batch_size, num_keys)

    Returns:
        attn_output(torch.Tensor): shape (batch_size, num_queries, embed_dim)
        attn_weight(torch.Tensor): shape (batch_size, num_queries, num_keys)
    """
    batch_size, num_queries, num_keys, embed_dim = query.shape[0], query.shape[1], key.shape[1], key.shape[2]
    attn_weight = torch.matmul(query, torch.transpose(key, 1, 2))
    assert attn_weight.shape == torch.Size([batch_size, num_queries, num_keys])
    # mask attention weights
    attn_weight = attn_weight.masked_fill_(padding_mask.unsqueeze(1), float('-inf'))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attention pool
    attn_output = torch.matmul(attn_weight, key)
    assert attn_output.shape == torch.Size([batch_size, num_queries, embed_dim])
    return attn_output, attn_weight

class AttentionPool(nn.Module):

    def __init__(self, embed_dim, num_queries, dropout=0.1):
        super(AttentionPool, self).__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.rand(1, num_queries, embed_dim // num_queries))
        self.fc_list = nn.ModuleList([nn.Linear(embed_dim, embed_dim // num_queries, bias=False) for _ in range(num_queries)])
        self.fc_output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_weight_dropout = nn.Dropout(dropout)
        self.attn_output_dropout = nn.Dropout(dropout)

    def forward(self, input_tensor: torch.Tensor, padding_mask: torch.Tensor, return_attn_weights=False):
        """
        Args:
            input_tensor(torch.Tensor): shape (batch_size, sequence_length, embed_dim)
            padding_mask(torch.Tensor): shape (batch_size, sequence_length)

        Returns:
            output(torch.Tensor): shape (batch_size, embed_dim)
        """
        batch_size, sequence_length = input_tensor.shape[0], input_tensor.shape[1]
        # compute attention weights
        key = torch.cat([fc(input_tensor).unsqueeze(1) for fc in self.fc_list], dim=1)
        assert key.shape == torch.Size([batch_size, self.num_queries, sequence_length, self.embed_dim // self.num_queries])
        query = self.query.expand(batch_size, -1, -1).unsqueeze(-1)
        attn_weight = torch.matmul(key, query).squeeze(-1)
        assert attn_weight.shape == torch.Size([batch_size, self.num_queries, sequence_length])
        if return_attn_weights == True:
            return attn_weight

        # mask attention weights
        attn_weight = attn_weight.masked_fill_(padding_mask.unsqueeze(1), float('-inf'))
        # attn_weight *= (self.embed_dim / self.num_queries) ** -0.5
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if return_attn_weights == True:
            return attn_weight
        attn_weight = self.attn_weight_dropout(attn_weight)

        # attention pool
        attn_output = torch.matmul(torch.transpose(key, 2, 3), attn_weight.unsqueeze(-1)).squeeze(-1)
        assert attn_output.shape == torch.Size([batch_size, self.num_queries, self.embed_dim // self.num_queries])
        attn_output = attn_output.reshape(batch_size, -1)
        attn_output = self.fc_output(attn_output)
        attn_output = self.attn_output_dropout(attn_output)
        assert attn_output.shape == torch.Size([batch_size, self.embed_dim])

        return attn_output

        # output = []
        # for fc in self.fc_list:
        #     key = fc(input_tensor)
        #     assert key.shape == torch.Size([batch_size, sequence_length, self.embed_dim // self.num_queries])
        #     attn_output, _ = dot_attention(self.query.expand(batch_size, -1, -1), key, padding_mask)
        #     assert attn_output.shape == torch.Size([batch_size, self.num_queries, self.embed_dim // self.num_queries])
        #     output.append(attn_output)
        # output = torch.cat(output, dim=-1)
        # assert output.shape == torch.Size([batch_size, self.embed_dim])
        # return outputjiungdong
