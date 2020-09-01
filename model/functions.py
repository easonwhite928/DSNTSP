from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

from model.attention import Attention


def pad_sequence(sequence: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
    """
    Args:
        sequence(List[torch.Tensor]):

    Returns:
        padded_sequence(torch.Tensor)
        padding_mask(torch.Tensor)
    """
    lengths = [seq.shape[0] for seq in sequence]
    padded_sequence = rnn.pad_sequence(sequence, batch_first=True, padding_value=-1)
    batch_size, sequence_length = padded_sequence.shape[0], padded_sequence.shape[1]

    padding_mask = padded_sequence.new_ones((batch_size, sequence_length), dtype=torch.bool)
    for idx in range(batch_size): padding_mask[idx][:lengths[idx]] = False
    assert padding_mask.shape == torch.Size([batch_size, sequence_length])

    return padded_sequence, padding_mask


def din(seqs: torch.Tensor, padding_mask: torch.Tensor, item_embeds: torch.Tensor, attn_module: Attention):
    """
    Args:
        seqs (Tensor): shape (batch_size, sequence_length, embed_dim)
        padding_mask (Tensor): shape (batch_size, sequence_length)
        item_embeds (Tensor): shape (items_total, embed_dim)
        attn_module (Attention):

    Returns:
        attn_output (Tensor): shape (batch_size, items_total, embed_dim)
        """

    item_embeds = item_embeds.unsqueeze(0).expand(seqs.shape[0], -1, -1)
    attn_output, _ = attn_module(item_embeds, seqs, padding_mask=padding_mask)
    return attn_output


def compute_scores(users_embeds: torch.Tensor, items_embeds: torch.Tensor, items_bias: torch.Tensor) -> torch.Tensor:
    """
    Args:
        users_embeds(torch.Tensor): shape (batch_size, items_total, embed_dim)
        items_embeds(torch.Tensor): shape (items_total, embed_dim)
        items_bias(torch.Tensor): shape (items_total)

    Returns:
        scores(torch.Tensor): shape (batch_size, items_total)
    """
    scores = (users_embeds * items_embeds).sum(-1) + items_bias
    return scores


def time_encode(time_encoding_method: str,
                time_encoding: nn.Module,
                padded_items_seqs: torch.Tensor,
                padded_times_seqs: torch.Tensor) -> torch.Tensor:
    if time_encoding_method == 'none':
        output = padded_items_seqs
    elif time_encoding_method == 'positional':
        output = time_encoding(padded_items_seqs)
    elif time_encoding_method == 'timestamp':
        output = time_encoding(padded_items_seqs, padded_times_seqs)
    else:
        raise NotImplementedError()
    return output



def itm(items_seqs: List[torch.Tensor],
        item_embed: nn.Embedding,
        times_seqs: List[torch.Tensor],
        time_encoding_method: str,
        time_encoding: nn.Module,
        transformer_encoder: nn.TransformerEncoder,
        attn: Attention) -> torch.Tensor:
    """
    Args:
        items_seqs:
        item_embed:
        time_encoding_method:
        time_encoding:
        transformer_encoder:
        attn:

    Returns:
        user_embeds(torch.Tensor): shape (batch_size, items_total, embed_dim)
    """
    # item embedding
    items_seqs = [item_embed(items) for items in items_seqs]

    # pad item sequences
    padded_items_seqs, padding_mask = pad_sequence(items_seqs)
    padded_times_seqs, _ = pad_sequence(times_seqs)

    # print(f'itm {padded_items_seqs.shape}')

    # time encoding
    transformer_input = time_encode(time_encoding_method, time_encoding, padded_items_seqs, padded_times_seqs)

    # transformer
    transformer_input = torch.transpose(transformer_input, 0, 1)
    transformer_output = transformer_encoder(transformer_input, src_key_padding_mask=padding_mask)
    transformer_output = torch.transpose(transformer_output, 0, 1)
    user_embeds = din(transformer_output, padding_mask, item_embed.weight, attn)
    return user_embeds


def set_embedding(sets_seqs: List[List[torch.Tensor]], set_embed_module: nn.Module) -> List[torch.Tensor]:
    sets = []
    sets_length = []
    for sets_seq in sets_seqs:
        sets.extend(sets_seq)
        sets_length.append(len(sets_seq))
    sets_batch_size = len(sets)

    padded_sets, padding_mask = pad_sequence(sets)
    set_embeds = set_embed_module(padded_sets, padding_mask)
    assert set_embeds.shape == torch.Size([sets_batch_size, set_embeds.shape[1]])

    set_embed_seqs = torch.split(set_embeds, sets_length)
    return set_embed_seqs


def stm(sets_seqs: List[List[torch.Tensor]],
        item_embed: nn.Embedding,
        set_embed_module: nn.Module,
        times_seqs: List[torch.Tensor],
        time_encoding_method: str,
        time_encoding: nn.Module,
        transformer_encoder: nn.TransformerEncoder,
        attn: Attention) -> torch.Tensor:
    sets_seqs = [[item_embed(user_set) for user_set in sets] for sets in sets_seqs]

    set_embed_seqs = set_embedding(sets_seqs, set_embed_module)
    padded_set_embed_seqs, padding_mask = pad_sequence(set_embed_seqs)

    # print(f'stm {padded_set_embed_seqs.shape}')

    padded_times_seqs, _ = pad_sequence(times_seqs)

    # time encoding
    transformer_input = time_encode(time_encoding_method, time_encoding, padded_set_embed_seqs, padded_times_seqs)

    # transformer
    transformer_input = torch.transpose(transformer_input, 0, 1)
    transformer_output = transformer_encoder(transformer_input, src_key_padding_mask=padding_mask)
    transformer_output = torch.transpose(transformer_output, 0, 1)
    user_embeds = din(transformer_output, padding_mask, item_embed.weight, attn)
    return user_embeds
