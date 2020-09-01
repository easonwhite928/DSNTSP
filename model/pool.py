import torch
import torch.nn as nn


class Pool(nn.Module):

    def __init__(self, pool: str='sum'):
        """
        Args:
            pool(str): sum/mean/max
        """
        super(Pool, self).__init__()
        self.pool = pool

    def forward(self, input_tensor: torch.Tensor, padding_mask: torch.Tensor):
        """
        Args:
            input_tensor(torch.Tensor): shape (batch_size, sequence_length, embed_dim)
            padding_mask(torch.Tensor): shape (batch_size, sequence_length)

        Returns:
            output(torch.Tensor): shape (batch_size, embed_dim)
        """
        if self.pool == 'sum':
            output = input_tensor.sum(dim=-2)
        elif self.pool == 'mean':
            _, _, embed_dim = input_tensor.shape
            batch_size, sequence_length = padding_mask.shape
            lengths = -padding_mask.sum(-1) + sequence_length
            lengths = lengths.unsqueeze(-1).expand(batch_size, embed_dim)
            assert lengths.shape == torch.Size([batch_size, embed_dim])
            output = input_tensor.sum(dim=-2)
            output = output / lengths
        elif self.pool == 'max':
            output, _ = input_tensor.max(dim=-2)
        else: raise NotImplementedError()
        return output