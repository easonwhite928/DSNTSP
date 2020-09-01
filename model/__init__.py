# import torch
# import torch.nn as nn
# from data.temporal_sets_batch import TemporalSetsInput
#
#
# class BaseModel(nn.Module):
#
#     def __init__(self, items_total, embed_dim=64, dropout=0.1):
#         super(BaseModel, self).__init__()
#         self.items_total = items_total
#         self.embed_dim = embed_dim
#         self.item_embed = nn.Embedding(num_embeddings=items_total,
#                                        embedding_dim=embed_dim)
#         # feed user representation(learn from user behaviors) into simple MLP
#         self.output_net = nn.Sequential(
#             nn.Linear(embed_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, items_total)
#         )
#
#     def forward(self, input_batch: TemporalSetsInput):
#         """
#         Args:
#             input_batch (TemporalSetsInput):
#         Returns:
#             output (Tensor): (batch_size, items_total)
#         """
#         items_batch = input_batch.get_items()
#         batch_size = len(items_batch)
#         items_embed_pool = torch.stack([self.item_embed(items).sum(dim=-2) for items in items_batch])
#         assert items_embed_pool.shape == torch.Size([batch_size, self.embed_dim])
#
#         # simple
#         output = self.output_net(items_embed_pool)
#         assert output.shape == torch.Size([batch_size, self.items_total])
#
#         return output
#
#
#
# import torch
# import torch.nn as nn
# from data.temporal_sets_batch import TemporalSetsInput
#
#
# class BaseModelPlus(nn.Module):
#
#     def __init__(self, items_total, embed_dim=64, dropout=0.1):
#         super(BaseModelPlus, self).__init__()
#         self.items_total = items_total
#         self.embed_dim = embed_dim
#         self.item_embed = nn.Embedding(num_embeddings=items_total,
#                                        embedding_dim=embed_dim)
#         self.output_net = nn.Sequential(
#             nn.Linear(embed_dim * 2, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1)
#         )
#
#     def forward(self, input_batch: TemporalSetsInput):
#         """
#         Args:
#             input_batch (TemporalSetsInput):
#         Returns:
#             output (Tensor): (batch_size, items_total)
#         """
#         items_batch = input_batch.get_items()
#         batch_size = len(items_batch)
#         items_embed_pool = torch.stack([self.item_embed(items).sum(dim=-2) for items in items_batch])
#         assert items_embed_pool.shape == torch.Size([batch_size, self.embed_dim])
#
#         embedding_weight = self.item_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
#         assert embedding_weight.shape == torch.Size([batch_size, self.items_total, self.embed_dim])
#         output = items_embed_pool.unsqueeze(1).expand(-1, self.items_total, -1)
#         assert output.shape == torch.Size([batch_size, self.items_total, self.embed_dim])
#
#         output = torch.cat([output, embedding_weight], dim=-1)
#         assert output.shape == torch.Size([batch_size, self.items_total, self.embed_dim * 2])
#
#         torch.cuda.empty_cache()
#
#         output = self.output_net(output)
#         assert output.shape == torch.Size([batch_size, self.items_total, 1])
#
#         output = output.squeeze()
#         assert output.shape == torch.Size([batch_size, self.items_total])
#
#         # # implemented by for-loop
#         # embedding_weight = self.item_embed.weight  # (items_total, embed_dim)
#         # output = []
#         # for item_embed in embedding_weight:
#         #     item_embed = item_embed.unsqueeze(0).expand(batch_size, -1)
#         #     assert item_embed.shape == torch.Size([batch_size, self.embed_dim])
#         #     mlp_input = torch.cat([items_embed_pool, item_embed], dim=-1)
#         #     assert mlp_input.shape == torch.Size([batch_size, self.embed_dim * 2])
#         #     mlp_output = self.output_net(mlp_input)
#         #     assert mlp_output.shape == torch.Size([batch_size, 1])
#         #     output.append(mlp_output)
#         #
#         # output = torch.cat(output, dim=-1)
#         # assert output.shape == torch.Size([batch_size, self.items_total])
#
#         return output