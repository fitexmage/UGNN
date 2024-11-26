import torch
from torch import nn

from src.model.base_ugnn import BaseGegn, DeepGravityBlock


class DeepGravity(BaseGegn):
    def __init__(self,
                 grid_dim,
                 edge_dim,
                 config):
        super(DeepGravity, self).__init__()

        self.layer = self._make_layer(DeepGravityBlock,
                                      config.layer_channels,
                                      grid_dim * config.grid_num * 2 + edge_dim,
                                      True)

    def forward(self, o_grid_x, d_grid_x, edge_x):
        o_grid_x = o_grid_x.squeeze(0)
        d_grid_x = d_grid_x.squeeze(0)
        edge_x = edge_x.squeeze(0)
        concat_x = torch.cat([o_grid_x, d_grid_x, edge_x], 1)
        out = self.layer(concat_x)
        out = out.squeeze(-1)
        return out