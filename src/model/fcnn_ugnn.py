import torch
from torch import nn

from src.model.base_ugnn import BaseGegn, LinearBlock


class FcnnGegn(BaseGegn):
    def __init__(self,
                 grid_dim,
                 edge_dim,
                 external_dim,
                 config):
        super(FcnnGegn, self).__init__()

        self.grid_layer = self._make_layer(LinearBlock,
                                           config.grid_layer_channels,
                                           grid_dim * config.grid_num,
                                           False)

        self.edge_layer = self._make_layer(LinearBlock,
                                           config.edge_layer_channels,
                                           edge_dim,
                                           False)

        self.external_layer = self._make_layer(LinearBlock,
                                               config.external_layer_channels,
                                               external_dim,
                                               False)

        self.out_layer = self._make_layer(LinearBlock,
                                          config.out_layer_channels,
                                          config.grid_layer_channels[-1] * 2 + config.edge_layer_channels[-1] + config.external_layer_channels[-1],
                                          True)

    def forward(self, o_grid_x, d_grid_x, edge_x, external_x):
        o_grid_out = self.grid_layer(o_grid_x)
        d_grid_out = self.grid_layer(d_grid_x)
        edge_out = self.edge_layer(edge_x)
        external_out = self.external_layer(external_x)
        out = torch.cat([o_grid_out, d_grid_out, edge_out, external_out], 1)
        out = self.out_layer(out)
        out = out.squeeze(-1)
        return out