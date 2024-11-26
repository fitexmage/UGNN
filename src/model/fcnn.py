import torch
from torch import nn

from src.model.base_ugnn import BaseGegn, LinearBlock


class Fcnn(BaseGegn):
    def __init__(self,
                 grid_basic_dim,
                 grid_extra_dim,
                 edge_dim,
                 external_dim,
                 config,
                 grid_num=9):
        super(Fcnn, self).__init__()
        self.layer = self._make_layer(LinearBlock,
                                      config.layer_channels,
                                      grid_basic_dim*2*grid_num + grid_extra_dim*2*grid_num + edge_dim + external_dim,
                                      True)

    def forward(self, o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x):
        # print("o_grid_basic_x:", len(o_grid_basic_x[0]))
        # print("d_grid_basic_x:", len(d_grid_basic_x[0]))
        # print("o_grid_extra_x:", len(o_grid_extra_x[0]))
        # print("d_grid_extra_x:", len(d_grid_extra_x[0]))
        # print("edge_x:", len(edge_x[0]))
        # print("external_x:", len(external_x[0]))

        concat_x = torch.cat([o_grid_basic_x, o_grid_extra_x, d_grid_basic_x, d_grid_extra_x, edge_x, external_x], 1)
        # print(concat_x.shape)
        out = self.layer(concat_x)
        out = out.squeeze(-1)
        return out
