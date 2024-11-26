import torch
from torch import nn

from src.model.base_ugnn import BaseGegn, LinearBlock, ConvBlock


class CnnGegn(BaseGegn):
    def __init__(self,
                 grid_basic_dim,
                 grid_extra_dim,
                 edge_dim,
                 external_dim,
                 config):
        super(CnnGegn, self).__init__()

        print("grid basic dim:", grid_basic_dim)
        self.o_grid_basic_cnn_layer = self._make_layer(ConvBlock,
                                               config.o_grid_cnn_layer_channels,
                                               grid_basic_dim,
                                               False)

        self.d_grid_basic_cnn_layer = self._make_layer(ConvBlock,
                                               config.d_grid_cnn_layer_channels,
                                               grid_basic_dim,
                                               False)

        self.o_grid_basic_fcnn_layer = self._make_layer(LinearBlock,
                                                config.o_grid_fcnn_layer_channels,
                                                config.o_grid_cnn_layer_channels[-1],
                                                False)

        self.d_grid_basic_fcnn_layer = self._make_layer(LinearBlock,
                                                config.d_grid_fcnn_layer_channels,
                                                config.d_grid_cnn_layer_channels[-1],
                                                False)

        self.o_grid_extra_cnn_layer = self._make_layer(ConvBlock,
                                               config.o_grid_extra_cnn_layer_channels,
                                               grid_extra_dim,
                                               False)

        self.d_grid_extra_cnn_layer = self._make_layer(ConvBlock,
                                               config.d_grid_extra_cnn_layer_channels,
                                               grid_extra_dim,
                                               False)

        self.o_grid_extra_fcnn_layer = self._make_layer(LinearBlock,
                                                config.o_grid_extra_fcnn_layer_channels,
                                                config.o_grid_extra_cnn_layer_channels[-1],
                                                False)

        self.d_grid_extra_fcnn_layer = self._make_layer(LinearBlock,
                                                config.d_grid_extra_fcnn_layer_channels,
                                                config.d_grid_extra_cnn_layer_channels[-1],
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
                                          config.o_grid_fcnn_layer_channels[-1]+config.d_grid_fcnn_layer_channels[-1]+config.o_grid_extra_fcnn_layer_channels[-1]+
                                          config.d_grid_extra_fcnn_layer_channels[-1]+config.edge_layer_channels[-1]+config.external_layer_channels[-1],
                                          True)


    def get_grid_out(self, cnn_layer, fcnn_layer, grid_x):
        # grid_out = self.grid_extra_cnn_layer(grid_x)
        grid_out = cnn_layer(grid_x)
        grid_out = grid_out.view(grid_out.size(0), -1)
        grid_out = fcnn_layer(grid_out)
        return grid_out


    def forward(self, o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x):
        o_grid_basic_out = self.get_grid_out(self.o_grid_basic_cnn_layer, self.o_grid_basic_fcnn_layer, o_grid_basic_x)
        d_grid_basic_out = self.get_grid_out(self.d_grid_basic_cnn_layer, self.d_grid_basic_fcnn_layer, d_grid_basic_x)
        o_grid_extra_out = self.get_grid_out(self.o_grid_extra_cnn_layer, self.o_grid_extra_fcnn_layer, o_grid_extra_x)
        d_grid_extra_out = self.get_grid_out(self.d_grid_extra_cnn_layer, self.d_grid_extra_fcnn_layer, d_grid_extra_x)

        # print("edge_x:\n", edge_x)

        edge_out = self.edge_layer(edge_x)
        # print("edge_out:")
        # print(edge_out)
        external_out = self.external_layer(external_x)
        # print("external_out:\n", external_out)

        out = torch.cat([o_grid_basic_out, d_grid_basic_out, o_grid_extra_out, d_grid_extra_out, edge_out, external_out], 1)
        # print("concat out:\n", out)
        out = self.out_layer(out)
        # out = out.squeeze(-1)
        return out