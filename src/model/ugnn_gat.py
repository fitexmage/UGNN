import torch
from torch import nn

from src.model.base_ugnn import BaseGegn, LinearBlock, ConvBlock

from torch_geometric.nn import SAGEConv

# class SAGEConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super(SAGEConvBlock, self).__init__()
#         self.conv1 = SAGEConv(in_channels, out_channels)
        
#     def forward(self, x, edge_index):
#         # First SAGEConv layer
#         out = self.conv1(x, edge_index)
#         out = torch.relu(out)  # Adding a non-linearity after the first conv layer


#         return out

class SAGEConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super(SAGEConvBlock, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # First SAGEConv layer
        out = self.conv1(x, edge_index)
        out = torch.relu(out)  # Adding a non-linearity after the first conv layer

        # Second SAGEConv layer
        out = self.conv2(out, edge_index)
        out = torch.relu(out)  # Adding a non-linearity after the second conv layer

        return out



class SAGEGegn(BaseGegn):
    def __init__(self,
                 grid_basic_dim,
                 grid_extra_dim,
                 edge_dim,
                 external_dim,
                 config):
        super(SAGEGegn, self).__init__()

        print("grid basic dim:", grid_basic_dim)
        
        self.SAGEConv_layer = SAGEConvBlock(49, 128, 64)

        self.edge_layer = self._make_layer(LinearBlock,
                                           config.edge_layer_channels,
                                           edge_dim,
                                           False)

        self.external_layer = self._make_layer(LinearBlock,
                                               config.external_layer_channels,
                                               external_dim,
                                               False)

        self.out_layer = self._make_layer(LinearBlock,[128,64, 1],
                                          128+config.edge_layer_channels[-1]+config.external_layer_channels[-1],
                                          True)


    def get_grid_out(self, cnn_layer, fcnn_layer, grid_x):
        # grid_out = self.grid_extra_cnn_layer(grid_x)
        grid_out = cnn_layer(grid_x)
        grid_out = grid_out.view(grid_out.size(0), -1)
        grid_out = fcnn_layer(grid_out)
        return grid_out


    def forward(self, o_idx, d_idx, edge_x, external_x, x, edge_index, id_dict):

        self.SAGEConv_x = self.SAGEConv_layer(x, edge_index)
        
        o_idx = [id_dict[i.item()] for i in o_idx]
        d_idx = [id_dict[i.item()] for i in d_idx]
        
        o_grid = self.SAGEConv_x[o_idx]
        d_grid = self.SAGEConv_x[d_idx]

        edge_out = self.edge_layer(edge_x)
        # print("edge_out:")
        # print(edge_out)
        external_out = self.external_layer(external_x)
        # print("external_out:\n", external_out)

        out = torch.cat([o_grid, d_grid, edge_out, external_out], 1)
        # print("concat out:\n", out)
        out = self.out_layer(out)
        # out = out.squeeze(-1)
        return out