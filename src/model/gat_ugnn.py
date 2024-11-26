import torch
from torch import nn
from torch_geometric import nn as geo_nn
import numpy as np


class GatGegn(nn.Module):
    def __init__(self,
                 grid_dim,
                 edge_dim,
                 external_dim,
                 config):
        super(GatGegn, self).__init__()

        self.config = config

        self.edge_index_addition = self.get_edge_index_addition(self.config.train_batch_size)

        self.config.grid_layer_channels.insert(0, grid_dim)
        grid_layers = []
        for i in range(len(self.config.grid_layer_channels) - 1):
            hidden_i = (geo_nn.GATConv(self.config.grid_layer_channels[i], self.config.grid_layer_channels[i + 1]), "x, edge_index -> x")
            bn_i = nn.BatchNorm1d(self.config.grid_layer_channels[i + 1])
            relu_i = nn.PReLU()
            grid_layers.extend([hidden_i, bn_i, relu_i])
        self.grid_layer = geo_nn.Sequential("x, edge_index", grid_layers)

        self.config.edge_layer_channels.insert(0, edge_dim)
        edge_layers = []
        for i in range(len(self.config.edge_layer_channels) - 1):
            hidden_i = nn.Linear(self.config.edge_layer_channels[i], self.config.edge_layer_channels[i + 1])
            bn_i = nn.BatchNorm1d(self.config.edge_layer_channels[i + 1])
            relu_i = nn.PReLU()
            edge_layers.extend([hidden_i, bn_i, relu_i])
        self.edge_layer = nn.Sequential(*edge_layers)

        self.config.external_layer_channels.insert(0, external_dim)
        external_layers = []
        for i in range(len(self.config .external_layer_channels) - 1):
            hidden_i = nn.Linear(self.config.external_layer_channels[i], self.config.external_layer_channels[i + 1])
            bn_i = nn.BatchNorm1d(self.config.external_layer_channels[i + 1])
            relu_i = nn.PReLU()
            external_layers.extend([hidden_i, bn_i, relu_i])
        self.external_layer = nn.Sequential(*external_layers)

        self.config.out_layer_channels.insert(0, self.config.grid_layer_channels[-1] * 2 + self.config.edge_layer_channels[-1] + self.config.external_layer_channels[-1])
        total_layers = []
        for i in range(len(self.config.out_layer_channels) - 2):
            hidden_i = nn.Linear(self.config.out_layer_channels[i], self.config.out_layer_channels[i + 1])
            bn_i = nn.BatchNorm1d(self.config.out_layer_channels[i + 1])
            relu_i = nn.PReLU()
            total_layers.extend([hidden_i, bn_i, relu_i])
        total_out_layer = nn.Linear(self.config.out_layer_channels[-2], self.config.out_layer_channels[-1])
        total_layers.append(total_out_layer)
        self.total_layer = nn.Sequential(*total_layers)

    def get_edge_index_addition(self, batch_size):
        edge_index_addition = np.arange(0, batch_size)
        edge_index_addition = np.repeat(edge_index_addition, (self.config.grid_num - 1) * 2)[np.newaxis, :]
        edge_index_addition = np.repeat(edge_index_addition, 2, axis=0)
        edge_index_addition = torch.tensor(edge_index_addition, dtype=torch.long)
        edge_index_addition = edge_index_addition.to(self.config.device)
        return edge_index_addition

    def forward(self, o_grid_x, d_grid_x, edge_index, edge_x, external_x):
        batch_size = o_grid_x.size(0)

        o_grid_x = o_grid_x.reshape(o_grid_x.size(0) * o_grid_x.size(1), o_grid_x.size(2))
        d_grid_x = d_grid_x.reshape(d_grid_x.size(0) * d_grid_x.size(1), d_grid_x.size(2))
        edge_index = edge_index.permute(1, 0, 2)
        edge_index = edge_index.reshape(edge_index.size(0), edge_index.size(1) * edge_index.size(2))

        if batch_size == self.config.train_batch_size:
            edge_index_addition = self.edge_index_addition
        else:
            edge_index_addition = self.get_edge_index_addition(batch_size)
        edge_index = edge_index + edge_index_addition

        o_grid_out = self.grid_layer(o_grid_x, edge_index)
        d_grid_out = self.grid_layer(d_grid_x, edge_index)

        o_grid_out = o_grid_out.reshape(batch_size, self.config.grid_num, o_grid_out.size(1))
        d_grid_out = d_grid_out.reshape(batch_size, self.config.grid_num, d_grid_out.size(1))

        o_grid_out = o_grid_out[:, self.config.grid_num // 2]
        d_grid_out = d_grid_out[:, self.config.grid_num // 2]

        edge_out = self.edge_layer(edge_x)
        external_out = self.external_layer(external_x)

        total = torch.cat([o_grid_out, d_grid_out, edge_out, external_out], 1)
        total_out = self.total_layer(total)
        total_out = total_out.squeeze(-1)
        return total_out