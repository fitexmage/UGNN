import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DeepGravityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepGravityBlock, self).__init__()

        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BaseGegn(nn.Module):
    def __init__(self):
        super(BaseGegn, self).__init__()

    def _make_layer(self, block, layer_channels, in_channels, is_output_layer):
        layer_channels.insert(0, in_channels)
        layers = []
        if is_output_layer:
            for i in range(len(layer_channels) - 2):
                layers.append(block(layer_channels[i], layer_channels[i + 1]))
            layers.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
        else:
            for i in range(len(layer_channels) - 1):
                layers.append(block(layer_channels[i], layer_channels[i + 1]))
        return nn.Sequential(*layers)
