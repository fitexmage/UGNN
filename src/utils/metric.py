import torch
from torch import nn


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_loss = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.basic_loss(input, target))


class SSI(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.maximum(torch.tensor(0), input)
        return torch.sum(2 * torch.minimum(input, target) / (input + target)) / len(target)


class CPC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return (2 * torch.sum(torch.minimum(input, target))) / torch.sum(input + target)


class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.sum(torch.abs((target - input) / target)) / len(target)


class Pearson(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        concat = torch.concat([input.unsqueeze(0), target.unsqueeze(0)])
        return torch.corrcoef(concat)[0, 1]


class DeepGravityCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return -torch.sum(target / torch.sum(target) * nn.LogSoftmax(dim=0)(input))