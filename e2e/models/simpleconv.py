import torch
import torch.nn.functional as F
from torch import nn


class SmoothingLayer(nn.Module):
    def __init__(self, max_window_size):
        super(SmoothingLayer, self).__init__()
        self.max_window_size = max_window_size
        self.window_size = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        # Ensure window_size is within valid range
        window_size = torch.clamp(self.window_size, 1, self.max_window_size).int().item()

        # Apply moving average
        smoothed = F.avg_pool1d(x, window_size, stride=1, padding=window_size - 1)
        return smoothed


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=100,
        kernel_size=224,
        p=0.2,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.p = p
        self.k = kernel_size

        self.fc = nn.Linear(in_features=kernel_size, out_features=out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.dropout1d = nn.Dropout1d(p=p)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.bn3 = nn.BatchNorm1d(in_channels)
        # self.bn4 = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv3 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)

        self.smooth1 = SmoothingLayer(max_window_size=30)

    # def to(self, device):
    #     # TODO: implement
    #     pass

    def forward(self, x):
        r = x

        x = self.bn1(x)
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x) + r

        x = self.bn2(x)
        x = self.conv1(x)
        x = self.dropout1d(x) + r.view(-1, self.k, 1)

        x = self.bn3(x.view(-1, 1, self.k))
        x = self.conv2(x) + r.view(-1, self.k, 1)

        # x = self.bn4(x.view(-1, 1, self.k))
        # x = self.conv3(x) + r.view(-1, self.k, 1)

        x = x.view(-1, 1, self.k)
        x = self.smooth1(x)

        return x
