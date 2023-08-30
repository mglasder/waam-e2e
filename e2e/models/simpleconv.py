import torch.nn.functional as F
from torch import nn


class SimpleConv(nn.Module):
    def __init__(
        self,
        in_channels=1,
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
        # self.dropout = nn.Dropout(p=p)
        # self.dropout1d = nn.Dropout1d(p=p)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.bn3 = nn.BatchNorm1d(in_channels)
        # self.bn4 = nn.BatchNorm1d(in_channels)

        self.conv1 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)
        # self.conv3 = nn.Conv1d(in_channels, kernel_size, kernel_size, stride, padding)

        # self.smooth1 = SmoothingLayer(max_window_size=30)

    # def to(self, device):
    #     # TODO: implement
    #     pass

    def forward(self, x):
        r = x

        x = self.bn1(x)
        x = self.fc(x)
        x = self.act(x)
        x = F.dropout(x, p=self.p, training=self.training) + r

        x = self.bn2(x)
        x = self.conv1(x)
        x = F.dropout1d(x, p=self.p, training=self.training) + r.view(-1, self.k, 1)

        x = self.bn3(x.view(-1, 1, self.k))
        x = self.conv2(x) + r.view(-1, self.k, 1)

        # x = self.bn4(x.view(-1, 1, self.k))
        # x = self.conv3(x) + r.view(-1, self.k, 1)

        # x = x.view(-1, 1, self.k)
        # x = self.smooth1(x)

        return x
