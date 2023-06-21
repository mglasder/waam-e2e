from math import floor

from nest.engineering.utils.pydantic import BaseSettings
from torch import nn


def avgpool1d_output_size(l_in, padding, kernel_size, stride):
    return floor((l_in + 2 * padding - kernel_size) / stride + 1)


class ResNetConfig(BaseSettings):
    in_c: list[int] = [1, 8]
    out_c: list[int] = [8, 8]
    kernels: list[int] = [55, 3]
    strides: list[int] = [1, 1]
    paddings: list[int] = [27, 1]


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            act,
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )

        self.act = act
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = x + residual
        out = self.act(out)
        return out


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            act(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResNet1D(nn.Module):
    def __init__(self, conf: ResNetConfig, inout_length=224):
        super().__init__()

        self.conv1 = ConvBlock1D(1, conf.in_c[0], kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResidualBlock1D(conf.in_c[0], conf.out_c[0], conf.kernels[0], conf.strides[0], conf.paddings[0])
        self.layer2 = ResidualBlock1D(conf.in_c[1], conf.out_c[1], conf.kernels[1], conf.strides[1], conf.paddings[1])
        self.avgpool = nn.AvgPool1d(7, 1)
        self.fc = nn.Linear(400, inout_length)

    def to(self, device):
        # TODO: implement
        pass

    def forward(self, x):
        # passthrough of x
        # residual = x
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        # x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, -1)
        # x += residual
        return x
