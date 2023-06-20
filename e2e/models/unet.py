import torch
from nest.engineering.utils.pydantic import BaseSettings
from torch import nn


class EncoderConfig(BaseSettings):
    in_cs: list[int] = [1, 4, 8, 16]
    out_cs: list[int] = [4, 8, 16, 32]
    k_sz: list[int] = [3, 3, 3, 3]


class DecoderConfig(BaseSettings):
    in_cs: list[int] = [64, 32, 16, 8]
    out_cs: list[int] = [32, 16, 8, 4]
    k_sz: list[int] = [3, 3, 3, 3]


class UNet1D(nn.Module):
    def __init__(self, enconf: EncoderConfig, deconf: DecoderConfig, act=nn.ReLU()):
        super().__init__()

        self.enc = self._make_encoder_blocks(enconf, act=act)
        self.b = ConvBlock(
            in_c=enconf.out_cs[-1],
            out_c=deconf.in_cs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            act=nn.ReLU(),
        )
        self.dec = self._make_decoder_blocks(deconf, act=act)

        self.out = nn.Conv1d(deconf.out_cs[-1], 1, kernel_size=1, stride=1, padding=0)

    def to(self, device):
        for m in self.enc:
            m.to(device)
        self.b.block.to(device)
        for m in self.dec:
            m.to(device)
        self.out.to(device)

    @staticmethod
    def _make_encoder_blocks(enconf: EncoderConfig, act=nn.ReLU()):
        blocks = []
        for in_c, out_c, k_sz in zip(enconf.in_cs, enconf.out_cs, enconf.k_sz):
            blocks.append(EncoderBlock(in_c, out_c, k_sz, stride=1, padding=1, act=act))
        return blocks

    @staticmethod
    def _make_decoder_blocks(deconf: DecoderConfig, act=nn.ReLU()):
        blocks = []
        for in_c, out_c, k_sz in zip(deconf.in_cs, deconf.out_cs, deconf.k_sz):
            blocks.append(DecoderBlock(in_c, out_c, k_sz, stride=2, padding=0, act=act))
        return blocks

    def forward(self, x):
        s1, p1 = self.enc[0](x)
        s2, p2 = self.enc[1](p1)
        s3, p3 = self.enc[2](p2)
        s4, p4 = self.enc[3](p3)

        b = self.b(p4)

        d1 = self.dec[0](b, s4)
        d2 = self.dec[1](d1, s3)
        d3 = self.dec[2](d2, s2)
        d4 = self.dec[3](d3, s1)

        return self.out(d4)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, act=nn.ReLU()):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, stride, padding),
            nn.BatchNorm1d(out_c),
            act,
            nn.Conv1d(out_c, out_c, kernel_size, stride, padding),
            nn.BatchNorm1d(out_c),
            act,
        )

    def forward(self, x):
        x = self.block(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, act=nn.ReLU()):
        super().__init__()

        self.conv = ConvBlock(in_c, out_c, kernel_size, stride, padding, act=act)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2, padding=0, act=nn.ReLU()):
        super().__init__()

        self.upconv = nn.ConvTranspose1d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv = ConvBlock(out_c + out_c, out_c, act=act)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
