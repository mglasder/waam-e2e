import torch
from torch import nn
from torch.nn import functional as F

# import pytest


class SmoothingLayer(nn.Module):
    def __init__(self, max_window_size):
        super(SmoothingLayer, self).__init__()
        self.max_window_size = max_window_size
        self.window_size = nn.Parameter(torch.tensor(2.0), requires_grad=True)

    def forward(self, x):
        # Ensure window_size is within valid range
        window_size = torch.clamp(self.window_size, 1, self.max_window_size)
        window_size = round(window_size.item())
        window_size = window_size if window_size % 2 == 1 else window_size - 1

        padding = (window_size - 1) // 2
        smoothed = F.avg_pool1d(x, window_size, stride=1, padding=padding)

        return smoothed


class GaussianSmoothing(nn.Module):
    def __init__(self, max_window_size):
        super(GaussianSmoothing, self).__init__()
        self.max_window_size = max_window_size
        self.window_size = nn.Parameter(torch.tensor(2.0), requires_grad=True)

    @staticmethod
    def gaussian_weights(window_size):
        sigma = window_size / 6  # A heuristic to relate window size to sigma
        lin_space = torch.linspace(-(window_size // 2), window_size // 2, steps=window_size)
        weights = torch.exp(-(lin_space**2) / (2 * sigma**2))
        weights /= weights.sum()  # Normalize
        return weights

    def forward(self, x):
        window_size = torch.clamp(self.window_size, 1, self.max_window_size)
        window_size = int(window_size.item())
        window_size = window_size if window_size % 2 == 1 else window_size - 1

        weights = self.gaussian_weights(window_size).to(x.device).view(1, 1, -1)
        padding = (window_size - 1) // 2
        smoothed = F.conv1d(x, weights, padding=padding)

        return smoothed


def test_smoothing_layer():
    x = torch.randn(3, 1, 90)
    print(x.shape)
    smooth = GaussianSmoothing(max_window_size=30)
    print(smooth(x).shape)
    assert smooth(x).shape == x.shape


if __name__ == "__main__":
    test_smoothing_layer()
