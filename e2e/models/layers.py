import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, window_size, sigma_init=1.0):
        super(GaussianSmoothing, self).__init__()
        self.window_size = window_size
        self.padding = (window_size - 1) // 2
        self.sigma = nn.Parameter(torch.tensor(sigma_init), requires_grad=True)
        # wrap in nn.Parameter to move ot to same device as self.sigma,
        # but requires_grad=False so that it is not optimized
        self.positions = nn.Parameter(
            torch.linspace(-(window_size // 2), window_size // 2, steps=window_size), requires_grad=False
        )

    def gaussian_weights(self):
        weights = torch.exp(-self.positions**2 / (2 * self.sigma**2))
        # inplace operations cause problems in the backward pass -> weights = weights / weights.sum()
        weights = weights / weights.sum()  # Normalize
        return weights

    def forward(self, x):
        weights = self.gaussian_weights().view(1, 1, -1)
        smoothed = F.conv1d(x, weights, padding=self.padding)
        return smoothed


class MultiGaussianSmoothing(nn.Module):
    def __init__(self, output_length, window_size, sigma_inits=[1.0, 2.0, 3.0]):
        super(MultiGaussianSmoothing, self).__init__()

        self.smoothing_layers = nn.ModuleList(
            [GaussianSmoothing(window_size, sigma_init=sigma) for sigma in sigma_inits]
        )

        # Learnable mask for piecewise combination
        self.mask = nn.Parameter(torch.randn(3, output_length), requires_grad=True)

    def forward(self, x):
        smoothed_outputs = [layer(x) for layer in self.smoothing_layers]
        stacked_outputs = torch.stack(smoothed_outputs, dim=0)

        # Compute the softmax over the mask to get the piecewise combination weights
        weights = F.softmax(self.mask, dim=0).unsqueeze(0).unsqueeze(2)

        weighted_outputs = weights * stacked_outputs
        combined = torch.sum(weighted_outputs, dim=0)

        return combined


def test_smoothing_layer():
    x = torch.randn(3, 1, 90)
    smooth = GaussianSmoothing(window_size=15, sigma_init=1.0)
    assert smooth(x).shape == x.shape


def test_multi_smoothing_layer():
    x = torch.randn(3, 1, 90)
    smooth = MultiGaussianSmoothing(output_length=90, window_size=15, sigma_inits=[1.0, 2.0, 3.0])
    assert smooth(x).shape == x.shape


if __name__ == "__main__":
    test_smoothing_layer()
    test_multi_smoothing_layer()
