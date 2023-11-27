import torch

from e2e.bspline.BSplineLayer import BSplineLayer

if __name__ == "__main__":
    m = BSplineLayer(4, 4, n_bases=6, shared_weights=True, bias=False, weighted_sum=False)
    # (batch size, length, channel, basis functions)
    input = torch.randn(100, 162, 4)
    output = m(input)
    print(output.size())
