import pytest
import torch

from e2e.data.resampled import ResampledShapePointsDataset


@pytest.fixture
def dataset_cls():
    return ResampledShapePointsDataset(mirror=True, segment_length=10)


@pytest.mark.parametrize(
    "input_tensor, expected_output",
    [
        (
            torch.stack([torch.arange(1, 11), torch.arange(2, 12)]),
            torch.stack([torch.arange(10, 0, -1), torch.arange(11, 1, -1)]),
        )
    ],
)
def test_flip_points_tensor(dataset_cls, input_tensor, expected_output):
    flip = dataset_cls._flip_points_tensor
    result = flip(input_tensor)
    assert torch.allclose(result, expected_output)
