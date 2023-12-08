import pytest
import torch

from e2e.models.modelV2 import ModelPoints


class MockModelPoints(ModelPoints):
    def __init__(self, length_out: int) -> None:
        self.length_out = length_out


@pytest.fixture(params=[2])
def setup_class(request):
    obj = MockModelPoints(length_out=request.param)
    return obj


@pytest.mark.parametrize(
    "predictions, targets, expected_result",
    [
        # (torch.zeros((2, 2, 2)), torch.ones((2, 2, 2)), 1.414213 * 2),
        # (torch.zeros((2, 2, 4)), torch.ones((2, 2, 4)), 1.414213 * 4),
        (
            torch.tensor([[[0, 1], [0, 1]]]),
            torch.tensor([[[0, 0], [0, 0]]]),
            torch.sqrt(torch.tensor([2])) / 2,
        ),
        (
            torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]]),
            torch.tensor([[[4, 4], [4, 4]], [[4, 4], [4, 4]]]),
            torch.sqrt(torch.tensor([4**2 + 4**2])),
        ),
    ],
)
def test_loss(setup_class, predictions, targets, expected_result):
    obj = setup_class
    result = obj._loss(predictions, targets)

    assert torch.isclose(result, expected_result, atol=1e-5)
