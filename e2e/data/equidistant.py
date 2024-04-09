from typing import Optional

import torch

from e2e.data.dataset import WaamDataset, LineSegmentZ, IDs, Samples
from e2e.e2esim import interp_and_torch_stack
from e2e.helpers.resample import interp_equidistant


class ShapePointsEquidistant(WaamDataset):
    def __init__(self, segment_length=50):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.ids: Optional[IDs] = None

        self._seg_len = segment_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        # TODO: Remove hack: consumers of dataset expect fp data in fourth position -> empty tensor
        # inpt = torch.ones_like(inpt)
        return inpt, target, sample_id, torch.tensor([])

    def create(self, samples: Samples) -> WaamDataset:
        inputs = self._extract_inputs(samples)
        targets = self._extract_targets(samples)
        ids = self._get_ids(samples)
        self.inputs, self.targets, self.ids = inputs, targets, ids
        return self

    def _extract_inputs(self, samples: Samples) -> list[LineSegmentZ]:
        inputs = []
        for s in samples:
            xs = torch.tensor([p.x[0] for p in s.slice_aligned_before.points], dtype=torch.float32)
            ys = torch.tensor([p.y[0] for p in s.slice_aligned_before.points], dtype=torch.float32)

            left_fp_idx = s.footprint.left_idx
            right_fp_idx = s.footprint.right_idx

            torch_idx = s.torchposition.global_y_idx
            x_shift = xs[torch_idx].clone()
            y_shift = ys[torch_idx].clone()

            x_coords = xs[left_fp_idx:right_fp_idx].clone()
            y_coords = ys[left_fp_idx:right_fp_idx].clone()

            x_coords -= x_shift
            y_coords -= y_shift

            shape = torch.stack(
                (
                    x_coords,
                    y_coords,
                ),
            )
            shape_re = interp_and_torch_stack(interp_equidistant, shape, self._seg_len)

            self._verify_data(shape_re)

            inputs.append(shape_re)

        return inputs

    def _verify_data(self, shape: torch.Tensor):
        assert shape.shape == torch.Size([2, self._seg_len])
        assert shape.min() >= -10.0, f"min is {shape.min()}"
        assert shape.max() <= 10.0, f"max is {shape.max()}"
        # print(f"min is: {shape.min()}")
        # print(f"max is: {shape.max()}")
        assert shape.isnan().any() == False
        assert shape.isinf().any() == False
        assert shape.dtype == torch.float32

    def _extract_targets(self, samples: Samples) -> list[LineSegmentZ]:
        targets = []
        for s in samples:
            xs_before = torch.tensor([p.x[0] for p in s.slice_aligned_before.points], dtype=torch.float32)
            ys_before = torch.tensor([p.y[0] for p in s.slice_aligned_before.points], dtype=torch.float32)
            torch_idx = s.torchposition.global_y_idx

            x_shift = xs_before[torch_idx].clone()
            y_shift = ys_before[torch_idx].clone()

            xs = torch.tensor([p.x[0] for p in s.slice_aligned_after.points], dtype=torch.float32)
            ys = torch.tensor([p.y[0] for p in s.slice_aligned_after.points], dtype=torch.float32)

            if "Sim" in s.experiment:
                left_fp_idx = s.footprint_after.left_idx
                right_fp_idx = s.footprint_after.right_idx
            else:
                left_fp_idx = s.footprint.left_idx
                right_fp_idx = s.footprint.right_idx

            x_coords = xs[left_fp_idx:right_fp_idx].clone()
            y_coords = ys[left_fp_idx:right_fp_idx].clone()

            x_coords -= x_shift
            y_coords -= y_shift

            shape = torch.stack(
                (
                    x_coords,
                    y_coords,
                ),
            )
            shape_re = interp_and_torch_stack(interp_equidistant, shape, self._seg_len)
            targets.append(shape_re)

        return targets

    @staticmethod
    def _get_ids(samples: Samples) -> list[str]:
        ids = []
        for s in samples:
            id_ = s.experiment + "_" + str(s.bead_id)
            ids.append(id_)
        return ids
