from abc import abstractmethod
from typing import Optional, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset

from e2e.datamodules.sample import CrossSectionSample

IDs = TypeVar("IDs", bound=list[str])
Samples = TypeVar("Samples", bound=list[CrossSectionSample])
LineSegmentZ = TypeVar("LineSegmentZ", bound=torch.Tensor)


class WaamDataset(Dataset):
    @abstractmethod
    def create(self, samples: Samples):
        pass


# TODO: own module
class ShapeDataset(WaamDataset):
    def __init__(self, segment_length=224):
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

        return inpt, target, sample_id

    def create(self, samples: Samples) -> WaamDataset:
        self.inputs = self._extract_inputs(samples)
        self.targets = self._extract_targets(samples)
        self.ids = self._get_ids(samples)
        # TODO: mirror

        return self

    def _extract_inputs(self, samples: Samples) -> list[LineSegmentZ]:
        inputs = []
        for s in samples:
            torch_idx = s.torchposition.global_y_idx
            zs = self._get_segment_heights(s.slice_based_before.points, torch_idx)
            inputs.append(zs)

        return inputs

    def _extract_targets(self, samples: Samples) -> list[LineSegmentZ]:
        targets = []
        for s in samples:
            torch_idx = s.torchposition.global_y_idx
            zs = self._get_segment_heights(s.slice_based_after.points, torch_idx)
            targets.append(zs)

        return targets

    @staticmethod
    def _get_ids(samples: Samples) -> list[str]:
        ids = []
        for s in samples:
            id_ = s.experiment + "_" + str(s.bead_id)
            ids.append(id_)
        return ids

    def _get_segment_heights(self, points, torch_idx) -> torch.Tensor:
        """ys corresponds to global z-coordinates."""
        left = torch_idx - self._seg_len // 2
        right = torch_idx + self._seg_len // 2
        ys = np.array([p.y[0] for p in points])[left:right]
        return torch.tensor(ys, dtype=torch.float32)
