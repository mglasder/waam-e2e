from abc import abstractmethod
from typing import Optional, TypeVar
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from e2e.data.sample import CrossSectionSample, FootprintEdge
from torchvision import transforms as T

IDs = TypeVar("IDs", bound=list[str])
Samples = TypeVar("Samples", bound=list[CrossSectionSample])
LineSegmentZ = TypeVar("LineSegmentZ", bound=torch.Tensor)


class WaamDataset(Dataset):
    @abstractmethod
    def create(self, samples: Samples):
        pass


class ZeroRandomDataset(WaamDataset):
    def __init__(self, segment_length=224, num_samples=32):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.ids: Optional[IDs] = None

        self._seg_len = segment_length
        self._num_samples = num_samples

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return inpt, target, sample_id

    def create(self, samples: Samples) -> WaamDataset:
        self.inputs = [torch.zeros(self._seg_len) for _ in range(self._num_samples)]
        self.targets = [torch.randn(self._seg_len) for _ in range(self._num_samples)]
        self.ids = [f"dummy_{i}" for i in range(self._num_samples)]
        return self


class ShapeDataset(WaamDataset):
    def __init__(self, mirror=False, segment_length=224):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.ids: Optional[IDs] = None
        self.fp_idx: Optional[list[torch.tensor]] = None

        self.mirror = mirror
        self._seg_len = segment_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]
        footprint = self.fp_idx[idx]

        return inpt, target, sample_id, footprint

    def create(self, samples: Samples) -> WaamDataset:
        inputs = self._extract_inputs(samples)
        targets = self._extract_targets(samples)
        ids = self._get_ids(samples)
        fp_idx = self._extract_relative_footprint_idx(samples)

        if self.mirror:
            self.inputs, self.targets, self.ids, self.fp_idx = self._mirror_dataset(inputs, targets, ids, fp_idx)

        else:
            self.inputs, self.targets, self.ids, self.fp_idx = inputs, targets, ids, fp_idx

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

    def _extract_relative_footprint_idx(self, samples: Samples) -> list[torch.tensor]:
        footprint_idx = []
        mid = self._seg_len // 2
        for s in samples:
            footprint = s.footprint_based
            left = mid - int(np.abs(footprint.left_idx - s.torchposition.global_y_idx))
            right = mid + int(np.abs(footprint.right_idx - s.torchposition.global_y_idx))
            # TODO: do this the right way (filter out samples)
            if left < 0:
                left = 0
            if right > self._seg_len:
                right = self._seg_len - 1
            footprint_idx.append(torch.tensor([left, right]))
        return footprint_idx

    def _mirror_dataset(self, inputs, targets, sample_ids, footprint_idx):
        inputs_h = []
        targets_h = []
        sample_ids_h = []
        footprint_idx_h = []

        # reverse a torch tensor

        for inpt, target, id_, fp_idx in zip(inputs, targets, sample_ids, footprint_idx):
            inputs_h.append(inpt.flipud())
            targets_h.append(target.flipud())
            sample_ids_h.append(id_ + "_hflip")
            footprint_idx_h.append(torch.tensor([self._seg_len - fp_idx[1], self._seg_len - fp_idx[0]]))

        inputs_h = np.concatenate((inputs_h, inputs))
        targets_h = np.concatenate((targets_h, targets))
        sample_ids_h = np.concatenate((sample_ids_h, sample_ids))
        footprint_idx_h = np.concatenate((footprint_idx_h, footprint_idx))

        return inputs_h, targets_h, sample_ids_h, footprint_idx_h

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


class ResampledFootprintDataset(WaamDataset):
    def __init__(self, output_length=100):
        self.inputs: Optional[list[LineSegmentZ]] = None
        self.targets: Optional[list[LineSegmentZ]] = None
        self.footprint_idx: Optional[list[tuple[int, int]]] = None
        self.ids: Optional[IDs] = None

        self._out_len = output_length
        self._seg_len = 224

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]
        footprint = self.footprint_idx[idx]

        return inpt, target, sample_id, footprint

    def create(self, samples: Samples) -> WaamDataset:
        self.inputs = self._extract_inputs(samples)
        self.targets = self._extract_targets(samples)
        self.footprint_idx = self._extract_footprint_idx(samples)
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
            zs = self._get_output_segment_heights(s.slice_based_after.points, s.footprint_based)
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

    def _get_output_segment_heights(self, points, footprint: FootprintEdge):
        left = footprint.left_idx
        right = footprint.right_idx

        ys = np.array([p.y[0] for p in points])[left:right]
        xs = np.arange(0, len(ys)) / 10
        f = interp1d(xs, ys, kind="linear")
        xs_new = np.linspace(0, (len(ys) - 1) / 10, self._out_len)
        ys_new = f(xs_new)
        return torch.tensor(ys_new, dtype=torch.float32)

    @staticmethod
    def _extract_footprint_idx(samples: Samples) -> list[torch.tensor]:
        footprint_idx = []
        for s in samples:
            footprint = s.footprint_based
            left = footprint.left_idx
            right = footprint.right_idx
            footprint_idx.append(torch.tensor([left, right]))
        return footprint_idx
