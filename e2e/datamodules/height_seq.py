from abc import abstractmethod
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from e2e.datamodules.loader import EXPERIMENT, SampleLoader
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


# TODO: own module
class ShapePredictionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset: WaamDataset,
        batch_size=16,
        workers=1,
        split=[0.5, 0.5, 0],
        separate_test_set: Optional[EXPERIMENT] = None,
        seed=42,
    ):
        super().__init__()

        self.dataset = dataset

        self._num_workers = workers
        self._batch_sz = batch_size
        self._data_dir = data_dir
        self._split = split

        self._sep_ts_set = separate_test_set
        self._dataset_tr, self._dataset_vl, self._dataset_ts = None, None, None

        self._gen = torch.Generator().manual_seed(seed)

    def setup(self, stage: str) -> None:
        loader = SampleLoader(self._data_dir)
        cross_section_samples = loader.load(which="all")
        dataset = self.dataset.create(cross_section_samples)
        self._dataset_tr, self._dataset_vl, self._dataset_ts = self._random_split(dataset)

        if self._sep_ts_set:
            assert self._split[2] == 0
            self._dataset_ts = loader.load(which=[self._sep_ts_set])

    def _random_split(self, dataset):
        size = len(dataset)
        size_tr = int(size * self._split[0])
        size_vl = int(size * self._split[1])
        size_ts = size - size_tr - size_vl
        return random_split(dataset, [size_tr, size_vl, size_ts], generator=self._gen)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_tr, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_vl, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_ts, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=False)

    def predict_dataloader(self):
        return [
            DataLoader(self._dataset_tr, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._dataset_vl, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._dataset_ts, batch_size=self._batch_sz, num_workers=self._num_workers, shuffle=False),
        ]
