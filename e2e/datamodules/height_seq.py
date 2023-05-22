from abc import abstractmethod
from pathlib import Path
from typing import Optional, TypeVar

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from e2e.datamodules.loader import EXPERIMENT, SampleLoader
from e2e.datamodules.sample import CrossSectionSample

Inputs = TypeVar("Inputs", bound=list)
Targets = TypeVar("Targets", bound=list)
IDs = TypeVar("IDs", bound=list[str])
Samples = TypeVar("Samples", bound=list[CrossSectionSample])


class WaamDataset(Dataset):
    @abstractmethod
    def create(self, samples: Samples):
        pass


class ShapeDataset(WaamDataset):
    def __init__(self):
        self.inputs: Optional[Inputs] = None
        self.targets: Optional[Targets] = None
        self.ids: Optional[IDs] = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        target = self.targets[idx]
        sample_id = self.ids[idx]

        return inpt, target, sample_id

    def create(self, samples: Samples):
        # TODO: extract input, target, crop, mirror etc.
        pass

    def _extract_input(self, sample: Samples):
        pass

    def _extract_target(self):
        pass

    def _get_ids(self):
        pass


class ShapePredictionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: WaamDataset,
        split=[0.7, 0.3, 0],
        separate_test_set: Optional[EXPERIMENT] = None,
        seed=42,
    ):
        super().__init__()

        self.dataset = dataset

        # TODO: configure batch_size etc.
        self._num_workers = 8
        self._batch_size = 128
        self._sample_dir = Path("/.")
        self._split = split

        self._sep_ts_set = separate_test_set
        self._dataset_tr, self._dataset_vl, self._dataset_ts = None, None, None

        self._gen = torch.Generator().manual_seed(seed)

    def setup(self, stage: str) -> None:
        loader = SampleLoader(self._sample_dir)
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
        return DataLoader(self._dataset_tr, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_vl, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset_ts, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

    def predict_dataloader(self):
        return [
            DataLoader(self._dataset_tr, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._dataset_vl, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._dataset_ts, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
        ]
