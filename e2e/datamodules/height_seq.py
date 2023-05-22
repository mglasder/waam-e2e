from abc import abstractmethod
from pathlib import Path
from typing import Optional, TypeVar

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from e2e.datamodules.loader import SampleLoader
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
    def __init__(self, dataset: WaamDataset):
        super().__init__()
        self.dataset = dataset

        # TODO: configure batch_size etc.
        self._num_workers = 8
        self._batch_size = 128
        self._sample_dir = Path("/.")

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: str) -> None:
        # TODO: load data, select samples, create dataset, split
        loader = SampleLoader(self._sample_dir)
        cross_section_samples = loader.load(which="all")
        _ = self.dataset.create(cross_section_samples)
        # self_train = ...

    def _random_split(self):
        # TODO
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False)

    def predict_dataloader(self):
        return [
            DataLoader(self._train, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._val, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
            DataLoader(self._test, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=False),
        ]
