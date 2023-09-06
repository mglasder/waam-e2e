from pathlib import Path
from typing import Optional, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from e2e.data.dataset import WaamDataset
from e2e.data.loader import EXPERIMENT, SampleLoader


class ShapePredictionDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset: WaamDataset,
        batch_size=16,
        workers=1,
        split=[0.5, 0.5, 0],
        train_val_sets: Union[str, list[EXPERIMENT]] = "all",
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
        self._tr_val_sets = train_val_sets
        self._dataset_tr, self._dataset_vl, self._dataset_ts = None, None, None

        self._gen = torch.Generator().manual_seed(seed)

    def setup(self, stage: str) -> None:
        loader = SampleLoader(self._data_dir)
        cross_section_samples = loader.load(which=self._tr_val_sets)

        print(len(cross_section_samples))

        first_in_row_samples = [s for s in cross_section_samples if s.welding_params["weld_bead_nr"] == 1]
        for _ in range(2):
            cross_section_samples.extend(first_in_row_samples)

        print(len(cross_section_samples))

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
