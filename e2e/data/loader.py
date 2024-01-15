import glob
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
from tqdm import tqdm

from e2e.data.sample import CrossSectionSample


class EXPERIMENT(Enum):
    CONSTANT_EX3 = "0.2or_Ex_3_0302022"
    CONSTANT_EX4 = "0.4or_Ex_4_08082022"
    RANDOM_EX3 = "Random_Ex3_02032022"
    RANDOM_EX5 = "Random_EX5_08032022"
    RANDOM_EX6 = "Random_Ex6_09032022"
    SIMULATION_CUBE01 = "Simulation_Cube01"


class SampleLoader:
    def __init__(self, sample_dir: Union[str, Path], seed=42, workers=1):
        self._sample_dir = Path(sample_dir)
        self._generator = np.random.default_rng(seed=seed)
        self._workers = workers

    def load(self, which: Union[str, list[EXPERIMENT]] = "all", subset_size: float = 1) -> list[CrossSectionSample]:
        if which == "all":

            def condition(_: str) -> bool:
                return True

        elif isinstance(which, List):

            def condition(f: str) -> bool:
                return any([e.value in f for e in which])

        else:
            raise NotImplementedError

        filepaths = self._get_filepaths()

        if subset_size < 1:
            # choose random subset of size subset
            k = int(len(filepaths) * subset_size)
            print(f"loading {k} out of {len(filepaths)} samples.")
            filepaths = self._generator.choice(filepaths, k, replace=False)

        if self._workers > 1:
            return self._read_files_multithreading(filepaths, condition)
        else:
            return self._read_files(filepaths, condition)

    def _get_filepaths(self) -> np.ndarray:
        # assumes all data is in the provided directory without nested directories
        results = glob.glob(str(self._sample_dir / "*.pickle"))
        return np.array(results)

    @staticmethod
    def _read_files(filepaths: np.ndarray, condition: Callable[[str], bool]) -> list[CrossSectionSample]:
        return [CrossSectionSample.read_file(f) for f in tqdm(filepaths) if condition(f)]

    def _read_files_multithreading(
        self, filepaths: np.ndarray, condition: Callable[[str], bool]
    ) -> list[CrossSectionSample]:
        filepaths = [f for f in filepaths if condition(f)]

        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            results = tqdm(executor.map(CrossSectionSample.read_file, filepaths), total=len(filepaths))

        return list(results)
