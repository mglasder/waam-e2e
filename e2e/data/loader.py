import glob
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from tqdm import tqdm

from e2e.data.sample import CrossSectionSample


class EXPERIMENT(Enum):
    CONSTANT_EX3 = "0.2or_Ex_3_0302022"
    CONSTANT_EX4 = "0.4or_Ex_4_08082022"
    RANDOM_EX3 = "Random_Ex3_02032022"
    RANDOM_EX5 = "Random_EX5_08032022"
    RANDOM_EX6 = "Random_Ex6_09032022"


class SampleLoader:
    def __init__(self, sample_dir: Union[str, Path]):
        self._sample_dir = Path(sample_dir)

    def load(self, which: Union[str, list[EXPERIMENT]] = "all") -> list[CrossSectionSample]:
        if which == "all":

            def condition(f: str) -> bool:
                return True

            filepaths = self._get_filepaths()
            return self._read_files(filepaths, condition)

        elif isinstance(which, List):

            def condition(f: str) -> bool:
                return any([e.value in f for e in which])

            filepaths = self._get_filepaths()
            return self._read_files(filepaths, condition)

    def _get_filepaths(self) -> list[str]:
        # assumes all data is in the provided directory without nested directories
        results = glob.glob(str(self._sample_dir / "*.pickle"))
        return results

    @staticmethod
    def _read_files(filepaths: list[str], condition: Callable[[str], bool]) -> list[CrossSectionSample]:
        return [CrossSectionSample.read_file(f) for f in tqdm(filepaths) if condition(f)]
