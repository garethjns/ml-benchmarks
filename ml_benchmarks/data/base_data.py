import abc
from typing import Tuple

import numpy as np


class BaseData(abc.ABC):
    def __init__(self, seed: int = 123):
        self.seed = seed

    @abc.abstractmethod
    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
