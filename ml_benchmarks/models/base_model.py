import abc
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class BaseModel(abc.ABC):
    def __init__(self, seed: int = 123, n_jobs: int = -1):
        self.seed = seed
        self.n_jobs = n_jobs

    @abc.abstractmethod
    def build(self) -> Union[GridSearchCV, BaseEstimator]:
        pass
