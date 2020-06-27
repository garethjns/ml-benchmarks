import numpy as np
from sklearn.datasets import make_classification

from ml_benchmarks.data.base_data import BaseData


class Synthetic(BaseData):
    def __init__(self, *args, samples: int = 10000, n_features: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = samples
        self.n_features = n_features

    def build(self) -> np.ndarray:
        return make_classification(n_samples=self.samples, n_features=self.n_features, random_state=self.seed)
