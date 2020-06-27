import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from ml_benchmarks.data.base_data import BaseData
from ml_benchmarks.models.base_model import BaseModel


@dataclass
class Benchmark:
    model: BaseModel
    data: BaseData
    seed: int = 123
    n_runs: int = 5
    verbose: int = 1

    def run(self):
        times = []
        for r in range(self.n_runs):
            self.model.seed = self.seed + r
            self.data.seed = self.seed + r

            data: Tuple[np.ndarray, np.ndarray] = self.data.build()
            x, y = data

            t0 = time.time()
            model: GridSearchCV = self.model.build()
            model.fit(x, y, verbose=self.verbose)

            t1 = time.time()
            times.append(t1 - t0)

        self.times = times

    @property
    def result(self) -> pd.DataFrame:
        return pd.DataFrame({'Run': range(self.n_runs), 'Time': self.times})

    def plot_ts(self):
        sns.set()
        sns.lineplot(x='Run', y='Time', data=self.result)
        plt.show()

    def plot_box(self):
        sns.set()
        sns.boxplot(y='Time', data=self.result)
        plt.show()
