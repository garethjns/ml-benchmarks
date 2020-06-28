import numpy as np

from ml_benchmarks import __version__
from ml_benchmarks.benchmark.benchmark import Benchmark
from ml_benchmarks.data.synthetic import Synthetic
from ml_benchmarks.models.xgboost_sk_grid import XGBoostSKGrid


def run():
    bench = Benchmark(XGBoostSKGrid(n_jobs=-1, verbose=1), Synthetic(), n_runs=20)
    bench.run()
    print(bench.times)
    print(f"Version: {__version__}. "
          f"Mean time: {np.round(np.mean(bench.times), 2)} +/- {np.round(np.std(bench.times), 2)}")
    bench.plot_box()
    bench.plot_ts()


if __name__ == "__main__":
    run()
