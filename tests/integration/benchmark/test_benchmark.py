import unittest

from ml_benchmarks.benchmark.benchmark import Benchmark
from ml_benchmarks.data.synthetic import Synthetic
from ml_benchmarks.models.xgboost_sk_grid import XGBoostSKGrid


class TestBenchmark(unittest.TestCase):
    def test_run_xgboost_sk_grid_with_synthetic_data(self):
        bench = Benchmark(XGBoostSKGrid(n_jobs=-1, verbose=1), Synthetic(), n_runs=1)
        bench.run()

        self.assertIsInstance(bench.times, list)
