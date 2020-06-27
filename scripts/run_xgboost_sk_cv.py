from ml_benchmarks.benchmark.benchmark import Benchmark
from ml_benchmarks.data.synthetic import Synthetic
from ml_benchmarks.models.xgboost_sk_grid import XGBoostSKGrid


def run():
    bench = Benchmark(XGBoostSKGrid(n_jobs=-1, verbose=1), Synthetic(), n_runs=3)
    bench.run()
    print(bench.times)
    bench.plot_ts()
    bench.plot_box()


if __name__ == "__main__":
    run()
