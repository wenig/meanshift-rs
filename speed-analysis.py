import sys
from pathlib import Path
sys.path.remove(str(Path(".").absolute()))

from sklearn.cluster import MeanShift
from meanshift_rs import MeanShift as MeanShiftRS
from sklearn.datasets import load_wine
import numpy as np
import timeit


def run_original(data, n_threads: int):
    ms = MeanShift(n_jobs=n_threads)
    ms.fit_predict(data)


def run_rust(data, n_threads: int):
    ms = MeanShiftRS(n_threads=n_threads)
    data = [d for d in data]
    ms.fit(data)


def speed_test():
    print(f"algorithm\t|\tthreads\t|\ttime")
    print("------------------------------------------------------")
    data = load_wine()["data"].astype(np.float64)
    data = np.repeat(data, repeats=10, axis=0)

    rust_time = timeit.timeit(lambda: run_rust(data, 8), number=3)
    print(f"rust-rayon\t|\t8\t|\t{rust_time}")

    rust_time = timeit.timeit(lambda: run_rust(data, 4), number=3)
    print(f"rust-rayon\t|\t4\t|\t{rust_time}")

    rust_time = timeit.timeit(lambda: run_rust(data, 1), number=3)
    print(f"rust-rayon\t|\t1\t|\t{rust_time}")

    original_time = timeit.timeit(lambda: run_original(data, 8), number=3)
    print(f"original\t|\t8\t|\t{original_time}")


if __name__ == "__main__":
    speed_test()
