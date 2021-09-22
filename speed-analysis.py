from sklearn.cluster import MeanShift
from meanshift_rs_py import MeanShift as MeanShiftRS
from sklearn.datasets import load_iris, load_wine
import numpy as np
import timeit


def run_original(data, n_threads: int):
    ms = MeanShift(n_jobs=n_threads)
    ms.fit_predict(data)


def run_rust(data, n_threads: int, actors: bool):
    ms = MeanShiftRS(n_threads=n_threads, use_actors=actors)
    ms.fit(data)


def speed_test():
    print(f"algorithm\t|\ttime")
    print("---------------------------")
    data = load_wine()["data"].astype(np.float32)
    data = np.repeat(data, repeats=10, axis=0)

    rust_time = timeit.timeit(lambda: run_rust(data, 8, True), number=3)
    print(f"rust-actors\t|\t{rust_time}")

    original_time = timeit.timeit(lambda: run_original(data, 8), number=3)
    print(f"original\t|\t{original_time}")



if __name__ == "__main__":
    speed_test()
