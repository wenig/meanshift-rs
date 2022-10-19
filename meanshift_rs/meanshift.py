from __future__ import annotations

from .meanshift_rs import meanshift_algorithm
from typing import Optional, List
import numpy as np
import numpy.typing as npt


class MeanShift:
    def __init__(self,
                 n_threads: int = 1,
                 bandwidth: Optional[float] = None,
                 distance_measure: str = "euclidean"):
        self.n_threads = n_threads
        self.bandwidth = bandwidth
        self.distance_measure = distance_measure
        self.cluster_centers: Optional[npt.NDArray[np.float32]] = None
        self.labels: Optional[List[int]] = None

    def fit(self, X: List[npt.NDArray[np.float32]]) -> MeanShift:
        self.cluster_centers, self.labels = meanshift_algorithm(
            X,
            self.n_threads,
            self.bandwidth,
            self.distance_measure
        )
        return self
