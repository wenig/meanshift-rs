from __future__ import annotations

from .meanshift_rs import meanshift_algorithm
from typing import Optional, List
import numpy as np
import numpy.typing as npt


class MeanShift:
    """
    MeanShift class that calls the rust binding.

    Arguments
    ---------
    n_threads : int
        Threads used for running the algorithm (default=-1).
    bandwidth : Optional[float]
        Evtl. bandwidth value. If None, it will be estimated (default=None).
    distance_measure : str
        Distance measure to use inside the algorithm (default="euclidean").
    """
    def __init__(self,
                 n_threads: int = -1,
                 bandwidth: Optional[float] = None,
                 distance_measure: str = "euclidean"):
        self.n_threads = n_threads
        self.bandwidth = bandwidth
        self.distance_measure = distance_measure
        self.cluster_centers: Optional[npt.NDArray[np.float32]] = None
        self.labels: Optional[List[int]] = None

    def fit(self, X: List[npt.NDArray[np.float32]]) -> MeanShift:
        """
        Fit the model to the data.

        :param X: List[np.ndarray]
            List of points.

        :return:
            The fitted model.
        """
        self.cluster_centers, self.labels = meanshift_algorithm(
            X,
            self.n_threads,
            self.bandwidth,
            self.distance_measure
        )
        return self
