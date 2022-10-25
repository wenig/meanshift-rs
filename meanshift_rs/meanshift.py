from __future__ import annotations
from warnings import warn

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
            List of points with 1 dimension in the best case. If 2 dimensional arrays are used, only the first channel will be used for clustering.

        :return:
            The fitted model.
        """

        X = [self._make_1d(x) for x in X]

        self.cluster_centers, self.labels = meanshift_algorithm(
            X,
            self.n_threads,
            self.bandwidth,
            self.distance_measure
        )
        return self

    def _make_1d(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        dims = len(x.shape)
        if 1 < dims <= 2:
            return x[:, 0]
        elif dims > 2:
            raise ValueError("Time series of more than 2 dimensions are not supported!")
        return x

