from __future__ import annotations
from warnings import warn

from .meanshift_rs import meanshift_algorithm
from typing import Optional, List, Union
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

    def fit(self, X: Union[List[npt.NDArray[np.float32]], npt.NDArray[np.float32]]) -> MeanShift:
        """
        Fit the model to the data.

        :param X: Union[List[np.ndarray], np.ndarray]
            List of points with 1 dimension or 2 dimensional ndarray (n, dim).

        :return:
            The fitted model.
        """

        if type(X) == list:
            X = self._make_matrix(X)

        self.cluster_centers, self.labels = meanshift_algorithm(
            X,
            self.n_threads,
            self.bandwidth,
            self.distance_measure
        )
        return self

    def _make_matrix(self, X: List[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        max_dims = max(x.shape[0] for x in X)
        matrix = np.zeros((len(X), max_dims)) + np.nan
        for i, x in enumerate(X):
            matrix[i, :len(x)] = x
        return matrix
