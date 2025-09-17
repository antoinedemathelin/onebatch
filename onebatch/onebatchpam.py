"""
One-batch PAM (k-medoids) clustering.

This module provides `OneBatchPAM`, a fast, memory-efficient approximation of
Partitioning Around Medoids (PAM). It evaluates candidate swaps on a single
"batch" of distances to reduce compute and memory while preserving clustering
quality. Distances can be optionally reweighted by the current cluster sizes to
stabilize optimization. A Cython-accelerated inner loop (`swap_eager`) performs
an eager swap phase.

The public API mirrors scikit-learn estimators with `fit`, `predict`, and
`fit_predict` methods.
"""
import numpy as np
from sklearn.metrics import pairwise_distances

from .pam import swap_eager


class OneBatchPAM:
    """
    One-batch PAM (k-medoids) clustering with optional sample-weight reweighting.

    This estimator selects `n_medoids` medoid indices from the input dataset by
    optimizing the total distance to the nearest medoid. To reduce complexity,
    the optimization is performed against a single sampled batch of candidate
    columns from the distance matrix rather than the full pairwise matrix. The
    inner swap routine is implemented in Cython for speed.

    Parameters
    ----------
    n_medoids : int, default=10
        Number of medoids (clusters) to select. Must be <= number of samples.

    distance : str or callable, default='euclidean'
        Distance metric passed to `sklearn.metrics.pairwise_distances`. Use
        'precomputed' to provide your own distance matrix as `X` in `fit`.

    batch_size : {'auto'} or int, default='auto'
        Number of candidate columns (points) used as the distance batch when
        `distance != 'precomputed'`. If 'auto', a logarithmic heuristic based on
        the dataset size and `n_medoids` is used, clamped to `[n_medoids, n]`.

    weighting : bool, default=True
        If True, reweight columns by the current cluster sizes (based on the
        argmin over the current batch) to stabilize optimization on small or
        imbalanced datasets.

    max_iter : int, default=100
        Maximum number of swap steps for the internal PAM optimization.

    tol : float, default=1e-6
        Relative improvement tolerance to stop the swap phase.

    n_jobs : int or None, default=None
        Number of jobs for `pairwise_distances`. See scikit-learn for details.

    random_state : int, numpy.random.Generator, or None, default=None
        Random seed or Generator controlling the batch sampling and medoid
        initialization.

    n_threads : int or None, default=None
        Number of threads for the Cython-accelerated `swap_eager` kernel. If
        None, it lets the kernel choose a default.

    Attributes
    ----------
    medoid_indices_ : ndarray of shape (n_medoids,), dtype=int
        Indices of the selected medoids within the input `X` passed to `fit`.

    labels_ : ndarray of shape (n_samples,), dtype=int
        Index of the nearest medoid (in `[0, n_medoids)`) for each sample.

    inertia_ : float
        Final objective value (sum of distances to the assigned medoid) as
        reported by the internal optimization.

    dist_to_nearest_medoid_ : ndarray of shape (n_samples,), dtype=float32
        Distance from each sample to its nearest medoid.

    n_iter_ : int
        Number of swap steps performed by the optimizer.

    cluster_centers_ : ndarray of shape (n_medoids, n_features)
        The medoid feature vectors, i.e., `X[medoid_indices_]`.

    solution_ : dict
        Low-level result dictionary returned by `swap_eager`, exposed for
        advanced users. Contains keys: 'medoids', 'nearest', 'loss',
        'dist_to_nearest', and 'steps'.

    Notes
    -----
    - When `distance != 'precomputed'`, this estimator samples `batch_size`
      candidate columns and computes a dense distance matrix `Dist` of shape
      `(n_samples, batch_size)` using `pairwise_distances`. The matrix is then
      normalized by its maximum value to the range `[0, 1]` for numerical
      stability.
    - If `weighting=True`, each column is multiplied by the relative cluster
      size induced by the current argmin over `Dist`.
    - The internal kernel expects C-contiguous `float32` distances.

    Examples
    --------
    >>> from scLodestar._onebatchpam.onebatchpam import OneBatchPAM
    >>> import numpy as np
    >>> X = np.random.RandomState(0).randn(100, 8).astype(np.float32)
    >>> model = OneBatchPAM(n_medoids=5, random_state=0)
    >>> model.fit(X)
    OneBatchPAM(...)
    >>> model.medoid_indices_.shape
    (5,)
    >>> labels = model.predict(X)
    >>> labels.shape
    (100,)
    """

    def __init__(
        self,
        n_medoids=10,
        distance="euclidean",
        batch_size="auto",
        weighting=True,
        max_iter=100,
        tol=1e-6,
        n_jobs=None,
        random_state=None,
        n_threads=None,
    ):
        """
        Initialize the estimator with the given configuration.

        Parameters
        ----------
        n_medoids : int, default=10
            Number of medoids (clusters) to select. Must be <= number of samples.
        distance : str or callable, default='euclidean'
            Distance metric passed to `pairwise_distances`. Use 'precomputed' to
            pass a precomputed distance matrix to `fit`.
        batch_size : {'auto'} or int, default='auto'
            Size of the candidate batch used when computing distances.
        weighting : bool, default=True
            Whether to reweight columns by current cluster sizes during
            optimization.
        max_iter : int, default=100
            Maximum number of swap steps.
        tol : float, default=1e-6
            Relative improvement tolerance to stop.
        n_jobs : int or None, default=None
            Parallelism for `pairwise_distances`.
        random_state : int, numpy.random.Generator, or None, default=None
            Random seed or Generator controlling sampling and initialization.
        n_threads : int or None, default=None
            Threads for the Cython `swap_eager` kernel (None = library default).
        """
        self.n_medoids = n_medoids
        self.distance = distance
        self.batch_size = batch_size
        self.weighting = weighting
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_threads = n_threads


    def fit(self, X):
        """
        Find the medoids on the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, m)
            - If `distance != 'precomputed'`: feature matrix with dtype that can
              be safely cast to `float32`.
            - If `distance == 'precomputed'`: precomputed distances `Dist` with
              shape `(n_samples, m)`, where each column corresponds to distances
              from all samples to a candidate point. Values are expected to be
              finite and non-negative. Distances will be treated as `float32`.

        Returns
        -------
        self : OneBatchPAM
            Fitted estimator.

        Raises
        ------
        ValueError
            If `n_medoids` exceeds the number of samples.
        """
        if self.n_medoids > X.shape[0]:
            raise ValueError("The number of medoids cannot"
                             " exceed the dataset size")
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        rng = np.random.default_rng(self.random_state)
        
        if self.distance == "precomputed":
            Dist = X
            batch_size = X.shape[1]
        else:
            if self.batch_size == "auto":
                # slightly larger batch for stability on small N; clamp to [min,N]
                est = int(150. * np.log(max(2, X.shape[0] * max(1, self.n_medoids))))
                batch_size = min(max(est, self.n_medoids), X.shape[0])
            else:
                batch_size = int(self.batch_size)
            if batch_size > X.shape[0]:
                batch_size = X.shape[0]
            batch_indexes = rng.choice(X.shape[0], batch_size, replace=False)
            Dist = pairwise_distances(X, X[batch_indexes], metric=self.distance, n_jobs=self.n_jobs)
            maxv = Dist.max()
            if maxv > 0:
                np.divide(Dist, np.float32(maxv), out=Dist, casting='same_kind')

        if self.weighting:
            # compute argmin using current Dist columns only once
            assign = Dist.argmin(1)
            sample_weight = np.zeros(Dist.shape[1], dtype=np.float32)
            unique, counts = np.unique(assign, return_counts=True)
            sample_weight[unique] = counts.astype(np.float32)
            meanw = sample_weight.mean()
            if meanw > 0:
                sample_weight /= meanw
            np.multiply(Dist, sample_weight, out=Dist, casting='same_kind')

        # Ensure C-contiguous float32 distances for Cython
        Dist = np.ascontiguousarray(Dist, dtype=np.float32)

        medoids_init = rng.choice(X.shape[0], self.n_medoids, replace=False)
        medoids_init = np.array(medoids_init, dtype=np.dtype("i"))
        
        self.solution_ = swap_eager(Dist,
                                    medoids_init,
                                    self.n_medoids,
                                    self.max_iter,
                                    X.shape[0],
                                    batch_size,
                                    np.float32(self.tol),
                                    0 if self.n_threads is None else int(self.n_threads))
        self.medoid_indices_ = self.solution_["medoids"]
        self.labels_ = self.solution_["nearest"]
        self.inertia_ = self.solution_["loss"]
        self.dist_to_nearest_medoid_ = self.solution_["dist_to_nearest"]
        self.n_iter_ = self.solution_["steps"]
        self.cluster_centers_ = X[self.medoid_indices_]
        return self


    def predict(self, X):
        """
        Assign each sample in `X` to the nearest learned medoid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix for which to compute assignments. Must be compatible
            with the metric provided by `distance`.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=int
            Index of the nearest medoid (in `[0, n_medoids)`) for each sample.

        Notes
        -----
        This method is intended for use with `distance != 'precomputed'`.
        If the estimator was fitted with `distance='precomputed'`, this
        method will attempt to call `pairwise_distances` with the provided
        `distance` value and may not be applicable. In that case, prefer using
        `labels_` obtained during `fit` or compute distances to the
        `cluster_centers_` externally and take argmin.
        """
        Dist = pairwise_distances(X, self.cluster_centers_, metric=self.distance, n_jobs=self.n_jobs)
        return Dist.argmin(1)


    def fit_predict(self, X):
        """
        Fit the model and return the medoid indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, m)
            Training data or precomputed distances, as described in `fit`.

        Returns
        -------
        medoid_indices : ndarray of shape (n_medoids,), dtype=int
            Indices of the selected medoids within the input `X`.

        Notes
        -----
        Unlike scikit-learn's usual `fit_predict` (which returns labels),
        this implementation returns the selected medoid indices for convenience.
        Cluster labels after fitting are available via the `labels_` attribute.
        """
        self.fit(X)
        return self.medoid_indices_
