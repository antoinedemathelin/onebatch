import numpy as np
from sklearn.metrics import pairwise_distances

from .pam import swap_eager


class OneBatchPAM:

    def __init__(
        self,
        n_medoids=10,
        distance="euclidean",
        batch_size="auto",
        weighting=True,
        max_iter=100,
        tol=1e-6
    ):
        self.n_medoids = n_medoids
        self.distance = distance
        self.batch_size = batch_size
        self.weighting = weighting
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X):
        """
        Find the medoids.
        """
        if self.n_medoids > X.shape[0]:
            raise ValueError("The number of medoids cannot"
                             " exceed the dataset size")
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        if self.distance == "precomputed":
            Dist = X
        else:
            if self.batch_size == "auto":
                batch_size = int(100. * np.log(X.shape[0] * self.n_medoids))
            else:
                batch_size = self.batch_size
            if batch_size > X.shape[0]:
                batch_size = X.shape[0]
            batch_indexes = np.random.choice(X.shape[0],
                                             batch_size,
                                             replace=False)
            Dist = pairwise_distances(X, X[batch_indexes], metric=self.distance)   
            np.divide(Dist, np.float32(Dist.max()), out=Dist, casting='same_kind')

        if self.weighting:
            sample_weight = np.zeros(Dist.shape[1], dtype=np.float32)
            unique, counts = np.unique(Dist.argmin(1).ravel(), return_counts=True)
            sample_weight[unique] = counts
            sample_weight /= sample_weight.mean()
            sample_weight = sample_weight.astype(np.float32)
            np.multiply(Dist, sample_weight, out=Dist, casting='same_kind')

        medoids_init = np.random.choice(X.shape[0], self.n_medoids, replace=False)
        medoids_init = np.array(medoids_init, dtype=np.dtype("i"))
        
        self.solution_ = swap_eager(Dist,
                                    medoids_init,
                                    self.n_medoids,
                                    self.max_iter,
                                    X.shape[0],
                                    batch_size,
                                    np.float32(self.tol))
        self.medoid_indices_ = self.solution_["medoids"]
        self.labels_ = self.solution_["nearest"]
        self.inertia_ = self.solution_["loss"]
        self.dist_to_nearest_medoid_ = self.solution_["dist_to_nearest"]
        self.n_iter_ = self.solution_["steps"]
        self.cluster_centers_ = X[self.medoid_indices_]
        return self


    def predict(self, X):
        Dist = pairwise_distances(X, self.cluster_centers_, metric=self.distance)   
        return Dist.argmin(1)


    def fit_predict(self, X):
        self.fit(X)
        return self.medoid_indices_