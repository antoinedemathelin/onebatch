from .pam import swap_eager
import numpy as np
from sklearn.metrics import pairwise_distances


def one_batch_pam(X, K=1, distance="euclidean", batch_size=1000, verbose=1, weight="debias"):
    
    N = X.shape[0]    

    if weight == "lwcs":
        x_mean = X.mean(0)
        dist_to_mean = pairwise_distances(X, x_mean.reshape(1, -1), metric=distance).ravel()
        probas = 0.5 * (1 / N) + 0.5 * dist_to_mean / dist_to_mean.sum()
        probas /= probas.sum()
        batch_indexes = np.random.choice(N,
                                         batch_size,
                                         replace=False,
                                         p=probas)
        Dist = pairwise_distances(X, X[batch_indexes], metric=distance)   
        np.divide(Dist, np.float32(Dist.max()), out=Dist, casting='same_kind')
        sample_weight = probas[batch_indexes]
        sample_weight /= sample_weight.mean()
        sample_weight = sample_weight.astype(np.float32)
        Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
        np.multiply(Dist, sample_weight, out=Dist, casting='same_kind')
    else:
        batch_indexes = np.random.choice(N,
                                         batch_size,
                                         replace=False)
        Dist = pairwise_distances(X, X[batch_indexes], metric=distance)   
        np.divide(Dist, np.float32(Dist.max()), out=Dist, casting='same_kind')
    
        if weight == "debias":
            Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
        elif weight == "nniw":
            sample_weight = np.zeros(batch_size, dtype=np.float32)
            unique, counts = np.unique(Dist.argmin(1).ravel(), return_counts=True)
            sample_weight[unique] = counts
            sample_weight /= sample_weight.mean()
            sample_weight = sample_weight.astype(np.float32)
            Dist[batch_indexes, np.arange(batch_size)] = np.float32(1.)
            np.multiply(Dist, sample_weight, out=Dist, casting='same_kind')
        else:
            pass
    
    medoids = np.random.choice(N, K, replace=False)
    meds = np.array(medoids, dtype=np.dtype("i"))
    
    new_medoids = swap_eager(Dist, meds, K, 100, N, batch_size, np.float32(1e-6))
    return new_medoids