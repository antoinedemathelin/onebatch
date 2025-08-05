# OnebatchPAM

Implementation of [OneBatchPAM: A Fast and Frugal K-Medoids Algorithm](https://arxiv.org/pdf/2501.19285) (AAAI 2025)

## Install the package:
```
pip install git+https://github.com/antoinedemathelin/onebatch.git
```

## Minimal examples

From data:

```python
import numpy as np
from onebatch import OneBatchPAM

X = np.random.random((10000, 2))

km = OneBatchPAM(
    n_medoids=9,
    distance="euclidean",
    batch_size="auto",
    weighting=True,
    max_iter=100,
    tol=1e-6
)

km.fit(X)

medoids = km.medoid_indices_
```

From pre-computed distance matrix:
```python
import numpy as np
from sklearn.metrics import pairwise_distances
from onebatch import OneBatchPAM

X = np.random.random((10000, 2))

K = pairwise_distances(X, X[:1000], metric="l2")

km = OneBatchPAM(
    n_medoids=9,
    distance="precomputed",
    batch_size=None,
    weighting=True,
    max_iter=100,
    tol=1e-6
)

km.fit(K)

medoids = km.medoid_indices_
```

For visualization:
```python
import matplotlib.pyplot as plt

plt.plot(X[:, 0], X[:, 1], ".", alpha=0.3)
plt.plot(X[medoids, 0], X[medoids, 1], "o", c="k")
plt.show()
```
