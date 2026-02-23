import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from onebatch import OneBatchPAM


@pytest.fixture
def random_data():
    return np.random.RandomState(42).randn(200, 5).astype(np.float32)


class TestFitPredict:

    def test_fit_returns_self(self, random_data):
        model = OneBatchPAM(n_medoids=5, random_state=0)
        assert model.fit(random_data) is model

    def test_fit_attributes(self, random_data):
        model = OneBatchPAM(n_medoids=5, random_state=0)
        model.fit(random_data)
        assert len(model.medoid_indices_) == 5
        assert model.cluster_centers_.shape == (5, 5)
        assert isinstance(model.inertia_, float)
        assert model.inertia_ >= 0
        assert isinstance(model.n_iter_, int)
        assert model.n_iter_ >= 0

    def test_medoid_indices_valid(self, random_data):
        model = OneBatchPAM(n_medoids=5, random_state=0)
        model.fit(random_data)
        assert all(0 <= idx < random_data.shape[0] for idx in model.medoid_indices_)
        assert len(set(model.medoid_indices_)) == 5

    def test_predict_shape(self, random_data):
        model = OneBatchPAM(n_medoids=5, random_state=0)
        model.fit(random_data)
        labels = model.predict(random_data)
        assert labels.shape == (200,)
        assert all(0 <= l < 5 for l in labels)

    def test_fit_predict_returns_medoid_indices(self, random_data):
        model = OneBatchPAM(n_medoids=5, random_state=0)
        medoids = model.fit_predict(random_data)
        np.testing.assert_array_equal(medoids, model.medoid_indices_)


class TestPrecomputed:

    def test_precomputed_rectangular(self, random_data):
        n = random_data.shape[0]
        rng = np.random.RandomState(123)
        cand_idx = rng.choice(n, size=50, replace=False)
        dist_matrix = pairwise_distances(
            random_data, random_data[cand_idx], metric="euclidean"
        ).astype(np.float32)

        model = OneBatchPAM(n_medoids=5, distance="precomputed", random_state=0)
        model.fit(dist_matrix)
        assert len(model.medoid_indices_) == 5
        assert all(0 <= idx < n for idx in model.medoid_indices_)

    def test_precomputed_square(self, random_data):
        dist_matrix = pairwise_distances(random_data, metric="euclidean").astype(
            np.float32
        )
        model = OneBatchPAM(n_medoids=5, distance="precomputed", random_state=0)
        model.fit(dist_matrix)
        assert len(model.medoid_indices_) == 5


class TestDeterminism:

    def test_same_seed(self, random_data):
        m1 = OneBatchPAM(n_medoids=5, random_state=42)
        m1.fit(random_data)
        m2 = OneBatchPAM(n_medoids=5, random_state=42)
        m2.fit(random_data)
        np.testing.assert_array_equal(m1.medoid_indices_, m2.medoid_indices_)
        assert m1.inertia_ == m2.inertia_


class TestEdgeCases:

    def test_n_medoids_exceeds_raises(self):
        X = np.random.RandomState(0).randn(5, 3).astype(np.float32)
        model = OneBatchPAM(n_medoids=10, random_state=0)
        with pytest.raises(ValueError, match="exceed"):
            model.fit(X)

    def test_single_medoid(self, random_data):
        model = OneBatchPAM(n_medoids=1, random_state=0)
        model.fit(random_data)
        assert len(model.medoid_indices_) == 1

    def test_weighting_false(self, random_data):
        model = OneBatchPAM(n_medoids=5, weighting=False, random_state=0)
        model.fit(random_data)
        assert len(model.medoid_indices_) == 5

    def test_normalize_false(self, random_data):
        model = OneBatchPAM(n_medoids=5, normalize=False, random_state=0)
        model.fit(random_data)
        assert len(model.medoid_indices_) == 5

    def test_explicit_batch_size(self, random_data):
        model = OneBatchPAM(n_medoids=5, batch_size=30, random_state=0)
        model.fit(random_data)
        assert len(model.medoid_indices_) == 5

    def test_n_medoids_equals_n_samples(self):
        X = np.random.RandomState(0).randn(10, 3).astype(np.float32)
        model = OneBatchPAM(n_medoids=10, random_state=0)
        model.fit(X)
        assert len(model.medoid_indices_) == 10
