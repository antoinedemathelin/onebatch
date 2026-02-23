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


class TestSampleWeight:

    def test_uniform_weight_matches_no_weight(self, random_data):
        model1 = OneBatchPAM(n_medoids=5, random_state=42)
        model1.fit(random_data)

        uniform_w = np.ones(random_data.shape[0], dtype=np.float32)
        model2 = OneBatchPAM(n_medoids=5, random_state=42)
        model2.fit(random_data, sample_weight=uniform_w)

        np.testing.assert_array_equal(model1.medoid_indices_, model2.medoid_indices_)

    def test_sample_weight_changes_result(self, random_data):
        rng = np.random.RandomState(99)
        weights = rng.exponential(1.0, size=random_data.shape[0]).astype(np.float32)
        model = OneBatchPAM(n_medoids=5, random_state=42)
        model.fit(random_data, sample_weight=weights)
        assert len(model.medoid_indices_) == 5

    def test_sample_weight_ignored_when_no_weighting(self, random_data):
        model1 = OneBatchPAM(n_medoids=5, weighting=False, random_state=42)
        model1.fit(random_data)

        weights = np.ones(random_data.shape[0], dtype=np.float32) * 5.0
        model2 = OneBatchPAM(n_medoids=5, weighting=False, random_state=42)
        model2.fit(random_data, sample_weight=weights)

        np.testing.assert_array_equal(model1.medoid_indices_, model2.medoid_indices_)

    def test_sample_weight_with_precomputed(self, random_data):
        dist_matrix = pairwise_distances(random_data).astype(np.float32)
        weights = np.ones(random_data.shape[0], dtype=np.float32)
        model = OneBatchPAM(n_medoids=5, distance="precomputed", random_state=42)
        model.fit(dist_matrix, sample_weight=weights)
        assert len(model.medoid_indices_) == 5


class TestDistInit:

    def test_dist_init_large_no_effect(self, random_data):
        model1 = OneBatchPAM(n_medoids=5, random_state=42, batch_size=50)
        model1.fit(random_data)

        dist_init = np.full(50, 1e6, dtype=np.float32)
        model2 = OneBatchPAM(n_medoids=5, random_state=42, batch_size=50)
        model2.fit(random_data, Dist_init=dist_init)

        np.testing.assert_array_equal(model1.medoid_indices_, model2.medoid_indices_)

    def test_dist_init_small_changes_result(self, random_data):
        dist_init = np.full(50, 1e-6, dtype=np.float32)
        model = OneBatchPAM(n_medoids=5, random_state=42, batch_size=50)
        model.fit(random_data, Dist_init=dist_init)
        assert len(model.medoid_indices_) == 5

    def test_dist_init_precomputed(self, random_data):
        dist_matrix = pairwise_distances(random_data).astype(np.float32)
        batch_size = dist_matrix.shape[1]
        dist_init = np.full(batch_size, 1e6, dtype=np.float32)
        model = OneBatchPAM(n_medoids=5, distance="precomputed", random_state=42)
        model.fit(dist_matrix, Dist_init=dist_init)
        assert len(model.medoid_indices_) == 5


class TestFrozenMedoids:

    def test_frozen_medoids_basic(self, random_data):
        frozen = random_data[:3]
        model = OneBatchPAM(n_medoids=5, random_state=42)
        model.fit(random_data, frozen_medoids=frozen)
        assert len(model.medoid_indices_) == 5

    def test_frozen_and_dist_init_raises(self, random_data):
        frozen = random_data[:3]
        dist_init = np.ones(50, dtype=np.float32)
        model = OneBatchPAM(n_medoids=5, random_state=42, batch_size=50)
        with pytest.raises(ValueError, match="Cannot specify both"):
            model.fit(random_data, frozen_medoids=frozen, Dist_init=dist_init)

    def test_frozen_with_precomputed_raises(self, random_data):
        dist_matrix = pairwise_distances(random_data).astype(np.float32)
        frozen = random_data[:3]
        model = OneBatchPAM(n_medoids=5, distance="precomputed", random_state=42)
        with pytest.raises(ValueError, match="not supported"):
            model.fit(dist_matrix, frozen_medoids=frozen)


class TestCallableDistance:

    def test_callable_matches_string(self, random_data):
        model1 = OneBatchPAM(n_medoids=5, distance="euclidean", random_state=42)
        model1.fit(random_data)

        def my_euclidean(X, Y, n_jobs):
            return pairwise_distances(X, Y, metric="euclidean", n_jobs=n_jobs)

        model2 = OneBatchPAM(n_medoids=5, distance=my_euclidean, random_state=42)
        model2.fit(random_data)

        np.testing.assert_array_equal(model1.medoid_indices_, model2.medoid_indices_)

    def test_callable_predict(self, random_data):
        def my_euclidean(X, Y, n_jobs):
            return pairwise_distances(X, Y, metric="euclidean", n_jobs=n_jobs)

        model = OneBatchPAM(n_medoids=5, distance=my_euclidean, random_state=42)
        model.fit(random_data)
        labels = model.predict(random_data)
        assert labels.shape == (200,)
        assert all(0 <= l < 5 for l in labels)

    def test_callable_with_frozen(self, random_data):
        def my_euclidean(X, Y, n_jobs):
            return pairwise_distances(X, Y, metric="euclidean", n_jobs=n_jobs)

        frozen = random_data[:3]
        model = OneBatchPAM(n_medoids=5, distance=my_euclidean, random_state=42)
        model.fit(random_data, frozen_medoids=frozen)
        assert len(model.medoid_indices_) == 5
