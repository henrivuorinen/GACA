"""Checks for the GACA estimator (coreset, assignment, Mixture of Experts)."""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from gaca import GACA, assign


def _data(n=600, seed=0):
    X, _ = make_blobs(n_samples=n, centers=3, cluster_std=0.6, random_state=seed)
    rng = np.random.default_rng(seed)
    y = X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


def test_fit_is_reproducible_with_random_state():
    X, _ = _data()
    a = GACA(sample_size=200, random_state=7).fit(X)
    b = GACA(sample_size=200, random_state=7).fit(X)
    assert np.allclose(a.suns_, b.suns_)
    assert np.allclose(a.sun_masses_, b.sun_masses_)


def test_coreset_mass_matches_sample_size():
    X, _ = _data()
    model = GACA(sample_size=200, random_state=0).fit(X)
    assert abs(model.sun_masses_.sum() - 200) < 1e-5


def test_assign_labels_every_row():
    X, _ = _data()
    model = GACA(sample_size=200, random_state=0).fit(X)
    labels = model.assign(X)
    assert labels.shape == (len(X),)
    assert labels.min() >= 0 and labels.max() < model.n_suns_


def test_newton_assignment_matches_thesis_rule():
    X, _ = _data()
    model = GACA(sample_size=200, random_state=0, assignment='newton').fit(X)
    labels = model.assign(X)
    assert np.array_equal(labels, assign(X, model.suns_, model.sun_masses_))


def test_mixture_of_experts_predicts_finite_values():
    X, y = _data()
    model = GACA(sample_size=200, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))


def test_unsampled_anomaly_becomes_lone_sun_and_copies_join_it():
    X, _ = _data()
    anomaly = np.array([[40.0, 40.0], [40.01, 40.0], [-40.0, 40.0]])
    model = GACA(sample_size=200, random_state=0).fit(X)   # anomalies not in coreset
    genuine = model.assign(X)
    labels = model.assign(anomaly)
    assert not model.is_lone(genuine).any()
    assert model.is_lone(labels).all()
    assert labels[0] == labels[1] != labels[2]              # copies share one Lone Sun
    assert model.assign(anomaly[:1])[0] == labels[0]        # persists across batches


def test_saddle_linking_recovers_non_convex_clusters():
    X, y = make_moons(3000, noise=0.06, random_state=0)
    X = StandardScaler().fit_transform(X)
    newton = GACA(gamma_clustering=10, sample_size=1000, random_state=0,
                  assignment='newton').fit(X)
    linked = GACA(gamma_clustering=10, sample_size=1000, random_state=0,
                  link_tau=0.6).fit(X)
    assert adjusted_rand_score(y, newton.assign(X)) < 0.5
    assert adjusted_rand_score(y, linked.assign(X)) > 0.95


def test_unpulled_rows_are_predicted_by_the_global_model():
    X, y = _data()
    model = GACA(sample_size=200, random_state=0).fit(X, y)
    far = np.array([[100.0, -100.0]])
    assert np.allclose(model.predict(far), model.global_model_.predict(far))
