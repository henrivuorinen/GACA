"""Checks for the GACA estimator (coreset, assignment, Mixture of Experts)."""
import numpy as np
from sklearn.datasets import make_blobs

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
    assert labels.min() >= 0 and labels.max() < len(model.suns_)
    assert np.array_equal(labels, assign(X, model.suns_, model.sun_masses_))


def test_mixture_of_experts_predicts_finite_values():
    X, y = _data()
    model = GACA(sample_size=200, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
