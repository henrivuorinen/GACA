"""Physics sanity checks for Solar Genesis.

Run with ``pytest``. Running this file directly also draws the before/after plot.
"""
import numpy as np
from sklearn.datasets import make_blobs

from gaca import solar_genesis

GAMMA, THETA, EPSILON, N_ITER = 1.0, 0.5, 0.1, 20


def _three_blobs(n_samples=300):
    X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.6, random_state=42)
    return X


def test_mass_is_conserved():
    X = _three_blobs()
    _, masses = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER,
                              theta=THETA, epsilon=EPSILON)
    assert abs(masses.sum() - len(X)) < 1e-5


def test_active_set_only_coarsens():
    X = _three_blobs()
    history = []
    solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER, theta=THETA,
                  epsilon=EPSILON, history=history)
    assert history, "expected at least one iteration"
    for record in history:
        assert record['n_after'] <= record['n_before']


def test_three_blobs_condense_into_three_heavy_suns():
    X = _three_blobs()
    _, masses = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER,
                              theta=THETA, epsilon=EPSILON)
    top3 = np.sort(masses)[-3:].sum()
    assert top3 >= 0.9 * len(X)


def test_far_outlier_becomes_a_lone_sun():
    X = _three_blobs()
    outlier = np.array([[50.0, 50.0]])
    suns, masses = solar_genesis(np.vstack([X, outlier]), gamma=GAMMA,
                                 n_iterations=N_ITER, theta=THETA, epsilon=EPSILON)
    nearest = np.argmin(np.linalg.norm(suns - outlier, axis=1))
    assert masses[nearest] == 1.0
    assert np.allclose(suns[nearest], outlier[0])


def _plot():
    import matplotlib.pyplot as plt

    X = _three_blobs()
    suns, masses = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER,
                                 theta=THETA, epsilon=EPSILON, verbose=True)
    print(f"{len(X)} points -> {len(suns)} Suns, total mass {masses.sum():.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.3, s=10, label='Original data')
    plt.title("Initial state (t=0)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.1, s=5)
    plt.scatter(suns[:, 0], suns[:, 1], s=masses * 5, c='red', alpha=0.7,
                edgecolors='black', label='Suns')
    plt.title(f"Final state: {len(suns)} Suns")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _plot()


def test_exact_and_barnes_hut_agree():
    X = _three_blobs()
    s1, m1 = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER, theta=THETA,
                           epsilon=EPSILON, method='barnes_hut')
    s2, m2 = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER, theta=THETA,
                           epsilon=EPSILON, method='exact')
    assert len(s1) == len(s2)
    assert np.allclose(np.sort(m1), np.sort(m2))


def test_members_point_to_the_sun_each_row_condensed_into():
    X = _three_blobs()
    suns, masses, members = solar_genesis(X, gamma=GAMMA, n_iterations=N_ITER,
                                          epsilon=EPSILON, return_members=True)
    assert members.shape == (len(X),)
    assert np.array_equal(np.bincount(members, minlength=len(suns)), masses)


def test_critical_gap_matches_the_two_sun_map_and_grows_slowly():
    from gaca import critical_gap
    assert 3.0 < critical_gap(20, spread=0.0) < 3.3      # docs/theory.md, compact limit
    assert critical_gap(20, spread=1.0) > critical_gap(20, spread=0.0)
    assert critical_gap(200, spread=0.0) < 1.35 * critical_gap(20, spread=0.0)   # sqrt(ln T)
