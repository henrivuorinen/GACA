"""Checks for AutoGACA (preprocessing, bandwidth selection, outputs) and the CLI."""
import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from gaca import AutoGACA, Preprocessor, select_gamma


def _blobs_with_anomalies(seed=0):
    X, y = make_blobs(n_samples=3000, centers=[[0, 0, 0], [6, 0, 0], [0, 6, 0]],
                      cluster_std=0.8, random_state=seed)
    far = np.array([[30.0, 30.0, 30.0], [-30.0, 25.0, 0.0]])
    return np.vstack([X, far]), np.r_[y, [-1, -1]]


def test_finds_clusters_and_anomalies_without_settings():
    X, y = _blobs_with_anomalies()
    auto = AutoGACA().fit(X)
    genuine = y >= 0
    assert auto.n_clusters_ == 3
    assert adjusted_rand_score(y[genuine], auto.labels_[genuine]) > 0.95
    assert auto.anomaly_[~genuine].all()
    assert auto.anomaly_[genuine].mean() < 0.01
    assert auto.anomaly_score_[~genuine].min() > auto.anomaly_score_[genuine].max()


def test_labels_are_ordered_by_size():
    X, _ = _blobs_with_anomalies()
    auto = AutoGACA().fit(X)
    sizes = [c['size'] for c in auto.clusters_]
    assert sizes == sorted(sizes, reverse=True)


def test_preprocessor_drops_unusable_columns_and_logs_counts():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        'row_id': np.arange(n), 'label': rng.choice(['a', 'b'], n),
        'const': 3.0, 'x': rng.normal(size=n),
        'counts': np.round(np.exp(rng.normal(3, 1.5, n))),
        'mostly_missing': np.where(rng.random(n) < 0.8, np.nan, 1.0)})
    p = Preprocessor().fit(df)
    assert p.used_ == ['x', 'counts']
    assert p.log_['counts'] == 'log1p' and p.log_['x'] is None
    assert 'identifier' in p.decisions_['row_id']
    assert 'non-numeric' in p.decisions_['label']
    assert 'constant' in p.decisions_['const']
    assert 'missing' in p.decisions_['mostly_missing']


def test_pca_only_above_max_dims():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 25))
    assert Preprocessor(max_dims=30).fit(X).pca_ is None
    p = Preprocessor(max_dims=12).fit(X)
    assert p.pca_ is not None and 5 <= p.n_dims_ <= 12


def test_one_gaussian_has_no_stable_split():
    X = np.random.default_rng(0).normal(size=(2000, 3))
    sel = select_gamma(X)
    assert not sel['found'] or sel['k'] <= 2


def test_linking_recovers_moons():
    X, y = make_moons(2000, noise=0.06, random_state=0)
    auto = AutoGACA(link_tau=0.6).fit(X)
    assert adjusted_rand_score(y, auto.labels_) > 0.95


def test_assign_new_rows_keeps_anomaly_groups():
    X, _ = _blobs_with_anomalies()
    auto = AutoGACA().fit(X[:3000])                   # fitted without the anomalies
    out = auto.assign(np.array([[30.0, 30.0, 30.0], [30.01, 30.0, 30.0], [0.0, 0.0, 0.0]]))
    cluster = np.asarray(out['gaca_cluster'])
    group = np.asarray(out['gaca_anomaly_group'])
    assert list(cluster[:2]) == [-1, -1] and cluster[2] >= 0
    assert group[0] == group[1] >= 0


def test_cli_writes_labels_report_and_summary(tmp_path):
    pd = pytest.importorskip("pandas")
    from gaca.cli import main
    X, y = _blobs_with_anomalies()
    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    df['truth'] = y
    src = tmp_path / "data.csv"
    df.to_csv(src, index=False)
    for extra in ([], ['--chunksize', '700', '--fit-rows', '2000']):
        out = tmp_path / ("out" + ("_stream" if extra else ""))
        main(['run', str(src), '--exclude', 'truth', '--out', str(out), '--quiet'] + extra)
        labels = pd.read_csv(out / "labels.csv")
        assert len(labels) == len(df)
        assert labels.loc[labels.truth < 0, 'gaca_anomaly'].all()
        assert (out / "report.html").stat().st_size > 1000
        summary = json.loads((out / "summary.json").read_text())
        assert summary['columns_used'] == ['a', 'b', 'c']


def test_placeholder_codes_are_treated_as_missing():
    rng = np.random.default_rng(0)
    X = rng.normal(18, 1, (1000, 2))
    X[:20, 0] = -9999.0
    p = Preprocessor().fit(X)
    assert p.sentinels_ == {'x0': [-9999.0]}
    Z = p.transform(X)
    assert np.abs(Z[:20, 0]).max() < 1.0                 # imputed to the median, not extreme


def test_identifier_names_like_objid_are_dropped():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'specObjID': rng.permutation(10 ** 9 + np.arange(300)),
                       'a': rng.normal(size=300), 'b': rng.normal(size=300)})
    assert Preprocessor().fit(df).used_ == ['a', 'b']


def test_bandwidth_sets_gamma():
    X, _ = _blobs_with_anomalies()
    auto = AutoGACA(bandwidth=2.0).fit(X)
    assert auto.gamma_selection_ is None
    assert np.isclose(auto.gamma_, 1 / 8)
