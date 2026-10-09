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


def _two_level_data():
    centers = [[0, 0], [1.6, 0], [0.8, 1.4], [12, 0], [13.6, 0], [12.8, 1.4]]
    X, y = make_blobs(4000, centers=centers, cluster_std=0.3, random_state=0)
    return X, y


def test_hierarchy_finds_groups_and_subgroups():
    X, y = _two_level_data()
    auto = AutoGACA(scale='none').fit(X)
    ks = [lv['k'] for lv in auto.levels_]
    assert 2 in ks and 6 in ks
    two = next(lv for lv in auto.levels_ if lv['k'] == 2)
    six = next(lv for lv in auto.levels_ if lv['k'] == 6)
    assert adjusted_rand_score(y >= 3, two['labels']) > 0.95
    assert adjusted_rand_score(y, six['labels']) > 0.9


def test_hierarchy_levels_are_nested_and_named_by_path():
    X, _ = _two_level_data()
    auto = AutoGACA(scale='none').fit(X)
    for up, lo in zip(auto.levels_, auto.levels_[1:]):
        for c in range(lo['k']):
            parents = set(up['labels'][lo['labels'] == c].tolist())
            assert len(parents) == 1
            assert lo['names'][c].startswith(up['names'][parents.pop()] + '.')
    assert sum(lv['chosen'] for lv in auto.levels_) == 1


def test_hierarchy_columns_for_new_rows():
    pytest.importorskip("pandas")
    X, _ = _two_level_data()
    auto = AutoGACA(scale='none').fit(X)
    out = auto.assign(X[:50])
    cols = [c for c in out.columns if c.startswith('gaca_level_')]
    assert len(cols) == len(auto.levels_)
    assert (out[cols[-1]].to_numpy() == auto.result_[cols[-1]].to_numpy()[:50]).all()


def test_no_hierarchy_when_disabled_or_gamma_given():
    X, _ = _two_level_data()
    assert len(AutoGACA(scale='none', hierarchy=False).fit(X).levels_) == 1
    assert len(AutoGACA(scale='none', bandwidth=0.5).fit(X).levels_) == 1


def test_link_view_recovers_moons_and_is_reported():
    pd = pytest.importorskip("pandas")
    X, y = make_moons(2000, noise=0.06, random_state=0)
    auto = AutoGACA().fit(X)
    assert adjusted_rand_score(y, auto.linked_labels_) > 0.95
    assert auto.link_agreement_ < 0.8                      # differs: report shows it
    assert 'gaca_linked_cluster' in auto.result_.columns


def test_link_view_off_and_not_with_given_gamma():
    X, _ = _blobs_with_anomalies()
    assert AutoGACA(link_view=False).fit(X).linked_labels_ is None
    assert AutoGACA(bandwidth=2.0).fit(X).linked_labels_ is None


def test_parallel_and_serial_runs_agree():
    X, _ = _blobs_with_anomalies()
    a = AutoGACA(n_jobs=1, link_view=False).fit(X)
    b = AutoGACA(n_jobs=4, link_view=False).fit(X)
    assert np.array_equal(a.labels_, b.labels_)
    assert np.allclose(a.anomaly_score_, b.anomaly_score_)


def test_columns_with_separate_groups_keep_their_subgroups():
    centers = [[0, 0], [1.6, 0], [0.8, 1.4], [12, 0], [13.6, 0], [12.8, 1.4]]
    X, y = make_blobs(4000, centers=centers, cluster_std=0.3, random_state=0)
    auto = AutoGACA(link_view=False).fit(X)              # default robust scaling
    assert 'within-group' in auto.preprocessor_.decisions_['x0']
    assert max(adjusted_rand_score(y, lv['labels']) for lv in auto.levels_) > 0.9


def test_outliers_and_spikes_do_not_trigger_group_scaling():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 3000)
    with_outliers = np.r_[v, np.full(30, 40.0) + rng.normal(0, 0.1, 30)]   # 1% far away
    with_spike = np.r_[v, np.full(900, 5.0)]                                # imputed constant
    from gaca.auto import _within_mode_spread
    assert _within_mode_spread(with_outliers, 1.0) is None
    assert _within_mode_spread(with_spike, 1.0) is None


def test_resolution_separates_groups_at_least_that_far_apart():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.6, (500, 2)) + [i * 4.0, 0] for i in range(3)])
    assert AutoGACA(scale='none', resolution=4.0, link_view=False).fit(X).n_clusters_ == 3
    assert AutoGACA(scale='none', resolution=12.0, link_view=False).fit(X).n_clusters_ == 1


def test_resolution_and_bandwidth_are_exclusive():
    X, _ = _blobs_with_anomalies()
    with pytest.raises(ValueError):
        AutoGACA(bandwidth=1.0, resolution=3.0).fit(X)


def test_cluster_rules_describe_clusters_in_original_units():
    rng = np.random.default_rng(0)
    a = np.c_[rng.normal(10, 1, 1500), rng.normal(100, 5, 1500)]      # low x, any y
    b = np.c_[rng.normal(30, 1, 1500), rng.normal(100, 5, 1500)]      # high x
    auto = AutoGACA(link_view=False).fit(np.vstack([a, b]))
    rules = [c['rule'] for c in auto.clusters_]
    assert len(rules) == 2
    for r in rules:
        assert r['precision'] > 0.95 and r['recall'] > 0.95
        assert r['text'].startswith('x0')                              # split on x, not y
        thr = float(r['text'].split()[-1].replace(',', ''))
        assert 12 < thr < 28                                           # threshold in data units


def _stream_batch(n, p=(0.5, 0.3, 0.2), seed=0, extra=None):
    r = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [6, 0, 0], [0, 6, 0]], float)
    X = centers[r.choice(3, n, p=p)] + r.normal(0, 0.8, (n, 3))
    return X if extra is None else np.vstack([X, extra])


def test_drift_flags_real_changes_but_not_noise():
    auto = AutoGACA(link_view=False).fit(_stream_batch(4000, seed=1))
    same = [auto.drift(auto.assign(_stream_batch(300, seed=s))) for s in range(10, 20)]
    assert not any(d['drift'] for d in same)
    shifted = auto.drift(auto.assign(_stream_batch(2000, p=(0.2, 0.3, 0.5), seed=2)))
    assert shifted['drift'] and 'cluster mix' in shifted['reasons']
    far = np.random.default_rng(3).normal(0, 0.5, (100, 3)) + 20
    novel = auto.drift(auto.assign(_stream_batch(1900, seed=4, extra=far)))
    assert novel['drift'] and 'anomaly rate' in novel['reasons'] and novel['new_groups'] >= 1


def test_cli_streaming_writes_drift_file(tmp_path):
    pd = pytest.importorskip("pandas")
    from gaca.cli import main
    # a time-ordered file whose last tenth has a different mix
    X = np.vstack([_stream_batch(9000, seed=1), _stream_batch(1000, p=(0.1, 0.1, 0.8), seed=2)])
    src = tmp_path / "stream.csv"
    pd.DataFrame(X, columns=['a', 'b', 'c']).to_csv(src, index=False)
    out = tmp_path / "out"
    main(['run', str(src), '--chunksize', '1000', '--fit-rows', '5000', '--out', str(out),
          '--quiet', '--no-link-view'])
    drift = pd.read_csv(out / "drift.csv")
    assert len(drift) == 10
    assert not drift['drift'].iloc[:9].any()
    assert drift['drift'].iloc[-1] and not drift['drift'].iloc[0]


def test_compare_scores_against_known_labels_with_baselines():
    X, y = _blobs_with_anomalies()
    auto = AutoGACA(link_view=False).fit(X)
    cmp_ = auto.compare(np.where(y >= 0, y.astype(str), 'odd'), baselines=True)
    names = [m['method'] for m in cmp_['methods']]
    assert names[0] == 'GACA' and any('K-Means' in n for n in names) and 'HDBSCAN' in names
    assert cmp_['methods'][0]['ari'] > 0.95
    assert all(c['purity'] > 0.95 for c in cmp_['clusters'])


def test_cli_truth_column_is_excluded_and_reported(tmp_path):
    pd = pytest.importorskip("pandas")
    from gaca.cli import main
    X, y = _blobs_with_anomalies()
    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    df['kind'] = np.where(y >= 0, y, -1)
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)
    out = tmp_path / "o"
    main(['run', str(src), '--truth', 'kind', '--out', str(out), '--quiet', '--no-link-view'])
    summary = json.loads((out / "summary.json").read_text())
    assert summary['columns_used'] == ['a', 'b', 'c']
    assert summary['comparison']['methods'][0]['ari'] > 0.95
    assert 'Comparison with known labels' in (out / "report.html").read_text()


def test_level_choice_prefers_the_finer_of_equally_long_plateaus():
    # The benchmark's 'anisotropic' set. At some other sizes the coarser plateau
    # is a grid point longer and wins (see docs/theory.md section 5).
    X, y = make_blobs(6000, centers=4, cluster_std=0.8, random_state=3)
    X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])                     # the 'anisotropic' set
    auto = AutoGACA(link_view=False).fit(X)
    assert adjusted_rand_score(y, auto.labels_) > 0.9


def test_rare_dense_group_scores_as_anomalous_without_new_flags():
    rng = np.random.default_rng(0)
    bulk = np.vstack([rng.normal(0, 1, (3000, 3)), rng.normal([8, 0, 0], 1, (3000, 3))])
    rare = rng.normal([4, 9, 0], 0.3, (60, 3))                       # 1% of rows, dense, apart
    X = np.vstack([bulk, rare])
    on = AutoGACA(link_view=False).fit(X)
    off = AutoGACA(link_view=False, rare_share=0).fit(X)
    from sklearn.metrics import roc_auc_score
    truth = np.r_[np.zeros(len(bulk), bool), np.ones(len(rare), bool)]
    assert roc_auc_score(truth, on.anomaly_score_) > roc_auc_score(truth, off.anomaly_score_)
    assert roc_auc_score(truth, on.anomaly_score_) > 0.95
    assert np.array_equal(on.anomaly_, off.anomaly_)
