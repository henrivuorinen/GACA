"""The GACA estimator: Solar Genesis on a coreset, Particle Accretion for every
row, and an optional Mixture of Experts (one local model per Sun)."""
import numpy as np
from sklearn.base import BaseEstimator
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge, LogisticRegression

from .genesis import kernel_pull, saddle_link, solar_genesis


class GACA(BaseEstimator):
    """Gravitational Accretion Clustering Algorithm.

    ``fit(X)`` draws a uniform coreset of ``sample_size`` rows, runs Solar Genesis
    on it and stores the Suns (``suns_``, ``sun_masses_``). ``assign(X)`` routes
    any number of rows to those Suns. If ``y`` is passed to ``fit``, one local
    expert (Ridge, or logistic regression for ``task_type='classification'``) is
    trained per Sun and ``predict`` routes each row to its Sun's expert.

    Input is expected to be standardised and low-dimensional (roughly d <= 10;
    project with PCA first if needed, see the README).

    Assignment (``assignment``):

    * ``'pull'`` (default). A row goes to the Sun whose coreset members exert
      the largest Gaussian pull on it, sum_{j in Sun k} exp(-gamma ||x - z_j||^2),
      over its ``n_neighbors`` nearest coreset members. This keeps the shape the
      dynamics produced (crescents, rings) instead of cutting space into
      Voronoi-like cells. A row on which the total pull is below ``kappa_``
      would not move if it were inserted into the coreset (thesis Prop. 3.8), so
      it is a Lone Sun: ``assign`` registers it as a new Lone Sun, or adds it to
      an earlier one that pulls it. Anomalies therefore no longer have to be in
      the coreset to be isolated.
    * ``'newton'``. The thesis rule, argmax_k M_k / d_k^2.

    ``kappa`` is relative: the absolute threshold ``kappa_`` is ``kappa`` times
    the median external pull at the coreset members, so it is unaffected by the
    choice of gamma, dimension and coreset size.
    """

    def __init__(self, gamma_clustering=1.0, n_iterations=20,
                 theta=0.5, epsilon=0.05, sample_size=5000,
                 task_type='regression', random_state=None, eta=0.5,
                 min_expert_size=10, verbose=False, method='exact',
                 assignment='pull', kappa=1e-3, link_tau=None, n_neighbors=32,
                 lone_share=1e-3):
        self.gamma_clustering = gamma_clustering
        self.n_iterations = n_iterations
        self.theta = theta
        self.epsilon = epsilon
        self.sample_size = sample_size
        self.task_type = task_type
        self.random_state = random_state
        self.eta = eta
        # A Sun with fewer than this many training points does not get its own
        # regression expert: a Ridge fitted on one or two points in a 5-dimensional
        # space is degenerate. Such Suns fall back to a constant predictor, which is
        # the behaviour thesis section 4.3.4 describes.
        self.min_expert_size = min_expert_size
        self.verbose = verbose
        # 'exact' (vectorised, no approximation) or 'barnes_hut' (thesis tree).
        self.method = method
        self.assignment = assignment
        self.kappa = kappa
        # If set, Suns joined by a density bridge of at least link_tau x the
        # lower peak are merged after genesis (see genesis.saddle_link).
        self.link_tau = link_tau
        self.n_neighbors = n_neighbors
        # A genesis Sun holding at most this share of the coreset (and at least
        # mass 1) counts as a Lone Sun. The thesis definition is mass == 1, which
        # misses a rare anomaly group once two of its copies are sampled.
        self.lone_share = lone_share

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit(self, X, y=None):
        """Solar Genesis on a coreset, then (if ``y`` is given) local experts."""
        if self.assignment not in ('pull', 'newton'):
            raise ValueError(f"assignment must be 'pull' or 'newton', got {self.assignment!r}")
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        # 1. Subsampling (coreset extraction)
        rng = np.random.default_rng(self.random_state)
        if n_samples > self.sample_size:
            idx = rng.choice(n_samples, self.sample_size, replace=False)
            X_core = X[idx]
        else:
            X_core = X

        self._log(f"Fitting GACA: running Solar Genesis on {len(X_core)} points...")

        # 2. Solar Genesis: the Suns, their accumulated masses, and the Sun each
        #    coreset row condensed into
        self.suns_, self.sun_masses_, self.n_iters_, members = solar_genesis(
            X_core,
            gamma=self.gamma_clustering,
            n_iterations=self.n_iterations,
            theta=self.theta,
            epsilon=self.epsilon,
            eta=self.eta,
            method=self.method,
            return_iters=True,
            return_members=True,
            verbose=self.verbose,
        )

        self._log(f"Solar Genesis complete: {len(self.suns_)} Suns identified "
                  f"in {self.n_iters_} iterations.")

        # 2b. Pull-based assignment structures: the coreset members keep the label
        # of the Sun they condensed into, and the Lone-Sun threshold is set
        # relative to the typical pull inside the coreset.
        if self.assignment == 'pull':
            w = np.ones(len(X_core))
            pull = kernel_pull(X_core, X_core, w, self.gamma_clustering) - w
            self.kappa_ = self.kappa * float(np.median(pull))

            if self.link_tau is not None and len(self.suns_) > 1:
                new_label, _ = saddle_link(X_core, members, w, self.gamma_clustering,
                                           tau=self.link_tau, kappa=self.kappa_,
                                           pull=pull)
                M = np.bincount(new_label, weights=self.sun_masses_)
                self.suns_ = np.column_stack([
                    np.bincount(new_label, weights=self.sun_masses_ * self.suns_[:, j])
                    for j in range(self.suns_.shape[1])]) / M[:, None]
                self.sun_masses_ = M
                members = new_label[members]
                self._log(f"Saddle linking: {len(new_label)} -> {len(M)} Suns.")

            self.core_ = X_core.copy()
            self.core_labels_ = members
            self.core_tree_ = cKDTree(self.core_)
        self.n_genesis_suns_ = len(self.suns_)
        self.lone_pos_ = np.empty((0, X.shape[1]))
        self.lone_labels_ = np.empty(0, dtype=int)

        # 3. Local expert models. The WHOLE training set (not just the coreset)
        # is assigned to the Suns to train them. Rows that no Sun pulls are left
        # to the global model.
        if y is not None:
            self._log("Training local expert models...")
            y = np.asarray(y)
            assignments = self._route(X)
            self.local_models_ = []

            # Global fallback. Every Sun that receives no training point at all
            # routes here, as do Suns too small to support a local fit. Without it
            # a query routed to an unpopulated Sun would be predicted as 0.0, which
            # is not a neutral default for any real target.
            if self.task_type == 'classification':
                self.global_model_ = LogisticRegression(max_iter=500, solver='lbfgs')
                self.global_model_.fit(X, y)
                constant_of = lambda yc: np.bincount(yc.astype(int)).argmax()
            else:
                self.global_model_ = Ridge(alpha=1.0).fit(X, y)
                constant_of = lambda yc: float(np.mean(yc))

            for i in range(len(self.suns_)):
                mask = (assignments == i)
                X_cluster = X[mask]
                y_cluster = y[mask]

                if len(y_cluster) == 0:
                    # Unpopulated Sun: defer to the global model at predict time.
                    self.local_models_.append(('fallback', None))
                elif len(y_cluster) < self.min_expert_size:
                    # Lone / near-singleton Sun: constant predictor.
                    self.local_models_.append(('constant', constant_of(y_cluster)))
                elif self.task_type == 'classification':
                    if len(np.unique(y_cluster)) < 2:
                        self.local_models_.append(('constant', constant_of(y_cluster)))
                    else:
                        local = LogisticRegression(max_iter=500, solver='lbfgs')
                        local.fit(X_cluster, y_cluster)
                        self.local_models_.append(('model', local))
                else:
                    local = Ridge(alpha=1.0)
                    local.fit(X_cluster, y_cluster)
                    self.local_models_.append(('model', local))

            kinds = [k for k, _ in self.local_models_]
            self._log(f"  experts: {kinds.count('model')} local, "
                      f"{kinds.count('constant')} constant, "
                      f"{kinds.count('fallback')} unpopulated (routed to global)")

        return self

    def assign(self, X, return_pull=False):
        """Particle Accretion: the index of the Sun each row of X belongs to.

        Rows can be passed in batches of any size, so arbitrarily large data can
        be streamed through a fitted model. With ``assignment='pull'``, rows that
        no Sun pulls become Lone Suns with labels from ``n_genesis_suns_``
        upwards. Lone Suns persist across calls, so later copies of an anomaly
        join the Lone Sun of the first. With ``return_pull`` the total Gaussian
        pull (``'pull'``) or the winning Newtonian pull (``'newton'``) is
        returned as well.
        """
        X = np.asarray(X, dtype=float)
        if self.assignment == 'newton':
            return self._assign_to_suns(X, return_pull=return_pull)
        labels, pull = self._pull_route(X)
        lone = np.flatnonzero(labels < 0)
        if len(lone):
            labels[lone] = self._register_lone(X[lone])
        return (labels, pull) if return_pull else labels

    def fit_assign(self, X):
        """Fit on X and return the Sun label of every row."""
        return self.fit(X).assign(X)

    @property
    def n_suns_(self):
        """Number of Suns, including Lone Suns registered by ``assign``."""
        return self.n_genesis_suns_ + len(np.unique(self.lone_labels_))

    def is_lone(self, labels):
        """True where a label is a Lone Sun: one registered by ``assign``, or a
        genesis Sun whose coreset mass is at most max(1, lone_share x coreset)."""
        labels = np.asarray(labels)
        out = labels >= self.n_genesis_suns_
        core = ~out
        limit = max(1.0, self.lone_share * self.sun_masses_.sum())
        out[core] = self.sun_masses_[labels[core]] <= limit
        return out

    def _route(self, X):
        """Labels without registering new Lone Suns (-1 where nothing pulls)."""
        if self.assignment == 'newton':
            return self._assign_to_suns(X)
        return self._pull_route(X)[0]

    def _pull_route(self, X):
        """Per-Sun Gaussian pull of the nearest coreset members.

        Returns the strongest-pulling genesis Sun (-1 if the total pull is below
        ``kappa_``) and the total pull of every row.
        """
        n, n_suns = len(X), self.n_genesis_suns_
        K = min(self.n_neighbors, len(self.core_))
        labels = np.empty(n, dtype=int)
        total = np.empty(n)
        chunk = max(1000, int(2e7 // max(n_suns, 1)))   # bounds the n x K pull table
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            dist, idx = self.core_tree_.query(Xc, k=K, workers=-1)
            if K == 1:
                dist, idx = dist[:, None], idx[:, None]
            W = np.exp(-self.gamma_clustering * dist ** 2)
            S = self.core_labels_[idx]
            acc = np.zeros((len(Xc), n_suns))
            rows = np.arange(len(Xc))
            for c in range(K):
                acc[rows, S[:, c]] += W[:, c]
            labels[s:s + chunk] = acc.argmax(axis=1)
            total[s:s + chunk] = W.sum(axis=1)
        labels[total < self.kappa_] = -1
        return labels, total

    def _register_lone(self, X_lone):
        """Give each unpulled row a Lone Sun: the existing Lone Sun that pulls it
        at least ``kappa_``, or a new one. Rows are taken in order; this is
        meant for the rare rows that no genesis Sun pulls."""
        out = np.empty(len(X_lone), dtype=int)
        pos, lab = list(self.lone_pos_), list(self.lone_labels_)
        next_label = self.n_genesis_suns_ + (len(set(lab)))
        for r, x in enumerate(X_lone):
            best = -1
            if pos:
                w = np.exp(-self.gamma_clustering * np.sum((np.asarray(pos) - x) ** 2, axis=1))
                acc = np.bincount(np.asarray(lab) - self.n_genesis_suns_, weights=w)
                if acc.max() >= self.kappa_:
                    best = int(acc.argmax()) + self.n_genesis_suns_
            if best < 0:
                best, next_label = next_label, next_label + 1
            pos.append(x)
            lab.append(best)
            out[r] = best
        self.lone_pos_ = np.asarray(pos).reshape(-1, X_lone.shape[1])
        self.lone_labels_ = np.asarray(lab, dtype=int)
        return out

    def _assign_to_suns(self, X_batch, return_pull=False):
        """Vectorised Newtonian assignment (Mass / r^2) of rows to Suns.

        Returns the array of Sun assignments (one integer per row). If
        ``return_pull`` is True, also returns the strength of the winning pull
        for each row (used for OOD rejection).
        """
        sq_X = np.sum(X_batch ** 2, axis=1)[:, np.newaxis]
        sq_S = np.sum(self.suns_ ** 2, axis=1)[np.newaxis, :]
        dot_XS = np.dot(X_batch, self.suns_.T)
        dists_sq = np.maximum(sq_X + sq_S - 2 * dot_XS, 1e-9)
        pulls = self.sun_masses_ / dists_sq

        assignments = np.argmax(pulls, axis=1)
        if return_pull:
            return assignments, np.max(pulls, axis=1)
        return assignments

    def predict(self, X_new):
        """Predict with the expert of each row's Sun (hard routing).

        Rows routed to a Sun that received no training point, and (with
        ``assignment='pull'``) rows that no Sun pulls, are predicted by the
        global fallback model, not by 0.0.
        """
        X_new = np.asarray(X_new, dtype=float)
        assignments = self._route(X_new)
        predictions = self.global_model_.predict(X_new)

        for i, (kind, local) in enumerate(self.local_models_):
            mask = (assignments == i)
            if not np.any(mask):
                continue
            if kind == 'fallback':
                continue                      # already holds the global prediction
            elif kind == 'constant':
                predictions[mask] = local
            else:
                predictions[mask] = local.predict(X_new[mask])

        return predictions

    def predict_blended(self, X_new, top_k=3):
        """Soft-gated Mixture of Experts: blend the predictions of the ``top_k``
        strongest-pulling Suns, weighted by their normalised pull.

        Experimental. In the thesis, blending linear experts showed destructive
        interference, so ``predict`` (hard routing) is the reported method.
        """
        # 1. Calculate pulls (Mass / Distance^2)
        sq_X = np.sum(X_new ** 2, axis=1)[:, np.newaxis]
        sq_S = np.sum(self.suns_ ** 2, axis=1)[np.newaxis, :]
        dot_XS = np.dot(X_new, self.suns_.T)
        dists_sq = np.maximum(sq_X + sq_S - 2 * dot_XS, 1e-9)
        pulls = self.sun_masses_ / dists_sq  # Shape: (n_samples, n_suns)

        final_predictions = np.zeros(len(X_new))

        # 2. For each sample, find the top K pulling suns
        # (top K instead of ALL suns so distant noise cannot ruin the blend)
        top_k_indices = np.argsort(pulls, axis=1)[:, -top_k:]

        for i in range(len(X_new)):
            sample_x = X_new[i].reshape(1, -1)
            sample_pulls = pulls[i, top_k_indices[i]]

            # Normalise the pulls so they sum to 1.0
            weights = sample_pulls / np.sum(sample_pulls)

            blended_pred = 0.0
            for w, sun_idx in zip(weights, top_k_indices[i]):
                kind, local_model = self.local_models_[sun_idx]
                if kind == 'fallback':
                    pred = self.global_model_.predict(sample_x)[0]
                elif kind == 'constant':
                    pred = local_model
                else:
                    pred = local_model.predict(sample_x)[0]

                blended_pred += w * pred

            final_predictions[i] = blended_pred

        return final_predictions
