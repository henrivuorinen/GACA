"""The GACA estimator: Solar Genesis on a coreset, Particle Accretion for every
row, and an optional Mixture of Experts (one local model per Sun)."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, LogisticRegression

from .genesis import solar_genesis


class GACA(BaseEstimator):
    """Gravitational Accretion Clustering Algorithm.

    ``fit(X)`` draws a uniform coreset of ``sample_size`` rows, runs Solar Genesis
    on it and stores the Suns (``suns_``, ``sun_masses_``). ``assign(X)`` routes
    any number of rows to those Suns. If ``y`` is passed to ``fit``, one local
    expert (Ridge, or logistic regression for ``task_type='classification'``) is
    trained per Sun and ``predict`` routes each row to its Sun's expert.

    Input is expected to be standardised and low-dimensional (roughly d <= 10;
    project with PCA first if needed, see the README).
    """

    def __init__(self, gamma_clustering=1.0, n_iterations=20,
                 theta=0.5, epsilon=0.05, sample_size=5000,
                 task_type='regression', random_state=None, eta=0.5,
                 min_expert_size=10, verbose=False):
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

        # State variables
        self.suns_ = None
        self.sun_masses_ = None
        self.local_models_ = []
        self.global_model_ = None
        self.n_iters_ = None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def fit(self, X, y=None):
        """Solar Genesis on a coreset, then (if ``y`` is given) local experts."""
        n_samples = X.shape[0]

        # 1. Subsampling (coreset extraction)
        rng = np.random.default_rng(self.random_state)
        if n_samples > self.sample_size:
            idx = rng.choice(n_samples, self.sample_size, replace=False)
            X_core = X[idx]
        else:
            X_core = X

        self._log(f"Fitting GACA: running Solar Genesis on {len(X_core)} points...")

        # 2. Barnes-Hut gravity with merging: the Suns and their accumulated masses
        self.suns_, self.sun_masses_, self.n_iters_ = solar_genesis(
            X_core,
            gamma=self.gamma_clustering,
            n_iterations=self.n_iterations,
            theta=self.theta,
            epsilon=self.epsilon,
            eta=self.eta,
            return_iters=True,
            verbose=self.verbose,
        )

        self._log(f"Solar Genesis complete: {len(self.suns_)} Suns identified "
                  f"in {self.n_iters_} iterations.")

        # 3. Local expert models. The WHOLE training set (not just the coreset)
        # is assigned to the Suns to train them.
        if y is not None:
            self._log("Training local expert models...")
            assignments = self._assign_to_suns(X)
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

        Uses the Newtonian pull M_k / d^2. Rows can be passed in batches of any
        size, so arbitrarily large data can be streamed through a fitted model.
        With ``return_pull`` the winning pull strength is returned as well.
        """
        return self._assign_to_suns(X, return_pull=return_pull)

    def fit_assign(self, X):
        """Fit on X and return the Sun label of every row."""
        return self.fit(X).assign(X)

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

        Rows routed to a Sun that received no training point are predicted by the
        global fallback model, not by 0.0.
        """
        assignments = self._assign_to_suns(X_new)
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
