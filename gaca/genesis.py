"""Solar Genesis: the tree-accelerated, mass-conserving blurring mean-shift.

This module holds the expensive half of GACA, the particle simulation that is
run once on a coreset and returns the Suns (positions and masses):

    GACANode                    Barnes-Hut spatial tree over the active particles
    merge_connected_components  mass-conserving condensation C_epsilon
    solar_genesis               the full iteration Psi = C_epsilon o B_{gamma,eta}
    kernel_pull                 Gaussian pull sum_j w_j exp(-gamma ||x - z_j||^2)
    saddle_link                 joins Suns connected by a high-density bridge
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components


def _sqdist(A, B):
    """Pairwise squared Euclidean distances, clipped at zero."""
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :]
                      - 2.0 * A @ B.T, 0.0)


def kernel_pull(Q, Z, w, gamma, block=2048):
    """Exact Gaussian pull of the weighted particles (Z, w) at each row of Q:
    sum_j w_j exp(-gamma ||q - z_j||^2). Evaluated in row blocks."""
    out = np.empty(len(Q))
    for s in range(0, len(Q), block):
        out[s:s + block] = np.exp(-gamma * _sqdist(Q[s:s + block], Z)) @ w
    return out


def _transport_exact(pos, mass, gamma, eta, block=2048):
    """Exact damped blurring mean-shift step B_{gamma,eta}, vectorised in blocks.

    For the low-dimensional coresets GACA runs on, a dense blocked kernel sum is
    both exact and far faster than the interpreted Barnes-Hut traversal.
    """
    new = np.empty_like(pos)
    for s in range(0, len(pos), block):
        P = pos[s:s + block]
        W = mass[None, :] * np.exp(-gamma * _sqdist(P, pos))
        new[s:s + block] = (1.0 - eta) * P + eta * (W @ pos) / W.sum(1, keepdims=True)
    return new


class GACANode:
    """Latent-space spatial tree node (thesis Algorithm 2).

    Each node stores a mass summary of the points it bounds: the global indices of
    those points, the total mass, the mass-weighted centre of mass, and the node
    radius (the maximum distance from any contained point to the centre of mass).
    The radius is the size proxy used by the Multipole Acceptance Criterion in the
    kernel sum. Internal nodes are split on the maximum-spread dimension via a
    median partition; a node becomes a leaf when it holds at most ``leaf_size``
    points or has zero radius (all points coincident).
    """

    def __init__(self, points, masses, indices, leaf_size=1):
        self.indices = indices
        self.index_set = set(int(i) for i in indices)  # O(1) membership for the MAC guard

        pts = points[indices]
        ms = masses[indices]
        self.mass = float(ms.sum())
        self.com = np.average(pts, axis=0, weights=ms)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        self.radius = float(np.max(np.linalg.norm(pts - self.com, axis=1)))

        self.left = None
        self.right = None
        if len(indices) > leaf_size and self.radius > 0.0:
            r = int(np.argmax(maxs - mins))                    # maximum-spread dimension
            order = indices[np.argsort(points[indices, r])]    # median partition on r
            mid = len(order) // 2
            self.left = GACANode(points, masses, order[:mid], leaf_size)
            self.right = GACANode(points, masses, order[mid:], leaf_size)

    def is_leaf(self):
        return self.left is None and self.right is None


def _kernel_sum(p, q, node, points, masses, gamma, theta, delta_mac):
    """Tree-accelerated Gaussian kernel summation (thesis Algorithm 3).

    Returns the weighted numerator (sum_j w_j z_j) and scalar denominator
    (sum_j w_j) with w_j = m_j exp(-gamma ||p - z_j||^2). Their ratio is the
    approximate mean-shift target for query point p (with global index q).
    """
    if node is None:
        return np.zeros_like(p), 0.0

    if node.is_leaf():
        num = np.zeros_like(p)
        den = 0.0
        for j in node.indices:
            r2 = np.sum((p - points[j]) ** 2)
            w = masses[j] * np.exp(-gamma * r2)
            num = num + w * points[j]
            den += w
        return num, den

    dist = np.sqrt(np.sum((p - node.com) ** 2))
    # Multipole Acceptance Criterion: a distant node may be summarised by its
    # centre of mass, but a node containing the query itself is never accepted as
    # an aggregate -- it must be descended to the leaf, where the self term
    # (z_q - z_q = 0) contributes to the denominator without creating displacement.
    accept = (q not in node.index_set) and (node.radius / max(dist, delta_mac) < theta)
    if accept:
        w = node.mass * np.exp(-gamma * dist ** 2)
        return w * node.com, w

    num_l, den_l = _kernel_sum(p, q, node.left, points, masses, gamma, theta, delta_mac)
    num_r, den_r = _kernel_sum(p, q, node.right, points, masses, gamma, theta, delta_mac)
    return num_l + num_r, den_l + den_r


def _condense(points, masses, epsilon):
    """C_epsilon with the component label of every input particle."""
    n = len(points)
    if n <= 1:
        return points, masses, np.zeros(n, dtype=int)

    graph = radius_neighbors_graph(points, radius=epsilon, mode='connectivity',
                                   include_self=True)
    n_comp, labels = connected_components(graph, directed=False)

    new_masses = np.bincount(labels, weights=masses, minlength=n_comp)
    new_points = np.column_stack([
        np.bincount(labels, weights=masses * points[:, j], minlength=n_comp)
        for j in range(points.shape[1])]) / new_masses[:, None]
    return new_points, new_masses, labels


def merge_connected_components(points, masses, epsilon):
    """Mass-conserving condensation (thesis section 3.3, C_epsilon).

    Points within ``epsilon`` of one another form the edges of a graph; each
    connected component is replaced by its mass-weighted centre, carrying the
    component's total mass. Unlike a single-pass greedy absorption this merges
    chains transitively, realising the exact connected-component partition of the
    epsilon-neighbourhood graph, so the operation is order-independent and
    conserves total mass.
    """
    new_points, new_masses, _ = _condense(points, masses, epsilon)
    return new_points, new_masses


def solar_genesis(X, gamma, n_iterations, theta=0.5, epsilon=0.05, eta=0.5,
                  leaf_size=1, delta_mac=1e-9, tol=1e-7, return_iters=False,
                  history=None, snapshots=None, verbose=False, method='exact',
                  masses=None, plateau=None, return_members=False):
    """Damped, mass-conserving, tree-accelerated blurring mean-shift: one step is
    the operator Psi = C_epsilon o B_{gamma,eta} of thesis Chapter 3.

    Each iteration applies the two components in the order the thesis defines
    them: (i) build a GACANode tree over the active set, (ii) apply the damped
    blurring mean-shift transport u = (1-eta) z + eta * T(z) using the
    tree-accelerated kernel sum (this is B), and (iii) condense the transported
    particles into the connected components of the epsilon-neighbourhood graph,
    replacing each by its mass-weighted centre (this is C).

    Applying C last matters: the returned set is always post-condensation, so no
    two returned Suns are within ``epsilon`` of one another. It also means the raw
    input is never silently deduplicated before the dynamics start, so S_0 is
    exactly {(z_i, 1)} as Chapter 3 states.

    Parameters
    ----------
    X : array of shape (n, d)
        Points to condense, normally a standardised, low-dimensional coreset.
    gamma : float
        Gaussian bandwidth. Controls how many Suns survive.
    n_iterations : int
        Iteration cap.
    theta : float
        Barnes-Hut opening angle. Larger is faster and less exact.
    epsilon : float
        Condensation radius: particles closer than this merge.
    eta : float in (0, 1]
        Damping of the transport step (eqs. 3.9-3.11); eta = 1 is the fully
        overdamped step. The thesis uses 0.5.
    leaf_size, delta_mac : int, float
        GACANode leaf size and the MAC distance floor.
    tol : float
        Mean-movement convergence threshold.
    return_iters : bool
        Also return the number of iterations actually executed.
    history : list, optional
        If given, one record per iteration is appended with the active-particle
        count before and after condensation and the mean transport movement.
    snapshots : list, optional
        If given, the initial system and one ``(positions, masses)`` pair after
        each condensation are appended (as copies).
    verbose : bool
        Print one progress line per iteration.
    method : {'exact', 'barnes_hut'}
        'exact' (default) evaluates the Gaussian sum densely in vectorised blocks:
        no approximation, and much faster than the interpreted tree for the
        low-dimensional coresets GACA is meant for. 'barnes_hut' is the
        tree-accelerated operator of the thesis (``theta``, ``leaf_size`` and
        ``delta_mac`` only apply to it).
    masses : array of shape (n,), optional
        Initial particle masses (default 1 each), e.g. importance weights.
    plateau : int, optional
        Also stop once the number of active particles has not changed for this
        many consecutive iterations (the plateau rule of thesis Sec. 5.6).
    return_members : bool
        Also return, for every input row, the index of the Sun it condensed into.

    Returns
    -------
    suns : array of shape (k, d)
    masses : array of shape (k,)
        Total mass carried by each Sun; sums to ``len(X)``.
    iters : int
        Only when ``return_iters`` is True.
    members : array of shape (n,)
        Only when ``return_members`` is True.
    """
    if method not in ('exact', 'barnes_hut'):
        raise ValueError(f"method must be 'exact' or 'barnes_hut', got {method!r}")
    current_pos = np.copy(X).astype(float)
    current_mass = (np.ones(len(X)) if masses is None
                    else np.asarray(masses, dtype=float).copy())
    members = np.arange(len(X))
    unchanged = 0

    if snapshots is not None:          # S_0 = {(z_i, 1)}, before any transport
        snapshots.append((current_pos.copy(), current_mass.copy()))

    iters = 0
    for k in range(n_iterations):
        iters = k + 1
        n = len(current_pos)

        # 1-2. Damped blurring mean-shift transport (B_{gamma,eta})
        if method == 'exact':
            new_pos = _transport_exact(current_pos, current_mass, gamma, eta)
        else:
            # Build the spatial tree over the current active set
            root = GACANode(current_pos, current_mass, np.arange(n), leaf_size)
            new_pos = np.zeros_like(current_pos)
            for i in range(n):
                num, den = _kernel_sum(current_pos[i], i, root, current_pos,
                                       current_mass, gamma, theta, delta_mac)
                target = num / den if den > 0 else current_pos[i]
                new_pos[i] = (1.0 - eta) * current_pos[i] + eta * target

        movement = float(np.mean(np.linalg.norm(new_pos - current_pos, axis=1)))

        # 3. Mass-conserving condensation (C_epsilon)
        current_pos, current_mass, labels = _condense(new_pos, current_mass, epsilon)
        members = labels[members]
        unchanged = unchanged + 1 if len(current_pos) == n else 0

        if history is not None:
            history.append(dict(iteration=k + 1, n_before=n,
                                n_after=len(current_pos), movement=movement))
        if snapshots is not None:
            snapshots.append((current_pos.copy(), current_mass.copy()))

        if verbose:
            print(f"Iter {k + 1} | Points: {n} -> {len(current_pos)} | "
                  f"Avg Movement: {movement:.6f}")

        if movement < tol or len(current_pos) <= 1:
            break
        if plateau is not None and unchanged >= plateau:
            break

    out = (current_pos, current_mass)
    if return_iters:
        out += (iters,)
    if return_members:
        out += (members,)
    return out


def assign(X, suns, masses):
    """Particle Accretion: route each row of X to the Sun with the strongest
    Newtonian pull M_k / d^2 (vectorised, O(N K D)). Safe to call batch by batch."""
    d2 = np.maximum(np.sum(X ** 2, 1)[:, None] + np.sum(suns ** 2, 1)[None, :]
                    - 2 * X @ suns.T, 1e-9)
    return np.argmax(masses / d2, axis=1)


def saddle_link(Z, members, w, gamma, tau=0.5, kappa=0.0, n_neighbors=10,
                pull=None, protect=None):
    """Join Suns whose coreset members are connected by a high-density bridge.

    The flat output of Solar Genesis can only express Voronoi-like cells, and a
    high bandwidth splits a curved or elongated cluster into a chain of Suns.
    This follows the thesis's suggestion (Sec. 7.3) to decide on joining two Suns
    by the mass lying between them rather than the distance separating them.

    The density at each coreset member is its external pull (self excluded).
    For every k-nearest-neighbour edge whose ends belong to different Suns, the
    bridge height is the lowest density along the edge (ends, and three
    interior points). Edges are processed from the highest bridge down, and
    two groups are joined when the bridge reaches ``tau`` times the lower of
    their two peak densities (a persistence ratio, as in ToMATo) and is at least
    ``kappa``. Lone Suns, whose external pull is below ``kappa``, are never
    joined, and neither are Suns flagged in ``protect`` (a boolean array over
    Suns): a small group has a low peak, so almost any bridge would pass the
    ratio test and absorb it.

    Returns the new label of each original Sun (an array of length
    ``members.max() + 1``) and the external pull at every member.
    """
    if pull is None:
        pull = kernel_pull(Z, Z, w, gamma) - w
    n_suns = int(members.max()) + 1
    peak = np.zeros(n_suns)
    np.maximum.at(peak, members, pull)

    k = min(n_neighbors + 1, len(Z))
    _, nb = cKDTree(Z).query(Z, k=k)
    i = np.repeat(np.arange(len(Z)), k - 1)
    j = nb[:, 1:].ravel()
    keep = (members[i] != members[j]) & (i < j)
    if protect is not None:
        keep &= ~protect[members[i]] & ~protect[members[j]]
    i, j = i[keep], j[keep]
    bridge = np.minimum(pull[i], pull[j])
    for t in (0.25, 0.5, 0.75):
        bridge = np.minimum(bridge, kernel_pull(Z[i] + t * (Z[j] - Z[i]), Z, w, gamma))

    parent = np.arange(n_suns)
    top = peak.copy()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in np.argsort(-bridge):
        a, b = find(members[i[e]]), find(members[j[e]])
        if a != b and bridge[e] >= kappa and bridge[e] >= tau * min(top[a], top[b]):
            parent[b] = a
            top[a] = max(top[a], top[b])

    roots = np.array([find(a) for a in range(n_suns)])
    _, new_label = np.unique(roots, return_inverse=True)
    return new_label, pull


def critical_gap(n_iterations=20, eta=0.5, spread=1.0):
    """Smallest gap, in kernel widths h, at which two equal Gaussian clusters of
    spread ``spread`` (also in units of h) survive ``n_iterations`` steps as
    separate Suns. Uses the mean-field recursion of docs/theory.md section 1.5,
    with "the gap halves" as the merge criterion. At ``spread=0`` it is the
    equal-mass two-Sun map of section 1.1."""
    def final_gap(gap):
        s, g = spread, gap
        for _ in range(n_iterations):
            v = s * s + 1.0
            r = np.exp(-g * g / (2.0 * v))
            g *= 1.0 - 2.0 * eta * r / (1.0 + r) / v
            s *= 1.0 - eta / v
        return g

    lo, hi = 0.1, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if final_gap(mid) < 0.5 * mid else (lo, mid)
    return 0.5 * (lo + hi)


def resolution_to_gamma(resolution, n_iterations=20, eta=0.5):
    """The gamma at which groups at least ``resolution`` apart stay separate.

    The kernel width is h = resolution / critical_gap(...), computed for groups
    as wide as the kernel (spread = h). That is conservative for compact groups
    (for point masses the proved critical gap is smaller). Returns
    gamma = 1 / (2 h^2). For the defaults, h is about resolution / 3.3."""
    h = float(resolution) / critical_gap(n_iterations, eta, spread=1.0)
    return 1.0 / (2.0 * h * h)
