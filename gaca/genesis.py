"""Solar Genesis: the tree-accelerated, mass-conserving blurring mean-shift.

This module holds the expensive half of GACA, the particle simulation that is
run once on a coreset and returns the Suns (positions and masses):

    GACANode                    Barnes-Hut spatial tree over the active particles
    merge_connected_components  mass-conserving condensation C_epsilon
    solar_genesis               the full iteration Psi = C_epsilon o B_{gamma,eta}
"""
import numpy as np
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components


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


def merge_connected_components(points, masses, epsilon):
    """Mass-conserving condensation (thesis section 3.3, C_epsilon).

    Points within ``epsilon`` of one another form the edges of a graph; each
    connected component is replaced by its mass-weighted centre, carrying the
    component's total mass. Unlike a single-pass greedy absorption this merges
    chains transitively, realising the exact connected-component partition of the
    epsilon-neighbourhood graph, so the operation is order-independent and
    conserves total mass.
    """
    n = len(points)
    if n <= 1:
        return points, masses

    graph = radius_neighbors_graph(points, radius=epsilon, mode='connectivity',
                                   include_self=True)
    n_comp, labels = connected_components(graph, directed=False)

    new_points = np.zeros((n_comp, points.shape[1]))
    new_masses = np.zeros(n_comp)
    for c in range(n_comp):
        mask = labels == c
        m = masses[mask]
        new_masses[c] = m.sum()
        new_points[c] = np.average(points[mask], axis=0, weights=m)
    return new_points, new_masses


def solar_genesis(X, gamma, n_iterations, theta=0.5, epsilon=0.05, eta=0.5,
                  leaf_size=1, delta_mac=1e-9, tol=1e-7, return_iters=False,
                  history=None, snapshots=None, verbose=False):
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

    Returns
    -------
    suns : array of shape (k, d)
    masses : array of shape (k,)
        Total mass carried by each Sun; sums to ``len(X)``.
    iters : int
        Only when ``return_iters`` is True.
    """
    current_pos = np.copy(X).astype(float)
    current_mass = np.ones(len(X))

    if snapshots is not None:          # S_0 = {(z_i, 1)}, before any transport
        snapshots.append((current_pos.copy(), current_mass.copy()))

    iters = 0
    for k in range(n_iterations):
        iters = k + 1
        n = len(current_pos)

        # 1. Build the spatial tree over the current active set
        root = GACANode(current_pos, current_mass, np.arange(n), leaf_size)

        # 2. Damped blurring mean-shift transport (B_{gamma,eta})
        new_pos = np.zeros_like(current_pos)
        for i in range(n):
            num, den = _kernel_sum(current_pos[i], i, root, current_pos, current_mass,
                                   gamma, theta, delta_mac)
            target = num / den if den > 0 else current_pos[i]
            new_pos[i] = (1.0 - eta) * current_pos[i] + eta * target

        movement = float(np.mean(np.linalg.norm(new_pos - current_pos, axis=1)))

        # 3. Mass-conserving condensation (C_epsilon)
        current_pos, current_mass = merge_connected_components(new_pos, current_mass, epsilon)

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

    if return_iters:
        return current_pos, current_mass, iters
    return current_pos, current_mass


def assign(X, suns, masses):
    """Particle Accretion: route each row of X to the Sun with the strongest
    Newtonian pull M_k / d^2 (vectorised, O(N K D)). Safe to call batch by batch."""
    d2 = np.maximum(np.sum(X ** 2, 1)[:, None] + np.sum(suns ** 2, 1)[None, :]
                    - 2 * X @ suns.T, 1e-9)
    return np.argmax(masses / d2, axis=1)
