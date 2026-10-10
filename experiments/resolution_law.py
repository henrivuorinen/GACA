"""
The resolution law of Solar Genesis: how far apart two clusters must be to
survive as separate Suns. See docs/theory.md, section 1.

Part 1 (proved results, docs/theory.md sections 1.1 to 1.4): for two Suns
(point masses a, b), iterates the exact one-step gap map and checks that the
critical gap lies between the proven lower and upper bounds, for equal and
unequal masses.

Part 2 (approximation, section 1.5) compares three things for two equal
Gaussian clusters (spread sigma, gap Delta) under the damped blurring dynamics
with kernel width h (gamma = 1 / (2 h^2)):

  simulation   the actual solar_genesis, bisecting on Delta
  mean field   the two-line recursion for (sigma_t, Delta_t)
  closed form  the compact-cluster limit, Delta* = h sqrt(2 u*) with
               Ei(u*) - Ei(u*/4) = 4 eta T, and its approximation
               h sqrt(2 ln(4 eta T ln(4 eta T)))

    python experiments/resolution_law.py          # about 2 minutes
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expi

from gaca import solar_genesis

ETA = 0.5


# ----------------------------------------------------------------- two Suns

def kappa(r, a, b):
    """Theorem 1: one step shrinks the gap by the factor 1 - eta * kappa(r)."""
    return r * b / (a + b * r) + r * a / (b + a * r)


def merge_steps(g0, a, b, eps, eta=ETA, h=1.0, cap=10 ** 7):
    """Iterations until two isolated Suns are within eps (exact map)."""
    g, t = g0, 0
    while g >= eps and t < cap:
        g *= 1 - eta * kappa(np.exp(-g * g / (2 * h * h)), a, b)
        t += 1
    return t


def steps_upper_bound(g0, a, b, eps, eta=ETA):
    """Theorem 3 (i)."""
    k0 = eta * kappa(np.exp(-g0 * g0 / 2), a, b)
    return np.ceil(np.log(g0 / eps) / -np.log1p(-k0))


def steps_lower_bound(g0, a, b, eps, eta=ETA):
    """Theorem 3 (ii), maximised over the intermediate level m."""
    m = np.linspace(eps * 1.01, g0 * 0.999, 4000)
    km = eta * kappa(np.exp(-m * m / 2), a, b)
    return np.max(np.log(g0 / m) / -np.log1p(-km))


def two_sun_table(eps=0.01):
    print("Two Suns: critical gap Delta*(T)/h (merge takes exactly T steps), eta = 0.5, "
          f"eps = {eps} h\n")
    print(f"  {'masses':>8} {'T':>7} {'proven lower':>13} {'exact map':>10} "
          f"{'proven upper':>13} {'sqrt(2 ln T)':>13}")
    for a, b in [(1, 1), (10, 1), (1000, 1)]:
        for T in (20, 100, 1000, 10 ** 5):
            lower = brentq(lambda g: steps_upper_bound(g, a, b, eps) - T - 0.5, 0.3, 30.0)
            upper = brentq(lambda g: steps_lower_bound(g, a, b, eps) - T, 0.3, 30.0)
            lo, hi = 0.3, 30.0
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if merge_steps(mid, a, b, eps) <= T else (lo, mid)
            print(f"  {f'{a}:{b}':>8} {T:>7} {lower:13.2f} {lo:10.2f} {upper:13.2f} "
                  f"{np.sqrt(2 * np.log(T)):13.2f}")
    print()


# ------------------------------------------------------------ extended clusters

def separated(delta, sigma, T, h=1.0, eta=ETA, n=600, d=2, seed=0):
    """Do two Gaussian clusters at gap delta end as two heavy Suns?"""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, sigma, (n, d))
    B = rng.normal(0, sigma, (n, d))
    B[:, 0] += delta
    gamma = 1 / (2 * h * h)
    suns, m = solar_genesis(np.vstack([A, B]), gamma, T, epsilon=0.01 * h, eta=eta, tol=0)
    big = np.sort(m)[::-1]
    return len(big) >= 2 and big[1] >= 0.3 * 2 * n


def simulated_threshold(sigma, T, seeds=(0, 1)):
    lo, hi = 0.5, 8.0
    for _ in range(12):
        mid = (lo + hi) / 2
        ok = np.mean([separated(mid, sigma, T, seed=s) for s in seeds]) >= 0.5
        hi, lo = (mid, lo) if ok else (hi, mid)
    return (lo + hi) / 2


def mean_field_threshold(sigma, T, h=1.0, eta=ETA):
    """Smallest gap that is not halved within T steps of the recursion."""
    def final_gap(delta):
        s, g = sigma, delta
        for _ in range(T):
            v = s * s + h * h
            r = np.exp(-g * g / (2 * v))
            g *= 1 - 2 * eta * r / (1 + r) * h * h / v
            s *= 1 - eta * h * h / v
        return g
    return brentq(lambda d: final_gap(d) - d / 2, 0.1, 20.0)


def closed_form(T, eta=ETA):
    u = brentq(lambda u: expi(u) - expi(u / 4) - 4 * eta * T, 0.5, 50)
    approx = np.sqrt(2 * np.log(4 * eta * T * np.log(4 * eta * T)))
    return np.sqrt(2 * u), approx


def main():
    two_sun_table()
    print("Two Gaussian clusters: critical gap Delta*/h (eta = 0.5, 2-D)\n")
    for T in (20, 100):
        exact, approx = closed_form(T)
        print(f"T = {T} iterations   (compact limit: exact {exact:.2f}, approximation {approx:.2f}; "
              f"density dip needs 2 sqrt(sigma^2 + h^2))")
        print(f"  {'sigma/h':>8} {'simulation':>11} {'mean field':>11} {'density dip':>12}")
        for sigma in (0.25, 0.5, 1.0, 1.5, 2.0):
            print(f"  {sigma:8.2f} {simulated_threshold(sigma, T):11.2f} "
                  f"{mean_field_threshold(sigma, T):11.2f} {2 * np.sqrt(sigma ** 2 + 1):12.2f}",
                  flush=True)
        print()


if __name__ == "__main__":
    main()
