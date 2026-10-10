# Notes on the mathematics

This document records the mathematics behind GACA's behaviour and behind the
changes made after the thesis. Section 1 is a new result, a resolution law for
the dynamics. Its two-Sun part is proved; its extension to clusters with a
spread is an approximation verified by simulation. The other sections give the
reasoning behind the design choices in the code. Each section says what is
proved, what is approximated, and what rests only on experiments.

Notation follows the thesis: the Gaussian kernel is exp(−γ‖x − z‖²), with
kernel width h = 1/√(2γ); η is the damping; T is the number of iterations; ε
is the condensation radius.

---

## 1. The resolution law: when do two clusters stay apart?

This section has two parts with different status. Sections 1.1 to 1.4 are
**proved**: they concern two Suns, the point masses that clusters become after
condensation, whose dynamics are exact. Section 1.5 extends the result to
clusters that still have a spread, using an **approximation** that is checked
numerically but **not proved**.

Throughout, r(g) = exp(−g²/2h²) = exp(−γg²) is the kernel weight at distance
g, and the dynamics are the exact (not Barnes-Hut) operator of thesis Chapter
3: damped transport, then condensation.

### 1.1 The exact two-Sun map

**Theorem 1.** Let two Suns of masses a, b > 0 be at positions x, y with gap
g = ‖y − x‖, and assume no other particle exerts a pull (see Remark 1.6). One
damped transport step with η ∈ (0, ½] moves both along the segment between
them, and the new gap is

$$
g' = g\,\big(1 - \eta\,\kappa(r)\big), \qquad
\kappa(r) = \frac{r\,b}{a + b\,r} + \frac{r\,a}{b + a\,r}, \qquad r = r(g).
$$

The two merge in the condensation step if and only if g′ < ε.

*Proof.* The transport target of x is the kernel-weighted mean of both Suns,
in which x has kernel weight 1 and y has weight r:
T(x) = (a·x + b·r·y)/(a + b·r). So T(x) − x = (b·r/(a + b·r))(y − x), and
x′ = x + η(T(x) − x) moves towards y by the fraction ηbr/(a + br) of the gap.
By symmetry y moves towards x by the fraction ηar/(b + ar). Hence
y′ − x′ = (y − x)(1 − ηκ(r)). Each of the two terms of κ is below 1, so κ < 2
and ηκ < 1 for η ≤ ½. The Suns therefore do not cross, and the gap is
g(1 − ηκ). Condensation merges exactly the pairs closer than ε. ∎

(For equal masses κ = 2r/(1 + r) ≤ 1, so the theorem holds for all η ∈ (0, 1].)

### 1.2 Two isolated Suns always merge, and faster and faster

**Theorem 2.** Under the assumptions of Theorem 1, the gaps g_t are strictly
decreasing, the per-step shrink factor 1 − ηκ(r(g_t)) is strictly decreasing
as well, and g_t falls below ε after finitely many steps.

*Proof.* ∂/∂r [rb/(a + br)] = ab/(a + br)² > 0, and likewise for the second
term, so κ is strictly increasing in r, while r(g) is strictly decreasing in
g. Since κ > 0, g₁ < g₀. Inductively, g_{t+1} < g_t implies r_{t+1} > r_t,
then κ_{t+1} > κ_t, and the next factor is smaller. Finally
g_t ≤ g₀(1 − ηκ(r(g₀)))ᵗ → 0. ∎

This is the two-body case of the global coalescence of blurring mean-shift
that thesis Sec. 3.8 refers to: no separation survives forever, so the
question is only how long it survives.

### 1.3 How long the separation survives

Let N(g₀) be the number of steps until two Suns starting at gap g₀ merge.

**Theorem 3.** Write q(g) = −ln(1 − ηκ(r(g))). Then

$$
\text{(i)}\quad N(g_0) \le \left\lceil \frac{\ln(g_0/\varepsilon)}{q(g_0)} \right\rceil,
\qquad
\text{(ii)}\quad N(g_0) \ge \max_{\varepsilon < m < g_0} \frac{\ln(g_0/m)}{q(m)} .
$$

*Proof.* (i) By Theorem 2, every factor is at most the first one, so
g_t ≤ g₀·e^{−t·q(g₀)}, which is below ε once t ≥ ln(g₀/ε)/q(g₀).
(ii) Fix m ∈ (ε, g₀). As long as g_s ≥ m, κ(r(g_s)) ≤ κ(r(m)) (κ increases as
the gap shrinks), so g_t ≥ g₀·e^{−t·q(m)}. To merge, the gap must first drop
below m > ε, which takes at least ln(g₀/m)/q(m) steps. ∎

N is non-decreasing in g₀. The map g ↦ g(1 − ηκ(r(g))) is increasing, as a
product of two positive increasing factors, so a pair that starts further
apart is further apart at every step. For a run of T iterations, define the
**critical gap** Δ\*(T) as the smallest starting gap that survives T steps. Pairs further apart survive the
run. Theorem 3 brackets Δ\*(T). The table compares the brackets with the exact
map, iterated numerically (η = ½, ε = 0.01 h, units of h):

| masses a : b | T | proven lower bound | exact map | proven upper bound | √(2 ln T) |
|---|---|---|---|---|---|
| 1 : 1 | 20 | 1.58 | 2.88 | 3.60 | 2.45 |
| 1 : 1 | 100 | 2.40 | 3.72 | 4.09 | 3.03 |
| 1 : 1 | 1000 | 3.21 | 4.41 | 4.68 | 3.72 |
| 1 : 1 | 10⁵ | 4.41 | 5.45 | 5.65 | 4.80 |
| 10 : 1 | 20 | 2.21 | 3.51 | 4.09 | 2.45 |
| 10 : 1 | 1000 | 3.67 | 4.81 | 5.04 | 3.72 |
| 1000 : 1 | 20 | 3.70 | 4.78 | 5.19 | 2.45 |
| 1000 : 1 | 1000 | 4.75 | 5.75 | 5.94 | 3.72 |

### 1.4 The √(ln T) law

**Corollary 4.** For fixed masses, η and ε,

$$
\Delta^*(T) = h\sqrt{2\ln T}\;\big(1 + o(1)\big) \qquad (T \to \infty).
$$

*Proof.* Let c = min(a, b)/(a + b) > 0. For r ≤ 1, each term of κ is at least
c·r, and each is at most r·max(a, b)/min(a, b) =: C·r. We also use
x ≤ −ln(1 − x) ≤ x/(1 − x).

*Lower bound.* By Theorem 3(i), with q(g₀) ≥ ηκ ≥ 2ηc·r(g₀), merging within
T steps is guaranteed when ln(g₀/ε) ≤ (T − 1)·2ηc·r(g₀), that is, when
g₀² ≤ 2h² ln(2ηc(T − 1) / ln(g₀/ε)). For g₀ = O(√(ln T)), ln(g₀/ε) = O(ln ln T), so
Δ\*(T)² ≥ 2h² ln T − O(ln ln T).

*Upper bound.* In Theorem 3(ii) take m = g₀(1 − δ) with δ = 1/ln T. Then
ln(g₀/m) ≥ δ and q(m) ≤ 2ηC·r(m)/(1 − 2ηC·r(m)). Survival beyond T steps is
guaranteed when δ(1 − 2ηC·r(m)) > T·2ηC·r(m), which holds when
r(m) ≤ δ/(2ηC(T + δ)), that is, when m² ≥ 2h² ln(2ηC·T·ln T·(1 + o(1))).
With g₀ = m/(1 − δ): Δ\*(T)² ≤ 2h² ln T·(1 + O(ln ln T / ln T)).

Both bounds are 2h² ln T to leading order. ∎

In words: the dynamics resolve groups whose gap is more than about
h√(2 ln T). The number of iterations enters only through √(ln T). From
T = 20 to T = 200 the exact critical gap for equal Suns grows by only about
30%, so clusters merge rarely after the first iterations. This turns the
metastable plateaus of thesis Corollary 3.12 into a quantitative statement:
a pair at gap g survives until roughly T ≈ e^{g²/2h²}. The thesis's
convergence run (7 → 4 Suns at γ = 1 between 20 and 200 iterations, Sec. 5.6)
fits this picture.

**Unequal masses.** The table shows that a light Sun next to a heavy one is
absorbed from further away. By the same bounds, with mass ratio ρ = a/b ≫ 1,
the critical gap is approximately h√(2 ln(ηρT)). For example, a Lone Sun of
mass 1 next to a Sun of mass 1000 is absorbed within 20 iterations from
about 4.8 h, against 2.9 h between equal Suns. This makes precise why
anomalies close to a dense cluster are absorbed (thesis Sec. 6.4) while
distant ones survive.

**Remark (other particles).** Theorems 1 to 3 assume the pair feels no other
pull. By thesis Proposition 3.8, particles at distance at least D from both
Suns, with total mass M and within distance R, move each Sun by at most
η·M·R·e^{−D²/2h²}/m per step. The results hold up to this perturbation, which
is negligible once D exceeds the critical gap by a few h.

### 1.5 Clusters with a spread: an approximation (not proved)

Before condensation, clusters are clouds rather than points. For a Gaussian
cluster N(μ, σ²I) with many points, one step is exact:
σ_{t+1} = σ_t(1 − ηh²/(σ_t² + h²)). This is the known contraction of Gaussian
blurring mean-shift (Carreira-Perpiñán, 2006). For two clusters, *assume*
that each stays Gaussian and that the attraction can be evaluated at the
cluster centres. Theorem 1 with spreads gives the **mean-field recursion**

$$
\sigma_{t+1} = \sigma_t\Big(1 - \frac{\eta h^2}{\sigma_t^2 + h^2}\Big), \qquad
\Delta_{t+1} = \Delta_t\Big(1 - 2\eta\,\frac{r_t}{1 + r_t}\,\frac{h^2}{\sigma_t^2 + h^2}\Big), \qquad
r_t = e^{-\Delta_t^2 / 2(\sigma_t^2 + h^2)} .
$$

At σ = 0 it reduces to the exact equal-mass map of Theorem 1. The Gaussian
assumption is **not justified**: the transport of a mixture of Gaussians does
not keep the components Gaussian. The recursion is therefore a model, checked
against the real `solar_genesis` (merge criterion: the gap halves; units of h):

| σ/h | T = 20: simulation | model | T = 100: simulation | model | density dip, 2√(σ² + h²) |
|---|---|---|---|---|---|
| 0.25 | 2.89 | 3.15 | 3.70 | 3.76 | 2.06 |
| 0.50 | 2.98 | 3.18 | 3.71 | 3.76 | 2.24 |
| 1.00 | 3.25 | 3.31 | 3.88 | 3.83 | 2.83 |
| 1.50 | 3.42 | 3.56 | 4.08 | 4.00 | 3.61 |
| 2.00 | 3.30 | 3.87 | 4.28 | 4.27 | 4.47 |

- **Fit:** at T = 100 the model is within 0.08 h of the simulation. At T = 20
  it is up to 0.3 h conservative, and 0.6 h at σ = 2h.
- **Dimension:** the simulated threshold does not depend on dimension (σ =
  0.5h, T = 20: 2.98 h in 2-D, 2.95 h in 5-D, 2.99 h in 10-D).
- **Very wide clusters** (σ ≳ 2h at T = 20) do not contract within the run
  and break into fragments. Neither the model nor the theorems describe that
  regime.

**What it suggests.** A mixture of two equal Gaussians smoothed by the kernel
has a density dip between them only when Δ > 2√(σ² + h²). That is the
condition for mean-shift (non-blurring), and for any method that looks for
density valleys, to see two groups. The simulated GACA threshold grows much
more slowly with σ. From σ ≈ 1.5 h on it is *below* the density-dip
threshold: at σ = 2h the dynamics separate clusters 3.3 h apart, where the
density has no dip until 4.5 h. The mechanism is that each cluster contracts
before the pair merges, and a contracted pair is bimodal. This explains why
GACA separated overlapping Gaussian groups that HDBSCAN merged in the
benchmark, and why a density-dip test cannot reproduce GACA's main view
(section 8). The simulations establish this behaviour; a proof is open.

**Open problem.** Prove a version of Theorem 3 for clusters with a spread:
bound how far the transported cloud can be from Gaussian, or treat the
infinite-sample mixture directly. A proof would make the "dynamics see more
than the density" statement a theorem.

### 1.6 Practical consequences

- **Choosing γ from a resolution.** To separate groups at least Δ_min apart
  within the default 20 iterations, take h ≈ Δ_min/3.3, that is
  γ ≈ 5/Δ_min². `resolution_to_gamma` (and `AutoGACA(resolution=…)`,
  `gaca run --resolution`) computes h = Δ_min/critical_gap(T, η, spread = 1)
  from the mean-field recursion. The conversion assumes groups as wide as the
  kernel, which is conservative for compact groups. In a test with three
  groups at gap D, groups at D ≥ resolution always stayed separate, for
  spreads up to a third of the resolution.
- **Lone Suns next to big clusters.** These are absorbed within roughly
  h√(2 ln(ηρT)) for mass ratio ρ. Detecting anomalies near dense clusters
  therefore needs a smaller kernel than detecting distant ones, which is one
  reason the anomaly test of section 2 works on the pull instead.

---

## 2. The anomaly test for streamed rows

Thesis Proposition 3.8 bounds how far an isolated particle o moves in one step:
by η·M¬o·R¬o·e^{−γd²}/m_o. Equivalently, with P(x) = Σ_j w_j e^{−γ‖x − z_j‖²}
the *external pull* of the other particles, the step is at most
η·R·P/(m_o + P). A row x that was not sampled into the coreset is tested by
asking whether it would have moved had it been sampled, with mass 1 like every
coreset particle. If P(x) < κ, its displacement is a fraction κ/(1 + κ) of the
bound. It would have stayed put and become a Lone Sun, so it is registered as
one. Later rows that a registered Lone Sun pulls at least κ join it, so copies
of one anomaly form one group.

- **κ is relative.** The pull of a dense region grows like
  n_c·ρ·(π/γ)^{d/2}, so a fixed κ would mean something different for every γ,
  dimension and coreset size. The threshold is therefore κ_rel × the median
  external pull among coreset members, with κ_rel = 10⁻³ by default. A row is
  anomalous if it feels less than 0.1% of the pull a typical row feels.
- **The anomaly score** is −log₁₀(P(x)/median pull), computed in log space so
  that far rows keep a finite score that grows with distance. Only coreset
  members of regular clusters contribute. Otherwise the sampled copies of an
  anomaly group would pull each other and look ordinary.
- **Rare groups.** A dense little group forms a regular cluster, so its rows
  feel a strong pull from each other and would score as ordinary. That is the
  typical case for intrusion floods or rare machine states. In the score, a
  cluster holding a share s < s₀ of the rows therefore pulls with weight
  s/s₀ (s₀ = 5%, the conventional meaning of "rare", fixed before testing).
  A row in such a group scores roughly −log₁₀(s/s₀) higher, plus whatever its
  isolation from the bulk adds. The flags, that is which rows are declared
  anomalies, are unchanged.
  - On the development sets this raised the ROC AUC on KDD Cup attacks from
    0.41 to 0.79 and left the others within ±0.01.
  - On held-out shuttle data, average precision rose from 0.26 to 0.65 and
    AUC from 0.80 to 0.98.

---

## 3. The condensation radius, and why a fixed ε chains

At the first iteration, condensation merges every connected component of the
ε-neighbourhood graph, which is single-linkage clustering at height ε. Once ε
exceeds the typical nearest-neighbour spacing, this graph percolates:
components span whole dense structures and collapse each to its centre of
mass. For a ring, that centre lies in the empty middle, so two concentric
rings collapse onto the same point. This is what happened with the fixed
ε = 0.05 on dense 2-D data. The default is therefore

$$
\varepsilon = \min\!\left(0.05/\sqrt{\gamma},\; \tfrac12\,\mathrm{median\ nearest\text{-}neighbour\ distance}\right).
$$

The first term is the thesis value at γ = 1, scaled with the kernel width
(0.05/√γ = 0.071 h). The second keeps ε below the point spacing. On the 5-D
thesis data the first term applies, and the result is the thesis's 0.05.

---

## 4. The bandwidth relative to the data, and dimension

Thesis Sec. 3.1.1 shows that the kernel keeps its contrast in d dimensions if
γ = c/(2d). Its experiment (Table 1) nevertheless kept γ = 1 at every
dimension. At D = 45 the squared distances are about 90, every pull is about
e⁻⁹⁰, and every point becomes a singleton. GACA now sweeps γ = c/s², where s²
is the median squared distance between rows, so that c is dimensionless:

- **Informative columns:** with every column carrying signal, six clusters
  were recovered at ARI ≥ 0.98 from 10 to 50 dimensions (fixed γ = 1: ARI
  0.01 from 30 dimensions on).
- **Noise columns:** with five signal dimensions among noise, raw data held to
  about 15 dimensions (ARI 0.86) and fell to 0.17 at 30. PCA restored it
  (0.89 at five components). Concentration of distance is real for noise
  dimensions, but scaling γ removes most of the collapse the thesis attributed
  to dimension.
- **The fine end of the grid:** in data with structure at several scales, s²
  is set by the largest gaps, so the grid also runs until the kernel is about
  three nearest-neighbour distances wide.

---

## 5. Choosing the resolution: plateaus and stability

At each γ of the sweep, GACA is fitted on four random subsamples. The number
of clusters k (clusters holding at least 1% of rows) and the stability (mean
adjusted Rand index between the subsamples' labelings) are recorded.
Stability is computed only over rows that both labelings put in a cluster.
Otherwise a γ at which every row is a Lone Sun would look perfectly stable.

- **Choice:** the longest run of consecutive γ values with the same k ≥ 2.
  Among equally long runs, the finest one whose mean stability is within 0.1
  of the most stable. Within the run, the middle of its most stable stretch,
  because a plateau's edge tips over when the coreset size changes.
- **Why "finest within 0.1":** a stability estimated from four subsamples
  cannot tell 0.99 from 0.94. Breaking such ties by stability chose views
  that were too coarse.
  - On the development sets (synthetic data, scikit-learn classics, SDSS), the
    change raised the chosen level's mean ARI from 0.63 to 0.73 without a
    single regression.
  - On five held-out OpenML datasets, fixed before the rule was written, it
    rose from 0.30 to 0.41 (pendigits 0.07 to 0.51; one small regression,
    banknote 0.12 to 0.05).
- **Known limitation: plateau length is quantised.** Lengths count grid
  points, so they jump by whole steps with small changes in the data. On the
  anisotropic test set, the 2- and 4-cluster plateaus are 3 to 5 points long.
  Which one is longer, and therefore chosen, flips with the number of rows:
  ARI 1.00 at 3,000 and 6,000 rows, 0.33 at 4,000 and 8,000. Both levels are
  always in the hierarchy. A length tolerance would fix this case, but it
  was not adopted: it would have been designed on this one case and could
  not be checked on unseen data. Making the length comparison robust is
  future work.
- **Stability floor (0.8):** a split counts only with stability ≥ 0.8. On a
  single Gaussian the best "split" has stability 0.6 to 0.75, while real
  structure scored 0.89 or more in every test. Without the floor, featureless
  data produces spurious clusters.

---

## 6. The hierarchy as a scale-space tree

Sweeping the kernel width is the scale-space view of clustering (Witkin, 1983):
modes appear and merge as the scale changes, and their history forms a tree
(the mode tree of Minnotte & Scott, 1993). GACA builds the tree from its
stable plateaus. All levels are fitted on the same coreset, so every particle
has a cluster at every level.

- **Coarser levels:** a coarse cluster's children are the fine clusters most
  of whose particles it holds.
- **Finer levels:** a row is assigned, among the children of its own cluster,
  to the one that pulls it most.

The levels are therefore strictly nested, and the tree comes from the
dynamics rather than from merging labels after the fact. Section 1 says what
each level means: a level with kernel width h separates groups more than
about 3.2 h apart.

---

## 7. Scaling a column whose values form separate groups

Robust scaling divides a column by its interquartile range (IQR). If the
column's values form two groups far apart, the IQR spans the gap, so the
column is squeezed and the subgroups along it merge. A 1-D Gaussian mixture
(1 to 3 components, chosen by BIC) is therefore fitted to each column. Adjacent
components that overlap (Ashman's D = √2·|μ₁ − μ₂|/√(σ₁² + σ₂²) < 2; Ashman,
Bird & Zepf, 1994) are merged. Components with less than 5% of the rows, or
with almost no spread (an imputed constant), are ignored. If at least two
separated groups remain, the column is scaled by their pooled within-group
standard deviation, but never by more than the IQR value.

---

## 8. Saddle linking, and why it is a separate view

Linking joins two Suns when the density along the best edge between their
members (sampled at the ends and three interior points) stays above τ times
the lower of the two peaks: a persistence ratio, as in ToMATo (Chazal et al.,
2013). Small Suns are never joined. This repairs clusters that the dynamics
cut into pieces: a ring or crescent has no density dip along it, so its
pieces link up.

Section 1 shows why linking cannot be on by default. The dynamics separate
overlapping Gaussian clusters that have no density dip between them, and a
density-based linking test cannot tell those from the pieces of a ring. A
fixed τ and a significance test on the dip both failed to choose between the
two cases in the benchmark. GACA therefore computes the linked clustering as
a second view and reports both when they disagree.

---

## References

- M. Á. Carreira-Perpiñán (2006). Fast nonparametric clustering with Gaussian
  blurring mean-shift. *ICML*.
- A. P. Witkin (1983). Scale-space filtering. *IJCAI*.
- M. C. Minnotte and D. W. Scott (1993). The mode tree: a tool for
  visualization of nonparametric density features. *Journal of Computational
  and Graphical Statistics* 2(1).
- K. M. Ashman, C. M. Bird and S. E. Zepf (1994). Detecting bimodality in
  astronomical datasets. *The Astronomical Journal* 108.
- F. Chazal, L. J. Guibas, S. Y. Oudot and P. Skraba (2013). Persistence-based
  clustering in Riemannian manifolds. *Journal of the ACM* 60(6).
- M. Gavish and D. L. Donoho (2014). The optimal hard threshold for singular
  values is 4/√3. *IEEE Transactions on Information Theory* 60(8).
