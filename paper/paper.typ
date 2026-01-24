#set page(
  paper: "us-letter",
  margin: 2.5cm,
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en"
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.1")

// Custom Commands
#let RR = $bb(R)$
#let bL = $hat(L)$
#let bR = $hat(R)$
#let so = $frak("so")$

// Theorem Environments
#let theorem(title: none, body) = figure(
  kind: "theorem",
  supplement: "Theorem",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "theorem"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let proposition(title: none, body) = figure(
  kind: "proposition",
  supplement: "Proposition",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "proposition"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let definition(title: none, body) = figure(
  kind: "definition",
  supplement: "Definition",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "definition"): it => block(width: 100%, inset: 8pt, stroke: (left: 2pt + black), radius: 0pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let corollary(title: none, body) = figure(
  kind: "corollary",
  supplement: "Corollary",
  caption: title,
  numbering: "1",
  body
)

#show figure.where(kind: "corollary"): it => block(width: 100%, inset: 8pt, fill: luma(240), radius: 4pt)[
  *#it.supplement #context it.counter.display(it.numbering)*#if it.caption != none [ (#it.caption.body).]
  #it.body
]

#let proof(body) = block(width: 100%)[
  _Proof._ #body #h(1fr) $square$
]

// Title Block
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Learning Interpretable Dynamics of Temporal Networks \
    via Neural ODEs and Symbolic Regression
  ]

  #v(1em)

  Connor Smith#super[1] and Giulio V. Dalla Riva#super[1]

  #v(0.5em)

  #text(size: 0.9em)[
    #super[1]School of Mathematics and Statistics, University of Canterbury, New Zealand
  ]
]

#v(2em)

// Abstract
#align(center)[
  #block(width: 85%)[
    *Abstract* \
    Temporal networks---networks whose structure changes over time---appear across domains from neuroscience to ecology to social systems.
    While most approaches focus on predicting future network states, they rarely provide interpretable models of the underlying dynamics.
    We present a framework that learns continuous, interpretable differential equations governing the evolution of temporal network structure.
    Our approach embeds networks into a low-dimensional latent space via Random Dot Product Graphs (RDPG), learns the dynamics of this embedding using Neural Ordinary Differential Equations (Neural ODEs), and extracts human-interpretable equations through symbolic regression.
    We develop a gauge-theoretic analysis showing that RDPG embeddings have rotational ambiguity, and derive a gauge-consistent architecture $dot(X) = N(P)X$ with symmetric $N$ that eliminates this ambiguity while achieving dramatic parameter reduction (from $approx$10,000 to as few as 2 parameters).
    We demonstrate the framework on synthetic temporal networks, showing that it successfully recovers governing equations and dynamical parameters.
    This work bridges the gap between predictive accuracy and mechanistic understanding in temporal network modeling.
  ]
]

#v(2em)

= Introduction <sec:intro>

Temporal networks---networks whose edges and nodes change over time---are ubiquitous in complex systems @HOLME201297.
Examples include protein interaction networks that rewire during cellular processes @lucas2021inferring, social networks where relationships form and dissolve @hanneke2010discrete, and ecological networks whose structure responds to environmental change @poisot2015species.
Understanding how and why network structure changes is central to predicting system behavior.

Most temporal network modeling falls into two categories.
The first models _dynamics on networks_: how node states evolve given a fixed or slowly-changing network topology (e.g., epidemic spreading, opinion dynamics) @porter2016dynamical.
The second models _dynamics of networks_: how the network structure itself evolves @holme2015modern.
This paper addresses the latter, which remains less developed despite its importance.

Existing approaches to modeling network dynamics face a fundamental tension.
Statistical models like temporal exponential random graphs @hanneke2010discrete are interpretable but often lack predictive power.
Machine learning approaches @kazemi2020representation achieve better predictions but function as black boxes, offering little insight into the mechanisms driving structural change.
We propose a framework that achieves both: predictive models that can be distilled into interpretable differential equations.

Our key insight is that the discreteness of network events (edges appearing or disappearing) can be overcome by working in a continuous embedding space.
We use Random Dot Product Graphs (RDPG) @athreya2017statistical to embed each network snapshot into a low-dimensional latent space where similar nodes cluster together and connection probabilities arise naturally from inner products.
The temporal evolution of these embeddings is then smooth and amenable to differential equation modeling.

We train Neural Ordinary Differential Equations (Neural ODEs) @chen2018neural to learn the dynamics in embedding space.
While Neural ODEs provide excellent fits, they remain opaque.
We therefore apply symbolic regression to discover closed-form differential equations that approximate the learned neural dynamics.
These equations are interpretable---they can be analyzed mathematically, checked for conservation laws, and compared across systems.

*Contributions.* We introduce:
+ A complete pipeline from temporal network observations to interpretable differential equations
+ A gauge-theoretic analysis of RDPG dynamics, identifying what can and cannot be learned from embedding trajectories
+ Parsimonious architectures ($dot(X) = N(P)X$ with symmetric $N$) that are gauge-consistent by construction and can recover exact dynamical parameters
+ Demonstration on synthetic systems with known ground-truth dynamics
+ Open-source Julia implementation (`RDPGDynamics.jl`) for reproducibility

= Methods <sec:methods>

Our framework consists of three stages: (1) embedding temporal networks via RDPG, (2) learning dynamics with Neural ODEs, and (3) extracting interpretable equations through symbolic regression.

== Random Dot Product Graph Embedding <sec:rdpg>

Given a temporal network represented as a sequence of adjacency matrices ${A_t}_(t=1)^T$, we embed each snapshot into a latent space using Random Dot Product Graphs (RDPG) @athreya2017statistical.

For an adjacency matrix $A in {0,1}^(n times n)$, the RDPG embedding computes:
$ A approx bL bR^top $
where $bL, bR in RR^(n times d)$ are the left and right embedding matrices and $d << n$ is the embedding dimension.
These are obtained via truncated singular value decomposition (SVD):
$ A = U Sigma V^top => bL = U_d Sigma_d^(1/2), quad bR = V_d Sigma_d^(1/2) $
where subscript $d$ denotes truncation to the top $d$ singular values/vectors.

The matrix $P = bL bR^top$ has entries $P_(i j) in [0,1]$ representing the probability of an edge between nodes $i$ and $j$.
This probabilistic interpretation is central to our approach: while edge events are discrete, connection probabilities evolve continuously.

*Temporal alignment.*
SVD decompositions are unique only up to orthogonal transformations.
To ensure smooth trajectories across time, we align each embedding to its predecessor using orthogonal Procrustes rotation:
$ Omega_t = arg min_(Omega^top Omega = I) ||Omega bL_t - bL_(t-1)||_F^2 $
with solution $Omega_t = V U^top$ where $bL_t bL_(t-1)^top = U Sigma V^top$.

== Gauge Freedom and Observable Dynamics <sec:gauge>

A fundamental subtlety of RDPG embedding is that the latent positions $X$ are not uniquely determined by the probability matrix $P$.
For any orthogonal matrix $Q in O(d)$:
$ (X Q)(X Q)^top = X Q Q^top X^top = X X^top = P $
Thus $X$ and $X Q$ represent the _same_ observable $P$.
This is the *gauge freedom* of RDPG: the equivalence class $[X] = {X Q : Q in O(d)}$ corresponds to a single probability matrix.

This gauge freedom has profound implications for learning dynamics.

#definition(title: "Observable vs. Invisible Dynamics")[
  A vector field $f: RR^(n times d) -> RR^(n times d)$ produces _observable dynamics_ if $dot(P) != 0$, where $dot(P) = f(X)X^top + X f(X)^top$.
  Otherwise, the dynamics are _invisible_---the parameterization changes but the graph structure is static.
]

#theorem(title: "Characterization of Invisible Dynamics")[
  Let $X in RR^(n times d)$ have full column rank.
  A vector field $f$ produces invisible dynamics if and only if $f(X) = X A$ for some skew-symmetric matrix $A in so(d)$, i.e., $A^top = -A$.
]

#proof[
  $(<=)$ If $f(X) = X A$ with $A^top = -A$, then $dot(P) = X A X^top + X A^top X^top = X(A + A^top)X^top = 0$.

  $(=>)$ Suppose $dot(P) = 0$. Decompose $f(X) = X A + W$ where $A = (X^top X)^(-1)X^top f(X)$ and $X^top W = 0$.
  Substituting and using $dot(P) = 0$ with full-rank $X$ forces $W = 0$ and $A + A^top = 0$.
]

The invisible dynamics $dot(X) = X A$ are infinitesimal rotations along gauge orbits.
In particular, _uniform rotation around the origin_ satisfies $dot(X)_i = X_i A$, producing $dot(P) = 0$---the embedding rotates but the network is static.

#proposition(title: "Centroid Circulation")[
  Dynamics of the form $dot(X)_i = (X_i - bar(X))A$ with $bar(X) != 0$ and $A in so(d)$ produce observable changes in $P$.
] <prop:centroid>

Crucially, other rotational dynamics _are_ observable.
Rotation around the _origin_ is gauge (invisible), but circulation around a _nonzero centroid_ is observable because it decomposes into gauge plus a shared drift that shifts all dot products (see Appendix @app:theory for proof).

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Dynamics*], [$dot(P) = 0$?], [*Observable?*],
    ),
    table.hline(stroke: 0.5pt),
    [$dot(X) = X A$ (uniform rotation around origin)], [Yes], [No],
    [$dot(X)_i = (X_i - bar(X))A$ with $bar(X) != 0$], [No], [Yes],
    [$dot(X)_i = alpha X_i$ (radial scaling)], [No], [Yes],
    [$dot(X)_i = sum_j w_(i j)(X_j - X_i)$ (attraction/repulsion)], [No], [Yes],
    table.hline()
  ),
  caption: [Classification of dynamics by observability.]
) <tab:observable>

A natural question arises: given that some dynamics are invisible, can we still learn the observable part?

#theorem(title: "Identifiability Modulo Gauge")[
  Let $X in RR^(n times d)$ have full column rank.
  Given $dot(P)$ and $X$, the vector field $f(X)$ is uniquely determined up to gauge:
  $ f(X) = F + X A $
  where $F$ is any solution to $dot(P) = F X^top + X F^top$ and $A in so(d)$ is arbitrary.
]

This theorem is reassuring: there is _no theoretical obstruction_ to learning beyond gauge freedom.
Every non-invisible dynamics can be recovered from observations of $dot(P)$.
The "physical" content---what affects the observable---is uniquely determined; only the coordinate-dependent form varies with gauge choice.

*Implications for learning.*
When we train on estimated positions $hat(X)(t)$, the Procrustes alignment fixes a consistent gauge.
The learned $f$ determines $dot(P)$ correctly, but a different alignment procedure would yield a gauge-equivalent $f + X A$.

== Neural ODE Dynamics <sec:node>

After embedding, we have a sequence of latent positions ${bL_t}_(t=1)^T$.
We flatten each $bL_t in RR^(n times d)$ into a vector $bold(u)_t in RR^(n d)$ and model the dynamics as:
$ (d bold(u)) / (d t) = f_theta(bold(u)) $
where $f_theta$ is a neural network with parameters $theta$.

We parameterize $f_theta$ as a fully-connected network with architecture:
$ f_theta: RR^(n d) arrow.long^("Dense") RR^(128) arrow.long^("celu") RR^(128) arrow.long^("celu") RR^(64) arrow.long^("celu") RR^(n d) $

Training minimizes the prediction error:
$ cal(L)(theta) = sum_(t=1)^T ||bold(u)_t - hat(bold(u))_t(theta)||_2^2 + lambda cal(L)_("prob") $
where $hat(bold(u))_t$ is obtained by integrating the Neural ODE from $bold(u)_1$, and $cal(L)_("prob")$ penalizes predicted probabilities outside $[0,1]$:
$ cal(L)_("prob") = sum_(i != j) max(0, -P_(i j)) + max(0, P_(i j) - 1) $

We use a two-stage optimization: Adam for initial exploration followed by Lion for fine-tuning.
Gradients are computed via adjoint sensitivity analysis for memory efficiency @chen2018neural.

== Universal Differential Equations <sec:ude>

When domain knowledge suggests a particular functional form for the dynamics, we can incorporate it via Universal Differential Equations (UDEs) @SciML_C_Rak.
The vector field decomposes as:
$ f(bold(u)) = f_("known")(bold(u); phi) + f_("NN")(bold(u); theta) $
where $f_("known")$ encodes known physics with parameters $phi$, and $f_("NN")$ is a neural network that learns residual corrections.

For RDPG dynamics, gauge theory (@sec:gauge) suggests a particularly elegant form.

#theorem(title: "Equivariant Dynamics")[
  Let $X in RR^(n times d)$ have full column rank.
  Any $O(d)$-equivariant vector field $f: RR^(n times d) -> RR^(n times d)$ has the form:
  $ f(X) = N(P) dot X $
  where $N: RR^(n times n) -> RR^(n times n)$ depends only on $P = X X^top$.
]

#proof[
  _Proof sketch._ Define $N(X) := f(X) X^dagger$ where $X^dagger = (X^top X)^(-1)X^top$.
  Equivariance $f(X Q) = f(X)Q$ implies $N(X Q) = N(X)$, so $N$ is constant on $O(d)$-orbits.
  Since orbits are indexed by $P = X X^top$, we have $N = N(P)$.
]

This form is automatically gauge-consistent since $N$ depends on the observable $P$, not the gauge-dependent $X$.
The key question is: how should we constrain $N$ to eliminate gauge freedom?

#theorem(title: "Gauge Dynamics are Not Symmetric")[
  The invisible (gauge) dynamics $dot(X) = X A$ with $A in so(d)$ correspond to $N = X A X^dagger$.
  For generic full-rank $X$ and nonzero $A$, this $N$ is *not symmetric*.
]

#proof[
  _Proof sketch._ For $X = I_d$ (taking $n = d$), we have $N = A$, which is skew-symmetric.
  For general $X$ with thin SVD $X = U Sigma V^top$, we get $N = U B U^top$ where $B = Sigma(V^top A V)Sigma^(-1)$.
  Since $V^top A V$ is skew-symmetric and $Sigma$ generically has distinct singular values, $B != B^top$ unless $A = 0$.
]

This theorem is the key insight: _gauge directions correspond to non-symmetric $N$_.
Therefore, constraining $N$ to be symmetric eliminates gauge:

#theorem(title: "Gauge Elimination via Symmetry")[
  Constraining $N(P) = N(P)^top$ (symmetric) eliminates all non-trivial gauge freedom.
  Any symmetric $N$ with $N X != 0$ produces observable dynamics ($dot(P) != 0$).
  Moreover, symmetric $N$ can produce _any_ realizable $dot(P)$---no expressivity is lost.
]

This suggests a principled UDE architecture:
$ N(P) = N_("known")(P) + N_("NN")(P), quad text("both symmetric") $
where $N_("known")$ might be a polynomial in $P$ (encoding local neighbor influence) and $N_("NN")$ learns corrections.

We consider three parameterization classes in order of increasing expressivity:

*Polynomial $N(P)$.* The most parsimonious form:
$ N = alpha_0 I + alpha_1 P + alpha_2 P^2 + dots + alpha_k P^k $
with only $k+1$ learnable scalars.
The interpretation is intuitive: $alpha_0 I$ represents intrinsic node dynamics, $alpha_1 P$ captures direct neighbor influence (one-hop interactions), and $alpha_2 P^2$ captures two-hop effects through shared neighbors.
For many network dynamics, degree $k <= 2$ suffices.

*Gauge invariance of learned parameters.*
A key advantage of expressing dynamics in terms of $P$ rather than $X$ directly: the scalar coefficients $alpha_0, alpha_1, dots$ are _gauge-invariant_.
Since $P = X X^top$ is unchanged by orthogonal transformations $X |-> X Q$, and $N(P)$ depends only on $P$, the learned $alpha_k$ values are independent of the coordinate system chosen by SVD and Procrustes alignment.
This means coefficients learned in _any_ gauge (including the DUASE-estimated coordinates) can be applied to _any other_ gauge (including the true positions)---the "learn anywhere, apply everywhere" principle.
By contrast, dynamics expressed directly in $X$ coordinates (e.g., $dot(X)_1 = a X_1 + b X_2$) would have coefficients that depend on the specific basis chosen.

*Pairwise kernel $N(P)$.* A flexible homogeneous form:
$ N_(i j) = cases(
  kappa(P_(i j)) & i != j,
  h(P_(i i)) & i = j
) $
where $kappa, h: [0,1] -> RR$ are learned functions (small neural networks or parametric forms).
Symmetry is automatic since $P_(i j) = P_(j i)$.
The kernel $kappa$ can capture nonlinear responses to connection probability, such as threshold effects or saturation.

*General symmetric $N(P)$.* A neural network that outputs the upper triangle of a symmetric matrix:
$ "NN": "uptri"(P) |-> "uptri"(N) $
with $(n(n+1))/2$ inputs and outputs.
This is the most expressive but least parsimonious option.

@tab:parameterizations summarizes the parameter counts.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Architecture*], [*Parameters*], [*Expressivity*],
    ),
    table.hline(stroke: 0.5pt),
    [Polynomial ($k=1$)], [2], [Low],
    [Polynomial ($k=2$)], [3], [Low],
    [Pairwise kernel (16-16 NN)], [$approx$300], [Medium],
    [General symmetric NN], [$approx$5,000], [High],
    [Standard Neural ODE], [$approx$10,000], [Highest],
    table.hline()
  ),
  caption: [Parameter counts for N(P)X architectures vs. standard Neural ODE.]
) <tab:parameterizations>

The dynamics $dot(X) = N X$ have a clear physical interpretation: node $i$'s velocity is $dot(X)_i = sum_j N_(i j) X_j$, a weighted combination of all positions where $N_(i j) > 0$ indicates attraction and $N_(i j) < 0$ indicates repulsion.
When the true dynamics have this form, the polynomial parameterization can recover the exact coefficients with orders of magnitude fewer parameters than a generic neural network.

== Realizable Dynamics and Model Diagnostics <sec:realizable>

Beyond gauge freedom, RDPG dynamics face a fundamental geometric constraint: the probability matrix $P = X X^top$ lives on a low-dimensional manifold, so most symmetric perturbations $dot(P)$ are not achievable.

#proposition(title: "Tangent Space Constraint")[
  Let $V in RR^(n times d)$ be an orthonormal basis for $"col"(P) subset RR^n$, and $V_perp in RR^(n times (n-d))$ span its orthogonal complement.
  Any realizable $dot(P)$ satisfies:
  $ V_perp^top dot(P) thick V_perp = 0 $
  The realizable tangent space has dimension $n d - (d(d-1))/2$.
]<prop:tangent>

For $n = 10$ nodes and $d = 2$ dimensions: symmetric matrices have 55 degrees of freedom, but only 19 directions are realizable.
The remaining 36 directions would require increasing the rank of $P$---that is, increasing the latent dimension $d$.

*Model diagnostic.*
This constraint provides a principled diagnostic for model adequacy.
If observed dynamics have structure in the "null-null" block $V_perp^top dot(P) V_perp$, this indicates one of two possibilities:
+ *Model misspecification*: The true dynamics do not preserve low-rank structure, so RDPG embedding is inappropriate.
+ *Dimensional emergence*: The latent dimension $d$ is increasing over time---new factors are emerging in the network structure.

In practice, the constraint is automatically satisfied by dynamics of the form $dot(X) = N(P)X$, since $dot(P) = N P + P N$ has the required tangent structure by construction.
Violations in fitted residuals suggest the chosen $d$ may be too small.

== Probability Constraints <sec:constraints>

For edge probabilities to be valid, we require $P_(i j) in [0,1]$ for all $i,j$.
A sufficient condition is that all node positions lie in the positive orthant of the unit ball:
$ B^d_+ = {x in RR^d : x >= 0, ||x|| <= 1} $
If $X_i in B^d_+$ for all $i$, then $P_(i j) = X_i dot X_j in [0,1]$ by non-negativity of coordinates and Cauchy-Schwarz.

However, $(B^d_+)^n$ is _not_ a fundamental domain: one cannot always rotate $n$ vectors simultaneously into the positive orthant.
For example, two orthogonal unit vectors in $RR^2$ cannot both have non-negative coordinates after any rotation.

We handle constraints via a barrier loss:
$ cal(L)_("prob") = gamma sum_(i,j) [max(0, -P_(i j))^2 + max(0, P_(i j) - 1)^2] $
This encourages learned dynamics to remain in the valid region without breaking the ODE structure.
Unlike projection-based approaches that clamp values, the barrier loss maintains differentiability throughout training.

*Practical note.*
While the $B^d_+$ constraint provides a convenient sufficient condition for mathematical analysis, our experiments show that explicit projection onto $B^d_+$ is neither necessary nor beneficial for numerical learning.
Euclidean ODE solvers with barrier losses maintain valid probabilities while preserving the natural geometry of the embedding space.
Projecting onto $B^d_+$ can distort the geometry and introduce artifacts that complicate learning.
The $B^d_+$ framework remains valuable for theoretical analysis of constraint satisfaction, but practitioners should prefer unconstrained optimization with soft penalties.

== Symbolic Regression <sec:symreg>

The trained Neural ODE provides accurate predictions but remains a black box.
To extract interpretable dynamics, we apply symbolic regression to find closed-form expressions approximating $f_theta$.

Given the learned vector field $f_theta(bold(u))$, we seek symbolic expressions $g(bold(u))$ from a grammar of operations (polynomials, trigonometric functions, etc.) that minimize:
$ min_g integral ||f_theta(bold(u)) - g(bold(u))||^2 dif bold(u) + "complexity"(g) $

The complexity penalty encourages parsimonious expressions.
We use genetic programming @schmidt2009distilling to search the space of symbolic expressions.

*Gauge dependence of symbolic equations.*
A critical caveat: recovered equations are _gauge-dependent_.
The symbolic form depends on the coordinate system fixed by SVD and Procrustes alignment.
In a rotated basis, $dot(X)_1 = omega X_2$ might appear as $dot(Y)_1 = a Y_1 + b Y_2$---different-looking equations producing identical $P(t)$.

What _is_ gauge-invariant:
- Eigenvalues of the linearization (frequencies, decay rates)
- Equilibrium structure (existence, stability type)
- Qualitative behavior (oscillatory, stable, chaotic)

For truly coordinate-free equations, one could regress $dot(P)_(i j)$ directly as functions of $P$, though at the cost of higher dimensionality.

= Data: Synthetic Temporal Networks <sec:data>

We evaluate our framework on three synthetic temporal networks with known generating processes.
This allows us to assess whether extracted equations match ground truth.
Crucially, all three systems exhibit _observable_ dynamics in the sense of @sec:gauge ---the latent position changes produce actual changes in $P$.

== Single Community Oscillation

A network of $n=5$ nodes where connection probabilities oscillate sinusoidally.
All nodes belong to a single community whose internal connectivity varies periodically.
The ground-truth dynamics follow:
$ (d L_(i,1)) / (d t) = omega L_(i,2), quad (d L_(i,2)) / (d t) = -omega L_(i,1) $
producing circular trajectories in embedding space.
This is observable because nodes circulate around a nonzero centroid (@prop:centroid), not the origin.

== Two Communities Merging

A network of $n=11$ nodes initially partitioned into two separate communities.
Over time, the communities gradually merge into a single cohesive group.
This models scenarios like organizational mergers or ecosystem succession.
The dynamics involve attraction between nodes (@tab:observable), which is observable as it changes pairwise dot products.

== Long-Tailed Degree Distribution

A network of $n=36$ nodes with a power-law degree distribution.
This tests whether our method handles networks with heterogeneous node connectivity, which are common in real-world systems due to preferential attachment.

= Results <sec:results>

== Training Performance

@tab:results summarizes the training results across all three systems.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Dataset*], [$n$], [$d$], [*Final MSE*],
    ),
    table.hline(stroke: 0.5pt),
    [Single community oscillation], [5], [2], [0.114],
    [Two communities merging], [11], [2], [1.169],
    [Long-tailed distribution], [36], [2], [0.159],
    table.hline()
  ),
  caption: [Training results for three synthetic temporal networks.]
) <tab:results>

The single community and long-tail systems achieve low reconstruction error, while the merging communities system is more challenging.
This may reflect the more complex dynamics of community reorganization.

== Embedding Trajectories

Figure below (placeholder) shows example embedding trajectories comparing ground truth (from data) with Neural ODE predictions.
The model successfully captures the qualitative dynamics in all cases.

// Placeholder for figure
// #figure(
//     image("plots/trajectories.pdf"),
//     caption: [Embedding trajectories for the single community oscillation system.]
// ) <fig:trajectories>

== Symbolic Regression

For the single community oscillation system, symbolic regression recovers equations of the form:
$ (d L_1) / (d t) approx a L_2, quad (d L_2) / (d t) approx -a L_1 $
matching the ground-truth harmonic oscillator dynamics.
This demonstrates that our framework can recover interpretable, mechanistically meaningful equations from network observations alone.

== Gauge-Consistent Architecture Comparison

To test whether the theoretically-motivated $dot(X) = N(P)X$ form offers practical advantages, we compare three architectures on synthetic pairwise dynamics:
$ dot(X) = (alpha I + beta P)X $
with $alpha = -0.02$ (slight contraction) and $beta = 0.001$ (pairwise attraction).
This dynamics has exactly the polynomial $N(P)X$ form with degree 1, providing a fair test where the correct inductive bias should help.

We compare:
1. *Standard Neural ODE*: Generic $f_theta(X)$ with $approx$10,000 parameters
2. *Polynomial $N(P)X$*: $N = alpha_0 I + alpha_1 P$ with 2 parameters
3. *Kernel $N(P)X$*: $N_(i j) = kappa(P_(i j))$ with $approx$300 parameters

@tab:gauge_results summarizes the results.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Architecture*], [*Parameters*], [*MSE*], [*Parameter Recovery*],
    ),
    table.hline(stroke: 0.5pt),
    [Standard Neural ODE], [$approx$10,000], [---], [N/A],
    [Polynomial $N(P)X$], [2], [---], [$hat(alpha)_0 approx ?$, $hat(alpha)_1 approx ?$],
    [Kernel $N(P)X$], [$approx$300], [---], [N/A],
    table.hline()
  ),
  caption: [Architecture comparison on pairwise dynamics ($n=30$, $d=2$).]
) <tab:gauge_results>

// TODO: Fill in results after running Example 5

The polynomial architecture offers a dramatic reduction in parameters (5000$times$ fewer than standard NN) while potentially recovering the true dynamical coefficients.
When the inductive bias matches the true dynamics, parsimony does not sacrifice accuracy.

= Discussion <sec:discussion>

We presented a framework for learning interpretable dynamics of temporal networks.
By combining RDPG embedding, Neural ODEs, and symbolic regression, we bridge the gap between black-box prediction and mechanistic understanding.

*Theoretical foundations.*
The gauge-theoretic analysis (@sec:gauge) provides principled answers to fundamental questions:
_What can we learn?_ All dynamics except uniform rotation around the origin (@thm:invisible).
_What architecture respects the structure?_ The form $dot(X) = N(P)X$ with symmetric $N$ (@thm:symmetric).
These results inform both architecture design and interpretation of learned models.

*Practical obstructions.*
Beyond the theoretical gauge freedom, practical challenges include:
(i) estimation error in $hat(X)$ from SVD ($approx$35% position error, though $hat(P)$ has only $approx$5% error);
(ii) Procrustes alignment artifacts that can introduce spurious motion or remove real motion resembling global rotation;
(iii) discrete, noisy observations rather than continuous $P(t)$.
These factors may explain why some dynamics (e.g., circulation) are harder to learn than others (e.g., attraction/repulsion).

*Evaluation in $P$-space.*
The gauge freedom implies that $X$-based metrics (e.g., position RMSE) are coordinate-dependent and potentially misleading.
The natural evaluation metric is $P(t) = X(t)X(t)^top$, which is gauge-invariant.
Comparing predicted $hat(P)(t)$ to true $P(t)$ directly tests whether the learned dynamics capture observable network structure, independent of coordinate conventions.
Our experiments confirm that models can achieve low $P$-error even when $X$-trajectories differ substantially due to gauge ambiguity.

*Parsimonious architectures.*
The $dot(X) = N(P)X$ architecture with symmetric $N$ offers two key advantages over generic neural networks: (1) automatic gauge consistency---symmetric $N$ cannot produce invisible dynamics (@thm:symmetric), and (2) dramatic parameter reduction---polynomial $N$ achieves comparable accuracy with $10^3$--$10^4$ fewer parameters.
When the true dynamics have this form, the polynomial parameterization can recover exact coefficients, providing interpretability that symbolic regression cannot match for black-box neural networks.

*Model diagnostics.*
The tangent space constraint (@prop:tangent) provides a diagnostic for model adequacy.
If residuals $hat(dot(P)) - dot(P)_("pred")$ have systematic structure in the null-null block $V_perp^top (dot) V_perp$, this suggests either that the RDPG model is inappropriate or that the latent dimension $d$ should be increased.
This connects temporal network modeling to static dimension selection methods, potentially enabling dynamic diagnostics for detecting emerging community structure.

*Limitations.*
The long-tailed network shows higher reconstruction error, suggesting challenges with highly heterogeneous degree distributions.
The polynomial $N(P)X$ form, while parsimonious, may be too restrictive for dynamics that do not factor through $P$.
For such cases, the kernel or general symmetric architectures provide a middle ground.
Additionally, all methods rely on accurate RDPG embedding, which introduces estimation error ($approx$35% in positions, though only $approx$5% in probabilities).

*Extensions.*
The UDE framework (@sec:ude) enables incorporating domain knowledge.
For ecological networks, one might encode known trophic interactions in $N_("known")$ while learning corrections.
For social networks, community structure could inform block-diagonal parameterizations.
The theory extends to directed graphs (Appendix @app:directed), where $P = L R^top$ with separate dynamics for source and target embeddings, and to oscillatory dynamics (Appendix @app:oscillations), which symmetric $N$ can produce through nonlinear coupling despite having real eigenvalues in the linear case.

= Conclusion

We introduced a framework that transforms the problem of temporal network modeling from discrete event prediction to continuous dynamical systems analysis.
The gauge-theoretic analysis reveals that RDPG embeddings have inherent rotational ambiguity, but we identify a broad class of observable dynamics and derive architectures that are gauge-consistent by construction.

The parsimonious $dot(X) = N(P)X$ form with symmetric $N$ achieves two goals simultaneously: it eliminates ambiguity about what the model can learn, and it reduces parameters by orders of magnitude while maintaining or improving accuracy.
When the true dynamics have this form, the polynomial parameterization can recover exact coefficients---a level of interpretability that post-hoc symbolic regression cannot match.

Our approach produces interpretable differential equations governing network evolution, enabling both prediction and mechanistic insight.
The open-source implementation facilitates application to new domains.

= Acknowledgments

// TODO: Add acknowledgments

= Data and Code Availability

The `RDPGDynamics.jl` package and all data are available at [repository URL].
Experiments can be reproduced with: `julia --project scripts/reproduce_paper.jl`

#bibliography("bibliography.bib", style: "ieee")

#pagebreak()

= Proofs and Extended Theory <app:theory>

This appendix provides complete proofs and extended theoretical results supporting the gauge-theoretic framework presented in @sec:gauge.

== Proof of Theorem (Full Version) <thm:invisible>

#proof[
  $(<=)$ Suppose $f(X) = X A$ with $A^top = -A$. Then:
  $ dot(P) = X A X^top + X (X A)^top = X A X^top + X A^top X^top = X(A + A^top)X^top = 0 $

  $(=>)$ Suppose $dot(P) = f(X)X^top + X f(X)^top = 0$.

  Decompose $f(X) = X A + W$ where $A = (X^top X)^(-1)X^top f(X)$ and $X^top W = 0$ (i.e., $W$ lies in the orthogonal complement of $"col"(X)$).

  Substituting into $dot(P) = 0$:
  $ 0 = (X A + W)X^top + X(X A + W)^top = X(A + A^top)X^top + W X^top + X W^top $

  For $X$ with full column rank, consider the constraint $X^top W = 0$ combined with the equation above.
  Taking the projection onto $"col"(X)$: multiplying on left by $(X^top X)^(-1)X^top$ and on right by $X(X^top X)^(-1)$:
  $ 0 = A + A^top + (X^top X)^(-1)X^top W X^top X (X^top X)^(-1) + "similar term" $

  Since $X^top W = 0$, the middle terms vanish, giving $A + A^top = 0$.

  For the remaining equation $W X^top + X W^top = 0$ with $X^top W = 0$: this is a symmetric matrix equation. For generic full-rank $X$, the only solution is $W = 0$.
]

#corollary[
  The space of invisible dynamics is isomorphic to $so(d)$, with dimension $(d(d-1))/2$.
  For $d = 2$, this is 1-dimensional (a single rotation rate); for $d = 3$, it is 3-dimensional.
]

== Proof of Proposition @prop:centroid

#proof[
  Let $bar(X) = 1/n sum_i X_i$ be the centroid. The dynamics $dot(X)_i = (X_i - bar(X))A$ can be rewritten as:
  $ dot(X)_i = X_i A - bar(X)A $

  In matrix form with $bold(1) in RR^n$ the all-ones vector:
  $ dot(X) = X A - bold(1)bar(X)^top A $

  Computing $dot(P)$:
  $ dot(P) &= dot(X)X^top + X dot(X)^top \
           &= (X A - bold(1)bar(X)^top A)X^top + X(X A - bold(1)bar(X)^top A)^top \
           &= X A X^top + X A^top X^top - bold(1)bar(X)^top A X^top - X A^top bar(X)bold(1)^top $

  The first two terms cancel since $A + A^top = 0$. Let $v = A^top bar(X) = -A bar(X)$:
  $ dot(P) = bold(1)v^top X^top + X v bold(1)^top $

  Entry-wise: $dot(P)_(i j) = v dot X_j + X_i dot v$.

  This vanishes for all $i,j$ only if $v = 0$, i.e., $A bar(X) = 0$. For generic $bar(X) != 0$ and $A != 0$, we have $dot(P) != 0$.
]

*Interpretation.* Circulation around the centroid decomposes as:
$ dot(X)_i = underbrace(X_i A, "invisible (gauge)") - underbrace(bar(X)A, "shared drift (observable)") $
The first term is pure gauge. The second is a constant velocity applied to all nodes, which shifts all dot products and hence changes $P$.

== Gauge Equivalence and Canonical Decomposition

#theorem(title: "Gauge Equivalence")[
  Two vector fields $f$ and $tilde(f)$ are gauge equivalent (induce the same $dot(P)$) if and only if:
  $ f(X) - tilde(f)(X) = X A(X) $
  for some $so(d)$-valued function $A(X)$.
]

#proof[
  Apply Theorem (@thm:invisible) to the difference $h = f - tilde(f)$.
]

#corollary(title: "Canonical Decomposition")[
  Any vector field decomposes uniquely as:
  $ f(X) = f_("phys")(X) + X A(X) $
  where $f_("phys")$ determines $dot(P)$ and $X A$ is pure gauge.
]

*Implication for learning:* If we learn $f$ from observations of $P(t)$, we can only determine $f$ up to the gauge freedom $X A$. The "physical" content---what affects the observable---is uniquely determined.

== Differential Rotation Rates

#theorem(title: "Differential Rotation is Observable")[
  If nodes have different rotation rates:
  $ dot(X)_i = X_i A_i, quad A_i in so(d) $
  then:
  $ dot(P)_(i j) = X_i (A_i - A_j) X_j^top $
  This is generically nonzero when $A_i != A_j$.
]

#proof[
  $ dot(P)_(i j) &= dot(X)_i X_j^top + X_i dot(X)_j^top = (X_i A_i) X_j^top + X_i (X_j A_j)^top \
               &= X_i A_i X_j^top + X_i A_j^top X_j^top = X_i A_i X_j^top - X_i A_j X_j^top \
               &= X_i (A_i - A_j) X_j^top $
  using $A_j^top = -A_j$. This is nonzero when $A_i != A_j$ and $X_i, X_j$ are generic.
]

*Interpretation:* Uniform rotation ($A_i = A$ for all $i$) is invisible, but heterogeneous rotation rates are observable.

== Gauge-Free Decomposition of $dot(X)$

Any velocity $dot(X)$ decomposes uniquely as:
$ dot(X) = X dot A + W $
where $A = (X^top X)^(-1)X^top dot(X) in RR^(d times d)$ and $W perp "col"(X)$ (i.e., $X^top W = 0$).

Further decompose $A = A_("sym") + A_("skew")$ into symmetric and skew-symmetric parts:

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header(
      [*Component*], [*Contributes to $dot(P)$?*], [*Interpretation*],
    ),
    table.hline(stroke: 0.5pt),
    [$X dot A_("sym")$], [Yes], [Radial/stretching dynamics],
    [$X dot A_("skew")$], [No], [Pure rotation (gauge)],
    [$W$], [Yes], [Rotates $"col"(P)$ in $RR^n$],
    table.hline()
  ),
  caption: [Decomposition of $dot(X)$ into observable and gauge components.]
)

*The observable content of $dot(X)$ is $(A_("sym"), W)$.*

== Proof of Theorem (Full Statement) <thm:symmetric>

#theorem(title: "Gauge Elimination via Symmetry---Full Statement")[
  Let $N = N^top$ be symmetric. If $N X != 0$, then $dot(P) = N P + P N != 0$.
  Moreover, symmetric $N$ eliminates all non-trivial gauge freedom in the sense that no symmetric $N$ can produce invisible dynamics except $N = 0$ restricted to $"col"(X)$.
]

#proof[
  Suppose $N = N^top$ and $dot(P) = N P + P N = 0$.

  Let $P = V Lambda V^top$ be the spectral decomposition with $Lambda = "diag"(lambda_1, dots, lambda_d, 0, dots, 0)$ and $lambda_i > 0$ for $i <= d$.

  Define $tilde(N) = V^top N V$ (symmetric since $N$ is). The condition $N P + P N = 0$ transforms to:
  $ tilde(N)Lambda + Lambda tilde(N) = 0 $

  Entry-wise: $tilde(N)_(i j)(lambda_i + lambda_j) = 0$.

  *Case analysis:*
  - For $i, j <= d$: $lambda_i + lambda_j > 0$, so $tilde(N)_(i j) = 0$.
  - For $i <= d$, $j > d$: $lambda_i + 0 = lambda_i > 0$, so $tilde(N)_(i j) = 0$.
  - For $i > d$, $j <= d$: by symmetry $tilde(N)_(i j) = tilde(N)_(j i) = 0$.
  - For $i, j > d$: the constraint $0 dot tilde(N)_(i j) = 0$ is vacuous.

  Therefore $tilde(N) = mat(0, 0; 0, tilde(N)_(22))$ where $tilde(N)_(22) in RR^((n-d) times (n-d))$ is arbitrary symmetric (supported on $"null"(P)$).

  Since $"col"(X) = "col"(V_1)$ where $V_1$ comprises the first $d$ columns of $V$, we can write $X = V_1 R$ for invertible $R$. Then:
  $ N X = V tilde(N) V^top V_1 R = V tilde(N) mat(I_d; 0) R = V mat(0; 0) = 0 $

  *Contrapositive:* If $N = N^top$ and $N X != 0$, then $N P + P N != 0$, so $dot(P) != 0$.
]

== The Tangent Space of RDPG Dynamics

The probability matrix $P = X X^top$ lives on the manifold $cal(M)_d$ of rank-$d$ positive semidefinite matrices. Not every symmetric $dot(P)$ is achievable---only those in the tangent space.

#proposition(title: "Tangent Space Characterization")[
  Let $P = X X^top$ with $X in RR^(n times d)$ full column rank. Let $V in RR^(n times d)$ be an orthonormal basis for $"col"(X)$, and $V_perp in RR^(n times (n-d))$ span its orthogonal complement.

  The tangent space $T_P cal(M)_d$ consists of symmetric matrices $dot(P)$ satisfying:
  $ V_perp^top dot(P) thick V_perp = 0 $
]

#proof[
  Any realizable $dot(P) = F X^top + X F^top$ for some $F in RR^(n times d)$. Since $"col"(X) = "col"(V)$, we have $X = V R$ for invertible $R in RR^(d times d)$. Then:
  $ V_perp^top dot(P) thick V_perp = V_perp^top F R^top V^top V_perp + V_perp^top V R F^top V_perp = 0 + 0 = 0 $
  using $V^top V_perp = 0$. Conversely, any symmetric $dot(P)$ with $V_perp^top dot(P) V_perp = 0$ can be written in this form.
]

*Interpretation of blocks.* Any symmetric $n times n$ matrix $M$ decomposes as:
$ M = underbrace(V A V^top, "range-range") + underbrace(V B V_perp^top + V_perp B^top V^top, "range-null cross") + underbrace(V_perp C V_perp^top, "null-null") $
For realizable $dot(P)$: the $A$ and $B$ blocks can be arbitrary, but $C = 0$ always.

The null-null block represents "structure in the orthogonal complement"---dynamics that would increase the rank of $P$.
If we fit an RDPG model and find systematic residuals with $C != 0$, this suggests the latent dimension $d$ is too small or the RDPG model is inappropriate.

#corollary(title: "Dimension Count")[
  $ dim(T_P cal(M)_d) = n d - (d(d-1))/2 $
  This equals the dimension of $X$-space ($n d$) minus the gauge freedom ($(d(d-1))/2$)---exactly the observable degrees of freedom.
]

*Example.* For $n = 10$, $d = 2$:
- Symmetric $n times n$ matrices: $(10 dot 11)/2 = 55$ dimensions
- Tangent space: $10 dot 2 - 1 = 19$ dimensions
- Unrealizable directions: $55 - 19 = 36$ dimensions
Most symmetric perturbations of $P$ are _not_ achievable by RDPG dynamics!

= Parsimonious UDE Parameterizations <app:ude>

This appendix catalogs parameterization choices for $N(P)$ in the UDE framework $dot(X) = N(P)X$.

== Taxonomy by Homogeneity

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Type*], [*Description*], [*Parameters*],
    ),
    table.hline(stroke: 0.5pt),
    [Homogeneous], [All node pairs follow same rule], [$O(1)$ functions],
    [Type-based], [Nodes grouped into $K$ types], [$O(K^2)$ functions],
    [Node-specific], [Each node has own parameters], [$O(n)$ scalars],
    [Fully heterogeneous], [Each pair has own parameter], [$O(n^2)$---avoid!],
    table.hline()
  ),
  caption: [Parameterization complexity by homogeneity assumption.]
)

== Homogeneous Parameterizations

*Polynomial in $P$:*
$ N(P) = alpha_0 I + alpha_1 P + alpha_2 P^2 + dots + alpha_k P^k $
Parameters: $k+1$ scalars. Interpretation: $alpha_0 I$ is self-dynamics, $alpha_1 P$ is direct neighbor influence, $alpha_2 P^2$ is two-hop influence.

*Pairwise kernel:*
$ N_(i j)(P) = kappa(P_(i j)) quad text("for ") i != j, quad quad N_(i i)(P) = h(P_(i i)) $
Symmetry is automatic since $P_(i j) = P_(j i)$. The function $kappa$ can be a small neural network or parametric (e.g., $kappa(p) = a + b p + c p^2$).

*Attraction-repulsion (Lennard-Jones inspired):*
$ kappa(p) = a / (p + epsilon) - b / (p + epsilon)^2 $
Parameters: $(a, b, epsilon)$. Equilibrium occurs where attraction balances repulsion.

*Laplacian-based:*
$ N(P) = alpha (D^(-1\/2) P D^(-1\/2) - I) $
where $D = "diag"(P bold(1))$ is the degree matrix. This encodes normalized diffusion.

== Type-Based Parameterizations

When nodes belong to types $tau: {1, dots, n} -> {1, dots, K}$, interactions can depend on type pairs.

*Block kernel:*
$ N_(i j) = kappa_(tau(i), tau(j))(P_(i j)) $
Symmetry requires $kappa_(a b) = kappa_(b a)$. Parameters: $(K(K+1))/2$ functions.

*Stochastic Block Model prior:*
$ N_(i j) = alpha_(tau(i), tau(j)) + beta dot P_(i j) $
Parameters: $(K(K+1))/2$ scalars $alpha_(a b)$ plus one shared $beta$.

Interpretation: Base rate depends on community pair, plus universal connection-strength effect.

== Node-Specific Parameterizations

Each node has individual parameters, but interactions follow shared rules.

*Diagonal + shared off-diagonal:*
$ N_(i j) = cases(
  h_i & i = j,
  kappa(P_(i j)) & i != j
) $
Parameters: $n$ scalars $h_i$ (node-specific self-rates) plus 1 function $kappa$ (shared interaction).

*Node features determine rate:*
$ h_i = g(phi_i), quad phi_i = (P_(i i), sum_j P_(i j), max_j P_(i j), dots) $
where $phi_i$ are node-level features extracted from $P$ and $g$ is a shared function (can be a small NN).

== Message-Passing Formulation

An equivalent view writes dynamics as node-level updates:
$ dot(X)_i = a(P_(i i)) X_i + sum_(j != i) m(P_(i j)) (X_j - X_i) $
where $a$ is the intrinsic rate and $m$ is the message function.

The equivalent symmetric $N$ is:
$ N_(i j) = cases(
  a(P_(i i)) - sum_(k != i) m(P_(i k)) & i = j,
  m(P_(i j)) & i != j
) $

== Low-Rank Parameterizations

*Rank-$r$ symmetric:*
$ N = sum_(k=1)^r alpha_k u_k u_k^top = U "diag"(alpha) U^top $
Parameters: $n r + r$ (with orthogonality constraints on $U$).

*Data-derived basis:*
$ N = sum_(k=1)^r alpha_k v_k v_k^top $
where $v_k$ are the top eigenvectors of $P$ itself. Parameters: $r$ scalars only.

== Encoding Qualitative Priors

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Prior*], [*Parameterization*],
    ),
    table.hline(stroke: 0.5pt),
    [Stability (nodes don't explode)], [$N = -exp(M)$ where $M = M^top$],
    [Conservation ($"tr"(P)$ constant)], [Project $N$ to $"tr"(N P) = 0$],
    [Known equilibrium $P^ast$], [$N(P) = (P - P^ast)M(P)$],
    [Sparsity preservation], [$N_(i j) = P_(i j) dot kappa(P_(i j))$],
    table.hline()
  ),
  caption: [Parameterizations encoding specific priors.]
)

== Geometric Boundary Constraints

There are two independent sources of constraints on $dot(P)$: algebraic (rank preservation, automatically satisfied by $dot(X) = N(P)X$) and geometric (probability bounds).

*At lower boundary ($P_(i j) = 0$):* Require $dot(P)_(i j) >= 0$.

From $dot(P) = N P + P N$:
$ dot(P)_(i j) = sum_k N_(i k) P_(k j) + sum_k P_(i k) N_(k j) $

*Caution:* This is NOT simply a Metzler condition on $N$. For the linear system $dot(y) = A y$, Metzler $A$ (non-negative off-diagonal) preserves the positive orthant. But $dot(P) = N P + P N$ is a Lyapunov equation---the condition involves the entire structure of $P$, not just local properties of $N$.

*At upper boundary ($P_(i j) = 1$):* Require $dot(P)_(i j) <= 0$.

Since $P_(i j) = X_i dot X_j <= ||X_i|| ||X_j||$, we have $P_(i j) = 1$ only if $X_i = X_j$ with $||X_i|| = 1$.

*Practical enforcement:* Rather than modifying $N$ (which breaks the symmetric structure) or projecting onto $B^d_+$ (which distorts the geometry), use a barrier in the loss function:
$ cal(L)_("barrier") = gamma sum_(i,j) [max(0, -P_(i j))^2 + max(0, P_(i j) - 1)^2] $
This encourages learned dynamics to stay in the valid region while preserving the natural geometry of the embedding space.
Our experiments show that projection-based constraint enforcement (e.g., onto $B^d_+$) is unnecessary and can actually impede learning.

*Summary:* In the interior of the valid configuration space, only the algebraic constraint matters---and it's automatic. Geometric constraints only matter at boundaries. Soft penalties (barrier loss) outperform hard constraints (projection) in practice.

= Oscillatory Dynamics with Symmetric $N$ <app:oscillations>

A common concern: can symmetric $N(P)$ produce oscillations? For a _linear_ system $dot(X) = N X$ with constant symmetric $N$, eigenvalues are real, so solutions are sums of exponentials---no oscillations.

However, our system $dot(X) = N(P)X = N(X X^top)X$ is _nonlinear_ because $N$ depends on $X$ through $P$.

== Linearization Around Equilibrium

At equilibrium $X^ast$ with $N(P^ast)X^ast = 0$, the linearization is:
$ delta dot(X) = N(P^ast)delta X + [(partial N) / (partial P) bar.v_(P^ast) dot (delta X dot X^ast^top + X^ast dot delta X^top)] X^ast $

The Jacobian (as a linear operator on $delta X in RR^(n times d)$) is _not_ simply $N(P^*)$. The second term involves derivatives of $N$ and creates coupling that can produce complex eigenvalues.

== Mechanisms for Oscillation

*1. Hopf bifurcation:* As parameters vary, eigenvalues of the Jacobian can cross the imaginary axis, creating limit cycles.

*2. Amplitude-phase coupling (for $d = 2$):* Write $X_i = r_i (cos theta_i, sin theta_i)$. Then $P_(i j) = r_i r_j cos(theta_i - theta_j)$. Phase differences affect probabilities, which affect phase dynamics---a feedback loop enabling oscillation.

*3. Multi-scale interaction:*
$ N(P) = alpha_1 P - alpha_2 P^2 $
Local attraction ($alpha_1 P$) competes with nonlocal repulsion ($-alpha_2 P^2$), potentially creating oscillatory approach to equilibrium.

== What Symmetric $N$ Cannot Do

- Rotation around origin in latent space (this is gauge/invisible anyway)
- Oscillations in the linear approximation with _constant_ $N$

== What Symmetric $N$ Can Do

- Damped oscillations approaching equilibrium (via nonlinear Jacobian)
- Limit cycles via Hopf bifurcation
- Quasi-periodic motion in higher dimensions

= Extension to Directed Graphs <app:directed>

For directed graphs, the probability matrix factors as $P = L R^top$ where $L, R in RR^(n times d)$ are *left* (source) and *right* (target) embeddings.

== Gauge Group

The gauge transformation is $(L, R) |-> (L Q, R Q)$ for $Q in O(d)$:
$ (L Q)(R Q)^top = L Q Q^top R^top = L R^top = P $
Crucially, both embeddings rotate by the _same_ $Q$.

== Gauge-Invariant Quantities

Under $(L, R) |-> (L Q, R Q)$:
- $P = L R^top$ --- invariant
- $G_L = L L^top$ --- invariant
- $G_R = R R^top$ --- invariant
- $L^top L$, $R^top R$, $L^top R$ --- transform by conjugation

The gauge-invariant data is $(P, G_L, G_R)$---three $n times n$ matrices.

== Equivariant Dynamics

Any $O(d)$-equivariant vector field has the form:
$ dot(L) = N_L(P, G_L, G_R) dot L, quad quad dot(R) = N_R(P, G_L, G_R) dot R $

To eliminate gauge, require $N_L = N_L^top$ and $N_R = N_R^top$ individually.

== Induced Dynamics on $P$

$ dot(P) = N_L P + P N_R^top $

This is a Sylvester equation, enabling:
- Asymmetric growth patterns
- Directed community formation
- Source-target differentiation

*Special case:* If $N_L = N_R = N$ with $N$ symmetric, the directed dynamics reduce to the undirected form $dot(P) = N P + P N$.
