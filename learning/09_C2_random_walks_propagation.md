# Unit C2 — Random walks and label propagation

> **Course:** Drug repositioning with graph neural networks — a self-contained course
> **Chapter 09 of 20** · Track C (Graphs)

| | |
|---|---|
| **Prerequisites** | 02 A2 Linear algebra (eigenvalues, symmetric matrices, matrix inverse); 03 A3 Probability (distributions, expectation, conditional probability); 05 B2 Evaluation (AUC, AUPR, cold start); 08 C1 Graph theory basics (adjacency matrix, degree, kNN graphs, bipartite graphs) |
| **Estimated study time** | 8–10 hours (about 4 h reading with pen and paper, 2 h running and changing the code, 3 h exercises) |
| **Leads to** | 10 B4 Matrix factorisation (graph Laplacian regularisers), 11 C3 GCNs (a GCN layer *is* a normalised propagation step), 14 C6 Link prediction (cold start) |

## Learning objectives

After this chapter you will be able to:

1. **Write down** the transition matrix of a random walk on a weighted graph, and convert between row-, column- and symmetric normalisation, stating which one "averages" and which one "spreads mass".
2. **Compute by hand** the stationary distribution of a small undirected graph, prove that it is proportional to the degree, and explain when a walk fails to converge (reducibility, periodicity).
3. **Define** PageRank, personalised PageRank and random walk with restart (RWR), **derive** the closed form $\mathbf p = (1-\alpha)(I-\alpha W)^{-1}\mathbf e$, and evaluate it on a 3-node graph.
4. **Derive** the label-propagation fixed point $F^* = (1-\alpha)(I-\alpha S)^{-1}Y$ both from the iteration and from an energy (regularisation) objective, and contrast it with the harmonic ("clamped") solution.
5. **Prove** that these iterations converge whenever the spectral radius of the iteration matrix is below 1, and **predict** the number of iterations needed for a given tolerance.
6. **Explain** the bi-random walk of MBiRW line by line: the logistic similarity adjustment, the left walk on the drug network, the right walk on the disease network, the restart weight $\alpha$ and the step counts $l$ and $r$.
7. **Describe** a random walk on a heterogeneous drug–disease network (NRWRH) and build its block transition matrix.
8. **Implement** the project's one-step kNN propagation (`knn_kernel`, `propagation()`), and **explain with evidence** why it is so strong for diseases that have no known drugs (cold start), and why *more* propagation steps make cold start *worse*.

---

## 1. Motivation: why this matters for this project

Drug repositioning rests on one idea, usually called **guilt by association**:

> *A drug is likely to treat a disease if similar drugs treat it, or if it treats similar diseases.*

A random walk is the most direct way of turning that sentence into numbers. Picture a person who starts on a disease node, repeatedly hops to a *similar* disease (with probability proportional to similarity), and occasionally hops across a known indication to a drug. The places this walker visits most often are the "nearby" drugs, and those become your predictions.

Three places in this project depend on exactly this idea:

1. **MBiRW** (`src/drepo/methods.py`, class `MBiRW`), one of the five baselines, is a random walk with restart that alternates between the drug-similarity network and the disease-similarity network. With no learned parameters it reaches AUC 0.883 / AUPR 0.311 in 5-fold cross-validation on Fdataset.
2. **The multi-view propagation head** of our own model, MV-HGAT (`propagation()` inside `MVHGATMethod.fit_predict`, together with `knn_kernel` in `src/drepo/data.py`), adds one propagation score per similarity view to the final logit:
   $$\text{logit}(i,j) = \text{gate}(i,j)\,\mathbf h_i^\top W\mathbf h_j \;+\; \sum_v w_v\,P_v[i,j] \;+\; b .$$
   Each $P_v$ is **one step** of row-normalised, k-nearest-neighbour ($k=10$) propagation of the *visible* known links.
3. **Cold start.** When a disease has *no* known drugs (the leave-one-disease-out, or LODO, protocol), a learned embedding for that disease has nothing to learn from. Propagation through disease similarity still works. In the project's validation cold-start proxy on Fdataset, one-step propagation through the phenotype view `pheno_mim` alone reached AUPR **0.149**, the new MONDO semantic view `sem_mondo` **0.106**, the two summed **0.175**, and MBiRW **0.174** — while the random baseline is about **0.010** (the 1% positive rate). A one-line, parameter-free formula is roughly 17× better than random and as good as a full random-walk method.

**A concrete example.** Suppose a newly characterised epilepsy subtype has no recorded drug. Its ten most phenotypically similar diseases in `pheno_mim` are other epilepsies and seizure disorders. The one-step score for a drug $i$ is the similarity-weighted fraction of those ten neighbours that drug $i$ treats. Anticonvulsants such as valproate, which treat many of the neighbours, float to the top of the ranking — without any training at all. Much of this chapter explains *why* such a simple rule is so hard to beat, and how it relates to PageRank, label propagation and the GNN layers of later chapters.

> **Roadmap.** Section 2 builds random walks from scratch (Markov chains, normalisation, stationary distributions). Section 3 adds restarts (PageRank, personalised PageRank, RWR). Section 4 covers label propagation. Section 5 proves convergence. Section 6 covers the bi-random walk of MBiRW, Section 7 heterogeneous walks (NRWRH), Section 8 the project's one-step kNN propagation and cold start, and Section 9 links all of this to GNNs.

---

## 2. Random walks on graphs from first principles

### 2.1 Intuition: ink on a network

Put a drop of ink on one node of a graph. At every tick of the clock, each node passes its ink to its neighbours in proportion to the edge weights. Two things happen:

* **Short term:** the ink stays near where it started. How much ink a node holds after a few ticks measures how *close* it is to the starting node.
* **Long term:** on a connected graph the ink spreads everywhere and settles into a fixed pattern that no longer depends on the starting node — the **stationary distribution**.

Random-walk methods for link prediction use the *short-term* behaviour, because that is where the information about the starting node lives. The *long-term* behaviour is a nuisance (it only reflects how well connected each node is) — and the restart trick of Section 3 exists to stop the ink from forgetting where it came from.

### 2.2 Markov chains

A **Markov chain** on a finite set of states $\{1,\dots,n\}$ is a sequence of random states $X_0, X_1, X_2, \dots$ such that the next state depends only on the current one:
$$\Pr(X_{t+1}=j \mid X_t=i, X_{t-1},\dots,X_0) = \Pr(X_{t+1}=j\mid X_t=i) = P_{ij}.$$
This "memoryless" requirement is the **Markov property**. The numbers $P_{ij}$ form the **transition matrix** $P \in \mathbb R^{n\times n}$, with

$$P_{ij}\ge 0, \qquad \sum_{j} P_{ij} = 1 \quad\text{for every } i .$$

A non-negative matrix whose rows each sum to 1 is called **row-stochastic**. Row $i$ is a probability distribution: "where can I go from $i$, and with which probability?"

**Distributions evolve by matrix multiplication.** Let $\mathbf p_t$ be the distribution of $X_t$, written as a *row* vector ($\mathbf p_t^\top$ with entries $\Pr(X_t=i)$). By the law of total probability,
$$\Pr(X_{t+1}=j) = \sum_i \Pr(X_t=i)\,P_{ij} \quad\Longleftrightarrow\quad \mathbf p_{t+1}^\top = \mathbf p_t^\top P .$$
Applying this $t$ times gives $\mathbf p_t^\top = \mathbf p_0^\top P^t$, so $(P^t)_{ij}$ is the probability of being at $j$ after exactly $t$ steps when starting at $i$ (the **Chapman–Kolmogorov** relation).

### 2.3 From a graph to a transition matrix: three normalisations

Let $S$ be a non-negative, symmetric $n\times n$ weight (similarity) matrix of an undirected graph, and let $d_i = \sum_j S_{ij}$ be the weighted **degree** of node $i$, $D=\mathrm{diag}(d_1,\dots,d_n)$. The natural random walk moves from $i$ to $j$ with probability proportional to the edge weight:
$$P = D^{-1}S, \qquad P_{ij} = \frac{S_{ij}}{d_i}.$$
This is **row normalisation**. In this course you will meet three variants of "normalise $S$", and it pays to keep them apart:

| Name | Matrix | Row sums | Column sums | What multiplying does |
|---|---|---|---|---|
| Row-normalised | $P = D^{-1}S$ | 1 | not 1 | $P\mathbf f$: each node takes the **weighted average** of its neighbours' values ("pull") |
| Column-normalised | $W = SD^{-1} = P^\top$ | not 1 | 1 | $W\mathbf p$: each node **spreads its mass** to its neighbours ("push"); total mass is conserved |
| Symmetric | $\tilde S = D^{-1/2}SD^{-1/2}$ | not 1 | not 1 | a symmetric compromise; used by GCN, Zhou's label propagation, MBiRW |

The "pull" view is the one you will use most in this project. If $\mathbf f$ is a column of numbers on the nodes (for instance column $j$ of the association matrix $A$, which marks the drugs known to treat disease $j$), then
$$(P\mathbf f)_i = \sum_j \frac{S_{ij}}{d_i} f_j$$
is a weighted average of the values of $i$'s neighbours. Applied to a whole matrix, $PA$ replaces every drug's association row by the weighted average of its neighbours' rows. **This is exactly what `knn_kernel(S, k) @ A` does in the project** (Section 8).

The "push" view, $W\mathbf p$ with $W = SD^{-1}$ column-stochastic, moves probability mass forward one step: if $\mathbf p$ (a column vector) is where the walker is now, $W\mathbf p$ is where it will be after one step. It is just the transpose of the row-vector rule $\mathbf p^\top P$.

**The three normalisations have the same eigenvalues.** Because
$$\tilde S = D^{-1/2} S D^{-1/2} = D^{1/2}\,(D^{-1}S)\,D^{-1/2} = D^{1/2} P D^{-1/2},$$
$\tilde S$ and $P$ are *similar matrices* ($\tilde S = M P M^{-1}$ with $M = D^{1/2}$), and similar matrices share eigenvalues. And $W = P^\top$ has the same eigenvalues as $P$ because a matrix and its transpose always do. Since $\tilde S$ is symmetric, all these eigenvalues are **real**. They also lie in $[-1,1]$: for a row-stochastic matrix, $\lVert P\mathbf x\rVert_\infty \le \lVert\mathbf x\rVert_\infty$ (each entry of $P\mathbf x$ is an average of entries of $\mathbf x$), so no eigenvalue can exceed 1 in absolute value; and $\lambda=1$ is always attained, with eigenvector $\mathbf 1$ ($P\mathbf 1 = \mathbf 1$ because rows sum to 1).

#### Worked example 2.1 (by hand)

Take four drugs with similarity weights
$$S=\begin{pmatrix}0&2&1&0\\2&0&1&0\\1&1&0&1\\0&0&1&0\end{pmatrix}.$$
Degrees: $d = (3, 3, 3, 1)$. Row-normalising,
$$P=\begin{pmatrix}0&\tfrac23&\tfrac13&0\\\tfrac23&0&\tfrac13&0\\\tfrac13&\tfrac13&0&\tfrac13\\0&0&1&0\end{pmatrix}.$$
Start the walker at node 3 ($\mathbf p_0 = (0,0,0,1)$). Node 3's only neighbour is node 2, so $\mathbf p_1 = (0,0,1,0)$. From node 2 the walker goes to 0, 1 or 3 with probability $\tfrac13$ each: $\mathbf p_2 = (\tfrac13,\tfrac13,0,\tfrac13)$. One more step:
$$\mathbf p_3 = \tfrac13\,\text{row}_0 + \tfrac13\,\text{row}_1 + \tfrac13\,\text{row}_3
= \tfrac13(0,\tfrac23,\tfrac13,0)+\tfrac13(\tfrac23,0,\tfrac13,0)+\tfrac13(0,0,1,0) = (\tfrac29,\tfrac29,\tfrac59,0).$$
You will check these numbers with code in Section 10.

The eigenvalues of $P$ can also be found by hand. The vector $\mathbf x=(1,-1,0,0)$ gives $P\mathbf x = (-\tfrac23,\tfrac23,0,0) = -\tfrac23\mathbf x$, so $-\tfrac23$ is an eigenvalue. We know $1$ is an eigenvalue, and the trace (sum of eigenvalues) is $0$. The determinant (the product of the eigenvalues) works out to $\tfrac{4}{27}$ by cofactor expansion. Writing the remaining two eigenvalues as $\lambda_3,\lambda_4$: $1-\tfrac23+\lambda_3+\lambda_4=0$ and $1\cdot(-\tfrac23)\lambda_3\lambda_4 = \tfrac4{27}$, i.e. $\lambda_3+\lambda_4 = -\tfrac13$ and $\lambda_3\lambda_4 = -\tfrac29$. They are the roots of $x^2+\tfrac13x-\tfrac29=0$, namely $\tfrac13$ and $-\tfrac23$. So the spectrum is $\{1, \tfrac13, -\tfrac23, -\tfrac23\}$.

### 2.4 The stationary distribution

A distribution $\boldsymbol\pi$ (row vector, non-negative, summing to 1) is **stationary** if
$$\boldsymbol\pi^\top P = \boldsymbol\pi^\top ,$$
i.e. one more step of the walk leaves it unchanged. Equivalently, $\boldsymbol\pi$ is a *left* eigenvector of $P$ with eigenvalue 1 (a right eigenvector of $P^\top$).

**Theorem (stationary distribution of an undirected graph).** For the random walk $P=D^{-1}S$ on an undirected graph with total weight $\mathrm{vol}(G) = \sum_i d_i > 0$,
$$\pi_i = \frac{d_i}{\mathrm{vol}(G)}$$
is a stationary distribution.

*Proof.* $(\boldsymbol\pi^\top P)_j = \sum_i \frac{d_i}{\mathrm{vol}}\cdot\frac{S_{ij}}{d_i} = \frac{1}{\mathrm{vol}}\sum_i S_{ij} = \frac{d_j}{\mathrm{vol}} = \pi_j$, using the symmetry $S_{ij}=S_{ji}$ in the last step. $\square$

The proof actually shows something stronger, called **detailed balance** or **reversibility**: $\pi_i P_{ij} = S_{ij}/\mathrm{vol} = \pi_j P_{ji}$. The probability flow from $i$ to $j$ equals the flow from $j$ to $i$.

In worked example 2.1, $\boldsymbol\pi = (3,3,3,1)/10 = (0.3,0.3,0.3,0.1)$. The poorly connected node 3 is visited least.

**Important consequence for link prediction.** If you let a walk run for a long time without restarts, *every* starting node ends up with the same scores $\propto$ degree. A drug that is similar to many drugs (a "hub") gets a high score for everything. This is **popularity (degree) bias**, and it is a recurring enemy in this chapter.

#### Existence, uniqueness and convergence (Perron–Frobenius)

Three properties decide whether "run the walk long enough" gives a unique answer:

* **Irreducible** — every node can reach every other node (the graph is connected). If not, each connected component has its own stationary distribution and the long-run behaviour depends on the start.
* **Aperiodic** — the walk does not cycle with a fixed period. A **bipartite** graph is periodic with period 2: from one side you always land on the other side at odd steps. Formally, $-1$ is then an eigenvalue of $P$.
* **Finite** — always true here.

The **Perron–Frobenius theorem** (for non-negative matrices) guarantees that an irreducible chain has a *unique* stationary distribution with all entries positive. If the chain is also aperiodic, then $\mathbf p_0^\top P^t \to \boldsymbol\pi^\top$ from *every* starting distribution.

**Speed of convergence.** Order the eigenvalues by absolute value, $1=\lambda_1 > |\lambda_2| \ge \dots$. Decomposing the starting distribution in the eigenbasis, every component other than the stationary one is multiplied by $\lambda_k$ at each step, so the error shrinks roughly like $|\lambda_2|^t$. The quantity $1-|\lambda_2|$ is the **spectral gap**: a large gap means fast mixing. In example 2.1, $|\lambda_2| = \tfrac23$; to shrink an error of order 1 to $10^{-12}$ needs about $\ln 10^{-12}/\ln\tfrac23 \approx 68$ steps (the code in Section 10 needs 67).

**Fixing periodicity: the lazy walk.** Replace $P$ by $P_{\text{lazy}} = \tfrac12(I+P)$: at each step, stay put with probability ½. Eigenvalues become $\tfrac12(1+\lambda)\in[0,1]$, so $-1$ turns into $0$ and the walk is aperiodic. (Notice the project's `fill_missing` puts 1 on the diagonal of every similarity matrix, and MBiRW's `sym_norm(logistic(S))` keeps those self-loops — self-loops also break periodicity.)

---

## 3. Restarts: PageRank, personalised PageRank, random walk with restart

### 3.1 PageRank

PageRank (Brin & Page, 1998) ranks web pages by the stationary distribution of a "random surfer" who follows links. A plain walk on the web graph has two problems:

1. **Dangling nodes** — pages with no out-links. Their rows of $P$ are all zeros, so $P$ is not row-stochastic. Standard fix: replace a dangling row by the uniform distribution $\tfrac1n\mathbf 1^\top$ (the surfer types a random URL).
2. **Traps and disconnected parts** — groups of pages with no links out (a reducible chain), where the walker gets stuck and the stationary distribution is not unique.

The fix for the second problem is **teleportation**: at every step, with probability $\beta$ (the **damping factor**, classically $0.85$) follow a link; with probability $1-\beta$ jump to a page chosen from a **teleport distribution** $\mathbf v$ (uniform, $\mathbf v = \tfrac1n\mathbf 1$, for classic PageRank). The resulting **Google matrix**
$$G = \beta P + (1-\beta)\,\mathbf 1\mathbf v^\top$$
is row-stochastic and has all entries strictly positive, so it is irreducible and aperiodic: a unique stationary distribution $\mathbf r$ exists and the power iteration converges to it.

**PageRank as a linear system.** From $\mathbf r^\top = \mathbf r^\top G$ and $\mathbf r^\top\mathbf 1 = 1$:
$$\mathbf r^\top = \beta\,\mathbf r^\top P + (1-\beta)\,\mathbf v^\top
\quad\Longrightarrow\quad
\mathbf r^\top (I-\beta P) = (1-\beta)\mathbf v^\top
\quad\Longrightarrow\quad
\mathbf r = (1-\beta)\,(I-\beta P^\top)^{-1}\mathbf v .$$
The inverse exists because $\rho(\beta P^\top) = \beta < 1$ (Section 5).

**Convergence speed.** One can show that the second eigenvalue of $G$ satisfies $|\lambda_2(G)|\le\beta$. So the power iteration's error shrinks at least like $\beta^t$ *regardless of the graph* — for $\beta=0.85$, about 140 iterations reach $10^{-10}$. This graph-independence is a big reason PageRank scales to billions of pages.

#### Worked example 3.1 (by hand): PageRank of a 3-page web

Pages A, B, C with links A→B, B→C, C→A and C→B. Rows of $P$ (order A, B, C): A: $(0,1,0)$; B: $(0,0,1)$; C: $(\tfrac12,\tfrac12,0)$. No dangling nodes.

*Without teleport* ($\beta=1$): $\pi_A = \tfrac12\pi_C$, $\pi_B = \pi_A+\tfrac12\pi_C$, $\pi_C = \pi_B$. So $\pi_B=\pi_C$ and $\pi_A = \tfrac12\pi_C$; normalising, $\boldsymbol\pi=(0.2,0.4,0.4)$.

*With $\beta = 0.85$, uniform teleport* ($\tfrac{1-\beta}{3}=0.05$):
$$r_A = 0.05 + 0.85\cdot\tfrac12 r_C,\qquad r_B = 0.05 + 0.85\,(r_A + \tfrac12 r_C),\qquad r_C = 0.05 + 0.85\,r_B.$$
Substitute the first into the second: $r_B = 0.05 + 0.85(0.05 + 0.425r_C) + 0.425r_C = 0.0925 + 0.78625\,r_C$. Then $r_C = 0.05 + 0.85(0.0925+0.78625r_C) = 0.128625 + 0.6683125\,r_C$, so $r_C = 0.128625/0.3316875 \approx 0.3878$. Back-substituting, $r_B\approx 0.3974$, $r_A\approx 0.2148$ (exactly $(380, 703, 686)/1769$). Teleportation pulls the scores towards uniform: A rises from 0.2 to 0.215.

### 3.2 Personalised PageRank and random walk with restart

Now change the teleport distribution from uniform to a single **seed** node $s$: $\mathbf v = \mathbf e_s$. Every restart sends the walker *home*. The stationary distribution of this walk is the **personalised PageRank** (PPR) vector of $s$; its entries measure how close each node is to $s$ — exactly what we want for "which drugs are near this disease?". In bioinformatics the same construction is called **random walk with restart** (RWR; e.g. Köhler et al. 2008 for disease-gene prioritisation, Tong, Faloutsos & Pan 2006 for fast computation).

We use the column-vector ("push") convention common in bioinformatics papers. Let $W = SD^{-1}$ (column-stochastic) and let $\mathbf e$ be the restart distribution (one seed, or several seeds with weights summing to 1). With probability $\alpha$ the walker takes a step, with probability $1-\alpha$ it restarts:
$$\boxed{\;\mathbf p_{t+1} = \alpha\,W\mathbf p_t + (1-\alpha)\,\mathbf e\;}$$

> **Notation warning.** Papers disagree on which probability gets the name. Some write $(1-c)W\mathbf p + c\,\mathbf e$ with $c$ the *restart* probability; PageRank calls the *continue* probability the damping factor $\beta$. In this chapter, as in the project's `MBiRW` code, **$\alpha$ is the weight on the walk step and $1-\alpha$ the restart weight**. Some papers describing MBiRW loosely call $\alpha$ a "restart probability"; read the formula, not the name.

#### Derivation of the closed form

Assume the iteration converges to a fixed point $\mathbf p^*$. Then
$$\mathbf p^* = \alpha W\mathbf p^* + (1-\alpha)\mathbf e
\iff (I-\alpha W)\mathbf p^* = (1-\alpha)\mathbf e
\iff \boxed{\;\mathbf p^* = (1-\alpha)\,(I-\alpha W)^{-1}\mathbf e\;}$$
provided $I-\alpha W$ is invertible. It is: the eigenvalues of $W$ lie in $[-1,1]$, so those of $I-\alpha W$ lie in $[1-\alpha, 1+\alpha]$, none of which is 0 when $\alpha<1$.

**The walk-counting interpretation (Neumann series).** For any square matrix $M$ with spectral radius $\rho(M)<1$,
$$(I-M)^{-1} = \sum_{t=0}^{\infty} M^t \qquad\text{(Neumann series).}$$
*Proof sketch:* $(I-M)\sum_{t=0}^{T-1}M^t = I - M^T$ (the sum telescopes), and $M^T\to 0$ when $\rho(M)<1$ (Section 5). $\square$ Applying it with $M=\alpha W$:
$$\mathbf p^* = (1-\alpha)\sum_{t=0}^\infty \alpha^t\,W^t\mathbf e .$$
$W^t\mathbf e$ is where a walker that started at the seed is after exactly $t$ steps, and $(1-\alpha)\alpha^t$ is the probability that a walk between two restarts lasts exactly $t$ steps (a geometric distribution). So **RWR = a mixture of $t$-step walks, weighted geometrically towards short walks**. The mean walk length between restarts is $\alpha/(1-\alpha)$: 0.43 steps for $\alpha=0.3$, 1 step for $\alpha=0.5$, 5.7 steps for $\alpha=0.85$. Small $\alpha$ = local, large $\alpha$ = global.

**Mass is conserved.** Since $W$ is column-stochastic, $\mathbf 1^\top W = \mathbf 1^\top$, so $\mathbf 1^\top\mathbf p_{t+1} = \alpha\mathbf 1^\top\mathbf p_t + (1-\alpha)\mathbf 1^\top\mathbf e$; if $\mathbf p_0$ and $\mathbf e$ both sum to 1, every $\mathbf p_t$ does, and so does $\mathbf p^*$.

**Limits.** As $\alpha\to0$, $\mathbf p^*\to\mathbf e$ (the walker never leaves home). As $\alpha\to1$, $\mathbf p^*\to\boldsymbol\pi$ (the degree-proportional stationary distribution — all personalisation is lost). Section 10's code shows this transition on the 4-drug graph.

#### Worked example 3.2 (by hand): RWR on a 3-node path

Path $0-1-2$ with unit weights, so $d=(1,2,1)$ and
$$W = SD^{-1} = \begin{pmatrix}0&\tfrac12&0\\1&0&1\\0&\tfrac12&0\end{pmatrix}\quad(\text{columns sum to }1).$$
Seed $\mathbf e = (1,0,0)^\top$, $\alpha=\tfrac12$. The fixed-point equations $\mathbf p = \tfrac12W\mathbf p + \tfrac12\mathbf e$ read
$$p_0 = \tfrac14p_1 + \tfrac12,\qquad p_1 = \tfrac12(p_0+p_2),\qquad p_2 = \tfrac14p_1 .$$
Substituting the first and third into the second: $p_1 = \tfrac12(\tfrac14p_1+\tfrac12+\tfrac14p_1) = \tfrac14p_1+\tfrac14$, so $p_1=\tfrac13$, $p_0 = \tfrac{7}{12}\approx0.583$, $p_2 = \tfrac1{12}\approx0.083$. The scores sum to 1 and decay with distance from the seed: $0.583 > 0.333 > 0.083$.

Truncating the Neumann series shows how the answer is built from walks of increasing length: with 1 term, $(\tfrac12,0,0)$; with 2 terms, $(\tfrac12,\tfrac14,0)$; with 5 terms, $(0.578, 0.313, 0.078)$; with 30 terms the exact answer.

### 3.3 Three normalisations, one core matrix

Which normalisation you choose changes *degree bias*. All three versions are built from the same symmetric "core" matrix $(D-\alpha S)^{-1}$:

| Variant | Iteration | Closed form | Equivalent |
|---|---|---|---|
| Push (column-stochastic RWR) | $\mathbf p\leftarrow\alpha SD^{-1}\mathbf p+(1-\alpha)\mathbf e$ | $(1-\alpha)(I-\alpha SD^{-1})^{-1}\mathbf e$ | $(1-\alpha)\,D\,(D-\alpha S)^{-1}\,\mathbf e$ |
| Pull (row-stochastic averaging) | $\mathbf f\leftarrow\alpha D^{-1}S\mathbf f+(1-\alpha)\mathbf y$ | $(1-\alpha)(I-\alpha D^{-1}S)^{-1}\mathbf y$ | $(1-\alpha)\,(D-\alpha S)^{-1}D\,\mathbf y$ |
| Symmetric | $\mathbf f\leftarrow\alpha D^{-1/2}SD^{-1/2}\mathbf f+(1-\alpha)\mathbf y$ | $(1-\alpha)(I-\alpha\tilde S)^{-1}\mathbf y$ | $(1-\alpha)\,D^{1/2}(D-\alpha S)^{-1}D^{1/2}\,\mathbf y$ |

(To verify the first row: $(I-\alpha SD^{-1})D = D-\alpha S$, so $(I-\alpha SD^{-1})^{-1} = D(D-\alpha S)^{-1}$. The others are analogous; Exercise 6 asks you to check them.)

Reading the table: the push version multiplies the result by $D$ on the *output* side, so **high-degree nodes receive high scores** (degree bias, the PageRank behaviour). The pull version multiplies by $D$ on the *input* side, so a node's score is a weighted average and is not inflated by its own degree. The symmetric version splits the difference. A pleasant corollary: since the core is symmetric, the push version satisfies $p_s(i)/d_i = p_i(s)/d_s$ — personalised PageRank is symmetric once divided by degree.

---

## 4. Label propagation

### 4.1 The semi-supervised setting

Suppose a few nodes carry labels (for instance "treated by drug X": yes) and most do not. **Label propagation** spreads the known labels along the edges so that every node gets a soft label. The assumption is the **cluster** (or **smoothness**) assumption: strongly connected nodes tend to share labels.

Encode labels in a matrix $Y\in\mathbb R^{n\times c}$: $Y_{ic}=1$ if node $i$ is known to have label $c$, else 0. We seek a score matrix $F\in\mathbb R^{n\times c}$ and predict $\arg\max_c F_{ic}$ (or rank nodes by $F_{ic}$ for each $c$).

**Why this is our problem.** Take the drug graph and let the "labels" be the diseases: $Y = A$ (drugs × diseases). Column $j$ of $A$ says which drugs are known to treat disease $j$. Propagating the labels along drug similarity gives every drug a score for every disease — a drug-repositioning predictor. The same works on the disease graph with $Y=A^\top$. Label propagation and RWR are the same computation viewed column-wise: RWR handles one seed vector, label propagation handles a whole matrix of seed vectors at once.

### 4.2 Zhou et al. (2004): learning with local and global consistency

Let $\tilde S = D^{-1/2}SD^{-1/2}$ (with $S_{ii}=0$). Iterate
$$F^{(t+1)} = \alpha\,\tilde S F^{(t)} + (1-\alpha)\,Y, \qquad F^{(0)}=Y, \quad 0<\alpha<1 .$$
Each node receives information from its neighbours (first term) while keeping some of its initial label (second term).

**Fixed point from the iteration.** Unrolling,
$$F^{(t)} = (\alpha\tilde S)^t Y + (1-\alpha)\sum_{k=0}^{t-1}(\alpha\tilde S)^k Y .$$
Because $\rho(\alpha\tilde S) = \alpha < 1$, the first term vanishes and the sum tends to $(I-\alpha\tilde S)^{-1}$:
$$\boxed{\;F^* = (1-\alpha)\,(I-\alpha\tilde S)^{-1}\,Y\;}$$
(The factor $1-\alpha$ does not change rankings and is often dropped.)

**Fixed point from an objective.** Zhou et al. show $F^*$ minimises
$$\mathcal Q(F) = \frac12\sum_{i,j} S_{ij}\left\lVert \frac{F_i}{\sqrt{d_i}} - \frac{F_j}{\sqrt{d_j}}\right\rVert^2 + \mu\sum_i \lVert F_i - Y_i\rVert^2 ,$$
where $F_i$ is row $i$. The first term (**smoothness**) penalises different scores on strongly connected nodes; the second (**fitting**) keeps scores close to the known labels. To minimise, first rewrite the smoothness term. Expanding the square,
$$\frac12\sum_{ij}S_{ij}\left(\frac{\lVert F_i\rVert^2}{d_i} - \frac{2F_i^\top F_j}{\sqrt{d_id_j}} + \frac{\lVert F_j\rVert^2}{d_j}\right) = \sum_i\lVert F_i\rVert^2 - \sum_{ij}\tilde S_{ij}F_i^\top F_j = \mathrm{tr}\!\big(F^\top(I-\tilde S)F\big),$$
using $\sum_j S_{ij}/d_i = 1$. The matrix $\mathcal L = I-\tilde S$ is the **normalised graph Laplacian** (you met it in Unit A2). So
$$\mathcal Q(F) = \mathrm{tr}(F^\top\mathcal LF) + \mu\lVert F-Y\rVert_F^2 .$$
It is a convex quadratic (since $\mathcal L$ is positive semi-definite). Setting the gradient to zero:
$$2\mathcal LF + 2\mu(F-Y) = 0 \iff F - \tilde SF + \mu F = \mu Y \iff F = \frac{1}{1+\mu}\tilde SF + \frac{\mu}{1+\mu}Y .$$
With $\alpha = 1/(1+\mu)$ this is exactly the fixed point of the iteration: $F^* = (1-\alpha)(I-\alpha\tilde S)^{-1}Y$. So **label propagation = graph-Laplacian-regularised smoothing of the labels**. You will meet the same Laplacian penalty $\mathrm{tr}(U^\top LU)$ in SCMFDD (Chapter 10) — there it smooths latent factors instead of scores.

### 4.3 Zhu & Ghahramani (2002): clamped propagation and harmonic functions

The earlier algorithm of Zhu and Ghahramani treats known labels as **hard constraints**. Split nodes into labelled $L$ and unlabelled $U$. Repeat: (1) $F\leftarrow PF$ with $P=D^{-1}S$ (every node takes the average of its neighbours); (2) reset ("clamp") the labelled rows to $Y_L$.

At convergence the unlabelled scores are **harmonic**: each is the weighted average of its neighbours, $f_i = \sum_j P_{ij}f_j$ for $i\in U$. Writing the graph Laplacian $L = D-S$ in blocks, harmonicity means $(LF)_U = 0$, i.e. $L_{UU}F_U + L_{UL}F_L = 0$, so
$$F_U = L_{UU}^{-1}\,S_{UL}\,Y_L = (D_{UU}-S_{UU})^{-1}S_{UL}Y_L$$
(using $L_{UL} = -S_{UL}$ off the diagonal). This is the minimiser of the energy $\tfrac12\sum_{ij}S_{ij}(f_i-f_j)^2$ subject to $f_L=y_L$.

**Random-walk meaning.** $F_{ic}$ is the probability that a random walk started at unlabelled node $i$ hits a labelled node of class $c$ *before* any other labelled node (labelled nodes are **absorbing** states).

| | Zhou et al. (2004) | Zhu & Ghahramani (2002) |
|---|---|---|
| Labels | soft: fitted with weight $\mu$ | hard: clamped |
| Normalisation | symmetric $\tilde S$ | row-stochastic $P$ |
| Solution | $(1-\alpha)(I-\alpha\tilde S)^{-1}Y$ | $L_{UU}^{-1}S_{UL}Y_L$ |
| Can correct a noisy label? | yes | no |
| Random-walk view | walk with restart | absorbing walk |

#### Worked example 4.1 (by hand): harmonic labels on two triangles

Nodes $\{0,1,2\}$ form a triangle, $\{3,4,5\}$ another, and an edge $2$–$3$ bridges them. Node 0 is labelled A, node 5 is labelled B. Let $f_i$ be the probability of class A. Harmonicity:
$f_1=\tfrac12(f_0+f_2)=\tfrac12(1+f_2)$, $f_2=\tfrac13(f_0+f_1+f_3)$, $f_3=\tfrac13(f_2+f_4+f_5)$, $f_4=\tfrac12(f_3+f_5)=\tfrac12f_3$.
By the left–right symmetry, $f_3 = 1-f_2$ and $f_4 = 1-f_1$. Then $3f_2 = 1 + f_1 + 1 - f_2$, i.e. $4f_2 = 2 + f_1 = 2+\tfrac12(1+f_2)$, so $f_2 = \tfrac57\approx0.714$ and $f_1 = \tfrac67\approx0.857$. Nodes 1, 2 lean A; nodes 3, 4 lean B; the bridge nodes are the least certain — exactly the intuition.

One step of Zhou's iteration on the same graph with $\alpha=0.9$: $\tilde S_{10} = 1/\sqrt{d_1d_0} = 1/\sqrt{2\cdot 2} = 0.5$, so $F^{(1)}_1 = 0.9\cdot0.5\cdot(1,0) = (0.45, 0)$; $\tilde S_{20} = 1/\sqrt{3\cdot2}\approx0.408$, so $F^{(1)}_2 \approx (0.367, 0)$; and node 0 keeps $(1-\alpha)Y_0 = (0.1, 0)$. Section 10 runs both algorithms to convergence.

---

## 5. Convergence: the spectral radius decides everything

Every algorithm so far is an iteration of the form
$$\mathbf x_{t+1} = M\mathbf x_t + \mathbf b$$
(RWR: $M=\alpha W$; label propagation: $M=\alpha\tilde S$ applied column by column; PageRank: $M = \beta P^\top$). The single number that governs it is the **spectral radius**
$$\rho(M) = \max_k |\lambda_k(M)| .$$

**Theorem.** The iteration $\mathbf x_{t+1} = M\mathbf x_t+\mathbf b$ converges for every starting point $\mathbf x_0$ if and only if $\rho(M)<1$. The limit is $\mathbf x^* = (I-M)^{-1}\mathbf b$, and the error obeys $\mathbf x_t - \mathbf x^* = M^t(\mathbf x_0-\mathbf x^*)$.

*Proof.* If a limit $\mathbf x^*$ exists it satisfies $\mathbf x^*=M\mathbf x^*+\mathbf b$. Subtracting from the iteration, the error $\mathbf e_t = \mathbf x_t-\mathbf x^*$ obeys $\mathbf e_{t+1} = M\mathbf e_t$, so $\mathbf e_t = M^t\mathbf e_0$. It remains to show $M^t\to0$ iff $\rho(M)<1$.
($\Leftarrow$) Write $M$ in Jordan form $M = QJQ^{-1}$; $J^t$ has entries of the form $\binom{t}{k}\lambda^{t-k}$, each of which tends to 0 when $|\lambda|<1$ (the geometric decay beats the polynomial growth). Equivalently, by **Gelfand's formula** $\rho(M) = \lim_{t\to\infty}\lVert M^t\rVert^{1/t}$, so for any $\rho(M)<q<1$, $\lVert M^t\rVert\le q^t$ for large $t$.
($\Rightarrow$) If $|\lambda|\ge1$ for some eigenvalue with eigenvector $\mathbf v$, starting with $\mathbf e_0=\mathbf v$ gives $\lVert\mathbf e_t\rVert = |\lambda|^t\lVert\mathbf v\rVert\not\to0$. $\square$

**Practical bound.** For any induced matrix norm, $\rho(M)\le\lVert M\rVert$, and $\lVert\mathbf e_t\rVert\le\lVert M\rVert^t\lVert\mathbf e_0\rVert$. This gives quick guarantees:

* Row-stochastic $P$: $\lVert\alpha P\rVert_\infty = \alpha$ (maximum absolute row sum).
* Column-stochastic $W$: $\lVert\alpha W\rVert_1 = \alpha$ (maximum absolute column sum).
* Symmetric $\tilde S$: $\lVert\alpha\tilde S\rVert_2 = \alpha\,\rho(\tilde S) = \alpha$.

In all three cases $\lVert\mathbf e_t\rVert\le\alpha^t\lVert\mathbf e_0\rVert$, so reaching tolerance $\varepsilon$ needs at most about
$$t \approx \frac{\ln\varepsilon}{\ln\alpha}\quad\text{iterations}$$
(16 for $\alpha=0.3$ and $\varepsilon=10^{-8}$; 175 for $\alpha=0.9$; 1,833 for $\alpha=0.99$). On well-connected graphs it is often far fewer, because the error components along small eigenvalues die faster than $\alpha^t$.

**Why normalisation is not optional.** A raw similarity matrix has spectral radius roughly equal to its average degree. On a ring where every node has two neighbours with weight 1, $\rho(S)=2$; then $\alpha S$ with $\alpha=0.6$ has $\rho = 1.2>1$ and the iteration explodes (Section 10 shows it). Normalising brings the spectral radius to 1, and any $\alpha<1$ is then safe.

**Matrix-valued iterations with two sides.** MBiRW updates a matrix by multiplying on the left and on the right (Section 6). The linear map $R\mapsto \tfrac\alpha2(M_rR + RM_d)$ acts on $\mathrm{vec}(R)$ as the **Kronecker sum** $\tfrac\alpha2(I\otimes M_r + M_d^\top\otimes I)$, whose eigenvalues are $\tfrac\alpha2(\lambda_i+\mu_j)$ for eigenvalues $\lambda_i$ of $M_r$ and $\mu_j$ of $M_d$. Both lie in $[-1,1]$, so $|\tfrac\alpha2(\lambda_i+\mu_j)|\le\alpha<1$: the bi-random walk always converges.

**Speed versus meaning.** A large $\alpha$ (or many steps) costs iterations *and* changes what you compute: the scores drift towards the stationary distribution, which carries no information about the seed. Choosing $\alpha$ is a modelling decision first and a numerical one second.

---

## 6. The bi-random walk of MBiRW

### 6.1 The idea

The **bi-random walk** (BiRW) was introduced by Xie, Hwang and Kuang (2012) to prioritise disease genes using a gene network and a phenotype network joined by known gene–phenotype links. **MBiRW** (Luo et al., *Bioinformatics* 2016) applied it to drug repositioning on exactly the Fdataset benchmark used in this project. The idea: there are two similarity networks (drugs, diseases) joined by a bipartite association matrix. Instead of walking on one big graph, MBiRW keeps a score matrix $R$ (drugs × diseases) and updates it by walking along *both* networks:

```
               drug network M_r                     disease network M_d
        (drug i  ~  drug i')                    (disease j' ~ disease j)
               |                                         |
   LEFT walk:  R <- alpha * M_r @ R + (1-alpha) * A0     |
   "drug i borrows the scores of similar drugs i'        |
    for the same disease j"                              |
                                         RIGHT walk:  R <- alpha * R @ M_d + (1-alpha) * A0
                                         "disease j borrows the scores of
                                          similar diseases j' for the same drug i"
```

The names come from **which side of $R$ the similarity matrix multiplies**: $M_rR$ (left) mixes rows, i.e. moves along the drug network; $RM_d$ (right) mixes columns, i.e. moves along the disease network. Entry-wise:
$$(M_rR)_{ij} = \sum_{i'}(M_r)_{ii'}R_{i'j},\qquad (RM_d)_{ij} = \sum_{j'}R_{ij'}(M_d)_{j'j}.$$

### 6.2 Step 1: similarity adjustment with a logistic function

Raw similarities are numerous and mostly small. If you sum hundreds of small similarities, they swamp the few strong ones that actually carry the signal. MBiRW therefore passes every similarity through a steep logistic curve,
$$L(x) = \frac{1}{1+e^{cx+d}},\qquad c=-15,\quad d=\ln 9999 .$$
Some values: $L(0) = 1/(1+9999) = 0.0001$; $L(0.3)\approx0.009$; the midpoint $L=0.5$ is at $x = d/15 = \ln(9999)/15\approx0.614$; $L(0.8)\approx0.942$; $L(1)\approx0.997$. This is a **soft threshold**: similarities below about 0.5 are essentially switched off and those above about 0.7 are pushed close to 1. The project's kNN sparsification ($k=10$) solves the same problem with a hard, per-node cut instead of a global soft one.

(The original paper also builds "comprehensive" similarities: it uses the known associations to down-weight non-discriminative similarities and clusters drugs with the ClusterONE graph-clustering algorithm to strengthen within-cluster similarity. The project's re-implementation keeps only the logistic step; see its docstring.)

### 6.3 Step 2: normalisation and the initial matrix

The adjusted matrices are symmetrically normalised, $M_r = D_r^{-1/2}L(S_r)D_r^{-1/2}$ and $M_d = D_d^{-1/2}L(S_d)D_d^{-1/2}$ (the project's `sym_norm`). The initial score matrix is the known association matrix scaled to sum to 1, $A_0 = A/\sum_{ij}A_{ij}$ — the matrix analogue of a restart distribution.

### 6.4 Step 3: the bi-random walk iteration

For $t = 1,\dots,\max(l,r)$:
$$R_L = \alpha M_rR_{t-1} + (1-\alpha)A_0 \quad(\text{only if } t\le l),\qquad
R_R = \alpha R_{t-1}M_d + (1-\alpha)A_0\quad(\text{only if } t\le r),$$
$$R_t = \text{average of the walks still active}.$$
Each walk is an RWR step: a fraction $\alpha$ of the new score comes from neighbours, the rest $1-\alpha$ is re-injected from the known associations.

**What $\alpha$ trades off.** $\alpha$ balances **neighbour evidence** (propagation, generalisation) against **fidelity to the known links** (restart). With the project's default $\alpha=0.3$, 70% of each step's score comes straight from $A_0$, so the walk stays very local. Larger $\alpha$ spreads evidence further and helps sparse rows and columns, but blurs everything towards degree-driven, popularity-like scores.

**What $l$ and $r$ do.** They are the maximum numbers of steps of the left walk and the right walk. With $l=r=2$ (the defaults recommended by the MBiRW authors and used in the project, together with $\alpha=0.3$), each walk takes two steps, so evidence travels at most two hops on each network. Choosing $l\neq r$ lets one network be explored further than the other — useful when one similarity is denser or more trustworthy. In the code, once the shorter walk has used up its steps, only the longer walk continues (the `parts` list then has one element).

**The infinite-step limit.** If $l=r\to\infty$, the update is $R\leftarrow\tfrac\alpha2(M_rR+RM_d)+(1-\alpha)A_0$, which converges (Section 5) to the solution of the **Sylvester equation**
$$\Big(\tfrac\alpha2M_r-\tfrac12I\Big)R + R\Big(\tfrac\alpha2M_d-\tfrac12I\Big) = -(1-\alpha)A_0 .$$
With $\alpha=0.3$ the operator's spectral radius is at most 0.3, so after two steps the iteration is already close to this limit (Exercise 9 compares them).

#### Worked example 6.1 (by hand): one right step for a cold disease

Toy data: 4 drugs, 3 diseases; disease 2 has no known drug.
$$A=\begin{pmatrix}1&0&0\\1&1&0\\0&0&0\\0&1&0\end{pmatrix},\qquad
S_d=\begin{pmatrix}1&0.2&0.3\\0.2&1&0.9\\0.3&0.9&1\end{pmatrix}.$$
Logistic adjustment: $L(0.2)\approx0.002$, $L(0.3)\approx0.009$, $L(0.9)\approx0.987$, $L(1)\approx0.997$. Symmetric normalisation (row sums of $L(S_d)$: 1.008, 1.985, 1.992) gives
$$M_d\approx\begin{pmatrix}0.989&0.001&0.006\\0.001&0.502&0.496\\0.006&0.496&0.500\end{pmatrix}.$$
$A$ has 4 ones, so $A_0 = A/4$. One right step for column 2 (the cold disease), drug 1 (which treats diseases 0 and 1):
$$\alpha\,(A_0M_d)_{1,2} = 0.3\times(0.25\times0.006 + 0.25\times0.496) = 0.3\times0.1256\approx0.0377 .$$
Drug 3 (treats disease 1 only): $0.3\times0.25\times0.496\approx0.0372$. Drug 0 (treats disease 0 only, which is barely similar to disease 2): $0.3\times0.25\times0.006\approx0.0005$. Drug 2: 0. The cold disease inherits the drugs of its close neighbour, disease 1, as intended.

**The left walk cannot do this.** $(M_rR)_{:,2} = M_rR_{:,2}$: if column 2 of $R$ is all zeros, so is column 2 of $M_rR$. Left walks only redistribute scores *within* a column, among similar drugs. Only the right walk can move information *into* an empty column. After the first step has filled column 2 a little, the second left step can then spread it among similar drugs. This asymmetry is the heart of the cold-start story (Section 8).

---

## 7. Random walks on heterogeneous networks (NRWRH)

A different way to use two similarity networks is to glue them into **one** graph whose nodes are drugs *and* diseases (or, in the original paper, drugs and protein targets), and run a single RWR on it. This is **NRWRH** (Network-based Random Walk with Restart on the Heterogeneous network; Chen, Liu & Yan, *Molecular BioSystems* 2012), proposed for drug–target interaction prediction; the same construction has been used for drug–disease prediction.

**The heterogeneous transition matrix.** Order the nodes as (drugs, diseases). A walker on drug $i$ that has at least one known link **jumps** to the disease side with probability $\lambda$ (the **jumping probability**) and stays on the drug side with probability $1-\lambda$; a drug with no known link always stays. In block form, with row-normalisation inside each block,
$$M=\begin{pmatrix}M_{RR}&M_{RD}\\M_{DR}&M_{DD}\end{pmatrix},\qquad
\begin{aligned}
(M_{RD})_{ij} &= \lambda\,\frac{A_{ij}}{\sum_{j'}A_{ij'}},\\
(M_{RR})_{ii'} &= (1-\lambda)\,\frac{(S_r)_{ii'}}{\sum_{i''}(S_r)_{ii''}}\ \ (\text{or without the factor }1-\lambda\text{ if drug }i\text{ has no link}),
\end{aligned}$$
and symmetrically for the disease rows. Every row sums to 1.

**Seeds on both sides.** To score candidate drugs for disease $j$, NRWRH restarts to *two* kinds of seeds: the disease itself ($\mathbf v_0 = \mathbf e_j$ on the disease side) and its known drugs ($\mathbf u_0$ uniform over them on the drug side), mixed with a weight $\eta\in[0,1]$:
$$\mathbf p_0 = \begin{pmatrix}(1-\eta)\,\mathbf u_0\\ \eta\,\mathbf v_0\end{pmatrix},\qquad
\mathbf p_{t+1} = (1-c)\,M^\top\mathbf p_t + c\,\mathbf p_0 ,$$
with restart probability $c$. Iterate until $\lVert\mathbf p_{t+1}-\mathbf p_t\rVert$ is tiny, and rank drugs by their entries of the stationary $\mathbf p$. (For a cold disease, $\mathbf u_0$ is empty and all restart mass goes to the disease seed.)

**BiRW versus NRWRH.**

| | BiRW / MBiRW | NRWRH |
|---|---|---|
| State | a score *matrix* $R$ (all pairs at once) | a probability *vector* on drugs + diseases (one query at a time) |
| Crossing between types | implicit: $A_0$ is the restart | explicit: jump with probability $\lambda$ along known links |
| Parameters | $\alpha$, $l$, $r$ | restart $c$, jump $\lambda$, seed mix $\eta$ |
| Cost | two matrix products per step for *all* pairs | one walk per query disease (or drug) |

On the toy data of Section 6 both methods rank the drugs for the cold disease in the same order (1, 3, 0, 2) — see Section 10.

---

## 8. One-step kNN propagation — the project's multi-view propagation head

### 8.1 Definition

For a similarity view $S$ (drugs or diseases) and $k=10$, the project builds a **kNN kernel** $K$ (`knn_kernel` in `src/drepo/data.py`):

1. keep, in each row $i$, only the $k$ largest similarities to *other* nodes (and never a zero similarity);
2. set the diagonal to zero;
3. divide each row by its sum (rows with no neighbour stay all-zero).

So $K_{ii'} = S_{ii'}/\sum_{i''\in N_k(i)}S_{ii''}$ for $i'\in N_k(i)$, the $k$ nearest neighbours of $i$, and 0 otherwise. $K$ is row-stochastic (on rows that have neighbours) but **not symmetric**: $i'$ may be among $i$'s ten nearest without $i$ being among $i'$'s.

The propagation scores for one view are then

* **drug view** $v$: $\;P_v = K_vA$, i.e. $P_v[i,j] = \sum_{i'\in N_k(i)}K_v[i,i']\,A[i',j]$ — *"the similarity-weighted share of drug $i$'s 10 nearest drugs that treat disease $j$"*;
* **disease view** $u$: $\;P_u = (K_uA^\top)^\top = AK_u^\top$, i.e. $P_u[i,j] = \sum_{j'\in N_k(j)}K_u[j,j']\,A[i,j']$ — *"how strongly disease $j$'s 10 nearest diseases are treated by drug $i$"*.

Every entry lies in $[0,1]$. In the language of this chapter, $P_v$ is **one step of "pull" label propagation** on a kNN-sparsified graph, with no restart, and with the node itself excluded. In the language of recommender systems (Chapter 10) it is **item-based** (drug view) or **user-based** (disease view) **neighbourhood collaborative filtering** — but with neighbours defined by chemistry or phenotype rather than by co-occurrence in $A$.

#### Worked example 8.1 (by hand)

Same toy as before, drug similarity
$$S_r=\begin{pmatrix}1&0.8&0.3&0.1\\0.8&1&0.2&0.1\\0.3&0.2&1&0.7\\0.1&0.1&0.7&1\end{pmatrix}.$$
With $k=2$: drug 0's two nearest others are drugs 1 (0.8) and 2 (0.3), so $K_r[0,:] = (0, \tfrac{0.8}{1.1}, \tfrac{0.3}{1.1}, 0) = (0, 0.727, 0.273, 0)$. Then
$$P_r[0,0] = 0.727\cdot A[1,0] + 0.273\cdot A[2,0] = 0.727,\qquad P_r[0,1] = 0.727\cdot A[1,1] = 0.727,\qquad P_r[0,2] = 0 .$$
The disease view with $k=1$: disease 2's nearest other disease is disease 1 (0.9), so $K_d[2,:]=(0,1,0)$ and $P_d[i,2] = A[i,1]$: drugs 1 and 3 get score 1, drugs 0 and 2 get 0. Note also that diseases 0 and 1 both have disease 2 as their single nearest neighbour; since disease 2 has no links, $P_d[:,0]=P_d[:,1]=0$ — **a neighbour without links contributes nothing**, which is one reason $k$ should not be too small.

### 8.2 Why the diagonal is zeroed

If $K_{ii}>0$, then $P[i,j]$ would contain $A[i,j]$ itself: the score of a pair would include the answer. During training, that teaches the model the shortcut "score high if the link is already visible" (Section 10 of `docs/HOW_IT_WORKS.md`: training on visible links dropped validation AUC to 0.71). Zeroing the diagonal makes each score depend only on *other* drugs' (or diseases') links. A neat side-effect: for a disease view, column $j$ of $AK_u^\top$ never reads column $j$ of $A$, so computing it once on the full matrix gives exactly the leave-one-disease-out scores for *every* disease simultaneously. Section 10 uses this trick.

### 8.3 Why propagation is so strong in cold start

In leave-one-disease-out (LODO) evaluation, disease $j$'s whole column of $A$ is hidden. Go through the score components one by one:

| Component | Score for cold disease $j$ | Why |
|---|---|---|
| Drug-view propagation $K_vA$ | **0 for every drug** | $(K_vA)_{:,j} = K_vA_{:,j} = K_v\mathbf 0$ |
| Left walk of MBiRW | 0 (until the right walk fills the column) | same reason |
| Matrix-factorisation factor of disease $j$ | untrained (only pulled by regularisers) | the loss has no observed entry in column $j$ |
| GNN bilinear term | weak | disease $j$ has no `assoc` edges; only its similarity relations inform its embedding; the degree gate shrinks the term |
| **Disease-view propagation $AK_u^\top$** | **informative** | it reads the columns of $j$'s neighbours, which are fully observed |
| Right walk of MBiRW | informative | same mechanism, plus restart |

So in cold start, disease-side propagation is not just *a* signal: it is almost *the only* signal. It is also:

* **Parameter-free** — nothing to overfit with 1,933 links, and nothing that needs the cold disease's own links to be trained.
* **Aligned with how the benchmark was built** — diseases in Fdataset are defined by OMIM phenotypes, and phenotypically similar diseases (e.g. epilepsy subtypes) genuinely share treatments, so the homophily assumption holds strongly.
* **Hard to beat in general** — in recommender systems research, carefully tuned nearest-neighbour baselines have repeatedly matched or beaten complex neural models (Ferrari Dacrema et al., 2019).

Real numbers, computed in Section 10 on Fdataset with full leave-one-disease-out (all 313 diseases; random AUPR = 0.010):

| Cold-start scorer (Fdataset, LODO, pooled) | AUPR | AUC |
|---|---|---|
| Drug-view propagation (any drug view) | 0.010 (all scores 0) | 0.5 |
| One-step `gene_d` | 0.053 | 0.584 |
| One-step `sem_mondo` | 0.111 | 0.643 |
| One-step `pheno_mim` | 0.174 | 0.738 |
| One-step `pheno_mim` + `sem_mondo` | 0.191 | 0.753 |
| MBiRW ($\alpha=0.3$, $l=r=2$, CDK + `pheno_mim`) | 0.214 | 0.786 |

The project's documentation reports a slightly different *validation cold-start proxy* (a held-out subset used during tuning): 0.149 (`pheno_mim`), 0.106 (`sem_mondo`), 0.175 (both), 0.174 (MBiRW). The protocols differ, so the absolute numbers differ, but the conclusions are the same: disease-side propagation is 10–20× better than random, combining the two disease views helps, and MBiRW is a strong cold-start baseline.

### 8.4 Why *more* steps make cold start worse

If one step is good, are two better? Section 10 tests it on Fdataset with `pheno_mim`:

| Propagation steps ($AK^{t\top}$) | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| Cold-start AUPR | **0.174** | 0.119 | 0.110 | 0.076 |

Each extra step multiplies by $K$ again, so the rows of $K^t$ drift towards the stationary distribution of the kNN walk (Section 2.4). Two hops away, a neighbour-of-a-neighbour is much less related to the cold disease, and diseases that are central in the kNN graph (or that have many drugs) start to dominate every ranking. This is the same **over-smoothing** that limits the depth of GNNs (Chapter 11), and it is why MBiRW uses a small $\alpha$ and only two steps.

### 8.5 How the project combines the views

MV-HGAT computes one $P_v$ per view (three drug views, three disease views), stacks them, and adds
$$\sum_v w_v\,P_v[i,j],\qquad w_v = \text{softplus}(\theta_v)\times s \ge 0$$
to the logit, where $\theta_v$ and the scale $s$ are learned. Because the term is additive and $w_v\ge0$, $w_vP_v[i,j]$ is literally view $v$'s share of the logit — a faithful explanation. During training the $P_v$ are recomputed every epoch from the *visible* links only (some links are hidden, and 10% of diseases are made fully cold), so the weights $w_v$ learn what to do in exactly the situation they will face at test time. The **degree gate** multiplies the GNN term by $\sigma(a+b\log(1+\deg_i))\,\sigma(c+d\log(1+\deg_j))$, so for a disease with no visible links the propagation head dominates.

---

## 9. From propagation to graph neural networks

Random walks and label propagation are not a separate world from GNNs; they are the special case with **no learned weights**:

| Method | Update | Learned? |
|---|---|---|
| Label propagation | $F\leftarrow\alpha\tilde SF+(1-\alpha)Y$ | nothing |
| GCN layer (Kipf & Welling 2017) | $H\leftarrow\sigma(\hat SH\Theta)$, $\hat S$ = symmetric-normalised $S+I$ | $\Theta$ |
| SGC (Wu et al. 2019) | $\hat S^KX\Theta$ (propagate $K$ times, then one linear layer) | $\Theta$ |
| APPNP (Klicpera et al. 2019) | $Z = (1-\alpha)(I-\alpha\hat S)^{-1}f_\theta(X)$, "predict then propagate" with personalised PageRank | $f_\theta$ |
| Correct & Smooth (Huang et al. 2021) | simple predictor, then label propagation of its errors and of the labels | the simple predictor |
| **MV-HGAT (this project)** | GNN term **+** one-step label propagation per view | GNN, $W$, $w_v$, gate |

Two lessons carry over to the GNN chapters. First, the **normalisation** in a GCN layer ($D^{-1/2}(S+I)D^{-1/2}$) is the symmetric normalisation of Section 2.3, and for the same reasons (bounded spectrum, controlled degree bias). Second, **over-smoothing** — node embeddings becoming indistinguishable after many layers — is the convergence of $\hat S^t$ towards its stationary, rank-one limit. The Correct & Smooth paper showed that adding plain label propagation on top of a simple model can match or beat much larger GNNs; our propagation head is the same insight applied to drug repositioning.

---

## 10. Code: everything above, runnable

All examples use only NumPy, SciPy and scikit-learn and run in a second on a CPU. Run them with the project environment:

```powershell
cd "C:\Users\Abhineet Anand\Desktop\DrugRepositioning"
.venv\Scripts\python.exe example.py
```

Every output shown below was produced by running the code exactly as printed. (Tiny differences in the last digit can occur on other machines or library versions.)

### 10.1 Markov chain, stationary distribution and the three normalisations

This reproduces worked example 2.1: the walk from node 3, power iteration to the stationary distribution, the degree formula, the eigenvector method, and the fact that row- and symmetric normalisation have the same eigenvalues.

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# A small undirected, weighted graph on 4 nodes (think: 4 drugs, weights = similarity)
S = np.array([[0, 2, 1, 0],
              [2, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], dtype=float)
d = S.sum(1)                       # weighted degrees
P = S / d[:, None]                 # row-normalised transition matrix D^-1 S
print("degrees:", d)
print("P =\n", P)
print("row sums:", P.sum(1))

# Evolve a distribution that starts on node 3
p = np.array([0, 0, 0, 1.0])
for t in range(1, 4):
    p = p @ P                      # p_{t}^T = p_{t-1}^T P
    print(f"t={t}: p = {p}")

# Power iteration to the stationary distribution
p = np.full(4, 0.25)
for t in range(200):
    p_new = p @ P
    if np.abs(p_new - p).sum() < 1e-12:
        break
    p = p_new
print("power iteration converged after", t, "steps:", p)
print("degree formula  d / sum(d):        ", d / d.sum())

# Same thing from the left eigenvector of P with eigenvalue 1
w, V = np.linalg.eig(P.T)
pi = np.real(V[:, np.argmin(np.abs(w - 1))]); pi /= pi.sum()
print("eigenvector method:                ", pi)
print("eigenvalues of P:", np.sort(np.real(w))[::-1])

# Symmetric normalisation has the SAME eigenvalues
Dm = np.diag(1 / np.sqrt(d))
Ssym = Dm @ S @ Dm
print("eigenvalues of D^-1/2 S D^-1/2:", np.sort(np.linalg.eigvalsh(Ssym))[::-1])
```

Output:

```text
degrees: [3. 3. 3. 1.]
P =
 [[0.     0.6667 0.3333 0.    ]
 [0.6667 0.     0.3333 0.    ]
 [0.3333 0.3333 0.     0.3333]
 [0.     0.     1.     0.    ]]
row sums: [1. 1. 1. 1.]
t=1: p = [0. 0. 1. 0.]
t=2: p = [0.3333 0.3333 0.     0.3333]
t=3: p = [0.2222 0.2222 0.5556 0.    ]
power iteration converged after 67 steps: [0.3 0.3 0.3 0.1]
degree formula  d / sum(d):         [0.3 0.3 0.3 0.1]
eigenvector method:                 [0.3 0.3 0.3 0.1]
eigenvalues of P: [ 1.      0.3333 -0.6667 -0.6667]
eigenvalues of D^-1/2 S D^-1/2: [ 1.      0.3333 -0.6667 -0.6667]
```

Things to notice: the hand-computed $\mathbf p_3 = (\tfrac29,\tfrac29,\tfrac59,0)$ appears at $t=3$; three different methods agree on $\boldsymbol\pi = (0.3,0.3,0.3,0.1)$; the second-largest eigenvalue modulus is $\tfrac23$, which is why power iteration needed 67 steps to reach $10^{-12}$.

### 10.2 PageRank with a dangling node, and personalised PageRank

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Directed graph, edges i -> j.  Node 4 has no out-links (a "dangling" node);
# node 3 has no in-links.
edges = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 0), (3, 2)]
n = 5
Adj = np.zeros((n, n))
for i, j in edges:
    Adj[i, j] = 1
out = Adj.sum(1)
# row-normalise; a dangling row is replaced by the uniform distribution
P = np.where(out[:, None] > 0, Adj / np.maximum(out, 1)[:, None], 1 / n)
beta = 0.85                                   # follow a link w.p. beta, teleport w.p. 1-beta
G = beta * P + (1 - beta) / n                 # the "Google matrix": row-stochastic, all > 0

def pagerank_power(G, tol=1e-12):
    r = np.full(len(G), 1 / len(G))
    for it in range(1000):
        r_new = r @ G
        if np.abs(r_new - r).sum() < tol:
            return r_new, it + 1
        r = r_new

r, iters = pagerank_power(G)
print("PageRank (power iteration):", r, "iterations:", iters)

# Because P is row-stochastic, r^T (I - beta P) = (1-beta)/n 1^T  (a linear system)
r_lin = np.linalg.solve((np.eye(n) - beta * P).T, (1 - beta) / n * np.ones(n))
print("PageRank (linear solve):   ", r_lin, " sum =", round(r_lin.sum(), 6))
print("check node 3 by hand: 0.15/5 + 0.85*r4/5 =", round(0.15 / 5 + 0.85 * r_lin[4] / 5, 4))

# Personalised PageRank: teleport only to node 1
e = np.zeros(n); e[1] = 1
ppr = np.linalg.solve((np.eye(n) - beta * P).T, (1 - beta) * e)
print("PPR seeded at node 1:      ", ppr, " sum =", round(ppr.sum(), 6))
```

Output:

```text
PageRank (power iteration): [0.3171 0.1872 0.3113 0.0524 0.132 ] iterations: 47
PageRank (linear solve):    [0.3171 0.1872 0.3113 0.0524 0.132 ]  sum = 1.0
check node 3 by hand: 0.15/5 + 0.85*r4/5 = 0.0524
PPR seeded at node 1:       [0.2625 0.2865 0.2794 0.0249 0.1467]  sum = 1.0
```

Node 3 has no in-links, so it only receives teleport mass ($0.15/5$) plus its share of the dangling node's uniform redistribution ($0.85\,r_4/5$) — the code checks this by hand. Under personalised PageRank, node 1 (the seed) becomes the highest-ranked node and node 3, which is far from node 1, is nearly ignored.

### 10.3 Random walk with restart: closed form, iteration, Neumann series, effect of $\alpha$

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def rwr_closed(S, seed, alpha):
    """p = alpha * W p + (1-alpha) * e,  W = S D^-1 (column-stochastic)."""
    W = S / S.sum(0, keepdims=True)
    n = len(S)
    return (1 - alpha) * np.linalg.solve(np.eye(n) - alpha * W, seed)

def rwr_iter(S, seed, alpha, tol=1e-10):
    W = S / S.sum(0, keepdims=True)
    p = seed.copy()
    for t in range(1, 10_000):
        p_new = alpha * W @ p + (1 - alpha) * seed
        if np.abs(p_new - p).sum() < tol:
            return p_new, t
        p = p_new

# Path graph 0 - 1 - 2 (the hand-worked example)
S = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
e = np.array([1.0, 0, 0])
print("closed form :", rwr_closed(S, e, 0.5), " (7/12, 1/3, 1/12 =",
      np.array([7/12, 1/3, 1/12]), ")")
p, t = rwr_iter(S, e, 0.5)
print("iteration   :", p, "after", t, "steps")

# Neumann series: (1-a) * sum_t (a W)^t e  -- truncated after T terms
W = S / S.sum(0, keepdims=True)
for T in (1, 2, 5, 30):
    acc, term = np.zeros(3), e.copy()
    for _ in range(T):
        acc += term
        term = 0.5 * W @ term
    print(f"Neumann series, {T:2d} terms:", 0.5 * acc)

# Effect of alpha on the 4-drug graph of Example 1: how local is the walk?
S4 = np.array([[0, 2, 1, 0], [2, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], float)
e3 = np.array([0, 0, 0, 1.0])
for a in (0.1, 0.5, 0.85, 0.99):
    print(f"alpha={a:4}: RWR from node 3 ->", rwr_closed(S4, e3, a))
print("stationary distribution d/sum(d) ->", S4.sum(1) / S4.sum())
```

Output:

```text
closed form : [0.5833 0.3333 0.0833]  (7/12, 1/3, 1/12 = [0.5833 0.3333 0.0833] )
iteration   : [0.5833 0.3333 0.0833] after 35 steps
Neumann series,  1 terms: [0.5 0.  0. ]
Neumann series,  2 terms: [0.5  0.25 0.  ]
Neumann series,  5 terms: [0.5781 0.3125 0.0781]
Neumann series, 30 terms: [0.5833 0.3333 0.0833]
alpha= 0.1: RWR from node 3 -> [0.0032 0.0032 0.0905 0.903 ]
alpha= 0.5: RWR from node 3 -> [0.075 0.075 0.3   0.55 ]
alpha=0.85: RWR from node 3 -> [0.2145 0.2145 0.3281 0.2429]
alpha=0.99: RWR from node 3 -> [0.2937 0.2937 0.3026 0.1099]
stationary distribution d/sum(d) -> [0.3 0.3 0.3 0.1]
```

The closed form matches worked example 3.2 exactly ($\tfrac7{12},\tfrac13,\tfrac1{12}$). The truncated Neumann series shows the answer being assembled from walks of length 0, 1, 2, …. The last block is the most important lesson: at $\alpha=0.1$ the walk barely leaves node 3; at $\alpha=0.99$ it has almost forgotten node 3 and approaches the degree-proportional stationary distribution $(0.3,0.3,0.3,0.1)$.

### 10.4 Label propagation: Zhou et al. versus the harmonic solution

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Two triangles {0,1,2} and {3,4,5} joined by the bridge 2-3
W = np.zeros((6, 6))
for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
    W[i, j] = W[j, i] = 1
d = W.sum(1)
S = W / np.sqrt(np.outer(d, d))            # D^-1/2 W D^-1/2

Y = np.zeros((6, 2))
Y[0, 0] = 1                                 # node 0 is labelled class "A"
Y[5, 1] = 1                                 # node 5 is labelled class "B"

alpha = 0.9
# (1) Zhou et al. iteration  F <- alpha S F + (1 - alpha) Y
F = Y.copy()
for t in range(1, 1000):
    F_new = alpha * S @ F + (1 - alpha) * Y
    if np.abs(F_new - F).max() < 1e-12:
        break
    F = F_new
print(f"iterative F after {t} steps:\n", F_new)
# (2) closed form  F* = (1 - alpha) (I - alpha S)^-1 Y
F_star = (1 - alpha) * np.linalg.solve(np.eye(6) - alpha * S, Y)
print("closed form F*:\n", F_star)
print("predicted class per node:", F_star.argmax(1))
print("spectral radius of alpha*S:", np.abs(np.linalg.eigvals(alpha * S)).max().round(4))

# (3) Zhu & Ghahramani harmonic solution: labelled nodes are CLAMPED
L_idx, U_idx = [0, 5], [1, 2, 3, 4]
Lap = np.diag(d) - W
F_u = np.linalg.solve(Lap[np.ix_(U_idx, U_idx)], W[np.ix_(U_idx, L_idx)] @ Y[L_idx])
print("harmonic solution for unlabelled nodes 1-4:\n", F_u)
```

Output:

```text
iterative F after 74 steps:
 [[0.2597 0.0791]
 [0.1907 0.0791]
 [0.201  0.1185]
 [0.1185 0.201 ]
 [0.0791 0.1907]
 [0.0791 0.2597]]
closed form F*:
 [[0.2597 0.0791]
 [0.1907 0.0791]
 [0.201  0.1185]
 [0.1185 0.201 ]
 [0.0791 0.1907]
 [0.0791 0.2597]]
predicted class per node: [0 0 0 1 1 1]
spectral radius of alpha*S: 0.9
harmonic solution for unlabelled nodes 1-4:
 [[0.8571 0.1429]
 [0.7143 0.2857]
 [0.2857 0.7143]
 [0.1429 0.8571]]
```

The iterative and closed-form solutions agree; both triangles are labelled correctly; the spectral radius of $\alpha\tilde S$ is exactly $\alpha$; and the harmonic values $\tfrac67, \tfrac57$ match worked example 4.1. Note the different *scales*: Zhou's scores are small (they include the $1-\alpha$ factor and the symmetric normalisation), the harmonic scores are probabilities. Only the ranking matters.

### 10.5 Convergence and the spectral radius

A ring of 50 nodes mixes slowly, so the error really does decay like $\alpha^t$.

```python
import numpy as np

# A ring of 50 nodes (each node similar to its two neighbours): a slowly-mixing graph
n = 50
S = np.zeros((n, n))
for i in range(n):
    S[i, (i + 1) % n] = S[(i + 1) % n, i] = 1.0
d = S.sum(1)
M = S / np.sqrt(np.outer(d, d))                 # D^-1/2 S D^-1/2
y = np.zeros(n); y[0] = 1
print("spectral radius of M:", np.abs(np.linalg.eigvalsh(M)).max().round(4))

def steps_to(M, y, alpha, tol=1e-8, max_it=100_000):
    f_star = (1 - alpha) * np.linalg.solve(np.eye(len(y)) - alpha * M, y)
    f = y.copy()
    for t in range(1, max_it + 1):
        f = alpha * M @ f + (1 - alpha) * y
        err = np.abs(f - f_star).max()
        if err < tol:
            return t
        if err > 1e12:
            return f"diverged (error {err:.1e} at step {t})"
    return "not converged"

for a in (0.3, 0.5, 0.9, 0.99):
    bound = int(np.ceil(np.log(1e-8) / np.log(a)))
    print(f"alpha={a:<4}: steps to error 1e-8 = {steps_to(M, y, a):>4}   bound log(tol)/log(alpha) = {bound}")

# Without normalisation the spectral radius is 2, so alpha=0.6 gives rho(alpha*S)=1.2 > 1
print("rho(raw S) =", np.abs(np.linalg.eigvalsh(S)).max().round(4))
print("alpha=0.6 with raw S:", steps_to(S, y, 0.6))
print("alpha=0.4 with raw S:", steps_to(S, y, 0.4), "(rho = 0.8 < 1, so it converges)")
```

Output:

```text
spectral radius of M: 1.0
alpha=0.3 : steps to error 1e-8 =   13   bound log(tol)/log(alpha) = 16
alpha=0.5 : steps to error 1e-8 =   23   bound log(tol)/log(alpha) = 27
alpha=0.9 : steps to error 1e-8 =  143   bound log(tol)/log(alpha) = 175
alpha=0.99: steps to error 1e-8 = 1444   bound log(tol)/log(alpha) = 1833
rho(raw S) = 2.0
alpha=0.6 with raw S: diverged (error 1.2e+12 at step 164)
alpha=0.4 with raw S: 74 (rho = 0.8 < 1, so it converges)
```

The iteration counts track the bound $\ln\varepsilon/\ln\alpha$ (a little below it because the initial error is smaller than 1). Without normalisation, $\rho(\alpha S) = 0.6\times2 = 1.2 > 1$ and the error grows without limit; with $\alpha = 0.4$, $\rho = 0.8$ and it converges.

### 10.6 The bi-random walk of MBiRW on the toy data

This re-implements the project's `MBiRW.fit_predict` logic on the 4-drug, 3-disease toy (disease 2 is cold).

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Toy problem: 4 drugs x 3 diseases. Disease 2 is "cold": no known drug.
A = np.array([[1, 0, 0],
              [1, 1, 0],
              [0, 0, 0],
              [0, 1, 0]], float)
Sr = np.array([[1.0, 0.8, 0.3, 0.1],      # drug-drug similarity (unit diagonal)
               [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7],
               [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3],            # disease-disease similarity
               [0.2, 1.0, 0.9],
               [0.3, 0.9, 1.0]])

def logistic(S, c=-15.0, d=np.log(9999)):
    """MBiRW's similarity adjustment L(x) = 1 / (1 + exp(c*x + d))."""
    return 1 / (1 + np.exp(c * S + d))

def sym_norm(S):
    d = S.sum(1); d[d == 0] = 1
    d = 1 / np.sqrt(d)
    return S * d[:, None] * d[None, :]

print("logistic(x) for x = 0, 0.3, 0.614, 0.8, 1:",
      logistic(np.array([0, 0.3, 0.614, 0.8, 1.0])))

def birw(A, Sr, Sd, alpha=0.3, l=2, r=2, left=True, right=True):
    Mr, Md = sym_norm(logistic(Sr)), sym_norm(logistic(Sd))
    A0 = A / max(A.sum(), 1)
    R = A0.copy()
    for step in range(1, max(l, r) + 1):
        parts = []
        if left and step <= l:
            parts.append(alpha * Mr @ R + (1 - alpha) * A0)   # walk on the DRUG network
        if right and step <= r:
            parts.append(alpha * R @ Md + (1 - alpha) * A0)   # walk on the DISEASE network
        R = sum(parts) / len(parts)
    return R

R = birw(A, Sr, Sd)
print("bi-random walk scores R (x100):\n", 100 * R)
print("ranking of drugs for cold disease 2:", np.argsort(-R[:, 2]))
print("left walk only, column 2: ", birw(A, Sr, Sd, right=False)[:, 2])
print("right walk only, column 2:", 100 * birw(A, Sr, Sd, left=False)[:, 2], "(x100)")
print("---- details for the hand calculation ----")
print("logistic(Sd) =\n", logistic(Sd))
print("Md = sym_norm(logistic(Sd)) =\n", sym_norm(logistic(Sd)))
A0 = A / A.sum()
print("A0 = A / 4;  one right step, column 2: alpha * (A0 @ Md)[:,2] =", 0.3 * (A0 @ sym_norm(logistic(Sd)))[:, 2])
```

Output:

```text
logistic(x) for x = 0, 0.3, 0.614, 0.8, 1: [0.0001 0.0089 0.4999 0.9421 0.997 ]
bi-random walk scores R (x100):
 [[24.9293  1.8349  0.2997]
 [24.9439 21.0261  1.8976]
 [ 0.0278  1.6806  0.2455]
 [ 0.0112 21.1686  1.894 ]]
ranking of drugs for cold disease 2: [1 3 0 2]
left walk only, column 2:  [0. 0. 0. 0.]
right walk only, column 2: [0.0557 3.7785 0.     3.7228] (x100)
---- details for the hand calculation ----
logistic(Sd) =
 [[0.997  0.002  0.0089]
 [0.002  0.997  0.9865]
 [0.0089 0.9865 0.997 ]]
Md = sym_norm(logistic(Sd)) =
 [[0.9892 0.0014 0.0063]
 [0.0014 0.5021 0.496 ]
 [0.0063 0.496  0.5004]]
A0 = A / 4;  one right step, column 2: alpha * (A0 @ Md)[:,2] = [0.0005 0.0377 0.     0.0372]
```

Check the last line against worked example 6.1: 0.0377 for drug 1 and 0.0372 for drug 3. The left walk alone leaves the cold column at exactly zero; the right walk fills it; the bi-walk ranks drugs 1 and 3 (the drugs of the similar disease 1) first. The logistic values match Section 6.2, including $L(0.614)\approx0.5$.

### 10.7 One-step kNN propagation (the project's `knn_kernel`)

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], float)   # same toy as Example 6
Sr = np.array([[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7], [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.9], [0.3, 0.9, 1.0]])

def knn_kernel(S, k):
    """Row-normalised weights of each node's k most similar OTHER nodes
    (same logic as drepo.data.knn_kernel, without the NaN handling)."""
    W = S.copy()
    np.fill_diagonal(W, -np.inf)                       # never pick yourself
    idx = np.argpartition(-W, k, axis=1)[:, :k]        # k largest per row
    K = np.zeros_like(S)
    rows = np.repeat(np.arange(len(S)), k)
    vals = S[rows, idx.ravel()]
    keep = vals > 0                                    # never link on zero similarity
    K[rows[keep], idx.ravel()[keep]] = vals[keep]
    s = K.sum(1, keepdims=True); s[s == 0] = 1
    return K / s

Kr, Kd = knn_kernel(Sr, 2), knn_kernel(Sd, 1)
print("Kr (k=2) =\n", Kr)
print("Kd (k=1) =\n", Kd)
P_drug = Kr @ A            # "what do my most similar DRUGS treat?"
P_dis = (Kd @ A.T).T       # "which drugs do my most similar DISEASES use?"  (= A @ Kd.T)
print("drug-view propagation  Kr @ A =\n", P_drug)
print("disease-view propagation (Kd @ A.T).T =\n", P_dis)
print("column of the cold disease 2:  drug view", P_drug[:, 2], "| disease view", P_dis[:, 2])
```

Output:

```text
Kr (k=2) =
 [[0.     0.7273 0.2727 0.    ]
 [0.8    0.     0.2    0.    ]
 [0.3    0.     0.     0.7   ]
 [0.125  0.     0.875  0.    ]]
Kd (k=1) =
 [[0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]]
drug-view propagation  Kr @ A =
 [[0.7273 0.7273 0.    ]
 [0.8    0.     0.    ]
 [0.3    0.7    0.    ]
 [0.125  0.     0.    ]]
disease-view propagation (Kd @ A.T).T =
 [[0. 0. 0.]
 [0. 0. 1.]
 [0. 0. 0.]
 [0. 0. 1.]]
column of the cold disease 2:  drug view [0. 0. 0. 0.] | disease view [0. 1. 0. 1.]
```

The kernel rows match worked example 8.1. (Drug 3 has a tie between drugs 0 and 1 at similarity 0.1; `argpartition` broke it in favour of drug 0.) The drug view gives the cold disease 2 nothing; the disease view gives it exactly the drugs of disease 1. The project's own `drepo.data.knn_kernel` returns identical matrices on these inputs (this was checked).

### 10.8 Random walk on the heterogeneous network (NRWRH-style)

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)

A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], float)   # same toy again
Sr = np.array([[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7], [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.9], [0.3, 0.9, 1.0]])
nr, nd = A.shape

def hetero_transition(A, Sr, Sd, lam):
    """Row-stochastic transition matrix of the drug+disease network (NRWRH-style).
    A node with cross-type links jumps to the other type w.p. lam, else stays."""
    def rownorm(X):
        s = X.sum(1, keepdims=True); s[s == 0] = 1
        return X / s
    Wr, Wd = Sr - np.diag(np.diag(Sr)), Sd - np.diag(np.diag(Sd))   # no self-loops
    has_r, has_d = A.sum(1) > 0, A.sum(0) > 0
    M_rr = rownorm(Wr) * np.where(has_r, 1 - lam, 1.0)[:, None]
    M_rd = rownorm(A) * lam
    M_dd = rownorm(Wd) * np.where(has_d, 1 - lam, 1.0)[:, None]
    M_dr = rownorm(A.T) * lam
    return np.block([[M_rr, M_rd], [M_dr, M_dd]])

def nrwrh_scores(A, Sr, Sd, j, lam=0.5, c=0.3, eta=0.5, tol=1e-10):
    M = hetero_transition(A, Sr, Sd, lam)
    u0 = A[:, j] / A[:, j].sum() if A[:, j].sum() > 0 else np.zeros(nr)  # known drugs of j
    v0 = np.eye(nd)[j]                                                   # the disease itself
    if u0.sum() == 0:
        eta = 1.0                       # cold disease: all restart mass on the disease seed
    p0 = np.concatenate([(1 - eta) * u0, eta * v0])
    p = p0.copy()
    for _ in range(10_000):
        p_new = (1 - c) * M.T @ p + c * p0
        if np.abs(p_new - p).sum() < tol:
            break
        p = p_new
    return p_new[:nr], p_new[nr:]

M = hetero_transition(A, Sr, Sd, 0.5)
print("row sums of the heterogeneous transition matrix:", M.sum(1))
drug_p, dis_p = nrwrh_scores(A, Sr, Sd, j=2)
print("cold disease 2 -> drug probabilities:", drug_p, " ranking:", np.argsort(-drug_p))
print("                 disease probabilities:", dis_p, " total mass:", round(drug_p.sum() + dis_p.sum(), 6))
```

Output:

```text
row sums of the heterogeneous transition matrix: [1. 1. 1. 1. 1. 1. 1.]
cold disease 2 -> drug probabilities: [0.0489 0.0835 0.0261 0.0607]  ranking: [1 3 0 2]
                 disease probabilities: [0.1184 0.2624 0.4   ]  total mass: 1.0
```

The heterogeneous walk agrees with the bi-random walk on the ranking (1, 3, 0, 2). The disease-side probabilities show where the walker spends its time: mostly at the seed (0.40, because of restarts) and its close neighbour, disease 1.

### 10.9 Real data: cold start on Fdataset

This script reads the project's processed Fdataset (read-only) and evaluates leave-one-disease-out for one-step disease-view propagation, a drug view, and MBiRW. It takes about 15 seconds on a laptop CPU.

```python
import sys
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from drepo.data import knn_kernel, fill_missing          # read-only use of project code

z = np.load(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\data\processed\Fdataset.npz")
A = z["A"].astype(float)
dv = [str(v) for v in z["disease_view_names"]]
rv = [str(v) for v in z["drug_view_names"]]
y = A.ravel()
report = lambda name, S: print(f"{name:34s} AUPR {average_precision_score(y, S.ravel()):.3f}"
                               f"  AUC {roc_auc_score(y, S.ravel()):.3f}")
print("positive rate (random AUPR):", round(y.mean(), 4))

# Leave-one-disease-out with ONE-STEP disease-view propagation.
# knn_kernel has a zero diagonal, so column j of A @ K.T never uses column j itself:
# computing it once with the full A is exactly the LODO score for every disease.
for names in (["pheno_mim"], ["sem_mondo"], ["gene_d"], ["pheno_mim", "sem_mondo"]):
    P = sum(A @ knn_kernel(z["disease_views"][dv.index(n)], 10).T for n in names)
    report("one-step " + "+".join(names), P)

# A DRUG-view propagation cannot score a cold disease: column j of K_r @ A_train is 0
Kr = knn_kernel(z["drug_views"][rv.index("chem_cdk")], 10)
S = np.zeros_like(A)
for j in range(A.shape[1]):
    A_tr = A.copy(); A_tr[:, j] = 0
    S[:, j] = (Kr @ A_tr)[:, j]
print("drug-view chem_cdk, cold column all zero?", bool((S == 0).all()))

# MBiRW (project defaults alpha=0.3, l=r=2) under the same protocol
def sym_norm(S):
    d = S.sum(1); d[d == 0] = 1; d = 1 / np.sqrt(d); return S * d[:, None] * d[None, :]
lg = lambda S: 1 / (1 + np.exp(-15 * S + np.log(9999)))
Mr = sym_norm(lg(fill_missing(z["drug_views"][0])))
Md = sym_norm(lg(fill_missing(z["disease_views"][0])))
def mbirw(A_tr, alpha=0.3, l=2, r=2):
    A0 = A_tr / max(A_tr.sum(), 1); R = A0.copy()
    for step in range(1, max(l, r) + 1):
        parts = []
        if step <= l: parts.append(alpha * Mr @ R + (1 - alpha) * A0)
        if step <= r: parts.append(alpha * R @ Md + (1 - alpha) * A0)
        R = sum(parts) / len(parts)
    return R
S = np.zeros_like(A)
for j in range(A.shape[1]):
    A_tr = A.copy(); A_tr[:, j] = 0
    S[:, j] = mbirw(A_tr)[:, j]
report("MBiRW (cdk + pheno_mim)", S)
```

Output:

```text
positive rate (random AUPR): 0.0104
one-step pheno_mim                 AUPR 0.174  AUC 0.738
one-step sem_mondo                 AUPR 0.111  AUC 0.643
one-step gene_d                    AUPR 0.053  AUC 0.584
one-step pheno_mim+sem_mondo       AUPR 0.191  AUC 0.753
drug-view chem_cdk, cold column all zero? True
MBiRW (cdk + pheno_mim)            AUPR 0.214  AUC 0.786
```

These are the numbers in the table of Section 8.3.

---

## 11. In this project: the code, line by line

> Code excerpts below are copied from the project files. Comments such as `# (1)` were added for this explanation (they refer to the numbered notes under each excerpt), and `...` marks lines left out. Open the real file alongside.

### 11.1 Building the kNN kernels — `src/drepo/data.py`

```python
def fill_missing(S: np.ndarray) -> np.ndarray:
    """Uncovered entities get no neighbours in that view (only themselves)."""
    S = np.nan_to_num(S.copy(), nan=0.0)
    np.fill_diagonal(S, 1.0)
    return S
```

* Some views do not cover every entity (a biologic has no SMILES, so no chemical similarity; half the diseases have no CTD genes). Those rows are stored as `NaN`. `nan_to_num` turns them into 0 — "no similarity to anybody" — so an uncovered node simply gets no neighbours in that view.
* The diagonal is set to 1 (everything is identical to itself). That self-similarity is removed again in `knn_kernel`.

```python
def knn_mask(S, k, symmetric=True):
    S = fill_missing(S)
    n = S.shape[0]
    work = S.copy()
    np.fill_diagonal(work, -np.inf)                 # (1)
    k = min(k, n - 1)
    idx = np.argpartition(-work, k, axis=1)[:, :k]  # (2)
    M = np.zeros_like(S, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    vals = work[rows, idx.ravel()]
    keep = vals > 0                                 # (3)
    M[rows[keep], idx.ravel()[keep]] = True
    if symmetric:
        M |= M.T                                    # (4)
    np.fill_diagonal(M, True)
    return M
```

1. Setting the diagonal to $-\infty$ guarantees a node never selects itself as a neighbour.
2. `argpartition(-work, k)` puts the $k$ largest similarities of each row in the first $k$ positions (unordered) in $O(n)$ per row, cheaper than a full sort.
3. A zero similarity is never an edge — important for uncovered nodes, whose rows are all zero.
4. The *graph* used by the GNN is symmetrised ($i\sim j$ if either picks the other) and gets self-loops. `knn_kernel` asks for `symmetric=False` instead, so each node keeps exactly its *own* $k$ nearest neighbours.

```python
def knn_kernel(S: np.ndarray, k: int) -> np.ndarray:
    W = fill_missing(S) * knn_mask(S, k, symmetric=False)   # keep k nearest, with weights
    np.fill_diagonal(W, 0.0)                                # never use your own links
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0                                         # isolated rows stay all-zero
    return W / s                                            # row-normalise: an averaging operator
```

This is the matrix $K$ of Section 8.1: row-stochastic on non-isolated rows, zero diagonal, $k$ non-zeros per row. Its docstring states the "pull" interpretation exactly: `K @ A` is, for every node, the similarity-weighted average association profile of its neighbours.

### 11.2 Precomputing one kernel per view — `MVHGATMethod.build` in `src/drepo/methods.py`

```python
self._prop = {"drug": [t(knn_kernel(data.drug_view(v), c.k)) for v in rv],
              "disease": [t(knn_kernel(data.disease_view(v), c.k)) for v in dv]}
```

* `rv` and `dv` are the drug and disease view names (by default all of them: `chem_cdk`, `chem_ecfp`, `gene_r` and `pheno_mim`, `sem_mondo`, `gene_d`).
* `c.k = 10` neighbours, the same $k$ as the GNN's kNN graphs.
* `t(...)` converts to a float tensor on the device. The kernels depend only on similarities, never on links, so they are computed once per fit and can never leak test links.

### 11.3 The propagation scores — `propagation()` inside `MVHGATMethod.fit_predict`

```python
def propagation(Am):
    """One (drugs x diseases) score slice per view, from VISIBLE links only."""
    if not c.prop_head:
        return None
    Am = Am.float()
    return torch.stack([K @ Am for K in self._prop["drug"]] +
                       [(K @ Am.T).T for K in self._prop["disease"]])
```

* `Am` is the boolean drugs × diseases matrix of links that are **visible** at this moment. During training it changes every epoch: 20% of training links are hidden (`drop_edge`) and 10% of diseases lose all their links (`cold_frac`). At prediction time it is the full training matrix `A_full`.
* `K @ Am` for a drug kernel is $P_v = K_vA$: rows of similar drugs averaged.
* `(K @ Am.T).T` for a disease kernel is $P_u = (K_uA^\top)^\top = AK_u^\top$: columns of similar diseases averaged.
* `torch.stack` produces a tensor of shape (6, 593, 313) on Fdataset: one score slice per view, in the order given by `self.prop_names`.

Because the scores are recomputed from `Am` every epoch, a link that is hidden *and* supervised this epoch is never part of its own propagation score: the model learns to *predict* links from neighbours' links, exactly as it must at test time.

```python
def degrees(Am):
    Am = Am.float()
    return (Am.sum(1), Am.sum(0)) if c.degree_gate else None
```

The number of *visible* links of every drug (row sums) and every disease (column sums), fed to the degree gate. A disease that is cold this epoch has degree 0.

The training step calls `model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))`; after training, `P, deg = propagation(A_full), degrees(A_full)` are used for the final scores.

### 11.4 Combining the views — `MVHGAT` in `src/drepo/model.py`

```python
if n_prop:
    self.prop_w = nn.Parameter(torch.zeros(n_prop))
    self.prop_scale = nn.Parameter(torch.tensor(5.0))
    self.bias = nn.Parameter(torch.tensor(-3.0))
    self.gate = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 1.0]))
```

* `prop_w` — one raw weight per view, starting at 0, so every view starts with the same weight $\text{softplus}(0)\times5 = \ln 2\times5\approx3.47$.
* `prop_scale` — a shared learned scale (initially 5). Propagation scores lie in $[0,1]$; multiplying by a few units lets them move the logit substantially.
* `bias = -3` — the logit of a pair with no evidence starts at $\sigma(-3)\approx0.047$, a sensible prior for a 1%-positive problem trained with two negatives per positive.
* `gate` — the four numbers $(a,b,c,d)$ of the degree gate.

```python
def view_weights(self):
    return F.softplus(self.prop_w) * self.prop_scale

def gnn_gate(self, deg_r, deg_d):
    g = self.gate
    return torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))[:, None] * \
           torch.sigmoid(g[2] + g[3] * torch.log1p(deg_d))[None, :]
```

* `softplus` keeps every view weight non-negative, so a view can only *add* evidence; together with additivity this makes $w_vP_v[i,j]$ an honest per-view contribution (used by `MVHGATMethod.view_weights` and `occlusion` for interpretability).
* The gate is a product of a drug factor and a disease factor, each a sigmoid of $\log(1+\text{degree})$. At initialisation a disease with 0 visible links gives a factor $\sigma(0)=0.5$; with 10 links $\sigma(\ln 11)\approx0.92$. Training adjusts $a,b,c,d$; because 10% of diseases are cold in every epoch, the gate learns how much to distrust the GNN term for them.

```python
def forward(self, X, graphs, drop_rel=(), P=None, deg=None):
    Hr, Hd, betas = self.encode(X, graphs, drop_rel)
    logits = Hr @ self.W @ Hd.T                                    # GNN term (bilinear)
    if self.n_prop and P is not None:
        if deg is not None:
            logits = self.gnn_gate(*deg) * logits                  # shrink GNN term for low-degree pairs
        logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
    return logits, betas
```

`(self.view_weights()[:, None, None] * P).sum(0)` is $\sum_v w_vP_v$: the weights are reshaped to (6, 1, 1) so they broadcast over the (6, drugs, diseases) stack. The result is exactly the formula of Section 1.

### 11.5 The MBiRW baseline — `sym_norm` and `MBiRW` in `src/drepo/methods.py`

```python
def sym_norm(S):
    d = S.sum(1)
    d[d == 0] = 1           # isolated nodes: avoid division by zero
    d = 1 / np.sqrt(d)
    return S * d[:, None] * d[None, :]     # D^-1/2 S D^-1/2 via broadcasting
```

Multiplying by `d[:, None]` scales row $i$ by $d_i^{-1/2}$ and by `d[None, :]` scales column $j$ by $d_j^{-1/2}$: this is $D^{-1/2}SD^{-1/2}$ without ever forming the diagonal matrices.

```python
class MBiRW:
    def __init__(self, alpha=0.3, l=2, r=2):
        self.alpha, self.l, self.r = alpha, l, r

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        Sr, Sd = bench_sims(data)                                       # (1)
        logistic = lambda S: 1 / (1 + np.exp(-15 * S + np.log(9999)))   # (2)
        Mr, Md = sym_norm(logistic(Sr)), sym_norm(logistic(Sd))         # (3)
        A0 = A_train / max(A_train.sum(), 1)                            # (4)
        R = A0.copy()
        for step in range(1, max(self.l, self.r) + 1):                  # (5)
            parts = []
            if step <= self.l:
                parts.append(self.alpha * Mr @ R + (1 - self.alpha) * A0)   # left walk
            if step <= self.r:
                parts.append(self.alpha * R @ Md + (1 - self.alpha) * A0)   # right walk
            R = sum(parts) / len(parts)                                 # (6)
        return R
```

1. `bench_sims` returns the two benchmark similarities (`chem_cdk` for drugs, `pheno_mim` for diseases), with missing values filled — the same information the original paper used.
2. $L(x) = 1/(1+e^{-15x+\ln 9999})$, i.e. $c=-15$, $d=\ln9999$ (Section 6.2).
3. Symmetric normalisation of the adjusted similarities (self-loops kept).
4. $A_0 = A/\sum A$; `max(..., 1)` protects against an empty matrix.
5. Steps $1,\dots,\max(l,r)$; a walk participates only while its step budget lasts.
6. The average of the active walks. Both walks start from the *same* $R_{t-1}$ (a Jacobi-style update), so the order of the two `if` blocks does not matter.

`neg_mask` and `seed` are unused: MBiRW has nothing to train and no randomness. Its output is not a probability, but AUC and AUPR depend only on the ranking.

---

## 12. Common mistakes and misconceptions

1. **Mixing up row and column normalisation.** $D^{-1}S$ *averages* (rows sum to 1); $SD^{-1}$ *spreads mass* (columns sum to 1). Using the wrong one with the wrong multiplication side silently produces degree-biased scores. Check: after normalising, which sums equal 1?
2. **Forgetting to normalise at all.** Iterating with raw similarities: the spectral radius is about the average degree, and the iteration diverges for most $\alpha$ (Section 10.5).
3. **Believing the stationary distribution is the answer.** For link prediction the long-run distribution is *useless*: it is proportional to degree and identical for every seed. Restarts (or few steps) are what keep the seed's information.
4. **Thinking more steps or larger $\alpha$ are always better.** On Fdataset cold start, one step gives AUPR 0.174 and five steps 0.076. More propagation means more smoothing and more popularity bias.
5. **Leaving self-loops in a propagation kernel used on the labels.** If $K_{jj}>0$, the score of a pair contains its own label. On Fdataset this inflates "cold-start" AUPR from 0.174 to 0.848 — a pure leak (Exercise 12).
6. **Computing propagation from links that include test links.** Every propagation score must be computed from the training (visible) links of the current fold only. In `fit_predict`, propagation is always called on `Am` or `A_full = A_train > 0`, never on the full dataset.
7. **Expecting drug-side propagation to help a cold disease.** $K_vA_{:,j} = 0$ when column $j$ is empty. Only disease-side similarity (or other disease features) can score a brand-new disease, and only drug-side similarity can score a brand-new drug.
8. **Confusing the names of $\alpha$.** Some papers call the continue-probability $\alpha$, others the restart probability. In MBiRW's code $\alpha$ multiplies the walk term. Always read the formula.
9. **Assuming walk scores are probabilities of being a true indication.** RWR scores are visiting probabilities; label propagation scores are smoothed labels. They rank pairs but are not calibrated; for AUC/AUPR this is fine, for a decision threshold it is not.
10. **Treating a symmetric kNN graph and a kNN kernel as the same thing.** The project's GNN graph is symmetrised with self-loops (`knn_mask(..., symmetric=True)`), while the propagation kernel is directed, self-free and row-normalised (`knn_kernel`). They serve different purposes.
11. **Ignoring periodicity.** On a bipartite graph (such as the drug–disease association graph itself!) a plain walk oscillates. Restarts, self-loops or a lazy walk fix it.

---

## 13. Exercises

Difficulty: ★ conceptual, ★★ standard, ★★★ challenging. Try each before opening the solution.

**Exercise 1 (★, conceptual).** You want each drug's new association row to be the similarity-weighted *average* of its neighbours' rows. Should you compute $(D^{-1}S)A$ or $(SD^{-1})A$? What goes wrong with the other one?

<details><summary>Solution</summary>

Use $(D^{-1}S)A$. Row $i$ of $D^{-1}S$ sums to 1, so row $i$ of the product is $\sum_{i'}\frac{S_{ii'}}{d_i}A_{i',:}$, a convex combination (weighted average) of neighbours' rows. With $SD^{-1}$, entry $(i,i')$ is $S_{ii'}/d_{i'}$: each *neighbour* divides its contribution by its own degree. Row $i$'s weights then do not sum to 1, so a drug with many low-degree neighbours gets large scores and one with high-degree neighbours small scores. The result is a mass-spreading (push) operation, not an average, and it introduces degree effects that have nothing to do with drug $i$.
</details>

**Exercise 2 (★★, maths + code).** Consider the star graph: centre 0 connected to leaves 1, 2, 3 (unit weights). (a) Find the stationary distribution. (b) Is the walk aperiodic? Describe what power iteration does when started on leaf 1. (c) Show that the lazy walk $\tfrac12(I+P)$ converges, by running it on the 3-node path $0-1-2$ (also bipartite).

<details><summary>Solution</summary>

(a) Degrees $(3,1,1,1)$, volume 6, so $\boldsymbol\pi = (\tfrac12,\tfrac16,\tfrac16,\tfrac16)$.

(b) The star is bipartite (centre vs leaves), hence periodic with period 2: from leaf 1 the walker is at the centre after odd steps and uniformly on the leaves after even steps. The sequence $\mathbf p_t$ alternates between $(1,0,0,0)$ and $(0,\tfrac13,\tfrac13,\tfrac13)$ and never converges, although its time-average does converge to $\boldsymbol\pi$. Algebraically, $P$ has eigenvalue $-1$.

(c) The lazy walk's eigenvalues are $\tfrac12(1+\lambda)\in[0,1]$, so $-1$ becomes 0:

```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# Exercise: lazy walk on the bipartite path 0-1-2
S = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
P = S / S.sum(1, keepdims=True)
p = np.array([1.0, 0, 0])
for t in range(4):
    p = p @ P
    print("plain walk step", t + 1, p)
Pl = 0.5 * (np.eye(3) + P)
p = np.array([1.0, 0, 0])
for t in range(60):
    p = p @ Pl
print("lazy walk after 60 steps:", p)
print("eigenvalues plain:", np.sort(np.linalg.eigvals(P).real), " lazy:", np.sort(np.linalg.eigvals(Pl).real))
```

```text
plain walk step 1 [0. 1. 0.]
plain walk step 2 [0.5 0.  0.5]
plain walk step 3 [0. 1. 0.]
plain walk step 4 [0.5 0.  0.5]
lazy walk after 60 steps: [0.25 0.5  0.25]
eigenvalues plain: [-1.  0.  1.]  lazy: [0.  0.5 1. ]
```

The plain walk oscillates forever; the lazy walk reaches the stationary distribution $(\tfrac14,\tfrac12,\tfrac14)$ (degrees 1, 2, 1).
</details>

**Exercise 3 (★★, maths).** Prove that $P=D^{-1}S$ and $\tilde S = D^{-1/2}SD^{-1/2}$ have the same eigenvalues, and express the eigenvectors of $P$ in terms of those of $\tilde S$. Why does this imply that all eigenvalues of $P$ are real even though $P$ is not symmetric?

<details><summary>Solution</summary>

$\tilde S = D^{1/2}PD^{-1/2}$. If $\tilde S\mathbf u = \lambda\mathbf u$, then $D^{1/2}PD^{-1/2}\mathbf u = \lambda\mathbf u$; multiplying by $D^{-1/2}$ on the left, $P(D^{-1/2}\mathbf u) = \lambda(D^{-1/2}\mathbf u)$. So $\mathbf x = D^{-1/2}\mathbf u$ is an eigenvector of $P$ with the same eigenvalue. Conversely every eigenvector $\mathbf x$ of $P$ gives $\mathbf u = D^{1/2}\mathbf x$. The characteristic polynomials are equal because $\det(\tilde S-\lambda I) = \det(D^{1/2}(P-\lambda I)D^{-1/2}) = \det(P-\lambda I)$. $\tilde S$ is real symmetric, so by the spectral theorem its eigenvalues are real; $P$ shares them, so they are real too. (In particular the top eigenvector of $\tilde S$ is $D^{1/2}\mathbf 1$, which maps to $\mathbf 1$ for $P$.)
</details>

**Exercise 4 (★★, maths).** A 2-page web: page 0 links to page 1; page 1 has no out-links (dangling, so its row becomes uniform). Find the PageRank vector as a function of $\beta$. What is it for $\beta=0.85$, and what is the limit as $\beta\to1$?

<details><summary>Solution</summary>

After fixing the dangling row, $P = \begin{pmatrix}0&1\\\tfrac12&\tfrac12\end{pmatrix}$. PageRank satisfies $r_0 = \beta\cdot\tfrac12r_1 + \tfrac{1-\beta}2$ and $r_0+r_1=1$. Substituting $r_1 = 1-r_0$: $r_0(1+\tfrac\beta2) = \tfrac\beta2+\tfrac{1-\beta}{2} = \tfrac12$, so
$$r_0 = \frac{1}{2+\beta},\qquad r_1 = \frac{1+\beta}{2+\beta}.$$
For $\beta=0.85$: $r = (0.3509, 0.6491)$. As $\beta\to1$: $(\tfrac13,\tfrac23)$, which is the stationary distribution of $P$ itself ($\pi_0 = \tfrac12\pi_1$). As $\beta\to0$: $(\tfrac12,\tfrac12)$, the teleport distribution.
</details>

**Exercise 5 (★★, maths).** For column-stochastic $W$, $0\le\alpha<1$ and a restart distribution $\mathbf e\ge0$ summing to 1, prove that the RWR solution $\mathbf p^* = (1-\alpha)(I-\alpha W)^{-1}\mathbf e$ is non-negative and sums to 1. Then find its limits as $\alpha\to0$ and (for a connected, aperiodic graph) $\alpha\to1$.

<details><summary>Solution</summary>

By the Neumann series, $\mathbf p^* = (1-\alpha)\sum_{t\ge0}\alpha^tW^t\mathbf e$, a sum of non-negative terms (all of $W$, $\mathbf e$, $\alpha$ are non-negative), so $\mathbf p^*\ge0$. Since $\mathbf 1^\top W = \mathbf 1^\top$, $\mathbf 1^\top W^t\mathbf e = 1$ for every $t$, so $\mathbf 1^\top\mathbf p^* = (1-\alpha)\sum_t\alpha^t = 1$.
As $\alpha\to0$ only the $t=0$ term survives: $\mathbf p^*\to\mathbf e$. As $\alpha\to1$ the geometric weights $(1-\alpha)\alpha^t$ spread over ever longer walks; since $W^t\mathbf e\to\boldsymbol\pi$ (the stationary distribution, as a column vector) for a connected aperiodic graph, the weighted average tends to $\boldsymbol\pi$, independent of the seed. This is the loss of personalisation seen in Section 10.3.
</details>

**Exercise 6 (★★, coding).** Verify numerically, on a random symmetric similarity matrix, the "one core matrix" table of Section 3.3: push $=(1-\alpha)D(D-\alpha S)^{-1}$, pull $=(1-\alpha)(D-\alpha S)^{-1}D$, symmetric $=(1-\alpha)D^{1/2}(D-\alpha S)^{-1}D^{1/2}$. Also verify that $p_s(i)/d_i = p_i(s)/d_s$ for personalised PageRank, and which matrices have unit column or row sums.

<details><summary>Solution</summary>

```python
import numpy as np
rng = np.random.default_rng(0)
X = rng.random((6, 6)); S = (X + X.T) / 2; np.fill_diagonal(S, 0)
d = S.sum(1); D = np.diag(d); a = 0.8; I = np.eye(6)
core = np.linalg.inv(D - a * S)                       # symmetric
push = (1 - a) * np.linalg.inv(I - a * S @ np.diag(1 / d))      # column-stochastic RWR
pull = (1 - a) * np.linalg.inv(I - a * np.diag(1 / d) @ S)      # row-stochastic averaging
sym = (1 - a) * np.linalg.inv(I - a * np.diag(d ** -.5) @ S @ np.diag(d ** -.5))
print("push == D core     :", np.allclose(push, (1 - a) * D @ core))
print("pull == core D     :", np.allclose(pull, (1 - a) * core @ D))
print("sym  == D^.5 core D^.5:", np.allclose(sym, (1 - a) * np.diag(d ** .5) @ core @ np.diag(d ** .5)))
print("PPR symmetry p_s(i)/d_i == p_i(s)/d_s:", np.allclose(push / d[:, None], (push / d[:, None]).T))
print("columns of push sum to 1:", np.allclose(push.sum(0), 1), "| rows of pull sum to 1:", np.allclose(pull.sum(1), 1))
```

```text
push == D core     : True
pull == core D     : True
sym  == D^.5 core D^.5: True
PPR symmetry p_s(i)/d_i == p_i(s)/d_s: True
columns of push sum to 1: True | rows of pull sum to 1: True
```

The push matrix's column $s$ is the RWR vector of seed $s$ (columns sum to 1: mass conservation); the pull matrix's rows sum to 1 (each score is a weighted average of the labels $\mathbf y$, so it stays in $[0,1]$ when $\mathbf y$ does).
</details>

**Exercise 7 (★★, maths).** A path $0-1-2-3-4$ with unit weights. Node 0 is clamped to label value 1 and node 4 to 0. Compute the harmonic (Zhu–Ghahramani) values of nodes 1–3, and interpret them as probabilities of a random walk.

<details><summary>Solution</summary>

Harmonicity on a path means each interior value is the average of its two neighbours: $f_i = \tfrac12(f_{i-1}+f_{i+1})$, i.e. $f_{i+1}-f_i = f_i - f_{i-1}$: the values are in arithmetic progression. With $f_0=1$, $f_4=0$: $f = (1, \tfrac34, \tfrac12, \tfrac14, 0)$. Interpretation: $f_i$ is the probability that a symmetric random walk started at $i$ reaches node 0 before node 4 (the classic "gambler's ruin" probability $1 - i/4$).
</details>

**Exercise 8 (★, conceptual).** In MBiRW, (a) why can the left walk never give a non-zero score to a disease column that is all zeros? (b) Which walk is unable to score a brand-new *drug* (an all-zero row)? (c) What information would a method need to score a brand-new drug *and* a brand-new disease at the same time?

<details><summary>Solution</summary>

(a) The left walk computes $M_rR$; column $j$ of $M_rR$ is $M_r$ times column $j$ of $R$, and $M_r\mathbf 0 = \mathbf 0$. The restart term adds $(1-\alpha)A_0[:,j] = 0$. So the column stays zero unless the right walk has put something there.
(b) Symmetrically, the right walk $RM_d$ cannot fill an all-zero row; only the left walk (drug similarity) can score a new drug.
(c) A pair (new drug, new disease) has no links on either side: no neighbour-based walk can reach it, because every path from it to a known link must cross a known link of the new drug or the new disease. You need *features* that map each entity into a shared space — for example inductive matrix completion or a GNN with feature inputs (Chapter 10, Section on IMC), where the score is a function of the drug's and disease's features rather than of their links.
</details>

**Exercise 9 (★★★, maths + coding).** (a) Show that with $l=r\to\infty$ the MBiRW iteration converges, and that its limit solves the Sylvester equation of Section 6.4. (b) Solve it with `scipy.linalg.solve_sylvester` on the toy data and compare with the iteration and with the two-step result of Section 10.6.

<details><summary>Solution</summary>

(a) With both walks active, $R_t = \tfrac12[\alpha M_rR_{t-1} + (1-\alpha)A_0] + \tfrac12[\alpha R_{t-1}M_d + (1-\alpha)A_0] = \mathcal T(R_{t-1}) + (1-\alpha)A_0$ with $\mathcal T(R) = \tfrac\alpha2(M_rR+RM_d)$. In vectorised form $\mathcal T$ is $\tfrac\alpha2(I\otimes M_r + M_d^\top\otimes I)$, whose eigenvalues are $\tfrac\alpha2(\lambda_i+\mu_j)$. Since $M_r$, $M_d$ are symmetrically normalised non-negative matrices, $\lambda_i,\mu_j\in[-1,1]$ and $|\tfrac\alpha2(\lambda_i+\mu_j)|\le\alpha<1$. By Section 5 the iteration converges to the fixed point $R = \tfrac\alpha2(M_rR+RM_d)+(1-\alpha)A_0$. Moving everything to one side: $(\tfrac\alpha2M_r - \tfrac12I)R + R(\tfrac\alpha2M_d-\tfrac12I) = -(1-\alpha)A_0$, a Sylvester equation $aX+Xb=q$.

(b)

```python
import numpy as np
from scipy.linalg import solve_sylvester
np.set_printoptions(precision=4, suppress=True)
A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], float)
Sr = np.array([[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7], [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.9], [0.3, 0.9, 1.0]])
logistic = lambda S: 1 / (1 + np.exp(-15 * S + np.log(9999)))
def sym_norm(S):
    d = S.sum(1); return S / np.sqrt(np.outer(d, d))
Mr, Md = sym_norm(logistic(Sr)), sym_norm(logistic(Sd))
alpha = 0.3
A0 = A / A.sum()
R = A0.copy()
for t in range(200):                                  # l = r = "infinity"
    R = 0.5 * (alpha * Mr @ R + (1 - alpha) * A0) + 0.5 * (alpha * R @ Md + (1 - alpha) * A0)
# fixed point: (alpha/2 Mr - I/2) R + R (alpha/2 Md - I/2) = -(1 - alpha) A0
R_syl = solve_sylvester(alpha / 2 * Mr - np.eye(4) / 2, alpha / 2 * Md - np.eye(3) / 2, -(1 - alpha) * A0)
print("iteration == Sylvester solution:", np.allclose(R, R_syl))
print("100 x R_inf:\n", 100 * R_syl)
ev = np.add.outer(np.linalg.eigvalsh(Mr), np.linalg.eigvalsh(Md)) * alpha / 2
print("spectral radius of the BiRW operator:", np.abs(ev).max().round(4))
```

```text
iteration == Sylvester solution: True
100 x R_inf:
 [[24.9236  1.8369  0.3518]
 [24.9381 20.9767  1.8986]
 [ 0.0304  1.6839  0.2975]
 [ 0.0131 21.1147  1.8913]]
spectral radius of the BiRW operator: 0.3
```

The spectral radius is exactly $\alpha=0.3$ here (both top eigenvalues equal 1). Compare with the two-step matrix in Section 10.6: e.g. entry (1, 2) is 1.8976 after two steps and 1.8986 in the limit; entry (2, 2) moves from 0.2455 to 0.2975. With $\alpha=0.3$ two steps already capture most of the limit, and the rankings are almost unchanged.
</details>

**Exercise 10 (★★★, coding on project data).** Generalise the project's one-step disease-view propagation to $t$ steps, $P^{(t)} = A\,(K^t)^\top$, and measure cold-start (leave-one-disease-out) AUPR on Fdataset with `pheno_mim` for $t=1,2,3,5$. Be careful: why can you no longer compute all diseases at once from the full $A$? Explain the result.

<details><summary>Solution</summary>

For $t\ge2$, $K^t$ has a non-zero diagonal (walks $j\to j'\to j$ return home), so column $j$ of $A(K^t)^\top$ would read column $j$ of $A$ — the hidden answer. You must zero column $j$ first, for each disease separately:

```python
import sys
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
import numpy as np
from sklearn.metrics import average_precision_score
from drepo.data import knn_kernel
z = np.load(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\data\processed\Fdataset.npz")
A = z["A"].astype(float)
K = knn_kernel(z["disease_views"][[str(v) for v in z["disease_view_names"]].index("pheno_mim")], 10)
for steps in (1, 2, 3, 5):
    Kt = np.linalg.matrix_power(K, steps)
    S = np.zeros_like(A)
    for j in range(A.shape[1]):                  # leave-one-disease-out, done properly
        A_tr = A.copy(); A_tr[:, j] = 0
        S[:, j] = A_tr @ Kt[j]                   # = (A_tr @ Kt.T)[:, j]
    print(f"{steps} step(s): cold-start AUPR {average_precision_score(A.ravel(), S.ravel()):.3f}")
```

```text
1 step(s): cold-start AUPR 0.174
2 step(s): cold-start AUPR 0.119
3 step(s): cold-start AUPR 0.110
5 step(s): cold-start AUPR 0.076
```

Every extra step makes the cold-start ranking worse. Rows of $K^t$ converge towards the stationary distribution of the kNN walk, so all diseases' score columns become similar and dominated by diseases that are central in the kNN graph and by drugs with many indications (popularity). The informative signal is in the *immediate* neighbourhood. This is the empirical justification for the project's *one-step* head and for MBiRW's small $\alpha$ and $l=r=2$.
</details>

**Exercise 11 (★★, maths).** (a) How many iterations does RWR with $\alpha=0.85$ need to guarantee an error below $10^{-6}$ (starting error at most 1)? (b) For the ring graph of Section 10.5, which values of $\alpha$ make the *unnormalised* iteration $\mathbf f\leftarrow\alpha S\mathbf f + (1-\alpha)\mathbf y$ converge?

<details><summary>Solution</summary>

(a) Error $\le0.85^t$; need $t\ge\ln10^{-6}/\ln0.85 = 13.816/0.1625\approx85.0$, so 85 iterations (86 to be safe with rounding).
(b) The ring's adjacency matrix has eigenvalues $2\cos(2\pi k/50)$, so $\rho(S)=2$ and $\rho(\alpha S) = 2\alpha$. Convergence iff $2\alpha<1$, i.e. $\alpha<0.5$. That is why $\alpha=0.4$ converged and $\alpha=0.6$ diverged. After symmetric normalisation ($\tilde S = S/2$ on a 2-regular graph) any $\alpha<1$ works.
</details>

**Exercise 12 (★★, conceptual + project).** `knn_kernel` zeroes the diagonal. Suppose someone "simplifies" it by keeping the self-similarity (so each disease is one of its own neighbours with weight $S_{jj}=1$). (a) What happens to the cold-start evaluation trick of Section 8.2? (b) What happens during training of MV-HGAT?

<details><summary>Solution</summary>

(a) Column $j$ of $AK^\top$ would include $K_{jj}A_{:,j}$, i.e. the hidden answer for disease $j$. Computing the scores once on the full matrix would no longer be a valid LODO evaluation: it would be a leak. On Fdataset with `pheno_mim` this "simplified" kernel scores AUPR 0.848 and AUC 0.999 (self-weight about 0.235 per row on average) instead of the honest 0.174 — a spectacular and completely fake improvement.
(b) In training, $P_v[i,j]$ would contain the visible link $(i,j)$ itself. Hidden links are removed from `Am`, so for them the leak disappears, but the model would see a mismatch between visible pairs (with self-evidence) and hidden pairs (without), and the learned view weights would partly reward "the link is already there" — the trap described in Section 10 of `docs/HOW_IT_WORKS.md`. Zeroing the diagonal makes every propagation score depend only on *other* entities' links.
</details>

**Exercise 13 (★★, conceptual).** Explain why push-style RWR on the *drug–disease association graph* tends to recommend the same few drugs to every disease when $\alpha$ is large. Which drugs, and how could you reduce the effect?

<details><summary>Solution</summary>

As $\alpha\to1$ the RWR vector tends to the stationary distribution, which is proportional to degree. On the association graph a drug's degree is its number of known indications, so drugs with many indications (broad-spectrum agents such as corticosteroids, which treat dozens of inflammatory and autoimmune conditions) receive high scores for every disease. Remedies: use a small $\alpha$ or few steps (MBiRW: 0.3, two steps); use pull/symmetric rather than push normalisation (Section 3.3); divide scores by degree (using the symmetry $p_s(i)/d_i$); sparsify the similarity graph (kNN, logistic adjustment) so hubs do not connect to everything; or evaluate per disease so that popularity alone cannot win.
</details>

**Exercise 14 (★★★, design).** You add a seventh view, a disease view built from shared clinical-trial sponsors, which is *very* dense (every disease has similarity above 0.2 to 80% of others). Predict what happens to its one-step propagation scores with $k=10$ versus no kNN cut, and how the learned weight $w_v$ is likely to behave.

<details><summary>Solution</summary>

Without a kNN cut, each disease's propagation row averages over almost all diseases, so $P_v[:,j]\approx$ the average column of $A$ for every $j$: a popularity score identical for all diseases, which carries almost no disease-specific information (it can still help AUC slightly because popular drugs are more often positives, but not ranking within a disease). With $k=10$ only the ten most similar diseases count, so whatever specific structure the view has is preserved; if the top-10 neighbours are still essentially random, the scores will be noisy. During training the propagation head will learn a small $w_v$ for an uninformative view (softplus lets it approach 0), so the model is protected — but you would see it in `view_weights()` and in occlusion, and the extra view costs one more parameter and one more GNN relation. The lesson matches the project's finding that more views do not help automatically; sparsification and per-view weights are what let useful views through.
</details>

---

## 14. Answers to the self-check questions (PREREQUISITES.md, unit C2)

### Q1. In `MBiRW`, what does `alpha` trade off?

`alpha` is the weight on the **walk step** in each update, $R\leftarrow\alpha M_rR + (1-\alpha)A_0$ (and likewise on the right). It trades off two sources of evidence:

* **Neighbour evidence (generalisation).** The fraction $\alpha$ of the new score comes from similar drugs (left walk) or similar diseases (right walk). This is what lets the method predict links that are not in $A$ at all.
* **Fidelity to the known associations (restart).** The fraction $1-\alpha$ is re-injected from the normalised known links $A_0$ at every step. This anchors the walk to the observed data and keeps the information local.

Consequences of the setting: small $\alpha$ (the project uses 0.3, so 70% restart) gives sharp, local predictions dominated by close neighbours, and keeps hubs from taking over; large $\alpha$ spreads evidence further (helpful for very sparse rows or columns) but drives the scores towards the degree-proportional stationary distribution — popularity bias — and makes the iteration converge more slowly (error $\sim\alpha^t$). In walk terms, the expected walk length between restarts is $\alpha/(1-\alpha)$: under half a step at $\alpha=0.3$. $\alpha$ interacts with $l$ and $r$: with only two steps, $\alpha$ mainly controls how much the second-hop neighbours count.

### Q2. What does "left walk" versus "right walk" mean?

They are named after the side on which the similarity matrix multiplies the score matrix $R$ (drugs × diseases):

* **Left walk:** $M_rR$ — the drug similarity multiplies from the *left*, mixing *rows*. Entry $(i,j)$ becomes $\sum_{i'}(M_r)_{ii'}R_{i'j}$: drug $i$ borrows the scores that *similar drugs* have for the *same disease*. It is a random walk on the **drug network**, run for at most $l$ steps.
* **Right walk:** $RM_d$ — the disease similarity multiplies from the *right*, mixing *columns*. Entry $(i,j)$ becomes $\sum_{j'}R_{ij'}(M_d)_{j'j}$: disease $j$ borrows the scores that the *same drug* has for *similar diseases*. It is a random walk on the **disease network**, run for at most $r$ steps.

At each step MBiRW averages the two walks (while both are active). A practical consequence: only the right walk can put a non-zero score into an empty disease column, and only the left walk into an empty drug row.

### Q3. Why is propagation so strong for diseases with no known drugs?

1. **It is the only information path that still works.** A cold disease has an empty column. Drug-side propagation gives exactly zero for it ($K_vA_{:,j} = 0$); a matrix-factorisation or GNN embedding for it has no observed link to learn from; the GNN's collaborative signal (`assoc` edges) is absent. Disease-side propagation reads the columns of its *neighbours*, which are fully observed.
2. **The guilt-by-association assumption holds strongly in these benchmarks.** Diseases are OMIM phenotypes, and phenotypically similar diseases (subtypes of epilepsy, of cardiomyopathy, of cancer) really do share drugs. One-step propagation through `pheno_mim` alone reaches cold-start AUPR 0.15–0.17 against a random baseline of 0.010.
3. **It has no parameters to overfit and needs nothing from the cold disease itself.** With 1,933 known links, anything learned is noisy; a fixed weighted average is not. The project's learned part only chooses non-negative weights per view.
4. **One step is the right amount.** The most informative evidence is in the immediate neighbourhood; more steps (or a large $\alpha$) blur it towards popularity (cold-start AUPR falls from 0.174 at one step to 0.076 at five).
5. **Views are complementary.** Summing `pheno_mim` and `sem_mondo` beats either alone (0.175 vs 0.149 in the project's validation proxy; 0.191 vs 0.174 in full LODO), because they make different errors.

This is why MV-HGAT includes the propagation head and a degree gate: for a cold disease the gate shrinks the GNN term and the score is carried by the disease-view propagation terms.

---

## 15. Summary and cheat sheet

**Core formulas**

| Concept | Formula | Remember |
|---|---|---|
| Transition matrix | $P=D^{-1}S$, rows sum to 1 | $P\mathbf f$ = neighbour average ("pull") |
| Distribution update | $\mathbf p_{t+1}^\top = \mathbf p_t^\top P$ (or $\mathbf p_{t+1} = W\mathbf p_t$, $W=SD^{-1}$) | "push" conserves mass |
| Symmetric normalisation | $\tilde S = D^{-1/2}SD^{-1/2}$ | same eigenvalues as $P$, all in $[-1,1]$ |
| Stationary distribution (undirected) | $\pi_i = d_i/\sum_k d_k$ | carries no seed information; degree bias |
| Convergence rate of a walk | error $\sim\lvert\lambda_2\rvert^t$ | spectral gap $1-\lvert\lambda_2\rvert$ |
| PageRank | $\mathbf r = (1-\beta)(I-\beta P^\top)^{-1}\mathbf v$, $\beta=0.85$ | dangling rows → uniform |
| RWR / personalised PageRank | $\mathbf p = (1-\alpha)(I-\alpha W)^{-1}\mathbf e = (1-\alpha)\sum_t\alpha^tW^t\mathbf e$ | mean walk length $\alpha/(1-\alpha)$ |
| Label propagation (Zhou) | $F^* = (1-\alpha)(I-\alpha\tilde S)^{-1}Y$ | minimises $\mathrm{tr}(F^\top\mathcal LF)+\mu\lVert F-Y\rVert^2$, $\alpha = 1/(1+\mu)$ |
| Harmonic (Zhu) | $F_U = (D_{UU}-S_{UU})^{-1}S_{UL}Y_L$ | labels clamped; absorbing walk |
| Convergence | $\mathbf x\leftarrow M\mathbf x+\mathbf b$ converges iff $\rho(M)<1$ | iterations $\approx\ln\varepsilon/\ln\alpha$ |
| MBiRW logistic | $L(x) = 1/(1+e^{-15x+\ln9999})$ | soft threshold at $x\approx0.61$ |
| MBiRW step | $R\leftarrow\mathrm{avg}\{\alpha M_rR+(1-\alpha)A_0,\ \alpha RM_d+(1-\alpha)A_0\}$ | $\alpha=0.3$, $l=r=2$ |
| Project propagation | drug view $K_vA$; disease view $AK_u^\top$; $K$ = row-normalised top-$k$, zero diagonal | $k=10$, one step |
| MV-HGAT logit | $\text{gate}\cdot\mathbf h_i^\top W\mathbf h_j + \sum_vw_vP_v[i,j] + b$ | $w_v=\text{softplus}\cdot s\ge0$ |

**Key ideas in one breath.** A random walk is repeated multiplication by a normalised similarity matrix; left alone it forgets its start and converges to a degree-proportional distribution, so link prediction uses *short* walks or *restarts*. RWR, personalised PageRank and label propagation are the same linear system $(I-\alpha M)^{-1}$ in different normalisations, and they converge because $\rho(\alpha M)\le\alpha<1$. MBiRW walks on both the drug and the disease network around the score matrix. For a disease with no known drugs, only disease-side propagation carries information, and one step of it is remarkably strong; more steps over-smooth.

---

## 16. Further resources (all links checked)

**Courses and lecture notes**
* Stanford CS224W (Leskovec), 2021 lecture on PageRank, personalised PageRank and random walk with restart — slides: <https://snap.stanford.edu/class/cs224w-2021/slides/04-pagerank.pdf> (free)
* Stanford CS224W video lectures (2021 playlist, PageRank is lecture 4) — <https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn> (free)
* Current CS224W course page — <https://web.stanford.edu/class/cs224w/> (free)
* Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, ch. 5 "Link Analysis" (PageRank, teleportation, topic-sensitive PageRank) — <http://infolab.stanford.edu/~ullman/mmds/ch5.pdf>; book site <http://www.mmds.org/> (free)

**Books and surveys**
* Levin, Peres & Wilmer, *Markov Chains and Mixing Times* (stationary distributions, mixing, spectral gap; rigorous) — <https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf> (free)
* Spielman, *Spectral and Algebraic Graph Theory* (Laplacians, random walks, eigenvalues) — <https://www.cs.yale.edu/homes/spielman/sagt/> (free)
* Hamilton, *Graph Representation Learning*, ch. 2 and 5 (random walks, propagation, GNNs) — <https://www.cs.mcgill.ca/~wlh/grl_book/> (free)
* Langville & Meyer, "Deeper inside PageRank" (*Internet Mathematics*, 2004), the classic mathematical survey — <https://www.stat.uchicago.edu/~lekheng/meetings/mathofranking/ref/langville.pdf> (free)
* Gleich, "PageRank beyond the Web" (*SIAM Review*, 2015), PageRank across science including biology — <https://arxiv.org/abs/1407.5107> (free)
* Cowen, Ideker, Raphael & Sharan, "Network propagation: a universal amplifier of genetic associations" (*Nat. Rev. Genet.*, 2017) — <https://doi.org/10.1038/nrg.2017.38> (paid)

**Original papers**
* Brin & Page, "The anatomy of a large-scale hypertextual web search engine" (1998) — <http://infolab.stanford.edu/~backrub/google.html> (free)
* Zhou, Bousquet, Lal, Weston & Schölkopf, "Learning with local and global consistency" (NeurIPS 2003) — <https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html> (free)
* Zhu & Ghahramani, "Learning from labeled and unlabeled data with label propagation" (CMU tech report CMU-CALD-02-107, 2002) — <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf> (free)
* Tong, Faloutsos & Pan, "Fast random walk with restart and its applications" (ICDM 2006) — <https://doi.org/10.1109/ICDM.2006.70> (paid)
* Köhler, Bauer, Horn & Robinson, "Walking the interactome for prioritization of candidate disease genes" (*AJHG*, 2008), RWR in biology — <https://doi.org/10.1016/j.ajhg.2008.02.013> (free via PMC2427257)
* Chen, Liu & Yan, "Drug–target interaction prediction by random walk on the heterogeneous network" (NRWRH; *Mol. BioSyst.*, 2012) — <https://doi.org/10.1039/c2mb00002d> (paid; PubMed 22538619)
* Xie, Hwang & Kuang, "Prioritizing disease genes by bi-random walk" (PAKDD 2012) — <https://doi.org/10.1007/978-3-642-30220-6_25> (paid)
* Luo et al., "Drug repositioning based on comprehensive similarity measures and Bi-Random walk algorithm" (MBiRW; *Bioinformatics*, 2016) — <https://doi.org/10.1093/bioinformatics/btw228> (paid; PubMed 27153662; read the Methods section)

**Propagation meets GNNs**
* Klicpera, Bojchevski & Günnemann, "Predict then propagate: graph neural networks meet personalized PageRank" (APPNP, ICLR 2019) — <https://arxiv.org/abs/1810.05997> (free)
* Huang, He, Singh, Lim & Benson, "Combining label propagation and simple models out-performs graph neural networks" (Correct & Smooth, ICLR 2021) — <https://arxiv.org/abs/2010.13993> (free)
* Ferrari Dacrema, Cremonesi & Jannach, "Are we really making much progress? A worrying analysis of recent neural recommendation approaches" (RecSys 2019), on how strong simple neighbourhood baselines are — <https://arxiv.org/abs/1907.06902> (free)

---

## 17. Glossary

* **Absorbing state** — a state a Markov chain never leaves; labelled nodes in harmonic label propagation.
* **Aperiodic** — a chain that does not cycle with a fixed period; required for convergence of $\mathbf p_t$.
* **Bi-random walk (BiRW)** — two coupled random walks on two networks joined by an association matrix; MBiRW applies it to drugs and diseases.
* **Chapman–Kolmogorov relation** — $(P^t)_{ij}$ is the $t$-step transition probability.
* **Cold start** — predicting for an entity (here, a disease) with no known links.
* **Column-stochastic** — non-negative matrix whose columns sum to 1; multiplying a distribution by it conserves mass.
* **Damping factor** — PageRank's probability $\beta$ of following a link instead of teleporting.
* **Dangling node** — a node with no out-links; its transition row is undefined until fixed.
* **Degree (popularity) bias** — tendency of long walks to favour high-degree nodes.
* **Degree gate** — MV-HGAT's learned factor that shrinks the GNN term for entities with few visible links.
* **Detailed balance / reversibility** — $\pi_iP_{ij}=\pi_jP_{ji}$; holds for walks on undirected graphs.
* **Gelfand's formula** — $\rho(M)=\lim_t\lVert M^t\rVert^{1/t}$.
* **Google matrix** — $G=\beta P+(1-\beta)\mathbf 1\mathbf v^\top$.
* **Guilt by association** — similar entities share links; the assumption behind all propagation methods.
* **Harmonic function** — values equal to the weighted average of neighbours at every unlabelled node.
* **Heterogeneous network** — a graph with several node types (drugs, diseases, targets).
* **Irreducible** — every state reachable from every other (connected graph).
* **Jumping probability ($\lambda$)** — in NRWRH, probability of crossing from one node type to the other.
* **kNN kernel** — row-normalised matrix of each node's $k$ most similar other nodes (`knn_kernel`).
* **Label propagation** — spreading known labels over a graph by repeated neighbour averaging.
* **Lazy walk** — $\tfrac12(I+P)$; stays put half the time; removes periodicity.
* **Left / right walk** — MBiRW's walks on the drug network ($M_rR$) and the disease network ($RM_d$).
* **Logistic similarity adjustment** — MBiRW's soft threshold $L(x)=1/(1+e^{cx+d})$.
* **Markov chain / Markov property** — a random process whose next state depends only on the current state.
* **Neumann series** — $(I-M)^{-1}=\sum_tM^t$ when $\rho(M)<1$.
* **Normalised Laplacian** — $\mathcal L = I-D^{-1/2}SD^{-1/2}$; its quadratic form measures non-smoothness.
* **NRWRH** — random walk with restart on a heterogeneous network (Chen et al. 2012).
* **Over-smoothing** — scores or embeddings becoming indistinguishable after too much propagation.
* **PageRank** — stationary distribution of a link-following walk with uniform teleportation.
* **Perron–Frobenius theorem** — guarantees a unique positive stationary distribution for irreducible non-negative matrices.
* **Personalised PageRank (PPR)** — PageRank with teleportation to a chosen seed set.
* **Propagation head** — MV-HGAT's term $\sum_vw_vP_v[i,j]$.
* **Random walk with restart (RWR)** — a walk that returns to the seed with probability $1-\alpha$ at every step.
* **Row-stochastic** — non-negative matrix whose rows sum to 1.
* **Spectral gap** — $1-|\lambda_2|$; larger means faster mixing.
* **Spectral radius** — $\rho(M)=\max|\lambda_k(M)|$; decides convergence of linear iterations.
* **Stationary distribution** — $\boldsymbol\pi^\top P=\boldsymbol\pi^\top$.
* **Sylvester equation** — $aX+Xb=q$; the fixed point of the infinite bi-random walk.
* **Teleportation** — PageRank's random jump to a node drawn from $\mathbf v$.
