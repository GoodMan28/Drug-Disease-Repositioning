# Unit B4 — Matrix factorisation and recommender systems

> **Course:** Drug repositioning with graph neural networks — a self-contained course
> **Chapter 10 of 20** · Track B (Machine learning)

| | |
|---|---|
| **Prerequisites** | 02 A2 Linear algebra (rank, SVD, norms, graph Laplacian); 03 A3 Probability (likelihood, Bernoulli, Gaussian priors); 04 B1 Supervised learning (regularisation, gradient descent); 05 B2 Evaluation (AUC, AUPR, imbalance); 07 B3 PyTorch (autograd, Adam); 09 C2 Random walks and propagation (kNN propagation, Laplacian smoothing) |
| **Estimated study time** | 10–12 hours (about 5 h reading and derivations, 2 h code, 4 h exercises) |
| **Leads to** | 11 C3 GCNs (NIMCGCN, LAGCN), 14 C6 Link prediction (decoders, negative sampling), 18 E1 Leakage (what counts as a negative) |

## Learning objectives

After this chapter you will be able to:

1. **Translate** drug repositioning into recommender-system language (users, items, implicit feedback, cold start) and **implement** user-based and item-based neighbourhood collaborative filtering.
2. **Explain** why an unknown drug–disease pair is not a negative, name the three standard ways of training with only positives (weighting, negative sampling, ranking), and **quantify** how the negative-sampling ratio shifts predicted probabilities.
3. **Justify** the low-rank assumption with the SVD and the Eckart–Young theorem, and **compute** a best rank-$k$ approximation.
4. **Derive** the matrix-factorisation objective from a probabilistic model, and **derive and implement** both alternating least squares (ALS) and stochastic gradient descent (SGD) for it, including the weighted (implicit-feedback) version.
5. **Derive** the BPR loss from a Bayesian argument and **relate** it to AUC.
6. **Prove** the identity $\operatorname{tr}(U^\top LU) = \tfrac12\sum_{ij}S_{ij}\lVert\mathbf u_i-\mathbf u_j\rVert^2$, and **explain** how SCMFDD uses it to give cold entities sensible factors.
7. **State and sketch the proof** that the nuclear norm is the convex envelope of rank on the spectral-norm unit ball; **derive** the singular value thresholding (SVT) operator and **implement** the Cai–Candès–Shen algorithm.
8. **Describe** DRRS's heterogeneous block matrix, BNNR, inductive matrix completion and NIMCGCN, and **show** that a bilinear decoder is a learned matrix factorisation.
9. **Read** the project's `SCMFDD`, `DRRS`, `NIMCGCN` and MV-HGAT decoder code line by line and **interpret** their Fdataset results.

---

## 1. Motivation: drug repositioning *is* a recommendation problem

In 2006 Netflix offered one million dollars to anyone who could predict users' movie ratings 10% more accurately than its own system. The winning teams' central tool was **matrix factorisation**: describe every user and every movie by a short vector of hidden "taste" factors, and predict a rating by the inner product of the two vectors. Recommender systems have used the idea ever since.

Our problem has exactly the same shape. Replace users by **diseases**, movies by **drugs** and "watched and liked" by **known indication**:

| Recommender system | Drug repositioning (this project) |
|---|---|
| user | disease (313 in Fdataset) |
| item | drug (593 in Fdataset) |
| user–item interaction (purchase, click, play) | known indication, $A_{ij}=1$ (1,933 in Fdataset) |
| "recommend items to a user" | rank candidate drugs for a disease |
| user side information (age, country) | disease similarity views (`pheno_mim`, `sem_mondo`, `gene_d`) |
| item side information (genre, actors) | drug similarity views (`chem_cdk`, `chem_ecfp`, `gene_r`) |
| new user with no history | cold-start disease (leave-one-disease-out) |
| blockbuster item | drug with many indications (e.g. a corticosteroid) |
| unwatched movie | unknown pair — *not* a known "dislike" |

(The project stores $A$ as drugs × diseases, i.e. items × users; the mathematics is symmetric, so we will freely transpose.)

**This matters for the project in three concrete ways.**

1. **Two of the five baselines are matrix-factorisation / completion methods.** SCMFDD (similarity-constrained MF) reaches AUC 0.893 / **AUPR 0.495** on Fdataset 5-fold CV — the *highest* AUPR of all methods, slightly above our MV-HGAT (0.939 / 0.488). DRRS (matrix completion by singular value thresholding) reaches 0.879 / 0.389. You cannot interpret the results table without understanding what these methods do and why they are strong.
2. **Our own decoder is a learned factorisation.** MV-HGAT scores a pair by $\mathbf h_i^\top W\mathbf h_j$: an inner product of a drug factor and a disease factor, where the factors are produced by a graph neural network instead of being free parameters.
3. **The training labels are recommender-style implicit feedback.** We only know positives; the model is trained with **negative sampling** (`neg_ratio=2`) — a technique that comes straight from implicit-feedback recommendation.

> **Roadmap.** Section 2: collaborative filtering and the analogy. Section 3: implicit feedback and why unknown ≠ negative. Section 4: the low-rank assumption. Section 5: MF, ALS and SGD. Section 6: weighted MF. Section 7: negative sampling. Section 8: BPR. Section 9: graph-regularised MF (SCMFDD). Sections 10–11: matrix completion, nuclear norm, SVT. Section 12: DRRS and BNNR. Section 13: inductive matrix completion and NIMCGCN. Section 14: bilinear decoders.

---

## 2. Collaborative filtering

### 2.1 Three ways to recommend

* **Content-based**: recommend items whose *features* resemble items the user liked (a chemically similar drug). Needs features, ignores other users.
* **Collaborative filtering (CF)**: recommend items that *similar users* liked, where similarity is learned from the interaction matrix itself ("people who bought this also bought…"). Needs no features, but cannot handle a user or item with no interactions.
* **Hybrid**: combine both. Nearly every modern system, including MV-HGAT, is hybrid.

CF comes in two families: **neighbourhood methods** (this section) and **latent-factor methods** (matrix factorisation, Sections 4–9).

### 2.2 Neighbourhood CF: user-based and item-based

**Item-based CF** (here: drug-based). Two drugs are similar if they are used for the same diseases. With $\mathbf a_i$ the row of drug $i$, a common choice is cosine similarity
$$\text{sim}(i,i') = \frac{\mathbf a_i^\top\mathbf a_{i'}}{\lVert\mathbf a_i\rVert\,\lVert\mathbf a_{i'}\rVert}.$$
The score of drug $i$ for disease $j$ is a similarity-weighted average over the other drugs' links to $j$:
$$\hat a_{ij} = \frac{\sum_{i'\ne i}\text{sim}(i,i')\,A_{i'j}}{\sum_{i'\ne i}\text{sim}(i,i')}.$$
In practice one keeps only the $k$ most similar items.

**User-based CF** (here: disease-based) is the transpose: diseases are similar if they are treated by the same drugs, and $\hat a_{ij}$ averages drug $i$'s links to diseases similar to $j$.

**The project's propagation head is neighbourhood CF with side information.** In Chapter 09 you met $P_v = K_vA$ (drug view) and $P_u = AK_u^\top$ (disease view), where $K$ is the row-normalised $k$-nearest-neighbour kernel. These are *exactly* the item-based and user-based formulas above, except that the similarity comes from chemistry or phenotype rather than from co-occurrence in $A$. That difference is crucial for cold start: co-occurrence similarity of a disease with no drugs is undefined, while phenotype similarity is always available.

#### Worked example 2.1 (by hand)

Six drugs, five diseases (asthma, COPD, rheumatoid arthritis, lupus, psoriasis):
$$A=\begin{array}{c|ccccc}
&\text{asth}&\text{COPD}&\text{RA}&\text{lupus}&\text{psor}\\\hline
d_0&1&1&0&0&0\\
d_1&1&1&0&0&0\\
d_2&0&1&0&0&0\\
d_3&0&0&1&1&0\\
d_4&0&0&1&1&1\\
d_5&0&0&1&0&1
\end{array}$$
Cosine similarity of $d_2$ with $d_0$: $\mathbf a_2^\top\mathbf a_0 = 1$, $\lVert\mathbf a_2\rVert = 1$, $\lVert\mathbf a_0\rVert = \sqrt2$, so $\text{sim} = 1/\sqrt2\approx0.707$; the same with $d_1$; 0 with $d_3,d_4,d_5$. Item-based score of $d_2$ for asthma: $(0.707\cdot1 + 0.707\cdot1)/(0.707+0.707) = 1$. Drug 2 (a COPD drug) is recommended for asthma because both of its "neighbours" treat asthma — a sensible repositioning hypothesis. For $d_0$ and asthma (a known link), the neighbours are $d_1$ (sim 1) and $d_2$ (0.707): $(1\cdot1 + 0.707\cdot0)/1.707 = 0.586$. Section 15 reproduces the full matrices.

### 2.3 Why go beyond neighbours?

Neighbourhood CF only "sees" one hop: two drugs with no disease in common have similarity 0, even if they treat closely related diseases. Latent-factor models compress the whole matrix into a few factors, so evidence can flow through *chains* of co-occurrence, and noise is averaged out. The price: they need every entity to have some data (or side information) to estimate its factors.

---

## 3. Explicit versus implicit feedback: unknown ≠ negative

### 3.1 Two kinds of data

* **Explicit feedback**: the user states a preference, positive *or negative* — a 1-star rating is real evidence of dislike. Missing ratings are simply missing.
* **Implicit feedback**: we observe behaviour (purchases, clicks, listening time). We see only positive evidence. A missing interaction is **ambiguous**: the user may dislike the item, or may never have encountered it.

Drug–disease indications are **implicit feedback**. A 1 in $A$ means a recorded indication. A 0 lumps together:

1. pairs that were tested and failed (true negatives — rarely recorded in these databases);
2. pairs nobody has ever tested (the vast majority — *unknown*);
3. pairs that work but are not yet discovered or recorded — **exactly the repositioning opportunities we want to find**.

So a model that learned "every 0 is a negative" would be trained to *suppress* the very discoveries it is meant to make. This is also called **positive–unlabelled (PU) learning**. Worse, the missingness is **not at random**: well-studied, older drugs have many recorded indications; rare diseases have few. The pattern of zeros reflects research attention as much as biology.

### 3.2 One-class collaborative filtering

Pan et al. (2008) named this setting **one-class collaborative filtering (OCCF)** and described the two naive extremes:

* **AMAN — all missing as negative.** Treat every 0 as a true 0 and fit the full matrix. Simple, uses all cells, but biased: true-but-unknown positives are pushed down.
* **AMAU — all missing as unknown.** Ignore the zeros completely and fit only the 1s. Degenerate: predicting 1 everywhere fits the data perfectly.

Every practical method sits in between:

| Strategy | Idea | Where in this project |
|---|---|---|
| **Weighting** | use all zeros, but with low confidence | `weighted_bce` (NIMCGCN, LAGCN) up-weights positives; SCMFDD is plain AMAN |
| **Sampling** | each epoch, treat a small random sample of unknowns as negatives | MV-HGAT, `neg_ratio=2` |
| **Ranking** | only require a positive to score above an unknown | BPR (Section 8) |
| **Completion** | treat zeros as *unobserved* and constrain the solution (low rank) | DRRS, BNNR (Section 12) |

### 3.3 What this means for training labels and evaluation

* **Training labels.** A "negative" label in training means "*assumed* negative for this gradient step", not "known not to work". Methods should not be too confident about any single unknown — hence low weights, random re-sampling every epoch, or ranking losses.
* **Test negatives.** The project's evaluation (`evaluation.py`) treats all unknown pairs in the test fold as negatives. Some of them are undiscovered true indications, so the measured AUPR *underestimates* the real precision, and a highly-ranked "false positive" may be a correct prediction (the case studies in `scripts/06_case_study.py` check top predictions against ClinicalTrials.gov and the literature for exactly this reason).
* **Leakage rule.** Negatives must never be drawn from test cells: `neg_mask` excludes them.

Section 15's synthetic experiment makes this concrete: of the 100 best-scored test pairs that the data *label* as unknown, 8–17% (depending on the training loss) are in fact true but unrecorded indications, against a base rate of about 0.8% among all test unknowns.

---

## 4. The low-rank assumption

### 4.1 Latent factors

Suppose every drug can be described by $k$ hidden numbers — how strongly it is anti-inflammatory, how strongly it blocks a dopamine receptor, how much it suppresses the immune system — and every disease by how much it would benefit from each of those same $k$ mechanisms. If a drug treats a disease when their mechanism profiles align, then
$$A_{ij}\approx\sum_{f=1}^k U_{if}V_{jf} = \mathbf u_i^\top\mathbf v_j,\qquad A\approx UV^\top,\quad U\in\mathbb R^{n\times k},\ V\in\mathbb R^{m\times k}.$$
A product $UV^\top$ has **rank at most $k$**: it can be written as a sum of $k$ outer products $\sum_f\mathbf U_{:,f}\mathbf V_{:,f}^\top$. The **low-rank assumption** is that $k\ll\min(n,m)$ factors explain most of the matrix.

Why it is plausible for drug–disease data:
* there are far fewer *mechanisms* than drugs or diseases;
* drugs come in classes (statins, corticosteroids, β-blockers) whose rows look alike;
* diseases come in families (epilepsies, cardiomyopathies) whose columns look alike;
* so $A$ has an approximate **block structure**, and block-structured matrices are approximately low-rank.

Where it fails: unique drugs with idiosyncratic uses, and noise. Low rank is an approximation, enforced *softly*.

### 4.2 The SVD and the best rank-$k$ approximation

Every real $n\times m$ matrix has a **singular value decomposition**
$$A = U_A\Sigma V_A^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^\top,\qquad \sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0,$$
with orthonormal columns in $U_A$, $V_A$ and $r=\operatorname{rank}(A)$.

**Eckart–Young–Mirsky theorem.** For any $k<r$, the truncated SVD $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^\top$ is a best rank-$k$ approximation in both the Frobenius and spectral norms, with
$$\min_{\operatorname{rank}(X)\le k}\lVert A-X\rVert_F = \lVert A-A_k\rVert_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2},\qquad \min_{\operatorname{rank}(X)\le k}\lVert A-X\rVert_2 = \sigma_{k+1}.$$
*Sketch (spectral norm).* If $\operatorname{rank}(X)\le k$, its null space has dimension at least $m-k$, so it intersects the $(k+1)$-dimensional span of $\mathbf v_1,\dots,\mathbf v_{k+1}$ in some unit vector $\mathbf w$. Then $\lVert(A-X)\mathbf w\rVert = \lVert A\mathbf w\rVert\ge\sigma_{k+1}$. $\square$ (The Frobenius case follows from a similar argument applied to each $\sigma_{k+i}$, or from Weyl's inequalities.)

#### Worked example 4.1

The 6×5 matrix of worked example 2.1 is block-diagonal: a respiratory block $B=\begin{pmatrix}1&1\\1&1\\0&1\end{pmatrix}$ and an immunology block $C=\begin{pmatrix}1&1&0\\1&1&1\\1&0&1\end{pmatrix}$. The singular values of a block-diagonal matrix are those of its blocks. For $B$: $B^\top B=\begin{pmatrix}2&2\\2&3\end{pmatrix}$, eigenvalues $\tfrac{5\pm\sqrt{17}}2 = 4.562, 0.438$, so $\sigma = 2.136, 0.662$. For $C$ (Exercise 2): $\sigma = 2.414, 1, 0.414$. All five: $(2.414, 2.136, 1, 0.662, 0.414)$. Two large values, one per disease family — the matrix is "approximately rank 2". The rank-2 truncation keeps one factor per family; it assigns drug 2 a score of 0.485 for asthma (a plausible repositioning) and exactly 0 for every cross-family pair; its error is $\sqrt{1^2+0.662^2+0.414^2} = 1.269$, exactly as Eckart–Young predicts (Section 15).

### 4.3 Why not just take the SVD of $A$?

Truncated SVD of the 0/1 matrix ("PureSVD"; Cremonesi et al. 2010 found it a surprisingly strong top-N recommender) is an AMAN method: every zero is fitted as a real zero with full weight. It also cannot add regularisation, side information, or weights. Matrix factorisation generalises it.

---

## 5. Matrix factorisation: model, objective, ALS and SGD

### 5.1 The model

$$\hat a_{ij} = \mathbf u_i^\top\mathbf v_j\qquad(\text{optionally } + \mu + b_i + c_j)$$
with a global offset $\mu$ and per-drug and per-disease **biases** $b_i,c_j$ that capture "this drug is used a lot" and "this disease has many drugs". Biases matter in practice; we omit them in derivations for clarity.

### 5.2 The objective, and where it comes from

Let $\Omega$ be the set of observed cells. The regularised least-squares objective is
$$\boxed{\;\mathcal J(U,V) = \sum_{(i,j)\in\Omega}\big(a_{ij}-\mathbf u_i^\top\mathbf v_j\big)^2 + \lambda\big(\lVert U\rVert_F^2+\lVert V\rVert_F^2\big)\;}$$

**Probabilistic derivation (probabilistic matrix factorisation, PMF; Mnih & Salakhutdinov 2007).** Assume $a_{ij}\sim\mathcal N(\mathbf u_i^\top\mathbf v_j,\sigma^2)$ independently for $(i,j)\in\Omega$, and Gaussian priors $\mathbf u_i\sim\mathcal N(\mathbf 0,\sigma_U^2I)$, $\mathbf v_j\sim\mathcal N(\mathbf 0,\sigma_V^2I)$. The negative log-posterior is
$$-\ln p(U,V\mid A) = \frac{1}{2\sigma^2}\sum_{\Omega}(a_{ij}-\mathbf u_i^\top\mathbf v_j)^2 + \frac{1}{2\sigma_U^2}\lVert U\rVert_F^2 + \frac{1}{2\sigma_V^2}\lVert V\rVert_F^2 + \text{const}.$$
Multiplying by $2\sigma^2$, maximising the posterior (MAP) is minimising $\mathcal J$ with $\lambda = \sigma^2/\sigma_U^2$ (taking $\sigma_U=\sigma_V$). **L2 regularisation = a Gaussian prior on the factors.** For binary data, replacing the Gaussian likelihood by a Bernoulli one with $\Pr(a_{ij}=1) = \sigma(\mathbf u_i^\top\mathbf v_j)$ gives **logistic MF**, trained with binary cross-entropy — what MV-HGAT does with its learned factors.

**Properties.**
* $\mathcal J$ is **not convex** jointly in $(U,V)$ (because of the product), but it is **biconvex**: convex (a ridge regression) in $U$ for fixed $V$ and vice versa. ALS exploits this.
* **Non-identifiability.** For any invertible $Q$, $(UQ)(VQ^{-\top})^\top = UV^\top$. Regularisation removes all but orthogonal $Q$ (rotations). So individual latent dimensions have no fixed meaning; only inner products do. Beware of interpreting "factor 3" biologically.
* **Hidden convexity.** $\min_{UV^\top=X}\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2) = \lVert X\rVert_*$, the nuclear norm (Section 10.4, Exercise 9). L2-regularised MF is therefore secretly nuclear-norm-regularised matrix completion, written in factored form.

### 5.3 Alternating least squares (ALS)

**Derivation.** Fix $V$. Then $\mathcal J$ splits into one independent problem per drug:
$$\min_{\mathbf u_i}\ \sum_{j\in\Omega_i}(a_{ij}-\mathbf v_j^\top\mathbf u_i)^2 + \lambda\lVert\mathbf u_i\rVert^2,$$
where $\Omega_i$ are the observed diseases of drug $i$. This is ridge regression with design matrix $V_{\Omega_i}$ (the rows of $V$ for $j\in\Omega_i$) and targets $\mathbf a_{i,\Omega_i}$. Setting the gradient to zero:
$$-2V_{\Omega_i}^\top(\mathbf a_{i,\Omega_i}-V_{\Omega_i}\mathbf u_i)+2\lambda\mathbf u_i = 0
\quad\Longrightarrow\quad
\boxed{\;\mathbf u_i = \big(V_{\Omega_i}^\top V_{\Omega_i}+\lambda I_k\big)^{-1}V_{\Omega_i}^\top\mathbf a_{i,\Omega_i}\;}$$
The matrix is $k\times k$ and positive definite for $\lambda>0$, so the solution is unique. By symmetry, with $U$ fixed,
$$\mathbf v_j = \big(U_{\Omega^j}^\top U_{\Omega^j}+\lambda I_k\big)^{-1}U_{\Omega^j}^\top\mathbf a_{\Omega^j,j}.$$
ALS alternates the two half-steps. Each half-step *exactly minimises* $\mathcal J$ over one block, so $\mathcal J$ never increases; it converges to a stationary point (not necessarily the global minimum).

**Full-matrix (AMAN) special case.** If every cell is observed, $V_{\Omega_i}=V$ for all $i$ and all rows can be solved at once:
$$U = AV\,(V^\top V+\lambda I)^{-1},\qquad V = A^\top U\,(U^\top U+\lambda I)^{-1}.$$
**Cost.** Per sweep, $O(|\Omega|k^2 + (n+m)k^3)$; rows are independent, so ALS parallelises perfectly.

#### Worked example 5.1 (by hand): one ALS sweep

$A=\begin{pmatrix}1&0\\1&1\end{pmatrix}$ (2 drugs × 2 diseases, all cells observed), $k=1$, $\lambda=1$, start with $V = (1,1)^\top$.
* $U$-step: $V^\top V+\lambda = 2+1 = 3$. $u_0 = (1\cdot1+0\cdot1)/3 = \tfrac13$; $u_1 = (1+1)/3 = \tfrac23$.
* $V$-step: $U^\top U + \lambda = \tfrac19+\tfrac49+1 = \tfrac{14}9$. $v_0 = (\tfrac13\cdot1+\tfrac23\cdot1)\cdot\tfrac9{14} = \tfrac9{14}\approx0.643$; $v_1 = (\tfrac13\cdot0+\tfrac23\cdot1)\cdot\tfrac9{14} = \tfrac37\approx0.429$.
* Predictions $UV^\top = \begin{pmatrix}0.214&0.143\\0.429&0.286\end{pmatrix}$.
The unknown cell $(0,1)$ gets 0.143: drug 0 shares disease 0 with drug 1, which treats disease 1. The values are shrunk well below 1 because $\lambda=1$ is large for a 2×2 problem.

### 5.4 Stochastic gradient descent (SGD)

Write the objective as a sum over observed cells, spreading the regulariser over them:
$$\mathcal J = \sum_{(i,j)\in\Omega}\Big[(a_{ij}-\mathbf u_i^\top\mathbf v_j)^2 + \lambda(\lVert\mathbf u_i\rVert^2+\lVert\mathbf v_j\rVert^2)\Big]\quad(\text{a common, slightly re-weighted variant}).$$
For one cell, with error $e_{ij} = a_{ij}-\mathbf u_i^\top\mathbf v_j$, the gradients are $\partial/\partial\mathbf u_i = -2e_{ij}\mathbf v_j + 2\lambda\mathbf u_i$ and $\partial/\partial\mathbf v_j = -2e_{ij}\mathbf u_i+2\lambda\mathbf v_j$. Absorbing the 2 into the learning rate $\eta$:
$$\boxed{\;\mathbf u_i\leftarrow\mathbf u_i+\eta\,(e_{ij}\mathbf v_j-\lambda\mathbf u_i),\qquad\mathbf v_j\leftarrow\mathbf v_j+\eta\,(e_{ij}\mathbf u_i-\lambda\mathbf v_j)\;}$$
(both updates use the *old* values). Loop over observed cells in random order, for several epochs. This is the algorithm Simon Funk popularised during the Netflix Prize.

#### Worked example 5.2 (by hand): one SGD step

$k=1$, $u=0.5$, $v=0.4$, observed $a=1$, $\eta=0.1$, $\lambda=0.1$. Error $e = 1-0.2 = 0.8$.
$u\leftarrow0.5+0.1(0.8\cdot0.4-0.1\cdot0.5) = 0.5+0.1\cdot0.27 = 0.527$;
$v\leftarrow0.4+0.1(0.8\cdot0.5-0.1\cdot0.4) = 0.4+0.1\cdot0.36 = 0.436$.
New prediction $0.527\times0.436\approx0.230$, up from 0.200.

### 5.5 ALS versus SGD

| | ALS | SGD |
|---|---|---|
| Step | exact minimisation of a block | small gradient step on one cell |
| Hyper-parameters | $\lambda$, $k$, number of sweeps | $\lambda$, $k$, learning rate, epochs |
| Convergence | monotone, few sweeps | noisy, needs tuning of $\eta$ |
| Weighted/implicit data (all cells) | efficient (Section 6) | cost grows with number of cells |
| Flexibility (non-quadratic losses, BPR, side information) | limited | any differentiable loss (autograd) |
| Parallelism | rows independent | needs care (conflicting updates) |

The project's SCMFDD uses a third option: full-batch gradient descent with Adam on the whole objective, letting PyTorch compute gradients. It is simple and works on a GPU.

---

## 6. Weighted matrix factorisation for implicit feedback

### 6.1 The Hu–Koren–Volinsky model

Hu, Koren and Volinsky (2008) proposed the standard latent-factor model for implicit feedback. Split each observation into a **preference** and a **confidence**:
$$p_{ij} = \mathbb 1[r_{ij}>0],\qquad c_{ij} = 1+\alpha\,r_{ij},$$
where $r_{ij}\ge0$ is the raw interaction strength (number of plays, minutes watched). Then fit **all** cells, weighted by confidence:
$$\min_{U,V}\ \sum_{i,j}c_{ij}\big(p_{ij}-\mathbf u_i^\top\mathbf v_j\big)^2 + \lambda\big(\lVert U\rVert_F^2+\lVert V\rVert_F^2\big).$$
Every unknown cell is a *weak* zero (confidence 1); every observed interaction is a *strong* one (confidence $1+\alpha$; the paper reports $\alpha=40$ working well for TV-viewing data). For binary drug–disease data, $r_{ij} = A_{ij}$, so known links get weight $1+\alpha$ and unknowns weight 1. This is the **weighting** answer to "unknown ≠ negative": unknowns still pull scores down, but each one only a little.

### 6.2 ALS for weighted MF, and the speed trick

With $C^i = \mathrm{diag}(c_{i1},\dots,c_{im})$, the row update is weighted ridge regression:
$$\mathbf u_i = \big(V^\top C^iV+\lambda I\big)^{-1}V^\top C^i\mathbf p_i .$$
Computing $V^\top C^iV$ naively costs $O(mk^2)$ per row — too slow when $m$ is large. The trick:
$$V^\top C^iV = V^\top V + V^\top(C^i-I)V .$$
$V^\top V$ is shared by all rows (computed once per sweep), and $C^i-I$ is non-zero only on the $n_i$ observed cells of row $i$, so the second term costs $O(n_ik^2)$. Likewise $V^\top C^i\mathbf p_i$ only involves observed cells (since $p_{ij}=0$ elsewhere). A full sweep then costs $O(|\Omega|k^2+(n+m)k^3)$, linear in the number of observations even though all $n\times m$ cells are in the loss. Section 15 implements it.

### 6.3 Weighting in this project

`weighted_bce` in `src/drepo/methods.py` (used by NIMCGCN and LAGCN) is the cross-entropy analogue: all allowed cells are used, and positives are up-weighted by $\#\text{negatives}/\#\text{positives}$ (about 95 on Fdataset) so that both classes contribute equally to the loss. SCMFDD uses the pure AMAN choice: squared error with weight 1 on every cell of the training matrix.

---

## 7. Negative sampling

### 7.1 The idea

Instead of putting every unknown cell in the loss, **sample** a few unknown cells per epoch and treat them as negatives:
$$\mathcal L = -\sum_{(i,j)\in\mathcal P}\ln\sigma(s_{ij}) - \sum_{(i,j)\in\mathcal N_t}\ln\big(1-\sigma(s_{ij})\big),\qquad |\mathcal N_t| = \rho\,|\mathcal P|,$$
where $\mathcal P$ are the positives being supervised, $\mathcal N_t$ a fresh uniform random sample of unknown cells at epoch $t$, and $\rho$ the **negative sampling ratio**. Three benefits:

1. **Cost.** Each epoch touches $(1+\rho)|\mathcal P|$ cells instead of $n\times m$.
2. **Balance.** The loss sees a 1 : $\rho$ class ratio rather than 1 : 95.
3. **Soft negatives.** Each particular unknown cell is labelled negative only occasionally. A true-but-unknown indication is pushed down rarely, while the model is pulled up by consistent evidence from similar positives. Re-sampling every epoch is what makes this work; a fixed sample would harden those cells into permanent negatives.

### 7.2 What the ratio does

* **Small $\rho$** (1–2): balanced, fast; each epoch's estimate of "what negatives look like" is noisier.
* **Large $\rho$** (10–20): more negatives per positive, closer to AMAN; more true-but-unknown positives get labelled negative in each epoch, and the loss becomes dominated by easy negatives. In the synthetic experiment of Section 15, ratio 20 lowered AUPR from about 0.20 to 0.09.
* The project tuned $\rho\in\{1,2,5,10\}$ on a validation split (`scripts/05_sensitivity.py`) and uses **$\rho=2$**.

### 7.3 Negative sampling shifts probabilities but (ideally) not rankings

Suppose positives are kept with probability $s_+$ and negatives with probability $s_-$. Among the *training* examples at input $x$, Bayes' rule gives
$$\frac{\Pr_{\text{train}}(y=1\mid x)}{\Pr_{\text{train}}(y=0\mid x)} = \frac{s_+\Pr(y=1\mid x)}{s_-\Pr(y=0\mid x)}
\quad\Longrightarrow\quad
\text{logit}_{\text{train}}(x) = \text{logit}_{\text{true}}(x) + \ln\frac{s_+}{s_-}.$$
A model that fits the sampled data perfectly therefore outputs logits shifted by the **constant** $\ln(s_+/s_-)$. Adding a constant does not change the order of the scores, so AUC and AUPR (which depend only on the ranking) are unaffected in this ideal case; but the predicted probabilities are inflated and should not be read as real probabilities of a successful repositioning. In Section 15 the mean predicted probability on unknown cells falls from 0.39 to 0.18 as $\rho$ goes from 1 to 20 — the shift in action. (In practice, finite capacity and regularisation mean the ranking does change somewhat with $\rho$, which is why it is tuned.)

### 7.4 Sampling distributions

Uniform sampling over unknown cells is the default (and what the project does: `torch.randint` over `neg_pool`, with replacement). Alternatives: **popularity-based** sampling (pick negatives involving popular drugs more often, to stop the model from simply ranking by popularity), and **hard-negative** sampling (pick unknowns the current model scores highly). Hard negatives are risky here: in PU data the highest-scored unknowns are disproportionately *true* positives.

---

## 8. Bayesian personalised ranking (BPR)

### 8.1 From points to pairs

For ranking we do not need $\sigma(s_{ij})$ to be a calibrated probability; we need positives to be ranked **above** unknowns. Rendle et al. (2009) formalised this. For a user $u$ (here: a disease), every observed item $i$ (a known drug) should be preferred to every unobserved item $j$ (an unknown drug):
$$\mathcal D_S = \{(u,i,j): i\in I_u^+,\ j\notin I_u^+\}.$$
Note that this does *not* claim $j$ is bad, only that $i$ is (more likely) better — a much weaker and more honest assumption for PU data.

### 8.2 Derivation of the BPR criterion

Let $\hat x_{ui}$ be the model's score and $\hat x_{uij} = \hat x_{ui}-\hat x_{uj}$. Model the probability that $u$ prefers $i$ to $j$ as $\Pr(i>_uj\mid\Theta) = \sigma(\hat x_{uij})$, assume pairs are independent, and put a Gaussian prior $\Theta\sim\mathcal N(\mathbf 0,\lambda^{-1}I)$ on the parameters. The log-posterior is
$$\ln p(\Theta\mid >) = \sum_{(u,i,j)\in\mathcal D_S}\ln\sigma(\hat x_{uij}) - \frac{\lambda}{2}\lVert\Theta\rVert^2 + \text{const}.$$
Maximising it is **BPR-Opt**; equivalently minimise the **BPR loss**
$$\boxed{\;\mathcal L_{\text{BPR}} = -\sum_{(u,i,j)}\ln\sigma(\hat x_{ui}-\hat x_{uj}) + \lambda\lVert\Theta\rVert^2\;}$$

**Relation to AUC.** The per-user AUC is
$$\text{AUC}(u) = \frac{1}{|I_u^+|\,|I\setminus I_u^+|}\sum_{i\in I_u^+}\sum_{j\notin I_u^+}\mathbb 1[\hat x_{uij}>0].$$
BPR replaces the non-differentiable step $\mathbb 1[x>0]$ by the smooth $\ln\sigma(x)$. So **BPR is a differentiable surrogate for the average per-user AUC**.

### 8.3 Gradients and the LearnBPR algorithm

Since $\frac{d}{dx}\ln\sigma(x) = 1-\sigma(x) = \sigma(-x)$,
$$\frac{\partial}{\partial\theta}\ln\sigma(\hat x_{uij}) = \sigma(-\hat x_{uij})\,\frac{\partial\hat x_{uij}}{\partial\theta}.$$
For MF scores $\hat x_{ui} = \mathbf w_u^\top\mathbf h_i$:
$\partial\hat x_{uij}/\partial\mathbf w_u = \mathbf h_i-\mathbf h_j$, $\ \partial\hat x_{uij}/\partial\mathbf h_i = \mathbf w_u$, $\ \partial\hat x_{uij}/\partial\mathbf h_j = -\mathbf w_u$.
The factor $\sigma(-\hat x_{uij})$ is large when the pair is ranked *wrongly* and near zero when it is already ranked correctly by a wide margin: BPR focuses on mistakes. **LearnBPR** samples triples uniformly *with replacement* (bootstrap) and takes one SGD step per triple, because iterating user by user would give highly correlated updates.

#### Worked example 8.1 (by hand)

$\hat x_{ui}=0.2$ (known drug), $\hat x_{uj}=0.5$ (unknown drug): $\hat x_{uij}=-0.3$, the pair is mis-ranked. Loss $= -\ln\sigma(-0.3) = \ln(1+e^{0.3}) = 0.854$. Gradient weight $\sigma(0.3) = 0.574$. If the pair were correctly ranked with $\hat x_{uij}=+3$, the weight would be $\sigma(-3)=0.047$: almost no update. Exercise 6 carries out a full parameter update.

### 8.4 BPR for drug repositioning

User = disease (a column of $A$), positive item = a known drug of that disease, negative item = a drug with an unknown link to it. Two caveats:

* BPR only compares drugs **within** the same disease. Scores of different diseases are never compared, so they are not on a common scale. The project's headline metrics pool all test pairs (one global ranking), which BPR does not optimise. In the synthetic experiment of Section 15, BPR's per-disease AUC (0.794) is higher than its pooled AUC (0.776), and pointwise BCE with negative sampling does better on both.
* BPR outputs are not calibrated probabilities (the mean "probability" on unknowns stays near 0.5).

---

## 9. Graph-regularised matrix factorisation: SCMFDD

### 9.1 Similar entities should have similar factors

Plain MF knows nothing about chemistry or phenotypes. Side information enters naturally through a penalty that makes similar drugs have similar factor vectors:
$$\mathcal R(U) = \frac12\sum_{i,i'}S_{ii'}\lVert\mathbf u_i-\mathbf u_{i'}\rVert^2 .$$

**Lemma (Laplacian quadratic form).** For symmetric $S\ge0$ with degree matrix $D$ and Laplacian $L=D-S$,
$$\frac12\sum_{i,i'}S_{ii'}\lVert\mathbf u_i-\mathbf u_{i'}\rVert^2 = \operatorname{tr}(U^\top LU).$$
*Proof.* Expand: $\tfrac12\sum_{ii'}S_{ii'}(\lVert\mathbf u_i\rVert^2+\lVert\mathbf u_{i'}\rVert^2-2\mathbf u_i^\top\mathbf u_{i'}) = \sum_i d_i\lVert\mathbf u_i\rVert^2 - \sum_{ii'}S_{ii'}\mathbf u_i^\top\mathbf u_{i'} = \operatorname{tr}(U^\top DU)-\operatorname{tr}(U^\top SU)$. $\square$

**Worked example 9.1.** Three drugs with $S_{01}=1$, $S_{12}=2$ (others 0) and one-dimensional factors $\mathbf u=(1,3,0)$. Directly: $1\cdot(1-3)^2+2\cdot(3-0)^2 = 4+18=22$. Via the Laplacian: $D=\mathrm{diag}(1,3,2)$, $\mathbf u^\top D\mathbf u = 1+27+0 = 28$, $\mathbf u^\top S\mathbf u = 2(1\cdot1\cdot3 + 2\cdot3\cdot0) = 6$, so $\mathbf u^\top L\mathbf u = 22$. ✓

**What $\operatorname{tr}(U^\top LU)$ penalises:** differences between the factor vectors of *strongly similar* entities, weighted by their similarity. Dissimilar pairs ($S=0$) are unconstrained. The **normalised** version $\operatorname{tr}(U^\top(I-D^{-1/2}SD^{-1/2})U) = \tfrac12\sum S_{ii'}\lVert\mathbf u_i/\sqrt{d_i}-\mathbf u_{i'}/\sqrt{d_{i'}}\rVert^2$ compares degree-scaled factors, so high-degree entities do not dominate the penalty. This is the same smoothness term as label propagation (Chapter 09, Section 4.2) — there it smoothed scores, here it smooths latent factors.

### 9.2 The SCMFDD objective

Zhang et al. (2018) proposed **SCMFDD** (similarity-constrained MF for drug–disease associations):
$$\min_{X,Y}\ \frac12\sum_{ij}(a_{ij}-\mathbf x_i\mathbf y_j^\top)^2 + \frac\mu2\Big(\sum_i\lVert\mathbf x_i\rVert^2+\sum_j\lVert\mathbf y_j\rVert^2\Big) + \frac\lambda2\sum_{ij}\lVert\mathbf x_i-\mathbf x_j\rVert^2w^d_{ij} + \frac\lambda2\sum_{ij}\lVert\mathbf y_i-\mathbf y_j\rVert^2w^s_{ij},$$
with drug factors $\mathbf x_i$ (rows of $X$), disease factors $\mathbf y_j$, a drug similarity $W^d$ computed from drug features (substructures, targets, enzymes, pathways, drug–drug interactions) and a disease semantic similarity $W^s$. By the lemma, this equals
$$\tfrac12\lVert A-XY^\top\rVert_F^2+\tfrac\mu2(\lVert X\rVert_F^2+\lVert Y\rVert_F^2)+\lambda\operatorname{tr}(X^\top L_dX)+\lambda\operatorname{tr}(Y^\top L_sY).$$
Note: the sum is over **all** cells (AMAN). The authors tuned $\lambda,\mu\in\{2^{-3},\dots,2^3\}$ and the dimension $k$ as a percentage of the matrix size, reporting the best AUPR around $k=45\%$, $\mu=1$, $\lambda=4$ on their datasets.

### 9.3 Deriving the updates

**Per-row form (as in the paper).** Differentiating with respect to $\mathbf x_i$ (a row vector), all other rows fixed — note that $\mathbf x_i$ appears in both the $(i,j)$ and $(j,i)$ terms of the similarity sum:
$$\nabla_{\mathbf x_i} = (\mathbf x_iY^\top-\mathbf a_i)Y+\mu\mathbf x_i+\lambda\sum_j(w^d_{ij}+w^d_{ji})(\mathbf x_i-\mathbf x_j).$$
Setting it to zero gives a closed-form row update (the paper derives it as one Newton step, which is exact for a quadratic):
$$\mathbf x_i = \Big(\mathbf a_iY+\lambda\sum_j(w^d_{ij}+w^d_{ji})\mathbf x_j\Big)\Big(Y^\top Y+\mu I+\lambda\sum_j(w^d_{ij}+w^d_{ji})I\Big)^{-1}.$$
**Read it:** the new factor of drug $i$ is a blend of "what its own associations say" ($\mathbf a_iY$) and "the factors of its similar drugs" ($\sum_jw_{ij}\mathbf x_j$). For a **cold drug** ($\mathbf a_i=\mathbf 0$), $\mathbf x_i$ becomes a shrunk, similarity-weighted average of its neighbours' factors — exactly what plain MF cannot do (plain MF gives $\mathbf x_i=\mathbf 0$).

**Whole-matrix form (Sylvester equation).** For $\lVert A-UV^\top\rVert_F^2+\mu\lVert U\rVert_F^2+\lambda\operatorname{tr}(U^\top L_rU)$ (no ½ factors) and fixed $V$, the gradient in $U$ is $-2(A-UV^\top)V+2\mu U+2\lambda L_rU$. Setting it to zero:
$$\lambda L_rU + U(V^\top V+\mu I) = AV,$$
a **Sylvester equation** $aX+Xb=q$ that `scipy.linalg.solve_sylvester` solves exactly. Alternating this with the analogous $V$-equation is graph-regularised ALS (Section 15, example 6).

### 9.4 What it buys: cold entities

In the synthetic experiment of Section 15, 12 drugs lose all their links. Plain MF ($\lambda=0$) scores them all exactly 0 (AUC 0.5); graph regularisation lifts their AUC to 0.851 ($\lambda=1$) and 0.909 ($\lambda=5$), at a tiny cost in fitting the warm drugs. The Laplacian is the channel through which side information reaches entities without data.

---

## 10. Matrix completion and the nuclear norm

### 10.1 The problem

Observe the entries $M_{ij}$ for $(i,j)\in\Omega$ of an unknown $n_1\times n_2$ matrix, and assume $M$ has low rank. Let $\mathcal P_\Omega$ be the projection that keeps the observed entries and zeroes the rest. The natural formulation is
$$\min_X\ \operatorname{rank}(X)\quad\text{s.t.}\quad\mathcal P_\Omega(X)=\mathcal P_\Omega(M).$$
This is NP-hard in general: rank is a non-convex, discontinuous, combinatorial function (it counts non-zero singular values).

### 10.2 The nuclear norm

The **nuclear norm** (trace norm, Schatten-1 norm) is the sum of singular values:
$$\lVert X\rVert_* = \sum_i\sigma_i(X).$$
The analogy with sparse vectors is exact: rank counts non-zero singular values like $\lVert\mathbf x\rVert_0$ counts non-zero entries, and the nuclear norm sums their magnitudes like $\lVert\mathbf x\rVert_1$. Just as the $\ell_1$ norm promotes sparse vectors (lasso), the nuclear norm promotes low-rank matrices. The convex relaxation is
$$\min_X\ \lVert X\rVert_*\quad\text{s.t.}\quad\mathcal P_\Omega(X)=\mathcal P_\Omega(M),$$
a convex (semidefinite-representable) problem.

### 10.3 Why the nuclear norm? The convex envelope theorem

The **convex envelope** of a function $f$ on a convex set $C$ is the largest convex function $g$ with $g\le f$ on $C$ — the tightest convex under-approximation.

**Theorem (Fazel 2002).** On the set $C=\{X:\lVert X\rVert_2\le1\}$ (spectral norm at most 1), the convex envelope of $\operatorname{rank}(X)$ is $\lVert X\rVert_*$.

*Proof sketch.* A standard fact is that the convex envelope of $f$ equals its **biconjugate** $f^{**}$, where $f^*(Y) = \sup_{X\in C}\big(\langle X,Y\rangle-f(X)\big)$ and $\langle X,Y\rangle=\operatorname{tr}(X^\top Y)$.

*Step 1: compute $f^*$.* By von Neumann's trace inequality, $\langle X,Y\rangle\le\sum_i\sigma_i(X)\sigma_i(Y)$, with equality when $X$ and $Y$ share singular vectors. So it suffices to choose singular values $\sigma_i(X)\in[0,1]$. If $\operatorname{rank}X = r$, the best is $\sigma_i(X)=1$ for the $r$ largest $\sigma_i(Y)$:
$$f^*(Y) = \max_{r}\Big(\sum_{i\le r}\sigma_i(Y)-r\Big) = \sum_i\big(\sigma_i(Y)-1\big)_+ .$$
*Step 2: compute $f^{**}$ on $C$.* Again align singular vectors; then for each $i$, with $s=\sigma_i(X)\in[0,1]$ we maximise $g(y) = sy-(y-1)_+$ over $y\ge0$. For $y\le1$, $g(y)=sy\le s$ (attained at $y=1$); for $y>1$, $g(y) = 1-(1-s)y\le s$. So the maximum is $s$, and
$$f^{**}(X) = \sum_i\sigma_i(X) = \lVert X\rVert_*.\qquad\square$$
**Consequences.** (i) On the unit spectral ball, $\lVert X\rVert_*\le\operatorname{rank}(X)$, and no convex function is a tighter lower bound. (ii) For general $X$, scaling gives $\operatorname{rank}(X)\ge\lVert X\rVert_*/\lVert X\rVert_2$. Example: $X=\mathrm{diag}(3,1)$ has $\lVert X\rVert_*=4$, $\lVert X\rVert_2=3$, so $\operatorname{rank}\ge4/3$, i.e. at least 2 — correct.

### 10.4 When does it work? (Candès–Recht)

Candès and Recht (2009) proved that nuclear-norm minimisation recovers $M$ **exactly**, with high probability, if

1. $M$ (size $n\times n$, rank $r$) is **incoherent**: its singular vectors are spread out rather than concentrated on a few rows or columns;
2. the observed entries are sampled **uniformly at random**; and
3. there are enough of them — in their first theorem roughly $m\ge C\,n^{1.2}\,r\log n$ for small rank; later work (Candès–Tao 2010, Recht 2011) sharpened this to about $n\,r\log^2n$. Compare with the $r(2n-r)$ degrees of freedom of a rank-$r$ matrix.

Why incoherence: if $M = \mathbf e_1\mathbf e_1^\top$ (a single non-zero entry), no method can recover it without observing that entry.

**Reality check for drug repositioning.** None of the assumptions hold cleanly. Observations are not uniform (well-studied drugs are over-represented: missing *not* at random); zeros are not "unobserved" but "unknown" (so what is $\Omega$?); and a **cold disease has an entirely unobserved column**, which no low-rank argument can fill: nothing in the data constrains that column, and the minimum-nuclear-norm completion simply sets it to zero (zeroing a column never increases the nuclear norm, because $\lVert XD\rVert_*\le\lVert X\rVert_*\lVert D\rVert_2$ for a diagonal 0/1 matrix $D$). Completing such a column needs **side information**, which is what DRRS's block matrix and inductive matrix completion provide.

### 10.5 Nuclear norm and factorisation are the same thing

**Lemma.** $\lVert X\rVert_* = \min_{U,V:\ UV^\top=X}\tfrac12\big(\lVert U\rVert_F^2+\lVert V\rVert_F^2\big)$ (with inner dimension at least $\operatorname{rank}X$).
*Proof.* "$\le$": for any factorisation, $\lVert UV^\top\rVert_*\le\lVert U\rVert_F\lVert V\rVert_F\le\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$ (a Hölder-type inequality for Schatten norms, then the AM–GM inequality). "$\ge$": take the SVD $X=P\Sigma Q^\top$ and $U=P\Sigma^{1/2}$, $V=Q\Sigma^{1/2}$; then $\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2) = \tfrac12(\operatorname{tr}\Sigma+\operatorname{tr}\Sigma)=\lVert X\rVert_*$. $\square$
So regularised MF (Section 5) and nuclear-norm completion (this section) are two views of the same model: one convex but working with full matrices, the other non-convex but cheap.

---

## 11. Singular value thresholding (SVT)

### 11.1 The shrinkage operator

For $\tau\ge0$ and $Y = U\Sigma V^\top$ (SVD), define the **singular value shrinkage (thresholding) operator**
$$\boxed{\;\mathcal D_\tau(Y) = U\,\mathrm{diag}\big((\sigma_i-\tau)_+\big)\,V^\top\;}$$
It shrinks every singular value by $\tau$ and sets those below $\tau$ to zero, so it both reduces the nuclear norm and (usually) the rank.

**Theorem.** $\mathcal D_\tau(Y) = \arg\min_X\ \tfrac12\lVert X-Y\rVert_F^2+\tau\lVert X\rVert_*$. (It is the **proximal operator** of $\tau\lVert\cdot\rVert_*$.)

*Proof sketch.* The objective is strictly convex, so it suffices to show $\mathbf 0\in\hat X-Y+\tau\,\partial\lVert\hat X\rVert_*$ at $\hat X = \mathcal D_\tau(Y)$. The subdifferential of the nuclear norm at $X=U_0\Sigma_0V_0^\top$ (compact SVD) is $\{U_0V_0^\top+W:\ U_0^\top W=0,\ WV_0=0,\ \lVert W\rVert_2\le1\}$. Split $Y$'s SVD into the part with $\sigma_i>\tau$ ($U_0,\Sigma_0,V_0$) and the rest ($U_1,\Sigma_1,V_1$). Then $\hat X = U_0(\Sigma_0-\tau I)V_0^\top$ and
$$Y-\hat X = \tau U_0V_0^\top + U_1\Sigma_1V_1^\top = \tau\big(U_0V_0^\top+W\big),\qquad W = \tau^{-1}U_1\Sigma_1V_1^\top .$$
$W$ is orthogonal to $U_0$ and $V_0$, and $\lVert W\rVert_2\le1$ because every singular value in $\Sigma_1$ is at most $\tau$. So $Y-\hat X\in\tau\partial\lVert\hat X\rVert_*$. $\square$
This is the matrix analogue of **soft-thresholding** in the lasso, $\operatorname{sign}(y)(|y|-\tau)_+$, applied to singular values.

#### Worked example 11.1 (by hand)

* $Y$ with singular values $(3,1)$, $\tau=2$: new singular values $(1,0)$. The rank drops from 2 to 1; the singular vectors are unchanged.
* $Y=\begin{pmatrix}3&4\\0&0\end{pmatrix}$: rank 1 with $\sigma_1=\lVert(3,4)\rVert=5$. With $\tau=1$: $\sigma_1\to4$, so $\mathcal D_1(Y) = \tfrac45Y = \begin{pmatrix}2.4&3.2\\0&0\end{pmatrix}$.
* $\mathrm{diag}(5,2,0.5)$ with $\tau=1$: $\mathrm{diag}(4,1,0)$.

### 11.2 The Cai–Candès–Shen algorithm

Cai, Candès and Shen (2010) solve the slightly regularised problem
$$\min_X\ \tau\lVert X\rVert_*+\tfrac12\lVert X\rVert_F^2\quad\text{s.t.}\quad\mathcal P_\Omega(X) = \mathcal P_\Omega(M)$$
(whose solution tends to the nuclear-norm minimiser as $\tau\to\infty$) by the iteration
$$\boxed{\;X^k = \mathcal D_\tau(Y^{k-1}),\qquad Y^k = Y^{k-1}+\delta\,\mathcal P_\Omega(M-X^k),\qquad Y^0=0\;}$$

**Interpretation.** $Y$ is a Lagrange multiplier for the constraint, supported on $\Omega$. The first step minimises the Lagrangian over $X$ (which, by the theorem above, is a shrinkage); the second is a gradient-ascent step on the dual (Uzawa's algorithm). In words: *shrink to low rank; look at how far the result is from the observed entries; push $Y$ in that direction; repeat.*

**Parameters and guarantees.** The iteration converges for step sizes $0<\delta<2$. CCS recommend $\tau\approx5n$ (for $n\times n$) and $\delta = 1.2\,n_1n_2/|\Omega|$ (i.e. $1.2/p$ for sampling fraction $p$) in practice, noting that this can exceed 2 when $p<0.6$ but usually still works. They also warm-start $Y^0$ at a multiple of $\delta\mathcal P_\Omega(M)$ to skip the first iterations where $X^k=0$.

**Why it is efficient.** $Y^k$ is sparse (non-zero only on $\Omega$) and $X^k$ is low-rank, so only the singular values above $\tau$ are needed: a partial or randomised SVD suffices. Section 15 recovers a 100×80 rank-3 matrix from 40% of its entries to relative error $1.6\times10^{-4}$ in 135 iterations.

A close relative is **Soft-Impute** (Mazumder, Hastie & Tibshirani 2010), which solves $\min_X\tfrac12\lVert\mathcal P_\Omega(X-M)\rVert_F^2+\lambda\lVert X\rVert_*$ by repeating $X\leftarrow\mathcal D_\lambda\big(\mathcal P_\Omega(M)+\mathcal P_\Omega^\perp(X)\big)$: fill the missing entries with the current guess, shrink, repeat.

---

## 12. DRRS and BNNR: completing the heterogeneous matrix

### 12.1 The block matrix

DRRS (Luo et al., *Bioinformatics* 2018) stacks everything known into one symmetric $(n+m)\times(n+m)$ matrix,
$$T=\begin{pmatrix}S_r&A\\A^\top&S_d\end{pmatrix},$$
and completes it under a low-rank assumption with SVT. The predictions are the completed top-right block.

Why this helps:
* **More observed entries.** The similarity blocks are dense, so $T$ is mostly observed even though $A$ is 99% zeros.
* **A shared latent space.** If $T\approx ZZ^\top$-like with $Z = \binom{Z_r}{Z_d}$, then simultaneously $S_r\approx Z_rZ_r^\top$, $S_d\approx Z_dZ_d^\top$ and $A\approx Z_rZ_d^\top$: drugs and diseases are embedded in *one* space in which inner products explain both similarities and associations. This is collective matrix factorisation in completion form.
* **Cold columns are no longer empty.** A disease with no known drug still has a full row in the $S_d$ block, so its latent coordinates are pinned down by its similarities, and the low-rank structure transfers them into the $A$ block. Section 15 shows the cold disease of the toy problem receiving the ranking (1, 3, 0, 2), the same as the random-walk methods of Chapter 09, while SVT on $A$ alone leaves that column at exactly 0.

**What counts as observed.** DRRS takes $\Omega$ = the *non-zero* entries of $T$: known links and non-zero similarities. The zeros of $A$ are treated as **unobserved** (the AMAU stance), and the low-rank constraint is what stops the trivial all-ones answer. This contrasts with SCMFDD, which fits those zeros as real zeros.

### 12.2 The project's DRRS

`DRRS.fit_predict` implements CCS-style SVT with a randomised partial SVD (`torch.svd_lowrank`, 200 components), $\delta = 1.2N^2/|\Omega|$, a threshold proportional to the spectral norm of $T$, and 200 iterations (details in Section 16). One number worth knowing: on Fdataset $N=906$ and 52.6% of $T$ is non-zero, so $\delta\approx2.28$ — slightly above the theoretical limit of 2. It works here (the large threshold keeps the iteration stable), but if you change the data or parameters and see the iteration blow up, reduce $\delta$ first (Exercise 10 shows the failure on a toy).

### 12.3 BNNR: bounded nuclear norm regularisation

Exact agreement with every observed entry is a strong demand when the similarities are noisy, and SVT's completed values can fall outside $[0,1]$. **BNNR** (Yang et al., *Bioinformatics* 2019) relaxes both:
$$\min_{X,W}\ \lVert X\rVert_* + \frac\alpha2\big\lVert\mathcal P_\Omega(W)-\mathcal P_\Omega(T)\big\rVert_F^2\quad\text{s.t.}\quad X=W,\quad 0\le W\le1 .$$
The quadratic term *tolerates* noise in the observed entries instead of enforcing them exactly; the box constraint keeps every prediction a valid score in $[0,1]$. It is solved with ADMM (alternating direction method of multipliers): an SVT step for $X$, a closed-form update followed by clipping to $[0,1]$ for $W$, and a multiplier update. Its authors reported improvements over DRRS on the standard benchmarks.

---

## 13. Inductive matrix completion and NIMCGCN

### 13.1 Transductive versus inductive

MF and SVT are **transductive**: every entity has its own free parameters (a row of $U$, or a row/column of $X$). A drug that was not in the training matrix has no parameters, so it cannot be scored. **Inductive matrix completion** (IMC; Jain & Dhillon 2013; Natarajan & Dhillon 2014 for gene–disease prediction) makes the factors *functions of features*. With drug features $X\in\mathbb R^{n\times f_r}$ and disease features $Y\in\mathbb R^{m\times f_d}$:
$$A\approx XWY^\top = (XG)(YH)^\top,\qquad W=GH^\top,\ G\in\mathbb R^{f_r\times k},\ H\in\mathbb R^{f_d\times k}.$$
The number of parameters no longer grows with the number of entities, and a brand-new disease with a feature vector $\mathbf y$ gets embedding $H^\top\mathbf y$ immediately. A useful identity for fitting $W$ by least squares: $\operatorname{vec}(XWY^\top) = (Y\otimes X)\operatorname{vec}(W)$, which turns the problem into ordinary ridge regression in $\operatorname{vec}(W)$. In Section 15, IMC trained on 40 diseases scores 10 never-seen diseases with AUC 0.947.

IMC fails when the features do not explain the associations (then $W$ has nothing to learn), and its linear form limits what it can express — hence neural versions.

### 13.2 NIMCGCN

**NIMCGCN** (Li et al., *Bioinformatics* 2020; originally for miRNA–disease association) replaces the linear feature maps of IMC by graph convolutional encoders: one GCN on the drug similarity graph and one on the disease similarity graph, each followed by a learned linear projection, and an inner product decoder:
$$\hat A = \sigma\big(f_r(\text{GCN}_r(\hat S_r, X_r))\;f_d(\text{GCN}_d(\hat S_d, X_d))^\top\big).$$
In the project's re-implementation the input features are the similarity rows themselves, the graphs are 10-NN and symmetrically normalised, and training uses class-weighted BCE over all allowed cells (Section 16).

It is the weakest baseline on Fdataset (AUC 0.838, AUPR 0.096). Likely reasons, in decreasing order of importance: (i) **its encoders never see the known associations** — embeddings are computed from similarities only, so all collaborative signal must be squeezed into a single inner product, whereas MV-HGAT feeds visible association rows as features (`feat_assoc`) and as graph edges; (ii) the inputs are high-dimensional similarity rows and there are only ~1,500 training positives, so the GCN overfits easily; (iii) the positive weight of about 95 in `weighted_bce` makes gradients noisy; (iv) the architecture was designed for miRNA data with different similarity structure. This is a good illustration that "inductive and neural" does not automatically beat "transductive and simple".

---

## 14. Bilinear decoders are learned factorisations

MV-HGAT scores a pair with
$$\text{logit}_{ij} = \text{gate}(i,j)\;\mathbf h_i^\top W\mathbf h_j\;+\;\sum_vw_vP_v[i,j]\;+\;b ,$$
where $\mathbf h_i,\mathbf h_j\in\mathbb R^{d}$ ($d = 64\times3 = 192$ after the jumping-knowledge concatenation) come from the heterogeneous GAT encoder and $W\in\mathbb R^{d\times d}$ is learned.

**The bilinear term is matrix factorisation.** Any $W$ can be written $W = LR^\top$ (for instance from its SVD, $L = P\Sigma^{1/2}$, $R=Q\Sigma^{1/2}$). Then
$$\mathbf h_i^\top W\mathbf h_j = (L^\top\mathbf h_i)^\top(R^\top\mathbf h_j) = \mathbf u_i^\top\mathbf v_j,\qquad \mathbf u_i = L^\top\mathbf h_i,\ \mathbf v_j = R^\top\mathbf h_j,$$
so the logit matrix is $H_rWH_d^\top = UV^\top$ with rank at most $d$. The difference from classic MF is *where the factors come from*:

| | Classic MF | IMC | MV-HGAT |
|---|---|---|---|
| Drug factor $\mathbf u_i$ | free parameter vector | $G^\top\mathbf x_i$ (linear in features) | $L^\top\mathbf h_i$, $\mathbf h_i$ = GNN(features, similarity graphs, visible links) |
| New entity? | impossible | yes (features) | partly: needs to be a node in the graph (transductive graph, inductive features) |
| Side information | only via regularisers (SCMFDD) | via features | via features *and* message passing |
| Loss | squared error / BPR | squared error | BCE with negative sampling (logistic MF) |

Common decoder variants: plain **dot product** ($W=I$), **DistMult** (diagonal $W$, a weighted dot product, symmetric in $i,j$), **RESCAL** (full $W$, as here, which can model asymmetric drug–disease interactions between different embedding dimensions — "chemical dimension 3 of a drug matches mechanism dimension 7 of a disease"), and MLP decoders.

**A hybrid recommender.** The full MV-HGAT logit = a *learned factorisation* (latent-factor CF with GNN-computed factors) + a *neighbourhood* term (the propagation head, i.e. user-/item-based CF with side-information similarity) + a bias. Combining latent factors and neighbourhoods in one model is exactly what Koren (2008) did in "Factorization meets the neighborhood", one of the key Netflix Prize ideas. The degree gate decides how much to trust the factor term for each pair: little for cold entities, a lot for well-connected ones.

### 14.1 How the methods compare on Fdataset

| Method | Family | Uses unknowns as | 5-fold AUC | 5-fold AUPR |
|---|---|---|---|---|
| MV-HGAT (ours) | GNN factors + neighbourhood CF | sampled negatives (ratio 2) | **0.939** | 0.488 |
| SCMFDD (tuned $k=128$, $\lambda=2$) | graph-regularised MF | zeros (AMAN), squared loss | 0.893 | **0.495** |
| DRRS | nuclear-norm completion of $T$ | unobserved (AMAU + low rank) | 0.879 | 0.389 |
| MBiRW | bi-random walk (Chapter 09) | restart from known links | 0.883 | 0.311 |
| NIMCGCN | neural IMC | weighted negatives | 0.838 | 0.096 |

How to read it: SCMFDD's squared loss on a full, block-structured matrix reconstructs the dense "family blocks" sharply, which puts many true positives at the very top of the ranking (high AUPR). But AMAN pushes every unknown towards 0 with equal force, so the order *among* the many low-scoring pairs is poor, and the rare positives that fall outside the main blocks are ranked badly (lower AUC). MV-HGAT orders the whole list better (AUC 0.939) and, through its propagation head and degree gate, handles cold start, which SCMFDD cannot do without links. A fair summary for the paper: *on warm-start Fdataset, MV-HGAT and a tuned SCMFDD are tied on AUPR, MV-HGAT is clearly better on AUC*.

---

## 15. Code: everything above, runnable

All examples are CPU-only and finish in seconds. Run them with the project environment (`.venv\Scripts\python.exe example.py`). Every output below was produced by running the code exactly as printed; the last digit can differ on other machines or library versions.

### 15.1 Neighbourhood collaborative filtering (item- and user-based)

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# 6 drugs (rows) x 5 diseases (columns); 1 = known indication.
#             asthma COPD  RA  lupus psoriasis
A = np.array([[1,    1,    0,   0,    0],    # drug 0: an inhaled steroid
              [1,    1,    0,   0,    0],    # drug 1: a bronchodilator
              [0,    1,    0,   0,    0],    # drug 2
              [0,    0,    1,   1,    0],    # drug 3: an immunosuppressant
              [0,    0,    1,   1,    1],    # drug 4
              [0,    0,    1,   0,    1]],   # drug 5
             dtype=float)

def cosine_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    Xn = X / n
    return Xn @ Xn.T

# ITEM-based CF (items = drugs): drugs are similar if they treat the same diseases
Sim_drug = cosine_rows(A); np.fill_diagonal(Sim_drug, 0)
score_item = Sim_drug @ A / np.maximum(Sim_drug.sum(1, keepdims=True), 1e-12)
# USER-based CF (users = diseases): diseases are similar if the same drugs treat them
Sim_dis = cosine_rows(A.T); np.fill_diagonal(Sim_dis, 0)
score_user = A @ Sim_dis.T / np.maximum(Sim_dis.sum(1)[None, :], 1e-12)

print("drug-drug cosine (from co-indications):\n", Sim_drug)
print("item-based scores:\n", score_item)
print("user-based scores:\n", score_user)
unknown = A == 0
best = np.unravel_index(np.argmax(np.where(unknown, score_item, -1)), A.shape)
print("top unknown pair by item-based CF: drug", best[0], "disease", best[1])
```

Output:

```text
drug-drug cosine (from co-indications):
 [[0.    1.    0.707 0.    0.    0.   ]
 [1.    0.    0.707 0.    0.    0.   ]
 [0.707 0.707 0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.816 0.5  ]
 [0.    0.    0.    0.816 0.    0.816]
 [0.    0.    0.    0.5   0.816 0.   ]]
item-based scores:
 [[0.586 1.    0.    0.    0.   ]
 [0.586 1.    0.    0.    0.   ]
 [1.    1.    0.    0.    0.   ]
 [0.    0.    1.    0.62  1.   ]
 [0.    0.    1.    0.5   0.5  ]
 [0.    0.    1.    1.    0.62 ]]
user-based scores:
 [[1.   1.   0.   0.   0.  ]
 [1.   1.   0.   0.   0.  ]
 [1.   0.   0.   0.   0.  ]
 [0.   0.   0.5  0.62 1.  ]
 [0.   0.   1.   1.   1.  ]
 [0.   0.   0.5  1.   0.62]]
top unknown pair by item-based CF: drug 2 disease 0
```

The drug–drug cosine matrix shows the two families (drugs 0–2, drugs 3–5). The item-based score of drug 2 for asthma is 1.0, as computed by hand in worked example 2.1, and it is the top unknown pair. Item-based and user-based CF disagree in places (e.g. drug 4 / psoriasis, a *known* pair, gets 0.5 from items but 1.0 from users) — they look at different neighbourhoods.

### 15.2 SVD, Eckart–Young and the low-rank picture

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)
A = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0],
              [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 1]], float)   # Example 1

U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", s)
for k in (1, 2, 3):
    Ak = U[:, :k] * s[:k] @ Vt[:k]
    err = np.linalg.norm(A - Ak)
    print(f"rank-{k} approx: ||A - A_k||_F = {err:.3f};  sqrt(sum of dropped s^2) = "
          f"{np.sqrt((s[k:] ** 2).sum()):.3f}")
A2 = U[:, :2] * s[:2] @ Vt[:2]
print("rank-2 reconstruction:\n", A2)
print("drug 2 / asthma (unknown) gets", A2[2, 0].round(3), "; drug 0 / RA (unknown) gets", A2[0, 2].round(3))
```

Output:

```text
singular values: [2.414 2.136 1.    0.662 0.414]
rank-1 approx: ||A - A_k||_F = 2.484;  sqrt(sum of dropped s^2) = 2.484
rank-2 approx: ||A - A_k||_F = 1.269;  sqrt(sum of dropped s^2) = 1.269
rank-3 approx: ||A - A_k||_F = 0.781;  sqrt(sum of dropped s^2) = 0.781
rank-2 reconstruction:
 [[ 0.864  1.106  0.     0.     0.   ]
 [ 0.864  1.106 -0.    -0.    -0.   ]
 [ 0.485  0.621  0.     0.     0.   ]
 [ 0.     0.     0.854  0.604  0.604]
 [ 0.     0.     1.207  0.854  0.854]
 [ 0.     0.     0.854  0.604  0.604]]
drug 2 / asthma (unknown) gets 0.485 ; drug 0 / RA (unknown) gets 0.0
```

The error of the best rank-$k$ approximation equals $\sqrt{\sum_{i>k}\sigma_i^2}$ exactly. The rank-2 reconstruction keeps one factor per disease family: it suggests drug 2 for asthma (0.485) and never mixes the families (drug 0 / RA = 0).

### 15.3 Matrix factorisation by ALS and by SGD

A synthetic 20×15 matrix of exact rank 2 with 60% of entries observed; both algorithms must predict the 40% they never see.

```python
import numpy as np
rng = np.random.default_rng(0)

# Ground truth: a 20 x 15 matrix of exact rank 2, 60% of the entries observed
n, m, k = 20, 15, 2
U_true, V_true = rng.normal(size=(n, k)), rng.normal(size=(m, k))
M = U_true @ V_true.T
mask = rng.random((n, m)) < 0.6
print("observed entries:", mask.sum(), "of", n * m)

def rmse(X, sel):
    return np.sqrt(((X - M)[sel] ** 2).mean())

def als(M, mask, k=2, lam=0.01, iters=30):
    U = rng.normal(scale=0.1, size=(n, k)); V = rng.normal(scale=0.1, size=(m, k))
    I = np.eye(k)
    for it in range(iters):
        for i in range(n):                         # each row of U: a small ridge regression
            Vo = V[mask[i]]                        # factors of the diseases observed for drug i
            U[i] = np.linalg.solve(Vo.T @ Vo + lam * I, Vo.T @ M[i, mask[i]])
        for j in range(m):                         # each row of V: same thing, roles swapped
            Uo = U[mask[:, j]]
            V[j] = np.linalg.solve(Uo.T @ Uo + lam * I, Uo.T @ M[mask[:, j], j])
        if it in (0, 1, 4, 29):
            print(f"ALS iter {it + 1:2d}: train RMSE {rmse(U @ V.T, mask):.4f}   "
                  f"held-out RMSE {rmse(U @ V.T, ~mask):.4f}")
    return U, V

def sgd(M, mask, k=2, lam=0.01, lr=0.02, epochs=400):
    U = rng.normal(scale=0.1, size=(n, k)); V = rng.normal(scale=0.1, size=(m, k))
    obs = np.argwhere(mask)
    for ep in range(epochs):
        rng.shuffle(obs)
        for i, j in obs:
            e = M[i, j] - U[i] @ V[j]              # prediction error on one entry
            U[i], V[j] = U[i] + lr * (e * V[j] - lam * U[i]), V[j] + lr * (e * U[i] - lam * V[j])
        if ep + 1 in (1, 10, 100, 400):
            print(f"SGD epoch {ep + 1:3d}: train RMSE {rmse(U @ V.T, mask):.4f}   "
                  f"held-out RMSE {rmse(U @ V.T, ~mask):.4f}")
    return U, V

als(M, mask)
sgd(M, mask)
```

Output:

```text
observed entries: 168 of 300
ALS iter  1: train RMSE 0.9853   held-out RMSE 1.8775
ALS iter  2: train RMSE 0.4075   held-out RMSE 1.0264
ALS iter  5: train RMSE 0.0309   held-out RMSE 0.0906
ALS iter 30: train RMSE 0.0038   held-out RMSE 0.0078
SGD epoch   1: train RMSE 1.3236   held-out RMSE 1.3243
SGD epoch  10: train RMSE 1.1345   held-out RMSE 1.2265
SGD epoch 100: train RMSE 0.0218   held-out RMSE 0.0540
SGD epoch 400: train RMSE 0.0169   held-out RMSE 0.0345
```

ALS reaches a held-out RMSE below 0.01 in 30 sweeps; each sweep solves 35 tiny ridge regressions exactly. SGD needs hundreds of epochs over the 168 observed cells and stops at a slightly higher error (its learning rate and regularisation were not tuned). Both recover the hidden entries because the matrix really is low-rank and the observed cells are spread uniformly — the Candès–Recht conditions in miniature.

### 15.4 Weighted ALS for implicit feedback (Hu–Koren–Volinsky)

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(1)
A = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0],
              [0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 0, 1]], float)   # Example 1
n, m = A.shape

def implicit_als(A, k=2, alpha=10.0, lam=0.1, iters=20):
    """Hu, Koren & Volinsky (2008): preference p = A, confidence C = 1 + alpha*A.
    Minimise sum_ij C_ij (p_ij - u_i.v_j)^2 + lam (||U||^2 + ||V||^2)."""
    P, C = A, 1 + alpha * A
    U = rng.normal(scale=0.1, size=(n, k)); V = rng.normal(scale=0.1, size=(m, k))
    I = np.eye(k)
    for _ in range(iters):
        VtV = V.T @ V                              # shared by every row: the speed trick
        for i in range(n):
            Ci = C[i]                              # confidences of row i
            # V^T C_i V = V^T V + V^T (C_i - I) V ; only the observed cells change it
            A_i = VtV + (V.T * (Ci - 1)) @ V + lam * I
            U[i] = np.linalg.solve(A_i, (V.T * Ci) @ P[i])
        UtU = U.T @ U
        for j in range(m):
            Cj = C[:, j]
            A_j = UtU + (U.T * (Cj - 1)) @ U + lam * I
            V[j] = np.linalg.solve(A_j, (U.T * Cj) @ P[:, j])
    return U @ V.T

S = implicit_als(A)
print("implicit-feedback ALS scores:\n", S)
cand = [(i, j) for i in range(n) for j in range(m) if A[i, j] == 0]
top = sorted(cand, key=lambda ij: -S[ij])[:3]
print("top-3 unknown pairs:", [(int(i), int(j), round(float(S[i, j]), 3)) for i, j in top])
```

Output:

```text
implicit-feedback ALS scores:
 [[ 0.965  1.037  0.     0.    -0.   ]
 [ 0.965  1.037  0.     0.    -0.   ]
 [ 0.851  0.915  0.     0.    -0.   ]
 [ 0.     0.     0.979  0.929  0.929]
 [ 0.     0.     1.038  0.985  0.985]
 [-0.    -0.     0.979  0.929  0.929]]
top-3 unknown pairs: [(3, 4, 0.929), (5, 3, 0.929), (2, 0, 0.851)]
```

Known links (confidence 11) are fitted close to 1; unknowns (confidence 1) are allowed to stay above 0 where the low-rank structure says so. The top three unknown pairs complete the two family blocks: drug 3 / psoriasis, drug 5 / lupus and drug 2 / asthma.

### 15.5 Negative sampling ratios and BPR (PyTorch, CPU)

A synthetic 200×100 drug–disease matrix with a hidden rank-4 structure: 3% of pairs are true indications, but only 70% of those are "recorded" — the rest are true-but-unknown, exactly the PU situation of Section 3. Models are trained on recorded links with different negative-sampling ratios, and with BPR.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(2)

# ---- synthetic "drug x disease" data with a hidden low-rank structure ----------
rng = np.random.default_rng(0)
n, m, r = 200, 100, 4
logit_true = rng.normal(size=(n, r)) @ rng.normal(size=(r, m))
truth = logit_true > np.quantile(logit_true, 0.97)          # 3% true indications
known = truth & (rng.random((n, m)) < 0.7)                  # only 70% have been discovered
# test split: 20% of the known links + 20% of the unknown cells
pos, unk = np.flatnonzero(known), np.flatnonzero(~known)
test_pos = rng.choice(pos, len(pos) // 5, replace=False)
test_unk = rng.choice(unk, len(unk) // 5, replace=False)
A_train = known.ravel().copy(); A_train[test_pos] = False
neg_pool = np.setdiff1d(np.flatnonzero(~A_train), test_unk)  # never sample test cells
test_idx = np.concatenate([test_pos, test_unk])
y_test = np.r_[np.ones(len(test_pos)), np.zeros(len(test_unk))]
train_pos = np.flatnonzero(A_train)

def evaluate(S, name):
    s = S.ravel()[test_idx]
    # per-disease AUC: rank drugs WITHIN each disease column, then average over diseases
    col = test_idx % m
    per = [roc_auc_score(y_test[col == j], s[col == j]) for j in range(m)
           if 0 < y_test[col == j].sum() < (col == j).sum()]
    # among the 100 best-scored test cells labelled "unknown", how many are real indications?
    top_unk = test_unk[np.argsort(-S.ravel()[test_unk])[:100]]
    print(f"{name:17s} AUC {roc_auc_score(y_test, s):.3f} | per-disease AUC {np.mean(per):.3f} | "
          f"AUPR {average_precision_score(y_test, s):.3f} | mean p(unknown) "
          f"{S.ravel()[test_unk].mean():.3f} | true in top-100 'unknowns' {truth.ravel()[top_unk].mean():.2f}")

def train_mf(loss_kind, neg_ratio=2, k=8, epochs=300, seed=0):
    torch.manual_seed(seed)
    U = (0.1 * torch.randn(n, k)).requires_grad_()
    V = (0.1 * torch.randn(m, k)).requires_grad_()
    opt = torch.optim.Adam([U, V], lr=0.05, weight_decay=1e-3 if loss_kind == "bce" else 3e-3)
    P = torch.as_tensor(train_pos); pool = torch.as_tensor(neg_pool)
    usable = torch.zeros(n * m, dtype=torch.bool); usable[pool] = True
    for _ in range(epochs):
        X = (U @ V.T).reshape(-1)
        if loss_kind == "bce":                     # pointwise: positives vs sampled negatives
            neg = pool[torch.randint(len(pool), (len(P) * neg_ratio,))]
            logits = torch.cat([X[P], X[neg]])
            y = torch.cat([torch.ones(len(P)), torch.zeros(len(neg))])
            loss = F.binary_cross_entropy_with_logits(logits, y)
        else:                                      # BPR: known drug vs unknown drug, SAME disease
            Pr = P.repeat(5)                       # 5 sampled comparisons per positive
            j = Pr % m                             # disease (column) of each positive
            cell = torch.randint(n, (len(Pr),)) * m + j   # a random drug for that disease
            ok = usable[cell]                      # keep it only if it is a usable unknown
            loss = -F.logsigmoid(X[Pr[ok]] - X[cell[ok]]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(U @ V.T).detach().numpy()

print(f"rate of recorded links: {known.mean():.4f}; true rate: {truth.mean():.4f}")
for ratio in (1, 2, 5, 20):
    evaluate(train_mf("bce", ratio), f"BCE, neg ratio {ratio}")
evaluate(train_mf("bpr"), "BPR")
```

Output:

```text
rate of recorded links: 0.0209; true rate: 0.0300
BCE, neg ratio 1  AUC 0.891 | per-disease AUC 0.852 | AUPR 0.197 | mean p(unknown) 0.391 | true in top-100 'unknowns' 0.17
BCE, neg ratio 2  AUC 0.898 | per-disease AUC 0.860 | AUPR 0.198 | mean p(unknown) 0.310 | true in top-100 'unknowns' 0.16
BCE, neg ratio 5  AUC 0.900 | per-disease AUC 0.846 | AUPR 0.160 | mean p(unknown) 0.246 | true in top-100 'unknowns' 0.12
BCE, neg ratio 20 AUC 0.841 | per-disease AUC 0.729 | AUPR 0.091 | mean p(unknown) 0.184 | true in top-100 'unknowns' 0.08
BPR               AUC 0.776 | per-disease AUC 0.794 | AUPR 0.179 | mean p(unknown) 0.496 | true in top-100 'unknowns' 0.17
```

Reading the columns:

* **mean p(unknown)** falls steadily as the ratio grows (0.39 → 0.18): the calibration shift of Section 7.3.
* **AUPR** is best at ratio 1–2 and drops at 20: too many sampled negatives per epoch, many of them true-but-unknown.
* **true in top-100 'unknowns'**: among the 100 test pairs labelled unknown that the model ranks highest, 8–17% are in fact true indications, against a base rate of about 0.8%. These "false positives" are exactly the discoveries a repositioning model is meant to make — unknown ≠ negative.
* **BPR** has per-disease AUC (0.794) above its pooled AUC (0.776), as expected from a loss that only compares drugs within a disease; its outputs hover around 0.5 (uncalibrated). Here, pointwise BCE with a small ratio wins on every metric — one reason the project uses it.

### 15.6 Graph-regularised MF (SCMFDD-style) solved with Sylvester equations

```python
import numpy as np
from scipy.linalg import solve_sylvester
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)

# Synthetic data: 120 drugs x 60 diseases, hidden rank-4 structure, 5% links
n, m, r = 120, 60, 4
Ut, Vt = rng.normal(size=(n, r)), rng.normal(size=(m, r))
A = (Ut @ Vt.T > np.quantile(Ut @ Vt.T, 0.95)).astype(float)

def knn_sim(F, k=8):
    """Cosine similarity of noisy side features, kept for the k nearest neighbours."""
    F = F + 0.5 * rng.normal(size=F.shape)
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    S = np.clip(Fn @ Fn.T, 0, None); np.fill_diagonal(S, 0)
    keep = np.zeros_like(S, bool)
    keep[np.repeat(np.arange(len(S)), k), np.argsort(-S, 1)[:, :k].ravel()] = True
    return S * (keep | keep.T)

def norm_laplacian(S):
    d = S.sum(1); d[d == 0] = 1
    return np.eye(len(S)) - S / np.sqrt(np.outer(d, d))

Sr, Sd = knn_sim(Ut), knn_sim(Vt)          # "chemical" and "phenotype" similarity stand-ins
Lr, Ld = norm_laplacian(Sr), norm_laplacian(Sd)

cold = rng.choice(n, 12, replace=False)     # 12 drugs lose ALL their links (cold start)
A_tr = A.copy(); A_tr[cold] = 0

def graph_mf(A, Lr, Ld, k=16, mu=0.1, lam=1.0, iters=30):
    """min ||A - U V^T||_F^2 + mu(||U||^2 + ||V||^2) + lam (tr U^T Lr U + tr V^T Ld V).
    Each half-step is a Sylvester equation:  lam*Lr U + U (V^T V + mu I) = A V."""
    U = 0.1 * rng.normal(size=(A.shape[0], k)); V = 0.1 * rng.normal(size=(A.shape[1], k))
    I = np.eye(k)
    for _ in range(iters):
        U = solve_sylvester(lam * Lr, V.T @ V + mu * I, A @ V)
        V = solve_sylvester(lam * Ld, U.T @ U + mu * I, A.T @ U)
    return U @ V.T

for lam in (0.0, 1.0, 5.0):
    R = graph_mf(A_tr, Lr, Ld, lam=lam)
    cold_auc = roc_auc_score(A[cold].ravel(), R[cold].ravel())
    warm = np.setdiff1d(np.arange(n), cold)
    print(f"lambda={lam:3}: cold-drug AUC {cold_auc:.3f}   max |score| on cold rows "
          f"{np.abs(R[cold]).max():.3f}   warm-drug fit AUC {roc_auc_score(A_tr[warm].ravel(), R[warm].ravel()):.3f}")
```

Output:

```text
lambda=0.0: cold-drug AUC 0.500   max |score| on cold rows 0.000   warm-drug fit AUC 1.000
lambda=1.0: cold-drug AUC 0.851   max |score| on cold rows 0.131   warm-drug fit AUC 1.000
lambda=5.0: cold-drug AUC 0.909   max |score| on cold rows 0.249   warm-drug fit AUC 0.988
```

With $\lambda=0$ (plain MF on the full matrix) the 12 cold drugs get factors of exactly zero and therefore scores of exactly zero. With the Laplacian term, their factors are pulled towards their neighbours' factors, and their rankings become good (AUC 0.85–0.91) at almost no cost to the warm drugs.

### 15.7 Singular value thresholding (Cai–Candès–Shen)

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(0)

def shrink(Y, tau):
    """Singular value thresholding operator D_tau(Y) = U diag(max(s - tau, 0)) V^T."""
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s = np.maximum(s - tau, 0)
    return (U * s) @ Vt, int((s > 0).sum())

# Hand example: Y = [[3, 0], [0, 1]] rotated by a random orthogonal Q
Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
Y = Q @ np.diag([3.0, 1.0]) @ Q.T
X, rank = shrink(Y, 2.0)
print("singular values of Y:", np.linalg.svd(Y, compute_uv=False),
      "-> after D_2:", np.linalg.svd(X, compute_uv=False), " rank", rank)

# Matrix completion: 100 x 80 matrix of rank 3, 40% of entries observed
n1, n2, r = 100, 80, 3
M = rng.normal(size=(n1, r)) @ rng.normal(size=(r, n2))
Omega = rng.random((n1, n2)) < 0.4
tau = 5 * np.sqrt(n1 * n2)                       # Cai-Candes-Shen's suggested threshold
delta = 1.2 * n1 * n2 / Omega.sum()              # and step size
Y = np.zeros_like(M)
for k in range(1, 501):
    X, rank = shrink(Y, tau)
    resid = Omega * (M - X)
    Y += delta * resid
    rel_obs = np.linalg.norm(resid) / np.linalg.norm(Omega * M)
    if k in (1, 10, 50) or rel_obs < 1e-4:
        rel_all = np.linalg.norm(X - M) / np.linalg.norm(M)
        print(f"iter {k:3d}: rank {rank:2d}  rel. error on observed {rel_obs:.2e}  on ALL entries {rel_all:.2e}")
    if rel_obs < 1e-4:
        break
print("nuclear norm of the answer vs truth:", round(np.linalg.svd(X, compute_uv=False).sum(), 1),
      round(np.linalg.svd(M, compute_uv=False).sum(), 1))
```

Output:

```text
singular values of Y: [3. 1.] -> after D_2: [1. 0.]  rank 1
iter   1: rank  0  rel. error on observed 1.00e+00  on ALL entries 1.00e+00
iter  10: rank  3  rel. error on observed 1.51e-01  on ALL entries 2.02e-01
iter  50: rank  3  rel. error on observed 6.87e-03  on ALL entries 1.11e-02
iter 135: rank  3  rel. error on observed 9.81e-05  on ALL entries 1.60e-04
nuclear norm of the answer vs truth: 272.3 272.3
```

The first line is worked example 11.1. In the completion run, the very first iterate is 0 (because $Y^0=0$; CCS's warm start avoids this), the rank locks onto the true value 3 within 10 iterations, and the error on *all* entries (including the 60% never observed) tracks the error on the observed ones down to about $10^{-4}$. The nuclear norm of the answer equals that of the truth.

### 15.8 DRRS-style completion of the heterogeneous block matrix

The 4-drug, 3-disease toy of Chapter 09 (disease 2 has no known drug).

```python
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# The 4-drug x 3-disease toy (disease 2 has no known drug)
A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], float)
Sr = np.array([[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7], [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.9], [0.3, 0.9, 1.0]])

def svt_complete(T, Omega, tau, delta=1.0, iters=2000):
    # step size delta must stay in (0, 2) for the convergence guarantee
    Y = np.zeros_like(T)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        X = (U * np.maximum(s - tau, 0)) @ Vt
        Y += delta * Omega * (T - X)
    return X

T = np.block([[Sr, A], [A.T, Sd]])           # the DRRS heterogeneous matrix (7 x 7)
Omega = T != 0                               # "observed" = non-zero, as in DRRS
X = svt_complete(T, Omega, tau=0.5)
print("completed drug-disease block (DRRS-style):\n", X[:4, 4:])
print("ranking of drugs for cold disease 2:", np.argsort(-X[:4, 6]))

# The same algorithm on A ALONE: the cold column has no observed entry at all
X_A = svt_complete(A, A != 0, tau=0.5)
print("SVT on A alone, cold column:", X_A[:, 2])
```

Output:

```text
completed drug-disease block (DRRS-style):
 [[ 1.     0.136  0.026]
 [ 1.     1.     0.154]
 [ 0.067  0.095 -0.05 ]
 [ 0.031  1.     0.11 ]]
ranking of drugs for cold disease 2: [1 3 0 2]
SVT on A alone, cold column: [0. 0. 0. 0.]
```

The known links are reproduced exactly (they are in $\Omega$); the cold disease 2 receives the ranking (1, 3, 0, 2), the same as MBiRW and the heterogeneous random walk in Chapter 09. SVT on $A$ alone has no observed entry in column 2 and returns exactly zero there (Section 10.4).

### 15.9 The bilinear decoder is a factorisation; the degree gate

```python
import numpy as np
rng = np.random.default_rng(0)
np.set_printoptions(precision=3, suppress=True)

Hr, Hd = rng.normal(size=(5, 4)), rng.normal(size=(3, 4))    # drug / disease embeddings
W = rng.normal(size=(4, 4))                                  # learned bilinear matrix

logits = Hr @ W @ Hd.T                                       # the bilinear decoder

# Any W can be split as W = L R^T (here via its SVD), so the decoder is an inner product
# between "drug factors" U = Hr L and "disease factors" V = Hd R: a matrix factorisation.
P, s, Qt = np.linalg.svd(W)
L, R = P * np.sqrt(s), Qt.T * np.sqrt(s)
U, V = Hr @ L, Hd @ R
print("Hr W Hd^T equals U V^T:", np.allclose(logits, U @ V.T))
print("rank of the logit matrix:", np.linalg.matrix_rank(logits), "(at most min(#drugs, #diseases, embedding size) = 3)")

# DistMult-style diagonal W: a weighted dot product
w = rng.normal(size=4)
print("diagonal W equals weighted dot product:",
      np.allclose(Hr @ np.diag(w) @ Hd.T, (Hr * w) @ Hd.T))

# The degree gate of model.py at its initial values gate = [0, 1, 0, 1]
sig = lambda x: 1 / (1 + np.exp(-x))
for deg_r, deg_d in [(5, 0), (5, 1), (5, 10), (20, 30)]:
    g = sig(0 + 1 * np.log1p(deg_r)) * sig(0 + 1 * np.log1p(deg_d))
    print(f"drug with {deg_r:2d} links, disease with {deg_d:2d} links -> gate {g:.3f}")
```

Output:

```text
Hr W Hd^T equals U V^T: True
rank of the logit matrix: 3 (at most min(#drugs, #diseases, embedding size) = 3)
diagonal W equals weighted dot product: True
drug with  5 links, disease with  0 links -> gate 0.429
drug with  5 links, disease with  1 links -> gate 0.571
drug with  5 links, disease with 10 links -> gate 0.786
drug with 20 links, disease with 30 links -> gate 0.925
```

The first line confirms Section 14's identity numerically; the logit matrix's rank is limited by the smallest dimension. The gate rows show the degree gate at its initial parameters: a pair involving a disease with no visible link keeps only 43% of the GNN term, a pair of well-connected entities 93%. Training then adjusts the four gate parameters.

### 15.10 Inductive matrix completion on unseen diseases

```python
import numpy as np
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)

# Drugs and diseases come with side features (think: fingerprint bits, phenotype terms)
n, m, f_r, f_d = 80, 50, 10, 8
X = rng.normal(size=(n, f_r))                 # drug features
Y = rng.normal(size=(m, f_d))                 # disease features
W_true = rng.normal(size=(f_r, 2)) @ rng.normal(size=(2, f_d))   # low-rank interaction
A = (X @ W_true @ Y.T > 2.5).astype(float)    # links come from features through W
new = np.arange(40, 50)                       # 10 diseases never seen in training
old = np.arange(40)

# Inductive matrix completion, least-squares flavour:
#   min_W || A[:, old] - X W Y[old]^T ||_F^2 + lam ||W||_F^2
# vec(X W Y^T) = (Y kron X) vec(W)   (column-major vec)  -> a ridge regression in vec(W)
lam = 1.0
K = np.kron(Y[old], X)                         # (n*40) x (f_r*f_d) design matrix
b = A[:, old].ravel(order="F")
w = np.linalg.solve(K.T @ K + lam * np.eye(K.shape[1]), K.T @ b)
W = w.reshape(f_r, f_d, order="F")

S_new = X @ W @ Y[new].T                       # scores for the 10 NEW diseases
print("IMC on 10 unseen diseases: AUC", round(roc_auc_score(A[:, new].ravel(), S_new.ravel()), 3))
print("number of learned parameters:", W.size, "(vs", (n + m) * 2, "for a rank-2 U, V)")
```

Output:

```text
IMC on 10 unseen diseases: AUC 0.947
number of learned parameters: 80 (vs 260 for a rank-2 U, V)
```

The model never saw a single link of diseases 40–49, yet ranks drugs for them almost perfectly, because associations here really are a function of the features. With 80 parameters it is also far smaller than a transductive factorisation. On real data, features explain associations only partly, which is why NIMCGCN-style methods underperform when the features are weak.

---

## 16. In this project: the code, line by line

> Code excerpts below are copied from the project files. Comments such as `# (1)` were added for this explanation (they refer to the numbered notes under each excerpt), and `...` marks lines left out. Open the real file alongside.

All baselines share the interface `scores = method.fit_predict(data, A_train, neg_mask, seed)`, where `A_train` is the drugs × diseases matrix with the test links of the current fold set to 0, and `neg_mask` marks the cells that may be used as training negatives (unknown cells *outside* the test fold). `bench_sims(data)` returns the two benchmark similarities (`chem_cdk` for drugs, `pheno_mim` for diseases) with missing rows filled and a unit diagonal — the same information the original papers used.

### 16.1 SCMFDD — `class SCMFDD` in `src/drepo/methods.py`

```python
class SCMFDD:
    """Zhang et al. 2018. Similarity-constrained matrix factorisation:
    ||A - U V'||^2 + mu(||U||^2+||V||^2) + lam(tr U'Lr U + tr V'Ld V)."""
    def __init__(self, k=128, mu=0.05, lam=2.0, epochs=400, lr=0.02):
        ...
    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        Lr = t(np.eye(len(Sr)) - sym_norm(Sr))                         # (1)
        Ld = t(np.eye(len(Sd)) - sym_norm(Sd))
        A = t(A_train)                                                 # (2)
        U = nn.Parameter(0.1 * torch.randn(A.shape[0], self.k, device=DEVICE))   # (3)
        V = nn.Parameter(0.1 * torch.randn(A.shape[1], self.k, device=DEVICE))
        opt = torch.optim.Adam([U, V], lr=self.lr)                     # (4)
        for _ in range(self.epochs):
            loss = ((A - U @ V.T) ** 2).sum() + self.mu * (U.pow(2).sum() + V.pow(2).sum()) \
                + self.lam * (torch.trace(U.T @ Lr @ U) + torch.trace(V.T @ Ld @ V))   # (5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (U @ V.T).detach().cpu().numpy()                        # (6)
```

1. **Normalised Laplacians** $L = I - D^{-1/2}SD^{-1/2}$ of the drug and disease similarity graphs (`sym_norm` is $D^{-1/2}SD^{-1/2}$; the unit diagonal of the filled similarities acts as a self-loop). The paper writes the penalty with the unnormalised Laplacian ($\sum w_{ij}\lVert\mathbf x_i-\mathbf x_j\rVert^2 = 2\operatorname{tr}(X^\top(D-W)X)$); the normalised one compares degree-scaled factors and keeps the penalty's scale independent of how dense each similarity matrix is.
2. The **whole** training matrix, zeros included (test cells are zeros too): this is the AMAN choice of Section 3.2. `neg_mask` is not used — SCMFDD never samples negatives.
3. Factor matrices $U$ (593 × 128) and $V$ (313 × 128), initialised small. $k=128$ was tuned on the validation split (the paper used $k$ as a percentage of the matrix size).
4. Adam on the full objective instead of the paper's row-wise closed-form updates; both minimise the same kind of objective, Adam is simpler and runs on the GPU.
5. The loss is exactly Section 9.2's objective (without the ½ factors): squared reconstruction error over all cells $+$ $\mu$ times the L2 norms $+$ $\lambda$ times the two Laplacian trace penalties. `torch.trace(U.T @ Lr @ U)` is $\operatorname{tr}(U^\top L_rU)$. With $\mu=0.05$ and $\lambda=2$ (tuned), the smoothness term is much stronger than the plain norm penalty: similar drugs are strongly encouraged to share factors.
6. Scores are $UV^\top$, not probabilities (they can be slightly negative or above 1); only the ranking matters for AUC/AUPR.

Because of the Laplacian term, SCMFDD *can* give a cold disease a non-zero factor vector: its only data term is "fit zeros", but the smoothness term pulls it towards its phenotype neighbours (Section 9.4).

### 16.2 DRRS — `class DRRS`

```python
class DRRS:
    def __init__(self, tau_rel=0.005, iters=200, rank=200):
        ...
    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        n_r = Sr.shape[0]
        T = t(np.block([[Sr, A_train], [A_train.T, Sd]]))         # (1)
        Om = (T != 0).float()                                      # (2)
        N = T.shape[0]
        delta = 1.2 * N * N / Om.sum()                             # (3)
        tau = self.tau_rel * torch.linalg.matrix_norm(T, ord=2) * N / 10   # (4)
        Y = torch.zeros_like(T)
        X = T
        for _ in range(self.iters):
            U, S, V = torch.svd_lowrank(Y if Y.abs().sum() > 0 else T, q=self.rank, niter=2)  # (5)
            S = torch.clamp(S - tau, min=0)                        # (6)
            X = (U * S) @ V.T                                      # (7)
            Y = Y + delta * Om * (T - X)                           # (8)
        return X[:n_r, n_r:].cpu().numpy()                         # (9)
```

1. The heterogeneous block matrix $T$ of Section 12.1, $906\times906$ on Fdataset.
2. $\Omega$ = the non-zero entries: training links and non-zero similarities. Unknown drug–disease pairs (zeros) are *unobserved*.
3. CCS's step size $\delta = 1.2\,N^2/|\Omega|$; about 2.28 on Fdataset (52.6% of $T$ is observed).
4. The threshold $\tau$: a fixed fraction of the largest singular value of $T$ (about 118 on Fdataset), scaled by $N/10$; about 54 on Fdataset. Only singular values above this survive each shrinkage, so the completion is very low-rank.
5. A **randomised partial SVD** (200 components, 2 power iterations) instead of a full SVD — this is the "randomized algorithms" of DRRS's title and keeps each iteration cheap. In the first iteration $Y=0$, so it decomposes $T$ itself: a warm start giving $X^1 = \mathcal D_\tau(T)$ instead of the useless $X^1=0$.
6. Shrink the singular values by $\tau$ and drop the negative ones: $\mathcal D_\tau$.
7. Rebuild $X = U\,\mathrm{diag}(S)\,V^\top$ (`U * S` scales the columns of $U$; `torch.svd_lowrank` returns $V$, not $V^\top$).
8. The dual (Uzawa) step $Y\leftarrow Y+\delta\,\mathcal P_\Omega(T-X)$: `Om *` is $\mathcal P_\Omega$.
9. The completed drug–disease block is the score matrix.

### 16.3 NIMCGCN and `weighted_bce`

```python
def weighted_bce(logits, A, neg_mask):
    m = (A > 0) | neg_mask                                   # cells allowed in the loss
    pw = (m & (A == 0)).sum() / max((A > 0).sum(), 1)        # #negatives / #positives (~95)
    w = torch.where(A > 0, pw, torch.ones_like(A)) * m       # positives weighted pw, negatives 1, test cells 0
    return (F.binary_cross_entropy_with_logits(logits, A, reduction="none") * w).sum() / w.sum()
```

This is the **weighting** answer to implicit feedback in cross-entropy form: every allowed unknown is a negative with weight 1, every positive has weight ≈ 95 so the two classes contribute equally. Test cells get weight 0 (no leakage).

```python
class NIMCGCN:
    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        Ar = t(sym_norm(Sr * knn_mask(Sr, self.k)))          # 10-NN drug graph, D^-1/2 S D^-1/2
        Ad = t(sym_norm(Sd * knn_mask(Sd, self.k)))
        Xr, Xd = t(Sr), t(Sd)                                # features = similarity rows
        gr = _GCN([Xr.shape[1], self.h, self.o], self.dropout).to(DEVICE)   # 2-layer GCN, 128 -> 64
        gd = _GCN([Xd.shape[1], self.h, self.o], self.dropout).to(DEVICE)
        fr, fd = nn.Linear(self.o, self.o).to(DEVICE), nn.Linear(self.o, self.o).to(DEVICE)
        ...
        for _ in range(self.epochs):
            logits = fr(gr(Ar, Xr)) @ fd(gd(Ad, Xd)).T       # neural IMC: inner product of projected embeddings
            loss = weighted_bce(logits, A, nm)
            ...
        with torch.no_grad():
            return torch.sigmoid(fr(gr(Ar, Xr)) @ fd(gd(Ad, Xd)).T).cpu().numpy()
```

Line by line: build sparse, normalised similarity graphs; use each entity's similarity row as its input feature; encode drugs and diseases with separate two-layer GCNs; project each with a linear layer (the "$G$" and "$H$" of IMC, Section 13.1); score by inner product; train with `weighted_bce`. Note that **`A` appears only in the loss**, never in the encoder inputs — the root of its weak results (Section 13.2).

### 16.4 MV-HGAT: logistic factorisation with negative sampling

Configuration (`MVHGATConfig` in `src/drepo/methods.py`):

```python
neg_ratio: int = 2                    # sampled negatives per positive per epoch
```

Inside `MVHGATMethod.fit_predict`:

```python
pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)                          # (1)
neg_pool = t(np.flatnonzero((neg_mask & (A_train == 0)).ravel()), torch.long)     # (2)
...
for _ in range(c.epochs):
    ...                                     # hide ~20% of links + all links of ~10% of diseases
    if c.supervise_hidden:
        sup = pos[hide]                                                           # (3)
    ...
    logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
    logits = logits.reshape(-1)
    n_neg = min(neg_pool.numel(), sup.numel() * c.neg_ratio) if c.neg_ratio > 0 \
        else neg_pool.numel()                                                     # (4)
    neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]      # (5)
    y = torch.cat([torch.ones(sup.numel(), device=DEVICE),
                   torch.zeros(n_neg, device=DEVICE)])                            # (6)
    loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)   # (7)
```

1. Flat indices ($i\cdot313+j$) of all training positives.
2. The **negative pool**: cells that are unknown *and* allowed by `neg_mask` — never a test cell, never a known link. On an Fdataset 5-fold split, about 146,900 cells.
3. The positives supervised this epoch are the ones **hidden** from the input graph and features (about 28% of the ~1,546 training links: 20% random plus every link of the ~10% "cold" diseases), so the model must predict them, not copy them.
4. Number of negatives $=\rho\times|\text{sup}|$ with $\rho$ = `neg_ratio` = 2, so about 866 negatives per epoch. `neg_ratio=0` would use the whole pool (AMAN).
5. **Uniform sampling with replacement**, freshly every epoch. Each pool cell is drawn with probability about 0.6% per epoch, i.e. about 3.5 times over 600 epochs — each unknown is a *rare, soft* negative.
6. Labels: 1 for the supervised positives, 0 for the sampled unknowns.
7. Binary cross-entropy on the logits = **logistic matrix factorisation** (Section 5.2) whose factors come from the GNN. With positives kept at about 28% and negatives at about 0.6% per epoch, Section 7.3 predicts a logit inflation of roughly $\ln(0.28/0.0059)\approx3.9$: the output "probabilities" are much higher than real repositioning probabilities, but the ranking — all that AUC and AUPR measure — is unaffected by a constant shift.

The decoder (`MVHGAT.forward` in `src/drepo/model.py`):

```python
self.W = nn.Parameter(torch.empty(out, out))     # out = hidden * (layers + 1) = 192
nn.init.xavier_uniform_(self.W)
...
logits = Hr @ self.W @ Hd.T                      # bilinear: a learned factorisation of rank <= 192
```

`Hr` (593 × 192) and `Hd` (313 × 192) are the jumping-knowledge concatenations of the input projection and the two GAT layers. `Hr @ self.W @ Hd.T` is $H_rWH_d^\top = (H_rL)(H_dR)^\top$: matrix factorisation with encoder-computed factors (Section 14). Xavier initialisation keeps the initial logits at a sensible scale. The gate and the propagation head are then added as described in Chapter 09.

---

## 17. Common mistakes and misconceptions

1. **Treating unknown pairs as confirmed negatives.** They are unlabelled; some are the discoveries you are looking for. Use soft negatives (weights, re-sampled negatives, ranking losses) and never interpret a high-scoring "false positive" as a model error without checking the literature.
2. **Sampling negatives from the test fold.** Any test cell used as a training negative leaks the test labels (it tells the model "this cell is 0"). `neg_mask` exists to prevent this.
3. **Reading probabilities from a model trained with negative sampling.** The logits are shifted by $\ln(s_+/s_-)$; the outputs are rankings, not calibrated probabilities.
4. **Assuming a bigger negative ratio is "more data, so better".** More sampled negatives per epoch means more true-but-unknown positives labelled 0, and a loss dominated by easy negatives (ratio 20 hurt in Section 15.5).
5. **Interpreting individual latent factors.** MF factors are only defined up to rotation; "factor 7 = anti-inflammatory" is not meaningful without extra constraints (e.g. non-negativity).
6. **Forgetting regularisation in ALS.** Without $\lambda>0$, $V_{\Omega_i}^\top V_{\Omega_i}$ is singular for any drug with fewer than $k$ observed cells (Exercise 3), and factors explode.
7. **Expecting plain MF, BPR-MF or SVT on $A$ to score a cold disease.** An entity with no observed entries gets a zero (or untrained) factor. You need side information: a Laplacian penalty (SCMFDD), similarity blocks (DRRS), features (IMC, GNNs) or propagation.
8. **Mixing up what is "observed" in completion methods.** In DRRS the zeros of $A$ are *unobserved*; in SCMFDD they are *observed zeros*. Changing this changes the method completely.
9. **Using the CCS step size blindly.** $\delta=1.2/p$ exceeds the proven limit of 2 when fewer than 60% of entries are observed and can diverge (Exercise 10).
10. **Comparing BPR models with pooled metrics without thinking.** BPR does not put different diseases on a common scale; pooled AUPR rewards global calibration that BPR never optimised.
11. **Thinking "neural" or "inductive" beats "simple".** On Fdataset the simplest regularised MF (SCMFDD) has the best AUPR and the neural IMC (NIMCGCN) the worst. What information the model sees, and how unknowns are treated, matter more than architecture.
12. **Tuning $k$, $\lambda$, $\mu$ on the test folds.** The project tunes on a separate validation split (`scripts/05_sensitivity.py`); otherwise the comparison with baselines is unfair.

---

## 18. Exercises

Difficulty: ★ conceptual, ★★ standard, ★★★ challenging.

**Exercise 1 (★, conceptual).** In recommender-system language, what are (a) a cold-start user, (b) a popular item, (c) implicit feedback, and (d) item-based collaborative filtering, *in this project*? Where in the code does (d) appear?

<details><summary>Solution</summary>

(a) A disease with no known drug — the leave-one-disease-out setting (`evaluation.py::run_lodo`), also simulated in training by `cold_frac`. (b) A drug with many known indications (e.g. a corticosteroid); popularity bias means models tend to rank such drugs high for everything. (c) The known indications in $A$: positive-only evidence, where 0 means "unknown", not "dislike". (d) Scoring a drug for a disease by the links of *similar drugs* to that disease. In the project it is the drug-view propagation $P_v = K_vA$ inside `propagation()` in `MVHGATMethod.fit_predict`, with similarity taken from chemistry/genes rather than from co-occurrence; the disease-view terms are user-based CF.
</details>

**Exercise 2 (★★, maths).** Compute by hand the singular values of $C=\begin{pmatrix}1&1&0\\1&1&1\\1&0&1\end{pmatrix}$ (the immunology block of worked example 4.1). Hint: find the eigenvalues of $C^\top C$; one eigenvector is $(1,1,1)^\top$-like.

<details><summary>Solution</summary>

$C^\top C = \begin{pmatrix}3&2&2\\2&2&1\\2&1&2\end{pmatrix}$. Try $\mathbf x = (0,1,-1)$: $C^\top C\mathbf x = (0,1,-1)$, so $\mu=1$. The remaining eigenvectors are orthogonal to it, of the form $(a,b,b)$: $C^\top C(a,b,b) = (3a+4b,\ 2a+3b,\ 2a+3b)$. Eigen-equation: $3a+4b=\mu a$, $2a+3b=\mu b$. From the second, $a = (\mu-3)b/2$; substituting, $3(\mu-3)/2+4 = \mu(\mu-3)/2$, i.e. $\mu^2-6\mu+1=0$, so $\mu = 3\pm2\sqrt2 = 5.828,\ 0.172$. Singular values are square roots: $\sqrt{3+2\sqrt2} = 1+\sqrt2 = 2.414$, $1$, and $\sqrt{3-2\sqrt2} = \sqrt2-1 = 0.414$. Check: $\sum\sigma_i^2 = 5.828+1+0.172 = 7 = \lVert C\rVert_F^2$ (seven ones). ✓
</details>

**Exercise 3 (★★, maths).** (a) Redo worked example 5.1's $U$-step with $\lambda=0$. (b) Explain why ALS *needs* $\lambda>0$ when $k=2$ and some drug has exactly one observed disease.

<details><summary>Solution</summary>

(a) $V=(1,1)^\top$, $V^\top V=2$: $u_0 = (1\cdot1+0\cdot1)/2 = 0.5$, $u_1 = (1+1)/2 = 1$. Without shrinkage the factors are larger.
(b) The row update solves $(V_{\Omega_i}^\top V_{\Omega_i}+\lambda I)\mathbf u_i = V_{\Omega_i}^\top\mathbf a_{i,\Omega_i}$. With one observed disease, $V_{\Omega_i}$ is $1\times2$, so $V_{\Omega_i}^\top V_{\Omega_i}$ is a $2\times2$ matrix of rank 1: singular. Infinitely many $\mathbf u_i$ fit the single observation perfectly, and nothing stops them from being huge. $\lambda>0$ makes the matrix positive definite and picks the minimum-norm solution — the Gaussian prior of PMF at work. In any formulation where only some cells count as observed (e.g. completion-style training on known links), a 1%-dense matrix leaves many drugs with fewer observed cells than $k$, so this is the normal case in drug repositioning, not an edge case.
</details>

**Exercise 4 (★★, maths).** Continue worked example 5.2 with a second SGD step on the same cell ($a=1$, $\eta=0.1$, $\lambda=0.1$, starting from $u=0.527$, $v=0.436$). What is the new prediction?

<details><summary>Solution</summary>

Prediction $0.527\times0.436 = 0.2298$, error $e = 0.7702$.
$u\leftarrow0.527+0.1(0.7702\cdot0.436-0.1\cdot0.527) = 0.527+0.1(0.3358-0.0527) = 0.5553$;
$v\leftarrow0.436+0.1(0.7702\cdot0.527-0.1\cdot0.436) = 0.436+0.1(0.4059-0.0436) = 0.4722$.
New prediction $0.5553\times0.4722\approx0.262$. Progress is slow (0.200 → 0.230 → 0.262) because the factors are small and $\eta$ is small; the product structure means the step size effectively grows as the factors grow.
</details>

**Exercise 5 (★★, maths + project).** In one Fdataset 5-fold training run of MV-HGAT, there are about 1,546 training links, of which about 28% are supervised per epoch, the negative pool has about 146,941 cells, and `neg_ratio=2`. (a) Estimate the per-epoch sampling rates $s_+$ and $s_-$. (b) By how much are the trained logits shifted relative to "true" logits, in the idealised analysis of Section 7.3? (c) Does it change AUC?

<details><summary>Solution</summary>

(a) $s_+\approx0.28$ (a positive is supervised when hidden). Negatives per epoch $\approx2\times0.28\times1546\approx866$, so $s_-\approx866/146{,}941\approx0.0059$.
(b) $\ln(s_+/s_-)\approx\ln(0.28/0.0059)\approx\ln47.5\approx3.86$: logits are inflated by about 3.9, i.e. odds by a factor of about 48.
(c) No — a constant added to every logit preserves the ranking, so AUC and AUPR are unchanged (in the idealised analysis). It matters only if you read the sigmoid output as a probability. (In reality the shift is not perfectly constant because the model is not perfectly flexible, which is why `neg_ratio` is still tuned.)
</details>

**Exercise 6 (★★, maths).** One BPR step. Disease embedding $\mathbf w_u=(1,0)$, known drug $\mathbf h_i=(0.2,0.5)$, unknown drug $\mathbf h_j=(0.5,0.1)$, learning rate $\eta=0.1$, no regularisation. Compute the loss, take one gradient-ascent step on $\ln\sigma(\hat x_{uij})$ for all three vectors, and recompute $\hat x_{uij}$.

<details><summary>Solution</summary>

$\hat x_{ui} = 0.2$, $\hat x_{uj} = 0.5$, $\hat x_{uij}=-0.3$. Loss $=\ln(1+e^{0.3}) = 0.854$. Weight $g=\sigma(0.3) = 0.5744$.
$\mathbf w_u\leftarrow\mathbf w_u+\eta g(\mathbf h_i-\mathbf h_j) = (1,0)+0.05744\,(-0.3,0.4) = (0.9828, 0.0230)$;
$\mathbf h_i\leftarrow\mathbf h_i+\eta g\,\mathbf w_u = (0.2574, 0.5)$;
$\mathbf h_j\leftarrow\mathbf h_j-\eta g\,\mathbf w_u = (0.4426, 0.1)$ (all using the old $\mathbf w_u$).
New scores: $\hat x_{ui} = 0.9828\cdot0.2574+0.0230\cdot0.5 = 0.2645$; $\hat x_{uj} = 0.9828\cdot0.4426+0.0230\cdot0.1 = 0.4372$; $\hat x_{uij} = -0.173$ (was $-0.300$); the loss fell to 0.783. The disease vector rotated towards the known drug's direction and the two drug vectors moved apart.
</details>

**Exercise 7 (★★, maths).** (a) Prove that for the normalised Laplacian $\mathcal L = I-D^{-1/2}SD^{-1/2}$, $\operatorname{tr}(U^\top\mathcal LU) = \tfrac12\sum_{ij}S_{ij}\lVert\mathbf u_i/\sqrt{d_i}-\mathbf u_j/\sqrt{d_j}\rVert^2$. (b) In words, what does $\operatorname{tr}(U^\top LU)$ penalise in SCMFDD, and how does the normalised version differ?

<details><summary>Solution</summary>

(a) Let $\tilde U = D^{-1/2}U$ (rows $\mathbf u_i/\sqrt{d_i}$). Then $U^\top\mathcal LU = U^\top U - \tilde U^\top S\tilde U$ and $U^\top U = \tilde U^\top D\tilde U$, so $\operatorname{tr}(U^\top\mathcal LU) = \operatorname{tr}(\tilde U^\top(D-S)\tilde U) = \tfrac12\sum_{ij}S_{ij}\lVert\tilde{\mathbf u}_i-\tilde{\mathbf u}_j\rVert^2$ by the lemma of Section 9.1.
(b) It penalises **differences between the latent factors of similar entities**, each difference weighted by the similarity; pairs with zero similarity are free. In SCMFDD this forces chemically similar drugs (and phenotypically similar diseases) to have similar factors and hence similar predicted indication profiles, and gives entities without links the factors of their neighbours. The normalised version compares factors after dividing by $\sqrt{\text{degree}}$, so entities with very many similar neighbours do not dominate the penalty and the penalty's scale does not depend on the overall density of the similarity matrix.
</details>

**Exercise 8 (★★, maths).** Apply $\mathcal D_1$ (SVT with $\tau=1$) by hand to (a) $\begin{pmatrix}3&4\\0&0\end{pmatrix}$, (b) $\mathrm{diag}(5,2,0.5)$, (c) $\begin{pmatrix}0.6&0\\0&0.8\end{pmatrix}$. What is the nuclear norm before and after each?

<details><summary>Solution</summary>

(a) Rank 1, $\sigma_1=5$, singular vectors $\mathbf u=(1,0)$, $\mathbf v=(0.6,0.8)$. Shrunk: $\sigma_1=4$, so the result is $\tfrac45$ of the matrix, $\begin{pmatrix}2.4&3.2\\0&0\end{pmatrix}$. Nuclear norm $5\to4$.
(b) $\mathrm{diag}(4,1,0)$; rank $3\to2$; nuclear norm $7.5\to5$.
(c) Both singular values (0.8 and 0.6) are below 1: the result is the zero matrix; nuclear norm $1.4\to0$. Shrinkage removes everything weaker than $\tau$.
</details>

**Exercise 9 (★★★, proof).** Prove $\lVert X\rVert_* = \min_{UV^\top=X}\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$, and explain what it implies for L2-regularised matrix factorisation.

<details><summary>Solution</summary>

*Upper bound for the minimum.* With the SVD $X=P\Sigma Q^\top$, take $U=P\Sigma^{1/2}$, $V=Q\Sigma^{1/2}$: $UV^\top = X$ and $\lVert U\rVert_F^2 = \operatorname{tr}(\Sigma^{1/2}P^\top P\Sigma^{1/2}) = \operatorname{tr}\Sigma = \lVert X\rVert_*$, likewise for $V$. So the minimum is at most $\lVert X\rVert_*$.
*Lower bound.* For any $U,V$ with $UV^\top=X$: $\lVert X\rVert_* = \operatorname{tr}(P^\top XQ) = \operatorname{tr}(P^\top UV^\top Q) = \langle U^\top P, V^\top Q\rangle_F\le\lVert U^\top P\rVert_F\lVert V^\top Q\rVert_F\le\lVert U\rVert_F\lVert V\rVert_F\le\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$, using Cauchy–Schwarz, the fact that multiplying by a matrix with orthonormal columns cannot increase the Frobenius norm, and AM–GM ($ab\le\tfrac12(a^2+b^2)$).
*Implication.* $\min_{U,V}\lVert\mathcal P_\Omega(A-UV^\top)\rVert^2+\lambda(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$ equals $\min_X\lVert\mathcal P_\Omega(A-X)\rVert^2+2\lambda\lVert X\rVert_*$ over matrices of rank at most $k$. So regularised MF is nuclear-norm-regularised completion in disguise: $\lambda$ controls an effective rank, and choosing $k$ larger than necessary is harmless when $\lambda$ is well tuned (the project's SCMFDD uses $k=128$).
</details>

**Exercise 10 (★★, coding).** Run SVT on the toy $A$ alone and on the DRRS block matrix, with step sizes $\delta\in\{1, 1.9, 1.2/p\}$ where $p$ is the observed fraction. What happens, and what does it mean for the project's DRRS ($\delta\approx2.28$)?

<details><summary>Solution</summary>

```python
import numpy as np
A = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0]], float)   # toy of Example 8
Sr = np.array([[1.0, 0.8, 0.3, 0.1], [0.8, 1.0, 0.2, 0.1],
               [0.3, 0.2, 1.0, 0.7], [0.1, 0.1, 0.7, 1.0]])
Sd = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.9], [0.3, 0.9, 1.0]])

def svt_status(T, Omega, tau, delta, iters=2000):
    Y = np.zeros_like(T)
    for k in range(iters):
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        X = (U * np.maximum(s - tau, 0)) @ Vt
        Y = Y + delta * Omega * (T - X)
        if np.abs(Y).max() > 1e8:
            return f"diverged (|Y| > 1e8 after {k + 1} iterations)"
    return f"converged, max residual on observed entries {np.abs(Omega * (T - X)).max():.1e}"

for name, T in [("A alone", A), ("DRRS block matrix", np.block([[Sr, A], [A.T, Sd]]))]:
    Omega = T != 0
    heuristic = 1.2 * T.size / Omega.sum()
    print(f"{name}: observed fraction {Omega.mean():.2f}")
    for delta in (1.0, 1.9, heuristic):
        print(f"   delta = {delta:.2f}: {svt_status(T, Omega, 0.5, delta)}")
```

```text
A alone: observed fraction 0.33
   delta = 1.00: converged, max residual on observed entries 0.0e+00
   delta = 1.90: converged, max residual on observed entries 1.4e-15
   delta = 3.60: diverged (|Y| > 1e8 after 20 iterations)
DRRS block matrix: observed fraction 0.67
   delta = 1.00: converged, max residual on observed entries 1.2e-15
   delta = 1.90: converged, max residual on observed entries 2.9e-15
   delta = 1.78: converged, max residual on observed entries 4.0e-15
```

For $A$ alone only a third of the entries are observed, so the heuristic gives $\delta=3.6$, well above the proven limit of 2, and the iteration explodes within 20 steps. For the block matrix, $p=0.67$ and the heuristic $\delta=1.78<2$ is safe. The project's DRRS on Fdataset uses $\delta\approx2.28$ — above the limit; it is kept stable in practice by the large threshold (only a few singular values survive), but it is the first thing to reduce if a modified version diverges.
</details>

**Exercise 11 (★★, coding).** Using the synthetic data of Section 15.5, train the Hu–Koren–Volinsky weighted ALS with confidence $\alpha\in\{0,5,20,100\}$ ($\alpha=0$ is plain AMAN least squares) and compare test AUC/AUPR. Explain the trend.

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Same synthetic data and split as Example 5 (code repeated so this runs on its own)
rng = np.random.default_rng(0)
n, m, r = 200, 100, 4
logit_true = rng.normal(size=(n, r)) @ rng.normal(size=(r, m))
truth = logit_true > np.quantile(logit_true, 0.97)
known = truth & (rng.random((n, m)) < 0.7)
pos, unk = np.flatnonzero(known), np.flatnonzero(~known)
test_pos = rng.choice(pos, len(pos) // 5, replace=False)
test_unk = rng.choice(unk, len(unk) // 5, replace=False)
A_train = known.ravel().copy(); A_train[test_pos] = False
A_train = A_train.reshape(n, m).astype(float)
test_idx = np.concatenate([test_pos, test_unk])
y_test = np.r_[np.ones(len(test_pos)), np.zeros(len(test_unk))]

def implicit_als(P, alpha, k=8, lam=1.0, iters=15, seed=0):
    g = np.random.default_rng(seed)
    U = 0.1 * g.normal(size=(n, k)); V = 0.1 * g.normal(size=(m, k)); I = np.eye(k)
    C = 1 + alpha * P                          # confidence: 1 for unknowns, 1+alpha for known links
    for _ in range(iters):
        VtV = V.T @ V
        for i in range(n):
            U[i] = np.linalg.solve(VtV + (V.T * (C[i] - 1)) @ V + lam * I, (V.T * C[i]) @ P[i])
        UtU = U.T @ U
        for j in range(m):
            V[j] = np.linalg.solve(UtU + (U.T * (C[:, j] - 1)) @ U + lam * I, (U.T * C[:, j]) @ P[:, j])
    return U @ V.T

for alpha in (0, 5, 20, 100):
    s = implicit_als(A_train, alpha).ravel()[test_idx]
    print(f"alpha = {alpha:3d}: test AUC {roc_auc_score(y_test, s):.3f}  AUPR {average_precision_score(y_test, s):.3f}")
```

```text
alpha =   0: test AUC 0.773  AUPR 0.192
alpha =   5: test AUC 0.782  AUPR 0.216
alpha =  20: test AUC 0.778  AUPR 0.172
alpha = 100: test AUC 0.770  AUPR 0.179
```

A moderate confidence ($\alpha=5$) is best: it tells the model that known links are much more trustworthy than unknowns, so true-but-unknown cells are not forced to 0 as hard as in AMAN. Very large $\alpha$ makes the model fit the known links almost exactly and care little about the unknowns, which overfits the ~330 training links. The differences are modest here — the same lesson as the negative-sampling ratio: treat unknowns as *weak*, not absent and not certain, negatives. (Pointwise BCE with negative sampling, Section 15.5, reached AUPR ≈ 0.20 on the same split.)
</details>

**Exercise 12 (★★, conceptual).** DRRS treats the zeros of $A$ as *unobserved*; SCMFDD treats them as *observed zeros*. Give one advantage and one risk of each choice.

<details><summary>Solution</summary>

*DRRS (unobserved):* advantage — a true-but-unknown indication is never pushed towards 0 by the data term, matching the PU nature of the problem; risk — with so few positives, only the low-rank constraint (and the similarity blocks) prevent trivial solutions such as filling the block with large values; the predicted magnitudes depend heavily on $\tau$ and the similarity blocks, and the 1s alone carry little information about *where* not to predict.
*SCMFDD (observed zeros):* advantage — 99% of the cells give a strong, stable signal about which regions are empty, which sharpens the dense blocks and gives high precision at the top of the ranking (its AUPR is the best on Fdataset); risk — every undiscovered indication is actively trained towards 0 (AMAN bias), and the ordering among low-scored pairs is poor (lower AUC).
</details>

**Exercise 13 (★★, conceptual).** On Fdataset, SCMFDD has AUPR 0.495 versus MV-HGAT's 0.488, but AUC 0.893 versus 0.939. How can one method win on AUPR and lose on AUC? Which matters more for repositioning?

<details><summary>Solution</summary>

AUPR is dominated by the top of the ranking (precision among the highest scores); AUC weighs every positive–negative pair equally, including the long tail. SCMFDD reconstructs the dense blocks of $A$ very sharply, so its very top predictions are about as precise as MV-HGAT's; but positives outside those blocks (unusual indications) are scored near 0 among thousands of unknowns, which lowers AUC. MV-HGAT orders the whole list better. For repositioning, experimental follow-up is only feasible for the top few predictions per disease, so the top of the ranking (AUPR, precision@k) is what matters most in practice — but AUC tells you how robust a method is for diseases whose drugs are not "typical", and cold start (which SCMFDD barely handles) matters for new diseases. The honest summary: tied on AUPR, MV-HGAT better on AUC and cold start.
</details>

**Exercise 14 (★★★, design).** Which of the following can produce a meaningful score for a disease with *no* known drug, and through which mechanism? Plain MF; BPR-MF; SVT on $A$ alone; SCMFDD; DRRS; IMC; NIMCGCN; MV-HGAT.

<details><summary>Solution</summary>

| Method | Cold disease? | Mechanism |
|---|---|---|
| Plain MF | no | its factor is fitted to an all-zero (or empty) column → zero/untrained |
| BPR-MF | no | no positive for that disease → no training triple involves it |
| SVT on $A$ alone | no | column unobserved → minimum-nuclear-norm completion sets it to 0 |
| SCMFDD | yes (weakly) | the Laplacian term pulls its factor towards phenotype neighbours |
| DRRS | yes | its row in the $S_d$ block anchors its latent coordinates |
| IMC | yes | its embedding is a function of its features |
| NIMCGCN | yes | GCN embedding from its similarity row and neighbours |
| MV-HGAT | yes | disease-view propagation terms, similarity relations in the GNN, degree gate |

The general rule: cold start requires *side information* — similarity or features — to reach the entity through some channel that does not depend on its own links.
</details>

---

## 19. Answers to the self-check questions (PREREQUISITES.md, unit B4)

### Q1. Why is "unknown" not the same as "does not treat"? How does that affect training labels?

A 0 in the association matrix records the **absence of a recorded indication**, not evidence that the drug fails. It merges three situations: pairs that were tested and failed (rarely recorded in these databases), pairs that have never been tested (the vast majority), and pairs that work but are not yet discovered or recorded — which are precisely the targets of drug repositioning. The data are **implicit, positive-only feedback**; the problem is **positive–unlabelled** learning; and missingness is **not at random** (old, well-studied drugs and common diseases have more recorded links).

Consequences for training labels:

* A "negative" label is an *assumption* made for a gradient step. It should be **soft**: give unknowns low weight (weighted MF, `weighted_bce`), or label only a small, freshly re-sampled subset as negatives each epoch (negative sampling, MV-HGAT with `neg_ratio=2`), or use a ranking loss that only asks positives to beat unknowns (BPR), or treat unknowns as unobserved and rely on a structural prior (DRRS's low rank, BNNR).
* Hard, fixed negatives teach the model to suppress exactly the pairs it should discover; with many negatives per positive this effect grows (Section 15.5: ratio 20 lowered AUPR).
* Negatives must come only from allowed cells (`neg_mask`): never from test cells (leakage) and never from known links.
* At evaluation time, unknown test pairs are counted as negatives, so measured precision is a **lower bound**; a top-ranked "false positive" may be a correct prediction, which is why the case studies check top predictions against trials and literature.
* Predicted "probabilities" from such training are relative scores, not calibrated probabilities of therapeutic success.

### Q2. What does a negative sampling ratio of 5 mean in `MVHGATConfig`?

The field is `neg_ratio` ("sampled negatives per positive per epoch"). A ratio of 5 would mean: in **every epoch**, for each positive link supervised in that epoch, **5 cells are drawn uniformly at random (with replacement) from the negative pool** — unknown cells allowed by `neg_mask`, never test cells — and labelled 0 in the binary cross-entropy loss. So each epoch's loss sees positives and sampled unknowns in a 1 : 5 ratio; next epoch, a new random set is drawn.

Details and consequences:

* The code computes `n_neg = min(len(neg_pool), len(sup) * neg_ratio)`; `neg_ratio=0` would use the entire pool (all unknowns as negatives).
* The project's current default is **`neg_ratio=2`**, chosen on the validation split from the grid {1, 2, 5, 10} in `scripts/05_sensitivity.py` (the self-check question was written when 5 was under consideration).
* Going from 2 to 5 makes positives 1/6 instead of 1/3 of each epoch's loss, labels 2.5× as many unknowns per epoch as negatives (more true-but-unknown indications get pushed down), and, by Section 7.3, shifts the logits down by about $\ln(5/2)\approx0.92$ relative to ratio 2 — which changes the outputs' scale but, ideally, not the ranking.
* Each unknown cell remains a *rare* negative: with about 433 supervised positives per epoch and a pool of about 146,900, ratio 5 would draw about 2,165 negatives per epoch, so each cell would be labelled negative in about 1.5% of epochs.

### Bonus (from unit A2): What does $\operatorname{tr}(U^\top LU)$ penalise?

$\operatorname{tr}(U^\top LU) = \tfrac12\sum_{ij}S_{ij}\lVert\mathbf u_i-\mathbf u_j\rVert^2$: the squared distances between the factor vectors of *similar* entities, weighted by their similarity. Minimising it makes similar drugs (or diseases) have similar latent factors, which is how SCMFDD injects chemical and phenotypic similarity into MF and gives entities with few or no links sensible factors (Section 9, Exercise 7).

---

## 20. Summary and cheat sheet

| Concept | Formula / rule | Remember |
|---|---|---|
| Analogy | disease = user, drug = item, indication = implicit interaction | cold disease = new user |
| Item-based CF | $\hat a_{ij} = \sum_{i'}\text{sim}(i,i')A_{i'j}/\sum_{i'}\text{sim}(i,i')$ | = project's drug-view propagation (with side-info similarity) |
| Implicit feedback | 0 = unknown, not negative (PU learning, MNAR) | AMAN vs AMAU; weight, sample, rank, or complete |
| Low rank | $A\approx UV^\top$, rank $\le k$ | Eckart–Young: truncated SVD is optimal, error $\sqrt{\sum_{i>k}\sigma_i^2}$ |
| MF objective | $\sum_\Omega(a_{ij}-\mathbf u_i^\top\mathbf v_j)^2+\lambda(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$ | = MAP with Gaussian priors (PMF) |
| ALS row update | $\mathbf u_i=(V_{\Omega_i}^\top V_{\Omega_i}+\lambda I)^{-1}V_{\Omega_i}^\top\mathbf a_{i,\Omega_i}$ | exact block minimisation, monotone |
| SGD update | $\mathbf u_i\mathrel{+}=\eta(e_{ij}\mathbf v_j-\lambda\mathbf u_i)$, $\mathbf v_j\mathrel{+}=\eta(e_{ij}\mathbf u_i-\lambda\mathbf v_j)$ | flexible, needs tuned $\eta$ |
| Weighted MF | $\sum c_{ij}(p_{ij}-\mathbf u_i^\top\mathbf v_j)^2$, $c=1+\alpha r$ | trick: $V^\top C^iV = V^\top V+V^\top(C^i-I)V$ |
| Negative sampling | $\rho$ fresh uniform unknowns per positive per epoch | logit shift $\ln(s_+/s_-)$; project $\rho=2$ |
| BPR | $-\sum\ln\sigma(\hat x_{ui}-\hat x_{uj})+\lambda\lVert\Theta\rVert^2$ | smooth per-user AUC; within-user only |
| Laplacian penalty | $\operatorname{tr}(U^\top LU)=\tfrac12\sum S_{ij}\lVert\mathbf u_i-\mathbf u_j\rVert^2$ | similar entities → similar factors |
| SCMFDD | $\lVert A-UV^\top\rVert^2+\mu(\ldots)+\lambda(\operatorname{tr}U^\top L_rU+\operatorname{tr}V^\top L_dV)$ | AMAN; Sylvester per half-step |
| Nuclear norm | $\lVert X\rVert_*=\sum\sigma_i = \min_{UV^\top=X}\tfrac12(\lVert U\rVert_F^2+\lVert V\rVert_F^2)$ | convex envelope of rank on $\lVert X\rVert_2\le1$ |
| SVT operator | $\mathcal D_\tau(Y)=U(\Sigma-\tau I)_+V^\top$ | prox of $\tau\lVert\cdot\rVert_*$ |
| CCS iteration | $X=\mathcal D_\tau(Y)$; $Y\mathrel{+}=\delta\mathcal P_\Omega(M-X)$ | $0<\delta<2$; heuristic $1.2/p$ |
| DRRS | complete $\begin{pmatrix}S_r&A\\A^\top&S_d\end{pmatrix}$, $\Omega$ = non-zeros | zeros of $A$ unobserved |
| BNNR | $\lVert X\rVert_*+\tfrac\alpha2\lVert\mathcal P_\Omega(W-T)\rVert^2$, $X=W$, $0\le W\le1$ | noise-tolerant, bounded, ADMM |
| IMC | $A\approx XWY^\top$ | inductive: new entities via features |
| Bilinear decoder | $\mathbf h_i^\top W\mathbf h_j = (L^\top\mathbf h_i)^\top(R^\top\mathbf h_j)$ | learned factorisation, encoder-computed factors |

**Fdataset 5-fold:** MV-HGAT 0.939 / 0.488 · SCMFDD 0.893 / 0.495 · DRRS 0.879 / 0.389 · MBiRW 0.883 / 0.311 · NIMCGCN 0.838 / 0.096 (AUC / AUPR).

**Key ideas in one breath.** Drug repositioning is implicit-feedback recommendation; unknown pairs are unlabelled, so they must be soft negatives. The low-rank assumption says a few mechanisms explain the indication matrix; MF fits it with ALS or SGD, and its L2 penalty is secretly a nuclear norm, the tightest convex proxy for rank, whose proximal operator is singular value thresholding. Side information enters through Laplacian penalties (SCMFDD), similarity blocks (DRRS), or features (IMC, GNNs) — and only side information can reach a cold entity. MV-HGAT's decoder is a learned factorisation plus a neighbourhood term, trained as logistic MF with negative sampling.

---

## 21. Further resources (all links checked)

**Courses and books**
* Google for Developers, *Recommendation Systems* course (candidate generation, matrix factorisation, WALS, softmax models; short and practical) — <https://developers.google.com/machine-learning/recommendation> (free)
* Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, ch. 9 "Recommendation Systems" (CF, UV-decomposition, the Netflix challenge) — <http://infolab.stanford.edu/~ullman/mmds/ch9.pdf>; book site <http://www.mmds.org/> (free)
* Hastie, Tibshirani & Wainwright, *Statistical Learning with Sparsity*, ch. 7 "Matrix Decompositions, Approximations, and Completion" (nuclear norm, Soft-Impute, theory) — <https://hastie.su.domains/StatLearnSparsity/> (free PDF)

**Recommender-system classics**
* Koren, Bell & Volinsky, "Matrix factorization techniques for recommender systems" (*IEEE Computer*, 2009) — <https://doi.org/10.1109/MC.2009.263> (paid)
* Simon Funk, "Netflix Update: Try This at Home" (2006), the blog post that popularised SGD for MF — <https://sifter.org/~simon/journal/20061211.html> (free)
* Mnih & Salakhutdinov, "Probabilistic matrix factorization" (NeurIPS 2007) — <https://proceedings.neurips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html> (free)
* Hu, Koren & Volinsky, "Collaborative filtering for implicit feedback datasets" (ICDM 2008) — <http://yifanhu.net/PUB/cf.pdf> (free author copy; DOI 10.1109/ICDM.2008.22)
* Pan et al., "One-class collaborative filtering" (ICDM 2008) — <https://doi.org/10.1109/ICDM.2008.16> (paid)
* Koren, "Factorization meets the neighborhood" (KDD 2008), latent factors + neighbourhoods in one model — <https://doi.org/10.1145/1401890.1401944> (paid)
* Rendle, Freudenthaler, Gantner & Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback" (UAI 2009) — <https://arxiv.org/abs/1205.2618> (free)
* Cremonesi, Koren & Turrin, "Performance of recommender algorithms on top-N recommendation tasks" (RecSys 2010; PureSVD) — <https://doi.org/10.1145/1864708.1864721> (paid)
* Rendle, Krichene, Zhang & Anderson, "Neural collaborative filtering vs. matrix factorization revisited" (RecSys 2020), why a dot product is hard to beat — <https://arxiv.org/abs/2005.09683> (free)
* Rao, Yu, Ravikumar & Dhillon, "Collaborative filtering with graph information: consistency and scalable methods" (NeurIPS 2015), graph-regularised MF — <https://proceedings.neurips.cc/paper/2015/hash/f4573fc71c731d5c362f0d7860945b88-Abstract.html> (free)

**Matrix completion and the nuclear norm**
* Fazel, *Matrix Rank Minimization with Applications* (PhD thesis, Stanford 2002), the convex-envelope theorem — <https://faculty.washington.edu/mfazel/thesis-final.pdf> (free)
* Recht, Fazel & Parrilo, "Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization" (*SIAM Review*, 2010) — <https://arxiv.org/abs/0706.4138> (free)
* Candès & Recht, "Exact matrix completion via convex optimization" (2009) — <https://arxiv.org/abs/0805.4471> (free)
* Cai, Candès & Shen, "A singular value thresholding algorithm for matrix completion" (*SIAM J. Optim.*, 2010) — <https://arxiv.org/abs/0810.3286> (free)
* Mazumder, Hastie & Tibshirani, "Spectral regularization algorithms for learning large incomplete matrices" (Soft-Impute; *JMLR*, 2010) — <https://www.jmlr.org/papers/v11/mazumder10a.html> (free)
* Jain & Dhillon, "Provable inductive matrix completion" (2013) — <https://arxiv.org/abs/1306.0626> (free)
* Natarajan & Dhillon, "Inductive matrix completion for predicting gene–disease associations" (*Bioinformatics*, 2014) — <https://doi.org/10.1093/bioinformatics/btu269> (free via PMC4058925)

**The drug-repositioning methods in this chapter**
* Zhang et al., "Predicting drug-disease associations by using similarity constrained matrix factorization" (SCMFDD; *BMC Bioinformatics*, 2018) — <https://doi.org/10.1186/s12859-018-2220-4> (free, open access); code <https://github.com/xiangyue9607/SCMFDD>
* Luo et al., "Computational drug repositioning using low-rank matrix approximation and randomized algorithms" (DRRS; *Bioinformatics*, 2018) — <https://doi.org/10.1093/bioinformatics/bty013> (paid; PubMed 29365057)
* Yang et al., "Drug repositioning based on bounded nuclear norm regularization" (BNNR; *Bioinformatics*, 2019) — <https://doi.org/10.1093/bioinformatics/btz331> (free via PMC6612853); code <https://github.com/BioinformaticsCSU/BNNR>
* Li et al., "Neural inductive matrix completion with graph convolutional networks for miRNA-disease association prediction" (NIMCGCN; *Bioinformatics*, 2020) — <https://doi.org/10.1093/bioinformatics/btz965> (paid; PubMed 31904845)

---

## 22. Glossary

* **ADMM** — alternating direction method of multipliers; splits a constrained problem into simple alternating steps (used by BNNR).
* **ALS (alternating least squares)** — fix one factor matrix, solve the other exactly by ridge regression, alternate.
* **AMAN / AMAU** — "all missing as negative" / "all missing as unknown", the two naive extremes of one-class CF.
* **Bias terms** — per-user / per-item offsets $b_i$, $c_j$ added to the factor score.
* **Bilinear decoder** — score $\mathbf h_i^\top W\mathbf h_j$; a learned factorisation.
* **BPR** — Bayesian personalised ranking; pairwise loss $-\ln\sigma(\hat x_{ui}-\hat x_{uj})$.
* **Cold start** — an entity with no interactions (new user/item; here a disease with no known drug).
* **Collaborative filtering** — recommending from the interaction patterns of many users and items.
* **Confidence weight** — $c_{ij}=1+\alpha r_{ij}$ in weighted MF.
* **Convex envelope** — largest convex function below a given function on a set.
* **DistMult / RESCAL** — diagonal / full bilinear decoders.
* **DRRS** — completion of the heterogeneous block matrix by SVT with randomised SVD.
* **Eckart–Young–Mirsky theorem** — truncated SVD gives the best low-rank approximation.
* **Explicit / implicit feedback** — stated preferences (both signs) / observed behaviour (positives only).
* **Hybrid recommender** — combines collaborative and content (side-information) signals.
* **Incoherence** — singular vectors spread across coordinates; needed for completion guarantees.
* **Inductive matrix completion (IMC)** — $A\approx XWY^\top$; factors are functions of features.
* **Item-based / user-based CF** — neighbourhood methods averaging over similar items / users.
* **Latent factor** — hidden dimension shared by users and items in MF.
* **Logistic MF** — MF with a Bernoulli likelihood, trained with binary cross-entropy.
* **Low-rank assumption** — a few factors explain most of the matrix.
* **Matrix completion** — recovering unobserved entries of a low-rank matrix.
* **Missing not at random (MNAR)** — the pattern of missing entries depends on the values or on other variables (e.g. research attention).
* **Negative sampling / ratio** — training on a random sample of unknowns as negatives; $\rho$ negatives per positive per epoch.
* **NIMCGCN** — neural IMC with GCN encoders.
* **Nuclear norm** — $\lVert X\rVert_*=\sum_i\sigma_i(X)$.
* **One-class collaborative filtering (OCCF)** — CF with positive-only data.
* **PMF** — probabilistic matrix factorisation; MF as MAP estimation.
* **Positive–unlabelled (PU) learning** — learning from positives and unlabelled examples.
* **Proximal operator** — $\arg\min_X\tfrac12\lVert X-Y\rVert^2+g(X)$; for the nuclear norm, SVT.
* **Randomised SVD** — fast approximate partial SVD using random projections.
* **SCMFDD** — similarity-constrained MF with Laplacian penalties.
* **SGD** — stochastic gradient descent, here one observed cell at a time.
* **Singular value thresholding (SVT)** — $\mathcal D_\tau$: shrink singular values by $\tau$; also the CCS algorithm built on it.
* **Soft-Impute** — completion by repeated fill-in and SVT.
* **Spectral norm** — largest singular value $\lVert X\rVert_2$.
* **Sylvester equation** — $aX+Xb=q$; arises in graph-regularised ALS.
* **Transductive / inductive** — predictions only for entities seen in training / also for new entities.
* **Weighted MF (WMF)** — MF over all cells with per-cell confidence weights.
