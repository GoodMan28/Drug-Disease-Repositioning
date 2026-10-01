# Unit A2 — Linear Algebra for Machine Learning and Graphs

> Course: *Computational Drug Repositioning with Graph Neural Networks* — Track A (Foundations), Unit 2 of 20.
> Companion files: `docs/PREREQUISITES.md` (study plan), `docs/HOW_IT_WORKS.md` (project walkthrough).

---

## 0. Front matter

**Prerequisites.** Unit A1 (NumPy arrays, shapes, broadcasting, `@`). From school mathematics:
solving two linear equations, summation notation $\sum$, square roots. No prior linear algebra
course is assumed; every concept is defined before it is used.

**Estimated study time.** 26–32 hours:

| Part | Hours |
|---|---|
| §2–§4 vectors, matrix multiplication, transpose, special matrices | 6 |
| §5 rank | 2 |
| §6 eigendecomposition | 4 |
| §7 SVD and low-rank approximation | 5 |
| §8–§9 norms, trace, matrix completion | 3 |
| §10–§11 graph Laplacians and smoothness | 4 |
| §12–§13 project walkthrough, common mistakes | 2 |
| §14 exercises | 5 |

**Learning objectives.** After this unit you will be able to:

1. Compute a matrix product by hand and interpret it in four ways (dot products, column mixing,
   **row mixing**, sum of outer products); explain what $SA$, $AS_d$, $GG^\top$ and $H_r W H_d^\top$
   compute in the project.
2. Use the transpose rules, including $(AB)^\top = B^\top A^\top$, and recognise symmetric,
   diagonal, orthogonal and positive semidefinite matrices; prove that every Gram matrix is PSD.
3. Define rank, compute it for small matrices, and explain why $\operatorname{rank}(UV^\top) \le k$.
4. Compute the eigenvalues/eigenvectors of a 2 × 2 matrix by hand, state the spectral theorem
   for symmetric matrices and prove its two key facts (real eigenvalues, orthogonal eigenvectors).
5. Compute an SVD by hand for a 2 × 2 matrix, derive it from the eigendecomposition of $A^\top A$,
   and use the Eckart–Young–Mirsky theorem to quantify the error of a rank-$k$ approximation.
6. Compute Frobenius, spectral and nuclear norms, relate each to singular values, and explain why
   the nuclear norm is used as a convex surrogate for rank (and what singular value thresholding does).
7. Build the degree matrix, the Laplacian $L = D - S$ and the normalised operators
   $D^{-1/2} S D^{-1/2}$ and $I - D^{-1/2} S D^{-1/2}$; prove the quadratic-form identities and the
   spectral bounds; explain why normalisation is needed.
8. Prove $\operatorname{tr}(U^\top L U) = \tfrac12 \sum_{ij} S_{ij}\lVert u_i - u_j\rVert^2$ and
   explain why it is a smoothness penalty, including its gradient.
9. Read `methods.py::sym_norm`, `DRRS`, `SCMFDD`, `MBiRW`, `data.py::knn_kernel` and the
   bilinear decoder in `model.py::MVHGAT.forward`, and state the linear algebra behind every line.

---

## 1. Motivation: a graph *is* a matrix

Every model in this project, including the baselines, is built from a handful of matrix
operations:

| Project code | Linear algebra |
|---|---|
| `data.py::knn_kernel` → `K @ A` | row mixing: each drug gets the weighted average association profile of its neighbours |
| `methods.py::MBiRW` → `Mr @ R`, `R @ Md` | left multiplication mixes drug rows; right multiplication mixes disease columns |
| `methods.py::sym_norm` | $D^{-1/2} S D^{-1/2}$, the symmetric normalisation of a graph |
| `methods.py::SCMFDD` | $\lVert A - UV^\top\rVert_F^2 + \lambda \operatorname{tr}(U^\top L_r U) + \dots$: low rank + Laplacian smoothness |
| `methods.py::DRRS` | singular value thresholding: SVD, shrink singular values, rebuild |
| `model.py::MVHGAT.forward` | bilinear decoder $H_r W H_d^\top$ |
| `similarity.py::jaccard`, `cosine_cross` | $GG^\top$ counts shared genes; normalised inner products |
| `model.py::DenseGAT` | $a^\top[x\Vert y] = a_1^\top x + a_2^\top y$ (block multiplication) |

**A concrete example.** In the multi-view propagation head of MV-HGAT, for a drug view the
score slice is `K @ Am`, where `K` is the (593 × 593) row-normalised kNN kernel of, say,
`chem_ecfp` and `Am` the (593 × 313) matrix of visible links. Row $i$ of the result is

$$ (K A)_{i,:} = \sum_{k} K_{ik}\, A_{k,:}, $$

a weighted average of the association rows of drug $i$'s ten most similar drugs. If eight of
the ten structural neighbours of a drug treat hypertension, the drug gets a high hypertension
score. That single matrix product is the "guilt-by-association" principle of drug repositioning
written in one line — and, as Unit A2 will show, it is also one step of a random walk, one layer of
a (linear) graph convolution, and one gradient step of a Laplacian smoothness penalty. On
Fdataset, combining just one drug view and one disease view this way already reaches a test AUC
of about 0.84 in a 5-fold split (we compute this in §12).

The second big idea is **low rank**. Fdataset's $593 \times 313$ matrix has 185,609 cells but
only 1,933 ones. If it were an arbitrary pattern, no method could guess the missing ones. The
working hypothesis of DRRS, SCMFDD and our bilinear decoder is that the *underlying* matrix of
"how plausible is drug $i$ for disease $j$" is approximately **low rank**: explained by a small
number of hidden factors (mechanisms, target families, disease pathways). Section 7 makes this
precise and §12 shows that a rank-10 approximation of the training matrix already ranks hidden
test links with AUC ≈ 0.81, whereas the full-rank matrix (which simply memorises the training
links) is no better than chance.

---

## 2. Vectors, matrices and notation

### 2.1 Notation used in this unit

* Scalars: lowercase italic, $a, \lambda, \sigma$.
* Vectors: lowercase, e.g. $x \in \mathbb{R}^n$, always **column** vectors ($n \times 1$).
  $x_i$ is the $i$-th entry. $\mathbf{1}$ is the all-ones vector.
* Matrices: uppercase, $A \in \mathbb{R}^{n\times m}$ has $n$ rows and $m$ columns; $A_{ij}$ is
  the entry in row $i$, column $j$; $A_{i,:}$ is row $i$ (a row vector); $A_{:,j}$ is column $j$.
* $A^\top$: transpose. $I$ or $I_n$: identity. $\operatorname{diag}(d)$: diagonal matrix with $d$ on the diagonal.
* Project symbols: $A$ = drug × disease association matrix ($n_r \times n_d$);
  $S_r$, $S_d$ = drug/disease similarity matrices; $D$ = degree matrix; $L$ = Laplacian;
  $U, V$ = factor matrices; $H_r, H_d$ = embedding matrices (one row per drug/disease).

In NumPy, a 1-D array of length $n$ is neither a row nor a column; `x[:, None]` makes it a
column and `x[None, :]` a row (Unit A1 §4.9). `@` is matrix multiplication.

### 2.2 Vectors, dot products, lengths and angles

The **dot product** (inner product) of $x, y \in \mathbb{R}^n$ is
$$x^\top y = \sum_{i=1}^n x_i y_i .$$
The **Euclidean length** (2-norm) is $\lVert x\rVert_2 = \sqrt{x^\top x}$. The angle $\theta$
between non-zero vectors satisfies the **cosine formula**
$$\cos\theta = \frac{x^\top y}{\lVert x\rVert_2\,\lVert y\rVert_2} \in [-1, 1].$$
(The bound is the Cauchy–Schwarz inequality $|x^\top y| \le \lVert x\rVert\lVert y\rVert$.)

**Binary vectors are sets.** If $x, y \in \{0,1\}^n$ are indicator vectors of gene sets $X, Y$,
then $x^\top y = |X\cap Y|$ (each shared gene contributes $1\cdot 1$), and
$\lVert x\rVert_2^2 = |X|$. Hence
$$\cos\theta = \frac{|X\cap Y|}{\sqrt{|X|\,|Y|}}, \qquad
\text{Jaccard}(X,Y) = \frac{|X\cap Y|}{|X| + |Y| - |X\cap Y|}.$$
`similarity.py::cosine_cross` computes the first between drug and disease gene profiles (the
"gene bridge"); `jaccard` computes the second between drugs (or between diseases).

**Worked example.** Drug gene set $X = \{\text{PTGS1}, \text{PTGS2}, \text{NFKB1}\}$, disease gene set
$Y = \{\text{PTGS2}, \text{NFKB1}, \text{IL6}, \text{TNF}\}$. Over the vocabulary
(PTGS1, PTGS2, NFKB1, IL6, TNF): $x = (1,1,1,0,0)$, $y = (0,1,1,1,1)$. Then $x^\top y = 2$,
$\lVert x\rVert = \sqrt3$, $\lVert y\rVert = 2$, cosine $= 2/(2\sqrt3) = 0.577$, Jaccard $= 2/(3+4-2) = 0.4$.

### 2.3 Linear combinations and the outer product

A **linear combination** of vectors $v_1,\dots,v_k$ is $c_1 v_1 + \dots + c_k v_k$ with scalar
weights $c_i$. Nearly everything in this unit is a linear combination in disguise.

The **outer product** of $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^m$ is the $n\times m$ matrix
$x y^\top$ with entries $(xy^\top)_{ij} = x_i y_j$. Every row is a multiple of $y^\top$ and every
column a multiple of $x$: it is the simplest possible non-zero matrix (rank 1, §5). Example:
$\begin{pmatrix}1\\2\end{pmatrix}\begin{pmatrix}3 & 4 & 5\end{pmatrix} = \begin{pmatrix}3&4&5\\6&8&10\end{pmatrix}$.

---

## 3. Matrix multiplication: four ways to read the same product

### 3.1 Definition

For $A \in \mathbb{R}^{n\times p}$ and $B \in \mathbb{R}^{p\times m}$, the product
$C = AB \in \mathbb{R}^{n\times m}$ has entries
$$C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}.$$
**Shape rule:** $(n\times p)(p\times m) = (n \times m)$; the inner dimensions must agree and
disappear. Computing $C$ this way costs $n\,p\,m$ multiplications.

The formula is one fact, but it can be *read* in four ways. Each reading is the right mental
picture for a different part of the project.

### 3.2 View 1 — every entry is a dot product

$C_{ij} = A_{i,:}\, B_{:,j}$: row $i$ of $A$ dotted with column $j$ of $B$. Example from the
project: the bilinear decoder's logit for drug $i$ and disease $j$ is
$h_i^\top W h_j$, i.e. entry $(i,j)$ of $H_r W H_d^\top$.

### 3.3 View 2 — columns of $AB$ mix the columns of $A$

Column $j$ of $C$ is $A$ times column $j$ of $B$:
$$C_{:,j} = A\,B_{:,j} = \sum_k B_{kj}\, A_{:,k}.$$
So each column of the product is a linear combination of the **columns of $A$**, with weights
taken from a column of $B$.

### 3.4 View 3 — rows of $AB$ mix the rows of $B$ ("mixing rows")

Row $i$ of $C$ is row $i$ of $A$ times $B$:
$$C_{i,:} = A_{i,:}\,B = \sum_k A_{ik}\, B_{k,:}.$$
Each row of the product is a linear combination of the **rows of $B$**, with weights taken from row
$i$ of $A$. **This is the single most important picture for graphs.** If $A$ is a similarity or
adjacency matrix and $B$ holds one row of data per node, then $AB$ replaces every node's row by a
similarity-weighted combination of the rows of the nodes it is connected to — that is **message
passing**. A GCN layer $\hat{A} H W$ first mixes rows ($\hat{A} H$: aggregate neighbours' features)
and then transforms columns ($\cdot W$: a learned linear map of the feature dimensions).

### 3.5 View 4 — a sum of outer products

$$AB = \sum_{k=1}^{p} A_{:,k}\, B_{k,:} \qquad\text{(column $k$ of $A$ times row $k$ of $B$).}$$
Each term is a rank-1 outer product. A factorisation $A \approx UV^\top = \sum_{f=1}^{k} U_{:,f} V_{:,f}^\top$
is therefore a sum of $k$ rank-1 "patterns"; the SVD (§7) is the best such sum.

### 3.6 Worked example by hand: similarity times associations

Three drugs, two diseases. Drug similarity and known associations:
$$
S = \begin{pmatrix} 1 & 0.8 & 0 \\ 0.8 & 1 & 0.5 \\ 0 & 0.5 & 1\end{pmatrix},\qquad
A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1\end{pmatrix}
\quad(\text{drug 1 treats disease 1, drug 3 treats disease 2}).
$$
View 3 (row mixing), row by row:

* Row 1: $1\cdot(1,0) + 0.8\cdot(0,0) + 0\cdot(0,1) = (1, 0)$.
* Row 2: $0.8\cdot(1,0) + 1\cdot(0,0) + 0.5\cdot(0,1) = (0.8, 0.5)$.
* Row 3: $0\cdot(1,0) + 0.5\cdot(0,0) + 1\cdot(0,1) = (0, 1)$.

$$SA = \begin{pmatrix} 1 & 0 \\ 0.8 & 0.5 \\ 0 & 1\end{pmatrix}.$$
Drug 2 had *no* known indication. After one multiplication it has scores 0.8 for disease 1
(because it is very similar to drug 1) and 0.5 for disease 2 (moderately similar to drug 3).
Check entry $(2,1)$ with view 1: row 2 of $S$ dotted with column 1 of $A$ is
$0.8\cdot1 + 1\cdot0 + 0.5\cdot0 = 0.8$. ✓

Now multiply on the **right** by a disease similarity
$S_d = \begin{pmatrix}1 & 0.6\\0.6 & 1\end{pmatrix}$:
$$A S_d = \begin{pmatrix} 1 & 0.6 \\ 0 & 0 \\ 0.6 & 1\end{pmatrix}.$$
Right-multiplication mixes **columns**: drug 1 now also scores 0.6 for disease 2, because disease 2
resembles disease 1, which drug 1 treats. So:

> **$S_r A$ spreads evidence along drug similarity; $A S_d$ spreads it along disease similarity.**
> MBiRW alternates both (`Mr @ R` and `R @ Md`); MV-HGAT's propagation head computes both
> (`K @ Am` for drug views, `(K @ Am.T).T` $= A_m K^\top$ for disease views).

```python
import numpy as np
S = np.array([[1, 0.8, 0], [0.8, 1, 0.5], [0, 0.5, 1]])
A = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
Sd = np.array([[1, 0.6], [0.6, 1]])
print(S @ A)
print(A @ Sd)
# view 3 explicitly: row 2 of S@A is a weighted sum of the rows of A
print(sum(S[1, k] * A[k] for k in range(3)))
# view 4 explicitly: sum of outer products
print(sum(np.outer(S[:, k], A[k]) for k in range(3)))
```

Output:

```text
[[1.  0. ]
 [0.8 0.5]
 [0.  1. ]]
[[1.  0.6]
 [0.  0. ]
 [0.6 1. ]]
[0.8 0.5]
[[1.  0. ]
 [0.8 0.5]
 [0.  1. ]]
```

### 3.7 Algebraic properties

* **Associative:** $(AB)C = A(BC)$. **Distributive:** $A(B + C) = AB + AC$.
* **Not commutative:** in general $AB \ne BA$ (often only one of them is even defined). Above,
  $S_r A$ is defined but $A S_r$ is not.
* **Identity:** $I_n A = A = A I_m$.
* **Scalars commute:** $A(cB) = c(AB)$.

**Associativity and cost.** The bilinear decoder computes $H_r W H_d^\top$ with
$H_r \in \mathbb{R}^{593\times 192}$, $W \in \mathbb{R}^{192\times192}$, $H_d \in \mathbb{R}^{313 \times 192}$
(192 = 64 × 3: the input embedding and the two layer outputs, concatenated by jumping knowledge). Left to right: $(H_r W)$ costs $593\cdot192\cdot192 \approx 21.9$M
multiplications, then $(\cdot)H_d^\top$ costs $593\cdot192\cdot313 \approx 35.6$M. The order matters
much more for chains like $x^\top (S^{10} y)$: computing $S^{10}$ first costs ten $n^3$ products,
whereas $S(S(\cdots(Sy)))$ costs ten $n^2$ matrix–vector products. **Multiply into vectors, not
matrices into matrices, whenever you can.**

### 3.8 Matrices as linear maps

A matrix $A\in\mathbb{R}^{n\times m}$ defines a function $x \mapsto Ax$ from $\mathbb{R}^m$ to
$\mathbb{R}^n$ that is **linear**: $A(\alpha x + \beta y) = \alpha Ax + \beta Ay$. Conversely, every
linear map between these spaces is a matrix (its columns are the images of the basis vectors
$e_1, \dots, e_m$, since $Ae_j = A_{:,j}$). Matrix multiplication is **composition** of maps:
$(AB)x = A(Bx)$ — "first apply $B$, then $A$". This is why stacking linear layers without a
non-linearity collapses to a single matrix, and why GNN layers need activations.

### 3.9 Products count paths

If $M$ is a 0/1 matrix of edges from set $\mathcal{X}$ to set $\mathcal{Y}$ and $N$ from
$\mathcal{Y}$ to $\mathcal{Z}$, then $(MN)_{xz} = \sum_y M_{xy}N_{yz}$ counts the paths
$x \to y \to z$. Examples in the project:

* $G_r G_r^\top$ (drugs × genes times genes × drugs): number of genes shared by two drugs →
  numerator of the `gene_r` Jaccard view.
* $G_r G_d^\top$: number of drug → gene → disease paths → numerator of the gene-bridge cosine.
* $A A^\top$ (drugs × drugs): number of diseases two drugs share; $A^\top A$: number of drugs two
  diseases share — "co-indication" similarities.
* For a square adjacency matrix, $(M^t)_{ij}$ counts walks of length $t$ from $i$ to $j$. A
  2-layer GNN uses $\hat A(\hat A H)$, so information from 2-hop neighbours arrives — this is the
  answer to Unit C3's "why can a 2-layer GNN see 2-hop neighbours?".

```python
import numpy as np
Gr = np.array([[1, 1, 1, 0, 0],      # drug 0: PTGS1 PTGS2 NFKB1
               [0, 1, 0, 0, 1],      # drug 1: PTGS2 TNF
               [0, 0, 0, 1, 0]])     # drug 2: IL6
Gd = np.array([[0, 1, 1, 1, 1],      # disease 0: PTGS2 NFKB1 IL6 TNF
               [1, 0, 0, 0, 0]])     # disease 1: PTGS1
print("shared genes drug-drug:\n", Gr @ Gr.T)
print("drug->gene->disease paths:\n", Gr @ Gd.T)
cos = (Gr @ Gd.T) / np.sqrt(Gr.sum(1))[:, None] / np.sqrt(Gd.sum(1))[None, :]
print("cosine (gene bridge):\n", cos.round(3))
M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])   # path graph 0-1-2
print("walks of length 2:\n", M @ M)
```

Output:

```text
shared genes drug-drug:
 [[3 1 0]
 [1 2 0]
 [0 0 1]]
drug->gene->disease paths:
 [[2 1]
 [2 0]
 [1 0]]
cosine (gene bridge):
 [[0.577 0.577]
 [0.707 0.   ]
 [0.5   0.   ]]
walks of length 2:
 [[1 0 1]
 [0 2 0]
 [1 0 1]]
```

Check by hand: drug 0 and disease 0 share PTGS2 and NFKB1 → 2 paths; cosine
$= 2/\sqrt{3\cdot4} = 0.577$, as in §2.2. In the path graph $0 - 1 - 2$ there is exactly one walk of
length 2 from 0 to 2 (via 1), and two from 1 back to 1 (via 0 and via 2).

---

## 4. The transpose and special matrices

### 4.1 Transpose

The **transpose** $A^\top$ of $A \in \mathbb{R}^{n \times m}$ is the $m\times n$ matrix with
$(A^\top)_{ij} = A_{ji}$: rows become columns. In the project, `A.T` is the disease × drug matrix;
the relation `assoc>disease` uses `A.T` because messages flow from drugs to diseases.

Rules (all easy to check entry-wise): $(A^\top)^\top = A$; $(A + B)^\top = A^\top + B^\top$;
$(cA)^\top = cA^\top$; and the important one:

**Theorem.** $(AB)^\top = B^\top A^\top$.

*Proof.* $((AB)^\top)_{ij} = (AB)_{ji} = \sum_k A_{jk}B_{ki} = \sum_k (B^\top)_{ik}(A^\top)_{kj} = (B^\top A^\top)_{ij}$. $\square$

(The order reverses, like taking off socks and shoes.) Consequences:

* The dot product is $x^\top y = y^\top x$ (a $1\times1$ matrix equals its transpose).
* In the propagation head, `(K @ Am.T).T` $= (K A^\top)^\top = A K^\top$: a disease-view kernel applied
  on the right, with $K^\top$ because `K` is row-normalised (not symmetric).
* $x^\top W y$ is a **bilinear form**: linear in $x$ for fixed $y$ and vice versa. Its matrix form
  $H_r W H_d^\top$ gives all pairs at once (view 1).

### 4.2 Block matrices

Matrices can be partitioned into blocks, and products computed block-wise as if the blocks were
numbers (provided the shapes match). Two project uses:

1. **DRRS and LAGCN** build the heterogeneous matrix
   $$T = \begin{pmatrix} S_r & A \\ A^\top & S_d \end{pmatrix} \in \mathbb{R}^{(n_r + n_d)\times(n_r+n_d)}$$
   with `np.block([[Sr, A_train], [A_train.T, Sd]])`. Because $S_r, S_d$ are symmetric,
   $T^\top = \begin{pmatrix} S_r^\top & (A^\top)^\top \\ A^\top & S_d^\top\end{pmatrix} = T$: $T$ is symmetric.
2. **GAT's attention split.** GAT scores an edge with $a^\top [x \Vert y]$, where $[x\Vert y]$ stacks two
   vectors. Splitting $a = \begin{pmatrix}a_1\\a_2\end{pmatrix}$ into blocks gives
   $a^\top[x\Vert y] = a_1^\top x + a_2^\top y$. That is why `model.py::DenseGAT` stores two vectors
   `a_dst` and `a_src` and adds the two scores with broadcasting (`[:, None, :]` + `[None, :, :]`)
   instead of concatenating every pair — the same number, at a fraction of the memory.

### 4.3 Diagonal matrices: scaling rows and columns

For $d \in \mathbb{R}^n$, $\operatorname{diag}(d)$ has $d_i$ on the diagonal and zeros elsewhere.

**Fact.** $\operatorname{diag}(d)\,M$ multiplies **row** $i$ of $M$ by $d_i$; $M \operatorname{diag}(d)$
multiplies **column** $j$ of $M$ by $d_j$.

*Proof.* $(\operatorname{diag}(d)M)_{ij} = \sum_k d_i\,[k = i]\,M_{kj} = d_i M_{ij}$; similarly on the right. $\square$

So $\operatorname{diag}(d)\,S\,\operatorname{diag}(d)$ has entries $d_i S_{ij} d_j$. NumPy never needs to
form the diagonal matrix: `S * d[:, None] * d[None, :]` gives the same result with $O(n^2)$ work
instead of the $O(n^3)$ of two dense products. This is exactly `methods.py::sym_norm`.

### 4.4 Symmetric matrices

$M$ is **symmetric** if $M = M^\top$. Similarity matrices are symmetric (sim$(i,j)$ = sim$(j,i)$), as
are $GG^\top$, $A^\top A$, $D^{-1/2}SD^{-1/2}$ and the Laplacian. A kNN mask built with
`symmetric=False` is **not** symmetric (i choosing j does not mean j chooses i) — that is why
`knn_mask` symmetrises with `M |= M.T` for the GNN graphs, and why `knn_kernel` (row-normalised)
is not symmetric either. Symmetric matrices have the best spectral theory (§6.3).

### 4.5 Orthogonal matrices

$Q \in \mathbb{R}^{n\times n}$ is **orthogonal** if $Q^\top Q = I$, i.e. its columns are
**orthonormal** (unit length, mutually perpendicular). Then $Q^{-1} = Q^\top$ and
$\lVert Qx\rVert_2 = \lVert x\rVert_2$ (proof: $\lVert Qx\rVert^2 = x^\top Q^\top Q x = x^\top x$).
Orthogonal matrices are rotations/reflections: they change direction but never length. A
rectangular $U \in \mathbb{R}^{n\times k}$ with $U^\top U = I_k$ has orthonormal columns (but
$UU^\top \ne I_n$ unless $k = n$; $UU^\top$ is the projection onto the span of those columns).

### 4.6 Positive semidefinite (PSD) matrices

A symmetric $M$ is **positive semidefinite** ($M \succeq 0$) if $x^\top M x \ge 0$ for every $x$,
and **positive definite** if $x^\top M x > 0$ for every $x \ne 0$.

**Theorem (Gram matrices are PSD).** For any $G \in \mathbb{R}^{n\times p}$, $M = GG^\top$ is PSD.

*Proof.* $x^\top GG^\top x = (G^\top x)^\top (G^\top x) = \lVert G^\top x\rVert_2^2 \ge 0$. $\square$

So the shared-gene count matrix $G_rG_r^\top$ is PSD, as is any matrix of inner products of
feature vectors (a **kernel matrix**). PSD similarity matrices are "geometrically honest": they
can be realised as inner products of points in some space. It is a known (but non-obvious) result that
the Tanimoto/Jaccard similarity of binary vectors is also PSD (proved by Bouchard, Jousselme &
Doré, 2013; the "Tanimoto kernel" of Ralaivola et al., 2005, relies on the same property). Not every similarity is: the Wang semantic
similarity with a *maximum over term pairs*, as in `wang_similarity`, need not be. The code below
checks the six project views numerically (the smallest eigenvalue must be $\ge 0$ for PSD; §6
explains why):

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
d = D.load("F")
for name in d.drug_view_names:
    S = D.fill_missing(d.drug_view(name))
    print(f"{name:10s} symmetric={np.allclose(S, S.T)}  min eigenvalue={np.linalg.eigvalsh(S).min():+.4f}")
for name in d.disease_view_names:
    S = D.fill_missing(d.disease_view(name))
    print(f"{name:10s} symmetric={np.allclose(S, S.T)}  min eigenvalue={np.linalg.eigvalsh(S).min():+.4f}")
```

Output:

```text
chem_cdk   symmetric=True  min eigenvalue=-0.0000
chem_ecfp  symmetric=True  min eigenvalue=-0.0000
gene_r     symmetric=True  min eigenvalue=-0.0000
pheno_mim  symmetric=True  min eigenvalue=+0.0001
sem_mondo  symmetric=True  min eigenvalue=-1.8893
gene_d     symmetric=True  min eigenvalue=-0.0000
```

Values like `-0.0000` are rounding error around zero (PSD up to floating point). `sem_mondo` has a
clearly negative eigenvalue: it is a perfectly usable *similarity*, but not a valid *kernel*. That
matters for methods that require kernels (e.g. kernel regression), not for kNN graphs.

### 4.7 Stochastic matrices

A non-negative matrix whose rows each sum to 1 is **row-stochastic** ($P\mathbf{1} = \mathbf{1}$).
It is the transition matrix of a random walk: $P_{ij}$ is the probability of stepping from $i$ to
$j$. `knn_kernel` returns a row-stochastic matrix (for nodes that have neighbours), so `K @ A` is a
*weighted average* of neighbours' rows rather than a weighted sum — the scale of the result does
not depend on how many neighbours a node has.

---

## 5. Linear independence, span and rank

### 5.1 Definitions

* Vectors $v_1, \dots, v_k$ are **linearly independent** if the only solution of
  $c_1 v_1 + \dots + c_k v_k = 0$ is $c_1 = \dots = c_k = 0$; otherwise one of them is a linear
  combination of the others (it is **redundant**).
* The **span** of a set of vectors is the set of all their linear combinations.
* A **basis** of a subspace is a linearly independent set that spans it; all bases have the same
  size, the **dimension**.
* The **column space** of $A$ is the span of its columns ($\{Ax : x\}$); the **row space** is the
  span of its rows.
* The **rank** of $A$, $\operatorname{rank}(A)$, is the dimension of its column space: the maximum
  number of linearly independent columns.

**Theorem (row rank = column rank).** The maximum number of independent rows equals the maximum
number of independent columns.

*Proof.* Let $r$ be the column rank and let the columns of $C \in \mathbb{R}^{n\times r}$ be a basis
of the column space. Every column of $A$ is a combination of the columns of $C$, so $A = CR$ for some
$R \in \mathbb{R}^{r\times m}$ (column $j$ of $R$ holds the coefficients of column $j$ of $A$). By view 3,
every row of $A = CR$ is a combination of the $r$ rows of $R$, so the row rank is at most $r$.
Applying the same argument to $A^\top$ shows column rank ≤ row rank. $\square$

Consequences used constantly:

1. $\operatorname{rank}(A) \le \min(n, m)$. A matrix with $\operatorname{rank}(A) = \min(n,m)$ is **full rank**.
2. $\operatorname{rank}(AB) \le \min(\operatorname{rank} A, \operatorname{rank} B)$ — columns of $AB$
   lie in the column space of $A$ (view 2), rows of $AB$ in the row space of $B$ (view 3).
3. An outer product $xy^\top$ ($x, y \ne 0$) has rank exactly 1.
4. **If $U \in \mathbb{R}^{n\times k}$ and $V \in \mathbb{R}^{m \times k}$, then $\operatorname{rank}(UV^\top) \le k$.**
   A $k$-factor model can only ever produce rank-$k$ score matrices, however large $n$ and $m$ are.
5. The bilinear decoder's logits $H_r W H_d^\top$ have rank at most 192 (the embedding width), so
   this term alone is a low-rank model; MV-HGAT adds the propagation terms on top.

### 5.2 Worked example (by hand)

$$M = \begin{pmatrix}1 & 2 & 3\\ 2 & 4 & 6\\ 1 & 0 & 1\end{pmatrix}.$$
Row 2 is twice row 1, so the rows are dependent. Rows 1 and 3 are independent (neither is a multiple
of the other). So $\operatorname{rank}(M) = 2$. Column view: column 3 = column 1 + column 2
($3 = 1+2$, $6 = 2 + 4$, $1 = 1 + 0$), and columns 1, 2 are independent — again rank 2.

**Rank in the project data.** Fdataset's $A$ has 313 columns, but only 246 *distinct* ones
(many rare diseases are treated by exactly the same single drug — Unit A1, Exercise 14). Identical
columns are dependent, and further dependencies exist, so the rank is 238, not 313.

### 5.3 Degrees of freedom: why low rank means "few parameters"

A general $n \times m$ matrix has $nm$ free entries. A rank-$k$ matrix can be written $UV^\top$ with
$k(n + m)$ numbers (fewer, $k(n+m) - k^2$, after removing redundancy). For Fdataset with $k = 10$:
$10\cdot(593 + 313) = 9{,}060$ numbers instead of 185,609. With only 1,933 known ones, there is no
hope of estimating 185,609 free values, but estimating a few thousand is plausible. **That is the
whole reason low-rank models can fill in missing entries.**

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
M = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]])
print("rank M =", np.linalg.matrix_rank(M))
rng = np.random.default_rng(0)
U, V = rng.normal(size=(593, 10)), rng.normal(size=(313, 10))
print("rank(U V^T) =", np.linalg.matrix_rank(U @ V.T))
A = D.load("F").A
print("Fdataset A: shape", A.shape, "rank", np.linalg.matrix_rank(A),
      "distinct columns", len(np.unique(A, axis=1).T))
```

Output:

```text
rank M = 2
rank(U V^T) = 10
Fdataset A: shape (593, 313) rank 238 distinct columns 246
```

`np.linalg.matrix_rank` counts singular values above a small tolerance (§7): in floating point,
"exactly dependent" becomes "dependent up to rounding error".

---

## 6. Eigenvalues, eigenvectors and the spectral theorem

### 6.1 Definition and intuition

A non-zero vector $v$ is an **eigenvector** of a square matrix $M$ with **eigenvalue** $\lambda$ if
$$Mv = \lambda v.$$
Most vectors change direction when multiplied by $M$; eigenvectors are the special directions that
are only stretched (by $\lambda$; flipped if $\lambda < 0$). Since $Mv = \lambda v \iff (M - \lambda I)v = 0$
has a non-zero solution exactly when $M - \lambda I$ is singular, the eigenvalues are the roots of the
**characteristic polynomial** $\det(M - \lambda I) = 0$ (degree $n$, so $n$ roots counted with
multiplicity, possibly complex).

### 6.2 Worked example (by hand)

$M = \begin{pmatrix} 2 & 1 \\ 1 & 2\end{pmatrix}$ (think: two similar drugs).
$\det(M - \lambda I) = (2-\lambda)^2 - 1 = 0 \Rightarrow 2 - \lambda = \pm 1 \Rightarrow \lambda_1 = 3,\ \lambda_2 = 1.$

* $\lambda_1 = 3$: $(M - 3I)v = \begin{pmatrix}-1 & 1\\ 1 & -1\end{pmatrix} v = 0 \Rightarrow v_1 = \tfrac{1}{\sqrt2}(1, 1)^\top$.
* $\lambda_2 = 1$: $(M - I)v = \begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix} v = 0 \Rightarrow v_2 = \tfrac{1}{\sqrt2}(1, -1)^\top$.

Check: $Mv_1 = \tfrac1{\sqrt2}(3, 3)^\top = 3v_1$ ✓. The two eigenvectors are perpendicular
($v_1^\top v_2 = 0$). Interpretation: the "agreement" direction $(1,1)$ (both drugs score the same)
is amplified 3×; the "disagreement" direction $(1,-1)$ is left unchanged. Repeatedly multiplying by
$M$ makes any vector look more and more like $(1,1)$ — **smoothing**.

Contrast: the rotation $R = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$ has
$\det(R - \lambda I) = \lambda^2 + 1$, roots $\pm i$: no real direction is preserved by a 90° rotation.
Non-symmetric matrices can have complex eigenvalues; symmetric ones cannot, as we now prove.

### 6.3 The spectral theorem for symmetric matrices

**Theorem (spectral theorem).** If $M \in \mathbb{R}^{n\times n}$ is symmetric, then all its eigenvalues
are real, and there is an orthonormal basis of $\mathbb{R}^n$ made of eigenvectors of $M$. Equivalently
$$M = Q\Lambda Q^\top = \sum_{i=1}^n \lambda_i\, q_i q_i^\top,$$
with $Q = [q_1, \dots, q_n]$ orthogonal and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$.

We prove the two facts that do most of the work; the existence of a full orthonormal basis when
eigenvalues repeat is proved by induction in any linear algebra text (e.g. Strang, *Introduction to
Linear Algebra*, ch. 6; Axler, *Linear Algebra Done Right*, ch. 7).

*(a) Eigenvalues are real.* Let $Mv = \lambda v$ with $v \in \mathbb{C}^n$, $v \ne 0$, and write
$v^*$ for the conjugate transpose. Then $v^* M v = \lambda\, v^* v = \lambda \lVert v\rVert^2$. The
number $v^*Mv$ is real, because its complex conjugate is $(v^*Mv)^* = v^* M^* v = v^* M v$ (using
$M^* = M^\top = M$ for a real symmetric matrix). Since $\lVert v\rVert^2 > 0$ is real, $\lambda$ is real. $\square$

*(b) Eigenvectors of different eigenvalues are orthogonal.* Let $Mu = \lambda u$, $Mv = \mu v$,
$\lambda \ne \mu$. Then
$\lambda\, u^\top v = (Mu)^\top v = u^\top M^\top v = u^\top M v = \mu\, u^\top v$, so
$(\lambda - \mu)\,u^\top v = 0$ and $u^\top v = 0$. $\square$

**Reading $M = \sum_i \lambda_i q_i q_i^\top$.** A symmetric matrix is a weighted sum of rank-1
projections onto perpendicular directions. Multiplying a vector $x$ by $M$: expand
$x = \sum_i c_i q_i$ with $c_i = q_i^\top x$; then $Mx = \sum_i \lambda_i c_i q_i$. Each
component along an eigenvector is scaled by its eigenvalue, independently of the others.

### 6.4 Consequences

**Powers.** $M^t = Q\Lambda^t Q^\top$, so $M^t x = \sum_i \lambda_i^t c_i q_i$. If one eigenvalue is
largest in absolute value, its component dominates as $t$ grows: $M^t x$ lines up with the top
eigenvector. For a normalised graph operator (§10) the top eigenvector is essentially "the same value
everywhere, adjusted for degree"; repeated propagation therefore washes out the differences between
nodes. This is the linear-algebra core of **over-smoothing** in deep GNNs (Unit C3), and the reason the
project uses only 2 GNN layers, kNN-sparsified graphs and jumping-knowledge concatenation.

**Power iteration.** The same observation gives an algorithm for the top eigenvector: start from a
random $x$, repeat $x \leftarrow Mx / \lVert Mx\rVert$. The error shrinks like $|\lambda_2/\lambda_1|^t$.

**Rayleigh quotient.** For symmetric $M$ with eigenvalues $\lambda_{\min} = \lambda_n \le \dots \le \lambda_1 = \lambda_{\max}$,
$$\lambda_{\min} \le R(x) = \frac{x^\top M x}{x^\top x} \le \lambda_{\max}\qquad(x \ne 0),$$
with equality at the corresponding eigenvectors.
*Proof.* With $x = \sum c_i q_i$: $x^\top M x = \sum_i \lambda_i c_i^2$ and $x^\top x = \sum_i c_i^2$
(orthonormality), so $R(x)$ is a weighted average of the $\lambda_i$ with weights $c_i^2/\sum c_j^2$. $\square$

**PSD ⇔ non-negative eigenvalues.** If all $\lambda_i \ge 0$ then $x^\top M x = \sum \lambda_i c_i^2 \ge 0$.
Conversely if $M \succeq 0$ then $\lambda_i = q_i^\top M q_i \ge 0$. (That is why §4.6 tested PSD via the
smallest eigenvalue.)

**Trace and determinant.** $\operatorname{tr}(M) = \sum_i \lambda_i$ and $\det(M) = \prod_i \lambda_i$
(for any square matrix, counting complex eigenvalues). Check on the example: $\operatorname{tr} = 4 = 3 + 1$, $\det = 3 = 3\cdot1$.

**Numerics.** Use `np.linalg.eigh` (or `eigvalsh`) for symmetric matrices: it is faster, more
accurate, returns real eigenvalues in **ascending** order and orthonormal eigenvectors (as columns).
`np.linalg.eig` is for general matrices and may return complex numbers. Eigenvectors are defined only
up to sign (and, for repeated eigenvalues, up to rotation within the eigenspace), so do not be surprised
if NumPy returns $-v$.

```python
import numpy as np
M = np.array([[2.0, 1.0], [1.0, 2.0]])
lam, Q = np.linalg.eigh(M)                      # ascending eigenvalues, eigenvectors in columns
print(lam)
print(Q)
print(np.allclose(Q @ np.diag(lam) @ Q.T, M), np.allclose(Q.T @ Q, np.eye(2)))
print(np.linalg.eig(np.array([[0.0, -1.0], [1.0, 0.0]]))[0])   # rotation: complex eigenvalues

# power iteration: converges to the top eigenvector (1,1)/sqrt(2) up to sign
x = np.array([1.0, 0.0])
for t in range(1, 6):
    x = M @ x
    x /= np.linalg.norm(x)
    print(t, x.round(4), "Rayleigh quotient", round(x @ M @ x, 4))
```

Output:

```text
[1. 3.]
[[-0.70710678  0.70710678]
 [ 0.70710678  0.70710678]]
True True
[0.+1.j 0.-1.j]
1 [0.8944 0.4472] Rayleigh quotient 2.8
2 [0.7809 0.6247] Rayleigh quotient 2.9756
3 [0.7328 0.6805] Rayleigh quotient 2.9973
4 [0.7158 0.6983] Rayleigh quotient 2.9997
5 [0.71   0.7042] Rayleigh quotient 3.0
```

The error in the direction shrinks by $\lambda_2/\lambda_1 = 1/3$ each step, and the Rayleigh quotient
approaches $\lambda_{\max} = 3$ even faster (its error shrinks by $(1/3)^2$ per step).

---

## 7. The singular value decomposition (SVD) and low-rank approximation

### 7.1 Why another decomposition?

Eigendecomposition needs a square matrix, and a nice one (symmetric) to be well behaved. The
association matrix $A$ is $593\times313$ — not even square. The **SVD** works for *every* matrix and
is the single most useful matrix factorisation in data science.

### 7.2 The theorem

**Theorem (SVD).** Every $A \in \mathbb{R}^{n \times m}$ of rank $r$ can be written
$$A = U\Sigma V^\top = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top,$$
where $U = [u_1,\dots,u_n] \in \mathbb{R}^{n\times n}$ and $V = [v_1, \dots, v_m]\in\mathbb{R}^{m\times m}$ are
orthogonal, and $\Sigma \in \mathbb{R}^{n\times m}$ is zero except for the **singular values**
$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ on its diagonal. The $u_i$ are the **left singular
vectors**, the $v_i$ the **right singular vectors**. Keeping only the first $r$ columns gives the
**compact SVD** $A = U_r\Sigma_rV_r^\top$.

*Construction (proof sketch).* $A^\top A \in \mathbb{R}^{m\times m}$ is symmetric and PSD (a Gram matrix),
so by the spectral theorem $A^\top A = V\Lambda V^\top$ with orthonormal $v_i$ and $\lambda_i \ge 0$. Set
$\sigma_i = \sqrt{\lambda_i}$, and for $\sigma_i > 0$ define $u_i = Av_i / \sigma_i$. These are orthonormal:
$$u_i^\top u_j = \frac{v_i^\top A^\top A v_j}{\sigma_i\sigma_j} = \frac{\lambda_j\, v_i^\top v_j}{\sigma_i\sigma_j} = \delta_{ij}.$$
For $\sigma_i = 0$, $\lVert Av_i\rVert^2 = v_i^\top A^\top A v_i = \lambda_i = 0$, so $Av_i = 0$. Hence
$Av_i = \sigma_i u_i$ for all $i$, i.e. $AV = U\Sigma$ (after completing the $u_i$ to an orthonormal basis),
and multiplying by $V^\top$ on the right gives $A = U\Sigma V^\top$. $\square$

By the same construction, $AA^\top = U\Sigma\Sigma^\top U^\top$: **the $u_i$ are eigenvectors of $AA^\top$, the
$v_i$ eigenvectors of $A^\top A$, and $\sigma_i^2$ their common non-zero eigenvalues.** The number of
non-zero singular values is the rank. For a symmetric PSD matrix the SVD and the eigendecomposition
coincide; for a symmetric matrix with negative eigenvalues, $\sigma_i = |\lambda_i|$.

**Geometry.** $Ax = U\Sigma V^\top x$: rotate/reflect ($V^\top$), stretch axis $i$ by $\sigma_i$ ($\Sigma$),
rotate/reflect again ($U$). The unit sphere is mapped to an ellipsoid whose semi-axes are
$\sigma_i u_i$. $\sigma_1$ is the largest stretch any unit vector can experience.

**Interpretation for drug–disease data.** $A = \sum_i \sigma_i u_i v_i^\top$ decomposes the association
matrix into rank-1 "patterns". Pattern $i$ says: drugs with large $|u_i|$ entries tend to treat
diseases with large $|v_i|$ entries (with signs indicating which group goes with which). The top
patterns capture the broad, repeated structure (classes of drugs used for classes of diseases); the tail
captures idiosyncratic single links and noise.

### 7.3 Worked example (by hand)

$$A = \begin{pmatrix} 3 & 0 \\ 4 & 5\end{pmatrix}.$$
**Step 1.** $A^\top A = \begin{pmatrix}3 & 4\\0 & 5\end{pmatrix}\begin{pmatrix}3 & 0\\4 & 5\end{pmatrix} = \begin{pmatrix}25 & 20\\ 20 & 25\end{pmatrix}.$

**Step 2.** Eigenvalues: $(25 - \lambda)^2 = 400 \Rightarrow \lambda = 45$ or $5$. Eigenvectors (as in §6.2):
$v_1 = \tfrac1{\sqrt2}(1, 1)^\top$, $v_2 = \tfrac{1}{\sqrt2}(1, -1)^\top$.

**Step 3.** $\sigma_1 = \sqrt{45} = 3\sqrt5 \approx 6.708$, $\sigma_2 = \sqrt5 \approx 2.236$.

**Step 4.** $u_1 = Av_1/\sigma_1 = \tfrac{1}{\sqrt2}(3, 9)^\top/(3\sqrt5) = \tfrac{1}{\sqrt{10}}(1, 3)^\top$;
$u_2 = Av_2/\sigma_2 = \tfrac{1}{\sqrt2}(3, -1)^\top/\sqrt5 = \tfrac{1}{\sqrt{10}}(3, -1)^\top$.
Check $u_1^\top u_2 = (3 - 3)/10 = 0$ ✓.

**Step 5 — rebuild.** Since $\sigma_1/(\sqrt{10}\sqrt2) = 3\sqrt5/\sqrt{20} = 3/2$ and
$\sigma_2/(\sqrt{10}\sqrt2) = 1/2$:
$$\sigma_1u_1v_1^\top = \tfrac32\begin{pmatrix}1 & 1\\3 & 3\end{pmatrix} = \begin{pmatrix}1.5 & 1.5\\4.5 & 4.5\end{pmatrix},\quad
\sigma_2u_2v_2^\top = \tfrac12\begin{pmatrix}3 & -3\\-1 & 1\end{pmatrix} = \begin{pmatrix}1.5 & -1.5\\-0.5 & 0.5\end{pmatrix},$$
and the sum is $\begin{pmatrix}3 & 0\\4 & 5\end{pmatrix} = A$ ✓. The first term alone is the **best rank-1
approximation** of $A$; its error is $\begin{pmatrix}1.5 & -1.5\\-0.5 & 0.5\end{pmatrix}$, whose Frobenius norm
is $\sqrt{2.25 + 2.25 + 0.25 + 0.25} = \sqrt5 = \sigma_2$ — exactly as the next theorem predicts.

```python
import numpy as np
A = np.array([[3.0, 0.0], [4.0, 5.0]])
U, s, Vt = np.linalg.svd(A)
print("singular values:", s, " (3*sqrt5, sqrt5) =", 3 * np.sqrt(5), np.sqrt(5))
print("U =\n", U.round(4), "\nV^T =\n", Vt.round(4))        # columns may differ from ours by a sign
print("rebuild ok:", np.allclose(U @ np.diag(s) @ Vt, A))
A1 = s[0] * np.outer(U[:, 0], Vt[0])
print("best rank-1:\n", A1.round(4))
print("error (Frobenius):", np.linalg.norm(A - A1), "= sigma_2")
print("eig(A^T A):", np.linalg.eigvalsh(A.T @ A), "= s^2 =", s**2)
```

Output:

```text
singular values: [6.70820393 2.23606798]  (3*sqrt5, sqrt5) = 6.708203932499369 2.23606797749979
U =
 [[-0.3162 -0.9487]
 [-0.9487  0.3162]] 
V^T =
 [[-0.7071 -0.7071]
 [-0.7071  0.7071]]
rebuild ok: True
best rank-1:
 [[1.5 1.5]
 [4.5 4.5]]
error (Frobenius): 2.23606797749979 = sigma_2
eig(A^T A): [ 5. 45.] = s^2 = [45.  5.]
```

### 7.4 Low-rank approximation: the Eckart–Young–Mirsky theorem

Define the **truncated SVD** $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$ (keep the top $k$ patterns).

**Theorem (Eckart–Young 1936; Mirsky 1960).** For every matrix $B$ with $\operatorname{rank}(B) \le k$,
$$\lVert A - B\rVert_2 \ \ge\ \lVert A - A_k\rVert_2 = \sigma_{k+1},\qquad
\lVert A - B\rVert_F \ \ge\ \lVert A - A_k\rVert_F = \sqrt{\textstyle\sum_{i > k}\sigma_i^2}.$$
So the truncated SVD is the **best** rank-$k$ approximation in both the spectral and the Frobenius
norm (both norms are defined in §9).

*Proof for the spectral norm.* First, $A - A_k = \sum_{i>k}\sigma_iu_iv_i^\top$ is itself an SVD whose
largest singular value is $\sigma_{k+1}$, so $\lVert A - A_k\rVert_2 = \sigma_{k+1}$. Now let
$\operatorname{rank}(B) \le k$. Its null space $\{x : Bx = 0\}$ has dimension at least $m - k$. The span of
$v_1,\dots,v_{k+1}$ has dimension $k+1$. Two subspaces of $\mathbb{R}^m$ whose dimensions add up to more
than $m$ intersect in a non-zero vector, so there is a unit vector $w = \sum_{i=1}^{k+1} c_i v_i$ with
$Bw = 0$ and $\sum c_i^2 = 1$. Then
$$\lVert (A - B)w\rVert^2 = \lVert Aw\rVert^2 = \Big\lVert \sum_{i=1}^{k+1} \sigma_i c_i u_i\Big\rVert^2 = \sum_{i=1}^{k+1}\sigma_i^2 c_i^2 \ \ge\ \sigma_{k+1}^2,$$
so $\lVert A - B\rVert_2 \ge \sigma_{k+1}$. $\square$
(The Frobenius statement has a similar but longer proof; see Strang's *Linear Algebra and Learning
from Data*, §I.9, or the MIT 18.065 lecture on Eckart–Young.)

**Energy captured.** Because $\lVert A\rVert_F^2 = \sum_i \sigma_i^2$ (§9), the fraction of the "energy"
kept by $A_k$ is $\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$. This is the standard way to choose $k$
(and is exactly the "explained variance" of PCA when the columns are centred).

### 7.5 Low rank in the project: what the data actually show

The cell below (i) prints the singular-value spectrum of Fdataset's $A$, and (ii) uses the truncated SVD
of the **training** matrix of fold 0 (5-fold split of `kfold_splits`) as a recommender: score each test
cell by $A_k$ and compute AUC/AUPR on that fold's test positives and negatives.

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import kfold_splits, metrics

A = D.load("F").A
s = np.linalg.svd(A, compute_uv=False)
energy = np.cumsum(s**2) / np.sum(s**2)
print("top singular values:", s[:6].round(2))
print("energy kept by k = 10, 50, 100, 200:", energy[[9, 49, 99, 199]].round(3))

f, test_pos, test_neg = next(kfold_splits(A, 5, seed=0))
A_tr = A.copy().ravel(); A_tr[test_pos] = 0; A_tr = A_tr.reshape(A.shape)
idx = np.concatenate([test_pos, test_neg])
y = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]
U, s_tr, Vt = np.linalg.svd(A_tr, full_matrices=False)
for k in [1, 5, 10, 20, 50, 100, 313]:
    A_k = (U[:, :k] * s_tr[:k]) @ Vt[:k]             # U_k diag(s_k) V_k^T via broadcasting
    m = metrics(y, A_k.ravel()[idx])
    print(f"rank {k:3d}: AUC {m['AUC']:.3f}  AUPR {m['AUPR']:.3f}")
```

Output:

```text
top singular values: [12.7  11.13  9.27  8.03  7.98  7.93]
energy kept by k = 10, 50, 100, 200: [0.382 0.757 0.905 0.993]
rank   1: AUC 0.642  AUPR 0.075
rank   5: AUC 0.768  AUPR 0.241
rank  10: AUC 0.810  AUPR 0.286
rank  20: AUC 0.797  AUPR 0.293
rank  50: AUC 0.750  AUPR 0.342
rank 100: AUC 0.683  AUPR 0.230
rank 313: AUC 0.482  AUPR 0.084
```

Read these numbers carefully — they contain the whole story of low-rank matrix completion:

1. **The observed matrix is not strongly low rank.** The top 10 singular values carry only about 38% of
   the energy, and it takes about 100 to reach 90%. A sparse 0/1 matrix with many rare diseases (97
   diseases in Fdataset have exactly one known drug) has a long tail of small singular values.
2. **Yet a low-rank approximation generalises.** With $k = 10$ the hidden test links are ranked with
   AUC ≈ 0.81 and AUPR ≈ 0.29 (random AUPR ≈ 0.01). The rank-10 model cannot memorise the training
   matrix, so it is forced to explain it with shared patterns, and those patterns put mass on the
   missing cells that fit them.
3. **Full rank memorises.** At $k = 313$, $A_k = A_{\text{train}}$ exactly: every test cell scores 0, so the
   test positives are indistinguishable from the negatives and the AUC is at chance level (0.48 here;
   the tiny non-zero test scores are floating-point rounding noise of order $10^{-15}$, and ranking noise
   is meaningless). In between, $k$ trades fitting the training data against generalising — the
   bias–variance trade-off, controlled here by rank. Note also that the two metrics need not agree on
   the best $k$: AUC peaks around $k = 10$, AUPR (which rewards the very top of the ranking) around $k = 50$.

So the low-rank hypothesis concerns the *underlying* matrix of indication propensities; the observed
matrix is that low-rank signal, heavily **subsampled** (most true indications are unknown) and corrupted
by noise. Matrix-completion methods (DRRS), factorisation methods (SCMFDD) and bilinear decoders all
bet on this structure, and add similarity information (§10–§11) to help where the links alone are too
sparse.

### 7.6 Computing SVDs in practice

* `np.linalg.svd(A, full_matrices=False)` returns $U_{n\times p}$, $s$ (a 1-D array, descending),
  $V^\top_{p\times m}$ with $p = \min(n,m)$. Cost $O(nm\min(n,m))$.
* `np.linalg.svd(A, compute_uv=False)` returns only singular values (cheaper).
* Rebuilding $U_k\operatorname{diag}(s_k)V_k^\top$: write `(U[:, :k] * s[:k]) @ Vt[:k]` — broadcasting
  multiplies column $i$ of $U$ by $s_i$ (= right-multiplying by a diagonal matrix, §4.3). DRRS writes
  `(U * S) @ V.T` for the same reason.
* **Randomised SVD** (`torch.svd_lowrank`, `sklearn.utils.extmath.randomized_svd`; Halko, Martinsson &
  Tropp 2011) computes only the top $q$ singular triplets: multiply $A$ by a random $m \times q$ matrix to
  capture its dominant column space, orthonormalise, and do a small exact SVD in that subspace. Cost
  $O(nmq)$ instead of $O(nm\min(n,m))$; accurate when the spectrum decays. DRRS calls it with `q=200`
  inside a 200-iteration loop, where a full SVD each time would be wasteful.

---

## 8. The trace

The **trace** of a square matrix is the sum of its diagonal entries:
$\operatorname{tr}(M) = \sum_i M_{ii}$. It looks humble, but it is the tool that turns sums over
matrix entries into compact algebra.

**Properties.**

1. Linear: $\operatorname{tr}(A + B) = \operatorname{tr}A + \operatorname{tr}B$, $\operatorname{tr}(cA) = c\operatorname{tr}A$; and $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$.
2. **Cyclic:** for $A \in \mathbb{R}^{n\times m}$, $B\in\mathbb{R}^{m\times n}$, $\operatorname{tr}(AB) = \operatorname{tr}(BA)$
   (even though $AB$ is $n\times n$ and $BA$ is $m \times m$).
   *Proof.* $\operatorname{tr}(AB) = \sum_i\sum_k A_{ik}B_{ki} = \sum_k\sum_i B_{ki}A_{ik} = \operatorname{tr}(BA)$. $\square$
   Hence $\operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA)$ — cyclic shifts only, not arbitrary reorderings.
3. **Frobenius inner product:** $\operatorname{tr}(A^\top B) = \sum_{i,j} A_{ij}B_{ij} =: \langle A, B\rangle_F$,
   the dot product of the two matrices viewed as long vectors. In particular
   $\operatorname{tr}(A^\top A) = \sum_{ij} A_{ij}^2$.
4. **Sum of eigenvalues:** $\operatorname{tr}(M) = \sum_i\lambda_i$. For symmetric $M = Q\Lambda Q^\top$:
   $\operatorname{tr}(Q\Lambda Q^\top) = \operatorname{tr}(\Lambda Q^\top Q) = \operatorname{tr}\Lambda$ by cyclicity.
5. A quadratic form is a trace: $x^\top M x = \operatorname{tr}(x^\top M x) = \operatorname{tr}(M x x^\top)$.
6. $\operatorname{tr}(U^\top M U) = \sum_{f=1}^{k} U_{:,f}^\top M\, U_{:,f}$: the trace of $U^\top M U$
   sums the quadratic forms of the **columns** of $U$ (its diagonal entries are exactly those forms).

Property 6 is what makes $\operatorname{tr}(U^\top L U)$ in SCMFDD meaningful (§11). Property 3 suggests
a cheaper implementation: $\operatorname{tr}(U^\top (LU)) = \sum_{ij} U_{ij}(LU)_{ij}$, i.e.
`(U * (L @ U)).sum()`, which avoids forming the $k\times k$ matrix $U^\top L U$.

```python
import numpy as np
rng = np.random.default_rng(0)
A, B = rng.normal(size=(4, 3)), rng.normal(size=(3, 4))
print(np.isclose(np.trace(A @ B), np.trace(B @ A)))                 # cyclic
print(np.isclose(np.trace(A.T @ A), (A**2).sum()))                  # Frobenius
M = rng.normal(size=(5, 5)); M = M + M.T
print(np.isclose(np.trace(M), np.linalg.eigvalsh(M).sum()))         # sum of eigenvalues
U = rng.normal(size=(5, 2))
print(np.isclose(np.trace(U.T @ M @ U), (U * (M @ U)).sum()),        # cheaper formula
      np.isclose(np.trace(U.T @ M @ U), sum(U[:, f] @ M @ U[:, f] for f in range(2))))
```

Output:

```text
True
True
True
True True
```

---

## 9. Norms, the nuclear norm and matrix completion

### 9.1 Vector norms

A **norm** measures size: $\lVert x\rVert \ge 0$ with equality only for $x = 0$; $\lVert cx\rVert = |c|\lVert x\rVert$;
$\lVert x + y\rVert \le \lVert x\rVert + \lVert y\rVert$ (triangle inequality). The common ones:
$\lVert x\rVert_2 = \sqrt{\sum x_i^2}$ (Euclidean), $\lVert x\rVert_1 = \sum |x_i|$,
$\lVert x\rVert_\infty = \max|x_i|$. The "$\ell_0$ norm" $\lVert x\rVert_0$ = number of non-zeros is not a
norm (it fails homogeneity) but appears in sparsity problems.

### 9.2 Three matrix norms

| Norm | Definition | In singular values | Interpretation |
|---|---|---|---|
| **Frobenius** $\lVert A\rVert_F$ | $\sqrt{\sum_{ij}A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}$ | $\sqrt{\sum_i \sigma_i^2}$ | the matrix as a long vector; least-squares error |
| **Spectral** $\lVert A\rVert_2$ | $\max_{\lVert x\rVert_2 = 1}\lVert Ax\rVert_2$ | $\sigma_1$ | largest stretch; operator norm |
| **Nuclear** $\lVert A\rVert_*$ | — | $\sum_i \sigma_i$ | convex surrogate of rank |

*Proof of the singular-value formulas.* Frobenius: $\operatorname{tr}(A^\top A) = \operatorname{tr}(V\Sigma^\top U^\top U\Sigma V^\top)
= \operatorname{tr}(\Sigma^\top\Sigma\, V^\top V) = \sum_i\sigma_i^2$. Spectral: for a unit $x$ write
$c = V^\top x$ (also a unit vector, as $V$ is orthogonal); then $\lVert Ax\rVert^2 = \lVert U\Sigma c\rVert^2 = \sum_i \sigma_i^2c_i^2 \le \sigma_1^2$,
with equality at $x = v_1$. $\square$

All three are **unitarily invariant** ($\lVert QAR\rVert = \lVert A\rVert$ for orthogonal $Q$, $R$) because they
depend only on the singular values. For a matrix of rank $r$,
$$\sigma_1 = \lVert A\rVert_2 \;\le\; \lVert A\rVert_F \;\le\; \lVert A\rVert_* \;\le\; \sqrt{r}\,\lVert A\rVert_F, \qquad \lVert A\rVert_F \le \sqrt r\, \lVert A\rVert_2 .$$
(The middle inequalities compare the $\ell_\infty$, $\ell_2$ and $\ell_1$ norms of the vector of singular
values; the right-hand ones are Cauchy–Schwarz on $r$ entries.)

**Worked example.** For $A = \begin{pmatrix}3 & 0\\4 & 5\end{pmatrix}$ (§7.3), $\sigma = (3\sqrt5, \sqrt5)$:
$\lVert A\rVert_F = \sqrt{9 + 0 + 16 + 25} = \sqrt{50} \approx 7.071$ and also $\sqrt{45 + 5} = \sqrt{50}$ ✓;
$\lVert A\rVert_2 = 3\sqrt5 \approx 6.708$; $\lVert A\rVert_* = 4\sqrt5 \approx 8.944$; and
$6.708 \le 7.071 \le 8.944 \le \sqrt2\cdot7.071 = 10$ ✓.

**For a 0/1 matrix**, $\lVert A\rVert_F^2$ is simply the number of ones: Fdataset has
$\lVert A\rVert_F = \sqrt{1933} \approx 43.97$.

**Where they appear in the project.**

* SCMFDD's data-fit term $\lVert A - UV^\top\rVert_F^2 = \sum_{ij}(A_{ij} - u_i^\top v_j)^2$ is a
  squared Frobenius norm (note that it treats *every* unknown cell as a 0 to be fitted — the
  "unknown ≠ negative" problem of Unit B4), and its regulariser $\mu(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)$
  is the matrix version of weight decay.
* DRRS sets its threshold relative to the spectral norm:
  `tau = self.tau_rel * torch.linalg.matrix_norm(T, ord=2) * N / 10` (`ord=2` = largest singular value).
* The nuclear norm is the objective that singular value thresholding minimises (below).

A remarkable link between the two (Srebro, Rennie & Jaakkola 2005; Recht, Fazel & Parrilo 2010):
$$\lVert X\rVert_* = \min_{U, V:\ UV^\top = X} \tfrac12\left(\lVert U\rVert_F^2 + \lVert V\rVert_F^2\right).$$
So penalising the Frobenius norms of the factors, as SCMFDD and every weight-decayed factorisation
model does, is secretly a nuclear-norm (low-rank-encouraging) penalty on the product.

### 9.3 Why the nuclear norm stands in for rank

$\operatorname{rank}(X)$ is the number of non-zero singular values — the $\ell_0$ "norm" of the vector
$\sigma(X)$. $\lVert X\rVert_*$ is the $\ell_1$ norm of the same vector. In ordinary sparse regression,
$\ell_1$ (the lasso) replaces the intractable $\ell_0$ because it is the tightest convex function below
$\ell_0$ on the unit box. The matrix analogue (Fazel 2002): on the set $\{X : \lVert X\rVert_2 \le 1\}$, the
nuclear norm is the **convex envelope** of the rank (the largest convex function that is everywhere
$\le$ rank). Minimising rank is NP-hard in general; minimising the nuclear norm is a convex problem.

**Matrix completion.** Let $\Omega$ be the set of observed cells and $P_\Omega(X)$ keep the entries in
$\Omega$ and zero the rest. The ideal problem "find the lowest-rank matrix agreeing with the observed
entries" is relaxed to
$$\min_X\ \lVert X\rVert_* \quad\text{subject to}\quad P_\Omega(X) = P_\Omega(M).$$
Candès & Recht (2009) proved that, if $M$ is low rank, its singular vectors are "incoherent" (not
concentrated on a few rows/columns) and enough entries are observed uniformly at random, this convex
program recovers $M$ **exactly** with high probability. Drug–disease data violate the assumptions
(entries are not missing at random, unknowns are mixed with true negatives), which is why DRRS treats
it as a heuristic and evaluates by cross-validation.

### 9.4 Singular value thresholding (SVT)

**Scalar warm-up: soft thresholding.** Minimise $f(x) = \tfrac12(x - y)^2 + \tau|x|$ over $x\in\mathbb{R}$
($\tau > 0$). For $x > 0$, $f'(x) = x - y + \tau = 0 \Rightarrow x = y - \tau$, valid if $y > \tau$; for $x < 0$,
$x = y + \tau$, valid if $y < -\tau$; otherwise the minimum is at the kink $x = 0$. So the minimiser is the
**soft-threshold** $\operatorname{sign}(y)\max(|y| - \tau, 0)$: shrink towards 0 by $\tau$, and kill anything
smaller than $\tau$.

**Matrix version.** For $Y = U\operatorname{diag}(\sigma)V^\top$ define
$$\mathcal{D}_\tau(Y) = U\operatorname{diag}\big(\max(\sigma_i - \tau, 0)\big)V^\top .$$
**Theorem (Cai, Candès & Shen 2010).** $\mathcal{D}_\tau(Y) = \arg\min_X\ \tfrac12\lVert X - Y\rVert_F^2 + \tau\lVert X\rVert_*$.

*Why (sketch).* Both terms are unitarily invariant, and (by von Neumann's trace inequality) for fixed
singular values $\lVert X - Y\rVert_F$ is smallest when $X$ shares $Y$'s singular vectors. The problem then
separates into one scalar soft-thresholding problem per singular value (with $\sigma_i \ge 0$). $\square$

So $\mathcal{D}_\tau$ is "the closest matrix to $Y$ that is pushed towards low rank": small singular values
(noise) are set to zero, large ones (signal) are shrunk by $\tau$. The SVT algorithm alternates

$$X^{(t)} = \mathcal{D}_\tau\big(Y^{(t-1)}\big), \qquad Y^{(t)} = Y^{(t-1)} + \delta\, P_\Omega\big(M - X^{(t)}\big),$$

starting from $Y^{(0)} = 0$. $Y$ accumulates the residual on the observed entries (it is a dual
variable; the iteration is Uzawa's method / dual gradient ascent), and $X$ is its low-rank shrinkage.
It converges to the solution of $\min \tau\lVert X\rVert_* + \tfrac12\lVert X\rVert_F^2$ s.t.
$P_\Omega(X) = P_\Omega(M)$, which approaches the nuclear-norm problem as $\tau$ grows. Cai, Candès &
Shen use $\tau = 5n$ for an $n\times n$ matrix (our demo uses $5\sqrt{nm}$ for a rectangular one) and step $\delta = 1.2/p$ ($p$ = fraction observed) — exactly the
`delta = 1.2 * N * N / Om.sum()` of `methods.py::DRRS`.

**Demonstration.** Hide half the entries of a random rank-2 matrix and recover them:

```python
import numpy as np
rng = np.random.default_rng(0)
n, m, r = 40, 30, 2
M = rng.normal(size=(n, r)) @ rng.normal(size=(r, m))        # ground truth, rank 2
Om = rng.random((n, m)) < 0.5                                 # observe ~50% of the cells

def svt(Y, tau):
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    return (U * s) @ Vt, int((s > 0).sum())

tau, delta = 5 * np.sqrt(n * m), 1.2 / Om.mean()
Y = np.zeros_like(M)
for it in range(1, 501):
    X, rank = svt(Y, tau)
    Y += delta * Om * (M - X)
    if it in (1, 10, 50, 100, 500):
        err = np.linalg.norm((X - M)[~Om]) / np.linalg.norm(M[~Om])
        print(f"iter {it:3d}: rank {rank}, relative error on HIDDEN cells {err:.4f}")

# the prox property, checked numerically: D_tau(Y) beats random perturbations of itself
f = lambda X, Y, t: 0.5 * np.linalg.norm(X - Y)**2 + t * np.linalg.norm(X, "nuc")
Y0 = rng.normal(size=(6, 5)); X0, _ = svt(Y0, 1.0)
print(all(f(X0, Y0, 1.0) <= f(X0 + 0.01 * rng.normal(size=X0.shape), Y0, 1.0) for _ in range(200)))
```

Output:

```text
iter   1: rank 0, relative error on HIDDEN cells 1.0000
iter  10: rank 2, relative error on HIDDEN cells 0.3005
iter  50: rank 3, relative error on HIDDEN cells 0.0438
iter 100: rank 3, relative error on HIDDEN cells 0.0229
iter 500: rank 3, relative error on HIDDEN cells 0.0003
True
```

From the observed half alone, SVT reconstructs the hidden half almost perfectly, because only
$2(40 + 30) - 4 = 136$ numbers are really unknown and 600 cells are observed. (The printed rank is 3, not 2:
SVT solves a slightly regularised version of the problem, and a third, tiny singular component survives
the threshold; it carries almost no energy, as the 0.03% error shows.) On real drug–disease data the recovery is far
from perfect, for the reasons in §7.5 and §9.3, but the mechanism is the same.

---

## 10. Graphs as matrices: degree, Laplacian and normalisation

### 10.1 Similarity graphs

A weighted undirected graph on $n$ nodes is a symmetric matrix $S \in \mathbb{R}^{n\times n}$ with
$S_{ij} \ge 0$: the weight of the edge between $i$ and $j$ (0 = no edge). A similarity view *is* such a
graph (a complete one); a kNN-sparsified view `S * knn_mask(S, k)` keeps only strong edges. The
**degree** of node $i$ is $d_i = \sum_j S_{ij}$ — in matrix form $d = S\mathbf{1}$ — and the
**degree matrix** is $D = \operatorname{diag}(d)$. With self-similarity $S_{ii} = 1$ included (as in the
project's similarity matrices), $d_i \ge 1$.

### 10.2 The (unnormalised) graph Laplacian

$$L = D - S .$$
**Theorem (quadratic form).** For symmetric $S$ and any $x \in \mathbb{R}^n$,
$$x^\top L x = \frac12\sum_{i,j} S_{ij}\,(x_i - x_j)^2 .$$
*Proof.* $x^\top L x = \sum_i d_i x_i^2 - \sum_{i,j}S_{ij}x_ix_j$. Since $d_i = \sum_j S_{ij}$,
$\sum_i d_i x_i^2 = \sum_{i,j}S_{ij}x_i^2 = \tfrac12\sum_{i,j}S_{ij}x_i^2 + \tfrac12\sum_{i,j}S_{ij}x_j^2$ (the
second form uses $S_{ij} = S_{ji}$ and renames indices). Hence
$x^\top Lx = \tfrac12\sum_{i,j}S_{ij}(x_i^2 - 2x_ix_j + x_j^2) = \tfrac12\sum_{i,j}S_{ij}(x_i - x_j)^2$. $\square$

Read $x$ as a **signal on the nodes** (one number per drug, e.g. a latent factor or a score). Then
$x^\top L x$ is large when strongly connected nodes ($S_{ij}$ large) have different values: it measures
how **rough** the signal is on the graph (it is also called the *Dirichlet energy*). Consequences:

* $L$ is PSD (the sum is non-negative), so its eigenvalues are $\ge 0$.
* $L\mathbf{1} = d - d = 0$: constant signals have zero roughness; 0 is always an eigenvalue.
* $x^\top L x = 0$ iff $x$ is constant on every connected component; so the multiplicity of the
  eigenvalue 0 equals the number of connected components.
* Self-loops do not matter: $S_{ii}$ is added to both $D_{ii}$ and $S_{ii}$ and cancels in $L$; in the
  quadratic form $(x_i - x_i)^2 = 0$.
* Eigenvectors with small eigenvalues are smooth signals (by the Rayleigh quotient, §6.4); those with
  large eigenvalues oscillate across edges. This is the "graph Fourier" view used in spectral GNNs and
  spectral clustering.

### 10.3 Normalisation: $D^{-1}S$ and $D^{-1/2}SD^{-1/2}$

Using $S$ itself to propagate (`S @ X`) has two problems. A node's aggregated value is a *sum* over its
neighbours, so it grows with degree: hubs get huge values and dominate. And repeated multiplication
scales like $\lambda_{\max}(S)^t$, which explodes or vanishes. Two normalisations fix this:

* **Random-walk normalisation** $P = D^{-1}S$: row $i$ is divided by $d_i$, so $P$ is row-stochastic and
  $PX$ averages neighbours (this is what `knn_kernel` does, with the self-loop removed).
* **Symmetric normalisation**
  $$\tilde S = D^{-1/2} S D^{-1/2}, \qquad \tilde S_{ij} = \frac{S_{ij}}{\sqrt{d_i\,d_j}},$$
  which divides each edge by the geometric mean of its endpoints' degrees. It keeps the matrix
  **symmetric** (so the spectral theorem applies) and penalises an edge if *either* endpoint is a hub.
  This is `methods.py::sym_norm`, and (with self-loops added) the GCN propagation matrix of Kipf &
  Welling (2017).

The two are closely related: $P = D^{-1/2}\tilde S D^{1/2}$, so $P$ and $\tilde S$ have the **same
eigenvalues** (they are *similar* matrices).

The **normalised Laplacian** is
$$L_{\text{sym}} = I - D^{-1/2}SD^{-1/2} = D^{-1/2}\,L\,D^{-1/2}.$$

**Theorem.** For $S$ symmetric with non-negative entries and all $d_i > 0$:
1. $x^\top L_{\text{sym}} x = \frac12\sum_{i,j}S_{ij}\left(\frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}}\right)^2$;
2. the eigenvalues of $L_{\text{sym}}$ lie in $[0, 2]$, and those of $\tilde S$ in $[-1, 1]$;
3. $\tilde S$ has eigenvalue exactly 1 with eigenvector $D^{1/2}\mathbf{1} = (\sqrt{d_1},\dots,\sqrt{d_n})$.

*Proof.* (1) Put $y = D^{-1/2}x$. Then $x^\top L_{\text{sym}}x = y^\top D^{1/2}D^{-1/2}LD^{-1/2}D^{1/2}y = y^\top Ly$,
and apply §10.2 with $y_i = x_i/\sqrt{d_i}$.
(2) Non-negativity follows from (1). For the upper bound use $(a - b)^2 \le 2a^2 + 2b^2$ and $S_{ij}\ge0$:
$x^\top L_{\text{sym}}x \le \tfrac12\sum_{i,j}S_{ij}\left(\tfrac{2x_i^2}{d_i} + \tfrac{2x_j^2}{d_j}\right) = \sum_i\tfrac{x_i^2}{d_i}\sum_jS_{ij} + \sum_j\tfrac{x_j^2}{d_j}\sum_i S_{ij} = 2\lVert x\rVert^2$.
By the Rayleigh quotient all eigenvalues of $L_{\text{sym}}$ are in $[0,2]$; since $\tilde S = I - L_{\text{sym}}$, its
eigenvalues are $1 - [0,2] = [-1, 1]$.
(3) $\tilde S D^{1/2}\mathbf{1} = D^{-1/2}S\mathbf{1} = D^{-1/2}d = D^{1/2}\mathbf{1}$. $\square$

So multiplying by $\tilde S$ never blows up (spectral radius 1), which is why GCN-style propagation can be
stacked. The eigenvalue $-1$ occurs only for graphs with a bipartite connected component; adding
self-loops (the project's similarity matrices have $S_{ii} = 1$, GCN adds $I$) pushes the bottom of the
spectrum away from $-1$, which makes propagation less oscillatory. (Wu et al. 2019, "Simplifying graph
convolutional networks", analyse this "renormalisation trick" spectrally.)

### 10.4 Worked example (by hand): the path graph

Unweighted path $1 - 2 - 3$: $S = \begin{pmatrix}0&1&0\\1&0&1\\0&1&0\end{pmatrix}$, $d = (1, 2, 1)$.
$$L = D - S = \begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix},\qquad
\tilde S = \begin{pmatrix}0&\tfrac1{\sqrt2}&0\\\tfrac1{\sqrt2}&0&\tfrac1{\sqrt2}\\0&\tfrac1{\sqrt2}&0\end{pmatrix}.$$
Eigenvalues of $L$: $\det(L - \lambda I) = (1-\lambda)\big[(2-\lambda)(1-\lambda) - 1\big] - (1 - \lambda) = (1-\lambda)(\lambda^2 - 3\lambda) = -\lambda(\lambda-1)(\lambda-3)$, so $\{0, 1, 3\}$ with eigenvectors
$(1,1,1)$ (constant; Rayleigh quotient 0), $(1,0,-1)$ (a smooth "ramp"; quotient 1) and $(1,-2,1)$
(sign changes across both edges; quotient 3) — the smoother the signal, the smaller its eigenvalue. For $\tilde S$, $\det(\tilde S - \lambda I) = -\lambda^3 + \lambda$, so eigenvalues
$\{-1, 0, 1\}$ and $L_{\text{sym}}$ has $\{0, 1, 2\}$ — the extreme value 2 because a path is bipartite.
The eigenvector for $\tilde S$'s eigenvalue 1 is $\propto D^{1/2}\mathbf 1 = (1, \sqrt2, 1)$ ✓.

**A similarity matrix with self-loops** (the $S$ of §3.6): $S = \begin{pmatrix}1&0.8&0\\0.8&1&0.5\\0&0.5&1\end{pmatrix}$,
$d = (1.8, 2.3, 1.5)$. Then $\tilde S_{11} = 1/1.8 = 0.556$, $\tilde S_{12} = 0.8/\sqrt{1.8\cdot2.3} = 0.8/2.035 = 0.393$,
$\tilde S_{22} = 1/2.3 = 0.435$, $\tilde S_{23} = 0.5/\sqrt{2.3\cdot1.5} = 0.5/1.857 = 0.269$, $\tilde S_{33} = 1/1.5 = 0.667$,
$\tilde S_{13} = 0$.

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo.methods import sym_norm          # importing methods also imports torch (CPU is fine)

S_path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
d = S_path.sum(1)
L = np.diag(d) - S_path
print("eig L     :", np.linalg.eigvalsh(L).round(4))
print("eig S~    :", np.linalg.eigvalsh(sym_norm(S_path)).round(4))
print("eig L_sym :", np.linalg.eigvalsh(np.eye(3) - sym_norm(S_path)).round(4))
x = np.array([1.0, 1.0, 5.0])
print("x^T L x =", x @ L @ x, "= 1/2 sum S_ij (x_i - x_j)^2 =",
      0.5 * sum(S_path[i, j] * (x[i] - x[j])**2 for i in range(3) for j in range(3)))

S = np.array([[1, 0.8, 0], [0.8, 1, 0.5], [0, 0.5, 1]])
print(sym_norm(S).round(3))
Dm = np.diag(1 / np.sqrt(S.sum(1)))
print("sym_norm == D^-1/2 S D^-1/2:", np.allclose(sym_norm(S), Dm @ S @ Dm))
P = S / S.sum(1, keepdims=True)                      # random-walk normalisation
print("same eigenvalues as D^-1 S:", np.allclose(np.sort(np.linalg.eigvals(P).real),
                                                 np.linalg.eigvalsh(sym_norm(S))))
```

Output:

```text
eig L     : [0. 1. 3.]
eig S~    : [-1.  0.  1.]
eig L_sym : [0. 1. 2.]
x^T L x = 16.0 = 1/2 sum S_ij (x_i - x_j)^2 = 16.0
[[0.556 0.393 0.   ]
 [0.393 0.435 0.269]
 [0.    0.269 0.667]]
sym_norm == D^-1/2 S D^-1/2: True
same eigenvalues as D^-1 S: True
```

### 10.5 The real graphs

The project's kNN graphs are large, sparse and have hubs. The spectrum of the symmetric-normalised
`chem_cdk` kNN graph (with self-loops, as built for NIMCGCN) confirms the theory:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from scipy.sparse.csgraph import connected_components
from drepo import data as D
from drepo.methods import sym_norm

d = D.load("F")
S = D.fill_missing(d.drug_view("chem_cdk"))
M = D.knn_mask(S, 10)
W = S * M                                   # kNN-sparsified weighted graph (self-loops kept)
deg = W.sum(1)
print("degree min/median/max:", deg.min().round(2), np.median(deg).round(2), deg.max().round(2))
ev = np.linalg.eigvalsh(sym_norm(W))
print("eigenvalues of D^-1/2 W D^-1/2 in [%.3f, %.3f]" % (ev.min(), ev.max()))
L_sym = np.eye(len(W)) - sym_norm(W)
print("smallest eigenvalues of L_sym:", np.linalg.eigvalsh(L_sym)[:3].round(4))
print("connected components:", connected_components(M)[0])
top = np.linalg.eigh(sym_norm(W))[1][:, -1]
print("top eigenvector proportional to sqrt(degree):",
      np.allclose(np.abs(top), np.sqrt(deg) / np.linalg.norm(np.sqrt(deg))))
```

Output:

```text
degree min/median/max: 1.85 6.74 43.13
eigenvalues of D^-1/2 W D^-1/2 in [-0.263, 1.000]
smallest eigenvalues of L_sym: [-0.      0.0189  0.0386]
connected components: 1
top eigenvector proportional to sqrt(degree): True
```

Exactly one zero eigenvalue of $L_{\text{sym}}$ (one connected component), the top eigenvalue of
$\tilde S$ equal to 1 with eigenvector $\propto\sqrt{d}$, and the bottom of the spectrum far from $-1$
thanks to the self-loops.

---

## 11. The smoothness penalty $\operatorname{tr}(U^\top L U)$

### 11.1 From one signal to an embedding matrix

Let $U \in \mathbb{R}^{n\times k}$ hold a $k$-dimensional latent vector $u_i^\top$ (row $i$) for each of $n$
drugs. By trace property 6 and §10.2 applied to each column $U_{:,f}$:
$$\operatorname{tr}(U^\top L U) = \sum_{f=1}^k U_{:,f}^\top L\, U_{:,f} = \sum_{f=1}^k \frac12\sum_{i,j}S_{ij}(U_{if} - U_{jf})^2 = \frac12\sum_{i,j} S_{ij}\,\lVert u_i - u_j\rVert_2^2 .$$
And with the normalised Laplacian (what SCMFDD's code uses):
$$\operatorname{tr}(U^\top L_{\text{sym}} U) = \frac12\sum_{i,j}S_{ij}\left\lVert \frac{u_i}{\sqrt{d_i}} - \frac{u_j}{\sqrt{d_j}}\right\rVert_2^2 .$$

**What it penalises.** The latent vectors of *similar* drugs being *far apart*. Each pair contributes its
squared distance, weighted by its similarity: two drugs with similarity 0.9 and very different latent
factors cost a lot; two unrelated drugs (similarity 0) can be anywhere. Minimising
$$\underbrace{\lVert A - UV^\top\rVert_F^2}_{\text{fit known links}} + \underbrace{\mu(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)}_{\text{small factors}} + \underbrace{\lambda\big(\operatorname{tr}(U^\top L_rU) + \operatorname{tr}(V^\top L_dV)\big)}_{\text{similar} \Rightarrow \text{close}}$$
(SCMFDD, Zhang et al. 2018) therefore trades off fitting the observed links against keeping the
factor matrices **smooth on the similarity graphs**. A drug with *no* known links receives no signal from
the first term, but the third term pulls its factors towards those of its similar drugs, so it inherits
their predicted indications: guilt by association, enforced as a regulariser. The normalised version
makes the penalty insensitive to how many neighbours a drug has (hubs would otherwise dominate the sum).

**Worked example (by hand).** Path graph $1 - 2 - 3$ (unweighted), one-dimensional factors:
* $u = (1, 1, 5)$: $u^\top L u = (1-1)^2 + (1-5)^2 = 16$ (summing once per edge equals
  $\tfrac12\sum_{i,j}$ over ordered pairs).
* $u = (1, 1, 1)$: 0 — perfectly smooth.
* $u = (1, 3, 5)$: $(1-3)^2 + (3-5)^2 = 8$ — a smooth ramp is cheaper than a jump of the same total size,
  because squares punish big differences.
* Two-dimensional $U = \begin{pmatrix}1&0\\1&0\\5&2\end{pmatrix}$:
  $\operatorname{tr}(U^\top L U) = \lVert u_1 - u_2\rVert^2 + \lVert u_2 - u_3\rVert^2 = 0 + (16 + 4) = 20$.

### 11.2 The gradient: smoothing is propagation

**Fact.** For symmetric $L$, $\nabla_U \operatorname{tr}(U^\top L U) = 2LU$.

*Proof.* Perturb $U$ by $E$: $\operatorname{tr}((U+E)^\top L(U+E)) - \operatorname{tr}(U^\top LU) = \operatorname{tr}(E^\top LU) + \operatorname{tr}(U^\top LE) + \operatorname{tr}(E^\top LE)$.
By transpose invariance and symmetry, $\operatorname{tr}(U^\top LE) = \operatorname{tr}(E^\top L^\top U) = \operatorname{tr}(E^\top LU)$.
So the first-order change is $2\operatorname{tr}(E^\top LU) = \langle E, 2LU\rangle_F$, i.e. the gradient is $2LU$. $\square$

A gradient step on the penalty alone, with step size $\eta$ and weight $\lambda$, is
$$U \leftarrow U - 2\eta\lambda L_{\text{sym}}U = (1 - 2\eta\lambda)\,U + 2\eta\lambda\,\tilde S\, U,$$
a blend of each drug's own vector and the normalised average of its neighbours' vectors — **one step of
graph propagation**, the same operation as a (linear) GCN layer $\tilde S H$. This is a deep connection:
Laplacian regularisation (SCMFDD), label propagation/random walks (MBiRW) and graph convolution (GCN,
GAT) all push information along the same similarity edges; they differ in whether the propagation is
fixed or learned, and whether it acts on scores, factors or features.

```python
import numpy as np
L = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
for u in ([1, 1, 5], [1, 1, 1], [1, 3, 5]):
    u = np.array(u, dtype=float)
    print(u, "-> u^T L u =", u @ L @ u)
U = np.array([[1, 0], [1, 0], [5, 2]], dtype=float)
print("tr(U^T L U) =", np.trace(U.T @ L @ U))

# numerical gradient check of d/dU tr(U^T L U) = 2 L U
f = lambda U: np.trace(U.T @ L @ U)
G = np.zeros_like(U); h = 1e-6
for i in range(3):
    for j in range(2):
        E = np.zeros_like(U); E[i, j] = h
        G[i, j] = (f(U + E) - f(U - E)) / (2 * h)
print("numerical gradient == 2 L U:", np.allclose(G, 2 * L @ U, atol=1e-4))

# gradient descent on the penalty alone smooths U along the graph
for step in range(3):
    U = U - 0.1 * 2 * L @ U
    print(step, U[:, 0].round(3), "penalty", round(f(U), 3))
```

Output:

```text
[1. 1. 5.] -> u^T L u = 16.0
[1. 1. 1.] -> u^T L u = 0.0
[1. 3. 5.] -> u^T L u = 8.0
tr(U^T L U) = 20.0
numerical gradient == 2 L U: True
0 [1.  1.8 4.2] penalty 8.0
1 [1.16 2.12 3.72] penalty 4.352
2 [1.352 2.248 3.4  ] penalty 2.662
```

Each step pulls the outlier (5) towards its neighbour and the penalty falls; the *mean* of each column
($\mathbf{1}^\top U$) is unchanged because $\mathbf{1}^\top L = 0$ — smoothing redistributes, it does not
create or destroy "mass". Run long enough with no data term, the penalty alone would collapse all
connected drugs onto one point (the eigenvalue-0 eigenvector): that is why it is always combined with a
data-fit term and a weight $\lambda$.

---

## 12. In this project: the linear algebra in the code

Snippets are quoted from the repository; the first comment names file and function.

### 12.1 `methods.py::sym_norm` — $D^{-1/2} S D^{-1/2}$

```py
# src/drepo/methods.py :: sym_norm
def sym_norm(S):
    d = S.sum(1)                          # degrees d = S 1           (n,)
    d[d == 0] = 1                         # isolated node: avoid division by zero
    d = 1 / np.sqrt(d)                    # diagonal of D^{-1/2}      (n,)
    return S * d[:, None] * d[None, :]    # diag(d) S diag(d), i.e. S_ij / sqrt(d_i d_j)
```

Line 4 is §4.3 with broadcasting: `d[:, None]` scales rows, `d[None, :]` scales columns — $O(n^2)$
instead of two $O(n^3)$ products with diagonal matrices. The guard on line 3 leaves an isolated node's
(zero) row unchanged rather than producing NaN. The same normalisation appears in `LAGCN.embed` in
PyTorch form: `d = Hm.sum(1).clamp(min=1e-12).rsqrt()` (`rsqrt` = $1/\sqrt{\cdot}$) followed by
`Hm * d[:, None] * d[None, :]`.

Where it is used: `NIMCGCN` normalises kNN-sparsified similarity graphs for its GCNs
(`sym_norm(Sr * knn_mask(Sr, self.k))`); `MBiRW` normalises logistic-adjusted similarities;
`SCMFDD` builds its Laplacians from it.

### 12.2 `data.py::knn_kernel` and the propagation head — row mixing

`knn_kernel` returns $K = D_W^{-1}W$, the random-walk normalisation (§10.3) of the kNN-sparsified
similarity without self-loops (Unit A1 §6.3). In `MVHGATMethod.fit_predict`:

```py
# src/drepo/methods.py :: MVHGATMethod.fit_predict :: propagation
return torch.stack([K @ Am for K in self._prop["drug"]] +
                   [(K @ Am.T).T for K in self._prop["disease"]])
```

* `K @ Am` (drug view): $(KA)_{ij} = \sum_k K_{ik}A_{kj}$ = weighted share of drug $i$'s neighbours that
  treat disease $j$ — view 3 (row mixing).
* `(K @ Am.T).T` (disease view) $= A K^\top$: $(AK^\top)_{ij} = \sum_l A_{il}K_{jl}$ = weighted share of disease
  $j$'s neighbours that drug $i$ treats — column mixing.
* `torch.stack` makes a $(V, n_r, n_d)$ tensor; the model combines it as
  `(self.view_weights()[:, None, None] * P).sum(0)` $= \sum_v w_v P_v$, broadcasting one weight over each slice.

How strong is this simple linear algebra? Using one drug view and one disease view on fold 0:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import kfold_splits, metrics

d = D.load("F"); A = d.A
f, tp, tn = next(kfold_splits(A, 5, seed=0))
A_tr = A.copy().ravel(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
idx = np.concatenate([tp, tn]); y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]

Kr = D.knn_kernel(d.drug_view("chem_cdk"), 10)       # 593 x 593, rows sum to 1
Kd = D.knn_kernel(d.disease_view("pheno_mim"), 10)   # 313 x 313
print("row sums of Kr:", np.unique(Kr.sum(1).round(6)))
for name, P in [("drug view  K_r A     ", Kr @ A_tr),
                ("disease view A K_d^T ", A_tr @ Kd.T),
                ("sum of both          ", Kr @ A_tr + A_tr @ Kd.T)]:
    m = metrics(y, P.ravel()[idx])
    print(f"{name}: AUC {m['AUC']:.3f}  AUPR {m['AUPR']:.3f}")
```

Output:

```text
row sums of Kr: [1.]
drug view  K_r A     : AUC 0.731  AUPR 0.193
disease view A K_d^T : AUC 0.706  AUPR 0.160
sum of both          : AUC 0.838  AUPR 0.322
```

Two fixed matrix products, no training, already give AUC ≈ 0.84 — comparable to the rank-10 SVD of §7.5.
This is why HOW_IT_WORKS.md reports that "this simple neighbour propagation is very strong on these
benchmarks", and why MV-HGAT keeps it as an explicit, interpretable head and lets the GNN learn the rest.

### 12.3 `methods.py::MBiRW` — alternating left and right multiplication

```py
# src/drepo/methods.py :: MBiRW.fit_predict
Mr, Md = sym_norm(logistic(Sr)), sym_norm(logistic(Sd))
A0 = A_train / max(A_train.sum(), 1)
R = A0.copy()
for step in range(1, max(self.l, self.r) + 1):
    parts = []
    if step <= self.l:
        parts.append(self.alpha * Mr @ R + (1 - self.alpha) * A0)   # walk on the drug graph
    if step <= self.r:
        parts.append(self.alpha * R @ Md + (1 - self.alpha) * A0)   # walk on the disease graph
    R = sum(parts) / len(parts)
```

`Mr @ R` mixes rows (spreads scores between similar drugs), `R @ Md` mixes columns (between similar
diseases). Each update is a **random walk with restart**: with weight $\alpha$ take a propagation step,
with weight $1-\alpha$ jump back to the known links $A_0$. Because $\lVert\tilde S\rVert_2 \le 1$ (§10.3), the
iteration is stable. The "left walk" is the drug-side (left) multiplication, the "right walk" the
disease-side (right) multiplication; Unit C2 studies them in depth.

### 12.4 `methods.py::DRRS` — singular value thresholding on a block matrix

```py
# src/drepo/methods.py :: DRRS.fit_predict
T = t(np.block([[Sr, A_train], [A_train.T, Sd]]))   # (n_r+n_d) square, symmetric (§4.2)
Om = (T != 0).float()                                # "observed" = non-zero entries
N = T.shape[0]
delta = 1.2 * N * N / Om.sum()                       # step 1.2/p  (p = observed fraction)
tau = self.tau_rel * torch.linalg.matrix_norm(T, ord=2) * N / 10   # threshold ~ spectral norm
Y = torch.zeros_like(T)
X = T
for _ in range(self.iters):
    U, S, V = torch.svd_lowrank(Y if Y.abs().sum() > 0 else T, q=self.rank, niter=2)
    S = torch.clamp(S - tau, min=0)                  # soft-threshold the singular values
    X = (U * S) @ V.T                                # D_tau(Y) = U diag(S) V^T
    Y = Y + delta * Om * (T - X)                     # accumulate residual on observed cells
return X[:n_r, n_r:].cpu().numpy()                   # the drug x disease block
```

Line by line against §9.4: `np.block` builds the heterogeneous matrix; `Om` is $P_\Omega$ as a 0/1 mask
(element-wise product = keep observed cells) — note DRRS's modelling choice that **every zero**,
including unknown drug–disease pairs and zero similarities, is treated as *missing*, and every non-zero
as observed; `delta` is the Cai–Candès–Shen step $1.2/p$; `tau` scales with $\sigma_1(T)$ so that the
threshold is meaningful whatever the matrix's magnitude; `torch.svd_lowrank` is the randomised SVD of
§7.6 returning $U$, $S$ and $V$ (not $V^\top$), so $\mathcal{D}_\tau(Y) = U\operatorname{diag}(S)V^\top$ is
`(U * S) @ V.T`; on the first iteration $Y = 0$, so the code factorises $T$ instead to obtain sensible
singular vectors. The prediction is the completed off-diagonal block: drug–disease scores borrowed from
the low-rank structure *shared* with the similarity blocks.

### 12.5 `methods.py::SCMFDD` — low rank plus Laplacian smoothness

```py
# src/drepo/methods.py :: SCMFDD.fit_predict
Lr = t(np.eye(len(Sr)) - sym_norm(Sr))     # L_sym for drugs  = I - D^-1/2 S_r D^-1/2
Ld = t(np.eye(len(Sd)) - sym_norm(Sd))     # L_sym for diseases
U = nn.Parameter(0.1 * torch.randn(A.shape[0], self.k))   # n_r x k drug factors (k = 128)
V = nn.Parameter(0.1 * torch.randn(A.shape[1], self.k))   # n_d x k disease factors
...
loss = ((A - U @ V.T) ** 2).sum() + self.mu * (U.pow(2).sum() + V.pow(2).sum()) \
    + self.lam * (torch.trace(U.T @ Lr @ U) + torch.trace(V.T @ Ld @ V))
...
return (U @ V.T).detach().cpu().numpy()
```

* `((A - U @ V.T) ** 2).sum()` $= \lVert A - UV^\top\rVert_F^2$.
* `U.pow(2).sum() + V.pow(2).sum()` $= \lVert U\rVert_F^2 + \lVert V\rVert_F^2$ — a nuclear-norm-like penalty on $UV^\top$ (§9.2).
* `torch.trace(U.T @ Lr @ U)` $= \tfrac12\sum_{ij}(S_r)_{ij}\lVert u_i/\sqrt{d_i} - u_j/\sqrt{d_j}\rVert^2$ (§11).
  `(U * (Lr @ U)).sum()` would compute the same number without the $k\times k$ intermediate.
* Autograd computes the gradients, including $2\lambda L_r U$ from the trace term, and Adam updates
  $U$, $V$. The output $UV^\top$ has rank at most $k = 128$.

### 12.6 `model.py` — bilinear decoder, gate, attention as row mixing

```py
# src/drepo/model.py :: MVHGAT.forward
Hr, Hd, betas = self.encode(X, graphs, drop_rel)     # (593 x 192), (313 x 192)
logits = Hr @ self.W @ Hd.T                          # bilinear: logits_ij = h_i^T W h_j
...
logits = self.gnn_gate(*deg) * logits                # element-wise (Hadamard) product
logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
```

* `Hr @ self.W @ Hd.T` is $H_rWH_d^\top$ (view 1: every entry is $h_i^\top W h_j$). With $W = I$ it would be a
  plain dot product; a full learned $W$ lets drug dimension $a$ interact with disease dimension $b$ and need
  not be symmetric. Its rank is at most 192: the GNN term is a learned low-rank factorisation
  ($U = H_rW$, $V = H_d$).
* `gnn_gate` returns `sigmoid(...)[:, None] * sigmoid(...)[None, :]` — an **outer product** (rank-1
  matrix) of a per-drug and a per-disease gate, built by broadcasting, then multiplied element-wise with
  the logits.
* Inside `DenseGAT.forward`, `torch.einsum("dsh,shk->dhk", att, zs)` is, for each head $h$, the matrix
  product $\text{att}_{:,:,h}\,Z_{:,h,:}$ of a (destination × source) attention matrix with the source
  features: **row mixing with learned weights**. Because softmax makes each row of `att` sum to 1, it is a
  learned row-stochastic matrix — a data-dependent $D^{-1}S$.
* The attention score `(zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]`
  is the block identity $a^\top[x\Vert y] = a_1^\top x + a_2^\top y$ of §4.2.

### 12.7 `similarity.py` — Gram matrices

`jaccard` computes `inter = G @ G.T` (a PSD Gram matrix counting shared genes) and
`cosine_cross` computes `(G1 @ G2.T) / (n1 * n2.T)` with `n1 = sqrt(G1.sum(1, keepdims=True))`:
since $G$ is binary, $\lVert g_i\rVert_2 = \sqrt{\sum_k g_{ik}}$, so this is exactly the cosine of §2.2,
i.e. $D_1^{-1/2}(G_1G_2^\top)D_2^{-1/2}$ — a symmetric-normalisation of the drug→gene→disease path counts.

---

## 13. Common mistakes and misconceptions

| Mistake / misconception | Why it is wrong | Correct view |
|---|---|---|
| "`*` and `@` are the same" | `*` is element-wise (Hadamard); `@` is matrix multiplication | `S * M` masks; `S @ A` propagates |
| Writing `S @ A` when the evidence should spread over diseases | left multiplication mixes **rows** (drugs) | use `A @ S_d` (or `(K @ A.T).T`) for disease similarity |
| $(AB)^\top = A^\top B^\top$ | order reverses | $(AB)^\top = B^\top A^\top$ |
| "A kNN graph is symmetric" | i choosing j ≠ j choosing i | symmetrise (`M \|= M.T`) if an undirected graph is needed |
| Forming `np.diag(d) @ S @ np.diag(d)` | $O(n^3)$ and memory-heavy | broadcasting `S * d[:, None] * d[None, :]` |
| Using `np.linalg.eig` on symmetric matrices | slower, complex output, unordered | `np.linalg.eigh` / `eigvalsh` (ascending) |
| Comparing eigen/singular vectors across runs | defined only up to sign (or rotation for repeated values) | compare subspaces or $\lvert v\rvert$, not raw vectors |
| "`np.linalg.svd` returns $V$" | it returns $V^\top$ (`Vt`); `torch.svd_lowrank` returns $V$ | read the docs; rebuild with `U * s @ Vt` or `(U * S) @ V.T` |
| "Low rank means the observed matrix has few non-zero singular values" | the observed sparse matrix has a long tail (rank 238) | the *underlying* propensity matrix is approximately low rank |
| "More rank = better fit = better model" | full rank memorises training links (AUC ≈ 0.5 on test) | choose $k$ by validation |
| "The nuclear norm is the rank" | it is the sum of singular values — a convex surrogate | rank = # non-zero $\sigma$; nuclear = $\sum\sigma$ |
| "Similarity matrices are always PSD" | `sem_mondo` has a negative eigenvalue | check with `eigvalsh` if a method needs a kernel |
| Normalising with a zero-degree node | division by zero → NaN/inf | guard (`d[d == 0] = 1`) |
| "$L = D - S$ changes with self-loops" | self-loops cancel in $L$ (not in $D^{-1/2}SD^{-1/2}$!) | know which operator your code uses |
| Forgetting the ½ in $\tfrac12\sum_{i,j}$ | ordered pairs count each edge twice | $\tfrac12\sum_{i,j}$ = $\sum_{\text{edges}}$ |
| "The Laplacian penalty alone gives good factors" | its minimiser is constant on components | always pair with a data-fit term |
| Computing $\operatorname{tr}(U^\top L U)$ by forming a big matrix | wasteful | `(U * (L @ U)).sum()` |

---

## 14. Exercises

Graded: **[C]** conceptual, **[M]** mathematical (by hand), **[P]** programming. Run programming
solutions from the project root.

**Exercise 1 [M].** With $S = \begin{pmatrix}1 & 0.5\\0.5 & 1\end{pmatrix}$ (two drugs) and
$A = \begin{pmatrix}1 & 0 & 1\\0 & 1 & 0\end{pmatrix}$ (two drugs × three diseases), compute $SA$ by the
row-mixing view and interpret row 1. Is $AS$ defined?

<details><summary>Solution</summary>

Row 1 of $SA$ $= 1\cdot(1,0,1) + 0.5\cdot(0,1,0) = (1, 0.5, 1)$; row 2 $= 0.5\cdot(1,0,1) + 1\cdot(0,1,0) = (0.5, 1, 0.5)$.
Drug 1 keeps its own indications (weight 1 from $S_{11}$) and acquires half a vote for disease 2 from its
similar drug 2. $AS$ is **not** defined: $(2\times3)(2\times2)$, inner dimensions 3 ≠ 2.

```python
import numpy as np
S = np.array([[1, 0.5], [0.5, 1]]); A = np.array([[1, 0, 1], [0, 1, 0]])
print(S @ A)
try:
    A @ S
except ValueError as e:
    print("A @ S:", type(e).__name__)
```

Output:

```text
[[1.  0.5 1. ]
 [0.5 1.  0.5]]
A @ S: ValueError
```

</details>

**Exercise 2 [M].** Prove (a) $(ABC)^\top = C^\top B^\top A^\top$; (b) $x^\top W y = y^\top W^\top x$;
(c) that $H_rWH_d^\top$ is symmetric when $H_r = H_d$ and $W = W^\top$.

<details><summary>Solution</summary>

(a) Apply $(XY)^\top = Y^\top X^\top$ twice: $((AB)C)^\top = C^\top(AB)^\top = C^\top B^\top A^\top$.
(b) $x^\top Wy$ is $1\times1$, hence equal to its transpose $(x^\top W y)^\top = y^\top W^\top x$.
(c) $(HWH^\top)^\top = (H^\top)^\top W^\top H^\top = HWH^\top$. (In the project $H_r \ne H_d$ — different node
types — so the question of symmetry does not even arise; the decoder is a drug × disease matrix.)

</details>

**Exercise 3 [M].** Compute `sym_norm(S)` for $S = \begin{pmatrix}1 & 0.5\\0.5 & 1\end{pmatrix}$ by hand, its
eigenvalues and eigenvectors, and the normalised Laplacian's eigenvalues.

<details><summary>Solution</summary>

$d = (1.5, 1.5)$, so $\tilde S = S/1.5 = \begin{pmatrix}2/3 & 1/3\\1/3 & 2/3\end{pmatrix}$. As in §6.2 the
eigenvectors are $(1,1)/\sqrt2$ and $(1,-1)/\sqrt2$, with eigenvalues $2/3 + 1/3 = 1$ and $2/3 - 1/3 = 1/3$.
$L_{\text{sym}} = I - \tilde S$ has eigenvalues $0$ and $2/3$. The top eigenvalue is 1 with eigenvector
$\propto D^{1/2}\mathbf1 \propto (1, 1)$ ✓ (§10.3).

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo.methods import sym_norm
S = np.array([[1, 0.5], [0.5, 1]])
print(sym_norm(S).round(4), np.linalg.eigvalsh(sym_norm(S)).round(4),
      np.linalg.eigvalsh(np.eye(2) - sym_norm(S)).round(4))
```

Output:

```text
[[0.6667 0.3333]
 [0.3333 0.6667]] [0.3333 1.    ] [-0.      0.6667]
```

</details>

**Exercise 4 [M].** Find the ranks of
$P = \begin{pmatrix}1&2\\2&4\\3&6\end{pmatrix}$, $Q = \begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$ and
$xy^\top$ for $x = (1, -1, 2)$, $y = (3, 0)$. Then argue why a model $\hat A = UV^\top$ with
$U \in \mathbb{R}^{593\times 128}$ can never reach rank 238.

<details><summary>Solution</summary>

$P$: column 2 = 2 × column 1 → rank 1. $Q$: row 3 = row 1 + row 2, rows 1 and 2 independent → rank 2.
$xy^\top$: an outer product of non-zero vectors → rank 1. $\operatorname{rank}(UV^\top) \le \operatorname{rank}(U) \le 128 < 238$
(§5.1, consequence 2); SCMFDD with $k = 128$ therefore cannot reproduce the training matrix exactly —
which is intended: it must generalise.

```python
import numpy as np
P = np.array([[1, 2], [2, 4], [3, 6]]); Q = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
print([int(np.linalg.matrix_rank(M)) for M in (P, Q, np.outer([1, -1, 2], [3, 0]))])
```

Output:

```text
[1, 2, 1]
```

</details>

**Exercise 5 [M].** Find the eigenvalues and eigenvectors of the *non-symmetric* matrix
$N = \begin{pmatrix}4 & 1\\2 & 3\end{pmatrix}$. Are the eigenvectors orthogonal? What does this tell you
about the spectral theorem?

<details><summary>Solution</summary>

$\det(N - \lambda I) = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2)$.
$\lambda = 5$: $(N - 5I)v = \begin{pmatrix}-1&1\\2&-2\end{pmatrix}v = 0 \Rightarrow v = (1, 1)$.
$\lambda = 2$: $\begin{pmatrix}2&1\\2&1\end{pmatrix}v = 0 \Rightarrow v = (1, -2)$.
$(1,1)\cdot(1,-2) = -1 \ne 0$: **not orthogonal**. The eigenvalues are real here, but orthogonality of
eigenvectors is guaranteed only for symmetric matrices. (Check: $\operatorname{tr} N = 7 = 5 + 2$, $\det N = 10 = 5\cdot2$.)

```python
import numpy as np
lam, V = np.linalg.eig(np.array([[4.0, 1.0], [2.0, 3.0]]))
print(lam.real, (V / V[0]).real.round(4))   # columns rescaled so that the first entry is 1
```

Output:

```text
[5. 2.] [[ 1.  1.]
 [ 1. -2.]]
```

</details>

**Exercise 6 [M].** Compute the SVD of $B = \begin{pmatrix}1&1\\0&0\\1&1\end{pmatrix}$ by hand, and its
Frobenius, spectral and nuclear norms.

<details><summary>Solution</summary>

All rows are multiples of $(1,1)$ → rank 1. $B^\top B = \begin{pmatrix}2&2\\2&2\end{pmatrix}$, eigenvalues 4 and 0,
$v_1 = (1,1)/\sqrt2$. $\sigma_1 = 2$, $u_1 = Bv_1/2 = (\sqrt2, 0, \sqrt2)/2 = (1, 0, 1)/\sqrt2$.
So $B = 2\,u_1v_1^\top$ (check: $2\cdot\tfrac12\begin{pmatrix}1&1\\0&0\\1&1\end{pmatrix}$ ✓). With a single singular
value 2: $\lVert B\rVert_F = \lVert B\rVert_2 = \lVert B\rVert_* = 2$ (all norms coincide for rank 1);
directly, $\lVert B\rVert_F = \sqrt{4\cdot1} = 2$ ✓.

```python
import numpy as np
B = np.array([[1, 1], [0, 0], [1, 1]], dtype=float)
print(np.linalg.svd(B, compute_uv=False).round(6),
      np.linalg.norm(B, "fro"), np.linalg.norm(B, 2).round(6), np.linalg.norm(B, "nuc").round(6))
```

Output:

```text
[2. 0.] 2.0 2.0 2.0
```

</details>

**Exercise 7 [M].** A matrix has singular values $(5, 3, 1)$. Give the spectral- and Frobenius-norm errors
of the best rank-1 and rank-2 approximations, and the fraction of energy kept by each.

<details><summary>Solution</summary>

Rank 1: spectral error $\sigma_2 = 3$; Frobenius error $\sqrt{3^2 + 1^2} = \sqrt{10} \approx 3.162$; energy
$25/35 \approx 71.4\%$. Rank 2: spectral error $\sigma_3 = 1$; Frobenius error $1$; energy $34/35 \approx 97.1\%$.

```python
import numpy as np
rng = np.random.default_rng(0)
Q1, _ = np.linalg.qr(rng.normal(size=(4, 3))); Q2, _ = np.linalg.qr(rng.normal(size=(3, 3)))
M = Q1 @ np.diag([5.0, 3.0, 1.0]) @ Q2.T                 # a matrix with exactly these singular values
U, s, Vt = np.linalg.svd(M, full_matrices=False)
for k in (1, 2):
    Mk = (U[:, :k] * s[:k]) @ Vt[:k]
    print(k, round(np.linalg.norm(M - Mk, 2), 4), round(np.linalg.norm(M - Mk, "fro"), 4),
          round((s[:k]**2).sum() / (s**2).sum(), 4))
```

Output:

```text
1 3.0 3.1623 0.7143
2 1.0 1.0 0.9714
```

</details>

**Exercise 8 [M].** For the weighted triangle with $S_{12} = 2$, $S_{13} = 1$, $S_{23} = 0$ (no self-loops):
write $D$ and $L$, verify $L\mathbf1 = 0$, and compute $x^\top L x$ for $x = (1, 0, 3)$ both with the matrix
and with the edge formula.

<details><summary>Solution</summary>

$d = (3, 2, 1)$, $L = \begin{pmatrix}3&-2&-1\\-2&2&0\\-1&0&1\end{pmatrix}$; row sums are 0, so $L\mathbf1 = 0$.
Edge formula: $2(1-0)^2 + 1(1-3)^2 + 0 = 2 + 4 = 6$. Matrix: $Lx = (3 - 3, -2, -1 + 3) = (0, -2, 2)$,
$x^\top Lx = 0 + 0 + 6 = 6$ ✓.

```python
import numpy as np
S = np.array([[0, 2, 1], [2, 0, 0], [1, 0, 0]], dtype=float)
L = np.diag(S.sum(1)) - S
x = np.array([1.0, 0.0, 3.0])
print(L, L @ np.ones(3), x @ L @ x)
```

Output:

```text
[[ 3. -2. -1.]
 [-2.  2.  0.]
 [-1.  0.  1.]] [0. 0. 0.] 6.0
```

</details>

**Exercise 9 [M].** Show that $\operatorname{tr}(U^\top L U) \ge 0$ for any $U$ when $S \ge 0$, and that it is 0
iff all rows of $U$ are equal within each connected component. Then express it through the eigenpairs
$(\lambda_i, q_i)$ of $L$.

<details><summary>Solution</summary>

By §11, $\operatorname{tr}(U^\top L U) = \tfrac12\sum_{ij}S_{ij}\lVert u_i - u_j\rVert^2$, a sum of non-negative terms. It is 0
iff $u_i = u_j$ whenever $S_{ij} > 0$, i.e. (following paths) iff $u$ is constant on each connected component.
Spectrally, write $L = \sum_i \lambda_i q_iq_i^\top$:
$\operatorname{tr}(U^\top LU) = \sum_i \lambda_i \operatorname{tr}(U^\top q_iq_i^\top U) = \sum_i \lambda_i \lVert q_i^\top U\rVert_2^2$.
The penalty charges each "graph frequency" component of $U$ by its eigenvalue: smooth components
($\lambda \approx 0$) are free, oscillating ones (large $\lambda$) are expensive — a low-pass filter.

</details>

**Exercise 10 [P].** Implement `sym_norm` with explicit diagonal matrices, check it equals the project's
version on a random 300 × 300 similarity matrix, and time both on 2,000 × 2,000.

<details><summary>Solution</summary>

```python
import sys, time; sys.path.insert(0, "src")
import numpy as np
from drepo.methods import sym_norm

def sym_norm_diag(S):
    d = S.sum(1); d[d == 0] = 1
    Dm = np.diag(1 / np.sqrt(d))
    return Dm @ S @ Dm

rng = np.random.default_rng(0)
X = rng.random((300, 300)); S = (X + X.T) / 2
print("equal:", np.allclose(sym_norm(S), sym_norm_diag(S)))
X = rng.random((2000, 2000)); S = (X + X.T) / 2

def best_time(f, reps=3):                  # best of 3 runs: timings are noisy
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(S); times.append(time.perf_counter() - t0)
    return min(times)

print(f"broadcasting {best_time(sym_norm):.3f}s   diagonal matrices {best_time(sym_norm_diag):.3f}s")
```

Output:

```text
equal: True
broadcasting 0.026s   diagonal matrices 0.283s
```

(Timings vary; the broadcasting version does $O(n^2)$ work, the diagonal-matrix version $O(n^3)$, so the
gap widens with $n$.)

</details>

**Exercise 11 [P].** Use power iteration to find the top eigenvalue of `sym_norm` of the kNN-sparsified
`pheno_mim` disease graph of Fdataset, and compare with `np.linalg.eigvalsh`. Why does it converge to 1?

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.methods import sym_norm
d = D.load("F")
S = D.fill_missing(d.disease_view("pheno_mim"))
N = sym_norm(S * D.knn_mask(S, 10))
x = np.random.default_rng(0).random(len(N))
for t in range(500):
    x = N @ x; x /= np.linalg.norm(x)
print("power iteration:", round(x @ N @ x, 6), "| eigvalsh max:", round(np.linalg.eigvalsh(N).max(), 6))
```

Output:

```text
power iteration: 1.0 | eigvalsh max: 1.0
```

By §10.3 the largest eigenvalue of $D^{-1/2}SD^{-1/2}$ is exactly 1 (eigenvector $\propto\sqrt d$) when all
weights are non-negative. Power iteration converges because the start vector has a positive component along
that eigenvector, and the other eigenvalues are smaller in absolute value (the self-loops keep the bottom of
the spectrum away from $-1$). Convergence speed is governed by the second-largest $|\lambda|$; if the graph has
several weakly connected clusters this is close to 1 and convergence is slow — that is why we iterate 500 times.

</details>

**Exercise 12 [P].** Evaluate the truncated-SVD recommender of §7.5 on **all five folds** for
$k \in \{5, 10, 20, 50\}$ and report mean AUC and AUPR. Which $k$ would you pick, and on which data should
you pick it in a real study?

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import kfold_splits, metrics
A = D.load("F").A
res = {k: [] for k in (5, 10, 20, 50)}
for f, tp, tn in kfold_splits(A, 5, seed=0):
    A_tr = A.copy().ravel(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
    U, s, Vt = np.linalg.svd(A_tr, full_matrices=False)
    idx = np.concatenate([tp, tn]); y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    for k in res:
        res[k].append(metrics(y, ((U[:, :k] * s[:k]) @ Vt[:k]).ravel()[idx]))
for k, ms in res.items():
    print(f"k={k:2d}: AUC {np.mean([m['AUC'] for m in ms]):.3f}  AUPR {np.mean([m['AUPR'] for m in ms]):.3f}")
```

Output:

```text
k= 5: AUC 0.774  AUPR 0.230
k=10: AUC 0.809  AUPR 0.316
k=20: AUC 0.808  AUPR 0.367
k=50: AUC 0.765  AUPR 0.361
```

Averaged over folds, AUC is flat between $k = 10$ and $k = 20$ and drops at 50, while AUPR peaks at $k = 20$; the right choice depends on which metric your study
prioritises (for heavily imbalanced data, AUPR — Unit B2). Crucially, $k$ is a hyper-parameter and must be chosen
on a **validation split carved out of the training data**, never on the test folds, otherwise the reported
numbers are optimistically biased (Units B1, E1). Choosing it here on test folds is fine only because this is an
illustration.

</details>

**Exercise 13 [P].** Build a toy SCMFDD: 3 drugs × 4 diseases where drug 2 has **no** known links but is very
similar to drug 0. Minimise
$\lVert A - UV^\top\rVert_F^2 + \mu(\lVert U\rVert_F^2 + \lVert V\rVert_F^2) + \lambda\operatorname{tr}(U^\top L_rU)$ with hand-written
gradients for $\lambda \in \{0, 2, 10\}$ and inspect drug 2's predicted row.

<details><summary>Solution</summary>

Gradients: $\nabla_U = -2(A - UV^\top)V + 2\mu U + 2\lambda L_rU$, $\nabla_V = -2(A - UV^\top)^\top U + 2\mu V$.

```python
import numpy as np
A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]], dtype=float)   # drug 2: no links
S = np.array([[1, .1, .9], [.1, 1, .1], [.9, .1, 1]])                     # drug 2 ~ drug 0
d = S.sum(1); Lr = np.eye(3) - S / np.sqrt(np.outer(d, d))                # I - D^-1/2 S D^-1/2
mu, lr = 0.05, 0.01
for lam in (0, 2, 10):
    rng = np.random.default_rng(0)
    U, V = 0.1 * rng.normal(size=(3, 2)), 0.1 * rng.normal(size=(4, 2))
    for _ in range(5000):
        R = A - U @ V.T
        gU = -2 * R @ V + 2 * mu * U + 2 * lam * Lr @ U
        gV = -2 * R.T @ U + 2 * mu * V
        U -= lr * gU; V -= lr * gV
    print(f"lambda={lam:2d}  drug 2 predicted row: {(U @ V.T)[2].round(2)}")
```

Output:

```text
lambda= 0  drug 2 predicted row: [0. 0. 0. 0.]
lambda= 2  drug 2 predicted row: [0.14 0.05 0.14 0.05]
lambda=10  drug 2 predicted row: [0.31 0.12 0.31 0.12]
```

With $\lambda = 0$ drug 2's row is exactly 0: nothing connects it to anything. With $\lambda > 0$ the Laplacian term
pulls $u_2$ towards $u_0$, and drug 2's scores take drug 0's *pattern* (diseases 0 and 2 high, 1 and 3 low), even
though it has no links. The scores stay modest because the Frobenius data term treats drug 2's unknown cells as
zeros to be fitted — the "unknown ≠ negative" tension that motivates negative sampling (Unit B4). Larger $\lambda$
transfers more.

</details>

**Exercise 14 [P].** On Fdataset's $A$, verify numerically: $\lVert A\rVert_F^2$ = number of ones $= \sum\sigma_i^2$;
$\lVert A\rVert_2 = \sigma_1$; $\lVert A\rVert_* = \sum\sigma_i$; and the factor identity
$\lVert A\rVert_* = \tfrac12(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)$ for $U = U_r\Sigma_r^{1/2}$, $V = V_r\Sigma_r^{1/2}$.

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
A = D.load("F").A.astype(np.float64)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(round(np.linalg.norm(A, "fro")**2, 6), A.sum(), (s**2).sum().round(6))
print(np.linalg.norm(A, 2).round(6), s[0].round(6))
print(np.linalg.norm(A, "nuc").round(4), s.sum().round(4))
Uf, Vf = U * np.sqrt(s), Vt.T * np.sqrt(s)                        # balanced factors, A = Uf Vf^T
print(np.allclose(Uf @ Vf.T, A), (0.5 * ((Uf**2).sum() + (Vf**2).sum())).round(4))
```

Output:

```text
1933.0 1933.0 1933.0
12.699401 12.699401
517.8739 517.8739
True 517.8739
```

The balanced factorisation attains the minimum in $\lVert X\rVert_* = \min_{UV^\top = X}\tfrac12(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)$:
each factor's squared Frobenius norm is $\sum_i\sigma_i$, so their average is the nuclear norm.

</details>

**Exercise 15 [C].** DRRS declares "observed" = non-zero (`Om = (T != 0)`). List the consequences of this choice
for drug–disease prediction and one alternative.

<details><summary>Solution</summary>

(1) Every unknown drug–disease pair is treated as *missing*, not as a negative: the algorithm is free to fill it,
which is what we want for prediction (it never forces unknowns to 0, unlike SCMFDD's Frobenius term). (2) Zero
similarities are also treated as missing, so the method may "invent" similarity between unrelated drugs if the
low-rank structure suggests it; sparse (kNN) similarity inputs would therefore change DRRS's behaviour a lot.
(3) Because all observed association entries are 1s, the observed set carries no explicit negative information,
so the completed values are only meaningful as a *ranking*. Alternative: treat a sample of unknown cells as
observed zeros with a lower weight (weighted matrix completion / negative sampling), or use a separate mask for
the similarity blocks.

</details>

---

## 15. Answers to the PREREQUISITES.md self-check questions (Unit A2)

### Q1. What does multiplying a similarity matrix $S$ ($n\times n$) by $A$ ($n\times m$) do to each row of $A$?

**Short answer.** It replaces row $i$ by a **similarity-weighted combination of all rows of $A$**:
$(SA)_{i,:} = \sum_k S_{ik}A_{k,:}$. Entry $(i, j)$ becomes $\sum_k S_{ik}A_{kj}$ = the total similarity of
entity $i$ to the entities known to be linked to column $j$. It is one step of propagation (message passing)
along the similarity graph: guilt by association.

**In depth.**

* *Derivation.* By the definition of the product, row $i$ of $SA$ is row $i$ of $S$ times $A$, which is the linear
  combination of the rows of $A$ with coefficients $S_{i1},\dots,S_{in}$ (§3.4). Rows of entities dissimilar to $i$
  ($S_{ik} = 0$) contribute nothing; very similar ones contribute almost their whole row; $S_{ii} = 1$ keeps $i$'s
  own row in the mix.
* *Worked numbers* (§3.6): drug 2 with no indications, similarity 0.8 to drug 1 (treats disease 1) and 0.5 to
  drug 3 (treats disease 2), gets row $(0.8, 0.5)$.
* *Scale and normalisation.* With raw $S$ the result is a weighted **sum**: entities with many or strong
  neighbours (hubs) and diseases with many drugs get inflated scores. Row-normalising ($K = D^{-1}S$, as
  `knn_kernel` does) turns it into a weighted **average** — "what fraction of my neighbours treat $j$?" —
  and symmetric normalisation ($D^{-1/2}SD^{-1/2}$, `sym_norm`) corrects for both endpoints' degrees and keeps
  the operator's spectral radius at 1 so repeated multiplication is stable (§10.3). Sparsifying $S$ to $k$
  nearest neighbours first keeps only informative neighbours; a dense $S$ averages over everybody and blurs
  all rows towards the global mean.
* *Left vs right.* $S_rA$ (drug similarity on the left) mixes **drug rows**: it spreads each disease column's
  known drugs to similar drugs. To spread along **disease** similarity you multiply on the right: $AS_d$ mixes
  columns. Consequence for cold start (leave-one-disease-out): the held-out disease's column of $A_{\text{train}}$
  is all zero, and $(S_rA)_{:,j} = S_r A_{:,j} = 0$ — drug-side propagation can say nothing about a disease with no
  known drugs. Only right multiplication by a disease similarity (or a disease-side GNN relation) can fill that
  column. This is exactly why the disease views matter so much in LODO:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import metrics
d = D.load("F"); A = d.A
j = int(A.sum(0).argmax())                               # hold out the most-treated disease
A_tr = A.copy(); A_tr[:, j] = 0
Kr = D.knn_kernel(d.drug_view("chem_cdk"), 10)
Kd = D.knn_kernel(d.disease_view("pheno_mim"), 10)
drug_side = (Kr @ A_tr)[:, j]
disease_side = (A_tr @ Kd.T)[:, j]
print("drug-side scores for the held-out disease, max:", drug_side.max())
print("disease-side AUC for the held-out disease:", round(metrics(A[:, j], disease_side)["AUC"], 3))
```

Output:

```text
drug-side scores for the held-out disease, max: 0.0
disease-side AUC for the held-out disease: 0.869
```

* *Connections.* The same product is one step of a random walk (MBiRW's `Mr @ R`), the aggregation step of a
  GCN layer ($\hat A H$) before its learned transform, a gradient step of the Laplacian penalty (§11.2), and the
  propagation head of MV-HGAT (`K @ Am`), which reaches AUC ≈ 0.84 on its own when a drug view and a disease
  view are combined (§12.2).

### Q2. Why does a low-rank assumption make sense for drug–disease matrices?

**Short answer.** Because drug–disease relationships are generated by a comparatively small number of shared
underlying factors — mechanisms of action, target families, pathways, therapeutic classes — so the matrix of
indication propensities is approximately a sum of a few rank-1 patterns "drugs of type $f$ treat diseases of type
$f$". A low-rank model has few parameters ($k(n+m)$ instead of $nm$), which is the only way to infer
hundreds of thousands of unknown cells from a couple of thousand known ones.

**In depth.**

* *Biological argument.* Drugs cluster into classes that share targets (e.g. many β-blockers, many NSAIDs), and
  each class is used across a family of related diseases (cardiovascular, inflammatory…). If drug $i$ is described
  by its "affinity" $u_{if}$ to mechanism $f$ and disease $j$ by its "dependence" $v_{jf}$ on that mechanism, a
  natural model of propensity is $\sum_f u_{if}v_{jf} = (UV^\top)_{ij}$: rank $k$ = number of mechanisms.
* *Data argument.* The observed matrix contains exact redundancy (246 distinct columns among 313; rank 238), and
  its leading singular vectors capture shared structure: a rank-10 reconstruction of a training fold ranks hidden
  links with AUC ≈ 0.81 and AUPR ≈ 0.29, versus ≈ 0.5 and ≈ 0.01 by chance (§7.5).
* *Statistical argument.* A rank constraint (or its convex surrogate, the nuclear norm; or Frobenius penalties on
  factors, which are equivalent, §9.2) is an **inductive bias**: it forbids the model from memorising the training
  matrix (full rank reproduces it exactly and generalises at chance level) and forces it to explain the known links
  with shared patterns, which then assign mass to the unknown cells that fit those patterns. Theory (Candès &
  Recht) shows exact recovery is possible under assumptions when the truth is low rank.
* *Caveats — what the assumption does not say.* The **observed** binary matrix is not low rank (its spectrum has a
  long tail: 10 components hold only 38% of the energy) because of rare diseases with one idiosyncratic drug,
  noise, and the fact that most true indications are unobserved. Entries are also not missing at random (well-studied
  diseases have more known drugs), which violates the recovery theorems. That is why the project's methods combine
  low rank with side information: DRRS completes a block matrix that includes the similarity matrices, SCMFDD adds
  Laplacian smoothness, and MV-HGAT's bilinear decoder (rank ≤ 192) is complemented by a similarity-propagation head.

### Q3. What does $\operatorname{tr}(U^\top L U)$ penalise?

**Short answer.** Differences between the latent vectors of similar entities:
$$\operatorname{tr}(U^\top L U) = \tfrac12\sum_{i,j}S_{ij}\,\lVert u_i - u_j\rVert_2^2 \quad\text{for } L = D - S,$$
so each pair of drugs (or diseases) pays its similarity times the squared distance between their latent factor
vectors. It is a **smoothness** penalty: latent factors must vary slowly over the similarity graph. With the
normalised Laplacian used in SCMFDD's code, the vectors are first divided by $\sqrt{d_i}$:
$\tfrac12\sum_{ij}S_{ij}\lVert u_i/\sqrt{d_i} - u_j/\sqrt{d_j}\rVert^2$, which stops high-degree nodes dominating.

**In depth.**

* *Proof.* The diagonal entries of $U^\top LU$ are the quadratic forms $U_{:,f}^\top LU_{:,f}$ of its columns
  (trace property 6), and each equals $\tfrac12\sum_{ij}S_{ij}(U_{if} - U_{jf})^2$ (§10.2). Summing over $f$ turns the
  squared differences of coordinates into squared Euclidean distances of rows (§11.1).
* *Worked example.* Path graph, $u = (1, 1, 5)$ costs 16, $(1, 3, 5)$ costs 8, $(1,1,1)$ costs 0 (§11.1).
* *Spectral view.* $\operatorname{tr}(U^\top LU) = \sum_i\lambda_i\lVert q_i^\top U\rVert^2$ (Exercise 9): components of $U$ along
  smooth eigenvectors of $L$ (small $\lambda$) cost little, oscillating ones cost a lot — a low-pass filter on the
  graph. The constant direction is free, so the penalty never shrinks $U$ as a whole (that is the job of the
  $\mu\lVert U\rVert_F^2$ term).
* *Dynamics.* Its gradient is $2LU$; a gradient step mixes each $u_i$ with the normalised average of its
  neighbours — one graph-propagation step (§11.2). So in SCMFDD, a drug with no known links is pulled towards the
  factors of its similar drugs and inherits their predicted indications (Exercise 13). This is guilt by association
  imposed as a prior.
* *Trade-off.* The weight $\lambda$ (`lam = 2.0` in `SCMFDD`) balances fit against smoothness: too small, and
  similarity is ignored; too large, and factors of connected entities collapse together so the model cannot express
  differences between similar drugs that do treat different diseases. Like rank, it is tuned on validation data.
* *Not the same as* $\lVert U\rVert_F^2$: Frobenius shrinks every factor towards 0 independently; the Laplacian
  term only cares about *differences along edges*.

---

## 16. Summary and cheat sheet

| Concept | Formula / fact | Project |
|---|---|---|
| matrix product | $C_{ij} = \sum_k A_{ik}B_{kj}$; $(n\times p)(p\times m) = n\times m$ | everywhere |
| row mixing | $(SA)_{i,:} = \sum_k S_{ik}A_{k,:}$ | `K @ Am`, `Mr @ R`, GNN aggregation |
| column mixing | $(AS)_{:,j} = \sum_l S_{lj} A_{:,l}$ | `(K @ Am.T).T`, `R @ Md` |
| outer-product view | $AB = \sum_k A_{:,k}B_{k,:}$ | $UV^\top = \sum_f u_fv_f^\top$ |
| transpose | $(AB)^\top = B^\top A^\top$ | `A.T` for disease←drug messages |
| diagonal scaling | $\operatorname{diag}(d)M$ rows, $M\operatorname{diag}(d)$ columns | `S * d[:, None] * d[None, :]` |
| Gram matrix | $GG^\top \succeq 0$ | `G @ G.T` in `jaccard` |
| rank | # independent columns = # non-zero $\sigma$; $\operatorname{rank}(UV^\top)\le k$ | rank(A) = 238 |
| eigen | $Mv = \lambda v$; symmetric ⇒ $M = Q\Lambda Q^\top$, real $\lambda$, orthonormal $Q$ | `eigh` |
| Rayleigh | $\lambda_{\min} \le x^\top Mx/x^\top x \le \lambda_{\max}$ | PSD test, spectral bounds |
| SVD | $A = U\Sigma V^\top = \sum\sigma_iu_iv_i^\top$; $\sigma_i^2 = \lambda_i(A^\top A)$ | DRRS |
| Eckart–Young | best rank-$k$: $A_k$; errors $\sigma_{k+1}$ (2-norm), $\sqrt{\sum_{i>k}\sigma_i^2}$ (F) | low-rank recommenders |
| Frobenius | $\lVert A\rVert_F^2 = \sum A_{ij}^2 = \operatorname{tr}(A^\top A) = \sum\sigma_i^2$ | SCMFDD fit term |
| spectral | $\lVert A\rVert_2 = \sigma_1$ | DRRS `tau` |
| nuclear | $\lVert A\rVert_* = \sum\sigma_i = \min_{UV^\top = A}\tfrac12(\lVert U\rVert_F^2 + \lVert V\rVert_F^2)$ | matrix completion |
| SVT | $\mathcal{D}_\tau(Y) = U(\Sigma - \tau)_+V^\top = \arg\min\tfrac12\lVert X - Y\rVert_F^2 + \tau\lVert X\rVert_*$ | `DRRS` loop |
| trace | $\operatorname{tr}(AB) = \operatorname{tr}(BA)$; $\operatorname{tr}(A^\top B) = \sum A_{ij}B_{ij}$; $\operatorname{tr} = \sum\lambda$ | `torch.trace` |
| degree / Laplacian | $d = S\mathbf1$, $L = D - S$, $x^\top Lx = \tfrac12\sum S_{ij}(x_i - x_j)^2$, $L\mathbf1 = 0$ | — |
| sym. normalisation | $\tilde S = D^{-1/2}SD^{-1/2}$, eigenvalues in $[-1, 1]$, top eigvec $\propto\sqrt d$ | `sym_norm` |
| normalised Laplacian | $L_{\text{sym}} = I - \tilde S$, eigenvalues in $[0, 2]$ | `SCMFDD` `Lr`, `Ld` |
| random-walk normalisation | $P = D^{-1}S$, row-stochastic, same eigenvalues as $\tilde S$ | `knn_kernel` |
| smoothness | $\operatorname{tr}(U^\top LU) = \tfrac12\sum S_{ij}\lVert u_i - u_j\rVert^2$; gradient $2LU$ | `SCMFDD` |
| bilinear decoder | $\text{logit}_{ij} = h_i^\top Wh_j$, all pairs $H_rWH_d^\top$, rank ≤ width | `MVHGAT.forward` |

**Five sentences to remember.** (1) Left multiplication mixes rows, right multiplication mixes columns.
(2) Symmetric matrices are sums of perpendicular rank-1 pieces weighted by real eigenvalues. (3) Every matrix is a
sum of rank-1 pieces weighted by singular values, and truncating that sum is the best low-rank approximation.
(4) Normalise graphs by degree so propagation averages instead of sums and stays stable. (5) $\operatorname{tr}(U^\top LU)$
is the total disagreement of embeddings across similarity edges; its gradient is propagation.

---

## 17. Further resources (curated)

All links were checked to resolve when this unit was written. **Free** unless marked **Paid**.

**Visual intuition first**

* [3Blue1Brown — *Essence of Linear Algebra* (YouTube playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) and
  [topic page](https://www.3blue1brown.com/topics/linear-algebra) — **Free.** The best geometric intuition for
  matrices as transformations, determinants and eigenvectors; watch before §3 and §6.
* [Margalit & Rabinoff — *Interactive Linear Algebra* (Georgia Tech)](https://textbooks.math.gatech.edu/ila/) — **Free.**
  Textbook with interactive figures; strong on rank, eigenvalues and orthogonality.

**University courses**

* [MIT 18.06 Linear Algebra (Gilbert Strang, OCW)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) and the
  self-paced [18.06SC](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — **Free.** The classic course:
  lectures, problem sets with solutions, exams. Lectures 1–3 (multiplication), 9–10 (rank), 21–25 (eigen,
  symmetric), 29–30 (SVD).
* [MIT 18.065 Matrix Methods in Data Analysis, Signal Processing, and Machine Learning (Strang, OCW)](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) — **Free.**
  The follow-up course that this unit most resembles: SVD, Eckart–Young, norms, low rank, graph Laplacians.
* [Stanford CS229 — Linear Algebra Review and Reference (Kolter & Do)](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Free.**
  A compact reference of the identities used in ML; keep it open while doing exercises.
* [fast.ai — Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra) — **Free.**
  Code-first notebooks on SVD, randomized SVD and why numerical algorithms matter.

**Textbooks**

* Gilbert Strang, [*Linear Algebra and Learning from Data*](https://math.mit.edu/~gs/learningfromdata/) — **Paid** (book;
  the site has sample sections and video links). Chapters I.1–I.9 are exactly the prerequisite list; I.9 proves Eckart–Young.
* Gilbert Strang, [*Introduction to Linear Algebra*](https://math.mit.edu/~gs/linearalgebra/) — **Paid** (book; the site has free
  problem solutions and summaries). The standard first textbook matching 18.06.
* Boyd & Vandenberghe, [*Introduction to Applied Linear Algebra — Vectors, Matrices, and Least Squares* (VMLS)](https://web.stanford.edu/~boyd/vmls/) — **Free** PDF.
  Very applied; excellent for matrix multiplication, norms and least squares; comes with Python companion.
* Deisenroth, Faisal & Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/) — **Free** PDF.
  Chapters 2–4 (linear algebra, analytic geometry, matrix decompositions) with ML-oriented examples.
* Sheldon Axler, [*Linear Algebra Done Right*, 4th ed.](https://linear.axler.net/) — **Free** (open access).
  The rigorous, proof-based treatment of the spectral theorem and SVD for when you want full proofs.
* [*The Matrix Cookbook* (Petersen & Pedersen)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — **Free.**
  Look-up table of identities and matrix derivatives (e.g. $\nabla_U\operatorname{tr}(U^\top LU)$).
* Jeremy Kun, [*Singular Value Decomposition* (blog series)](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/) — **Free.**
  A programmer's derivation of the SVD from scratch.

**Graphs and Laplacians**

* Ulrike von Luxburg (2007), [*A Tutorial on Spectral Clustering*](https://arxiv.org/abs/0711.0189) — **Free.**
  The clearest derivation of unnormalised/normalised Laplacians and their properties (§10).
* Daniel Spielman, [*Spectral and Algebraic Graph Theory* (book draft, Yale)](http://cs-www.cs.yale.edu/homes/spielman/sagt/) — **Free.**
  Rigorous course notes on Laplacian spectra, random walks and quadratic forms.
* Fan Chung, [*Spectral Graph Theory* (CBMS lectures; chapters online)](https://mathweb.ucsd.edu/~fan/research/revised.html) — **Free** chapters.
  The reference for the normalised Laplacian $I - D^{-1/2}SD^{-1/2}$ and its $[0, 2]$ spectrum.
* William Hamilton, [*Graph Representation Learning*](https://www.cs.mcgill.ca/~wlh/grl_book/) — **Free** pre-print.
  Ch. 2 covers adjacency/Laplacian matrices and their link to GNNs.
* Distill, [*Understanding Convolutions on Graphs*](https://distill.pub/2021/understanding-gnns/) — **Free.**
  Interactive explanation of how Laplacians and polynomial filters become GNN layers.

**Papers behind the project's methods**

* Kipf & Welling (2017), [*Semi-Supervised Classification with Graph Convolutional Networks*](https://arxiv.org/abs/1609.02907) — **Free.** Introduces $\tilde D^{-1/2}(A + I)\tilde D^{-1/2}$.
* Wu et al. (2019), [*Simplifying Graph Convolutional Networks*](https://arxiv.org/abs/1902.07153) — **Free.** Spectral analysis of the self-loop "renormalisation trick" (low-pass filtering).
* Cai, Candès & Shen (2010), [*A Singular Value Thresholding Algorithm for Matrix Completion*](https://arxiv.org/abs/0810.3286) — **Free.** The SVT algorithm used by DRRS, including the prox theorem.
* Candès & Recht (2009), [*Exact Matrix Completion via Convex Optimization*](https://arxiv.org/abs/0805.4471) — **Free.** When nuclear-norm minimisation recovers a low-rank matrix.
* Recht, Fazel & Parrilo (2010), [*Guaranteed Minimum-Rank Solutions … via Nuclear Norm Minimization*](https://arxiv.org/abs/0706.4138) — **Free.** The nuclear norm as convex surrogate of rank; factor-norm identity.
* Halko, Martinsson & Tropp (2011), [*Finding Structure with Randomness*](https://arxiv.org/abs/0909.4061) — **Free.** Randomised SVD, as in `torch.svd_lowrank`.
* Zhang et al. (2018), [*Predicting drug–disease associations by using similarity constrained matrix factorization* (SCMFDD)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2220-4) — **Free** (open access).
* Luo et al. (2018), [*Computational drug repositioning using low-rank matrix approximation and randomized algorithms* (DRRS)](https://digitalcommons.odu.edu/computerscience_fac_pubs/362/) — repository page **free**; journal full text in *Bioinformatics* 34:1904 may be **Paid**.

**Practice**

* [Khan Academy — Linear algebra](https://www.khanacademy.org/math/linear-algebra) — **Free.** Many short worked examples and exercises if any basic step (determinants, solving systems) is rusty.

---

## 18. Glossary

* **basis / dimension** — a linearly independent spanning set; its size.
* **bilinear form / decoder** — $x^\top Wy$, linear in each argument; the decoder scores pairs as $h_i^\top Wh_j$.
* **block matrix** — a matrix partitioned into sub-matrices that multiply like scalars.
* **characteristic polynomial** — $\det(M - \lambda I)$; its roots are the eigenvalues.
* **column space / row space** — span of the columns / rows.
* **convex envelope** — the largest convex function lying below a given function on a set.
* **degree / degree matrix** — $d_i = \sum_jS_{ij}$; $D = \operatorname{diag}(d)$.
* **Dirichlet energy** — $x^\top Lx$, the roughness of a node signal on a graph.
* **eigenvalue / eigenvector** — $Mv = \lambda v$, $v \ne 0$.
* **Eckart–Young–Mirsky theorem** — truncated SVD is the best rank-$k$ approximation (spectral and Frobenius norms).
* **energy (spectral)** — $\sum_{i\le k}\sigma_i^2/\sum_i\sigma_i^2$, fraction of squared Frobenius norm captured.
* **Frobenius norm** — $\sqrt{\sum A_{ij}^2}$.
* **Gram matrix** — a matrix of inner products $GG^\top$; always PSD.
* **Hadamard product** — element-wise product (`*`).
* **identity matrix** — $I$, ones on the diagonal.
* **incoherence** — a condition that singular vectors are spread out, needed for exact matrix completion.
* **Laplacian (graph)** — $L = D - S$; normalised $L_{\text{sym}} = I - D^{-1/2}SD^{-1/2}$.
* **linear combination / linear map** — weighted sum of vectors / function preserving sums and scalings (a matrix).
* **linear independence** — no vector is a combination of the others.
* **low-rank approximation** — approximating a matrix by one of rank $k$, e.g. truncated SVD or $UV^\top$.
* **matrix completion** — recovering missing entries of a (presumed low-rank) matrix.
* **nuclear norm** — $\sum_i\sigma_i$; convex surrogate of rank.
* **orthogonal matrix / orthonormal columns** — $Q^\top Q = I$.
* **outer product** — $xy^\top$, a rank-1 matrix.
* **over-smoothing** — node representations becoming indistinguishable after many propagation steps.
* **positive semidefinite (PSD)** — symmetric with $x^\top Mx \ge 0$ for all $x$; equivalently all eigenvalues $\ge 0$.
* **power iteration** — repeated multiplication and normalisation, converging to the top eigenvector.
* **projection** — $UU^\top$ for orthonormal $U$; maps onto the span of $U$'s columns.
* **random-walk normalisation** — $D^{-1}S$, a row-stochastic transition matrix.
* **rank** — number of linearly independent columns (= rows) = number of non-zero singular values.
* **Rayleigh quotient** — $x^\top Mx/x^\top x$, between the extreme eigenvalues.
* **row-stochastic** — non-negative with rows summing to 1.
* **singular value decomposition (SVD)** — $A = U\Sigma V^\top$ with orthogonal $U, V$ and non-negative diagonal $\Sigma$.
* **singular value thresholding (SVT)** — shrinking singular values by $\tau$ (prox of the nuclear norm); the matrix-completion algorithm built on it.
* **soft thresholding** — $\operatorname{sign}(y)\max(|y| - \tau, 0)$.
* **spectral norm** — $\sigma_1$, the largest stretch of a unit vector.
* **spectral theorem** — real symmetric matrices have real eigenvalues and an orthonormal eigenbasis.
* **symmetric matrix** — $M = M^\top$.
* **symmetric normalisation** — $D^{-1/2}SD^{-1/2}$.
* **trace** — sum of diagonal entries = sum of eigenvalues.
* **transpose** — $A^\top$, rows and columns swapped.
* **truncated SVD** — $A_k = \sum_{i\le k}\sigma_iu_iv_i^\top$.
* **unitarily invariant norm** — unchanged by orthogonal transformations on either side.
