# Unit C3 — Graph Neural Networks: Message Passing and GCN

> **Course:** Drug repositioning with graph neural networks — a self-contained course
> **Chapter:** 11 of 20 · **Track:** C (Graphs) · **Unit code:** C3

| | |
|---|---|
| **Prerequisites** | A1 (NumPy, broadcasting, boolean masks), A2 (matrix products, symmetric matrices, eigen-decomposition, the graph Laplacian), A3 (expectation and variance, lightly), B1 (supervised learning, train/test splits), B3 (PyTorch modules, autograd, Adam, dropout), C1 (adjacency matrices, degree, bipartite and kNN graphs), C2 (transition matrices, label propagation). B4 (embeddings and dot-product decoders) helps for Section 11. |
| **Estimated study time** | 12–15 hours: about 6 h reading and redoing the worked examples by hand, 4 h running and modifying the code, 3–5 h on the exercises. |
| **Code environment** | `C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv` (PyTorch 2.x, NumPy, scikit-learn). Every code block in this chapter was run on CPU and its printed output is shown underneath. No `torch_geometric` is needed: like the project, we build everything from dense matrices. |

## Learning objectives

When you finish this chapter you will be able to:

1. **Explain** in two or three sentences why an ordinary multilayer perceptron (MLP) cannot be applied directly to a graph, using the words *permutation*, *invariance* and *equivariance* correctly.
2. **Write down** the general message-passing template (message → aggregate → update) and **prove** that sum, mean and max aggregation are permutation invariant and that a GCN layer is permutation equivariant.
3. **Derive** the GCN layer of Kipf & Welling (2017) from a first-order Chebyshev approximation of a spectral graph filter, including the *renormalisation trick* $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$, and **explain** why the trick is needed.
4. **Compute by hand** one GCN layer on a 4-node graph and check the result with NumPy.
5. **Implement from scratch** (dense PyTorch) a GCN layer, a GraphSAGE mean layer with neighbour sampling, a max-pool aggregator and a GIN layer, and train a small GCN for node classification.
6. **State** what the 1-Weisfeiler–Lehman (WL) test is, why message-passing GNNs are at most as powerful as it, and why GIN uses a *sum* aggregator.
7. **Explain and demonstrate** that the receptive field of an $L$-layer GNN is the $L$-hop neighbourhood, and **diagnose** over-smoothing and over-squashing; **choose** among remedies (kNN sparsification, residual connections, jumping knowledge, DropEdge, normalisation).
8. **Distinguish** transductive from inductive learning and **say precisely** which kind the project's model is and what that implies for a brand-new drug.
9. **Translate** between dense-matrix and sparse edge-list implementations and estimate their memory cost.
10. **Read** `_GCN`, `NIMCGCN`, `LAGCN` (`src/drepo/methods.py`), `knn_mask` (`src/drepo/data.py`) and the jumping-knowledge concat in `MVHGAT.encode` (`src/drepo/model.py`) line by line, and explain the design choice behind every line.

---

## 1. Motivation: why this matters for *this* project

The project predicts new **drug → disease indications**. On Fdataset we have 593 drugs, 313 diseases and only 1,933 known links: the association matrix $A$ is about **1.04 % ones**. Most of what we know about a drug is therefore *not* its own row of $A$ but **who it resembles**: drugs with similar chemistry (`chem_cdk`, `chem_ecfp`), drugs that touch similar genes (`gene_r`); diseases with similar phenotypes (`pheno_mim`), a similar place in the MONDO ontology (`sem_mondo`) or similar genes (`gene_d`).

Each of these similarity "views" is turned into a **k-nearest-neighbour graph** (k = 10, symmetrised, with self-loops) by `data.py::knn_mask`. Here is what those graphs actually look like on Fdataset (computed with the project's own `knn_mask`; "isolated" means no neighbour other than itself, typically because the drug has no SMILES or the disease has no genes in CTD):

| View | Nodes | Undirected edges | Degree min / mean / max | Isolated |
|---|---|---|---|---|
| `chem_cdk` (drug) | 593 | 4,384 | 10 / 14.8 / 94 | 0 |
| `chem_ecfp` (drug) | 593 | 4,067 | 0 / 13.7 / 49 | 15 |
| `gene_r` (drug) | 593 | 3,866 | 0 / 13.0 / 49 | 81 |
| `pheno_mim` (disease) | 313 | 2,124 | 10 / 13.6 / 31 | 0 |
| `sem_mondo` (disease) | 313 | 2,492 | 0 / 15.9 / 128 | 9 |
| `gene_d` (disease) | 313 | 782 | 0 / 5.0 / 40 | 153 |

So a drug is a **node** in several graphs, and its "meaning" for our task is spread across its neighbourhoods. In Unit C2 you met a *fixed* way to exploit this — propagation / random walks such as MBiRW: "score drug $i$ for disease $j$ by how strongly $i$'s neighbours are linked to $j$". That idea is strong, but its rules are hand-written: every neighbour counts in proportion to a fixed similarity, every view is weighted by a fixed constant, and only raw links are propagated.

A **graph neural network (GNN)** is *learned* propagation. Each layer lets every node collect information from its neighbours, transform it with trainable weights, and combine it with its own state. After two layers a drug's vector summarises its 2-hop neighbourhood — including, through the known-link edges, *the other drugs that treat the same diseases*. That is exactly the collaborative-filtering signal recommender systems exploit (Unit B4), now expressed on a graph.

**A concrete example from the project.** Two of the baselines in `methods.py` are GCNs: **NIMCGCN** (separate GCNs on the drug and disease similarity graphs) and **LAGCN** (one GCN on the combined drug–disease graph with attention over layers). Our own model, **MV-HGAT**, is also a message-passing GNN, but with attention (Unit C4), one relation per view (Unit C5) and careful link-prediction training (Unit C6). On Fdataset 5-fold cross-validation:

| Method | AUC | AUPR |
|---|---|---|
| MV-HGAT (ours) | 0.939 | 0.488 |
| NIMCGCN | 0.838 | 0.096 |
| LAGCN | 0.833 | 0.133 |

Same data, same splits, all three are "GNNs" — yet the AUPR differs by a factor of four or five. By the end of this chapter you will be able to point at *specific lines* of the baselines and say which design decisions plausibly cost them performance (dense vs sparse graphs and over-smoothing; which edges carry messages; what the training loss sees). You will also understand a lesson the project learned the hard way: a plain GNN that was supervised on links *visible in its own input graph* learned to **detect** edges rather than **predict** them (validation AUC 0.71) until training was changed to supervise on links hidden from the graph (0.92). Section 11 reproduces that effect on a toy problem.

---

## 2. Why standard neural networks fail on graphs

### 2.1 Three awkward properties of graphs

Suppose you want a neural network that outputs one vector per drug, using the drug–drug similarity graph. The obvious idea is: "feed the adjacency matrix $A$ (593 × 593) and the feature matrix $X$ into an MLP". This fails for three reasons.

1. **No canonical order.** Images have a natural pixel order and sentences a natural word order. Graph nodes do not: drug #17 is #17 only because of how the CSV was sorted. If we re-sort the drugs alphabetically, $A$ and $X$ have their rows (and $A$ its columns) shuffled, but *it is the same graph*. A function that gives different answers for the same graph written in a different order is learning an accident of bookkeeping.
2. **Variable size.** An MLP has a fixed input width. Fdataset has 593 drugs, Cdataset 663. A model whose first layer has $593^2$ inputs cannot even be applied to Cdataset.
3. **Irregular neighbourhoods.** A pixel always has 8 neighbours in fixed relative positions (up, down-left, …), which is what lets a CNN share a 3 × 3 filter across the image. In `chem_cdk` a drug has between 10 and 94 neighbours and there is no "up" or "left" neighbour — just an unordered set.

What we want is a model that (a) does not care about node order, (b) works for any number of nodes, and (c) shares parameters across nodes the way a CNN shares a filter across pixels. The way to get all three is to **build layers out of operations on unordered neighbour sets**.

### 2.2 Permutations, invariance and equivariance — precisely

A **permutation matrix** $P \in \{0,1\}^{N\times N}$ has exactly one 1 in each row and each column. Multiplying $X$ (rows = nodes) on the left by $P$ reorders the rows: if $P$ has its 1 in row $i$ at column $\pi(i)$, then $(PX)_i = X_{\pi(i)}$. Relabelling the nodes of a graph changes its adjacency matrix to $PAP^\top$ (rows *and* columns reordered) and its features to $PX$.

Let $f$ be a function of a graph $(A, X)$.

* $f$ is **permutation invariant** if $f(PAP^\top, PX) = f(A, X)$ for every permutation $P$. Use: graph-level outputs ("is this molecule toxic?"). The output does not depend on the labelling.
* $f$ is **permutation equivariant** if $f(PAP^\top, PX) = P\,f(A, X)$. Use: node-level outputs (one vector per node). Relabelling the input relabels the output *in the same way* and changes nothing else.

Every GNN layer we will meet is equivariant; a graph-level read-out (sum or mean over nodes) on top of an equivariant layer is invariant. In link prediction (our task) the score of the pair (drug $i$, disease $j$) must be the same whatever order the drugs and diseases are listed in — equivariance in both node sets.

**Why an MLP on the flattened adjacency matrix is not equivariant.** Write $\text{vec}(A)$ for the 16 numbers of a 4 × 4 adjacency matrix laid out in a row. An MLP computes $\tanh(\text{vec}(A)\,W)$ with a fixed weight matrix $W$, where each weight is attached to a *position* in the vector ("the entry in row 2, column 3"). Relabelling the nodes moves entries to different positions, so they meet different weights, and the outputs change in an arbitrary way. The code below checks both claims numerically.

```python
# file: c3_03_equivariance.py
import numpy as np
rng = np.random.default_rng(0)

def gcn_layer(A, X, W):
    At = A + np.eye(len(A))
    d = At.sum(1) ** -0.5
    return np.maximum((d[:, None] * At * d[None, :]) @ X @ W, 0)

A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], dtype=float)
X = rng.normal(size=(4, 3))
W = rng.normal(size=(3, 2))

perm = np.array([2, 0, 3, 1])              # new order: old node 2 first, ...
P = np.eye(4)[perm]                         # permutation matrix, (P X)[i] = X[perm[i]]

H = gcn_layer(A, X, W)
H_perm = gcn_layer(P @ A @ P.T, P @ X, W)
print("equivariant  f(PAP^T, PX) == P f(A, X):", np.allclose(H_perm, P @ H))

# a graph-level readout (sum over nodes) is invariant
print("invariant    sum f(PAP^T, PX) == sum f(A, X):",
      np.allclose(H_perm.sum(0), H.sum(0)))

# A "naive" MLP that reads the flattened adjacency matrix is NOT equivariant
W_mlp = rng.normal(size=(16, 4))
naive = lambda A: np.tanh(A.reshape(-1) @ W_mlp)       # one score per node slot
print("naive MLP scores, original order :", np.round(naive(A), 3))
print("naive MLP scores, permuted input :", np.round(naive(P @ A @ P.T), 3))
print("permuted original scores (wanted):", np.round(P @ naive(A), 3))
```

Output:

```text
equivariant  f(PAP^T, PX) == P f(A, X): True
invariant    sum f(PAP^T, PX) == sum f(A, X): True
naive MLP scores, original order : [ 0.992 -0.968  0.148  0.956]
naive MLP scores, permuted input : [ 0.848  0.981  0.996 -0.124]
permuted original scores (wanted): [ 0.148  0.992  0.956 -0.968]
```

The GCN layer passes both tests; the naive MLP gives a completely different set of numbers for the same graph.

### 2.3 The inductive bias of a GNN: locality + weight sharing

A CNN encodes two beliefs about images: (i) what matters for a pixel is mostly nearby (**locality**) and (ii) the same pattern means the same thing wherever it appears (**weight sharing**). A GNN encodes the graph versions of these beliefs:

* **Locality:** a node's new representation depends on its own state and its neighbours' states only.
* **Weight sharing:** the same transformation (same weight matrices) is used at every node, whatever its degree or position.
* **Permutation symmetry:** neighbours are an *unordered multiset*, combined by an order-independent function.

These three choices give us everything we asked for in Section 2.1. Weight sharing makes the parameter count independent of the number of nodes, so the same model can (in principle) run on 593 or 663 drugs.

---

## 3. The message-passing framework

### 3.1 The template

Almost every GNN in use — GCN, GraphSAGE, GIN, GAT, and the project's MV-HGAT — is an instance of one template, popularised as the *message passing neural network* (MPNN) by Gilmer et al. (2017). Let $h_i^{(l)} \in \mathbb{R}^{d_l}$ be the representation (embedding, hidden state) of node $i$ after layer $l$, with $h_i^{(0)} = x_i$, the input features. Let $\mathcal N(i)$ be the neighbours of $i$. One layer does three things:

$$
\begin{aligned}
\textbf{message:}\quad & m_{j\to i}^{(l)} = \psi^{(l)}\big(h_i^{(l)},\ h_j^{(l)},\ e_{ij}\big) && \text{for every } j \in \mathcal N(i)\\
\textbf{aggregate:}\quad & m_i^{(l)} = \bigoplus_{j\in\mathcal N(i)} m_{j\to i}^{(l)} && \bigoplus \in \{\textstyle\sum, \text{mean}, \max, \text{attention-weighted sum}, \dots\}\\
\textbf{update:}\quad & h_i^{(l+1)} = \phi^{(l)}\big(h_i^{(l)},\ m_i^{(l)}\big)
\end{aligned}
$$

* $\psi$ (the **message function**) says what neighbour $j$ tells $i$. It may depend on the edge feature $e_{ij}$ (e.g. the similarity value).
* $\bigoplus$ (the **aggregation**) must be **permutation invariant**: it takes a multiset of messages and returns one vector, and it must not care about their order. It must also handle any number of messages.
* $\phi$ (the **update function**) combines the node's old state with the aggregated message — e.g. a linear layer + nonlinearity, possibly with a skip connection.

```
                 layer l                                   layer l+1
   h_j1 ──ψ──► m_j1→i ─┐
   h_j2 ──ψ──► m_j2→i ─┼──► ⊕ (sum/mean/max/attention) ──► m_i ──┐
   h_j3 ──ψ──► m_j3→i ─┘                                         ├──► φ ──► h_i'
   h_i ──────────────────────────────────────────────────────────┘
```

All nodes are updated *simultaneously* from the layer-$l$ states, and the same $\psi, \phi$ are used at every node (weight sharing).

**The template for the models in this course:**

| Model | message $\psi$ | aggregate $\bigoplus$ | update $\phi$ |
|---|---|---|---|
| GCN (Kipf & Welling 2017) | $\frac{1}{\sqrt{\tilde d_i \tilde d_j}} W h_j$ | sum over $\mathcal N(i)\cup\{i\}$ | $\sigma(\cdot)$ |
| GraphSAGE-mean (Hamilton et al. 2017) | $h_j$ | mean over (sampled) $\mathcal N(i)$ | $\sigma(W[h_i \,\Vert\, m_i])$, then $\ell_2$-normalise |
| GIN (Xu et al. 2019) | $h_j$ | sum | $\text{MLP}((1+\epsilon)h_i + m_i)$ |
| GAT (Veličković et al. 2018; Unit C4) | $\alpha_{ij} W h_j$, $\alpha$ learned | sum (weights sum to 1) | $\sigma(\cdot)$ |
| MV-HGAT `HeteroLayer` | per relation: GAT message | attention over relations (Unit C5) | $\text{ELU}(\text{LayerNorm}(z_i + W_{\text{skip}} h_i))$ |

### 3.2 Proof: sum, mean and max aggregation are permutation invariant

Let the neighbours of $i$ be listed as $j_1, \dots, j_n$ and let $\pi$ be any re-ordering of $\{1,\dots,n\}$.

* **Sum.** $\sum_{k=1}^n m_{j_{\pi(k)}} = \sum_{k=1}^n m_{j_k}$ because vector addition is commutative and associative: a finite sum may be reordered term by term without changing its value.
* **Mean.** $\frac1n \sum_k m_{j_{\pi(k)}} = \frac1n\sum_k m_{j_k}$: the sum is unchanged (above) and $n$ is unchanged by re-ordering.
* **Max** (element-wise). For each coordinate $c$, $\max_k m_{j_{\pi(k)},c} = \max_k m_{j_k,c}$ because the maximum of a finite set depends only on the set, not the listing order.

Each is a function of the **multiset** $\{\!\{m_{j_1}, \dots, m_{j_n}\}\!\}$ only. $\square$

(Not every reasonable-sounding aggregator is invariant. GraphSAGE's **LSTM aggregator** reads neighbours as a sequence, so it *does* depend on order; the authors patch this by feeding a random permutation each time. Attention-weighted sums are invariant because the weight of each neighbour is computed from that neighbour's own features, not from its position.)

### 3.3 Proof: a message-passing layer is permutation equivariant

Relabel nodes with a bijection $\pi$: new node $i$ is old node $\pi(i)$, so $h'_i = h_{\pi(i)}$, and $j$ is a neighbour of $i$ in the new graph iff $\pi(j)$ is a neighbour of $\pi(i)$ in the old graph. The new output at node $i$ is
$$
\phi\Big(h_{\pi(i)},\ \bigoplus_{j:\ \pi(j)\in\mathcal N(\pi(i))} \psi(h_{\pi(i)}, h_{\pi(j)})\Big)
= \phi\Big(h_{\pi(i)},\ \bigoplus_{k\in\mathcal N(\pi(i))} \psi(h_{\pi(i)}, h_k)\Big)
= \text{old output at node } \pi(i),
$$
where the middle step substitutes $k = \pi(j)$ (a bijection between the two neighbour sets) and uses the invariance of $\bigoplus$. So the output is the old output relabelled the same way: equivariance. In matrix form, for the GCN layer: $\widehat{PAP^\top} = P\hat A P^\top$ (degrees are permuted along with the nodes), hence
$$\sigma\big(P\hat A P^\top\, P X\, W\big) = \sigma\big(P \hat A X W\big) = P\,\sigma(\hat A X W),$$
because $P^\top P = I$ and an element-wise $\sigma$ commutes with row reordering. $\square$

### 3.4 Matrix form

When the message is a linear function of $h_j$ that is weighted by a fixed edge coefficient $c_{ij}$, and aggregation is a sum, a whole layer is one matrix product:
$$
h_i' = \sigma\Big(\sum_j c_{ij}\, W h_j\Big)\quad\Longleftrightarrow\quad H' = \sigma(C\,H\,W^\top)
$$
(with $H$ having nodes as rows; in code `nn.Linear` stores $W$ so that `lin(H) = H @ W.T`). $C$ is a **propagation matrix**: $C = \hat A$ for GCN, $C = D^{-1}A$ for mean aggregation, $C = A + (1+\epsilon)I$ for GIN, $C = $ the attention matrix $[\alpha_{ij}]$ for GAT. This is why the project can implement GNNs with nothing but dense matrix multiplications, and why "GNN layer = (weighted) neighbourhood averaging + a shared linear map + a nonlinearity" is a good mental model.

### 3.5 The receptive field: $L$ layers see $L$ hops

**Claim.** In an $L$-layer message-passing GNN, $h_i^{(L)}$ depends only on the input features of nodes within graph distance $L$ of $i$ (its **$L$-hop neighbourhood**), and in general on all of them.

**Proof by induction.** $L=0$: $h_i^{(0)} = x_i$ depends on $i$ only (distance 0). Step: $h_i^{(l+1)}$ is a function of $h_i^{(l)}$ and $\{h_j^{(l)}: j \in \mathcal N(i)\}$. By hypothesis $h_j^{(l)}$ depends on nodes within distance $l$ of $j$, and every such node is within distance $l+1$ of $i$ (go $i \to j$, then $\le l$ more steps). So $h_i^{(l+1)}$ depends only on nodes within distance $l+1$. $\square$

In matrix form, with self-loops: $(\hat A^L)_{ik} \neq 0$ exactly when there is a walk of length $\le L$ from $i$ to $k$, because the self-loops let a walk "wait". Ignoring nonlinearities, $H^{(L)} \approx \hat A^L X W$, so node $i$ mixes in exactly the nodes $k$ with $(\hat A^L)_{ik} \ne 0$.

A useful picture is the **computation tree** (unrolled neighbourhood) of a node. For node 3 of our running 4-node graph (edges 0–1, 0–2, 1–2, 2–3), a 2-layer GNN computes:

```
                        h_3^(2)
              ┌────────────┴─────────────┐
           h_3^(1)                     h_2^(1)
         ┌────┴────┐          ┌──────┬───┴───┬──────┐
       x_3       x_2        x_2    x_0     x_1    x_3        (self-loops included)
```

After one layer node 3 knows only about itself and node 2; after two layers it has heard from nodes 0 and 1 *through* node 2.

**In the project.** MV-HGAT has 2 layers. Through a drug-view relation, a drug hears from its similar drugs and their similar drugs. Through the `assoc>drug` / `assoc>disease` relations, the 2-hop path *drug → disease → drug* means "drugs that treat the same diseases as me" — classic collaborative filtering — and *disease → drug → disease* means "diseases treated by the same drugs as me". That 2-hop signal is the reason to use 2 layers rather than 1.

---

## 4. The spectral motivation (in brief)

GCN was not invented as "average your neighbours"; it was *derived* as a cheap approximation of a convolution defined through the graph's eigenvectors. You do not need spectral graph theory to *use* a GCN, but it explains the strange-looking normalisation, why GCN is a **low-pass filter**, and why depth leads to over-smoothing. We go through it at the level of a graduate course, briefly.

### 4.1 The Laplacian measures how "wiggly" a signal is

A **graph signal** is one number per node, $x \in \mathbb R^N$ (e.g. one feature column). For an unweighted graph with adjacency $A$ and degree matrix $D = \text{diag}(d_1,\dots,d_N)$, the **combinatorial Laplacian** is $L = D - A$ and the **symmetric normalised Laplacian** (Unit A2) is
$$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}.$$
Both are symmetric and positive semi-definite, thanks to the identities
$$x^\top (D - A) x = \sum_{(i,j)\in E} (x_i - x_j)^2, \qquad x^\top L_{\text{sym}}\, x = \sum_{(i,j)\in E}\Big(\frac{x_i}{\sqrt{d_i}} - \frac{x_j}{\sqrt{d_j}}\Big)^2 .$$
(Expand the square and collect the $x_i^2$ terms: each node appears in $d_i$ edges, which produces the $D$ part; the cross terms produce $-A$.) The quadratic form is small when neighbours carry similar values: it is the graph's measure of **non-smoothness**, also called the **Dirichlet energy**.

### 4.2 Eigenvectors are graph "frequencies"; the graph Fourier transform

Since $L_{\text{sym}}$ is real symmetric it has an orthonormal eigenbasis: $L_{\text{sym}} = U \Lambda U^\top$ with $U^\top U = I$ and eigenvalues $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_N \le 2$. For an eigenvector $u_k$, $u_k^\top L_{\text{sym}} u_k = \lambda_k$, so **small eigenvalues belong to smooth eigenvectors** (low frequency: neighbours agree) and **large eigenvalues to oscillating eigenvectors** (high frequency: neighbours disagree). This is the graph analogue of sines and cosines of increasing frequency.

The **graph Fourier transform** (GFT) of a signal is its coordinates in this basis, and the inverse transform rebuilds the signal:
$$\hat x = U^\top x,\qquad x = U\hat x = \sum_k \hat x_k u_k .$$

A **spectral filter** rescales each frequency by a response $g(\lambda)$:
$$y = U\, g(\Lambda)\, U^\top x,\qquad g(\Lambda) = \text{diag}(g(\lambda_1),\dots,g(\lambda_N)).$$
A *low-pass* filter keeps small $\lambda$ (smooth parts) and damps large $\lambda$.

**Worked example.** Take the running 4-node graph (edges 0–1, 0–2, 1–2, 2–3; degrees $2,2,3,1$) and the signal $x = (1, 0, 1, 0)$.

```python
# file: c3_02_spectral.py
import numpy as np
np.set_printoptions(precision=3, suppress=True)

A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], dtype=float)
d = A.sum(1)
Dm = np.diag(d ** -0.5)
L = np.eye(4) - Dm @ A @ Dm                 # normalised Laplacian
lam, U = np.linalg.eigh(L)                  # L = U diag(lam) U^T
print("eigenvalues of L:", lam)
print("eigenvectors (columns), sign-fixed:")
U = U * np.sign(U[0])                       # make first entry positive for display
print(U)

x = np.array([1., 0., 1., 0.])              # a signal on the nodes
x_hat = U.T @ x                             # graph Fourier transform
print("x_hat = U^T x :", x_hat)
print("inverse GFT   :", U @ x_hat)

# a low-pass filter g(lam) = 1 - lam/2 applied in the spectral domain ...
g = 1 - lam / 2
y_spec = U @ (g * x_hat)
# ... is the same as multiplying by the polynomial (I - L/2) in the node domain
y_node = (np.eye(4) - L / 2) @ x
print("filtered (spectral):", y_spec)
print("filtered (node)    :", y_node)

# First-order Chebyshev with lambda_max = 2, theta = theta0 = -theta1
M = np.eye(4) + Dm @ A @ Dm
print("eig(I + D^-1/2 A D^-1/2):", np.linalg.eigvalsh(M))
At = A + np.eye(4)
Dt = np.diag(At.sum(1) ** -0.5)
A_hat = Dt @ At @ Dt
print("eig(A_hat) renormalised :", np.linalg.eigvalsh(A_hat))
for name, P in [("I + D^-1/2AD^-1/2", M), ("A_hat", A_hat)]:
    v = x.copy()
    for _ in range(10):
        v = P @ v
    print(f"10 applications of {name:18s}: norm = {np.linalg.norm(v):9.3f}")
```

Output:

```text
eigenvalues of L: [-0.     0.771  1.5    1.729]
eigenvectors (columns), sign-fixed:
[[ 0.5    0.436  0.707  0.244]
 [ 0.5    0.436 -0.707  0.244]
 [ 0.612 -0.29   0.    -0.736]
 [ 0.354 -0.732 -0.     0.583]]
x_hat = U^T x : [ 1.112  0.146  0.707 -0.491]
inverse GFT   : [1. 0. 1. 0.]
filtered (spectral): [0.704 0.454 0.704 0.289]
filtered (node)    : [0.704 0.454 0.704 0.289]
eig(I + D^-1/2 A D^-1/2): [0.271 0.5   1.229 2.   ]
eig(A_hat) renormalised : [-0.148  0.     0.564  1.   ]
10 applications of I + D^-1/2AD^-1/2 : norm =  1139.070
10 applications of A_hat             : norm =     1.077
```

How to read it (the `-0.` is a tiny negative rounding error of an exact 0):

* $\lambda_1 = 0$ with eigenvector $(0.5, 0.5, 0.612, 0.354)$. This is $D^{1/2}\mathbf 1 / \lVert D^{1/2}\mathbf 1\rVert = (\sqrt2,\sqrt2,\sqrt3,1)/\sqrt8$: the "DC component", perfectly smooth in the normalised sense.
* $\lambda_3 = 1.5$ has eigenvector $(0.707, -0.707, 0, 0)$: "node 0 minus node 1", a local oscillation between two nodes that share all their neighbours.
* The low-pass filter $g(\lambda) = 1-\lambda/2$ multiplies the four Fourier coefficients by $1, 0.614, 0.25, 0.135$; the result is the same whether computed spectrally or as the matrix polynomial $I - L/2$ in the node domain. **A polynomial in $L$ is a spectral filter that can be applied without any eigen-decomposition.**

### 4.3 Why not learn $g$ directly? Three problems

A first "spectral GNN" (Bruna et al. 2014) learned the $N$ numbers $g(\lambda_k)$ directly. That has three problems:

1. **Cost:** you need the eigen-decomposition ($O(N^3)$) and two dense $N\times N$ products per layer.
2. **No locality:** an arbitrary $g$ mixes every node with every node.
3. **No transfer:** the basis $U$ belongs to *one* graph; filters learned on Fdataset's drug graph mean nothing on Cdataset's.

### 4.4 Polynomial filters and Chebyshev polynomials (ChebNet)

Choose $g$ to be a polynomial of degree $K$: $g(\lambda) = \sum_{k=0}^K \theta_k \lambda^k$. Then $U g(\Lambda) U^\top = \sum_k \theta_k L^k$, which fixes all three problems: only $K+1$ parameters, no eigenvectors, and **$K$-localised** — $(L^k)_{ij} = 0$ whenever $i$ and $j$ are more than $k$ hops apart, by the same walk-counting argument as Section 3.5.

Defferrard et al. (2016, "ChebNet") used the **Chebyshev polynomials** $T_0(x) = 1$, $T_1(x) = x$, $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$, which form a numerically stable basis on $[-1, 1]$. Since the eigenvalues of $L_{\text{sym}}$ lie in $[0, \lambda_{\max}] \subseteq [0, 2]$, they are first mapped to $[-1,1]$ by $\tilde\Lambda = \frac{2}{\lambda_{\max}}\Lambda - I$, i.e. $\tilde L = \frac{2}{\lambda_{\max}} L_{\text{sym}} - I$. The filter is
$$ g_{\theta} \star x \;\approx\; \sum_{k=0}^{K} \theta_k\, T_k(\tilde L)\, x, $$
computed with the recursion $\bar x_k = 2\tilde L \bar x_{k-1} - \bar x_{k-2}$, so the cost is $K$ sparse matrix–vector products: $O(K|E|)$.

### 4.5 Deriving the GCN layer (Kipf & Welling 2017)

Kipf & Welling made three simplifications.

**Step 1 — first order, $K = 1$.** Keep only $T_0$ and $T_1$:
$$ g_\theta \star x \approx \theta_0 x + \theta_1 \tilde L x. $$

**Step 2 — assume $\lambda_{\max} \approx 2$** (the upper bound; the network's weights can absorb the scale). Then $\tilde L = L_{\text{sym}} - I = -D^{-1/2} A D^{-1/2}$ and
$$ g_\theta \star x \approx \theta_0\, x - \theta_1\, D^{-1/2} A D^{-1/2}\, x. $$
This is now a 1-hop operation: each node combines itself ($\theta_0$) with a degree-normalised sum of its neighbours ($\theta_1$).

**Step 3 — tie the two parameters**, $\theta = \theta_0 = -\theta_1$, to reduce overfitting:
$$ g_\theta \star x \approx \theta\,\big(I + D^{-1/2} A D^{-1/2}\big)\, x. $$
In spectral terms the response is $\theta(2 - \lambda)$: a **low-pass filter** that keeps the smooth component ($\lambda = 0$, gain $2\theta$) and kills the highest possible frequency ($\lambda = 2$, gain 0). (Our worked example used $1-\lambda/2$, which is exactly this filter with $\theta = \tfrac12$.)

**Step 4 — the renormalisation trick.** The eigenvalues of $I + D^{-1/2} A D^{-1/2}$ are $2 - \lambda_k \in [0, 2]$ — in our example $0.271, 0.5, 1.229, 2$. Stacking many layers multiplies by this matrix repeatedly, so components with eigenvalue 2 grow like $2^L$ and those near 0 vanish: **numerical instability and exploding/vanishing signals and gradients**. The output above shows it: ten applications blow the norm of $x$ up to 1139. Kipf & Welling therefore replaced
$$ I + D^{-1/2} A D^{-1/2} \;\longrightarrow\; \hat A = \tilde D^{-1/2} \tilde A\, \tilde D^{-1/2},\qquad \tilde A = A + I,\quad \tilde D_{ii} = \sum_j \tilde A_{ij} = d_i + 1. $$
In words: **add self-loops first, then normalise**. The eigenvalues of $\hat A$ lie in $(-1, 1]$ (our example: $-0.148, 0, 0.564, 1$), and ten applications leave the norm at 1.077.

*Why $(-1, 1]$?* $\hat A = \tilde D^{-1/2}\tilde A \tilde D^{-1/2}$ is similar to $\tilde D^{-1}\tilde A = \tilde D^{-1/2}\hat A\,\tilde D^{1/2}$, the transition matrix of a random walk (Unit C2): non-negative with rows summing to 1, so every eigenvalue satisfies $|\mu| \le 1$ (any row-stochastic matrix has spectral radius 1). $\mu = 1$ is attained with eigenvector $\tilde D^{1/2}\mathbf 1$. And $\mu = -1$ would require a bipartite component, which the self-loops rule out. (Wu et al. 2019, "SGC", also show that adding self-loops *shrinks* the largest eigenvalue of the normalised Laplacian, making the filter more strongly low-pass.)

**Step 5 — many channels and many filters.** With an input feature matrix $X \in \mathbb R^{N\times C}$ ($C$ channels) and $F$ output filters, each output channel is a sum over input channels of filtered signals; collecting the $\theta$s into a matrix $\Theta \in \mathbb R^{C\times F}$:
$$ Z = \hat A\, X\, \Theta. $$
Adding a nonlinearity and stacking gives the **GCN layer**:
$$ \boxed{\,H^{(l+1)} = \sigma\big(\hat A\, H^{(l)}\, W^{(l)}\big),\qquad \hat A = \tilde D^{-1/2}(A + I)\tilde D^{-1/2}\,} $$
and the 2-layer model of the paper for node classification:
$$ Z = \mathrm{softmax}\big(\hat A\ \mathrm{ReLU}(\hat A X W^{(0)})\ W^{(1)}\big). $$

Node-wise, the same layer reads
$$ h_i^{(l+1)} = \sigma\Big( \sum_{j \in \mathcal N(i)\cup\{i\}} \frac{1}{\sqrt{\tilde d_i\,\tilde d_j}}\; W^{(l)\top} h_j^{(l)} \Big). $$

Its cost is $O(|E|\,C\,F)$ with a sparse $\hat A$ — linear in the number of edges.

**The take-away.** A GCN layer is (i) a fixed low-pass graph filter $\hat A$ that smooths features over 1-hop neighbourhoods, followed by (ii) a learned, node-shared linear map $W$ and (iii) a nonlinearity. Everything about over-smoothing in Section 8 follows from (i).

---

## 5. The GCN layer in detail

### 5.1 Why *symmetric* normalisation? Three choices of propagation matrix

| Propagation matrix | Node-wise meaning | Row sums | Comment |
|---|---|---|---|
| $\tilde A = A + I$ (sum) | $\sum_{j} h_j$ | $\tilde d_i$ | Scale grows with degree; hub nodes explode; used by GIN (with an MLP to fix scale) |
| $\tilde D^{-1}\tilde A$ (mean, "random-walk") | $\frac{1}{\tilde d_i}\sum_j h_j$ | 1 | Plain average; each neighbour counts the same regardless of *its* degree |
| $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ (symmetric, GCN) | $\sum_j \frac{h_j}{\sqrt{\tilde d_i \tilde d_j}}$ | $\approx 1$ | Down-weights messages from **high-degree** neighbours; symmetric matrix |

The symmetric choice has an intuitive reading: a neighbour $j$ that is connected to *everything* (a hub — e.g. a "promiscuous" drug that is similar to 94 others in `chem_cdk`) is less informative about any particular neighbour, so its message is shrunk by $1/\sqrt{\tilde d_j}$. It also keeps the matrix symmetric, which is what makes the spectral derivation work. The price is that row sums are not exactly 1 (see the worked example), so the scale of $h_i$ drifts slightly with degree.

**Weighted graphs.** Nothing in the derivation requires 0/1 edges. With a weighted adjacency $S$ (similarities), $\tilde D_{ii} = \sum_j \tilde S_{ij}$ and $\hat S = \tilde D^{-1/2}\tilde S\tilde D^{-1/2}$. The NIMCGCN baseline does exactly this with kNN-masked similarities (Section 12).

### 5.2 Worked example by hand: one GCN layer on four drugs

Four drugs, D0–D3, in a chemical kNN graph with edges D0–D1, D0–D2, D1–D2, D2–D3. Each drug has two input features, and the layer maps 2 → 2 features.

$$
A = \begin{pmatrix}0&1&1&0\\1&0&1&0\\1&1&0&1\\0&0&1&0\end{pmatrix},\quad
X = \begin{pmatrix}1&0\\0&2\\1&0\\0&1\end{pmatrix},\quad
W = \begin{pmatrix}1&-1\\0.5&0.8\end{pmatrix}.
$$

**Step 1 — self-loops and degrees.** $\tilde A = A + I$. Row sums give $\tilde d = (3, 3, 4, 2)$.

**Step 2 — normalise.** $\hat A_{ij} = \tilde A_{ij}/\sqrt{\tilde d_i \tilde d_j}$:

* $\hat A_{00} = \hat A_{01} = \hat A_{11} = 1/\sqrt{3\cdot 3} = 1/3 \approx 0.3333$
* $\hat A_{02} = \hat A_{12} = 1/\sqrt{3\cdot4} = 1/\sqrt{12} \approx 0.2887$
* $\hat A_{22} = 1/4 = 0.25$, $\quad\hat A_{23} = 1/\sqrt{4\cdot 2} = 1/\sqrt8 \approx 0.3536$, $\quad\hat A_{33} = 1/2$

$$
\hat A \approx \begin{pmatrix}0.3333&0.3333&0.2887&0\\0.3333&0.3333&0.2887&0\\0.2887&0.2887&0.25&0.3536\\0&0&0.3536&0.5\end{pmatrix}
$$

**Step 3 — aggregate: $\hat A X$.** Row by row:

* D0: $\tfrac13(1,0) + \tfrac13(0,2) + 0.2887(1,0) = (0.6220,\ 0.6667)$
* D1: identical coefficients → $(0.6220,\ 0.6667)$
* D2: $0.2887(1,0) + 0.2887(0,2) + 0.25(1,0) + 0.3536(0,1) = (0.5387,\ 0.9309)$
* D3: $0.3536(1,0) + 0.5(0,1) = (0.3536,\ 0.5)$

**Step 4 — transform: $(\hat A X)W$.** For D0: $(0.6220\cdot1 + 0.6667\cdot0.5,\ 0.6220\cdot(-1) + 0.6667\cdot0.8) = (0.9553,\ -0.0887)$. Similarly D2 → $(1.0041,\ 0.2060)$ and D3 → $(0.6036,\ 0.0464)$.

**Step 5 — ReLU** sets the negative $-0.0887$ to 0.

```python
# file: c3_01_gcn_by_hand.py
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# 4 drugs; edges = "is among my most similar drugs"
A = np.array([[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]], dtype=float)

A_tilde = A + np.eye(4)                     # add self-loops
d_tilde = A_tilde.sum(1)                    # degrees incl. self-loop
D_inv_sqrt = np.diag(1 / np.sqrt(d_tilde))
A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt   # the GCN propagation matrix
print("degrees with self-loops:", d_tilde)
print("A_hat =\n", A_hat)

X = np.array([[1., 0.],                      # 2 input features per drug
              [0., 2.],
              [1., 0.],
              [0., 1.]])
W = np.array([[1.0, -1.0],                   # 2 -> 2 weight matrix
              [0.5,  0.8]])

AX = A_hat @ X
print("A_hat @ X =\n", AX)
Z = AX @ W
print("A_hat @ X @ W =\n", Z)
H = np.maximum(Z, 0)                         # ReLU
print("H1 = ReLU(A_hat X W) =\n", H)

# random-walk (mean) normalisation for comparison
A_rw = np.diag(1 / d_tilde) @ A_tilde
print("row-normalised D^-1 (A+I) =\n", A_rw)
print("row sums of A_hat :", A_hat.sum(1))
print("row sums of A_rw  :", A_rw.sum(1))
```

Output:

```text
degrees with self-loops: [3. 3. 4. 2.]
A_hat =
 [[0.3333 0.3333 0.2887 0.    ]
 [0.3333 0.3333 0.2887 0.    ]
 [0.2887 0.2887 0.25   0.3536]
 [0.     0.     0.3536 0.5   ]]
A_hat @ X =
 [[0.622  0.6667]
 [0.622  0.6667]
 [0.5387 0.9309]
 [0.3536 0.5   ]]
A_hat @ X @ W =
 [[ 0.9553 -0.0887]
 [ 0.9553 -0.0887]
 [ 1.0041  0.206 ]
 [ 0.6036  0.0464]]
H1 = ReLU(A_hat X W) =
 [[0.9553 0.    ]
 [0.9553 0.    ]
 [1.0041 0.206 ]
 [0.6036 0.0464]]
row-normalised D^-1 (A+I) =
 [[0.3333 0.3333 0.3333 0.    ]
 [0.3333 0.3333 0.3333 0.    ]
 [0.25   0.25   0.25   0.25  ]
 [0.     0.     0.5    0.5   ]]
row sums of A_hat : [0.9553 0.9553 1.1809 0.8536]
row sums of A_rw  : [1. 1. 1. 1.]
```

**Three lessons hidden in this tiny example.**

1. **D0 and D1 now have identical embeddings although their inputs differed** ($(1,0)$ vs $(0,2)$). They have the same closed neighbourhood $\{0,1,2\}$ and the same degree, so $\hat A$ has identical rows 0 and 1, and any GCN layer maps them to the same vector. In spectral language, $e_0 - e_1$ is an eigenvector of $\hat A$ with eigenvalue 0 (the "0" in `eig(A_hat)` above): the GCN filter *deletes* that frequency entirely. This is over-smoothing in miniature (Section 8). A skip connection that adds $W_{\text{skip}} h_i$ — as `HeteroLayer` does — would keep them apart.
2. **The order $(\hat A X) W = \hat A (X W)$ is free** (associativity). In code we usually compute $XW$ first because it is cheaper when the output width is smaller than the input width.
3. **Symmetric normalisation does not make rows sum to 1** (D2's row sums to 1.18, D3's to 0.85); mean normalisation does.

### 5.3 Training a GCN: semi-supervised node classification

The original paper's setting is **semi-supervised node classification**: one graph, every node has features, a *few* nodes have labels, and we want labels for the rest. Loss = cross-entropy on the labelled nodes only; the unlabelled nodes still participate in message passing, which is how label information flows to them. The paper used a 2-layer GCN, 16 hidden units, dropout 0.5, L2 regularisation $5\cdot10^{-4}$, Adam with learning rate 0.01, 200 epochs — the values used below.

Our toy graph is a **stochastic block model** (SBM): 60 nodes in two communities, edge probability 0.20 inside a community and 0.02 across. Each node has 8 features of pure noise, plus a weak class signal ($\pm1$) added to feature 0. Only **3 nodes per class are labelled**. We compare the GCN with *the very same network* whose $\hat A$ is replaced by the identity — i.e. an MLP that ignores the graph.

```python
# file: c3_04_gcn_node_classification.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---- a toy graph: 2 communities of 30 nodes (a "stochastic block model") ----
n_per, n_cls = 30, 2
y = np.repeat(np.arange(n_cls), n_per)                     # labels 0..0,1..1
N = len(y)
same = y[:, None] == y[None, :]
prob = np.where(same, 0.20, 0.02)                          # dense inside, sparse across
upper = np.triu(rng.random((N, N)) < prob, 1)
A = (upper | upper.T).astype(np.float32)                   # symmetric, no self-loops
X = rng.normal(size=(N, 8)).astype(np.float32)             # noisy features ...
X[:, 0] += np.where(y == 1, 1.0, -1.0)                     # ... with a weak class signal
print(f"nodes={N}, edges={int(A.sum() // 2)}, "
      f"edges inside a community={int((A * same).sum() // 2)}")

# 3 labelled nodes per class, the rest are test nodes
train_idx = np.concatenate([np.where(y == c)[0][:3] for c in range(n_cls)])
test_mask = np.ones(N, bool); test_mask[train_idx] = False

def normalise(A):
    """A_hat = D~^-1/2 (A + I) D~^-1/2  (Kipf & Welling renormalisation trick)."""
    A = A + torch.eye(A.shape[0])
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]

class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=True)
    def forward(self, A_hat, H):
        return A_hat @ self.lin(H)          # (A_hat H) W = A_hat (H W): order is free

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.l1, self.l2 = GCNLayer(d_in, d_hid), GCNLayer(d_hid, d_out)
        self.dropout = dropout
    def forward(self, A_hat, X):
        h = F.dropout(X, self.dropout, self.training)
        h = F.relu(self.l1(A_hat, h))
        h = F.dropout(h, self.dropout, self.training)
        return self.l2(A_hat, h)            # logits

class MLP(nn.Module):                        # same model with A_hat = I (no graph)
    def __init__(self, d_in, d_hid, d_out, dropout=0.5):
        super().__init__()
        self.net = GCN(d_in, d_hid, d_out, dropout)
    def forward(self, A_hat, X):
        return self.net(torch.eye(X.shape[0]), X)

def run(model_cls, seed):
    torch.manual_seed(seed)
    At, Xt, yt = normalise(torch.tensor(A)), torch.tensor(X), torch.tensor(y)
    model = model_cls(8, 16, n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(200):
        model.train()
        loss = F.cross_entropy(model(At, Xt)[train_idx], yt[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(At, Xt).argmax(1).numpy()
    return (pred[test_mask] == y[test_mask]).mean()

for name, cls in [("MLP (ignores graph)", MLP), ("2-layer GCN", GCN)]:
    accs = [run(cls, s) for s in range(5)]
    print(f"{name:20s} test accuracy over 5 seeds: {np.mean(accs):.3f} +- {np.std(accs):.3f}")
```

Output:

```text
nodes=60, edges=215, edges inside a community=194
MLP (ignores graph)  test accuracy over 5 seeds: 0.544 +- 0.025
2-layer GCN          test accuracy over 5 seeds: 0.959 +- 0.007
```

With six labels and noisy features the MLP is barely better than a coin; the GCN reaches 96 % because 90 % of edges (194/215) stay inside a community, so averaging over neighbours cancels the feature noise and spreads the few labels. This is **homophily** at work: *connected nodes tend to be alike*. GCNs are strong exactly when the graph is homophilous with respect to the target — and the whole premise of similarity-based drug repositioning ("similar drugs treat similar diseases") is a homophily assumption.

> **Note on reproducibility.** The numbers above come from CPU runs with the seeds shown. On another machine or PyTorch version the last digit can differ slightly; the qualitative picture will not.

### 5.4 Parameters do not depend on the graph size

A GCN layer from $d_\text{in}$ to $d_\text{out}$ features has $d_\text{in} d_\text{out} + d_\text{out}$ parameters (weights + bias), whether the graph has 60 nodes or 60 million. This is weight sharing again — and it is why, *architecturally*, a GCN can be applied to a new graph (Section 9 explains why that is not the whole story).

---

## 6. GraphSAGE: sampling, aggregators and inductive learning

### 6.1 The problem GraphSAGE solves

Hamilton, Ying & Leskovec (2017) wanted node embeddings for graphs that **keep growing** (new Reddit posts, new papers) and are **huge** (hundreds of thousands of nodes, millions of edges). Two features of the original GCN get in the way:

* It is trained **full-batch** on one fixed graph: every epoch multiplies by the whole $\hat A$.
* Its symmetric normalisation and the setup of the experiments tie it to the training graph: the paper's framing is *transductive* (Section 9).

GraphSAGE ("SAmple and aggreGatE") reframes a layer as a *function a node applies to its neighbourhood*, so it can be applied to any node, including one that did not exist during training — **inductive** learning.

### 6.2 The algorithm

For each layer $k = 1, \dots, K$ and each node $v$:
$$
\begin{aligned}
h_{\mathcal N(v)}^{(k)} &= \text{AGGREGATE}_k\big(\{\!\{ h_u^{(k-1)} : u \in \mathcal S_k(v) \}\!\}\big) \\
h_v^{(k)} &= \sigma\Big(W^{(k)} \big[\, h_v^{(k-1)} \,\Vert\, h_{\mathcal N(v)}^{(k)} \big]\Big) \\
h_v^{(k)} &\leftarrow h_v^{(k)} / \lVert h_v^{(k)} \rVert_2
\end{aligned}
$$

* $\mathcal S_k(v)$ is a **fixed-size uniform sample** of $v$'s neighbours (with replacement if the degree is smaller). The paper used $K = 2$ with $S_1 = 25$, $S_2 = 10$.
* $\Vert$ is **concatenation**. Keeping the node's own state separate from its neighbours' (rather than mixing it in as a self-loop, as GCN does) acts like a skip connection: the model can tell "me" from "my neighbourhood". (Compare with lesson 1 of Section 5.2.)
* The $\ell_2$ normalisation puts every embedding on the unit sphere.

**Why sampling?** With full neighbourhoods, a 2-layer computation tree for one node touches all its 2-hop neighbours — in a social network with average degree 100, that is about $10^4$ nodes per target node, and the count explodes with depth. With samples, the tree has at most $S_1 S_2 = 250$ leaves: constant memory per target node, which enables **mini-batch training**. Sampling also acts as a regulariser (a cousin of DropEdge, Section 8.6).

### 6.3 The aggregators

| Aggregator | Formula | Notes |
|---|---|---|
| **Mean** | $\frac{1}{\lvert\mathcal S(v)\rvert}\sum_{u} h_u$ | Simple, invariant. The "GCN variant" instead averages over $\{v\}\cup\mathcal S(v)$ and skips the concatenation — nearly a GCN layer. |
| **Max-pooling** | $\max_u \sigma(W_{\text{pool}} h_u + b)$ (element-wise) | Each neighbour first goes through a small neural layer; the max picks, per feature, the most "activated" neighbour. Invariant. |
| **LSTM** | $\text{LSTM}(h_{u_{\pi(1)}}, \dots, h_{u_{\pi(n)}})$ | Most expressive in practice, but **not** permutation invariant; applied to a random permutation $\pi$ of the neighbours each time. |

In the paper the LSTM and pooling aggregators were the strongest on its benchmarks, with mean close behind.

### 6.4 Code: mean and max-pool layers, sampling, and an inductive test

The script below (i) checks numerically that both aggregators are permutation invariant, (ii) trains a 2-layer GraphSAGE-mean model on one random graph $G_a$ with **≤ 5 sampled neighbours per node per epoch**, and (iii) applies the trained model *without retraining* to a **new graph $G_b$** with new nodes (generated from the same process). An MLP baseline gets the same treatment.

```python
# file: c3_05_graphsage.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_sbm(seed, n_per=30, p_in=0.20, p_out=0.02, signal=1.0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_per)
    same = y[:, None] == y[None, :]
    up = np.triu(rng.random((len(y), len(y))) < np.where(same, p_in, p_out), 1)
    A = (up | up.T).astype(np.float32)
    X = rng.normal(size=(len(y), 8)).astype(np.float32)
    X[:, 0] += np.where(y == 1, signal, -signal)
    return torch.tensor(A), torch.tensor(X), torch.tensor(y)

def sample_neighbours(A, S, gen):
    """Keep at most S random neighbours per node (GraphSAGE-style sampling)."""
    scores = torch.rand(A.shape, generator=gen) * A        # random score on real edges
    k = min(S, A.shape[1])
    top = scores.topk(k, dim=1)
    M = torch.zeros_like(A)
    M.scatter_(1, top.indices, (top.values > 0).float())   # never keep a non-edge
    return M

class SAGEMean(nn.Module):
    """h_i' = ReLU( W [ h_i || mean_{j in N(i)} h_j ] ),  then L2-normalise."""
    def __init__(self, d_in, d_out, act=True):
        super().__init__()
        self.lin = nn.Linear(2 * d_in, d_out)
        self.act = act
    def forward(self, M, H):
        deg = M.sum(1, keepdim=True).clamp(min=1)          # isolated node -> mean of nothing = 0
        neigh = (M @ H) / deg
        out = self.lin(torch.cat([H, neigh], 1))
        if self.act:
            out = F.normalize(F.relu(out), dim=1)
        return out

class SAGEMaxPool(nn.Module):
    """h_i' = ReLU( W [ h_i || max_{j in N(i)} ReLU(W_pool h_j + b) ] )."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.pool = nn.Linear(d_in, d_in)
        self.lin = nn.Linear(2 * d_in, d_out)
    def forward(self, M, H):
        P = F.relu(self.pool(H))                                   # N, d
        masked = P[None, :, :].masked_fill(M[:, :, None] == 0, float("-inf"))
        neigh = masked.max(1).values                               # N, d
        neigh = torch.where(torch.isinf(neigh), torch.zeros_like(neigh), neigh)
        return F.relu(self.lin(torch.cat([H, neigh], 1)))

# 1) permutation invariance of the aggregators, checked numerically
torch.manual_seed(0)
H = torch.randn(5, 3)
M = torch.tensor([[0, 1, 1, 1, 0]] + [[0] * 5] * 4, dtype=torch.float32)  # node 0 has nbrs 1,2,3
perm = torch.tensor([0, 3, 1, 2, 4])                    # shuffle nbrs 1,2,3 among themselves
Mp, Hp = M[perm][:, perm], H[perm]
lay = SAGEMean(3, 2)
print("mean aggregator, node 0 :", torch.allclose(lay(M, H)[0], lay(Mp, Hp)[0]))
lay2 = SAGEMaxPool(3, 2)
print("max-pool aggregator, node 0:", torch.allclose(lay2(M, H)[0], lay2(Mp, Hp)[0]))

# 2) inductive learning: train on graph G_a, apply to a NEW graph G_b
class SAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = SAGEMean(8, 16), SAGEMean(16, 2, act=False)
    def forward(self, M, X):
        return self.l2(M, self.l1(M, X))

A_a, X_a, y_a = make_sbm(seed=1)            # training graph (all labels known)
A_b, X_b, y_b = make_sbm(seed=2)            # unseen graph, unseen nodes
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
model = SAGE()
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(100):
    M = sample_neighbours(A_a, S=5, gen=gen)   # fresh sample of <= 5 neighbours each epoch
    loss = F.cross_entropy(model(M, X_a), y_a)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc_a = (model(A_a, X_a).argmax(1) == y_a).float().mean().item()
    acc_b = (model(A_b, X_b).argmax(1) == y_b).float().mean().item()
print(f"accuracy on training graph G_a: {acc_a:.3f}")
print(f"accuracy on NEW graph G_b     : {acc_b:.3f}   (never seen during training)")

# baseline: the same network with every neighbourhood emptied = a plain MLP
torch.manual_seed(0)
mlp = SAGE()
opt = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
Z_a, Z_b = torch.zeros_like(A_a), torch.zeros_like(A_b)
for epoch in range(100):
    loss = F.cross_entropy(mlp(Z_a, X_a), y_a)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    acc_b = (mlp(Z_b, X_b).argmax(1) == y_b).float().mean().item()
print(f"MLP baseline on NEW graph G_b : {acc_b:.3f}")
```

Output:

```text
mean aggregator, node 0 : True
max-pool aggregator, node 0: True
accuracy on training graph G_a: 1.000
accuracy on NEW graph G_b     : 0.783   (never seen during training)
MLP baseline on NEW graph G_b : 0.667
```

Points to notice:

* At **test time** we used the **full** neighbourhoods (`A_b`), not samples. This is standard: sampling is a training-time device; the mean over the full set is the quantity it estimates.
* The training-graph accuracy of 1.0 versus 0.78 on $G_b$ shows overfitting on a tiny graph, but the model *transfers*: it beats the MLP on a graph it has never seen. This is possible only because nothing in the model refers to a particular node — the input is a node's own features plus an aggregate of its neighbours' features, both of which exist for new nodes.
* `deg.clamp(min=1)` makes an isolated node's neighbour-mean a zero vector instead of a division by zero; the max-pool layer handles the same case by replacing $-\infty$ with 0. **Every aggregator needs a defined answer for an empty neighbourhood.** (Unit C4 shows how the project's attention layer handles this.)

---

## 7. How powerful is message passing? The WL test and GIN (brief)

### 7.1 The 1-Weisfeiler–Lehman colour refinement

How do you test whether two graphs are the same up to relabelling (**isomorphic**)? No polynomial-time algorithm is known in general, but a cheap heuristic, the **1-dimensional Weisfeiler–Lehman test (1-WL)**, works for most graphs:

1. Give every node the same colour (or a colour encoding its features).
2. Repeat: each node's new colour = a hash of *(its own colour, the multiset of its neighbours' colours)*.
3. Stop when the colouring no longer gets finer. If the two graphs' **histograms of colours** differ at any round, they are certainly not isomorphic. If they never differ, 1-WL "cannot tell" (they may or may not be isomorphic).

Look at step 2: it *is* message passing, with an injective (hash) aggregation of the neighbour multiset.

### 7.2 The theorem (Xu, Hu, Leskovec & Jegelka 2019)

* **Upper bound.** Any message-passing GNN maps two nodes (graphs) to different vectors *only if* 1-WL gives them different colours. Message passing can never distinguish more than 1-WL can.
* **Achievability.** A GNN is *as powerful as* 1-WL if its aggregate-and-update is **injective on multisets** (and so is its graph read-out).

**Sum** aggregation (followed by an MLP) can be injective on multisets of features; **mean** and **max** cannot. Mean only sees *proportions* and max only sees the *set of distinct values*:

```python
# file: c3_06_wl_gin.py
import numpy as np
import torch
import torch.nn as nn

# 1) Which aggregators can tell these neighbour multisets apart?
a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
pairs = {
    "{a,b} vs {a,a,b,b}": ([a, b], [a, a, b, b]),
    "{a,a,b} vs {a,b}  ": ([a, a, b], [a, b]),
    "{a} vs {a,a}      ": ([a], [a, a]),
}
for name, (m1, m2) in pairs.items():
    out = []
    for agg, f in [("sum", np.sum), ("mean", np.mean), ("max", np.max)]:
        same = np.allclose(f(m1, axis=0), f(m2, axis=0))
        out.append(f"{agg}: {'SAME' if same else 'different'}")
    print(name, " | ".join(out))

# 2) 1-WL colour refinement
def wl_colours(adj, iters=3):
    n = len(adj)
    col = [0] * n                                    # every node starts with the same colour
    for _ in range(iters):
        sig = [(col[i], tuple(sorted(col[j] for j in adj[i]))) for i in range(n)]
        table = {s: k for k, s in enumerate(sorted(set(sig)))}
        col = [table[s] for s in sig]
    return sorted(col)

hexagon = {i: [(i - 1) % 6, (i + 1) % 6] for i in range(6)}
two_triangles = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
print("WL colours, hexagon      :", wl_colours(hexagon))
print("WL colours, two triangles:", wl_colours(two_triangles))
print("WL can distinguish them?  ", wl_colours(hexagon) != wl_colours(two_triangles))

# 3) A GIN layer: h_i' = MLP( (1 + eps) h_i + sum_j h_j )
class GINLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Linear(d_out, d_out))
    def forward(self, A, H):
        return self.mlp((1 + self.eps) * H + A @ H)

def to_dense(adj):
    A = torch.zeros(len(adj), len(adj))
    for i, nb in adj.items():
        A[i, nb] = 1.0
    return A

torch.manual_seed(0)
gin = GINLayer(1, 4)
for name, g in [("hexagon", hexagon), ("two triangles", two_triangles)]:
    A = to_dense(g)
    readout = gin(A, torch.ones(6, 1)).sum(0)      # sum-pool all nodes -> graph vector
    print(f"GIN graph readout, {name:13s}:", readout.detach().numpy().round(4))

# 4) A pair that sum (GIN/WL) separates but a mean aggregator does not
path3 = {0: [1], 1: [0, 2], 2: [1]}
triangle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
print("WL separates path vs triangle?", wl_colours(path3) != wl_colours(triangle))
for name, g in [("path", path3), ("triangle", triangle)]:
    A = to_dense(g)
    H = torch.ones(3, 1)
    mean_agg = (A @ H) / A.sum(1, keepdim=True)          # mean of neighbours
    sum_agg = A @ H                                      # sum of neighbours
    print(f"{name:8s} mean-agg per node: {mean_agg.squeeze().tolist()}  "
          f"sum-agg per node: {sum_agg.squeeze().tolist()}")
```

Output:

```text
{a,b} vs {a,a,b,b} sum: different | mean: SAME | max: SAME
{a,a,b} vs {a,b}   sum: different | mean: different | max: SAME
{a} vs {a,a}       sum: different | mean: SAME | max: SAME
WL colours, hexagon      : [0, 0, 0, 0, 0, 0]
WL colours, two triangles: [0, 0, 0, 0, 0, 0]
WL can distinguish them?   False
GIN graph readout, hexagon      : [ 1.0069 -3.4128  6.096   1.8941]
GIN graph readout, two triangles: [ 1.0069 -3.4128  6.096   1.8941]
WL separates path vs triangle? True
path     mean-agg per node: [1.0, 1.0, 1.0]  sum-agg per node: [1.0, 2.0, 1.0]
triangle mean-agg per node: [1.0, 1.0, 1.0]  sum-agg per node: [2.0, 2.0, 2.0]
```

Read the output in three parts:

* **Aggregators.** Sum separates all three pairs; mean fails whenever the *proportions* match; max fails whenever the *sets of distinct values* match.
* **WL's blind spot.** A 6-cycle and two disjoint triangles are both 2-regular: every node sees two neighbours of the same colour forever, so 1-WL — and therefore *any* message-passing GNN, GIN included — gives identical outputs (the identical GIN read-outs). Yet one graph contains triangles and the other does not. Plain message passing cannot count cycles reliably.
* **Mean vs sum.** A path of 3 nodes and a triangle *are* separated by WL (degrees differ) and by sum aggregation, but a mean aggregator with constant input features sees "1.0" everywhere: it has lost the degree information.

### 7.3 GIN

The **Graph Isomorphism Network** turns the theorem into a layer:
$$ h_v^{(k)} = \text{MLP}^{(k)}\Big( (1 + \epsilon^{(k)})\, h_v^{(k-1)} + \sum_{u\in\mathcal N(v)} h_u^{(k-1)} \Big), $$
with $\epsilon$ a learnable (or fixed) scalar that lets the model distinguish "me" from "a neighbour that looks like me", and an MLP (not a single linear layer) because one-layer perceptrons are not universal approximators of multiset functions. For graph classification, GIN sums node vectors at **every** layer and concatenates the results (a jumping-knowledge read-out, Section 8.7).

### 7.4 Does expressiveness matter for *our* project?

Much less than in molecule classification. WL's blind spots come from nodes that look identical (same features, same-looking neighbourhoods). In MV-HGAT every drug starts with its **own row of similarities to all drugs plus its own row of known links**: features are essentially unique per node, and with unique features even 1-WL tells all nodes apart. Our bottleneck is the *signal* (1 % positives, noisy similarities), not the ability to distinguish nodes. This is also why the project uses attention-weighted *averages* (bounded scale, robust to the hugely varying degrees: 10 to 94 neighbours in `chem_cdk`, up to 84 drugs for one disease) rather than GIN-style sums.

---

## 8. Depth: receptive field, over-smoothing, over-squashing

A CNN for images is often 50+ layers deep. Most successful GNNs have **2–4 layers**. Why?

### 8.1 Over-smoothing: what happens when you keep averaging

Li, Han & Wu (2018) observed that a GCN layer is a form of **Laplacian smoothing**: $\hat A = I - \tilde L_{\text{sym}}$ (where $\tilde L_{\text{sym}}$ is the normalised Laplacian of the graph *with* self-loops), so $\hat A x$ moves each value towards a degree-weighted average of its neighbours. Smoothing is *why* GCNs work (Section 5.3: noise cancels inside communities). But repeated smoothing makes **all node representations converge to the same thing**, and the classes become inseparable: **over-smoothing**.

**The limit, exactly.** Take a connected graph with self-loops. $\hat A$ is symmetric with eigen-decomposition $\hat A = \sum_k \mu_k u_k u_k^\top$, where $\mu_1 = 1 > |\mu_2| \ge \dots$ (Section 4.5) and $u_1 = \tilde D^{1/2}\mathbf 1 / \lVert \tilde D^{1/2}\mathbf 1\rVert$. Then
$$ \hat A^L = \sum_k \mu_k^L\, u_k u_k^\top \;\xrightarrow[L\to\infty]{}\; u_1 u_1^\top ,\qquad\text{so}\qquad \hat A^L X \to u_1\,(u_1^\top X). $$
Every row of the limit is $\sqrt{\tilde d_i}$ times **the same vector** $u_1^\top X / \lVert\tilde D^{1/2}\mathbf1\rVert$. All nodes point in the same direction (cosine similarity 1), differing only by a degree-dependent length. The only information left is the degree! The speed of convergence is governed by $|\mu_2|^L$: the closer $|\mu_2|$ is to 1 (a small **spectral gap** $1-|\mu_2|$), the slower the smoothing. With learned weights and ReLUs between the layers the picture is more complicated, but Oono & Suzuki (2020) proved that, under conditions on the weights' singular values, the distance from this "information-less" subspace still shrinks exponentially with depth.

A graph with several connected components converges to one such vector *per component* — information is not mixed across components.

### 8.2 Demo: dense similarity graph vs kNN graph

This experiment mirrors a real design decision in the project. We place 40 "drugs" in two chemical families, build (a) the **dense** similarity graph (everyone connected to everyone with a Gaussian-kernel weight — what you get if you use a similarity matrix directly as the adjacency) and (b) a **kNN graph** with $k = 5$, built with the same recipe as `data.py::knn_mask`. Then we apply $\hat A$ repeatedly to random features (no weights — pure propagation) and measure the mean cosine similarity between all node pairs (1 = everyone identical) and the **family gap**: mean within-family cosine minus mean between-family cosine (how much family structure the embeddings carry).

```python
# file: c3_07_oversmoothing.py
import numpy as np
rng = np.random.default_rng(0)

# 40 "drugs" in two chemical families (clusters) in a 5-D descriptor space
N = 40
fam = np.repeat([0, 1], N // 2)
pts = rng.normal(size=(N, 5)) + np.where(fam[:, None] == 1, 1.5, -1.5) * np.eye(5)[0]
D2 = ((pts[:, None] - pts[None]) ** 2).sum(-1)
S = np.exp(-D2 / D2.mean())                       # dense similarity in (0, 1]

def knn_mask(S, k):                               # same recipe as drepo/data.py::knn_mask
    W = S.copy(); np.fill_diagonal(W, -np.inf)
    idx = np.argpartition(-W, k, axis=1)[:, :k]
    M = np.zeros_like(S, bool)
    M[np.repeat(np.arange(len(S)), k), idx.ravel()] = True
    M |= M.T; np.fill_diagonal(M, True)
    return M.astype(float)

def gcn_norm(W):                                  # D^-1/2 W D^-1/2 (W already has self-loops)
    d = W.sum(1) ** -0.5
    return d[:, None] * W * d[None, :]

graphs = {"dense similarity": gcn_norm(S), "kNN (k=5)": gcn_norm(knn_mask(S, 5))}

def stats(H):
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    C = Hn @ Hn.T
    within = C[fam[:, None] == fam[None, :]].mean()
    between = C[fam[:, None] != fam[None, :]].mean()
    return C[~np.eye(N, dtype=bool)].mean(), within - between

X = rng.normal(size=(N, 16))                      # random input features
for name, P in graphs.items():
    lam = np.sort(np.abs(np.linalg.eigvalsh(P)))[::-1]
    print(f"\n{name}: |lambda_2| = {lam[1]:.3f}   cross-family edges = "
          f"{int(((P > 0) & (fam[:, None] != fam[None, :])).sum() // 2)}")
    print("  layers  mean cos-sim  family gap (within - between)")
    H = X.copy()
    for L in range(0, 65):
        if L in (0, 1, 2, 4, 8, 16, 32, 64):
            m, gap = stats(H)
            print(f"  {L:5d}   {m:10.3f}   {gap:10.3f}")
        H = P @ H                                  # one propagation step (no weights)
```

Output:

```text

dense similarity: |lambda_2| = 0.371   cross-family edges = 400
  layers  mean cos-sim  family gap (within - between)
      0        0.006        0.056
      1        0.839        0.109
      2        0.987        0.016
      4        1.000        0.000
      8        1.000        0.000
     16        1.000        0.000
     32        1.000        0.000
     64        1.000        0.000

kNN (k=5): |lambda_2| = 0.957   cross-family edges = 10
  layers  mean cos-sim  family gap (within - between)
      0        0.006        0.056
      1        0.186        0.382
      2        0.357        0.592
      4        0.508        0.682
      8        0.635        0.588
     16        0.784        0.360
     32        0.938        0.105
     64        0.996        0.007
```

(At layer 0 the features are random, so the small "gap" of 0.056 is noise.)

What this shows:

* **Dense similarity graph.** After **one** propagation step the mean cosine similarity jumps from 0.006 to 0.84, and by **four** steps every node is identical (cos-sim 1.000, gap 0). The graph has 400 cross-family edges — every pair is connected — so one averaging step already mixes the families. Its $|\mu_2| = 0.371$ is small: a huge spectral gap means *fast* convergence to the useless limit.
* **kNN graph.** Only 10 cross-family edges remain. Propagation first **sharpens** the family structure (gap rises from 0.06 to 0.68 at 4 steps: within-family neighbours are averaged together, noise cancels) and only *slowly* over-smooths ($|\mu_2| = 0.957$; at 64 steps it too has collapsed).
* **Moral:** kNN sparsification turns a graph that over-smooths in 1–2 layers into one where 2–8 layers of propagation are *useful*. This is the reason `knn_mask` exists (its docstring: "Dense similarity matrices connect everything to everything, which makes message passing blur all nodes together"). Note the useful window is still finite: depth must be chosen, not maximised.

### 8.3 Measuring over-smoothing

Two common diagnostics, both used above or in the exercises:

* **Mean pairwise cosine similarity** of node embeddings (→ 1 under over-smoothing).
* **Normalised Dirichlet energy** $E(H) = \mathrm{tr}(H^\top \tilde L H) / \mathrm{tr}(H^\top H)$ with $\tilde L = I - \hat A$: the share of the signal's energy that sits in "non-smooth" directions (→ 0 under over-smoothing). Exercise 12 computes it on the running example.

### 8.4 Over-squashing: too much information through too narrow a pipe

Over-smoothing is about representations becoming *too similar*. **Over-squashing** (Alon & Yahav 2021) is a different failure: the number of nodes in an $L$-hop neighbourhood can grow **exponentially** with $L$, but all of their information must be compressed into a **fixed-size vector** at the target node — and, worse, it has to pass through **bottleneck** edges on the way. Messages from far away are "squashed" and the gradient signal they produce is tiny, so the GNN fails on tasks that need **long-range** information. Topping et al. (2022) tied bottlenecks to negative graph curvature and proposed rewiring the graph to relieve them.

**Where it would bite in our project.** Through the association edges, a disease such as the one with 84 known drugs is a hub: in the drug → disease → drug two-hop step, everything that 84 drugs know passes through one 64-dimensional disease vector. The project keeps depth at 2, uses **attention** (which can focus on the few relevant messages instead of averaging them all, Unit C4), and separates relations so that each relation gets its *own* message vector before they are combined (Unit C5) — all of which reduce squashing compared with one big averaged neighbourhood.

### 8.5 Why depth is limited: summary

| Problem | Symptom | Cause |
|---|---|---|
| Over-smoothing | embeddings of all nodes converge; accuracy drops with depth | repeated low-pass filtering; $\hat A^L \to u_1 u_1^\top$ |
| Over-squashing | long-range dependencies not learned | exponentially growing neighbourhoods squeezed into a fixed vector through bottlenecks |
| Trainability | deep plain GNNs fail to train | vanishing/exploding gradients through many matrix products, as in deep MLPs |
| Overfitting | train ≫ test | more parameters, few labels (we have 1,933 positives) |

### 8.6 Remedies

* **Keep it shallow** (2–3 layers) — the default for good reasons.
* **Sparsify the graph** (kNN) — slows smoothing and removes noisy edges (Section 8.2).
* **Residual / skip connections:** $h^{(l+1)} = h^{(l)} + \text{GNNLayer}(h^{(l)})$, or a learned skip $W_s h^{(l)}$ when widths differ. The node keeps its own identity at every layer, so the "D0 = D1" collapse of Section 5.2 cannot happen. GCNII (Chen et al. 2020) goes further with an *initial residual* (mixing in $h^{(0)}$ at every layer) and identity mapping, training 64-layer GCNs.
* **Normalisation:** LayerNorm (used in the project), or PairNorm (Zhao & Akoglu 2020), which explicitly keeps the total pairwise distance between node embeddings constant.
* **DropEdge** (Rong et al. 2020): randomly delete a fraction of edges each epoch. It slows smoothing (a sparser graph has a smaller spectral gap) and regularises. The project's hidden-link training is a link-prediction-specific form of DropEdge on the association edges.
* **Jumping knowledge** — next.

### 8.7 Jumping-knowledge networks (Xu et al. 2018)

Xu et al. noticed that the *right* receptive field differs **per node**. In a graph with a dense core and a tree-like periphery, two layers from a core node already cover a huge, nearly random part of the graph (its influence distribution has spread), while two layers from a peripheral node cover only a handful of nodes. A single global depth is a compromise that is wrong for both.

**JK-Net** keeps every layer's representation and lets the final layer choose:
$$ h_v^{\text{final}} = \text{AGG}\big(h_v^{(0)}, h_v^{(1)}, \dots, h_v^{(L)}\big) $$
with three options:

* **Concatenation:** $[h^{(0)} \Vert h^{(1)} \Vert \dots \Vert h^{(L)}]$ followed by a linear layer. The same mixing for every node, but the downstream layer learns which depths matter. **This is what MV-HGAT uses** (`torch.cat(outs[...], 1)` in `MVHGAT.encode`).
* **Max-pooling** (element-wise over layers): each feature picks its best depth, node by node — node-adaptive without extra parameters.
* **LSTM-attention:** a bi-directional LSTM reads the sequence of layer outputs and produces attention scores over layers per node — the most flexible.

LAGCN's "layer attention" (Section 12.3) is a fourth, simpler variant: a learned softmax weighting of the layers, *shared by all nodes*.

**Demo: depth with and without residuals / JK.** Same toy SBM as Section 5.3; a stack of $L$ GCN layers of width 16 after an input layer; three variants.

```python
# file: c3_08_depth_residual_jk.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# same toy graph as before: 2 communities x 30 nodes, 3 labels per class
rng = np.random.default_rng(0)
y = np.repeat([0, 1], 30); N = 60
same = y[:, None] == y[None, :]
up = np.triu(rng.random((N, N)) < np.where(same, 0.20, 0.02), 1)
A = torch.tensor((up | up.T), dtype=torch.float32) + torch.eye(N)
d = A.sum(1).pow(-0.5); A_hat = d[:, None] * A * d[None, :]
X = rng.normal(size=(N, 8)).astype(np.float32); X[:, 0] += np.where(y == 1, 1.0, -1.0)
X, yt = torch.tensor(X), torch.tensor(y)
train = np.concatenate([np.where(y == c)[0][:3] for c in (0, 1)])
test = np.setdiff1d(np.arange(N), train)

class DeepGCN(nn.Module):
    """mode = 'plain' | 'residual' | 'jk' (jumping knowledge, concatenation)."""
    def __init__(self, L, mode, hid=16):
        super().__init__()
        self.inp = nn.Linear(8, hid)
        self.layers = nn.ModuleList(nn.Linear(hid, hid) for _ in range(L))
        self.mode = mode
        self.out = nn.Linear(hid * (L + 1) if mode == "jk" else hid, 2)
    def forward(self, A_hat, X):
        h = F.relu(self.inp(X))
        hs = [h]
        for lin in self.layers:
            new = F.relu(A_hat @ lin(h))
            h = h + new if self.mode == "residual" else new
            hs.append(h)
        return self.out(torch.cat(hs, 1) if self.mode == "jk" else h)

def acc(L, mode, seed):
    torch.manual_seed(seed)
    m = DeepGCN(L, mode)
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        loss = F.cross_entropy(m(A_hat, X)[train], yt[train])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (m(A_hat, X).argmax(1)[test] == yt[test]).float().mean().item()

print("layers   plain   residual   JK-concat   (test accuracy, mean of 5 seeds)")
for L in (1, 2, 4, 8, 16, 32):
    row = [np.mean([acc(L, mode, s) for s in range(5)]) for mode in ("plain", "residual", "jk")]
    print(f"{L:6d}   {row[0]:.3f}    {row[1]:.3f}      {row[2]:.3f}")
```

Output (this one takes a couple of minutes on a CPU):

```text
layers   plain   residual   JK-concat   (test accuracy, mean of 5 seeds)
     1   0.952    0.852      0.881
     2   0.978    0.952      0.933
     4   0.959    0.948      0.937
     8   0.563    0.967      0.941
    16   0.567    0.956      0.933
    32   0.544    0.948      0.933
```

* The **plain** GCN is best at 2 layers (0.978) and **collapses to chance (~0.55) from 8 layers on** — over-smoothing plus trainability problems.
* **Residual** and **JK** variants stay around 0.93–0.97 at *every* depth up to 32. They do not make depth *help* on this easy task, but they make it *safe*, which matters when you do not know the right depth in advance.
* At 1–2 layers the plain model is slightly better here: skip paths also carry the noisy raw features forward. There is no free lunch; always compare on validation data.

---

## 9. Transductive vs inductive learning

### 9.1 Definitions

* **Transductive:** training and prediction happen on **one fixed graph** whose nodes are all known in advance. The test nodes (or test edges) are *present* during training — their features and connections participate in message passing — only their **labels** are hidden. The goal is to label *those specific* nodes/edges.
* **Inductive:** the model learns a **rule** that can be applied to nodes, edges or whole graphs **not present during training**, without retraining.

These are properties of the *setup and the model's inputs*, not just of the architecture:

* The GCN *layer* has no node-specific parameters, so architecturally it could run on a new graph. But the original GCN experiments are transductive (test nodes are in the graph during training), and a GCN whose normalisation or input features depend on the full node set is tied to that node set.
* Anything that uses **node-ID embeddings** (a learned vector per node, as in matrix factorisation) or **features whose dimension is indexed by the training nodes** is inherently transductive: a new node has no embedding and no well-defined feature vector.
* GraphSAGE is the standard example of an inductive GNN: features are intrinsic node attributes, and the aggregators work on any neighbourhood (Section 6.4 showed it running on an unseen graph).

### 9.2 Our model is transductive — concretely why

Look at how MV-HGAT builds its input features (`MVHGATMethod.build`, `feats` and `features`):

* A drug's feature vector is the **concatenation of its rows in the three drug similarity matrices** — its similarity to each of the 593 training drugs, three times — plus its **row of visible known links** (313 entries, one per disease). Width $3 \times 593 + 313 = 2{,}092$ on Fdataset.
* A disease's features: its rows in the three disease similarity matrices plus its column of visible links: $3 \times 313 + 593 = 1{,}532$.
* The first layer `self.inp = nn.ModuleDict({"drug": nn.Linear(in_drug, hidden), ...})` has a weight for **each of those positions**, i.e. for "similarity to drug #17".

So the meaning of input coordinate 17 is "similarity to the 17th training drug". A brand-new drug (#594) would need a 2,095-wide vector; there is no weight for "similarity to drug #594", and the kNN graphs, the propagation kernels and the association matrix are all built for exactly 593 drugs. **To score a brand-new drug you must add it to the similarity matrices and the graphs and retrain** (or at least re-run `build` and fine-tune). Fdataset and Cdataset models cannot be swapped either.

What the model *can* handle is a **cold-start node that already exists in the graph**: a disease with no known drugs (as in leave-one-disease-out evaluation) still has its similarity rows, its kNN neighbours and a place in every matrix. That is a *transductive* cold start, and the project trains for it explicitly (`cold_frac`: each epoch 10 % of diseases lose all their links; the degree gate and propagation head then carry the prediction).

**How you could make it inductive** (future work): describe drugs by intrinsic features that exist for any molecule (e.g. a 2,048-bit ECFP fingerprint instead of "similarity to each training drug"), compute a new drug's kNN neighbours among the training drugs, and use aggregators that do not depend on the global node set (GraphSAGE-style). The price is usually some accuracy on the transductive benchmark.

---

## 10. Dense vs sparse implementations

There are two ways to implement the same layer:

* **Dense:** store $\hat A$ as an $N_\text{dst}\times N_\text{src}$ matrix and multiply. Memory $O(N^2)$, time $O(N^2 d)$. Trivial to write, fast on a GPU for small $N$, and masking (for attention) is just `masked_fill`. **The project does this** — e.g. `Ahat @ lin(x)` in `_GCN.forward`, `einsum("dsh,shk->dhk", ...)` in `DenseGAT.forward`.
* **Sparse / edge list:** store only the edges as two index arrays `src`, `dst` (and a weight per edge). A layer = compute one message per edge (`X[src]`), multiply by the edge weight, and **scatter-add** into the destination nodes (`index_add_`). Memory and time $O(|E|\,d)$. This is how `torch_geometric` and DGL work, and it is the only option for large graphs.

```python
# file: c3_09_dense_vs_sparse.py
import torch

torch.manual_seed(0)
# the 4-node example graph, as an EDGE LIST (both directions + self-loops)
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
src = torch.tensor([u for u, v in edges] + [v for u, v in edges] + [0, 1, 2, 3])
dst = torch.tensor([v for u, v in edges] + [u for u, v in edges] + [0, 1, 2, 3])
N, F_in = 4, 3
X = torch.randn(N, F_in)

# --- sparse "message passing" with scatter-add -------------------------------
deg = torch.zeros(N).index_add_(0, dst, torch.ones(len(dst)))      # in-degree incl. self-loop
norm = deg[src].rsqrt() * deg[dst].rsqrt()                          # 1/sqrt(d_i d_j) per edge
messages = norm[:, None] * X[src]                                   # one message per edge
out_sparse = torch.zeros(N, F_in).index_add_(0, dst, messages)      # aggregate = sum at dst

# --- torch.sparse matrix ------------------------------------------------------
A_sp = torch.sparse_coo_tensor(torch.stack([dst, src]), norm, (N, N), check_invariants=True)
out_spmm = torch.sparse.mm(A_sp, X)

# --- dense matrix (what the project does) -------------------------------------
A = torch.zeros(N, N); A[dst, src] = 1.0
d = A.sum(1).rsqrt()
out_dense = (d[:, None] * A * d[None, :]) @ X

print("scatter == dense:", torch.allclose(out_sparse, out_dense, atol=1e-6))
print("spmm    == dense:", torch.allclose(out_spmm, out_dense, atol=1e-6))
print("number of stored entries: dense", N * N, "| sparse", len(src))

# --- memory arithmetic for the project's graphs -------------------------------
for name, n_dst, n_src in [("Fdataset drug view", 593, 593),
                           ("Fdataset disease view", 313, 313),
                           ("1 million-node graph", 10**6, 10**6)]:
    dense_bytes = n_dst * n_src * 4                  # float32 adjacency
    approx_edges = n_dst * 15                        # ~ k=10 symmetrised + self-loop
    sparse_bytes = approx_edges * (8 + 8 + 4)        # two int64 indices + one float32 value
    print(f"{name:22s}: dense {dense_bytes / 2**20:10.2f} MiB  edge list {sparse_bytes / 2**20:7.2f} MiB")
```

Output:

```text
scatter == dense: True
spmm    == dense: True
number of stored entries: dense 16 | sparse 12
Fdataset drug view    : dense       1.34 MiB  edge list    0.17 MiB
Fdataset disease view : dense       0.37 MiB  edge list    0.09 MiB
1 million-node graph  : dense 3814697.27 MiB  edge list  286.10 MiB
```

The scatter version makes the message-passing template *literal*: `X[src]` = messages, `norm` = per-edge coefficient, `index_add_` = sum aggregation at `dst`. All three implementations agree. For the project, the dense matrices are around a megabyte each — dense is simpler and fast. For a million-node graph the dense adjacency would need about 3.6 TiB, while the edge list needs under 300 MiB. **Rule of thumb:** dense is fine up to a few thousand nodes per node type; beyond roughly $10^4$ nodes switch to edge lists (or a library such as PyTorch Geometric / DGL) and, for very large graphs, neighbour sampling.

(The approximation "15 edges per node" matches the real kNN graphs from Section 1: mean degree 13–16, plus the self-loop.)

---

## 11. GNNs for link prediction, and the detect-vs-predict trap

Unit C6 treats link prediction in depth; here is the minimum you need to read the project's GCN baselines. A GNN **encoder** produces an embedding for every drug ($h_i$) and every disease ($g_j$); a **decoder** turns a pair into a score: dot product $h_i^\top g_j$ (NIMCGCN, LAGCN), bilinear $h_i^\top W g_j$ (MV-HGAT) or an MLP. Training uses known links as positives and sampled unknown pairs as negatives with binary cross-entropy.

**The trap.** If the known links are *also edges of the message-passing graph* (and/or input features), then for every training positive $(i, j)$ the encoder can *see* the edge $i$–$j$. The cheapest way to minimise the loss is to learn "score high if the edge is in my input" — **edge detection**. At test time the link to be predicted is, by construction, *not* in the input, so that shortcut is useless. The fix used by MV-HGAT (`supervise_hidden=True`) is to hide a random 20 % of training links from the graph and features each epoch and compute the loss **only on those hidden links**, mimicking test conditions.

A toy reproduction: 60 "drugs" and 40 "diseases" with hidden mechanism groups; drug–disease links are likely within a group; noisy similarities reveal the groups; 20 % of links are held out for testing. The encoder is a 2-layer GCN on the block graph $\begin{psmallmatrix}K_r & A\\ A^\top & K_d\end{psmallmatrix}$ (like LAGCN, with kNN similarity blocks), the decoder is bilinear, and the two runs differ **only** in which positives the loss sees.

```python
# file: c3_10_detect_vs_predict.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- a toy drug-disease world with hidden "mechanism groups" ----------
rng = np.random.default_rng(0)
nr, nd, G = 60, 40, 4
gr, gd = rng.integers(G, size=nr), rng.integers(G, size=nd)       # mechanism group
A = (rng.random((nr, nd)) < np.where(gr[:, None] == gd[None, :], 0.25, 0.01)).astype(np.float32)

def noisy_similarity(groups):
    S = (groups[:, None] == groups[None, :]) * 0.6 + rng.random((len(groups),) * 2) * 0.5
    S = (S + S.T) / 2; np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)
Sr, Sd = noisy_similarity(gr), noisy_similarity(gd)

def knn(S, k=5):
    W = S.copy(); np.fill_diagonal(W, -np.inf)
    idx = np.argpartition(-W, k, axis=1)[:, :k]
    M = np.zeros_like(S, bool); M[np.repeat(np.arange(len(S)), k), idx.ravel()] = True
    M |= M.T; np.fill_diagonal(M, True)
    return M.astype(np.float32)
Kr, Kd = knn(Sr), knn(Sd)

# hold out 20% of the known links as the TEST set
pos = np.flatnonzero(A.ravel())
test_pos = rng.choice(pos, size=len(pos) // 5, replace=False)
A_train = A.copy().ravel(); A_train[test_pos] = 0; A_train = A_train.reshape(nr, nd)
test_cells = np.flatnonzero(A_train.ravel() == 0)                 # test pos + all unknowns
y_test = A.ravel()[test_cells]
print(f"links: total {int(A.sum())}, train {int(A_train.sum())}, test {len(test_pos)}")

def norm(M):
    d = M.sum(1).clamp(min=1e-9).rsqrt()
    return d[:, None] * M * d[None, :]

class HeteroGCN(nn.Module):
    """2-layer GCN on the block graph [[Kr, A], [A^T, Kd]] + bilinear decoder."""
    def __init__(self, d_in, hid=32):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, hid), nn.Linear(hid, hid)
        self.W = nn.Parameter(torch.eye(hid))
    def forward(self, Avis):
        top = torch.cat([torch.tensor(Kr), Avis], 1)
        bot = torch.cat([Avis.T, torch.tensor(Kd)], 1)
        Ablk = torch.cat([top, bot], 0)
        X = torch.cat([torch.cat([torch.tensor(Sr), Avis], 1),            # drug features
                       torch.cat([Avis.T, torch.tensor(Sd)], 1)], 0)      # disease features
        P = norm(Ablk)
        h = F.relu(P @ self.l1(X))
        h = P @ self.l2(h)
        return h[:nr] @ self.W @ h[nr:].T                                 # nr x nd logits

def train(hidden_supervision, seed=0, epochs=300, hide=0.2):
    torch.manual_seed(seed)
    At = torch.tensor(A_train)
    pos_t = torch.tensor(np.flatnonzero(A_train.ravel()))
    neg_pool = torch.tensor(np.flatnonzero(A_train.ravel() == 0))
    model = HeteroGCN(nr + nd)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        if hidden_supervision:                # hide some links, supervise ONLY on them
            h = torch.rand(len(pos_t)) < hide
            Avis = At.clone().view(-1); Avis[pos_t[h]] = 0; Avis = Avis.view(nr, nd)
            sup = pos_t[h]
        else:                                 # supervise on links the model can see
            Avis, sup = At, pos_t
        neg = neg_pool[torch.randint(len(neg_pool), (2 * len(sup),))]
        logits = model(Avis).view(-1)
        y = torch.cat([torch.ones(len(sup)), torch.zeros(len(neg))])
        loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s = model(At).view(-1).numpy()
    train_auc = roc_auc_score(A_train.ravel(), s)        # can it find links it SEES?
    return train_auc, roc_auc_score(y_test, s[test_cells]), average_precision_score(y_test, s[test_cells])

for name, flag in [("supervise on VISIBLE links", False), ("supervise on HIDDEN links ", True)]:
    r = np.array([train(flag, seed) for seed in range(3)]).mean(0)
    print(f"{name}: AUC on visible train links {r[0]:.3f} | TEST AUC {r[1]:.3f}  TEST AUPR {r[2]:.3f}")
```

Output:

```text
links: total 184, train 148, test 36
supervise on VISIBLE links: AUC on visible train links 0.932 | TEST AUC 0.731  TEST AUPR 0.040
supervise on HIDDEN links : AUC on visible train links 0.884 | TEST AUC 0.819  TEST AUPR 0.065
```

The model trained on visible links is *better at recognising links it can see* (0.932) and *worse at predicting the held-out ones* (0.731 vs 0.819 AUC). That is the project's 0.71 → 0.92 story in miniature. (Test AUPR is low for both because only 36 of about 2,250 candidate cells are positive — base rate 1.6 %; see Unit B2.)

---

## 12. In this project: walking through the code

Open `src/drepo/data.py`, `src/drepo/methods.py` and `src/drepo/model.py` side by side with this section.

### 12.1 `data.py::knn_mask` — turning a similarity matrix into a graph

```python
def knn_mask(S: np.ndarray, k: int, symmetric: bool = True) -> np.ndarray:
    S = fill_missing(S)
    n = S.shape[0]
    work = S.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(-work, k, axis=1)[:, :k]
    M = np.zeros_like(S, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    vals = work[rows, idx.ravel()]
    keep = vals > 0                         # never link on zero similarity
    M[rows[keep], idx.ravel()[keep]] = True
    if symmetric:
        M |= M.T
    np.fill_diagonal(M, True)
    return M
```

Line by line:

* `fill_missing(S)` replaces NaN (an entity not covered by this view: a biologic drug without SMILES, a disease without CTD genes) by 0 and sets the diagonal to 1. *We never invent a similarity.*
* `np.fill_diagonal(work, -np.inf)` stops a node from choosing itself as one of its $k$ neighbours.
* `np.argpartition(-work, k, axis=1)[:, :k]` finds the $k$ largest similarities per row in $O(n)$ per row without fully sorting (Unit A1).
* `keep = vals > 0` — a node whose row is all zeros (uncovered) gets **no** neighbours instead of $k$ arbitrary ones. That is why `gene_d` has 153 isolated diseases (Section 1).
* `M |= M.T` **symmetrises**: $i$–$j$ is an edge if $j$ is among $i$'s top $k$ *or* $i$ among $j$'s. So every covered node has at least $k$ neighbours, but popular nodes collect many more (max degree 94 in `chem_cdk`). Symmetry makes the graph undirected, matching the GCN derivation.
* `np.fill_diagonal(M, True)` adds **self-loops** — the $A + I$ of the renormalisation trick, built into the graph itself. Consequence: in every *view* relation each node has at least one neighbour (itself). Unit C4 shows why this matters for attention.

The *purpose* is stated in the docstring and demonstrated in Section 8.2: a dense similarity matrix "connects everything to everything, which makes message passing blur all nodes together".

### 12.2 `methods.py::sym_norm` and `_GCN` — the shared GCN machinery of the baselines

```python
def sym_norm(S):
    d = S.sum(1)
    d[d == 0] = 1
    d = 1 / np.sqrt(d)
    return S * d[:, None] * d[None, :]
```

`sym_norm` computes $D^{-1/2} S D^{-1/2}$ for a (weighted) matrix $S$, with a guard so that an all-zero row does not divide by zero. It does **not** add the identity itself: in NIMCGCN the self-loops come from `knn_mask` (diagonal `True`) times the similarity diagonal (1 after `fill_missing`), so `sym_norm(Sr * knn_mask(Sr, k))` is exactly the **weighted renormalised** GCN matrix $\tilde D^{-1/2}(S\odot M)\tilde D^{-1/2}$ of Section 5.1. (The broadcasting `S * d[:, None] * d[None, :]` is the cheap way to compute $D^{-1/2} S D^{-1/2}$ without building diagonal matrices.)

```python
class _GCN(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.Dropout(dropout)

    def forward(self, Ahat, x, return_all=False):
        outs = []
        for lin in self.lins:
            x = F.relu(Ahat @ lin(self.drop(x)))
            outs.append(x)
        return outs if return_all else x
```

* `dims` is a list of widths, e.g. `[593, 128, 64]` → two layers 593→128→64.
* `F.relu(Ahat @ lin(self.drop(x)))` is $H^{(l+1)} = \text{ReLU}(\hat A\,\text{dropout}(H^{(l)}) W + \mathbf 1 b^\top)$ — Kipf & Welling's layer, with dropout on the layer input as in the paper. Note the bias is added *before* propagation (inside `lin`), so each node receives $\hat A \mathbf 1 b^\top$, a degree-dependent multiple of $b$. A small difference from the paper, harmless in practice.
* ReLU is applied after **every** layer, including the last, so final embeddings are non-negative. NIMCGCN then applies a linear projection, which restores signed values; LAGCN uses the non-negative embeddings directly in a dot product, so every logit is $\ge 0$ and every predicted probability $\ge 0.5$. Ranking metrics (AUC, AUPR) do not care about a monotone shift, but the BCE loss can never push a negative pair below 0.5, which weakens the training signal — one more limitation of that re-implementation.
* `return_all=True` returns the list of every layer's output: the hook LAGCN uses for layer attention.

### 12.3 `NIMCGCN` (Li et al. 2020) — two GCNs + inductive matrix completion

```python
def fit_predict(self, data, A_train, neg_mask, seed=0):
    set_seed(seed)
    Sr, Sd = bench_sims(data)
    Ar = t(sym_norm(Sr * knn_mask(Sr, self.k)))
    Ad = t(sym_norm(Sd * knn_mask(Sd, self.k)))
    Xr, Xd = t(Sr), t(Sd)
    gr = _GCN([Xr.shape[1], self.h, self.o], self.dropout).to(DEVICE)
    gd = _GCN([Xd.shape[1], self.h, self.o], self.dropout).to(DEVICE)
    fr, fd = nn.Linear(self.o, self.o).to(DEVICE), nn.Linear(self.o, self.o).to(DEVICE)
    ...
    for _ in range(self.epochs):
        [m.train() for m in mods]
        logits = fr(gr(Ar, Xr)) @ fd(gd(Ad, Xd)).T
        loss = weighted_bce(logits, A, nm)
        ...
```

* `bench_sims(data)` returns only the **two benchmark similarities** (`chem_cdk`, `pheno_mim`) — the baselines use exactly what their papers used.
* `Ar`, `Ad`: kNN (k = 10) **weighted** graphs with self-loops, symmetric-normalised. Good practice against over-smoothing.
* `Xr, Xd = t(Sr), t(Sd)`: each drug's input feature is **its full similarity row** (593 numbers) — transductive features, just as in MV-HGAT.
* Two separate 2-layer GCNs (593 → 128 → 64 for drugs; 313 → 128 → 64 for diseases), then per-side linear projections `fr`, `fd` into a common space, and the **inner product** decoder: $\text{logit}_{ij} = (W_r h_i + b_r)^\top (W_d g_j + b_d)$. The original paper calls this *neural inductive matrix completion* (NIMC): matrix completion where the factors are produced by networks from side information instead of being free parameters per node.
* `weighted_bce(logits, A, nm)` is a **full-matrix** loss over every allowed cell, positives up-weighted by (#negatives / #positives) — roughly 95–100 on Fdataset — to balance the classes.

**What to notice.** `A_train` **never enters the encoder**: the graphs are drug–drug and disease–disease only. So NIMCGCN cannot fall into the detect-vs-predict trap — but it also gets no collaborative message passing ("drugs that treat my diseases"). All association knowledge must be squeezed into the weights through the loss. Plausible contributors to its low AUPR (0.096 on Fdataset; the project did not ablate these, so treat them as hypotheses): only two similarity views; a heavily up-weighted full-matrix loss, which tends to help AUC more than the precision of the top-ranked predictions; and an encoder whose only input is similarity, so two chemically similar drugs with different indications are hard to separate.

### 12.4 `LAGCN` (Yu et al. 2021) — one GCN on the heterogeneous graph + layer attention

```python
def fit_predict(self, data, A_train, neg_mask, seed=0):
    set_seed(seed)
    Sr, Sd = bench_sims(data)
    n_r = Sr.shape[0]
    H = t(np.block([[Sr, A_train], [A_train.T, Sd]]))
    gcn = _GCN([H.shape[1]] + [self.h] * self.L, self.dropout).to(DEVICE)
    att = nn.Parameter(torch.ones(self.L, device=DEVICE) / self.L)
    ...
    def embed(Hm):
        d = Hm.sum(1).clamp(min=1e-12).rsqrt()
        outs = gcn(Hm * d[:, None] * d[None, :], H, return_all=True)
        w = torch.softmax(att, 0)
        return sum(wi * o for wi, o in zip(w, outs))

    for _ in range(self.epochs):
        gcn.train()
        drop = assoc & (torch.rand_like(H) < self.de)
        E = embed(H.masked_fill(drop, 0))
        loss = weighted_bce(E[:n_r] @ E[n_r:].T, A, nm)
        ...
```

* `H = [[Sr, A], [Aᵀ, Sd]]` is a $(593+313)\times(593+313) = 906\times906$ **heterogeneous** adjacency: drug–drug similarity, drug–disease links, disease–disease similarity. It is used **both as the graph and as the input features** (`gcn(..., H, ...)`: node $i$'s features are its row of `H`).
* `embed` normalises symmetrically, $D^{-1/2} H D^{-1/2}$. No identity is added because the diagonal of `Sr` and `Sd` is already 1 (self-loops).
* `_GCN([906, 64, 64, 64])` → 3 GCN layers; `return_all=True` gives $H^{(1)}, H^{(2)}, H^{(3)}$.
* **Layer attention:** `att` holds $L$ learnable scores (initialised equal); `softmax(att)` turns them into weights $w_l \ge 0$, $\sum_l w_l = 1$, and the final embedding is $E = \sum_l w_l H^{(l)}$. This is a jumping-knowledge aggregator (Section 8.7) with one set of weights **shared by all nodes** — the model can learn "trust layer 1 more than layer 3", which counteracts over-smoothing in deep layers.
* Decoder: plain dot product `E[:n_r] @ E[n_r:].T`; loss: `weighted_bce` on the full matrix.
* `drop = assoc & (rand < self.de)`: DropEdge on the association block only — but the default `drop_edge=0.0` switches it off.

**What to notice — two issues this chapter equips you to see:**

1. **Dense similarity blocks.** `Sr` and `Sd` are used *without* kNN sparsification. On Fdataset every drug–drug similarity is non-zero (`Sr` is 100 % dense; `Sd` 78 %), and an average drug row has similarity mass 110.6 against association mass 3.3. **Only about 3 % of a drug's propagation weight goes to its diseases**, the rest is spread over all 593 drugs. Section 8.2 showed what one or two steps on such a graph do: everything converges. The layer attention mitigates this by leaning on early layers, but the collaborative signal is diluted from the first layer.
2. **The detection trap.** The training links are in the graph *and* in the features, and the loss is computed on exactly those links. Section 11 showed that this teaches edge detection rather than prediction.

Together these give a plausible (again, not ablated) explanation of why LAGCN's AUPR here is 0.133, far below MV-HGAT's 0.488, which uses kNN graphs, separate relations and hidden-link supervision. Note that these numbers come from *our re-implementations* on our splits; published results of the original authors can differ (the project's `HOW_IT_WORKS.md` §5.4 insists on saying so in the paper).

### 12.5 MV-HGAT: where the ideas of this chapter live

**Graph construction** (`MVHGATMethod.build`):

```python
for v in rv:
    relations[f"view:{v}"] = ("drug", "drug")
    graphs[f"view:{v}"] = t(knn_mask(data.drug_view(v), c.k), torch.bool)
...
if c.use_assoc_edges:
    relations["assoc>drug"] = ("drug", "disease")
    relations["assoc>disease"] = ("disease", "drug")
    A = t(A_train > 0, torch.bool)
    graphs["assoc>drug"], graphs["assoc>disease"] = A, A.T
```

One **boolean** kNN mask (k = 10) per similarity view and the training links in both directions: 3 + 3 + 2 = 8 relations by default (the gene bridge is an optional ablation). Each relation is a `(dst_type, src_type)` pair — `assoc>drug` means "drugs receive messages from diseases". Masks are boolean because the attention layer (Unit C4) learns its own edge weights instead of using $\hat A$.

**Hidden-link training** (`fit_predict`): each epoch `graphs["assoc>drug"], graphs["assoc>disease"] = Am, Am.T`, where `Am` contains only the *visible* links (20 % hidden, plus all links of 10 % "cold" diseases), and `sup = pos[hide]`. This is the fix for Section 11's trap and a DropEdge-style regulariser (Section 8.6) at the same time.

**One layer's update** (`HeteroLayer.forward`):

```python
new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
```

$z$ is the attention-combined message from all relations; `self.skip[t]` is a learned linear **skip connection** of the node's own previous state; `LayerNorm` keeps the scale stable; ELU is the nonlinearity; dropout regularises. Skip + normalisation are two of the over-smoothing remedies from Section 8.6, and the skip guarantees that nodes with identical neighbourhoods (like D0 and D1 in Section 5.2) keep different embeddings.

**Jumping knowledge** (`MVHGAT.encode`):

```python
def encode(self, X, graphs, drop_rel=()):
    h = {t: self.drop(F.elu(self.inp[t](X[t]))) for t in ("drug", "disease")}
    outs = {t: [h[t]] for t in h}
    all_betas = []
    for layer in self.layers:
        h, betas = layer(h, graphs, self.uniform, drop_rel)
        all_betas.append(betas)
        for t in h:
            outs[t].append(h[t])
    return torch.cat(outs["drug"], 1), torch.cat(outs["disease"], 1), all_betas
```

* `h` = $h^{(0)}$: the input projection (2,092 → 64 for drugs, 1,532 → 64 for diseases on Fdataset), with ELU and dropout. It contains **no** neighbour information.
* `outs[t]` collects $h^{(0)}, h^{(1)}, h^{(2)}$.
* `torch.cat(..., 1)` is the **JK concatenation**: each drug and disease ends with a $64 \times 3 = 192$-dimensional vector. Matching this, the constructor sets `out = hidden * (layers + 1)` and the bilinear decoder matrix is `W` of size 192 × 192: the decoder can learn, for example, that a drug's *0-hop* features should be matched with a disease's *2-hop* features.

Why include $h^{(0)}$? A pair's score can then use the node's *own* similarity profile and visible links undiluted by any neighbourhood averaging — protection against over-smoothing exactly where it would hurt most, for well-characterised drugs with strong individual signals. And the 2-hop part provides the collaborative "drug → disease → drug" signal of Section 3.5.

**Design summary.** The table maps this chapter's ideas to the project's choices:

| Concept (this chapter) | MV-HGAT choice | Where |
|---|---|---|
| sparse, homophilous graph | kNN, k = 10, symmetrised, self-loops | `data.py::knn_mask`, `build` |
| depth = receptive field | 2 layers (2-hop: co-treating drugs) | `MVHGATConfig.layers` |
| over-smoothing remedies | skip + LayerNorm, JK concat, kNN | `HeteroLayer.forward`, `MVHGAT.encode` |
| aggregation | attention-weighted mean within a relation, attention across relations | `DenseGAT`, `ViewAttention` (Units C4, C5) |
| dense implementation | boolean masks, `einsum` | `DenseGAT.forward` |
| transductive | similarity-row and link-row features | `build`, `features` |
| link-prediction training | hidden-link supervision (DropEdge-like) | `fit_predict` |

---

## 13. Common mistakes and misconceptions

1. **Forgetting self-loops.** Without $A + I$, a node's new state ignores its own features (only neighbours count), and the normalisation's eigenvalues can reach $-1$ on bipartite structures (oscillation). In the project the self-loops are inside `knn_mask`; if you build a graph by hand, add them.
2. **Normalising before adding self-loops.** $D^{-1/2}AD^{-1/2} + I$ is *not* the renormalised $\hat A$; it is the unstable $I + D^{-1/2}AD^{-1/2}$ of Section 4.5.
3. **Dividing by zero degrees.** Isolated nodes (153 diseases in `gene_d`!) give $d = 0$. Guard as `sym_norm` does (`d[d == 0] = 1`) or `clamp(min=...)`.
4. **"More layers = more power."** In GNNs depth beyond 2–4 usually hurts (Section 8.7 table: plain GCN drops to chance at 8 layers). Use residuals/JK if you go deeper, and pick depth on validation data.
5. **Using a dense similarity matrix as the adjacency.** Every node is connected to every node, so one layer averages almost everything (Section 8.2). Sparsify (kNN or a threshold).
6. **Confusing over-smoothing with over-squashing.** The first is *too similar* (too much mixing); the second is *too little* information arriving from far away through bottlenecks. Remedies differ (residuals vs rewiring/attention).
7. **Thinking a GCN is automatically inductive because it has no per-node parameters.** If the input features are indexed by the training nodes (similarity rows!), a new node has no valid input. MV-HGAT, NIMCGCN and LAGCN are all transductive for this reason.
8. **Supervising link prediction on edges visible in the input graph.** The model learns to detect edges (Section 11; the project's 0.71 → 0.92 lesson).
9. **Mean aggregation everywhere "because it is normalised".** Mean discards neighbourhood *size* (Section 7.2). Sometimes the count *is* the signal (e.g. how many drugs a disease already has). The project re-injects such counts explicitly through the degree gate.
10. **Training-time sampling at test time.** GraphSAGE samples during training; at inference use the full neighbourhood (or average many samples).
11. **Ignoring that $\hat A$'s rows don't sum to 1.** Symmetric normalisation makes high-degree nodes' outputs larger or smaller than the input scale; if you need true averages use $\tilde D^{-1}\tilde A$.
12. **Thinking the spectral view is required to compute a GCN.** It motivates the normalisation; the computation is a sparse/dense matrix product, no eigenvectors involved.

---

## 14. Exercises

Difficulty: ★ = conceptual / quick, ★★ = requires working, ★★★ = challenging. Try each before opening the solution.

**Exercise 1 (★, conceptual).** A colleague proposes to treat the 593 × 593 `chem_cdk` kNN graph as an image and run a 3 × 3 CNN over its adjacency matrix. Give two concrete reasons why this is a bad idea.

<details><summary>Solution</summary>

1. **The result depends on the arbitrary node order.** A 3 × 3 window covers entries $(i\pm1, j\pm1)$ — "drugs whose index is next to $i$". Index adjacency means nothing (the order comes from the CSV). Relabelling the drugs changes which entries sit next to each other, so the output changes for the same graph: the model is not permutation equivariant (Section 2.2).
2. **Locality in the image is not locality in the graph.** A node's neighbours are scattered across its whole row (anywhere among 593 columns), not in a 3 × 3 patch. The CNN's useful inductive bias (nearby pixels are related) does not hold.
3. (Bonus) It cannot be applied to Cdataset (663 nodes) without changing the architecture, and the number of meaningful entries is tiny (about 2.5 % are edges).
</details>

**Exercise 2 (★, math).** (a) Show that a permutation matrix satisfies $P^\top P = I$. (b) Show that $f(A, X) = AX$ is permutation equivariant and $g(A, X) = \mathbf 1^\top A X$ is permutation invariant.

<details><summary>Solution</summary>

(a) Column $c$ of $P$ is a standard basis vector $e_{\sigma(c)}$, with distinct $\sigma(c)$ for distinct columns (one 1 per row and column). So $(P^\top P)_{cc'} = e_{\sigma(c)}^\top e_{\sigma(c')} = [c = c']$, i.e. $P^\top P = I$.

(b) $f(PAP^\top, PX) = PAP^\top P X = PA(P^\top P)X = PAX = P f(A,X)$: equivariant. Then $g(PAP^\top, PX) = \mathbf 1^\top P A X = (P^\top \mathbf 1)^\top AX = \mathbf 1^\top AX = g(A, X)$, because permuting the all-ones vector leaves it unchanged: invariant. (Summing node outputs is how an equivariant layer becomes an invariant read-out.)
</details>

**Exercise 3 (★★, math).** For the path graph 0 – 1 – 2, compute $\hat A$ by hand, then $\hat A x$ and $\hat A^2 x$ for $x = (1, 0, 0)$. Which node first "hears" about node 0, and after how many layers? Check with NumPy.

<details><summary>Solution</summary>

$\tilde d = (2, 3, 2)$. $\hat A_{00} = \hat A_{22} = 1/2$, $\hat A_{11} = 1/3$, $\hat A_{01} = \hat A_{12} = 1/\sqrt6 \approx 0.4082$, $\hat A_{02} = 0$.

$\hat A x$ = first column of $\hat A$ = $(0.5, 0.4082, 0)$: after one layer node 2 knows nothing about node 0 (distance 2).

$\hat A^2 x = \hat A (0.5, 0.4082, 0)$:
* node 0: $0.5\cdot0.5 + 0.4082\cdot0.4082 = 0.25 + 0.1667 = 0.4167$
* node 1: $0.4082\cdot0.5 + 0.3333\cdot0.4082 = 0.2041 + 0.1361 = 0.3402$
* node 2: $0 + 0.4082\cdot0.4082 = 0.1667$

Node 2 first hears about node 0 after **2 layers**, exactly its distance — the receptive-field theorem of Section 3.5.

```python
# file: c3_ex03_path_graph.py
import numpy as np
np.set_printoptions(precision=4, suppress=True)

A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
At = A + np.eye(3); d = At.sum(1) ** -0.5
A_hat = d[:, None] * At * d[None, :]
x = np.array([1., 0., 0.])
print("A_hat =\n", A_hat)
print("A_hat x   =", A_hat @ x)
print("A_hat^2 x =", A_hat @ A_hat @ x)
```

```text
A_hat =
 [[0.5    0.4082 0.    ]
 [0.4082 0.3333 0.4082]
 [0.     0.4082 0.5   ]]
A_hat x   = [0.5    0.4082 0.    ]
A_hat^2 x = [0.4167 0.3402 0.1667]
```
</details>

**Exercise 4 (★★, math).** Prove that the eigenvalues of $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$ lie in $[0, 2]$ (graph without isolated nodes). Deduce the eigenvalue range of $I + D^{-1/2}AD^{-1/2}$.

<details><summary>Solution</summary>

For any $x$, with $y_i = x_i/\sqrt{d_i}$ (i.e. $x = D^{1/2} y$):
$$x^\top L_{\text{sym}}\,x = \sum_{(i,j)\in E}(y_i - y_j)^2 \ \ge 0,$$
so $L_{\text{sym}}$ is positive semi-definite: every eigenvalue $\ge 0$.

Upper bound: $(y_i - y_j)^2 \le 2y_i^2 + 2y_j^2$ (since $0 \le (y_i + y_j)^2$). Summing over edges, each node $i$ appears in $d_i$ edges:
$$\sum_{(i,j)\in E}(y_i-y_j)^2 \le 2\sum_i d_i y_i^2 = 2\sum_i x_i^2 = 2\,x^\top x.$$
So the Rayleigh quotient $x^\top L_{\text{sym}}x / x^\top x \in [0, 2]$, and so are all eigenvalues (they are the Rayleigh quotients of the eigenvectors). Equality 2 needs $y_i = -y_j$ on every edge: a bipartite component.

Since $I + D^{-1/2}AD^{-1/2} = 2I - L_{\text{sym}}$, its eigenvalues are $2 - \lambda \in [0, 2]$ — the instability Kipf & Welling's renormalisation removes (Section 4.5; our example had $0.271, 0.5, 1.229, 2$).
</details>

**Exercise 5 (★★, math).** Show that $u = \tilde D^{1/2}\mathbf 1$ is an eigenvector of $\hat A$ with eigenvalue 1. Then, assuming the graph is connected (so all other eigenvalues have $|\mu| < 1$), write the limit of $\hat A^L X$ and explain in one sentence why it is useless for classification.

<details><summary>Solution</summary>

$\hat A u = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}\tilde D^{1/2}\mathbf 1 = \tilde D^{-1/2}\tilde A\mathbf 1 = \tilde D^{-1/2}\tilde d = \tilde D^{1/2}\mathbf 1 = u$, using $\tilde A\mathbf 1 = \tilde d$ (row sums are degrees) and $\tilde D^{-1/2}\tilde d = (\tilde d_i/\sqrt{\tilde d_i})_i = (\sqrt{\tilde d_i})_i$.

With $u_1 = u/\lVert u\rVert$ and the spectral decomposition $\hat A^L = \sum_k \mu_k^L u_k u_k^\top$, every term with $|\mu_k| < 1$ vanishes, so
$$\hat A^L X \to u_1 u_1^\top X,\qquad (\hat A^L X)_i \to \frac{\sqrt{\tilde d_i}}{\lVert u\rVert}\,(u_1^\top X).$$
Every node's representation becomes the same vector $u_1^\top X$ scaled by $\sqrt{\tilde d_i}$: the only node-specific information left is the degree, so classes cannot be separated (unless they differ only in degree).
</details>

**Exercise 6 (★★, math).** On Fdataset the drug input width is 2,092. Count the parameters (weights + biases) of one layer 2,092 → 64 for: (a) a GCN layer, (b) a GraphSAGE-mean layer (concatenation version), (c) a GIN layer whose MLP is Linear(2092, 64) – ReLU – Linear(64, 64), with a learnable $\epsilon$. Which count depends on the number of drugs?

<details><summary>Solution</summary>

(a) GCN: $2092\cdot64 + 64 = 133{,}952$.
(b) GraphSAGE: the linear layer reads $[h_i \Vert m_i]$ of width $2\cdot2092$: $4184\cdot64 + 64 = 267{,}840$ — twice the GCN, because self and neighbourhood get separate weights.
(c) GIN: $133{,}952 + (64\cdot64 + 64) + 1 = 133{,}952 + 4{,}160 + 1 = 138{,}113$.

None of the *layer* counts depends on the number of nodes as such. But the **input width 2,092 does** — it is $3\times593 + 313$, built from similarity rows and link rows (Section 9.2). That is why the first layer, and with it the model, is tied to Fdataset's node set.
</details>

**Exercise 7 (★★, coding).** Write `gcn_norm(W, add_self_loops=True)` that computes the symmetric normalisation for a possibly **weighted** adjacency, and gives an **isolated node** a zero row instead of NaN when self-loops are off. Test it on the weighted graph `[[0,.9,0,0],[.9,0,.5,0],[0,.5,0,0],[0,0,0,0]]` with and without self-loops.

<details><summary>Solution</summary>

```python
# file: c3_ex07_gcn_norm.py
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def gcn_norm(W, add_self_loops=True):
    """Symmetric GCN normalisation for a (possibly weighted) adjacency matrix.
    Isolated nodes (zero degree) get a zero row instead of a division by zero."""
    W = W.astype(float).copy()
    if add_self_loops:
        W = W + np.eye(len(W))
    deg = W.sum(1)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    return inv_sqrt[:, None] * W * inv_sqrt[None, :]

W = np.array([[0, .9, 0, 0], [.9, 0, .5, 0], [0, .5, 0, 0], [0, 0, 0, 0]])
print(gcn_norm(W, add_self_loops=False))
print(gcn_norm(W))
```

```text
[[0.     0.8018 0.     0.    ]
 [0.8018 0.     0.5976 0.    ]
 [0.     0.5976 0.     0.    ]
 [0.     0.     0.     0.    ]]
[[0.5263 0.4215 0.     0.    ]
 [0.4215 0.4167 0.2635 0.    ]
 [0.     0.2635 0.6667 0.    ]
 [0.     0.     0.     1.    ]]
```

Check one entry by hand: without self-loops, degrees are $(0.9, 1.4, 0.5, 0)$, so entry $(0,1) = 0.9/\sqrt{0.9\cdot1.4} = 0.9/1.1225 = 0.8018$. ✓ With self-loops the isolated node 3 gets $\hat A_{33} = 1$: it simply keeps its own features — exactly what happens to an uncovered drug in a project view (its only `knn_mask` edge is the self-loop).
</details>

**Exercise 8 (★, coding).** Given a list of per-layer embeddings `[h0, h1, h2]` (each 5 × 4), implement JK **max-pooling** and compare its output shape with JK **concatenation**. What extra information does max-pooling give for interpretation?

<details><summary>Solution</summary>

```python
# file: c3_ex08_jk_maxpool.py
import torch
torch.manual_seed(0)
layers_out = [torch.randn(5, 4) for _ in range(3)]          # h0, h1, h2 for 5 nodes
jk_cat = torch.cat(layers_out, 1)
jk_max = torch.stack(layers_out, 0).max(0).values
print("concat shape:", tuple(jk_cat.shape), " max-pool shape:", tuple(jk_max.shape))
print("max-pool picks, per node and dim, which layer won:\n",
      torch.stack(layers_out, 0).argmax(0).numpy())
```

```text
concat shape: (5, 12)  max-pool shape: (5, 4)
max-pool picks, per node and dim, which layer won:
 [[2 2 2 2]
 [2 2 2 0]
 [2 1 0 2]
 [0 0 0 2]
 [0 0 0 1]]
```

Concatenation keeps all $3\times4 = 12$ numbers (and leaves the choice of depth to the next layer — MV-HGAT's bilinear `W`); max-pooling keeps the width at 4 and selects, **per node and per feature**, which depth wins. The `argmax` table is a crude, node-specific read-out of "which receptive field this node relied on" (here node 0 used 2-hop information everywhere, node 3 mostly its own features). Note: max-pooling requires all layers to have the same width.
</details>

**Exercise 9 (★★, conceptual).** MV-HGAT has 2 layers and 8 relations (3 drug views, 3 disease views, `assoc>drug`, `assoc>disease`). List four different kinds of 2-hop information a **drug** receives, and say which one is the "collaborative filtering" signal.

<details><summary>Solution</summary>

In layer 2 a drug aggregates from its neighbours' *layer-1* states, which already mix all of *their* relations. So any composition of two relation steps that ends at a drug is possible, e.g.:

1. **drug ← (chem view) drug ← (chem view) drug** — chemically similar drugs of my chemically similar drugs (a wider chemical neighbourhood).
2. **drug ← (chem view) drug ← (assoc) disease** — the diseases treated by drugs similar to me ("guilt by association", the same idea as the propagation head).
3. **drug ← (assoc) disease ← (assoc) drug** — drugs that treat the same diseases as I do: the **collaborative-filtering** signal ("users who liked what you liked").
4. **drug ← (assoc) disease ← (disease view) disease** — diseases similar to the diseases I treat (e.g. by phenotype or MONDO position).

Mixed views also occur (e.g. drug ← `gene_r` drug ← `chem_ecfp` drug). Because a cold disease has no assoc edges, paths 3–4 vanish for its drugs' predictions, which is why the degree gate and propagation head exist.
</details>

**Exercise 10 (★★, conceptual).** A new drug is approved tomorrow and you want MV-HGAT's prediction of its indications. List everything in the pipeline that has to change, and propose the cheapest valid procedure.

<details><summary>Solution</summary>

What depends on the node set:

* the three drug similarity matrices (new row and column: compute Tanimoto of its fingerprints and Jaccard of its CTD genes with all 593 drugs);
* the input features: every drug's feature vector gets 3 new coordinates ("similarity to the new drug") and the new drug gets a full vector → the input layer `inp["drug"]` changes width (2,092 → 2,095);
* the kNN masks of each drug view (the new drug enters some drugs' top-10 lists, so `knn_mask` must be recomputed), the propagation kernels (`knn_kernel`), and the association matrix (a new all-zero row);
* the degree gate input (degree 0 for the new drug).

Cheapest *valid* procedure: add the drug to all matrices, re-run `build`, and **retrain** (the input layer's shape changed, so old weights cannot be loaded as they are; initialising the new input columns at zero and fine-tuning is a possible shortcut but must be validated). The new drug is a cold-start node, so predictions will lean on its similarity views — expect accuracy closer to the cold-start evaluation than to warm 5-fold CV. This is precisely what "transductive" implies.
</details>

**Exercise 11 (★★★, analysis).** Using this chapter's tools, explain two independent reasons why the project's LAGCN re-implementation might underperform, and propose a concrete code change for each.

<details><summary>Solution</summary>

1. **Over-smoothing / signal dilution from dense similarity blocks.** `H = [[Sr, A],[Aᵀ, Sd]]` uses the full similarity matrices: 100 % of drug–drug entries are non-zero, and on average about 97 % of a drug row's mass is similarity, only about 3 % associations (Section 12.4). One propagation step averages each drug over all drugs (Section 8.2). *Change:* sparsify the blocks, e.g. `Sr * knn_mask(Sr, 10)` and `Sd * knn_mask(Sd, 10)`, and possibly rescale the association block so links carry a fixed share of each row.
2. **Detect-vs-predict leakage.** Training links are in the graph and in the features (`H` is also the input), and the loss is computed on them. *Change:* each epoch hide a random fraction of training links from both the graph and the features and compute the loss only on the hidden links plus sampled negatives (what `supervise_hidden` does in `MVHGATMethod.fit_predict`). Setting `drop_edge > 0` alone is not enough, because the loss would still include the visible links.

(A third, smaller point: `_GCN` applies ReLU after the last layer, so the dot-product decoder only sees non-negative embeddings and cannot express "anti-correlated" profiles; drop the final ReLU or add a linear projection as NIMCGCN does.) Any such change must be tuned on the validation split, never on the test folds.
</details>

**Exercise 12 (★★, coding).** Compute the normalised Dirichlet energy $E(H) = \mathrm{tr}(H^\top\tilde L H)/\mathrm{tr}(H^\top H)$, $\tilde L = I - \hat A$, of $H^{(k)} = \hat A^k X$ for the 4-node example ($X$ from Section 5.2) at $k = 0, 1, 2, 5, 10, 20$. How fast does it decay, and why?

<details><summary>Solution</summary>

```python
# file: c3_ex12_dirichlet.py
import numpy as np

def gcn_norm(W):
    W = W + np.eye(len(W)); d = W.sum(1) ** -0.5
    return d[:, None] * W * d[None, :]

A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], float)
P = gcn_norm(A)
L = np.eye(4) - P                                  # L~ = I - A_hat
X = np.array([[1., 0.], [0., 2.], [1., 0.], [0., 1.]])
H = X.copy()
for k in range(0, 21):
    if k in (0, 1, 2, 5, 10, 20):
        E = np.trace(H.T @ L @ H) / np.trace(H.T @ H)
        print(f"layer {k:2d}: normalised Dirichlet energy = {E:.2e}")
    H = P @ H
```

```text
layer  0: normalised Dirichlet energy = 5.72e-01
layer  1: normalised Dirichlet energy = 1.31e-02
layer  2: normalised Dirichlet energy = 1.22e-03
layer  5: normalised Dirichlet energy = 3.22e-05
layer 10: normalised Dirichlet energy = 1.05e-07
layer 20: normalised Dirichlet energy = 1.13e-12
```

The energy falls by orders of magnitude: geometrically. Write $H = \sum_k u_k (u_k^\top H)$ in $\hat A$'s eigenbasis (eigenvalues $1, 0.564, 0, -0.148$). Only the $\mu=1$ direction has zero energy ($\tilde L u_1 = 0$); every other component is multiplied by $\mu_k$ per layer, so its share of the energy shrinks like $(\mu_k/1)^{2L}$, dominated by $|\mu_2| = 0.564$: roughly a factor $0.564^2 \approx 0.32$ per layer asymptotically. The huge drop in the first layer comes from the components with $\mu = 0$ and $-0.148$ (including the "D0 minus D1" direction), which one layer deletes almost entirely.
</details>

**Exercise 13 (★★★, math).** Prove: in a $d$-regular graph (every node has degree $d$) with identical input features $x_i = c$ for all $i$, **any** message-passing GNN produces identical outputs at all nodes, at every layer. Relate this to Section 7's hexagon-vs-two-triangles example.

<details><summary>Solution</summary>

Induction on the layer $l$. Base: $h_i^{(0)} = c$ for all $i$. Step: suppose $h_i^{(l)} = a$ for every node. Node $i$'s neighbour multiset is $\{\!\{a, \dots, a\}\!\}$ ($d$ copies) — the same for every node because the graph is $d$-regular. Each message $\psi(a, a)$ is the same, so the aggregate $\bigoplus$ of $d$ identical messages is the same vector $b$ at every node, and $h_i^{(l+1)} = \phi(a, b)$ is the same for all $i$. $\square$

Consequently every graph-level read-out depends only on $(N, d, c)$: a 6-cycle and two triangles (both 2-regular, 6 nodes) receive identical representations — the GIN outputs in Section 7.2 were identical for exactly this reason. Breaking the symmetry needs informative node features (as our similarity-row features are), or more expressive architectures (higher-order WL, subgraph GNNs, positional encodings).
</details>

**Exercise 14 (★, conceptual).** Why does `knn_mask` contain `keep = vals > 0`? What would happen without it for an uncovered disease in `gene_d`?

<details><summary>Solution</summary>

For an uncovered disease (no genes in CTD) `fill_missing` sets the whole row to 0 (except the diagonal, which is then masked to $-\infty$ for neighbour selection). Without `keep`, `argpartition` would still return $k = 10$ indices — essentially **arbitrary** diseases tied at similarity 0 — and the disease would receive messages from 10 random diseases (and, after symmetrisation, send messages to them). That injects noise and, worse, *fake evidence* that attention could latch onto. With `keep`, the disease has only its self-loop in this view; its representation in that relation is just its own transformed state, and the view attention (Units C4, C5) can learn to rely on its other views. This matches the project's rule "we never invent a similarity value".
</details>

---

## 15. Answers to the self-check questions (PREREQUISITES.md, unit C3)

### Q1. Why can a 2-layer GNN "see" 2-hop neighbours?

Because each layer lets a node read the **current** states of its 1-hop neighbours, and those states were themselves built from *their* 1-hop neighbours in the previous layer. Formally (Section 3.5), by induction: $h_i^{(0)}$ depends only on $x_i$; if $h_j^{(l)}$ depends on nodes within $l$ hops of $j$, then $h_i^{(l+1)}$, a function of $h_i^{(l)}$ and $\{h_j^{(l)}\}_{j\in\mathcal N(i)}$, depends on nodes within $l + 1$ hops of $i$. In matrix form (ignoring nonlinearities) two layers compute $\hat A(\hat A X W_1)W_2 = \hat A^2 X W_1W_2$, and $(\hat A^2)_{ik} \ne 0$ exactly when $k$ is reachable from $i$ in at most 2 steps (self-loops let a walk pause). Exercise 3 showed node 2 of a path first receiving node 0's signal after exactly 2 layers.

**Why it matters here.** In MV-HGAT the 2-hop paths include **drug → disease → drug** (drugs that treat the same diseases — collaborative filtering) and **drug → similar drug → disease** (guilt by association), see Exercise 9. One layer would only see direct links and direct similarities. More than two would see 3-hop neighbourhoods but risk over-smoothing and over-squashing (Q2), and the JK concat lets the decoder use 0-, 1- and 2-hop information side by side.

### Q2. What is over-smoothing, and how does the kNN sparsification help?

**Over-smoothing** is the tendency of node representations to become indistinguishable as GNN layers are stacked. Each GCN-style layer is a low-pass filter (Laplacian smoothing, Li et al. 2018): it replaces a node's state by a weighted average over its neighbourhood. Repeating it drives the representations towards the dominant eigenvector of the propagation matrix: $\hat A^L X \to u_1 u_1^\top X$, where every node holds the same vector scaled by $\sqrt{\tilde d_i}$ (Section 8.1, Exercise 5). Classes then cannot be separated: in the depth demo a plain GCN fell from 0.978 accuracy (2 layers) to about 0.55 (8+ layers). The speed of the collapse is set by $|\mu_2|$: the larger the spectral gap $1 - |\mu_2|$, the faster.

**How kNN helps — three effects.**

1. **Fewer cross-cluster edges.** A dense similarity matrix connects every drug to every other one, so one layer already averages over *all* drugs (in the demo: 400 cross-family edges, cosine similarity 0.84 after one step and 1.000 after four). Keeping only each node's 10 most similar neighbours leaves mostly within-family edges (10 cross-family edges in the demo); averaging then mixes like with like, which *sharpens* the family structure first (gap 0.06 → 0.68 after four steps).
2. **Smaller spectral gap → slower convergence.** The sparse graph had $|\mu_2| = 0.957$ against 0.371 for the dense one, so the useless limit is approached far more slowly; two layers sit comfortably in the useful regime.
3. **Noise removal.** Weak similarities (most of the dense matrix) are mostly noise; dropping them stops noise from being averaged in. On Fdataset, LAGCN's dense blocks put about 97 % of a drug's propagation weight on similarities to *all* 593 drugs (Section 12.4).

The project adds further protection: only 2 layers, a learned skip connection plus LayerNorm in every `HeteroLayer`, and the JK concatenation that keeps the un-smoothed $h^{(0)}$ available to the decoder.

### Q3. Why is our model transductive, and what does that imply for brand-new drugs?

**Transductive** means the model is trained and used on one fixed set of nodes; only *labels* (here: links) are hidden, not nodes. MV-HGAT is transductive because of how its inputs are defined (Section 9.2):

* a drug's input vector is its similarity to **each of the 593 training drugs** in three views plus its row of visible links to the 313 diseases (2,092 numbers on Fdataset); the first linear layer has one weight column per such coordinate, i.e. per *specific training drug and disease*;
* the kNN graphs, propagation kernels, degree vectors and association matrix are all built for exactly this node set.

**Implications for a brand-new drug.** It has no valid input vector (the model has no weights for "similarity to the new drug"), is absent from all graphs, and the model cannot score it without rebuilding the matrices and graphs and **retraining** (Exercise 10). Models cannot be transferred between Fdataset and Cdataset either. Two things the model *can* do: (i) predict new indications for drugs and diseases that are already in the graph (the repositioning use case), and (ii) handle **existing nodes with no known links** — cold-start diseases — because they still have similarity rows and kNN neighbours; the project trains for this explicitly (cold-start practice, degree gate, propagation head). An inductive variant would need intrinsic node features (e.g. raw fingerprints), neighbour lists computed against the training nodes at prediction time, and aggregators independent of the global node set (GraphSAGE-style).

---

## 16. Summary and cheat sheet

**One-paragraph summary.** Graphs have no node order, variable size and irregular neighbourhoods, so GNNs build layers from permutation-invariant aggregations of neighbour messages, shared across nodes (message passing: message → aggregate → update). The GCN layer $H' = \sigma(\hat A H W)$, $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$, is a first-order, renormalised approximation of a spectral low-pass filter. GraphSAGE samples neighbours and concatenates self and neighbourhood, enabling inductive, mini-batch learning; GIN uses sum aggregation to match the 1-WL test's power. An $L$-layer GNN sees $L$ hops; depth is limited by over-smoothing (repeated averaging → identical embeddings) and over-squashing (exponentially many messages through bottlenecks); kNN sparsification, residuals, normalisation, DropEdge and jumping knowledge mitigate them. Models whose features are indexed by the training nodes are transductive. Dense implementations suit small graphs like ours; edge lists scale. In link prediction, never supervise on edges the encoder can see.

**Cheat sheet**

| Item | Formula / fact |
|---|---|
| Invariance / equivariance | $f(PAP^\top, PX) = f(A,X)$ / $= P f(A,X)$ |
| Message passing | $h_i' = \phi\big(h_i, \bigoplus_{j\in\mathcal N(i)} \psi(h_i, h_j, e_{ij})\big)$ |
| Laplacian, smoothness | $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$; $x^\top L_{\text{sym}}x = \sum_{(i,j)\in E}(x_i/\sqrt{d_i} - x_j/\sqrt{d_j})^2$; eigenvalues in $[0,2]$ |
| GFT | $\hat x = U^\top x$; filter $y = U g(\Lambda) U^\top x$; polynomial in $L$ = $K$-hop local |
| Chebyshev | $T_0 = 1, T_1 = x, T_k = 2xT_{k-1} - T_{k-2}$; $\tilde L = 2L/\lambda_{\max} - I$ |
| GCN derivation | $K=1$, $\lambda_{\max}\approx2$, $\theta = \theta_0 = -\theta_1$ → $\theta(I + D^{-1/2}AD^{-1/2})$ → renormalise |
| GCN layer | $H' = \sigma(\hat A H W)$, $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$, eigenvalues in $(-1, 1]$ |
| Node-wise GCN | $h_i' = \sigma\big(\sum_{j\in\mathcal N(i)\cup\{i\}} W h_j / \sqrt{\tilde d_i\tilde d_j}\big)$ |
| GraphSAGE | $h_v' = \sigma(W[h_v \Vert \text{AGG}(\{h_u\}_{u\in\mathcal S(v)})])$, $\ell_2$-normalise; mean / max-pool / LSTM; sample $S_1=25, S_2=10$ |
| GIN | $h_v' = \text{MLP}((1+\epsilon)h_v + \sum_u h_u)$; as powerful as 1-WL |
| Aggregator power | sum (multiset) > mean (distribution) > max (set) |
| Receptive field | $L$ layers = $L$ hops; $(\hat A^L)_{ik}\neq0$ iff walk of length $\le L$ |
| Over-smoothing limit | $\hat A^L X \to u_1u_1^\top X$, $u_1 \propto \tilde D^{1/2}\mathbf 1$; rate $\lvert\mu_2\rvert^L$ |
| Remedies | shallow, kNN, residual/skip, LayerNorm/PairNorm, DropEdge, JK (concat / max / LSTM-attn) |
| Transductive | fixed node set; features indexed by nodes → new node needs retraining |
| Dense vs sparse | $O(N^2)$ vs $O(\lvert E\rvert)$ memory; dense OK for $\lesssim 10^4$ nodes |
| Project | k = 10 kNN masks with self-loops; 2 layers; skip + LayerNorm; JK concat $h^0\Vert h^1\Vert h^2$ (192-d); bilinear decoder; hidden-link supervision |

---

## 17. Further resources (all links checked)

**Courses and books**
* [Stanford CS224W: Machine Learning with Graphs — course site](https://web.stanford.edu/class/cs224w/) — slides and notes; the GNN lectures (model, design space, theory/GIN) are the best lecture companion to this chapter. *Free.*
* [CS224W lecture videos (YouTube playlist)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn) — Jure Leskovec's recorded lectures; in the 2021 series, lectures 6–9 cover GNN basics, design space, GAT and expressiveness. *Free.*
* [William L. Hamilton, *Graph Representation Learning* (book)](https://www.cs.mcgill.ca/~wlh/grl_book/) — Ch. 5 (the GNN model, message passing), Ch. 6 (GNNs in practice), Ch. 7 (spectral and WL theory). *Free pre-print PDF; printed edition paid.*
* [Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* proto-book (arXiv)](https://arxiv.org/abs/2104.13478) — the unifying symmetry view (invariance/equivariance) at research depth. *Free.* Companion site with lectures: [geometricdeeplearning.com](https://geometricdeeplearning.com/). *Free.*

**Visual explanations**
* [Distill: "A Gentle Introduction to Graph Neural Networks"](https://distill.pub/2021/gnn-intro/) — interactive, intuition-first tour of message passing and pooling. *Free.*
* [Distill: "Understanding Convolutions on Graphs"](https://distill.pub/2021/understanding-gnns/) — spectral vs spatial convolutions, ChebNet → GCN, with interactive figures. *Free.*
* [Thomas Kipf: "Graph Convolutional Networks" (blog post)](https://tkipf.github.io/graph-convolutional-networks/) — the GCN author's own short explanation. *Free.*
* [Michael Bronstein, ICLR 2021 keynote "Geometric Deep Learning: The Erlangen Programme of ML" (video)](https://www.youtube.com/watch?v=w6Pw4MOzMuo) — why symmetry is the organising principle. *Free.*
* [Petar Veličković, "Theoretical Foundations of Graph Neural Networks" (video)](https://www.youtube.com/watch?v=uF53xsT7mjc) — derives GNNs from permutation symmetry; excellent after this chapter. *Free.*

**Primary papers** (all free on arXiv unless marked)
* [Kipf & Welling 2017, "Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) — the GCN paper; read Sections 2–3.
* [Defferrard, Bresson & Vandergheynst 2016, "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (ChebNet)](https://arxiv.org/abs/1606.09375).
* [Hammond, Vandergheynst & Gribonval 2011, "Wavelets on Graphs via Spectral Graph Theory"](https://arxiv.org/abs/0912.3848) — origin of the Chebyshev approximation trick.
* [Gilmer et al. 2017, "Neural Message Passing for Quantum Chemistry"](https://arxiv.org/abs/1704.01212) — the MPNN template.
* [Hamilton, Ying & Leskovec 2017, "Inductive Representation Learning on Large Graphs" (GraphSAGE)](https://arxiv.org/abs/1706.02216); project page: [snap.stanford.edu/graphsage](https://snap.stanford.edu/graphsage/).
* [Xu, Hu, Leskovec & Jegelka 2019, "How Powerful are Graph Neural Networks?" (GIN)](https://arxiv.org/abs/1810.00826).
* [Xu et al. 2018, "Representation Learning on Graphs with Jumping Knowledge Networks"](https://arxiv.org/abs/1806.03536).
* [Li, Han & Wu 2018, "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning"](https://arxiv.org/abs/1801.07606) — GCN as Laplacian smoothing; over-smoothing.
* [Oono & Suzuki 2020, "Graph Neural Networks Exponentially Lose Expressive Power for Node Classification"](https://arxiv.org/abs/1905.10947) — the theory of over-smoothing with weights and ReLU.
* [Rusch, Bronstein & Mishra 2023, "A Survey on Oversmoothing in Graph Neural Networks"](https://arxiv.org/abs/2303.10993) — measures and remedies.
* [Alon & Yahav 2021, "On the Bottleneck of Graph Neural Networks and its Practical Implications"](https://arxiv.org/abs/2006.05205) — over-squashing.
* [Topping et al. 2022, "Understanding over-squashing and bottlenecks on graphs via curvature"](https://arxiv.org/abs/2111.14522).
* [Wu et al. 2019, "Simplifying Graph Convolutional Networks" (SGC)](https://arxiv.org/abs/1902.07153) — GCN without nonlinearities; the self-loop/low-pass analysis.
* [Rong et al. 2020, "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification"](https://arxiv.org/abs/1907.10903).
* [Zhao & Akoglu 2020, "PairNorm: Tackling Oversmoothing in GNNs"](https://arxiv.org/abs/1909.12223).
* [Chen et al. 2020, "Simple and Deep Graph Convolutional Networks" (GCNII)](https://arxiv.org/abs/2007.02133).
* [Errica et al. 2020, "A Fair Comparison of Graph Neural Networks for Graph Classification"](https://arxiv.org/abs/1912.09893) — a cautionary tale on evaluation, relevant to comparing baselines.

**Baselines used in this project**
* [Li et al. 2020, "Neural inductive matrix completion with graph convolutional networks for miRNA-disease association prediction" (NIMCGCN), *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btz965) — *paywalled journal* (DOI verified via Crossref; the publisher blocks automated checks).
* [Yu et al. 2021, "Predicting drug–disease associations through layer attention graph convolutional network" (LAGCN), *Briefings in Bioinformatics*](https://doi.org/10.1093/bib/bbaa243) — *paywalled journal* (DOI verified via Crossref).

**Practical**
* [PyTorch Geometric tutorial: "Creating Message Passing Networks"](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html) — how the edge-list (sparse) version of this chapter's layers is written in the standard library. *Free.*

---

## 18. Glossary

* **Adjacency matrix ($A$)** — $N\times N$ matrix with $A_{ij} \ne 0$ iff there is an edge $j \to i$ (weighted or 0/1).
* **Aggregation ($\bigoplus$)** — permutation-invariant function combining a multiset of neighbour messages into one vector (sum, mean, max, attention-weighted sum).
* **Chebyshev polynomials** — $T_0 = 1$, $T_1 = x$, $T_k = 2xT_{k-1} - T_{k-2}$; a stable polynomial basis on $[-1, 1]$ used by ChebNet.
* **Computation tree** — the unrolled $L$-hop neighbourhood that determines one node's $L$-layer embedding.
* **Dirichlet energy** — $x^\top L x$; measures how much a signal varies across edges; → 0 under over-smoothing.
* **DropEdge** — randomly removing edges each training epoch; regulariser that slows over-smoothing.
* **Equivariance (permutation)** — relabelling the input nodes relabels the output identically: $f(PAP^\top, PX) = Pf(A,X)$.
* **GCN** — graph convolutional network, layer $\sigma(\hat A H W)$ (Kipf & Welling 2017).
* **GIN** — graph isomorphism network; sum aggregation + MLP; as powerful as 1-WL.
* **Graph Fourier transform** — coordinates of a node signal in the Laplacian's eigenbasis, $\hat x = U^\top x$.
* **GraphSAGE** — sample-and-aggregate GNN with concatenated self/neighbour representations; inductive.
* **Homophily** — the tendency of connected nodes to share labels/properties; GCNs rely on it.
* **Inductive learning** — learning a rule applicable to nodes/graphs not seen during training.
* **Invariance (permutation)** — the output does not change when nodes are relabelled.
* **Jumping knowledge (JK)** — combining the outputs of all layers (concat / max / LSTM-attention) for each node.
* **kNN graph** — graph keeping each node's $k$ most similar nodes as neighbours (symmetrised in this project).
* **Laplacian** — $L = D - A$ (combinatorial) or $I - D^{-1/2}AD^{-1/2}$ (symmetric normalised).
* **Layer attention** — LAGCN's learned softmax weighting of layer outputs, shared by all nodes.
* **Message passing** — the GNN template: compute messages from neighbours, aggregate, update.
* **Over-smoothing** — node embeddings becoming indistinguishable as depth grows.
* **Over-squashing** — exponentially large receptive fields compressed into fixed-size vectors through bottlenecks; long-range information is lost.
* **Permutation matrix** — 0/1 matrix with one 1 per row and column; reorders nodes.
* **Receptive field** — the set of nodes whose inputs can influence a node's output; $L$ hops for $L$ layers.
* **Renormalisation trick** — replacing $I + D^{-1/2}AD^{-1/2}$ by $\tilde D^{-1/2}(A + I)\tilde D^{-1/2}$ to keep eigenvalues in $(-1, 1]$.
* **Residual / skip connection** — adding the layer input (or a linear map of it) to the layer output.
* **Self-loop** — an edge from a node to itself; lets a node keep its own information during aggregation.
* **Spectral gap** — $1 - \lvert\mu_2\rvert$ for the propagation matrix; large gap → fast mixing → fast over-smoothing.
* **Spectral filter** — $U g(\Lambda) U^\top$; rescales each graph frequency by $g(\lambda)$.
* **Stochastic block model (SBM)** — random graph with communities and different within/between edge probabilities.
* **Transductive learning** — training and predicting on one fixed node set; only labels are hidden.
* **Weisfeiler–Lehman (1-WL) test** — iterative colour refinement by hashing (own colour, neighbour colour multiset); upper bound on message-passing GNN power.
