# Unit C1 — Graph Theory Basics

*Nodes, edges and matrices: the language in which MV-HGAT's input is written.*

---

## 0. Before you start

**Prerequisites (earlier units in this course)**

| Unit | What you need from it here |
|---|---|
| A1 Python / NumPy / pandas | boolean masks, `argpartition`, fancy indexing, `np.repeat` |
| A2 Linear algebra | matrix products, transpose, symmetric matrices, eigenvalues, positive semi-definiteness |
| B3 Neural networks & PyTorch (helpful, not required) | to connect "neighbourhood" with "message passing" and masked attention |

**Estimated study time:** 14–20 hours (6 h reading and re-deriving, 5 h code, 5–8 h exercises,
1–2 h reading `data.py`, `similarity.py` and `MVHGATMethod.build` alongside Section 14).

**Learning objectives.** After this unit you will be able to:

1. Define graphs precisely — directed/undirected, weighted, with or without self-loops — and say which kind
   each relation of the project is.
2. Convert between edge lists, adjacency lists, dense adjacency matrices and sparse (COO/CSR) matrices,
   and estimate the memory each needs.
3. Compute degrees, the degree matrix and density, and prove the handshake lemma.
4. Define walks, paths, distance, $k$-hop neighbourhoods and connected components, and compute them with
   NumPy/SciPy.
5. Prove that $(A^k)_{ij}$ counts walks of length $k$ and use it (e.g. to count triangles or common
   neighbours).
6. Recognise bipartite and $k$-partite graphs, write their block adjacency matrices, and interpret
   $BB^\top$ as a projection.
7. Describe heterogeneous (typed, multi-relational) graphs with a schema, and write down the project's
   schema with every relation's direction and matrix shape.
8. Build a $k$-nearest-neighbour graph from a similarity matrix by hand and in code, explain asymmetry and
   the symmetrisation choices (OR, AND, averaging), and explain self-loops and zero-similarity rules.
9. Define meta-paths, compute their commuting matrices by matrix products, normalise them (count, Jaccard,
   cosine, PathSim), and explain exactly how the drug–gene–disease meta-path becomes the gene-bridge
   relation in this project.
10. Define the graph Laplacian $L=D-A$ and its normalised form, prove $x^\top Lx=\tfrac12\sum_{ij}A_{ij}(x_i-x_j)^2$,
    and relate its zero eigenvalues to connected components.
11. Read `knn_mask`, `knn_kernel`, `topk_bipartite`, `cosine_cross` and `MVHGATMethod.build` line by line.

---

## 1. Motivation: why this unit matters for *this* project

MV-HGAT does not see "drugs" and "diseases"; it sees **a graph**: a set of nodes and several sets of
edges, each stored as a boolean matrix. Everything the model can learn flows along those edges. The
default Fdataset graph that `MVHGATMethod.build` constructs has:

* **906 nodes of two types**: 593 drug nodes and 313 disease nodes;
* **8 relations** (edge types): three drug–drug similarity graphs (`chem_cdk`, `chem_ecfp`, `gene_r`),
  three disease–disease similarity graphs (`pheno_mim`, `sem_mondo`, `gene_d`), and the known
  indications in both directions (`assoc>drug`, `assoc>disease`); optionally 2 more (`gene_bridge`);
* **sparse** connectivity: each similarity graph keeps only about the $k=10$ most similar neighbours per node,
  so a drug has on average about 14 neighbours per view instead of 592;
* a **third node type** — genes — that never appears as a node: it is folded into the graph through
  **meta-paths** (drug–gene–drug, disease–gene–disease, drug–gene–disease).

A concrete example: the drug–drug `chem_ecfp` graph on Fdataset has 593 nodes and 4,067 undirected edges
(plus self-loops), but it is **not connected**: it has 16 connected components — one giant component of
578 drugs and 15 isolated drugs. Those 15 are drugs without a usable SMILES string (biologics such as
proteins, or retired DrugBank IDs), so no
fingerprint, so no chemical neighbours. Does the model break for them? No — because the graph is
*heterogeneous* and *multi-view*: those drugs still have neighbours in `chem_cdk`, `gene_r` and the
association relation. Understanding why requires exactly the vocabulary of this unit: components,
isolated nodes, self-loops, relations, views.

The two self-check questions for this unit in `docs/PREREQUISITES.md` are:

1. *Our graph has drug, disease and gene nodes. Which edge types exist?*
2. *How is a drug–gene–disease meta-path turned into a direct drug–disease relation?*

They are answered in depth in Section 17.

---

## 2. What is a graph?

### 2.1 Intuition

A graph is a set of things together with a record of which pairs of things are related. The things are
**nodes** (also called *vertices*); the relationships are **edges** (also called *links*). A road map, a
social network, a molecule (atoms and bonds) and a drug–disease indication table are all graphs. The
power of the abstraction is that the same algorithms (shortest paths, components, random walks, GNNs)
apply to all of them.

### 2.2 Definitions

**Undirected graph.** $G=(V,E)$ where $V$ is a finite set of nodes and $E$ is a set of unordered pairs
$\{u,v\}$ with $u,v\in V$. We write $n=|V|$ and $m=|E|$, and $u\sim v$ when $\{u,v\}\in E$ ("$u$ is adjacent to $v$").
The two nodes of an edge are its **endpoints**; the edge is **incident** to them.

**Directed graph (digraph).** $E$ is a set of *ordered* pairs $(u,v)$, drawn as arrows $u\to v$: $u$ is the
**source** (tail) and $v$ the **target** (head). $(u,v)\in E$ does not imply $(v,u)\in E$.

**Self-loop.** An edge from a node to itself, $\{v,v\}$ or $(v,v)$.

**Multigraph.** A graph allowing several parallel edges between the same pair of nodes. A graph with no
self-loops and no parallel edges is **simple**.

**Weighted graph.** Each edge carries a number $w_{uv}$ (a similarity, a strength, a distance). An
unweighted graph is the special case where every weight is 1.

```
  undirected, unweighted      directed                 weighted, with self-loop
     1 ─── 2                    1 ──► 2                    ┌─┐
     │   ╱                      ▲   ╱                      1.0
     │  ╱                       │  ▼                       └ 1 ──0.9── 2
     3 ─── 4                    3 ◄── 4                      │
                                                            0.8
                                                             3
```

### 2.3 The project's graphs in this vocabulary

| Graph in the project | Directed? | Weighted? | Self-loops? |
|---|---|---|---|
| A raw similarity view, e.g. `chem_ecfp` ($593\times593$ Tanimoto) | undirected (symmetric) | yes, in $[0,1]$ | yes, diagonal $=1$ |
| `knn_mask(S, 10)` used for message passing | made undirected by symmetrisation | no (boolean) | yes |
| `knn_mask(S, 10, symmetric=False)` inside `knn_kernel` | **directed** ($i\to$ its top-$k$) | weights re-attached and row-normalised | removed |
| Known indications $A$ ($593\times313$) | undirected bipartite; stored as two directed relations | no | n/a (two node types) |
| Gene bridge `topk_bipartite(B, 10)` | **directed** (top-$k$ per row is not symmetric) | no | n/a |

Even "undirected" relations are used *directionally* in message passing: a relation is always
"destination ← source", and the project stores each relation as a boolean mask of shape
**(destination nodes, source nodes)** — `DenseGAT`'s docstring says "dense boolean adjacency (dst x src)".
For an undirected graph between nodes of the same type the mask is symmetric, so the distinction
disappears; for drug–disease edges we need two relations, `assoc>drug` with mask $A$ (drugs receive
from diseases) and `assoc>disease` with mask $A^\top$ (diseases receive from drugs).

---

## 3. Representing a graph in a computer

### 3.1 Four representations

Take the undirected graph on nodes $\{1,2,3,4,5\}$ with edges $\{1,2\},\{1,3\},\{2,3\},\{3,4\},\{4,5\}$ —
a triangle 1–2–3 with a tail 3–4–5. We will use this graph (call it $G_5$) throughout the chapter.

```
      1
     / \
    2───3───4───5
```

**Edge list** — just the pairs: `[(1,2), (1,3), (2,3), (3,4), (4,5)]`. Compact ($O(m)$), what files
usually contain, but answering "who are the neighbours of 3?" needs a scan of all edges.

**Adjacency list** — for each node, the list of its neighbours:
`1: [2,3]   2: [1,3]   3: [1,2,4]   4: [3,5]   5: [4]`. $O(n+m)$ memory; neighbour lookup is fast. This
is the classic representation for graph algorithms (BFS, DFS).

**Adjacency matrix** — the $n\times n$ matrix with $A_{ij}=1$ if $i\sim j$ and 0 otherwise:
$$
A=\begin{pmatrix}
0&1&1&0&0\\
1&0&1&0&0\\
1&1&0&1&0\\
0&0&1&0&1\\
0&0&0&1&0
\end{pmatrix}.
$$
Properties: for an undirected graph $A=A^\top$ (symmetric); the diagonal holds self-loops; for a weighted
graph $A_{ij}=w_{ij}$; for a directed graph the usual convention is $A_{ij}=1$ iff $i\to j$ (the project's
masks use the *transposed* convention, $M_{ij}=1$ iff $j$ sends to $i$; for symmetric relations they
coincide). Memory $O(n^2)$ regardless of $m$; edge lookup is $O(1)$; and — the reason it dominates in
machine learning — graph operations become **matrix operations** that run fast on GPUs.

**Incidence matrix** — $n\times m$, $B_{ve}=1$ if node $v$ is an endpoint of edge $e$. Rarely used in ML but it
gives the Laplacian as $L=B_{\pm}B_{\pm}^\top$ (Section 12) when one endpoint of each column is given $-1$.

### 3.2 Converting between them in NumPy

```python
import numpy as np

n = 5
edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]          # 1-based labels as in the text
E = np.array(edges) - 1                                   # 0-based indices for NumPy

A = np.zeros((n, n), dtype=int)
A[E[:, 0], E[:, 1]] = 1
A[E[:, 1], E[:, 0]] = 1                                   # undirected: store both directions
print(A)
print("symmetric:", (A == A.T).all())

adj_list = {i + 1: (np.flatnonzero(A[i]) + 1).tolist() for i in range(n)}
print("adjacency list:", adj_list)

back = [(int(i) + 1, int(j) + 1) for i, j in zip(*np.nonzero(np.triu(A)))]   # each edge once
print("edge list:", back)
```

**Output:**
```text
[[0 1 1 0 0]
 [1 0 1 0 0]
 [1 1 0 1 0]
 [0 0 1 0 1]
 [0 0 0 1 0]]
symmetric: True
adjacency list: {1: [2, 3], 2: [1, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4]}
edge list: [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
```

---

## 4. Degree, density and the degree matrix

### 4.1 Definitions

* The **degree** of node $v$ in an undirected graph, $\deg(v)$, is the number of edges incident to it
  (a self-loop is conventionally counted twice in pure graph theory, but in ML code the row sum of $A$ —
  which counts it once — is what everyone uses). In matrix form, $\deg(i)=\sum_j A_{ij}$, i.e.
  $\mathbf d = A\mathbf 1$.
* In a directed graph: **out-degree** $\deg^+(i)=\sum_j A_{ij}$ (row sums) and **in-degree**
  $\deg^-(j)=\sum_i A_{ij}$ (column sums).
* In a weighted graph, the row sum $\sum_j w_{ij}$ is the **weighted degree** or **strength**.
* The **degree matrix** is the diagonal matrix $D=\operatorname{diag}(\mathbf d)$.
* A node of degree 0 is **isolated**; a node of very high degree relative to the rest is a **hub**.
* **Density** of a simple undirected graph: $\rho = m/\binom n2 = 2m/(n(n-1))$, the fraction of possible
  edges that are present. **Average degree** $\bar d = 2m/n$.

For $G_5$: $\mathbf d=(2,2,3,2,1)$, $m=5$, $\bar d=2$, $\rho=5/10=0.5$.

### 4.2 The handshake lemma

> **Lemma.** In any undirected graph without self-loops, $\sum_{v\in V}\deg(v) = 2m$.

**Proof.** Count the pairs (node, incident edge). Counting by nodes gives $\sum_v\deg(v)$; counting by
edges gives 2 per edge (each edge has two endpoints). Both count the same set. $\square$

**Corollaries.** (i) The number of odd-degree nodes is even. (ii) $\bar d = 2m/n$. (iii) In matrix
language: $\mathbf 1^\top A\mathbf 1 = 2m$ — the sum of all entries of a symmetric adjacency matrix counts
every edge twice. For a bipartite graph with biadjacency $B$ (Section 8), the analogue is: sum of left
degrees = sum of right degrees = number of edges $=\mathbf 1^\top B\mathbf 1$.

### 4.3 Degree in the project

* **Association degree.** In Fdataset, a drug's degree in the known-indication graph is its number of
  known indications: between 1 and 22, mean 3.26 ($=1933/593$). A disease's degree ranges from 1 to 84,
  mean 6.18 ($=1933/313$). Both sums equal 1,933 — the handshake lemma for bipartite graphs.
* **The degree gate.** `fit_predict` computes `degrees(Am)` $=(A_m\mathbf 1,\ A_m^\top\mathbf 1)$ — the
  visible link counts per drug and per disease — and `MVHGAT.gnn_gate` turns them into a trust factor via
  $\sigma(a+b\log(1+\deg))$. Low-degree (cold-start) nodes get a small gate.
* **Degree normalisation.** Dividing by degrees appears everywhere: `knn_kernel` divides each row by its
  weighted degree (random-walk normalisation $D^{-1}W$); `sym_norm` computes $D^{-1/2}SD^{-1/2}$ (Section 12).

```python
import numpy as np

A = np.array([[0, 1, 1, 0, 0],
              [1, 0, 1, 0, 0],
              [1, 1, 0, 1, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 0]])
d = A.sum(1)
n, m = len(A), A.sum() // 2
print("degrees:", d.tolist(), "| sum =", d.sum(), "= 2m =", 2 * m)
print("density:", 2 * m / (n * (n - 1)), "| average degree:", 2 * m / n)
print("degree matrix D:\n", np.diag(d))

# directed example: in- and out-degree are column and row sums
Adir = np.array([[0, 1, 0],
                 [0, 0, 1],
                 [1, 1, 0]])
print("out-degree:", Adir.sum(1).tolist(), "in-degree:", Adir.sum(0).tolist())
```

**Output:**
```text
degrees: [2, 2, 3, 2, 1] | sum = 10 = 2m = 10
density: 0.5 | average degree: 2.0
degree matrix D:
 [[2 0 0 0 0]
 [0 2 0 0 0]
 [0 0 3 0 0]
 [0 0 0 2 0]
 [0 0 0 0 1]]
out-degree: [1, 1, 2] in-degree: [1, 2, 1]
```

---

## 5. Neighbourhoods, walks, paths and distance

### 5.1 Neighbourhoods

* The (open) **neighbourhood** of $v$: $\mathcal N(v)=\{u: u\sim v\}$. The **closed** neighbourhood
  $\mathcal N[v]=\mathcal N(v)\cup\{v\}$. Adding self-loops to a graph turns open neighbourhoods into closed
  ones, which is why GNN code adds self-loops: a node should hear its own message too. `knn_mask` ends with
  `np.fill_diagonal(M, True)` for exactly this reason.
* The **$k$-hop neighbourhood** $\mathcal N_k(v)$ is the set of nodes at distance at most $k$ from $v$
  (definition of distance below). In $G_5$: $\mathcal N_1(1)=\{2,3\}$, $\mathcal N_2(1)=\{2,3,4\}$,
  $\mathcal N_3(1)=\{2,3,4,5\}$.

**Why $k$-hop neighbourhoods matter.** One message-passing layer lets each node aggregate information
from its 1-hop neighbours. Two layers: from neighbours' neighbours — the 2-hop neighbourhood. MV-HGAT has
2 layers, so a drug's final embedding depends on everything within 2 hops *in the union of all relations*
(e.g. drug → similar drug → disease that drug treats). This is its **receptive field** (Unit C3).

### 5.2 Walks, trails, paths, cycles

| Term | Definition | In $G_5$ |
|---|---|---|
| **Walk** of length $k$ | sequence $v_0,v_1,\dots,v_k$ with $v_{t-1}\sim v_t$; nodes and edges may repeat | $1,2,1,3$ (length 3) |
| **Trail** | walk with no repeated edge | $1,2,3,1$ |
| **Path** | walk with no repeated node | $1,3,4,5$ (length 3) |
| **Cycle** | closed trail $v_0=v_k$, $k\ge3$, no other repetition | $1,2,3,1$ (the triangle) |

The **length** is the number of edges. The **distance** $\operatorname{dist}(u,v)$ is the length of a shortest
path ($\infty$ if none exists). The **diameter** is the largest finite distance. In $G_5$,
$\operatorname{dist}(1,5)=3$ and the diameter is 3.

Walks are the natural object for *matrices* (Section 6); paths are the natural object for *distances*.

### 5.3 Breadth-first search (BFS)

BFS computes distances from a source in $O(n+m)$ time: visit the source (distance 0), then all its
neighbours (distance 1), then their unvisited neighbours (distance 2), and so on, using a queue. Visiting
"layer by layer" guarantees that the first time a node is reached is via a shortest path.

```python
from collections import deque
import numpy as np

A = np.array([[0, 1, 1, 0, 0],
              [1, 0, 1, 0, 0],
              [1, 1, 0, 1, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 0]])

def bfs_distances(A, source):
    dist = np.full(len(A), -1)          # -1 = not reached yet
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(A[u]):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist

D = np.array([bfs_distances(A, s) for s in range(len(A))])
print("distance matrix:\n", D)
print("diameter:", D.max())
for k in (1, 2, 3):
    print(f"{k}-hop neighbourhood of node 1:", (np.flatnonzero((D[0] >= 1) & (D[0] <= k)) + 1).tolist())
```

**Output:**
```text
distance matrix:
 [[0 1 1 2 3]
 [1 0 1 2 3]
 [1 1 0 1 2]
 [2 2 1 0 1]
 [3 3 2 1 0]]
diameter: 3
1-hop neighbourhood of node 1: [2, 3]
2-hop neighbourhood of node 1: [2, 3, 4]
3-hop neighbourhood of node 1: [2, 3, 4, 5]
```

---

## 6. Powers of the adjacency matrix count walks

### 6.1 The theorem

> **Theorem.** Let $A$ be the adjacency matrix of a graph (directed or undirected, self-loops allowed).
> For every $k\ge1$, $(A^k)_{ij}$ equals the number of walks of length $k$ from $i$ to $j$.

**Proof** by induction on $k$. For $k=1$, $A_{ij}$ is 1 iff there is an edge $i\to j$, i.e. a walk of length
1. Suppose the claim holds for $k$. Every walk of length $k+1$ from $i$ to $j$ consists of a walk of length
$k$ from $i$ to some node $\ell$, followed by an edge $\ell\to j$. Grouping walks by their second-to-last
node $\ell$,
$$
\#\{\text{walks}_{k+1}(i\to j)\} = \sum_\ell \#\{\text{walks}_k(i\to\ell)\}\cdot A_{\ell j} = \sum_\ell (A^k)_{i\ell}A_{\ell j} = (A^{k+1})_{ij}. \qquad\square
$$

**Weighted version.** If $A$ holds weights, $(A^k)_{ij}=\sum_{\text{walks}}\prod_{\text{edges}}w$: the sum, over
all walks of length $k$, of the product of the edge weights along the walk. This is the form that matters
for similarity propagation: "how strongly is $i$ connected to $j$ through 2-step chains of similar
entities?"

### 6.2 Worked example on $G_5$

For an undirected simple graph, $(A^2)_{ij}=\sum_\ell A_{i\ell}A_{\ell j}=|\mathcal N(i)\cap\mathcal N(j)|$, the number of
**common neighbours**, and $(A^2)_{ii}=\deg(i)$ (walk out along an edge and straight back). Computing
the intersections from the adjacency list ($\mathcal N(1)=\{2,3\}$, $\mathcal N(2)=\{1,3\}$, $\mathcal N(3)=\{1,2,4\}$,
$\mathcal N(4)=\{3,5\}$, $\mathcal N(5)=\{4\}$):
$$
A^2=\begin{pmatrix}
2&1&1&1&0\\
1&2&1&1&0\\
1&1&3&0&1\\
1&1&0&2&0\\
0&0&1&0&1
\end{pmatrix}.
$$
Check one entry: $(A^2)_{14}=|\{2,3\}\cap\{3,5\}|=|\{3\}|=1$ — the single walk $1\to3\to4$. And $(A^2)_{34}=0$:
nodes 3 and 4 are adjacent but have no common neighbour, so there is no walk of length exactly 2.

Then $A^3=A^2A$. Row 1: $(A^3)_{1j}=\sum_\ell (A^2)_{1\ell}A_{\ell j}$; e.g.
$(A^3)_{13}=(A^2)_{11}A_{13}+(A^2)_{12}A_{23}+(A^2)_{14}A_{43}=2+1+1=4$. The four walks of length 3 from 1 to 3 are
$1{-}2{-}1{-}3$, $1{-}3{-}1{-}3$, $1{-}3{-}2{-}3$ and $1{-}3{-}4{-}3$ (the first factor, $(A^2)_{11}=2$, accounts for the
first two, which pass through node 1 at step 2). The full result:
$$
A^3=\begin{pmatrix}
2&3&4&1&1\\
3&2&4&1&1\\
4&4&2&4&0\\
1&1&4&0&2\\
1&1&0&2&0
\end{pmatrix}.
$$

**Triangles.** A closed walk of length 3 is a triangle traversed from one of its 3 nodes in one of 2
directions, so
$$
\#\text{triangles} = \frac{\operatorname{tr}(A^3)}{6} = \frac{2+2+2+0+0}{6}=1. \checkmark
$$

**Reachability.** $j$ is within $k$ hops of $i$ iff $\big((I+A)^k\big)_{ij}>0$ (the identity lets walks
"wait", turning "exactly $k$" into "at most $k$"). Equivalently, with self-loops already in the mask (as
in `knn_mask`), powers of the mask itself describe $k$-hop reachability — the receptive field of a $k$-layer
GNN.

```python
import numpy as np

A = np.array([[0, 1, 1, 0, 0],
              [1, 0, 1, 0, 0],
              [1, 1, 0, 1, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 0, 1, 0]])
A2, A3 = A @ A, A @ A @ A
print("A^2:\n", A2)
print("A^3:\n", A3)
print("diag(A^2) == degrees:", np.array_equal(np.diag(A2), A.sum(1)))
print("triangles = trace(A^3)/6 =", np.trace(A3) // 6)

# brute-force check: enumerate all walks of length 3 from node 1 to node 3 (0-based 0 -> 2)
walks = [(0, a, b, 2) for a in range(5) for b in range(5) if A[0, a] and A[a, b] and A[b, 2]]
print("walks 1->3 of length 3:", [tuple(v + 1 for v in w) for w in walks])

reach2 = np.linalg.matrix_power(np.eye(5, dtype=int) + A, 2) > 0
print("within 2 hops of node 1:", (np.flatnonzero(reach2[0]) + 1).tolist())
```

**Output:**
```text
A^2:
 [[2 1 1 1 0]
 [1 2 1 1 0]
 [1 1 3 0 1]
 [1 1 0 2 0]
 [0 0 1 0 1]]
A^3:
 [[2 3 4 1 1]
 [3 2 4 1 1]
 [4 4 2 4 0]
 [1 1 4 0 2]
 [1 1 0 2 0]]
diag(A^2) == degrees: True
triangles = trace(A^3)/6 = 1
walks 1->3 of length 3: [(1, 2, 1, 3), (1, 3, 1, 3), (1, 3, 2, 3), (1, 3, 4, 3)]
within 2 hops of node 1: [1, 2, 3, 4]
```

### 6.3 Why this is the heart of graph learning

* **Propagation = multiplication.** If $x$ holds one number per node, $(Ax)_i=\sum_{j\in\mathcal N(i)}x_j$ sums
  the neighbours' values; $(A^2x)_i$ sums over 2-step walks. A GCN layer is essentially
  $\sigma(\hat A H W)$ — one multiplication by a normalised adjacency per layer (Unit C3).
* **Meta-paths = products of different matrices** (Section 11): a drug–gene–disease walk count is
  $G_rG_d^\top$, a product of two *different* relation matrices.
* **Walk counts grow fast.** Entries of $A^k$ grow roughly like $\lambda_{\max}^k$ (the largest eigenvalue of
  $A$), and hub nodes accumulate huge counts. This is why propagation is always **normalised** by degrees
  (Section 12), and one reason deep GNNs over-smooth.

---

## 7. Connectivity and connected components

### 7.1 Definitions

* An undirected graph is **connected** if every pair of nodes is joined by a path.
* A **connected component** is a maximal connected subgraph: a set of nodes all mutually reachable, with
  no edges to the outside. The components partition $V$. An isolated node is a component of size 1.
* For directed graphs: **strongly connected** if every node can reach every other following arrow
  directions; **weakly connected** if it is connected once directions are ignored. Strongly connected
  components (SCCs) are the maximal strongly connected subsets.
* A **giant component** is a component containing a large fraction of all nodes.

Components are found by repeated BFS/DFS: start a BFS from an unlabelled node, label everything it reaches
with a new component id, repeat. Total cost $O(n+m)$. SciPy provides it as
`scipy.sparse.csgraph.connected_components`.

**Block structure.** If you reorder the nodes component by component, the adjacency matrix becomes
**block-diagonal** — no edges between blocks. This is the matrix fact behind Section 12's theorem on the
Laplacian's zero eigenvalues.

### 7.2 Connectivity of the project's graphs

The table below was computed from the processed Fdataset with the project's own `knn_mask(S, 10)` (the
code is in Section 14.6):

| View | Nodes | Undirected edges (excl. self-loops) | Degree min / mean / max | Components | Largest | Isolated |
|---|---|---|---|---|---|---|
| `chem_cdk` | 593 | 4,384 | 10 / 14.8 / 94 | 1 | 593 | 0 |
| `chem_ecfp` | 593 | 4,067 | 0 / 13.7 / 49 | 16 | 578 | 15 |
| `gene_r` | 593 | 3,866 | 0 / 13.0 / 49 | 82 | 512 | 81 |
| `pheno_mim` | 313 | 2,124 | 10 / 13.6 / 31 | 1 | 313 | 0 |
| `sem_mondo` | 313 | 2,492 | 0 / 15.9 / 128 | 11 | 271 | 9 |
| `gene_d` | 313 | 782 | 0 / 5.0 / 40 | 157 | 153 | 153 |

Lessons:

1. **Benchmark views are complete**, so every node gets its $k=10$ neighbours (minimum degree exactly 10) and
   the graph is connected.
2. **New views have holes.** Entities with no data (no SMILES, no CTD genes, not in MONDO) are isolated:
   15 drugs in `chem_ecfp`, 81 in `gene_r`, 9 diseases in `sem_mondo`, 111 in `gene_d`.
3. **Isolated ≠ missing data only.** In the processed `gene_d` matrix, 111 diseases have no curated genes
   (all-NaN rows), yet 153 are isolated:
   42 diseases *have* genes but share none with any other disease, so all their Jaccard
   similarities are 0, and `knn_mask` refuses to link on zero similarity (`keep = vals > 0`).
4. **Why the model survives.** An isolated node in one view still has a self-loop there (so its message in
   that view is just its own projected features) and has real neighbours in other views. View attention
   can learn to down-weight the uninformative view for exactly those nodes — one of the motivations for
   *node-specific* view attention (Unit C5).
5. **Hubs appear after symmetrisation.** `sem_mondo` has a node of degree 128 although every node chooses
   only 10 neighbours: 118 other diseases chose it (Section 10.4 explains this "hubness").

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# G5 plus a separate edge 6-7 and an isolated node 8
edges = [(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (6, 7)]
n = 8
A = np.zeros((n, n), dtype=int)
for u, v in edges:
    A[u - 1, v - 1] = A[v - 1, u - 1] = 1

def components_bfs(A):
    label = np.full(len(A), -1)
    c = 0
    for s in range(len(A)):
        if label[s] != -1:
            continue
        label[s] = c
        stack = [s]
        while stack:                               # depth-first; BFS works equally well
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if label[v] == -1:
                    label[v] = c
                    stack.append(v)
        c += 1
    return c, label

print("from scratch:", components_bfs(A))
print("scipy       :", connected_components(csr_matrix(A), directed=False))

# directed graph: weakly vs strongly connected
Ad = np.array([[0, 1, 0],
               [0, 0, 1],
               [0, 0, 0]])                         # 1 -> 2 -> 3, no way back
print("weak  :", connected_components(csr_matrix(Ad), directed=True, connection="weak")[0])
print("strong:", connected_components(csr_matrix(Ad), directed=True, connection="strong")[0])
```

**Output:**
```text
from scratch: (3, array([0, 0, 0, 0, 0, 1, 1, 2]))
scipy       : (3, array([0, 0, 0, 0, 0, 1, 1, 2], dtype=int32))
weak  : 1
strong: 3
```

---

## 8. Bipartite and $k$-partite graphs

### 8.1 Definitions

A graph is **bipartite** if its nodes can be split into two disjoint sets $U$ and $W$ such that every edge
joins a node of $U$ to a node of $W$ (no edges inside $U$ or inside $W$). More generally a graph is
**$k$-partite** if $V$ splits into $k$ parts with no edges inside any part. The drug–disease indication
graph is bipartite with $U=$ drugs and $W=$ diseases; the conceptual drug–gene–disease graph of CTD is
**tripartite** if we only consider drug–gene and gene–disease edges.

> **Theorem (König, 1936).** A graph is bipartite if and only if it contains no cycle of odd length.

*Proof sketch.* (⇒) Along any cycle the sides alternate $U,W,U,W,\dots$, so returning to the start takes
an even number of steps. (⇐) In each component pick a root $r$ and put $v$ in $U$ if $\operatorname{dist}(r,v)$ is
even, in $W$ otherwise. If an edge joined two nodes at distances of the same parity, the two shortest
paths plus that edge would contain an odd cycle. $\square$ (Algorithmically: 2-colour with BFS; a
conflict proves an odd cycle.) $G_5$ is not bipartite — it has the triangle 1–2–3.

### 8.2 The biadjacency matrix and the block adjacency matrix

For a bipartite graph we don't need an $(|U|+|W|)^2$ matrix: the **biadjacency matrix** $B\in\{0,1\}^{|U|\times|W|}$
with $B_{uw}=1$ iff $u\sim w$ says everything. **The project's association matrix $A$ (593 × 313) is
exactly the biadjacency matrix of the drug–disease graph.** Its row sums are drug degrees, its column sums
disease degrees.

If we do index all nodes together (drugs first, then diseases), the full adjacency matrix is the block
matrix
$$
\mathcal A=\begin{pmatrix}0 & B\\ B^\top & 0\end{pmatrix}.
$$
Squaring it shows something useful:
$$
\mathcal A^2=\begin{pmatrix}BB^\top & 0\\ 0 & B^\top B\end{pmatrix}.
$$
Walks of length 2 always return to the same side, and
* $(BB^\top)_{ii'}$ = number of diseases that drugs $i$ and $i'$ both treat (**co-indication** count);
* $(B^\top B)_{jj'}$ = number of drugs shared by diseases $j$ and $j'$.

These two matrices are the **one-mode projections** of the bipartite graph. In Fdataset, $AA^\top$ links
11,860 pairs of drugs that share at least one indication (the maximum overlap is 12 shared diseases).

**Heterogeneous block matrices in the baselines.** `DRRS` and `LAGCN` in `methods.py` build
`np.block([[Sr, A_train], [A_train.T, Sd]])` — a $906\times906$ matrix for Fdataset — i.e. a weighted
adjacency matrix of the graph whose nodes are all drugs and diseases, whose diagonal blocks are the
drug–drug and disease–disease similarity graphs and whose off-diagonal blocks are the bipartite indication
graph. It is the simplest way to put a heterogeneous graph into a single matrix (at the cost of forgetting
which edges are which type).

```python
import numpy as np

# 3 drugs x 4 diseases: a tiny "association matrix" = biadjacency matrix
B = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 0, 1]])
print("drug degrees:", B.sum(1).tolist(), "disease degrees:", B.sum(0).tolist(), "edges:", B.sum())

full = np.block([[np.zeros((3, 3), int), B], [B.T, np.zeros((4, 4), int)]])
sq = full @ full
print("off-diagonal blocks of the square are zero:", (sq[:3, 3:] == 0).all() and (sq[3:, :3] == 0).all())
print("B B^T (drug co-indication counts):\n", B @ B.T)
print("B^T B (disease shared-drug counts):\n", B.T @ B)

def is_bipartite(A):
    side = np.full(len(A), -1)
    for s in range(len(A)):
        if side[s] != -1:
            continue
        side[s], queue = 0, [s]
        while queue:
            u = queue.pop(0)
            for v in np.flatnonzero(A[u]):
                if side[v] == -1:
                    side[v] = 1 - side[u]; queue.append(v)
                elif side[v] == side[u]:
                    return False                  # odd cycle found
    return True

G5 = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
print("block graph bipartite:", is_bipartite(full), "| G5 bipartite:", is_bipartite(G5))
```

**Output:**
```text
drug degrees: [2, 2, 1] disease degrees: [1, 2, 1, 1] edges: 5
off-diagonal blocks of the square are zero: True
B B^T (drug co-indication counts):
 [[2 1 0]
 [1 2 0]
 [0 0 1]]
B^T B (disease shared-drug counts):
 [[1 1 0 0]
 [1 2 1 0]
 [0 1 1 0]
 [0 0 0 1]]
block graph bipartite: True | G5 bipartite: False
```

---

## 9. Heterogeneous and multi-relational graphs

### 9.1 Definitions

A **heterogeneous graph** (also *heterogeneous information network*, HIN) is a graph
$G=(V,E,\phi,\psi)$ with a **node-type map** $\phi:V\to\mathcal T_V$ and an **edge-type (relation) map**
$\psi:E\to\mathcal T_E$, where $|\mathcal T_V|+|\mathcal T_E|>2$ (more than one node type or more than one edge type).
Each relation $r\in\mathcal T_E$ connects a fixed source type to a fixed target type. The **network schema**
is the small "type-level" graph whose nodes are the node types and whose edges are the relations.

Related terms you will meet in papers:

* **Multi-relational graph** — one node set, several edge types; often written as triples
  $(h, r, t)$ ("head, relation, tail"), the format of **knowledge graphs** (e.g. *(aspirin, treats,
  headache)*).
* **Multiplex / multi-view / multi-layer graph** — the same node set with several edge sets, each a
  different "view" of how nodes relate. The project's three drug similarity graphs are a 3-layer multiplex
  graph on the 593 drugs.
* **Homogeneous graph** — one node type and one edge type (e.g. a single kNN similarity graph).

A heterogeneous graph is stored as **one adjacency matrix per relation**, of shape
(#target-type nodes × #source-type nodes) — in the project, a dictionary `graphs[relation_name]` of boolean
masks plus a dictionary `relations[relation_name] = (dst_type, src_type)`.

### 9.2 The project's schema

```
                     view:chem_cdk                            view:pheno_mim
                     view:chem_ecfp   ┌──────────┐            view:sem_mondo   ┌───────────┐
                     view:gene_r  ┌──►│   DRUG   │            view:gene_d  ┌──►│  DISEASE  │
                                  └───│  (593)   │                         └───│   (313)   │
                                      └──────────┘                             └───────────┘
                                         ▲    │        assoc>disease (Aᵀ)         ▲   │
                                         │    └──────────────────────────────────►┘   │
                                         └────────────────────────────────────────────┘
                                                       assoc>drug (A)
                         (optional)  gene_bridge>drug / gene_bridge>disease, from the
                                     drug→gene→disease meta-path (Section 11)
```

| Relation | (dst type, src type) | Mask shape (Fdataset) | Built by | Symmetric? |
|---|---|---|---|---|
| `view:chem_cdk`, `view:chem_ecfp`, `view:gene_r` | (drug, drug) | 593 × 593 | `knn_mask(S, 10)` | yes |
| `view:pheno_mim`, `view:sem_mondo`, `view:gene_d` | (disease, disease) | 313 × 313 | `knn_mask(S, 10)` | yes |
| `assoc>drug` | (drug, disease) | 593 × 313 | `A_train > 0` (visible links) | n/a |
| `assoc>disease` | (disease, drug) | 313 × 593 | its transpose | n/a |
| `gene_bridge>drug` (optional) | (drug, disease) | 593 × 313 | `topk_bipartite(B, 10)` | n/a |
| `gene_bridge>disease` (optional) | (disease, drug) | 313 × 593 | `topk_bipartite(B.T, 10)` | n/a, and **not** the transpose of the previous |

Two node types and 8 relation types by default (10 with the bridge). Notice that `assoc>disease` is
exactly the transpose of `assoc>drug` (an indication is symmetric: if drug $i$ treats disease $j$, disease
$j$ is treated by drug $i$), whereas the two bridge masks are built separately (each drug keeps its top-10
diseases; each disease keeps its top-10 drugs), so a drug→disease bridge edge need not have a reverse edge.
On Fdataset the drug-side bridge mask has 4,435 edges and the disease-side one 1,809.

**Why one matrix per relation rather than one big matrix?** Because the model gives every relation its
own parameters (`DenseGAT` per relation in a `ModuleDict`) and learns how much each node should trust
each relation (view attention). Merging relations into one matrix, as DRRS/LAGCN do, would throw away the
information "this edge came from chemistry, that one from phenotypes", which is exactly what the project
wants to exploit and interpret.

---

## 10. From a similarity matrix to a $k$-nearest-neighbour graph

### 10.1 A similarity matrix is already a graph — just a useless one

A similarity matrix $S\in[0,1]^{n\times n}$ (e.g. Tanimoto similarity of fingerprints) *is* the weighted
adjacency matrix of a **complete** graph: every pair connected with weight $S_{ij}$. For message passing
this is bad for two reasons:

1. **Over-smoothing.** If every drug aggregates from all 592 others, every embedding becomes nearly the
   global average; nodes become indistinguishable (Unit C3).
2. **Noise.** Most small similarities (Tanimoto 0.1–0.2 between unrelated molecules) carry no signal, but
   there are so many of them that together they swamp the few informative ones.

So we **sparsify**: keep only strong edges. Two standard ways:

* **$\varepsilon$-graph (threshold):** keep $\{i,j\}$ if $S_{ij}\ge\varepsilon$. Symmetric automatically, but node
  degrees can vary wildly — a node in a dense region gets hundreds of neighbours, a node in a sparse
  region none.
* **$k$-nearest-neighbour (kNN) graph:** each node keeps its $k$ most similar *other* nodes. Every node
  gets (up to) $k$ neighbours regardless of local density. This is the project's choice, with $k=10$.

### 10.2 kNN graphs are directed — and need symmetrising

"$j$ is among $i$'s $k$ nearest neighbours" does not imply "$i$ is among $j$'s". The raw kNN relation is a
**directed** graph with out-degree exactly $k$. To obtain an undirected graph we **symmetrise** its mask
$M$ ($M_{ij}=1$ iff $j\in\text{kNN}(i)$):

| Rule | Mask | Keeps $\{i,j\}$ if | Degrees | Effect |
|---|---|---|---|---|
| **OR** (union) | $M\lor M^\top$ | $i$ chose $j$ **or** $j$ chose $i$ | $\ge k$, can be large (hubs) | connected, robust; the project's choice (`M \|= M.T`) |
| **AND** (mutual kNN) | $M\land M^\top$ | both chose each other | $\le k$, can be 0 | very clean edges, but fragments the graph |
| **Average** (weighted) | $(W+W^\top)/2$ | as OR, weight halved if one-sided | — | used for weighted spectral methods |

Then add **self-loops** so each node keeps its own information (`np.fill_diagonal(M, True)`).

### 10.3 Worked example: a kNN graph from a 5 × 5 similarity matrix, $k=2$

Five drugs $a,b,c,d,e$ with similarity
$$
S=\begin{array}{c|ccccc}
 & a & b & c & d & e\\\hline
a & 1 & .9 & .8 & .1 & 0\\
b & .9 & 1 & .7 & .2 & .1\\
c & .8 & .7 & 1 & .6 & .3\\
d & .1 & .2 & .6 & 1 & .4\\
e & 0 & .1 & .3 & .4 & 1
\end{array}
$$

**Step 1 — each row's top-2, ignoring the diagonal.**

| Node | Similarities to others (sorted) | kNN ($k=2$) |
|---|---|---|
| $a$ | $b$ .9, $c$ .8, $d$ .1, $e$ 0 | $\{b,c\}$ |
| $b$ | $a$ .9, $c$ .7, $d$ .2, $e$ .1 | $\{a,c\}$ |
| $c$ | $a$ .8, $b$ .7, $d$ .6, $e$ .3 | $\{a,b\}$ |
| $d$ | $c$ .6, $e$ .4, $b$ .2, $a$ .1 | $\{c,e\}$ |
| $e$ | $d$ .4, $c$ .3, $b$ .1, $a$ 0 | $\{d,c\}$ |

Directed edges: $a{\to}b, a{\to}c, b{\to}a, b{\to}c, c{\to}a, c{\to}b, d{\to}c, d{\to}e, e{\to}d, e{\to}c$.

**Step 2 — OR-symmetrisation.** Undirected edges: $ab, ac, bc$ (all chosen both ways), $de$ (both ways),
$cd$ (only $d$ chose $c$), $ce$ (only $e$ chose $c$). Degrees: $a{:}2,\ b{:}2,\ c{:}4,\ d{:}2,\ e{:}2$. Node $c$ has
degree $4>k$: it is a *hub* — popular as a neighbour even though it did not choose $d$ or $e$ itself.

**Step 2′ — AND (mutual) instead.** Only $ab, ac, bc, de$ survive: the graph splits into **two components**
$\{a,b,c\}$ and $\{d,e\}$. Mutual kNN is cleaner but can disconnect a graph — another reason the project
uses OR.

**Step 3 — self-loops** on all five nodes.

**Step 4 — the row-normalised kNN kernel** (`knn_kernel`, used by the propagation head; Unit C2). Keep the
*directed* top-$k$ with their similarity weights, no self-loops, and divide each row by its sum:

| Row | Neighbours and weights | Normalised |
|---|---|---|
| $a$ | $b$ .9, $c$ .8 | $b$: .9/1.7 = 0.529, $c$: 0.471 |
| $b$ | $a$ .9, $c$ .7 | $a$: 0.5625, $c$: 0.4375 |
| $c$ | $a$ .8, $b$ .7 | $a$: 0.533, $b$: 0.467 |
| $d$ | $c$ .6, $e$ .4 | $c$: 0.6, $e$: 0.4 |
| $e$ | $d$ .4, $c$ .3 | $d$: 0.571, $c$: 0.429 |

If drug $c$ is known to treat disease $X$ and nobody else does, then $(KA)_{\cdot,X}$ gives
$a{:}\ 0.471,\ b{:}\ 0.4375,\ c{:}\ 0,\ d{:}\ 0.6,\ e{:}\ 0.429$ — "what share of my nearest neighbours' similarity mass
treats $X$". This is the guilt-by-association score of the propagation head, a weighted walk of length 1
in the kNN graph followed by a step along an indication edge (a drug–drug–disease meta-path, Section 11).

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import knn_mask, knn_kernel

S = np.array([[1.0, 0.9, 0.8, 0.1, 0.0],
              [0.9, 1.0, 0.7, 0.2, 0.1],
              [0.8, 0.7, 1.0, 0.6, 0.3],
              [0.1, 0.2, 0.6, 1.0, 0.4],
              [0.0, 0.1, 0.3, 0.4, 1.0]])
names = np.array(list("abcde"))

def knn_directed(S, k):
    """From scratch: M[i, j] = True iff j is one of i's k most similar OTHER nodes."""
    work = S.astype(float).copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1)[:, :k]
    M = np.zeros_like(S, dtype=bool)
    M[np.repeat(np.arange(len(S)), k), order.ravel()] = True
    return M

M = knn_directed(S, 2)
print("directed kNN:", {str(names[i]): names[M[i]].tolist() for i in range(5)})
OR, AND = M | M.T, M & M.T
print("OR  degrees:", dict(zip(names.tolist(), OR.sum(1).tolist())))
print("AND degrees:", dict(zip(names.tolist(), AND.sum(1).tolist())))

OR_loops = OR.copy(); np.fill_diagonal(OR_loops, True)
print("matches project's knn_mask(S, 2):", np.array_equal(OR_loops, knn_mask(S, 2)))
print("matches knn_mask(S, 2, symmetric=False) minus diagonal:",
      np.array_equal(M, knn_mask(S, 2, symmetric=False) & ~np.eye(5, dtype=bool)))

K = knn_kernel(S, 2)
print("kernel row sums:", K.sum(1).round(6).tolist())
A = np.zeros((5, 1)); A[2, 0] = 1                          # only drug c treats disease X
print("propagation score for X:", dict(zip(names.tolist(), (K @ A).ravel().round(4).tolist())))
```

**Output:**
```text
directed kNN: {'a': ['b', 'c'], 'b': ['a', 'c'], 'c': ['a', 'b'], 'd': ['c', 'e'], 'e': ['c', 'd']}
OR  degrees: {'a': 2, 'b': 2, 'c': 4, 'd': 2, 'e': 2}
AND degrees: {'a': 2, 'b': 2, 'c': 2, 'd': 1, 'e': 1}
matches project's knn_mask(S, 2): True
matches knn_mask(S, 2, symmetric=False) minus diagonal: True
kernel row sums: [1.0, 1.0, 1.0, 1.0, 1.0]
propagation score for X: {'a': 0.4706, 'b': 0.4375, 'c': 0.0, 'd': 0.6, 'e': 0.4286}
```

### 10.4 Practical details that matter

* **Choice of $k$.** Small $k$: sparse, precise, but more isolated or weakly connected nodes; large $k$:
  robust but noisier, more smoothing. $k=10$ was chosen on a validation split (`05_sensitivity.py` varies it).
* **Zero similarity is not an edge.** If a node has fewer than $k$ positive similarities (e.g. a disease
  whose genes overlap with nobody), `knn_mask` keeps only the positive ones (`keep = vals > 0`).
  Otherwise `argpartition` would pick arbitrary zero-similarity "neighbours".
* **Missing data.** NaN rows (no fingerprint, no genes) are turned into "no similarity to anyone, 1 to
  myself" by `fill_missing`, so such a node is isolated except for its self-loop.
* **Ties.** If several candidates share the $k$-th largest similarity, `argpartition` picks among them
  arbitrarily (deterministically for a given input, but not by any meaningful rule). With discrete
  similarities (Jaccard of small sets, ontology-based scores) ties are common. Keeping *all* tied
  neighbours would be an alternative.
* **Hubness.** After OR-symmetrisation some nodes are chosen by many others and get degrees far above
  $k$ — up to 128 in `sem_mondo`, where many diseases share the same close ancestor in the ontology and
  therefore pick the same few "central" diseases. Hubs dominate message passing (they appear in many
  neighbourhoods), which is one reason attention (rather than plain averaging) helps: a node can learn to
  down-weight a hub that is similar to everyone.
* **`argpartition` vs `argsort`.** Selecting the top-$k$ of each row with `np.argpartition` is $O(n)$ per row
  versus $O(n\log n)$ for a full sort; it returns the top-$k$ *unordered*, which is all we need (Unit A1).

---

## 11. Meta-paths: turning multi-hop typed paths into relations

### 11.1 Definitions

A **meta-path** is a path *at the level of types* in the network schema:
$$
\mathcal P:\ T_0 \xrightarrow{R_1} T_1 \xrightarrow{R_2} \cdots \xrightarrow{R_\ell} T_\ell ,
$$
e.g. Drug $\xrightarrow{\text{interacts}}$ Gene $\xrightarrow{\text{associated}}$ Disease, abbreviated D–G–Di
(or "drug–gene–disease"). A **path instance** is a concrete path in the graph that follows the type
sequence, e.g. *metformin → PRKAA1 → type-2 diabetes*. Meta-paths were formalised for heterogeneous
information networks by Sun et al. (2011, PathSim) and are the backbone of HAN (Wang et al. 2019; Unit C5).

**Each meta-path defines a new relation between its end types**: "$x$ and $y$ are connected by at least
one instance of $\mathcal P$", with a strength that grows with the number of instances.

### 11.2 The commuting matrix: meta-paths are matrix products

Let $R_t$ be the adjacency (biadjacency) matrix of relation $t$, of shape $|T_{t-1}|\times|T_t|$. By exactly
the argument of Section 6.1,
$$
M_{\mathcal P} = R_1R_2\cdots R_\ell,\qquad (M_{\mathcal P})_{xy} = \#\{\text{path instances of }\mathcal P\text{ from }x\text{ to }y\}.
$$
$M_{\mathcal P}$ is called the **commuting matrix** of the meta-path. (Strictly, it counts *walks* following
the type pattern; for meta-paths whose intermediate types differ from the end types, such as D–G–Di,
walks and paths coincide.)

For the project, let $G_r\in\{0,1\}^{n_r\times n_g}$ be the drug–gene matrix (CTD chemical–gene interactions)
and $G_d\in\{0,1\}^{n_d\times n_g}$ the disease–gene matrix (CTD curated gene–disease links). Then:

| Meta-path | Commuting matrix | Meaning of entry | Becomes in the project |
|---|---|---|---|
| drug–gene–drug | $G_rG_r^\top$ ($n_r\times n_r$) | # genes shared by two drugs | Jaccard-normalised → view `gene_r` |
| disease–gene–disease | $G_dG_d^\top$ ($n_d\times n_d$) | # genes shared by two diseases | Jaccard-normalised → view `gene_d` |
| drug–gene–disease | $G_rG_d^\top$ ($n_r\times n_d$) | # genes linking drug and disease | cosine-normalised, top-10 → `gene_bridge` |
| drug–disease–drug | $AA^\top$ | # shared indications | (not used as a relation — it would leak labels into a "similarity") |
| drug–drug–disease | $K_rA$ (similarity kernel × indications) | similarity-weighted neighbours' indications | the **propagation head** $P_v$ |
| drug–disease–disease | $A K_d^\top$ | indications of similar diseases | the disease-side propagation slices |

The last rows show that the project's propagation head is *also* a set of meta-paths: for drug view $v$,
`K @ Am` is the drug–drug($v$)–disease meta-path, and for disease view $u$, `(K @ Am.T).T` is the
drug–disease–disease($u$) meta-path (`propagation()` in `fit_predict`).

### 11.3 Normalisation: raw counts are biased

Raw path counts favour **well-studied** entities: a drug with 500 CTD genes shares *some* genes with
almost every disease; a drug with 3 genes shares few with anything. CTD has a strong literature bias, so
raw counts mostly measure "how famous is this drug". Normalisations fix the scale:

* **Jaccard** (for same-type pairs, binary profiles): $J(x,y)=\dfrac{|g_x\cap g_y|}{|g_x\cup g_y|}=\dfrac{M_{xy}}{M_{xx}+M_{yy}-M_{xy}}$
  with $M=GG^\top$. Used by `similarity.jaccard` for `gene_r` and `gene_d`.
* **Cosine** (works across types): $\cos(x,y)=\dfrac{g_x\cdot g_y}{\lVert g_x\rVert\,\lVert g_y\rVert}=\dfrac{(G_1G_2^\top)_{xy}}{\sqrt{|g_x|}\sqrt{|g_y|}}$
  for binary profiles — the path count divided by the geometric mean of the two degrees. Used by
  `similarity.cosine_cross` for the bridge; its docstring calls it "the degree-normalised count of
  drug → gene → disease paths".
* **PathSim** (Sun et al. 2011) for symmetric meta-paths: $\dfrac{2M_{xy}}{M_{xx}+M_{yy}}$ — the number of path
  instances between $x$ and $y$ relative to the number of "round trips" from each to itself.
* **Row normalisation** $D^{-1}M$ — turns counts into transition probabilities of a random walk along the
  meta-path (Unit C2).

### 11.4 Worked example: from gene profiles to a bridge relation

Three drugs and two diseases over five genes $g_1..g_5$:
$$
G_r=\begin{array}{c|ccccc} & g_1&g_2&g_3&g_4&g_5\\\hline d_1&1&1&1&0&0\\ d_2&0&1&1&1&0\\ d_3&0&0&0&0&1\end{array},\qquad
G_d=\begin{array}{c|ccccc} & g_1&g_2&g_3&g_4&g_5\\\hline s_1&1&1&0&0&0\\ s_2&0&0&1&1&1\end{array}.
$$

**Counts** $C=G_rG_d^\top$: $d_1$–$s_1$: genes $\{g_1,g_2\}$ → 2; $d_1$–$s_2$: $\{g_3\}$ → 1; $d_2$–$s_1$: $\{g_2\}$ → 1;
$d_2$–$s_2$: $\{g_3,g_4\}$ → 2; $d_3$–$s_1$: 0; $d_3$–$s_2$: $\{g_5\}$ → 1.

**Cosine.** Drug gene counts $(3,3,1)$, disease gene counts $(2,3)$:
$$
\cos=\begin{pmatrix} 2/\sqrt{3\cdot2} & 1/\sqrt{3\cdot3}\\ 1/\sqrt{3\cdot2} & 2/\sqrt{3\cdot3}\\ 0 & 1/\sqrt{1\cdot3}\end{pmatrix}
=\begin{pmatrix}0.8165&0.3333\\0.4082&0.6667\\0&0.5774\end{pmatrix}.
$$
Note how $d_3$–$s_2$ (one shared gene) scores 0.577, *higher* than $d_1$–$s_2$ (also one shared gene, 0.333):
$d_3$'s single gene is entirely "about" $s_2$, whereas $d_1$ has two other genes elsewhere.

**Top-$k$ sparsification** (`topk_bipartite`, here $k=1$). Per drug (rows): $d_1\to s_1$, $d_2\to s_2$, $d_3\to s_2$
— this is the mask of `gene_bridge>drug`. Per disease (rows of the transpose): $s_1\to d_1$, $s_2\to d_2$ — the
mask of `gene_bridge>disease`. The two are not transposes: $d_3$ receives a message from $s_2$, but $s_2$
does not receive one from $d_3$.

**Drug–gene–drug (same-type) Jaccard.** $G_rG_r^\top=\begin{pmatrix}3&2&0\\2&3&0\\0&0&1\end{pmatrix}$, so
$J(d_1,d_2)=2/(3+3-2)=0.5$, $J(d_1,d_3)=J(d_2,d_3)=0$; PathSim$(d_1,d_2)=2\cdot2/(3+3)=0.667$.

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.similarity import jaccard, cosine_cross
from drepo.data import topk_bipartite

Gr = np.array([[1, 1, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 1]])       # drugs x genes
Gd = np.array([[1, 1, 0, 0, 0],
               [0, 0, 1, 1, 1]])       # diseases x genes

C = Gr @ Gd.T
print("drug-gene-disease path counts:\n", C)
B = cosine_cross(Gr, Gd)
manual = C / np.sqrt(Gr.sum(1))[:, None] / np.sqrt(Gd.sum(1))[None, :]
print("cosine bridge:\n", B.round(4), "\nmatches counts / sqrt(deg x deg):", np.allclose(B, manual))
print("gene_bridge>drug    (top-1 per drug):\n", topk_bipartite(B, 1).astype(int))
print("gene_bridge>disease (top-1 per disease):\n", topk_bipartite(B.T, 1).astype(int))

M = Gr @ Gr.T
print("drug-gene-drug counts:\n", M)
print("Jaccard (gene_r-style view):\n", jaccard(Gr))
pathsim = 2 * M / (np.diag(M)[:, None] + np.diag(M)[None, :])
print("PathSim:\n", pathsim.round(3))
```

**Output:**
```text
drug-gene-disease path counts:
 [[2 1]
 [1 2]
 [0 1]]
cosine bridge:
 [[0.8165 0.3333]
 [0.4082 0.6667]
 [0.     0.5774]] 
matches counts / sqrt(deg x deg): True
gene_bridge>drug    (top-1 per drug):
 [[1 0]
 [0 1]
 [0 1]]
gene_bridge>disease (top-1 per disease):
 [[1 0 0]
 [0 1 0]]
drug-gene-drug counts:
 [[3 2 0]
 [2 3 0]
 [0 0 1]]
Jaccard (gene_r-style view):
 [[1.  0.5 0. ]
 [0.5 1.  0. ]
 [0.  0.  1. ]]
PathSim:
 [[1.    0.667 0.   ]
 [0.667 1.    0.   ]
 [0.    0.    1.   ]]
```

### 11.5 Why fold genes into meta-paths instead of adding gene nodes?

* **Size.** CTD covers tens of thousands of genes. A dense attention layer over drug–gene edges would
  need $(n_r\times n_g\times H)$ tensors: $593\cdot20{,}000\cdot4$ floats $\approx190$ MB *per relation per layer*
  before autograd's copies, on a 6 GB GPU (Section 13).
* **Signal.** Gene nodes would have no input features of their own and would need to be learned from
  scratch with few edges each.
* **Standard practice.** HAN (Unit C5) handles intermediate node types exactly this way: each meta-path
  between target-type nodes becomes one relation.
* **The cost.** Folding loses the identity of the individual genes (we know *how many* genes connect a
  drug and a disease, not *which*), and in this project the bridge turned out to carry almost no signal on
  its own (AUC 0.52–0.57, HOW_IT_WORKS §3.2 and §10), so it is off by default (`use_bridge=False`) and
  tested in the ablation.

---

## 12. The graph Laplacian (introduction)

### 12.1 Definition and the key identity

For an undirected graph with (possibly weighted) symmetric adjacency $A$ and degree matrix $D$, the
(combinatorial) **graph Laplacian** is
$$
L = D - A.
$$
For $G_5$:
$$
L=\begin{pmatrix}
2&-1&-1&0&0\\
-1&2&-1&0&0\\
-1&-1&3&-1&0\\
0&0&-1&2&-1\\
0&0&0&-1&1
\end{pmatrix}.
$$
Each row sums to zero ($L\mathbf 1=\mathbf 0$).

> **Identity.** For every $x\in\mathbb R^n$,
> $$x^\top Lx=\frac12\sum_{i,j}A_{ij}(x_i-x_j)^2=\sum_{\{i,j\}\in E}w_{ij}(x_i-x_j)^2 .$$

**Proof.** Expand the right-hand side:
$\tfrac12\sum_{ij}A_{ij}(x_i^2-2x_ix_j+x_j^2) = \tfrac12\left(\sum_i d_ix_i^2 - 2\sum_{ij}A_{ij}x_ix_j + \sum_j d_jx_j^2\right)
= x^\top Dx - x^\top Ax = x^\top Lx$, using $\sum_jA_{ij}=d_i$ and the symmetry of $A$. $\square$

**Interpretation: $x^\top Lx$ measures how *non-smooth* a signal is on the graph.** It is small when
connected nodes have similar values and large when edges join very different values. Examples on $G_5$:
the constant signal $x=\mathbf 1$ gives 0; the indicator $x=(1,1,1,0,0)$ of the set $\{1,2,3\}$ gives
$x^\top Lx=1$ — exactly the number of edges leaving the set (the single edge 3–4). In general, for an
indicator vector, $x^\top Lx$ is the **cut size**.

### 12.2 Spectral facts (stated, with short proofs)

1. **$L$ is positive semi-definite**: $x^\top Lx\ge0$ (a sum of squares with non-negative weights), so all
   eigenvalues are $\ge0$.
2. **$0$ is always an eigenvalue**, with eigenvector $\mathbf 1$.
3. **The multiplicity of eigenvalue 0 equals the number of connected components.** *Proof:* $Lx=0$ ⇔
   $x^\top Lx=0$ (for PSD $L$) ⇔ $x_i=x_j$ for every edge ⇔ $x$ is constant on each component. The space of
   such vectors has dimension = number of components. $\square$
4. The second-smallest eigenvalue (the **algebraic connectivity** or Fiedler value) is $>0$ iff the graph
   is connected; its eigenvector is the basis of spectral clustering.

### 12.3 Normalised versions

* **Symmetric normalised Laplacian:** $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$, with
  $x^\top L_{\text{sym}}x=\tfrac12\sum_{ij}A_{ij}\left(\dfrac{x_i}{\sqrt{d_i}}-\dfrac{x_j}{\sqrt{d_j}}\right)^2$; its eigenvalues lie in
  $[0,2]$. This is the formula in Unit A2, and $D^{-1/2}AD^{-1/2}$ is the normalised adjacency used by GCN
  (Unit C3).
* **Random-walk normalisation:** $P=D^{-1}A$ (rows sum to 1, a Markov transition matrix; Unit C2) and
  $L_{\text{rw}}=I-P$.

**In the project.**
* `sym_norm(S)` in `methods.py` computes $D^{-1/2}SD^{-1/2}$ (with $d=0$ replaced by 1 to avoid division by
  zero). `SCMFDD` builds `Lr = I - sym_norm(Sr)` — the normalised Laplacian of the drug similarity graph —
  and adds $\lambda\,\operatorname{tr}(U^\top L_rU)$ to its loss. By the identity above (applied column by column),
  $\operatorname{tr}(U^\top LU)=\tfrac12\sum_{ij}S_{ij}\lVert u_i/\sqrt{d_i}-u_j/\sqrt{d_j}\rVert^2$: **similar drugs are pushed
  to have similar latent factors**. This answers A2's self-check "what does $\operatorname{tr}(U^\top LU)$ penalise?".
* `knn_kernel` is a random-walk normalisation $D^{-1}W$ of the (directed, weighted) kNN graph.
* `NIMCGCN` uses `sym_norm(Sr * knn_mask(Sr, k))` as its GCN propagation matrix.

```python
import numpy as np

def laplacian(A):
    return np.diag(A.sum(1)) - A

G5 = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
L = laplacian(G5)
print("L =\n", L)
print("row sums:", L.sum(1).tolist())
x = np.array([1, 1, 1, 0, 0])
print("x^T L x =", x @ L @ x, "(edges leaving {1,2,3})")
rng = np.random.default_rng(0)
y = rng.normal(size=5)
pairwise = 0.5 * sum(G5[i, j] * (y[i] - y[j]) ** 2 for i in range(5) for j in range(5))
print("identity holds for a random y:", np.isclose(y @ L @ y, pairwise))
print("eigenvalues of L(G5):", np.linalg.eigvalsh(L).round(4).tolist())

# two components: G5 plus a separate edge 6-7
A2 = np.zeros((7, 7), dtype=int); A2[:5, :5] = G5; A2[5, 6] = A2[6, 5] = 1
ev = np.linalg.eigvalsh(laplacian(A2))
print("eigenvalues with 2 components:", ev.round(4).tolist(), "-> zeros:", int(np.sum(np.isclose(ev, 0))))

d = G5.sum(1)
Lsym = np.eye(5) - G5 / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
print("eigenvalues of L_sym (all in [0, 2]):", (np.linalg.eigvalsh(Lsym).round(4) + 0.0).tolist())
```

**Output:**
```text
L =
 [[ 2 -1 -1  0  0]
 [-1  2 -1  0  0]
 [-1 -1  3 -1  0]
 [ 0  0 -1  2 -1]
 [ 0  0  0 -1  1]]
row sums: [0, 0, 0, 0, 0]
x^T L x = 1 (edges leaving {1,2,3})
identity holds for a random y: True
eigenvalues of L(G5): [0.0, 0.5188, 2.3111, 3.0, 4.1701]
eigenvalues with 2 components: [0.0, 0.0, 0.5188, 2.0, 2.3111, 3.0, 4.1701] -> zeros: 2
eigenvalues of L_sym (all in [0, 2]): [0.0, 0.3459, 1.2975, 1.5, 1.8566]
```

---

## 13. Sparse vs dense representations

### 13.1 The memory arithmetic

A graph with $n$ nodes and $m$ stored (directed) entries:

| Representation | Memory | Fdataset `chem_ecfp` kNN graph ($n=593$, 8,727 stored entries) |
|---|---|---|
| Dense `bool` | $n^2$ bytes | 351,649 bytes ≈ 0.34 MB |
| Dense `float32` | $4n^2$ bytes | 1,406,596 bytes ≈ 1.34 MB |
| COO (`int64` row, `int64` col, `float32` value) | $20m$ bytes | ≈ 0.17 MB |
| CSR (`int32` indices, `int32` indptr, `float32` data) | $8m+4(n+1)$ bytes | ≈ 0.07 MB |

(8,727 = $2\times4{,}067$ undirected edges + 593 self-loops.) The density of these graphs is about 2.5%,
so sparse formats save ~20–40×. But at this size *every* format is tiny compared with a 6 GB GPU.

### 13.2 Sparse formats

* **COO (coordinate)**: three arrays `row`, `col`, `data`, one entry per stored element. Easy to build
  (it is just an edge list with weights); slow for row access.
* **CSR (compressed sparse row)**: `data` and `indices` (column indices) stored row after row, plus `indptr`
  of length $n+1$ where row $i$'s entries are `data[indptr[i]:indptr[i+1]]`. Fast row slicing and fast
  matrix–vector products; the standard format for computation. **CSC** is the same by columns.

For $G_5$ (0-based), CSR is: `indptr = [0, 2, 4, 7, 9, 10]`, `indices = [1, 2, 0, 2, 0, 1, 3, 2, 4, 3]`,
`data = [1]*10`. Row 2 (node 3) is `indices[4:7] = [0, 1, 3]` — nodes 1, 2, 4.

```python
import numpy as np
from scipy import sparse

G5 = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
csr = sparse.csr_matrix(G5)
print("indptr :", csr.indptr.tolist())
print("indices:", csr.indices.tolist())
print("neighbours of node 3:", (csr.indices[csr.indptr[2]:csr.indptr[3]] + 1).tolist())

coo = csr.tocoo()
print("COO edges:", list(zip((coo.row + 1).tolist(), (coo.col + 1).tolist()))[:4], "...")
x = np.arange(1.0, 6.0)
print("A @ x dense == sparse:", np.allclose(G5 @ x, csr @ x))
print("A^2 via sparse:\n", (csr @ csr).toarray())

# memory at project scale vs a hypothetical graph with 20,000 gene nodes
for n, nnz in ((593, 8727), (593 + 313 + 20000, 2_000_000)):
    dense = 4 * n * n
    csr_bytes = 8 * nnz + 4 * (n + 1)
    print(f"n={n:6d}: dense float32 {dense / 1e6:9.1f} MB | CSR {csr_bytes / 1e6:6.2f} MB")
```

**Output:**
```text
indptr : [0, 2, 4, 7, 9, 10]
indices: [1, 2, 0, 2, 0, 1, 3, 2, 4, 3]
neighbours of node 3: [1, 2, 4]
COO edges: [(1, 2), (1, 3), (2, 1), (2, 3)] ...
A @ x dense == sparse: True
A^2 via sparse:
 [[2 1 1 1 0]
 [1 2 1 1 0]
 [1 1 3 0 1]
 [1 1 0 2 0]
 [0 0 1 0 1]]
n=   593: dense float32       1.4 MB | CSR   0.07 MB
n= 20906: dense float32    1748.2 MB | CSR  16.08 MB
```

### 13.3 Why the project uses dense masks — and when you should not

**Dense is the right choice here** because the node sets are small (593–663 drugs, 313–409 diseases).
Dense boolean masks make the GAT a few large tensor operations (`masked_fill`, `softmax`, `einsum`)
that GPUs execute extremely fast, without scatter/gather kernels, and keep the code short and readable —
`DenseGAT` is about ten lines. The price is $O(n_{dst}\cdot n_{src}\cdot H)$ memory per relation, which is a few
MB here.

**Sparse becomes necessary** when $n$ grows: with explicit gene nodes (tens of thousands), a dense
$n\times n$ float32 matrix would be ~1.7 GB *before* attention heads and autograd intermediates. Libraries such
as PyTorch Geometric and DGL store graphs as edge lists (COO, `edge_index` of shape $2\times m$) and compute
attention only on existing edges with scatter operations — memory $O(m)$ instead of $O(n^2)$. PyTorch
itself has `torch.sparse_coo_tensor`/`torch.sparse_csr_tensor` for sparse matrix products.

**Rule of thumb.** Up to a few thousand nodes and moderate density: dense is simplest and fastest on a
GPU. Beyond ~10⁴ nodes, or very sparse graphs: sparse/edge-list representations.

---

## 14. In this project: from similarity matrices to the heterogeneous graph

### 14.1 `fill_missing` — what an uncovered entity looks like (`data.py`)

```py
def fill_missing(S: np.ndarray) -> np.ndarray:
    """Uncovered entities get no neighbours in that view (only themselves)."""
    S = np.nan_to_num(S.copy(), nan=0.0)
    np.fill_diagonal(S, 1.0)
    return S
```

The similarity functions in `similarity.py` mark rows they could not compute (no SMILES, no genes, no
MONDO term) with NaN. `fill_missing` turns NaN into 0 ("no evidence of similarity", not "known to be
dissimilar") and forces the diagonal to 1. Graph-theoretically: such a node becomes **isolated except for
its self-loop** in that view (Section 7.2).

### 14.2 `knn_mask` — the kNN graph, line by line

```py
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

| Line | Graph-theory meaning |
|---|---|
| `fill_missing(S)` | complete weighted graph with NaNs → 0, self-similarity 1 |
| `np.fill_diagonal(work, -np.inf)` | a node may not choose itself as one of its $k$ neighbours |
| `k = min(k, n - 1)` | cannot have more than $n-1$ other neighbours |
| `np.argpartition(-work, k, axis=1)[:, :k]` | per row, the column indices of the $k$ largest similarities (unordered) — the directed kNN out-neighbourhoods |
| `rows = np.repeat(np.arange(n), k)` | the source row of each of the $n\cdot k$ candidate edges (edge list in COO style: `rows`, `idx.ravel()`) |
| `keep = vals > 0` | drop candidate edges with zero similarity (Section 10.4) |
| `M[rows[keep], idx.ravel()[keep]] = True` | write the directed kNN graph into a dense boolean adjacency matrix |
| `M \|= M.T` | OR-symmetrisation: undirected kNN graph (Section 10.2) |
| `np.fill_diagonal(M, True)` | self-loops: closed neighbourhoods for message passing |

The result is **boolean** (unweighted). The weights are not lost: the GAT recomputes data-dependent edge
weights (attention), and the node features contain the similarity rows themselves.

### 14.3 `topk_bipartite` — top-$k$ on a rectangular matrix

```py
def topk_bipartite(B: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns per row of a rectangular score matrix (zeros never kept)."""
    M = np.zeros_like(B, dtype=bool)
    if k <= 0:
        return M
    k = min(k, B.shape[1])
    idx = np.argpartition(-B, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(B.shape[0]), k)
    keep = B[rows, idx.ravel()] > 0
    M[rows[keep], idx.ravel()[keep]] = True
    return M
```

The bipartite analogue of `knn_mask`: no diagonal to exclude (rows and columns are different node types),
no symmetrisation possible (the matrix is rectangular), so the result is a **directed** bipartite relation
"row node receives from its top-$k$ column nodes". That is why `build` calls it twice, on `B` and on `B.T`
(Section 9.2). (`argpartition(-B, k-1)` and `argpartition(-work, k)` both put the $k$ largest values in the
first $k$ positions; the second form just also fixes the $(k{+}1)$-th.)

### 14.4 `knn_kernel` — a weighted, row-normalised kNN graph

```py
def knn_kernel(S: np.ndarray, k: int) -> np.ndarray:
    W = fill_missing(S) * knn_mask(S, k, symmetric=False)
    np.fill_diagonal(W, 0.0)
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return W / s
```

Directed kNN mask × similarity = weighted directed kNN graph; remove self-loops; divide each row by its
weighted out-degree (random-walk normalisation $D^{-1}W$, Section 12.3), guarding isolated rows against
division by zero (they stay all-zero). `K @ A` then gives the drug–drug–disease meta-path scores of the
propagation head (Section 11.2); you computed one by hand in Section 10.3.

### 14.5 `jaccard` and `cosine_cross` — meta-paths through genes (`similarity.py`)

```py
    G = (G > 0).astype(np.float64)
    inter = G @ G.T
    size = G.sum(1)
    union = size[:, None] + size[None, :] - inter
```

`inter = G @ G.T` is the commuting matrix of the entity–gene–entity meta-path (shared-gene counts);
`size` is each entity's gene degree; `union` uses inclusion–exclusion $|a\cup b|=|a|+|b|-|a\cap b|$ with
broadcasting to form all pairs. Rows with no genes become NaN, later turned into isolated nodes by
`fill_missing`.

```py
    n1 = np.sqrt(G1.sum(1, keepdims=True))
    n2 = np.sqrt(G2.sum(1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        M = (G1 @ G2.T) / (n1 * n2.T)
    return np.nan_to_num(M)
```

`G1 @ G2.T` counts drug→gene→disease paths; `n1 * n2.T` is the $(n_r\times n_d)$ matrix of
$\sqrt{\deg_{\text{gene}}(\text{drug})}\sqrt{\deg_{\text{gene}}(\text{disease})}$, so `M` is the cosine of the binary gene
profiles (Section 11.3). Entities without genes give $0/0$ = NaN → 0: no bridge.

### 14.6 `MVHGATMethod.build` — assembling all relations (`methods.py`)

```py
        relations, graphs = {}, {}
        for v in rv:
            relations[f"view:{v}"] = ("drug", "drug")
            graphs[f"view:{v}"] = t(knn_mask(data.drug_view(v), c.k), torch.bool)
        for v in dv:
            relations[f"view:{v}"] = ("disease", "disease")
            graphs[f"view:{v}"] = t(knn_mask(data.disease_view(v), c.k), torch.bool)
        if c.use_assoc_edges:
            relations["assoc>drug"] = ("drug", "disease")
            relations["assoc>disease"] = ("disease", "drug")
            A = t(A_train > 0, torch.bool)
            graphs["assoc>drug"], graphs["assoc>disease"] = A, A.T
        if c.use_bridge:
            B = data.gene_bridge
            relations["gene_bridge>drug"] = ("drug", "disease")
            relations["gene_bridge>disease"] = ("disease", "drug")
            graphs["gene_bridge>drug"] = t(topk_bipartite(B, c.bridge_k), torch.bool)
            graphs["gene_bridge>disease"] = t(topk_bipartite(B.T, c.bridge_k), torch.bool)
        return relations, X, graphs
```

* `relations` is the **network schema** in code: relation name → (destination type, source type).
* `graphs` holds one boolean adjacency per relation, shape (destination nodes, source nodes), moved to the
  GPU by `t(..., torch.bool)`.
* One relation per similarity view: a **multiplex** graph on drugs and another on diseases.
* `A_train > 0` — the **biadjacency matrix** of the *training-fold* bipartite indication graph. Test links
  are never edges (leakage prevention, Unit E1), and during training `fit_predict` further replaces these
  two masks each epoch by `Am`/`Am.T` with 20% of links hidden.
* The bridge relations come from the drug–gene–disease meta-path (`data.gene_bridge` is the
  `cosine_cross` matrix computed in `02_build_features.py`), sparsified with `topk_bipartite`.

The code below runs the real `build` on Fdataset (with the bridge switched on), on the CPU, and checks
the graph-theoretic properties claimed in this chapter; then it recomputes the connectivity table of
Section 7.2.

```python
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import load, knn_mask
from drepo.methods import MVHGATMethod

data = load("F")
relations, X, graphs = MVHGATMethod(use_bridge=True).build(data, data.A)
print("feature shapes:", {k: tuple(v.shape) for k, v in X.items()})
print(f"{'relation':22s} {'dst <- src':18s} {'shape':11s} {'entries':>7s}  symmetric  self-loops")
for r, (dst, src) in relations.items():
    M = graphs[r].cpu().numpy()
    sq = M.shape[0] == M.shape[1]
    print(f"{r:22s} {dst + ' <- ' + src:18s} {str(M.shape):11s} {int(M.sum()):7d}  "
          f"{str((M == M.T).all()) if sq else '-':9s}  {str(M.diagonal().all()) if sq else '-'}")
A = graphs["assoc>drug"].cpu().numpy()
print("assoc>disease is the transpose of assoc>drug:", np.array_equal(graphs["assoc>disease"].cpu().numpy(), A.T))
Bd, Bs = graphs["gene_bridge>drug"].cpu().numpy(), graphs["gene_bridge>disease"].cpu().numpy()
print("gene_bridge>disease is the transpose of gene_bridge>drug:", np.array_equal(Bs, Bd.T))
print("handshake (bipartite): drug-degree sum", int(A.sum(1).sum()), "= disease-degree sum", int(A.sum(0).sum()))

print("\nview        comps  largest  isolated  deg min/mean/max")
for name, S in list(zip(data.drug_view_names, data.drug_views)) + list(zip(data.disease_view_names, data.disease_views)):
    M = knn_mask(S, 10)
    deg = M.sum(1) - 1                                     # exclude the self-loop
    nc, lab = connected_components(csr_matrix(M), directed=False)
    print(f"{str(name):11s} {nc:5d}  {np.bincount(lab).max():7d}  {int((deg == 0).sum()):8d}  "
          f"{deg.min()}/{deg.mean():.1f}/{deg.max()}")
```

**Output:**
```text
feature shapes: {'drug': (593, 1779), 'disease': (313, 939)}
relation               dst <- src         shape       entries  symmetric  self-loops
view:chem_cdk          drug <- drug       (593, 593)     9361  True       True
view:chem_ecfp         drug <- drug       (593, 593)     8727  True       True
view:gene_r            drug <- drug       (593, 593)     8325  True       True
view:pheno_mim         disease <- disease (313, 313)     4561  True       True
view:sem_mondo         disease <- disease (313, 313)     5297  True       True
view:gene_d            disease <- disease (313, 313)     1877  True       True
assoc>drug             drug <- disease    (593, 313)     1933  -          -
assoc>disease          disease <- drug    (313, 593)     1933  -          -
gene_bridge>drug       drug <- disease    (593, 313)     4435  -          -
gene_bridge>disease    disease <- drug    (313, 593)     1809  -          -
assoc>disease is the transpose of assoc>drug: True
gene_bridge>disease is the transpose of gene_bridge>drug: False
handshake (bipartite): drug-degree sum 1933 = disease-degree sum 1933

view        comps  largest  isolated  deg min/mean/max
chem_cdk        1      593         0  10/14.8/94
chem_ecfp      16      578        15  0/13.7/49
gene_r         82      512        81  0/13.0/49
pheno_mim       1      313         0  10/13.6/31
sem_mondo      11      271         9  0/15.9/128
gene_d        157      153       153  0/5.0/40
```

Read the first table as a census of the graph MV-HGAT sees: the similarity relations store between
1,877 (`gene_d`) and 9,361 (`chem_cdk`) entries, self-loops included — roughly 2–5% of all possible
entries (a full $593\times593$ drug matrix has 351,649). The features are the raw similarity rows (3 views × 593 = 1,779
columns for drugs); `fit_predict` appends the 313-column visible association row, giving the 2,092 input
features counted in Unit B3.

---

## 15. Common mistakes and misconceptions

1. **Confusing the two adjacency conventions.** Textbooks use $A_{ij}=1$ for $i\to j$; the project's masks
   are (destination, source). For symmetric relations it does not matter; for bipartite and top-$k$
   relations it does. Always ask "who receives from whom?".
2. **Forgetting that kNN is directed.** "$j$ is in $i$'s top-10" ≠ "$i$ is in $j$'s top-10". After OR-symmetrisation
   degrees are $\ge k$, not $=k$.
3. **Expecting every node to have exactly $k$ neighbours.** Zero-similarity exclusion and missing data make
   some nodes isolated; OR-symmetrisation makes others hubs.
4. **Forgetting self-loops** in GNN graphs — a node then cannot "hear itself", and a node isolated in a view
   would produce an empty (NaN) attention row.
5. **Treating the bipartite association matrix as a square adjacency matrix.** $A$ is $593\times313$; $A^2$ is not
   even defined. Use $AA^\top$ / $A^\top A$ or the block matrix.
6. **Assuming `topk_bipartite(B.T, k)` equals `topk_bipartite(B, k).T`.** It does not (Section 9.2).
7. **Using raw meta-path counts as similarities.** Hubs (well-studied drugs) dominate; normalise (Jaccard,
   cosine, PathSim).
8. **Building a similarity or meta-path from the full association matrix** (e.g. $AA^\top$ with test links)
   — label leakage. The project computes no view from $A$; the propagation head uses only the *visible* links.
9. **Thinking a walk is a path.** $(A^k)_{ij}$ counts walks, which may revisit nodes; counting simple paths is
   much harder (#P-hard in general).
10. **Dense matrices by reflex.** Fine at 1,000 nodes; disastrous at 100,000. Do the memory arithmetic first.
11. **Square test matrices.** A bug that swaps drugs and diseases is invisible if both sets have the same size.
12. **Thinking "isolated in one view" means "the model knows nothing".** Other relations still connect the
    node — that is the point of a multi-view heterogeneous graph.
13. **Reading $L$'s smallest eigenvalue as "how disconnected"** — it is always 0; the *number* of zero
    eigenvalues counts components, and the second-smallest measures connectivity.

---

## 16. Exercises

Graded **[C]** conceptual, **[M]** mathematical, **[P]** programming; ★ to ★★★ difficulty.

**Exercise 1 [C ★].** For each of the relations `view:sem_mondo`, `assoc>disease` and `gene_bridge>drug`,
state the destination and source node types, the shape of its mask on Cdataset (663 drugs, 409 diseases),
and whether the mask is symmetric.

<details><summary>Solution</summary>

* `view:sem_mondo`: disease ← disease, $409\times409$, symmetric (OR-symmetrised kNN with self-loops).
* `assoc>disease`: disease ← drug, $409\times663$, rectangular (the transpose of `assoc>drug`, $663\times409$).
* `gene_bridge>drug`: drug ← disease, $663\times409$, rectangular; and not the transpose of
  `gene_bridge>disease` (each side keeps its own top-$k$).
</details>

**Exercise 2 [M ★].** Prove the bipartite handshake lemma: in a bipartite graph with biadjacency $B$, the
sum of left degrees equals the sum of right degrees equals the number of edges. Check it on Fdataset's
numbers (mean drug degree 3.26 over 593 drugs, mean disease degree 6.18 over 313 diseases).

<details><summary>Solution</summary>

Every edge has exactly one endpoint on each side. Summing left degrees counts each edge once (from its left
end); so does summing right degrees. In matrix form: $\mathbf 1^\top(B\mathbf 1)=(\mathbf 1^\top B)\mathbf 1=\mathbf 1^\top B\mathbf 1=m$.
Fdataset: $593\times3.2597=1933.0$ and $313\times6.1757=1933.0$, both equal to the 1,933 known links.
</details>

**Exercise 3 [M ★★].** Without computing $A^4$ fully, find the number of closed walks of length 4 starting
and ending at node 1 of $G_5$. List them.

<details><summary>Solution</summary>

$(A^4)_{11}=\sum_\ell (A^2)_{1\ell}(A^2)_{\ell1}=\lVert\text{row 1 of }A^2\rVert^2$ (as $A^2$ is symmetric)
$=2^2+1^2+1^2+1^2+0^2=7$. Grouping by the node $\ell$ reached after 2 steps: via $\ell=1$ there are $2\times2=4$
walks ($1{-}2{-}1{-}2{-}1$, $1{-}2{-}1{-}3{-}1$, $1{-}3{-}1{-}2{-}1$, $1{-}3{-}1{-}3{-}1$); via $\ell=2$: $1{-}3{-}2{-}3{-}1$; via $\ell=3$:
$1{-}2{-}3{-}2{-}1$; via $\ell=4$: $1{-}3{-}4{-}3{-}1$. Total 7.

```python
import numpy as np
from itertools import product
A = np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
print("(A^4)_11 =", np.linalg.matrix_power(A, 4)[0, 0])
walks = [(0, a, b, c, 0) for a, b, c in product(range(5), repeat=3)
         if A[0, a] and A[a, b] and A[b, c] and A[c, 0]]
print(len(walks), [ "-".join(str(v + 1) for v in w) for w in walks])
```

**Output:**
```text
(A^4)_11 = 7
7 ['1-2-1-2-1', '1-2-1-3-1', '1-2-3-2-1', '1-3-1-2-1', '1-3-1-3-1', '1-3-2-3-1', '1-3-4-3-1']
```
</details>

**Exercise 4 [M ★★].** For a simple undirected graph prove (a) $(A^2)_{ij}=|\mathcal N(i)\cap\mathcal N(j)|$ for
$i\ne j$; (b) $\operatorname{tr}(A^2)=2m$; (c) $\operatorname{tr}(A^3)=6\times\#\text{triangles}$.

<details><summary>Solution</summary>

(a) $(A^2)_{ij}=\sum_\ell A_{i\ell}A_{\ell j}$; the term is 1 exactly when $\ell$ is adjacent to both $i$ and $j$.
(b) $(A^2)_{ii}=\sum_\ell A_{i\ell}^2=\sum_\ell A_{i\ell}=\deg(i)$ (entries are 0/1), and $\sum_i\deg(i)=2m$ by the
handshake lemma. (c) $(A^3)_{ii}$ counts closed walks of length 3 from $i$. In a simple graph (no
self-loops) such a walk $i{-}a{-}b{-}i$ needs $a\ne i$, $b\ne a$, $b\ne i$, so $\{i,a,b\}$ is a triangle. Each triangle
yields 6 closed walks: 3 choices of starting node × 2 directions. Summing over $i$ gives $6T$.
</details>

**Exercise 5 [M ★★].** Four diseases $p,q,r,s$ have similarities
$S_{pq}=.5,\ S_{pr}=.2,\ S_{ps}=.1,\ S_{qr}=.3,\ S_{qs}=.3,\ S_{rs}=.9$ (symmetric, diagonal 1). (a) Build the $k=1$ kNN
graph, OR-symmetrise, add self-loops, and count components. (b) For $k=2$, what is ambiguous, and how many
undirected non-loop edges does the OR graph have in each case?

<details><summary>Solution</summary>

(a) Top-1: $p\to q$ (.5), $q\to p$ (.5), $r\to s$ (.9), $s\to r$ (.9). OR graph: edges $pq$, $rs$ — **two
components** $\{p,q\}$, $\{r,s\}$. Even OR-symmetrisation can leave a kNN graph disconnected when $k$ is small.
(b) Top-2: $p\to\{q,r\}$, $r\to\{s,q\}$, $s\to\{r,q\}$, and $q\to\{p,\ ?\}$ where $r$ and $s$ **tie** at 0.3 — the
choice is arbitrary (whatever `argpartition` returns). Either way the OR graph contains $pq, pr, qr, qs, rs$
(note $qr$ and $qs$ are already present from $r$'s and $s$'s choices), so it has 5 edges and is the same in
both cases; only the *directed* graph (and therefore `knn_kernel`) differs.

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import knn_mask
from scipy.sparse.csgraph import connected_components
S = np.array([[1, .5, .2, .1], [.5, 1, .3, .3], [.2, .3, 1, .9], [.1, .3, .9, 1]])
for k in (1, 2):
    M = knn_mask(S, k)
    print(f"k={k}: undirected edges={int((M.sum() - 4) // 2)}, components={connected_components(M)[0]}")
    print("   directed choice of q:", np.flatnonzero(knn_mask(S, k, symmetric=False)[1] & (np.arange(4) != 1)).tolist())
```

**Output:**
```text
k=1: undirected edges=2, components=2
   directed choice of q: [0]
k=2: undirected edges=5, components=1
   directed choice of q: [0, 2]
```
</details>

**Exercise 6 [M ★★].** Drugs $x,y$ and diseases $u,v,w$ over genes $g_1..g_4$:
$x=\{g_1,g_2\}$, $y=\{g_2,g_3,g_4\}$; $u=\{g_1\}$, $v=\{g_2,g_3\}$, $w=\{g_4\}$. Compute the drug–gene–disease
count matrix, the cosine bridge, and the `gene_bridge>drug` mask for $k=1$. Which disease would drug $y$
be connected to, and why does $w$ lose to $v$ even though $y$ "covers" $w$ completely?

<details><summary>Solution</summary>

Counts: $x$: $(u,v,w)=(1,1,0)$; $y$: $(0,2,1)$. Cosine with gene degrees $x{:}2,\ y{:}3$; $u{:}1,\ v{:}2,\ w{:}1$:
$x$–$u=1/\sqrt2=0.7071$, $x$–$v=1/\sqrt4=0.5$, $x$–$w=0$; $y$–$u=0$, $y$–$v=2/\sqrt6=0.8165$, $y$–$w=1/\sqrt3=0.5774$.
Top-1: $x\to u$, $y\to v$. $w$'s single gene is entirely inside $y$'s profile, but cosine is symmetric in
the two profiles: $y$ shares two genes with $v$ and $v$ has only two genes, so their profiles overlap more
in both directions. (A one-sided "coverage" score $|g_y\cap g_w|/|g_w|$ would rank $w$ first — choosing the
normalisation is a modelling decision.)

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.similarity import cosine_cross
from drepo.data import topk_bipartite
Gr = np.array([[1, 1, 0, 0], [0, 1, 1, 1]])
Gd = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
print(Gr @ Gd.T)
print(cosine_cross(Gr, Gd).round(4))
print(topk_bipartite(cosine_cross(Gr, Gd), 1).astype(int))
```

**Output:**
```text
[[1 1 0]
 [0 2 1]]
[[0.7071 0.5    0.    ]
 [0.     0.8165 0.5774]]
[[1 0 0]
 [0 1 0]]
```
</details>

**Exercise 7 [M ★★★].** (a) Show that for an indicator vector $x=\mathbb 1_S$ of a node set $S$,
$x^\top Lx$ equals the number of edges between $S$ and its complement. (b) Prove that every eigenvalue of
$L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$ lies in $[0,2]$ (assume no isolated nodes).

<details><summary>Solution</summary>

(a) $x^\top Lx=\sum_{\{i,j\}\in E}(x_i-x_j)^2$. The term is 1 iff exactly one endpoint is in $S$, else 0.
(b) Let $\tilde A=D^{-1/2}AD^{-1/2}$ and $y=D^{-1/2}x$. Then
$x^\top(I-\tilde A)x=\tfrac12\sum_{ij}A_{ij}\left(\tfrac{x_i}{\sqrt{d_i}}-\tfrac{x_j}{\sqrt{d_j}}\right)^2\ge0$ and, by the same expansion
with a plus sign, $x^\top(I+\tilde A)x=\tfrac12\sum_{ij}A_{ij}\left(\tfrac{x_i}{\sqrt{d_i}}+\tfrac{x_j}{\sqrt{d_j}}\right)^2\ge0$. (Both expansions
use $\sum_jA_{ij}=d_i$, so $\tfrac12\sum_{ij}A_{ij}(x_i^2/d_i+x_j^2/d_j)=x^\top x$.) The first gives $\lambda(L_{\text{sym}})\ge0$;
the second gives $\lambda(\tilde A)\ge-1$, hence $\lambda(L_{\text{sym}})=1-\lambda(\tilde A)\le2$. Equality at 2 occurs iff
some component is bipartite.
</details>

**Exercise 8 [P ★★].** Write `my_knn_mask(S, k, mutual=False)` from scratch (handle NaN rows like
`fill_missing`, exclude zero similarities, add self-loops; `mutual=True` uses AND instead of OR). Test that
`mutual=False` reproduces the project's `knn_mask` on 20 random similarity matrices with some NaN rows,
and report how many edges AND keeps compared with OR on one of them.

<details><summary>Solution</summary>

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import knn_mask

def my_knn_mask(S, k, mutual=False):
    S = np.nan_to_num(S, nan=0.0)
    n = len(S)
    work = S.copy(); np.fill_diagonal(work, -np.inf)
    k = min(k, n - 1)
    top = np.argsort(-work, axis=1, kind="stable")[:, :k]
    M = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in top[i]:
            if work[i, j] > 0:
                M[i, j] = True
    M = (M & M.T) if mutual else (M | M.T)
    np.fill_diagonal(M, True)
    return M

rng = np.random.default_rng(0)
ok = 0
for trial in range(20):
    n = int(rng.integers(8, 40))
    X = rng.random((n, n)); S = (X + X.T) / 2; np.fill_diagonal(S, 1.0)
    S[rng.random(n) < 0.15] = np.nan                     # some uncovered entities ...
    S[:, np.isnan(S).all(1)] = np.nan                    # ... in both rows and columns
    ok += np.array_equal(my_knn_mask(S, 5), knn_mask(S, 5))
print("identical to project's knn_mask:", ok, "/ 20")
OR, AND = my_knn_mask(S, 5), my_knn_mask(S, 5, mutual=True)
print("last matrix: n =", n, "| OR edges:", (OR.sum() - n) // 2, "| AND edges:", (AND.sum() - n) // 2)
```

**Output:**
```text
identical to project's knn_mask: 20 / 20
last matrix: n = 28 | OR edges: 81 | AND edges: 44
```

The random similarities have no ties, so the stable sort and `argpartition` choose the same neighbours. The
mutual (AND) graph keeps noticeably fewer edges than the OR graph.
</details>

**Exercise 9 [P ★★].** Are the two chemical views redundant? On Fdataset, compute the Jaccard overlap of
the *edge sets* (excluding self-loops) of the `chem_cdk` and `chem_ecfp` kNN graphs ($k=10$).

<details><summary>Solution</summary>

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import load, knn_mask
data = load("F")
off = ~np.eye(data.n_drugs, dtype=bool)
E1 = knn_mask(data.drug_view("chem_cdk"), 10) & off
E2 = knn_mask(data.drug_view("chem_ecfp"), 10) & off
print("shared edges:", int((E1 & E2).sum() // 2), "| union:", int((E1 | E2).sum() // 2),
      "| edge Jaccard: %.3f" % ((E1 & E2).sum() / (E1 | E2).sum()))
```

**Output:**
```text
shared edges: 1399 | union: 7052 | edge Jaccard: 0.198
```

Only about one edge in five is shared: two fingerprints of the *same* molecules produce substantially
different neighbourhoods. This supports treating them as separate relations and letting attention weigh
them (and is consistent with ECFP's better stand-alone AUPR reported in HOW_IT_WORKS §10).
</details>

**Exercise 10 [P ★★★].** Compute the 2-hop **receptive field** of drugs in the union of all default
relations on Fdataset: build the $906\times906$ boolean block adjacency (OR of the three drug views on the
drug block, OR of the three disease views on the disease block, $A$ and $A^\top$ off-diagonal, self-loops), then
use $(\mathcal A^2>0)$. How many nodes, and how many diseases, does an average drug "see" after 2 layers?
What does this suggest about over-smoothing?

<details><summary>Solution</summary>

```python
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import load, knn_mask
d = load("F")
nr, nd = d.A.shape
H = np.zeros((nr + nd, nr + nd), dtype=bool)
for v in d.drug_view_names:
    H[:nr, :nr] |= knn_mask(d.drug_view(v), 10)
for v in d.disease_view_names:
    H[nr:, nr:] |= knn_mask(d.disease_view(v), 10)
H[:nr, nr:] = d.A > 0
H[nr:, :nr] = (d.A > 0).T
np.fill_diagonal(H, True)
H1 = H.astype(np.int64)
R2 = (H1 @ H1) > 0
print("mean 1-hop neighbourhood of a drug (incl. itself): %.1f" % H[:nr].sum(1).mean())
print("mean 2-hop receptive field of a drug: %.1f of %d nodes" % (R2[:nr].sum(1).mean(), nr + nd))
print("mean number of diseases within 2 hops of a drug: %.1f of %d" % (R2[:nr, nr:].sum(1).mean(), nd))
```

**Output:**
```text
mean 1-hop neighbourhood of a drug (incl. itself): 39.5
mean 2-hop receptive field of a drug: 566.1 of 906 nodes
mean number of diseases within 2 hops of a drug: 121.2 of 313
```

With all relations combined, two hops already reach roughly 60% of the graph. A third or fourth layer
would make almost every node's receptive field the whole graph, so embeddings would tend to converge
(over-smoothing, Unit C3) — one reason MV-HGAT uses only 2 layers, keeps $k$ small, adds skip connections,
and concatenates layer outputs (jumping knowledge).
</details>

**Exercise 11 [C ★★].** Why does the project *not* add a drug–drug "co-indication" view built from
$AA^\top$ (drug–disease–drug meta-path), although co-indication is an excellent similarity signal?

<details><summary>Solution</summary>

(1) **Leakage.** If computed from the full $A$, test links would enter the graph through the back door
(two drugs sharing a held-out indication would become neighbours). (2) **Even from training links only**,
it must be recomputed every fold and — in this project — every epoch from the *visible* links, otherwise it
reveals the links hidden for hidden-link supervision. (3) **Redundancy.** The model already has the
`assoc` relations; two layers of message passing along drug→disease→drug *is* the co-indication meta-path,
computed from exactly the visible links. HOW_IT_WORKS §6 (assumption 7) states the design rule: no
similarity view is computed from the association matrix.
</details>

**Exercise 12 [P ★].** Store Fdataset's association matrix in CSR format. Report its number of stored
entries, its memory compared with a dense `float32` array, and the drug with the most indications.

<details><summary>Solution</summary>

```python
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.data import load
d = load("F")
C = sparse.csr_matrix(d.A)
csr_bytes = C.data.nbytes + C.indices.nbytes + C.indptr.nbytes
print("stored entries:", C.nnz, "| density: %.4f" % (C.nnz / np.prod(C.shape)))
print("CSR bytes:", csr_bytes, "| dense float32 bytes:", d.A.astype(np.float32).nbytes)
deg = np.diff(C.indptr)                                 # row degrees straight from indptr
print("max degree:", deg.max(), "->", d.drug_names[deg.argmax()])
```

**Output:**
```text
stored entries: 1933 | density: 0.0104
CSR bytes: 17840 | dense float32 bytes: 742436
max degree: 22 -> Methotrexate
```

`np.diff(indptr)` gives every row's number of stored entries — the degree — without touching the data.
</details>

**Exercise 13 [C ★★].** The raw gene-bridge matrix `data.gene_bridge` is non-zero for about 10% of all
drug–disease pairs on Fdataset. Why is it sparsified to the top 10 per node instead of using `B > 0` as the
mask, and why is the bridge off by default?

<details><summary>Solution</summary>

`B > 0` would create roughly $0.10\times593\times313\approx18{,}900$ edges, dominated by weak, single-gene overlaps of
well-studied drugs (literature bias): message passing would mostly average noise, exactly the problem
kNN sparsification solves for similarity views (Section 10.1). Top-$k$ keeps the strongest, degree-normalised
(cosine) connections per node. It is off by default because, even sparsified, the meta-path predicted
indications at near-random level on its own (AUC 0.52–0.57, AUPR ≈ positive rate), so it adds parameters
and noise without signal on these benchmarks; it remains available as an ablation ("+ bridge").
</details>

---

## 17. Answers to the self-check questions (PREREQUISITES.md, Unit C1)

### Q1. Our graph has drug, disease and gene nodes. Which edge types exist?

**Two levels to distinguish: the conceptual (data) graph and the graph the model actually uses.**

**(a) The conceptual tripartite heterogeneous graph** built from the data sources has three node types
and these edge types:

| Edge type | Between | Source in this project | Directed? Weighted? |
|---|---|---|---|
| chemical similarity (×2: CDK, ECFP) | drug – drug | benchmark `DrugSim`; RDKit Morgan + Tanimoto | undirected, weighted |
| phenotype similarity | disease – disease | benchmark MimMiner | undirected, weighted |
| ontology (semantic) similarity | disease – disease | Wang similarity on MONDO | undirected, weighted |
| known indication | drug – disease | benchmark association matrix (training fold only) | undirected (bipartite), unweighted |
| chemical–gene interaction | drug – gene | CTD chemical–gene interactions (human) | unweighted |
| gene–disease association | gene – disease | CTD *curated* (marker/mechanism) links only | unweighted |

What is **absent**: no gene–gene edges (no protein–protein interaction network is used); no CTD
*inferred* or *therapeutic* gene–disease edges (they are partly derived from known treatments — leakage);
CTD curated chemical–disease links are used only to validate case studies, never as edges.

**(b) The graph MV-HGAT actually operates on** has only **two node types** (drug, disease), because gene
nodes are folded into meta-paths (Section 11), and these relation types (as built by
`MVHGATMethod.build`):

1. `view:chem_cdk` (drug ← drug) — kNN of benchmark chemical similarity;
2. `view:chem_ecfp` (drug ← drug) — kNN of ECFP4 Tanimoto;
3. `view:gene_r` (drug ← drug) — kNN of Jaccard on CTD gene sets = the **drug–gene–drug meta-path**;
4. `view:pheno_mim` (disease ← disease) — kNN of MimMiner;
5. `view:sem_mondo` (disease ← disease) — kNN of MONDO semantic similarity;
6. `view:gene_d` (disease ← disease) — kNN of Jaccard on curated disease genes = the
   **disease–gene–disease meta-path**;
7. `assoc>drug` (drug ← disease) — visible training indications $A_m$;
8. `assoc>disease` (disease ← drug) — the same edges reversed, $A_m^\top$;
9. *(optional, off by default)* `gene_bridge>drug` (drug ← disease) — top-10 diseases per drug by the
   **drug–gene–disease meta-path** (cosine);
10. *(optional)* `gene_bridge>disease` (disease ← drug) — top-10 drugs per disease by the same meta-path.

So 8 relations by default and 10 with the bridge. All six view relations are undirected (OR-symmetrised)
with self-loops; the association relations are a bipartite graph stored as two directed relations; the
bridge relations are directed and mutually non-transposed. Two more "relations" exist outside the GNN, in
the decoder's propagation head: the drug–drug($v$)–disease and drug–disease–disease($u$) meta-paths `K @ Am`,
one per view (Section 11.2).

### Q2. How is a drug–gene–disease meta-path turned into a direct drug–disease relation?

Step by step, with the code that does it:

1. **Two bipartite graphs.** From CTD, build the binary drug–gene matrix $G_r$ ($n_r\times n_g$, chemical–gene
   interactions) and the binary disease–gene matrix $G_d$ ($n_d\times n_g$, curated gene–disease links), over
   the same gene columns (`02_build_features.py`).
2. **Count meta-path instances with a matrix product.** By the walk-counting theorem (Section 6.1, applied
   to typed relations in Section 11.2), $C=G_rG_d^\top$ has $C_{ij}=$ the number of genes $g$ with
   drug $i\to g\to$ disease $j$, i.e. the number of drug–gene–disease path instances. This is the
   **commuting matrix** of the meta-path.
3. **Normalise for degree bias.** Divide by $\sqrt{|g_i|}\sqrt{|g_j|}$ (each entity's gene count):
   $B_{ij}=C_{ij}/\sqrt{|g_i||g_j|}$, the cosine similarity of the two gene profiles —
   `similarity.cosine_cross(G1, G2)` computes `(G1 @ G2.T) / (n1 * n2.T)`. Without this step, drugs with
   hundreds of CTD genes would be bridged to almost every disease. Entities with no genes get 0.
4. **Store it** as `data.gene_bridge` ($n_r\times n_d$) in the processed `.npz` file.
5. **Sparsify into edges.** Like a similarity matrix, $B$ is too dense to be a graph (about 10% of pairs
   non-zero on Fdataset). `data.topk_bipartite(B, 10)` keeps, for each drug, its 10 highest-scoring
   diseases (never zero scores) → the boolean mask of relation `gene_bridge>drug` (drug ← disease);
   `topk_bipartite(B.T, 10)` keeps each disease's top 10 drugs → `gene_bridge>disease`. On Fdataset these
   have 4,435 and 1,809 edges (diseases without genes contribute none).
6. **Use it as a relation.** `MVHGATMethod.build` registers both in `relations` with types
   ("drug", "disease") and ("disease", "drug") when `use_bridge=True`. From then on the model treats them
   exactly like the association edges: a dedicated `DenseGAT` per direction computes attention-weighted
   messages along the bridge edges, and the view attention decides, per node, how much to trust them.

The result is a **direct** drug–disease edge that summarises an indirect, two-hop, typed path. What is
gained: a disease with no known drugs (cold start) can still receive messages from plausible drugs through
shared genes, and the graph stays small (no gene nodes). What is lost: which genes were shared, and any
weighting of genes by importance. In this project the evaluation showed the bridge carried almost no
predictive signal on these benchmarks (AUC 0.52–0.57 alone), so it is off by default and tested as
"+ bridge" in the ablation. The same three-step recipe — **product of relation matrices → normalisation →
sparsification** — also produced the `gene_r` and `gene_d` views from the drug–gene–drug and
disease–gene–disease meta-paths (with Jaccard instead of cosine, then `knn_mask`).

---

## 18. Summary and cheat sheet

**Core definitions**

| Concept | Definition / formula |
|---|---|
| Graph | $G=(V,E)$, $n=\lvert V\rvert$, $m=\lvert E\rvert$ |
| Adjacency matrix | $A_{ij}=1$ (or $w_{ij}$) if $i\sim j$; symmetric iff undirected |
| Project mask convention | `mask[dst, src]` — who receives from whom |
| Degree / degree matrix | $\mathbf d=A\mathbf 1$, $D=\operatorname{diag}(\mathbf d)$; in/out-degree = column/row sums |
| Handshake | $\sum_v\deg(v)=2m$; bipartite: $\sum$ left $=\sum$ right $=m$ |
| Density | $2m/(n(n-1))$ |
| $k$-hop neighbourhood | nodes at distance $\le k$; reachable iff $((I+A)^k)_{ij}>0$ |
| Walk counting | $(A^k)_{ij}$ = # walks of length $k$; weighted: sum of weight products |
| Common neighbours / triangles | $(A^2)_{ij}$; $\operatorname{tr}(A^3)/6$ |
| Components | BFS/DFS labelling; `scipy.sparse.csgraph.connected_components` |
| Bipartite | no odd cycles; biadjacency $B$; full $\begin{pmatrix}0&B\\B^\top&0\end{pmatrix}$; projections $BB^\top$, $B^\top B$ |
| Heterogeneous graph | $(V,E,\phi,\psi)$; schema; one matrix per relation |
| kNN graph | per node top-$k$ (directed) → OR / AND symmetrise → self-loops |
| Meta-path | type sequence; commuting matrix $R_1R_2\cdots R_\ell$ |
| Normalisations | Jaccard $\frac{M_{xy}}{M_{xx}+M_{yy}-M_{xy}}$; cosine $\frac{C_{xy}}{\sqrt{d_xd_y}}$; PathSim $\frac{2M_{xy}}{M_{xx}+M_{yy}}$ |
| Laplacian | $L=D-A$, $x^\top Lx=\sum_{\{i,j\}\in E}w_{ij}(x_i-x_j)^2$; #zero eigenvalues = #components |
| Normalised | $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$ (eigenvalues in $[0,2]$); $P=D^{-1}A$ |
| Memory | dense $O(n^2)$; CSR $O(n+m)$ |

**Project map**

| Code | Graph concept |
|---|---|
| `data.fill_missing` | missing data → isolated node with self-loop |
| `data.knn_mask` | OR-symmetrised kNN graph with self-loops, no zero-similarity edges |
| `data.topk_bipartite` | directed top-$k$ bipartite relation |
| `data.knn_kernel` | weighted directed kNN, random-walk normalised; `K @ A` = drug–drug–disease meta-path |
| `similarity.jaccard` | normalised entity–gene–entity meta-path (`gene_r`, `gene_d`) |
| `similarity.cosine_cross` | normalised drug–gene–disease meta-path (bridge) |
| `MVHGATMethod.build` | network schema (`relations`) + one mask per relation (`graphs`) |
| `methods.sym_norm`, `SCMFDD` | $D^{-1/2}SD^{-1/2}$, Laplacian regulariser $\operatorname{tr}(U^\top LU)$ |
| `np.block([[Sr, A], [A.T, Sd]])` (DRRS, LAGCN) | heterogeneous graph squashed into one weighted adjacency |

**Numbers to remember (Fdataset):** 593 drugs, 313 diseases, 1,933 links (density 1.04%); drug degree
1–22 (mean 3.26), disease degree 1–84 (mean 6.18); 8 relations (10 with bridge); kNN $k=10$ gives mean
degree ≈ 13–16 per view (only 5 in the sparse `gene_d` view); two layers reach ≈ 60% of all nodes.

---

## 19. Further resources (curated; all links checked)

**Courses**

* **Stanford CS224W: Machine Learning with Graphs** (Jure Leskovec) — <https://web.stanford.edu/class/cs224w/>
  (slides, notes, Colabs) and the lecture videos on YouTube —
  <https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn> — *Free.* Lectures 1–2 cover
  graph basics, representations, heterogeneous graphs and node/link features; later lectures cover GNNs,
  heterogeneous GNNs and knowledge graphs (Units C3–C6). An archived offering with complete materials:
  <https://snap.stanford.edu/class/cs224w-2023/>.
* **MIT 6.042J Mathematics for Computer Science** — <https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/>
  — *Free.* The graph-theory lectures and textbook chapters (walks, paths, connectivity, bipartite graphs,
  matchings, colouring) with rigorous proofs at an introductory level.

**Books**

* **William L. Hamilton — *Graph Representation Learning*** — <https://www.cs.mcgill.ca/~wlh/grl_book/> —
  *Free pre-publication PDF* (print version paid). Chapter 1 (graph definitions, multi-relational and
  heterogeneous graphs) and chapter 2 (node statistics, neighbourhood overlap, Laplacians and spectral
  methods) are the ideal next read after this unit.
* **Albert-László Barabási — *Network Science*** — <https://networksciencebook.com/> — *Free online.*
  Chapter 2, "Graph Theory" (<https://networksciencebook.com/chapter/2>), covers degree, adjacency,
  bipartite networks, paths, connectedness and clustering with excellent figures; later chapters explain
  hubs and degree distributions.
* **Reinhard Diestel — *Graph Theory*** (Springer GTM 173) — <https://diestel-graph-theory.com/> — *Main text
  free to view online; e-book and print paid.* The rigorous mathematical reference (chapter 1 for all
  basic definitions and proofs).
* **Douglas B. West — *Introduction to Graph Theory*** (Pearson, 2nd ed.) — *Paid; widely available in
  libraries.* The standard undergraduate textbook, with many exercises.
* **Mark Newman — *Networks*** (Oxford University Press, 2nd ed., 2018) — *Paid.* Chapters on mathematics
  of networks (adjacency matrices, walks, Laplacian, bipartite networks) written for scientists.

**Spectral graph theory and kNN graphs**

* Ulrike von Luxburg, "A Tutorial on Spectral Clustering" (2007) — <https://arxiv.org/abs/0711.0189> —
  *Free.* The clearest treatment of similarity graphs ($\varepsilon$, kNN, mutual kNN), Laplacians and their
  properties; sections 2–3 match Sections 10 and 12 of this unit.
* Daniel Spielman, *Spectral and Algebraic Graph Theory* (Yale lecture notes/book draft) —
  <https://www.cs.yale.edu/homes/spielman/sagt/> — *Free.* The Laplacian in depth.

**Heterogeneous graphs and meta-paths**

* Sun, Han, Yan, Yu & Wu (2011), "PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous
  Information Networks", *PVLDB* — <https://www.vldb.org/pvldb/vol4/p992-sun.pdf> — *Free.* The paper that
  formalised meta-paths and commuting matrices.
* Wang et al. (2019), "Heterogeneous Graph Attention Network" (HAN) — <https://arxiv.org/abs/1903.07293> —
  *Free.* Meta-paths as relations plus two levels of attention; the direct ancestor of MV-HGAT (Unit C5).
* Schlichtkrull et al. (2018), "Modeling Relational Data with Graph Convolutional Networks" (R-GCN) —
  <https://arxiv.org/abs/1703.06103> — *Free.* One weight matrix per relation in a multi-relational graph.
* Dong, Chawla & Swami (2017), metapath2vec — project page with paper and code:
  <https://ericdongyx.github.io/metapath2vec/m2v.html> — *Free.* Meta-path-guided random walks for
  embeddings.

**Visual and practical**

* Sanchez-Lengeling et al. (2021), "A Gentle Introduction to Graph Neural Networks", *Distill* —
  <https://distill.pub/2021/gnn-intro/> — *Free.* Interactive figures on adjacency matrices, adjacency
  lists and message passing; the bridge from this unit to C3.
* SciPy documentation: sparse matrices — <https://docs.scipy.org/doc/scipy/reference/sparse.html>; graph
  routines (`connected_components`, shortest paths, Laplacian) —
  <https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html>. *Free.*

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **Adjacency list** | For each node, the list of its neighbours. |
| **Adjacency matrix** | $n\times n$ matrix with $A_{ij}$ = 1 (or a weight) if $i$ and $j$ are connected. |
| **Biadjacency matrix** | $\lvert U\rvert\times\lvert W\rvert$ matrix of a bipartite graph; the project's association matrix $A$. |
| **Bipartite graph** | Nodes split into two sets with edges only between the sets; equivalently no odd cycles. |
| **BFS (breadth-first search)** | Layer-by-layer graph traversal that computes shortest-path distances in unweighted graphs. |
| **Closed neighbourhood** | $\mathcal N[v]=\mathcal N(v)\cup\{v\}$; what a node sees when self-loops are added. |
| **Commuting matrix** | Product of relation matrices along a meta-path; counts its path instances. |
| **Connected component** | Maximal set of mutually reachable nodes. |
| **COO / CSR / CSC** | Sparse formats: coordinate lists / compressed rows / compressed columns. |
| **Cosine similarity** | $\frac{x\cdot y}{\lVert x\rVert\lVert y\rVert}$; for binary profiles, shared count over $\sqrt{d_xd_y}$. |
| **Cycle** | Closed path with at least 3 nodes and no other repetition. |
| **Degree / strength** | Number of incident edges / sum of incident edge weights. |
| **Degree matrix** | Diagonal matrix $D$ of degrees. |
| **Density** | Fraction of possible edges present. |
| **Diameter** | Largest shortest-path distance in a connected graph. |
| **Directed graph** | Edges are ordered pairs (arrows), source → target. |
| **Distance** | Length of a shortest path. |
| **$\varepsilon$-graph** | Similarity graph keeping pairs with similarity above a threshold. |
| **Giant component** | A component containing a large fraction of the nodes. |
| **Handshake lemma** | Sum of degrees = $2m$. |
| **Heterogeneous graph (HIN)** | Graph with several node and/or edge types, given by type maps $\phi,\psi$. |
| **Hub** | Node with unusually high degree. |
| **Hubness** | Tendency of some points to appear in very many kNN lists. |
| **In-/out-degree** | Number of incoming / outgoing arrows of a node. |
| **Isolated node** | Node with no edges (other than possibly a self-loop). |
| **Jaccard similarity** | $\lvert a\cap b\rvert/\lvert a\cup b\rvert$ for sets. |
| **$k$-hop neighbourhood** | Nodes within distance $k$; the receptive field of a $k$-layer GNN. |
| **kNN graph** | Each node linked to its $k$ most similar nodes; directed until symmetrised. |
| **$k$-partite graph** | Nodes split into $k$ parts with no edges inside a part. |
| **Knowledge graph** | Multi-relational graph stored as (head, relation, tail) triples. |
| **Laplacian** | $L=D-A$; normalised $L_{\text{sym}}=I-D^{-1/2}AD^{-1/2}$. |
| **Meta-path** | Sequence of node types connected by relations, e.g. drug–gene–disease. |
| **Multigraph** | Graph allowing parallel edges. |
| **Multiplex / multi-view graph** | Same node set, several edge sets (e.g. three drug similarity graphs). |
| **Mutual kNN** | kNN graph keeping only edges chosen by both endpoints (AND). |
| **Neighbourhood** | $\mathcal N(v)$, the nodes adjacent to $v$. |
| **Network schema** | Type-level graph of a heterogeneous graph (node types and relations). |
| **One-mode projection** | $BB^\top$ or $B^\top B$: links nodes of one side through shared neighbours. |
| **Path** | Walk without repeated nodes. |
| **PathSim** | Meta-path similarity $2M_{xy}/(M_{xx}+M_{yy})$. |
| **Receptive field** | Set of nodes whose features can influence a node's output after $k$ layers. |
| **Relation** | An edge type, with fixed source and destination node types. |
| **Self-loop** | Edge from a node to itself. |
| **Simple graph** | No self-loops, no parallel edges. |
| **Sparse / dense representation** | Storing only non-zero entries / storing all $n^2$ entries. |
| **Symmetrisation** | Turning a directed graph into an undirected one (OR, AND or weight averaging). |
| **Walk** | Sequence of adjacent nodes, repeats allowed; counted by powers of $A$. |
| **Weighted graph** | Edges carry numerical weights. |
