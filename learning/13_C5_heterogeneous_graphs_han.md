# Unit C5: Heterogeneous Graphs and the Heterogeneous Graph Attention Network (HAN)

> **Track C: Graphs and graph neural networks** · Chapter 13 of the learning track
>
> **Prerequisites:** C1 *Graph theory basics* (adjacency matrices, bipartite graphs, k-NN graphs), C3 *Message passing and GCN*, C4 *Attention and GAT*, B3 *Neural networks and PyTorch*, A2 *Linear algebra* (matrix products, block matrices).
>
> **Estimated study time:** 8–10 hours. Plan on about 3 h for the theory (sections 2–3), 2 h running and modifying the code (section 4), 1.5 h on the project walkthrough (section 5), and 2–3 h on the exercises.

---

## Learning objectives

When you finish this unit you should be able to do the following.

1. **Define** a heterogeneous information network (HIN) formally: node-type map, edge-type map, network schema. **Draw** the network schema of this project, including the gene nodes that the code folds away.
2. **Compute by hand** the commuting matrix of a meta-path (for example drug–gene–drug) and turn it into a similarity with count, PathSim, Jaccard or cosine normalisation.
3. **Explain** with a concrete example why a single-weight (homogeneous) GNN mixes incompatible signals, and **name** the three mechanisms by which it does so.
4. **Write down** the R-GCN layer equation, **derive** the parameter counts of the full, basis-decomposed and block-diagonal variants, and **compute** one R-GCN layer by hand on a five-node graph.
5. **Write down** both levels of HAN (node-level and semantic-level attention), **derive** the semantic-attention formula and its softmax Jacobian, and **compute** semantic weights by hand.
6. **Contrast** global (HAN) and node-specific (this project's `ViewAttention`) view attention: what each can express, what each costs, and how masking of missing views works.
7. **List** the main view-fusion strategies (concatenation, mean, attention, gating, late fusion) and **predict** which one suits a given situation.
8. **Summarise** HGT, MAGNN, HetGNN and Simple-HGN in a sentence or two each, including what Simple-HGN taught the field about evaluation.
9. **Walk through** `model.py::ViewAttention`, `HeteroLayer` and `MVHGAT` line by line, and **map** every line to an equation in this chapter.
10. **Interpret** the project's ablation numbers on views and view attention with appropriate caution: paired per-fold differences, one repeat, and AUC vs AUPR.

---

## Notation used in this chapter

| Symbol | Meaning |
|---|---|
| $\mathcal{G}=(\mathcal{V},\mathcal{E})$ | a graph with node set $\mathcal{V}$ and edge set $\mathcal{E}$ |
| $\phi:\mathcal{V}\to\mathcal{T}$ | node-type map (here $\mathcal{T}=\{\text{drug},\text{disease},\text{gene}\}$) |
| $\psi:\mathcal{E}\to\mathcal{R}$ | edge-type (relation) map |
| $\mathcal{N}^r(i)$ | neighbours of node $i$ under relation $r$ |
| $\Phi$ | a meta-path, e.g. $\text{Drug}\xrightarrow{\text{targets}}\text{Gene}\xrightarrow{\text{targeted by}}\text{Drug}$ |
| $\mathcal{N}^\Phi(i)$ | meta-path-based neighbours of $i$ (nodes reachable from $i$ along $\Phi$) |
| $M_\Phi$ | commuting matrix of $\Phi$: product of the adjacency matrices along the path |
| $h_i$, $h_i'$ | node representation before / after a layer (row vectors in code, column vectors in maths) |
| $W_r$ | relation-specific weight matrix (R-GCN) |
| $\alpha^\Phi_{ij}$ | node-level attention of $i$ on neighbour $j$ within $\Phi$ |
| $z_i^\Phi$ | node $i$'s embedding computed from meta-path / relation $\Phi$ only |
| $\beta_\Phi$, $\beta_i^\Phi$ | semantic (view) attention weight: global (HAN) or node-specific (this project) |
| $\sigma$ | an activation; $\mathrm{sigmoid}$ is written out when meant |

A note on **"view", "relation" and "meta-path"**: in this project the three words often refer to the same object. A *view* is one way of measuring similarity, for example `chem_ecfp`. Inside the model each view becomes a *relation*, a set of typed edges such as drug ← drug. Some of those relations come from *meta-paths*: `gene_r` is the drug–gene–drug meta-path, collapsed into a direct drug–drug edge. Section 2.2 makes the distinction precise.

---

## 1. Motivation: why this matters for *this* project

### 1.1 One graph, many kinds of evidence

Look at the evidence available about one drug in Fdataset, say a small-molecule anti-inflammatory:

- It has a SMILES string, so it has **two chemical views**: CDK fingerprints (`chem_cdk`, from the benchmark) and ECFP4 fingerprints (`chem_ecfp`, computed by us). In each view it has 10 nearest-neighbour drugs.
- It has CTD gene interactions, so it has a **gene view** (`gene_r`, Jaccard overlap of gene sets) with another 10 neighbours.
- It has a few **known indications** (on average $1933/593\approx 3.3$ per drug in Fdataset). These are edges to *disease* nodes, not to drugs.

A biologic such as a heparin, on the other hand, **has no SMILES**. In both chemical views it has no neighbours at all; only its gene view and its known links carry information.

On the disease side, an old OMIM syndrome may have a MimMiner phenotype vector (`pheno_mim`) and a place in the MONDO hierarchy (`sem_mondo`), but **no curated CTD genes**. About half of the diseases are in that situation (only 51% of Fdataset diseases have curated genes, `HOW_IT_WORKS.md` §2.3), so it has no `gene_d` neighbours.

These sources differ in every way that matters to a neural network:

| | `chem_ecfp` edge | `gene_r` edge | `assoc` edge (known indication) |
|---|---|---|---|
| connects | drug–drug | drug–drug | drug–disease |
| meaning | "looks alike chemically" | "perturbs the same genes" | "is used to treat" |
| typical degree | 10–20 (symmetrised k-NN) | 10–20, or 0 if no genes | ~3 per drug, ~6 per disease |
| reliability | good for close analogues | weak (curation bias) | ground truth, but only in training |
| feature space of the neighbour | drug | drug | **disease** |

A **heterogeneous graph** is a graph that records these differences explicitly: every node has a *type* and every edge has a *relation type*. A **heterogeneous GNN** is a GNN that uses those types, for example by giving each relation its own weights and by learning how much to trust each relation.

### 1.2 What goes wrong if you ignore the types

The baseline LAGCN in `methods.py` builds one big homogeneous matrix

$$
H=\begin{bmatrix} S_r & A \\ A^\top & S_d\end{bmatrix}
$$

and runs an ordinary GCN on it. Every edge is the same kind of edge, and every layer has **one** weight matrix. On Fdataset 5-fold CV our re-implementation of LAGCN reaches AUC 0.833 and AUPR 0.133, while MV-HGAT reaches 0.939 and 0.488 (`results/RESULTS.md`). Many things differ between the two models (features, decoder, training tricks), so **the whole gap cannot be blamed on homogeneity**. Still, it illustrates the point of this unit: if the model cannot tell "similar to" from "treats", it has to untangle the two signals from the numbers alone, and with ~1,900 labelled links it usually cannot.

### 1.3 What this unit gives you

MV-HGAT, the project's model, is a **heterogeneous multi-view GNN** in the style of HAN:

1. each relation gets its own GAT (**node-level attention**, relation-specific weights, as in R-GCN and HAN);
2. each drug and each disease then **weighs its relation messages** with a learned softmax (**semantic-level / view attention**, as in HAN, but *per node*);
3. gene nodes are not explicit; they are **folded into meta-paths** (drug–gene–drug, disease–gene–disease, and optionally drug–gene–disease).

After this unit, every line of `model.py::ViewAttention` and `HeteroLayer` will map to an equation you have derived yourself.

---

## 2. Core theory

### 2.1 Heterogeneous information networks

**Intuition.** A homogeneous graph is a phone book: everyone is a "person" and every link means "knows". A heterogeneous graph is a database: there are tables of drugs, diseases and genes, and several kinds of links between them, each with its own meaning.

**Definition (heterogeneous information network, Sun & Han; Shi et al. 2017).** A HIN is a graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ together with

- a node-type mapping $\phi:\mathcal{V}\to\mathcal{T}$, and
- an edge-type mapping $\psi:\mathcal{E}\to\mathcal{R}$,

such that $|\mathcal{T}|+|\mathcal{R}|>2$. If $|\mathcal{T}|=|\mathcal{R}|=1$ the graph is homogeneous. A graph with one node type but several edge types, such as drugs connected by a chemical view and a gene view, is already heterogeneous. Such a graph is often called a **multiplex** or **multi-relational** graph.

**Definition (network schema).** The *network schema* $\mathcal{S}=(\mathcal{T},\mathcal{R})$ is the "type-level" graph. It has one node per type and one edge per relation, and each relation $r$ connects a source type to a target type, written $r: T_s\to T_t$. The schema is the blueprint of the HIN: every actual edge must be an instance of a schema edge.

The schema of this project, with genes still present, looks like this:

```
                    chem_cdk, chem_ecfp                pheno_mim, sem_mondo
                     (similarity, k-NN)                 (similarity, k-NN)
                        ┌───────┐                          ┌───────┐
                        │       ▼                          │       ▼
                      ┌───────────┐     treats (assoc)   ┌───────────┐
                      │   DRUG    │ ───────────────────► │  DISEASE  │
                      │  (593)    │ ◄─────────────────── │   (313)   │
                      └───────────┘    treated-by        └───────────┘
                            │  ▲                               │  ▲
              interacts-with│  │                   associated- │  │
               (CTD chem-gene) │                   with (CTD   │  │
                            ▼  │                   curated)    ▼  │
                      ┌──────────────────────────────────────────────┐
                      │                    GENE                      │
                      └──────────────────────────────────────────────┘
```

**Relations versus their inverses.** Message passing is directional: a message flows *from* a source node *to* a target node. An undirected "treats" edge therefore gives two relations: `assoc>drug` (disease → drug messages, the drug *receives*) and `assoc>disease` (drug → disease messages). R-GCN calls these a relation and its *inverse relation*. The project creates both explicitly in `MVHGATMethod.build`:

```python
relations["assoc>drug"] = ("drug", "disease")      # (destination type, source type)
relations["assoc>disease"] = ("disease", "drug")
```

Note the code's convention: the tuple is **(destination, source)**, because the dense adjacency masks are stored as `dst x src`.

**Typed adjacency matrices.** The cleanest way to store a HIN is as a set of adjacency matrices, one per relation, each of shape $|\mathcal{V}_{T_t}|\times|\mathcal{V}_{T_s}|$. In the project these are boolean masks: `graphs["view:chem_ecfp"]` is $593\times 593$, `graphs["assoc>drug"]` is $593\times 313$, `graphs["assoc>disease"]` is its transpose. There is no single "big" adjacency matrix; that is the whole point.

> **Terminology.** *Multi-relational*: several edge types. *Multiplex*: the same node set with several layers of edges (our drug–drug views). *Multi-view*: several feature sets or graphs describing the same objects (our six similarity views). *Knowledge graph*: a multi-relational graph of (head, relation, tail) facts, usually with many relation types and no node features. All of them are special cases of, or close relatives of, the HIN.

### 2.2 Meta-paths: composing relations

**Intuition.** Two drugs that never appear together in the data can still be related *through* something else: they hit the same gene, or they treat the same disease. A **meta-path** names such an indirect relationship at the level of types.

**Definition (meta-path).** A meta-path $\Phi$ is a path in the network schema:

$$
\Phi:\; T_1 \xrightarrow{r_1} T_2 \xrightarrow{r_2} \cdots \xrightarrow{r_\ell} T_{\ell+1},
$$

which describes the *composite relation* $r_1\circ r_2\circ\cdots\circ r_\ell$ between types $T_1$ and $T_{\ell+1}$. When the relations are unambiguous we abbreviate by type initials: **DGD** (drug–gene–drug), **SGS** (disease–gene–disease; S for "sickness" to avoid a clash with D), **DGS** (drug–gene–disease), **DSD** (drug–disease–drug: two drugs sharing an indication).

A **path instance** of $\Phi$ is a concrete path in the graph whose node types and edge types follow $\Phi$, e.g. *aspirin → PTGS2 → ibuprofen* is an instance of DGD.

**Definition (meta-path-based neighbours).** $\mathcal{N}^\Phi(i)$ is the set of nodes reachable from $i$ by at least one instance of $\Phi$. HAN includes $i$ itself in $\mathcal{N}^\Phi(i)$ (a self-loop). For a symmetric meta-path such as DGD, $i$ is usually reachable from itself anyway.

**Commuting matrix.** Let $R\in\{0,1\}^{n_\text{drug}\times n_\text{gene}}$ be the drug–gene adjacency and $Q\in\{0,1\}^{n_\text{dis}\times n_\text{gene}}$ the disease–gene adjacency. Then the number of path instances is a matrix product:

$$
M_{DGD}=RR^\top,\qquad M_{SGS}=QQ^\top,\qquad M_{DGS}=RQ^\top,\qquad M_{DSD}=AA^\top .
$$

Why? $(RR^\top)_{ij}=\sum_g R_{ig}R_{jg}$ counts genes $g$ that both $i$ and $j$ touch, which is exactly the number of DGD path instances from $i$ to $j$. In general, for $\Phi=r_1\circ\cdots\circ r_\ell$ with adjacency matrices $A_{r_1},\dots,A_{r_\ell}$:

$$
M_\Phi = A_{r_1}A_{r_2}\cdots A_{r_\ell}, \qquad (M_\Phi)_{ij}=\#\{\text{instances of }\Phi\text{ from } i \text{ to } j\}.
$$

Then $\mathcal{N}^\Phi(i)=\{j: (M_\Phi)_{ij}>0\}$.

**From counts to similarities.** Raw counts favour "hub" nodes: a heavily studied drug with 2,000 CTD genes shares *some* gene with almost everything. Several normalisations exist:

| Normalisation | Formula (symmetric meta-path) | Comment |
|---|---|---|
| raw count | $M_{ij}$ | biased towards hubs |
| **PathSim** (Sun et al. 2011) | $\dfrac{2M_{ij}}{M_{ii}+M_{jj}}$ | for symmetric meta-paths; equals 1 for identical nodes |
| **Jaccard** on neighbour sets | $\dfrac{\lvert G_i\cap G_j\rvert}{\lvert G_i\cup G_j\rvert}=\dfrac{M_{ij}}{M_{ii}+M_{jj}-M_{ij}}$ | what the project uses for `gene_r`, `gene_d` |
| **cosine** of profiles | $\dfrac{r_i\cdot q_j}{\lVert r_i\rVert\,\lVert q_j\rVert}$ | works for asymmetric meta-paths such as DGS (the project's gene bridge) |

#### Worked example 1: meta-paths on a three-drug, two-disease, three-gene graph

Take drugs $d_1,d_2,d_3$, diseases $s_1,s_2$, genes $g_1,g_2,g_3$:

- $d_1$ interacts with $g_1,g_2$; $d_2$ with $g_2$; $d_3$ with $g_3$.
- $s_1$ is associated with $g_1,g_2$; $s_2$ with $g_3$.
- Known indications: $d_1$ treats $s_1$; $d_3$ treats $s_2$.

$$
R=\begin{bmatrix}1&1&0\\0&1&0\\0&0&1\end{bmatrix},\quad
Q=\begin{bmatrix}1&1&0\\0&0&1\end{bmatrix},\quad
A=\begin{bmatrix}1&0\\0&0\\0&1\end{bmatrix}.
$$

*DGD.* $M_{DGD}=RR^\top$. Entry $(1,1)$: $d_1$ has 2 genes, so $1\cdot1+1\cdot1+0=2$. Entry $(1,2)$: they share only $g_2$, so 1. Entry $(2,2)=1$, $(3,3)=1$, everything else 0:

$$
M_{DGD}=\begin{bmatrix}2&1&0\\1&1&0\\0&0&1\end{bmatrix}.
$$

*PathSim* for $(d_1,d_2)$: $2\cdot1/(2+1)=0.667$. *Jaccard*: $|\{g_2\}|/|\{g_1,g_2\}|=1/2=0.5$. The two normalisations disagree on the number but agree on the ranking, which is what a k-NN graph uses.

*DGS (the "gene bridge").* $M_{DGS}=RQ^\top$: $d_1$ shares 2 genes with $s_1$, $d_2$ shares 1, $d_3$ shares 1 with $s_2$:

$$
M_{DGS}=\begin{bmatrix}2&0\\1&0\\0&1\end{bmatrix},\qquad
\cos(d_2,s_1)=\frac{1}{\sqrt1\cdot\sqrt2}=0.707.
$$

*DSD.* $M_{DSD}=AA^\top=\mathrm{diag}(1,0,1)$. No two drugs share an indication, so the co-indication meta-path gives **no** off-diagonal neighbours, and $d_2$ (no known indication) is not even connected to itself through it. Keep this in mind: in section 2.6, $d_2$'s DSD view will be *empty*, and we will need a mask.

You will verify all these numbers in code (section 4.1).

**How the project folds gene nodes into meta-paths.** The graph used by MV-HGAT has **no gene nodes**. Instead:

| Meta-path | Becomes | Normalisation | Where |
|---|---|---|---|
| Drug–Gene–Drug | drug view `gene_r` | Jaccard of CTD gene sets, then k-NN (k=10) | `02_build_features.py`, then `knn_mask` |
| Disease–Gene–Disease | disease view `gene_d` | Jaccard of curated gene sets, then k-NN | same |
| Drug–Gene–Disease | relation `gene_bridge` (optional) | cosine of gene profiles, top-10 diseases per drug (`topk_bipartite`) | `data.gene_bridge`, `MVHGATMethod.build` |

This is the standard HAN approach ("meta-path-based neighbours") and it has clear **advantages**: the graph stays small (no ~20,000 gene nodes), every relation connects the two types we actually predict between, and the Jaccard normalisation is computed once, outside the network. It has **costs** too. First, the *identity of the intermediate gene is lost*: the model knows that two drugs share genes but not *which* genes, so it cannot learn that sharing PTGS2 matters more than sharing CYP3A4. MAGNN (section 2.8) was designed to fix this. Second, the *choice of meta-paths is manual*. HGT avoids it by stacking layers over the raw schema.

**Meta-path length and explosion.** The number of path instances grows roughly like the product of average degrees along the path. With CTD genes this is severe: DGD through a hub gene connects hundreds of drugs. This is one reason the project sparsifies every meta-path view to its top-10 neighbours.

### 2.3 Why a homogeneous GNN mixes incompatible signals

Recall the GCN / mean-aggregation layer from C3 on a homogeneous graph:

$$
h_i' = \sigma\Big(W\,\frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} h_j\Big).
$$

Put drugs and diseases into one node set and union all edge types. Three distinct things go wrong.

**(1) Semantic mixing: one $W$ for every relation.** Suppose drug $d_1$ has a chemically similar neighbour $d_2$ and a known indication $s_1$. The aggregated message is $W(h_{d_2}+h_{s_1})/2$. A sum is *permutation invariant and type blind*: from the sum alone the layer cannot tell whether a component came from "a drug like me" or from "a disease I treat". Those mean opposite things for prediction. A disease I treat is a *target*; a drug like me is a *peer*. Formally, the layer's output is a function of the multiset $\{h_j\}$, and the edge type is not part of that multiset.

**(2) Feature-space mismatch.** Drug features and disease features live in different spaces. In the project the raw drug input has $3\times593+313$ dimensions (three similarity rows plus the visible link row), while the raw disease input has $3\times313+593$. They cannot even be added without a projection. A homogeneous GCN forces them into one space by padding or by a shared input matrix. LAGCN does the latter: each node's input row is its row of $H$. A drug's row and a disease's row then share columns that mean different things.

**(3) Degree and scale imbalance.** With k=10 symmetrised k-NN graphs, a drug has roughly 10–20 neighbours *per view*, so 30–60 similarity neighbours over three views, against ~3 indication neighbours. A shared mean aggregator gives the indication signal a weight of about $3/(3+45)\approx 6\%$. The most predictive relation is drowned by sheer count. Per-relation aggregation normalises *within* each relation, so every relation gets a fair hearing, and a learned weight then decides how much each one counts.

**Remedies, in increasing order of flexibility:**

1. *Relation-specific weights.* Give each relation its own $W_r$ and normalise within each relation. This is **R-GCN** (2.4).
2. *Relation-specific attention.* Inside each relation, learn which neighbours matter (node-level attention); across relations, learn which relations matter (semantic-level attention). This is **HAN** (2.5).
3. *Type-dependent attention parameters everywhere.* Queries, keys and values depend on the source type, target type and edge type. This is **HGT** (2.8).

### 2.4 R-GCN: relational graph convolution (Schlichtkrull et al. 2018)

**Intuition.** Keep the GCN idea of "average your neighbours, then transform", but do it **once per relation, each with its own transformation**, and then add the results.

**The layer.**

$$
\boxed{\,h_i^{(l+1)} = \sigma\Big( W_0^{(l)} h_i^{(l)} \;+\; \sum_{r\in\mathcal{R}}\;\sum_{j\in\mathcal{N}^r(i)} \frac{1}{c_{i,r}}\, W_r^{(l)} h_j^{(l)} \Big)\,}
$$

- $\mathcal{R}$ contains every relation **and its inverse**, so information flows both ways along a directed fact.
- $W_0$ is a self-connection: the node keeps its own information. In the paper it is a special "self-loop relation".
- $c_{i,r}$ is a normalisation constant, typically $c_{i,r}=|\mathcal{N}^r(i)|$ (mean within the relation). It is chosen per problem, and a learned attention can replace it.
- The paper uses $\sigma=\text{ReLU}$ between layers.

Compare this with point (3) of 2.3: since each relation is normalised by its *own* degree, the three indication neighbours get a message of the same scale as the forty-five similarity neighbours.

**The problem: parameters grow with $|\mathcal{R}|$.** With $d$-dimensional hidden states, a full R-GCN layer has $|\mathcal{R}|\,d^2$ relation parameters. Knowledge graphs such as FB15k-237 have hundreds of relations, many with only a handful of edges. A $d\times d$ matrix learned from five edges will overfit. The paper offers two regularisers.

**Basis decomposition.** Every relation matrix is a learned linear combination of $B$ shared basis matrices:

$$
W_r = \sum_{b=1}^{B} a_{rb}\, V_b,\qquad V_b\in\mathbb{R}^{d_\text{out}\times d_\text{in}},\; a_{rb}\in\mathbb{R}.
$$

Parameters: $B\,d_\text{in}d_\text{out} + |\mathcal{R}|B$. *Interpretation:* the bases are shared "transformation styles", and each relation picks its own mix. Rare relations borrow statistical strength from frequent ones, because the $V_b$ are trained by all relations. This is **weight sharing across relations**, the same idea as low-rank matrix factorisation applied to the tensor of relation matrices: stacking all $W_r$ into a tensor $\mathcal{W}\in\mathbb{R}^{|\mathcal{R}|\times d_\text{out}\times d_\text{in}}$, the basis decomposition says $\mathcal{W}$ has rank at most $B$ along its relation mode.

**Block-diagonal decomposition.** Every relation matrix is block-diagonal:

$$
W_r = \mathrm{blockdiag}\big(Q_{1r},\dots,Q_{Br}\big),\qquad Q_{br}\in\mathbb{R}^{(d_\text{out}/B)\times(d_\text{in}/B)}.
$$

Parameters: $|\mathcal{R}|\,B\,(d_\text{in}/B)(d_\text{out}/B)=|\mathcal{R}|\,d_\text{in}d_\text{out}/B$. *Interpretation:* the hidden dimensions are split into $B$ groups that do not talk to each other within a relation. It is a **sparsity constraint**, sensible when the latent features cluster into groups that interact mainly among themselves.

**Parameter count at the project's scale** (8 relations, $d_\text{in}=d_\text{out}=64$, excluding $W_0$), verified in section 4.2:

| Variant | Formula | Parameters |
|---|---|---|
| full | $8\cdot64^2$ | 32,768 |
| basis, $B=2$ | $2\cdot64^2+8\cdot2$ | 8,208 |
| basis, $B=4$ | $4\cdot64^2+8\cdot4$ | 16,416 |
| block, $B=4$ | $8\cdot64^2/4$ | 8,192 |
| block, $B=16$ | $8\cdot64^2/16$ | 2,048 |

With only 8 relations the project does not need these tricks. But if you added, say, one relation per CTD interaction type (binding, expression up, expression down, ...), basis decomposition would become attractive.

#### Worked example 2: one R-GCN layer by hand

Nodes: $d_1,d_2,d_3,s_1,s_2$ with 2-dimensional features (row vectors, multiplied on the right as in PyTorch's `h @ W`):

$$
x_{d_1}=[1,0],\; x_{d_2}=[0,1],\; x_{d_3}=[1,1],\; x_{s_1}=[1,0],\; x_{s_2}=[0,1].
$$

Relations: $r_0$ = *shares-gene* ($d_1\leftrightarrow d_2$, from DGD in example 1), $r_1$ = *treats* (messages drug → disease: $d_1\to s_1$, $d_3\to s_2$), $r_2$ = *treated-by* (disease → drug: $s_1\to d_1$, $s_2\to d_3$). Every node has at most one neighbour per relation, so $c_{i,r}=1$.

Basis decomposition with $B=2$: $V_1=I$, $V_2=\begin{bmatrix}0&1\\1&0\end{bmatrix}$ (swap the two coordinates), and coefficients

$$
a=\begin{bmatrix}0.5&0\\0&1\\1&0.5\end{bmatrix}\;\Rightarrow\;
W_{\text{shares}}=0.5I,\quad W_{\text{treats}}=V_2,\quad W_{\text{treated-by}}=I+0.5V_2=\begin{bmatrix}1&0.5\\0.5&1\end{bmatrix}.
$$

$W_0=I$, activation ReLU.

- $d_1$: self $[1,0]$; shares-gene from $d_2$: $[0,1]\cdot0.5I=[0,0.5]$; treated-by from $s_1$: $[1,0]W_\text{tb}=[1,0.5]$. Sum $=[2,1]$.
- $d_2$: self $[0,1]$; shares-gene from $d_1$: $[0.5,0]$. Sum $=[0.5,1]$.
- $d_3$: self $[1,1]$; treated-by from $s_2$: $[0,1]W_\text{tb}=[0.5,1]$. Sum $=[1.5,2]$.
- $s_1$: self $[1,0]$; treats from $d_1$: $[1,0]V_2=[0,1]$. Sum $=[1,1]$.
- $s_2$: self $[0,1]$; treats from $d_3$: $[1,1]V_2=[1,1]$. Sum $=[1,2]$.

All entries are positive, so ReLU changes nothing. Notice that $d_1$ and $s_1$ start from the **same** feature vector $[1,0]$ but end up different, because their neighbours arrive through different relations with different matrices. That is exactly what a homogeneous GCN cannot do.

**R-GCN for link prediction.** In the original paper R-GCN is an *encoder*. Links $(s,r,o)$ are scored with a DistMult decoder $f(s,r,o)=e_s^\top R_r e_o$ ($R_r$ diagonal), trained with sampled negatives. This **encoder–decoder** pattern is exactly what MV-HGAT does with a bilinear decoder (Unit C6). On FB15k-237 the R-GCN encoder improved substantially over a decoder-only DistMult baseline (the abstract reports a 29.8% relative improvement).

### 2.5 HAN: the Heterogeneous Graph Attention Network (Wang et al. 2019)

HAN asks two questions at every node:

1. *Within* one meta-path, **which neighbours** matter? This is **node-level attention**, a GAT restricted to $\mathcal{N}^\Phi(i)$.
2. *Across* meta-paths, **which meta-paths** matter? This is **semantic-level attention**.

```
             meta-path Φ1 neighbours          meta-path Φ2 neighbours
                  j1  j2  j3                       k1   k2
                   \  |  /                          \  /
           node-level attention (GAT, Φ1)   node-level attention (GAT, Φ2)
                     │  α^Φ1_ij                     │  α^Φ2_ik
                     ▼                              ▼
                   z_i^Φ1                         z_i^Φ2
                       \                          /
                        \   semantic attention  /
                         \     β_Φ1    β_Φ2    /
                          ▼                   ▼
                       z_i = β_Φ1 z_i^Φ1 + β_Φ2 z_i^Φ2   →  classifier / decoder
```

#### Step 0: type-specific projection

Different node types have different feature spaces (problem 2 in 2.3). HAN first maps each type into a common space with a type-specific matrix:

$$
h_i' = M_{\phi(i)}\, h_i .
$$

In the project this is `MVHGAT.inp`, one `nn.Linear` for drugs (input size $3\cdot593+313$) and one for diseases, both to 64 dimensions.

#### Step 1: node-level attention (within meta-path $\Phi$)

Exactly GAT (Unit C4), restricted to meta-path neighbours (including $i$ itself) and with an attention vector $a_\Phi$ *specific to the meta-path*:

$$
e^\Phi_{ij} = \mathrm{LeakyReLU}\big(a_\Phi^\top [\,h_i' \,\Vert\, h_j'\,]\big),\qquad
\alpha^\Phi_{ij} = \frac{\exp(e^\Phi_{ij})}{\sum_{k\in\mathcal{N}^\Phi(i)} \exp(e^\Phi_{ik})},
$$

$$
z_i^\Phi = \sigma\Big(\sum_{j\in\mathcal{N}^\Phi(i)} \alpha^\Phi_{ij}\, h_j'\Big),
$$

with $K$ heads concatenated: $z_i^\Phi = \Vert_{k=1}^{K}\,\sigma\big(\sum_j \alpha^{\Phi,k}_{ij} h_j'\big)$. Two remarks:

- **Asymmetry.** $e^\Phi_{ij}\neq e^\Phi_{ji}$ in general, because $a_\Phi$ has a "target half" and a "source half". In the project's `DenseGAT` these halves are `a_dst` and `a_src` (Unit C4, self-check 1).
- **Relation-specific weights.** HAN's paper shares the type projection across meta-paths and makes only $a_\Phi$ meta-path-specific. The project goes further: every relation has its own `W_src`, `W_dst`, `a_src`, `a_dst` (one `DenseGAT` per relation). That is the **R-GCN idea of relation-specific weights inside a HAN-style architecture**.

After this step each node has $P$ embeddings $z_i^{\Phi_1},\dots,z_i^{\Phi_P}$, one per meta-path. Each tells a different story.

#### Step 2: semantic-level attention (across meta-paths), with a derivation

We want a weight $\beta_\Phi\ge0$ for each meta-path, with $\sum_\Phi\beta_\Phi=1$. It must be **learned from data** and **comparable across meta-paths**. Here is the construction step by step.

*(a) Score each meta-path-specific embedding with one shared scoring function.* Use a one-layer MLP followed by a learned "query" vector $q$ (additive or Bahdanau-style attention):

$$
s_i^\Phi = q^\top \tanh\big(W z_i^\Phi + b\big).
$$

$W$, $b$ and $q$ are **shared by all meta-paths**. If each meta-path had its own scorer, a high score would only mean "this scorer outputs large numbers", and scores would not be comparable. With a shared scorer, $s_i^\Phi>s_i^\Psi$ means "by the same yardstick, $z_i^\Phi$ looks more like what the query is looking for". The $\tanh$ keeps each coordinate in $(-1,1)$, so a single huge coordinate cannot dominate, and $|s_i^\Phi|\le\lVert q\rVert_1$.

*(b) Summarise over nodes (the HAN choice).* HAN wants one importance per meta-path *for the whole graph*, so it averages:

$$
w_\Phi = \frac{1}{|\mathcal{V}|}\sum_{i\in\mathcal{V}} q^\top \tanh\big(W z_i^\Phi + b\big).
$$

(In practice the average runs over the nodes of the target type.)

*(c) Normalise with a softmax*, giving positive weights that sum to one:

$$
\boxed{\;\beta_\Phi = \frac{\exp(w_\Phi)}{\sum_{\Psi}\exp(w_\Psi)},\qquad
z_i = \sum_{\Phi}\beta_\Phi\, z_i^\Phi\;}
$$

*(d) Use $z_i$ downstream*: a classifier with cross-entropy in the paper; a bilinear decoder in this project.

**Why a softmax and not independent sigmoids?** The softmax makes meta-paths *compete*: raising one weight necessarily lowers the others. You can see this in the Jacobian. Differentiating $\beta_\Phi=e^{w_\Phi}/\sum_\Psi e^{w_\Psi}$:

$$
\frac{\partial \beta_\Phi}{\partial w_\Psi}
=\frac{\delta_{\Phi\Psi}e^{w_\Phi}\sum e^{w}-e^{w_\Phi}e^{w_\Psi}}{(\sum e^{w})^2}
=\beta_\Phi\big(\delta_{\Phi\Psi}-\beta_\Psi\big).
$$

Two consequences follow.

1. $\sum_\Phi \partial\beta_\Phi/\partial w_\Psi=\beta_\Psi-\beta_\Psi\sum_\Phi\beta_\Phi=0$: the total weight is conserved, and gradient that raises one view must lower the others.
2. The diagonal term $\beta_\Phi(1-\beta_\Phi)$ is at most $1/4$ and **vanishes when $\beta_\Phi\to 0$ or $1$**. A view whose weight has collapsed to zero receives almost no gradient and tends to stay "switched off". This is one reason attention weights can get stuck early in training, and why they should not be read as a careful verdict on a view's value.

**The fused embedding is a convex combination.** Since $\beta_\Phi\ge0$ and $\sum\beta_\Phi=1$, $z_i$ lies in the convex hull of $\{z_i^\Phi\}$ and $\lVert z_i\rVert\le\max_\Phi\lVert z_i^\Phi\rVert$ (triangle inequality). Attention can *select* and *blend* views, but it cannot *amplify* agreement: two views that both say "+1" fuse to "+1", not "+2". Concatenation and gating (section 2.7) do not have this limit.

#### Worked example 3: semantic attention by hand

Three drugs, two meta-paths, embeddings already computed by node-level attention:

| | $z^{DGD}$ | $z^{DSD}$ |
|---|---|---|
| $d_1$ | $[1,0]$ | $[2,1]$ |
| $d_2$ | $[0.5,0.5]$ | $[0,0]$ |
| $d_3$ | $[0,0]$ | $[1,1]$ |

Take $W=I$, $b=0$, $q=[1,1]$, so $s_i^\Phi=\tanh(z_{i,1}^\Phi)+\tanh(z_{i,2}^\Phi)$. Use $\tanh(0.5)=0.4621$, $\tanh(1)=0.7616$, $\tanh(2)=0.9640$.

- DGD: $d_1$: $0.7616+0=0.7616$; $d_2$: $2\times0.4621=0.9242$; $d_3$: $0$. Mean $w_{DGD}=1.6858/3=0.5619$.
- DSD: $d_1$: $0.9640+0.7616=1.7256$; $d_2$: $0$; $d_3$: $1.5232$. Mean $w_{DSD}=3.2488/3=1.0829$.
- Softmax: $e^{0.5619}=1.7540$, $e^{1.0829}=2.9533$, so $\beta_{DGD}=1.7540/4.7073=0.3726$ and $\beta_{DSD}=0.6274$.

**Every drug** now uses 37% DGD and 63% DSD. Look at $d_2$: it has **no** co-indicated drugs (its DSD embedding is the zero vector, because it has no known indication), yet 63% of its fused embedding comes from that empty view: $z_{d_2}=0.3726\,[0.5,0.5]=[0.186,0.186]$. Its only real evidence has been diluted to 37% strength.

**Node-specific alternative.** Skip the averaging, and softmax per node: $\beta_i^\Phi=\exp(s_i^\Phi)/\sum_\Psi\exp(s_i^\Psi)$.

- $d_1$: $(0.7616,1.7256)\to(0.276,0.724)$
- $d_2$: $(0.9242,0)\to(0.716,0.284)$
- $d_3$: $(0,1.5232)\to(0.179,0.821)$

Now $d_2$ leans on its gene view. Add a **validity mask**, setting $s=-\infty$ for views in which the node has no neighbours, and $d_2$ gets $\beta=(1,0)$ and $z_{d_2}=[0.5,0.5]$: its real evidence at full strength. This is precisely what the project's `ViewAttention` does with `s.masked_fill(~valid, -inf)`. All of these numbers are reproduced in section 4.3.

#### Complexity and training

Node-level attention costs $O(\sum_\Phi|\mathcal{E}_\Phi|\cdot d)$ for the scores, plus the projections. Semantic attention costs $O(P\,|\mathcal{V}|\,d\,d_\text{att})$. Both are linear in graph size for sparse meta-path graphs. The project uses *dense* $n\times n$ masks (`DenseGAT`), which costs $O(n^2)$ per relation but is perfectly fine for 593 drugs and 313 diseases, and is much simpler than sparse scatter operations.

HAN was introduced for **semi-supervised node classification** on ACM, DBLP and IMDB (e.g. ACM papers with meta-paths Paper–Author–Paper and Paper–Subject–Paper) and trained with cross-entropy on labelled nodes. Like GCN and GAT, HAN as published is **transductive**: it embeds the nodes of one fixed graph.

### 2.6 Node-specific versus global view attention

| | Global (HAN) | Node-specific (project's `ViewAttention`) |
|---|---|---|
| weight | $\beta_\Phi$, one per meta-path | $\beta_i^\Phi$, one per node per relation |
| formula | softmax over $\Phi$ of the **node-averaged** score | softmax over $\Phi$ of the **node's own** score |
| parameters | $W,b,q$ | the same $W,b,q$ (no extra parameters!) |
| expresses | "PSP is more informative than PAP **for this dataset**" | "**this** disease relies on phenotype; **that** one on genes" |
| missing views | cannot handle per node (one weight for all) | handled by masking $s_i^\Phi=-\infty$ |
| variance | low: averaging over hundreds of nodes smooths noise | higher: one node's score can be noisy |
| interpretability | one global ranking of meta-paths | a per-node profile, e.g. for case studies |
| failure mode | dilutes nodes for which a globally good view is empty or noisy | may overfit, i.e. pick views by accident on small data |

Note the subtle row on parameters. Node-specific attention costs **nothing extra**: the same scorer is evaluated per node and simply not averaged. The price is statistical, not computational: each node's weight is estimated from that node's evidence alone.

**What node-specific attention buys in this project** (self-check 2, answered in full in section 8):

1. *Coverage heterogeneity.* Some drugs have no SMILES, and half of the diseases have no curated genes. A global weight would apply the same mix to every node. Node-specific weights plus masking let each entity use the views it actually has.
2. *Reliability heterogeneity.* Even when a view exists, its quality varies from node to node: well-studied drugs have rich CTD gene sets (literature bias), obscure ones have a handful of genes. A per-node weight can reflect this.
3. *Interpretability per entity.* For a case study we can report "for disease X the model relied 45% on gene evidence", which a single global number cannot say.

**A crucial limitation.** The score $s_i^\Phi=q^\top\tanh(Wz_i^\Phi+b)$ looks only at the **message** $z_i^\Phi$, not at the node's own state $h_i$. The attention can learn "messages that look like *this* are trustworthy", but it cannot learn "for nodes like *me*, view A is trustworthy". If a view is *misleading* (confident but wrong) rather than *noisy*, its messages look just as confident as the good view's, and the scorer cannot tell them apart. The experiment in section 4.4 and Exercise 10 demonstrate this. A *context-aware* score $q^\top\tanh(W[z_i^\Phi\Vert h_i]+b)$ is the natural extension.

### 2.7 Multi-view learning and fusion strategies

**Multi-view learning** studies how to learn from several descriptions (views) of the same objects. Two classic principles guide it:

- **Consensus:** views should agree on the underlying structure. Co-training and co-regularisation push predictions from different views to agree.
- **Complementarity:** each view carries information the others lack. The point of fusion is to collect it.

Our six similarity views are a textbook multi-view setting: chemical structure, gene effects, phenotype and ontology position each describe drugs or diseases from a different angle.

**When to fuse.**

| Stage | Name | Project example |
|---|---|---|
| inputs | **early fusion** | `feat_views="all"`: concatenate the similarity rows of all views into the input features |
| inside the encoder | **intermediate fusion** | `ViewAttention`: fuse per-relation messages at every layer |
| outputs | **late fusion** | the propagation head: $\sum_v w_v P_v[i,j]$, one additive score per view |

MV-HGAT uses **all three**. That is unusual and deliberate. Early fusion gives the input layer everything. Intermediate fusion lets each node choose its views. Late fusion gives an interpretable, per-view additive explanation (Unit C6 and E2).

**How to fuse.** Let $m_i^1,\dots,m_i^P\in\mathbb{R}^d$ be node $i$'s view messages.

| Strategy | Formula | Extra params | Properties |
|---|---|---|---|
| concatenation | $z_i = U[m_i^1\Vert\cdots\Vert m_i^P]$ | $P d\cdot d$ | most expressive *linear* fusion; can weight views differently per dimension; parameters grow with $P$; no explicit view weights to inspect; cannot handle a missing view (zeros are fed in) |
| mean / sum | $z_i=\frac1P\sum_v m_i^v$ | 0 | robust, no overfitting; noisy views dilute good ones; the project's `uniform_attention=True` ablation (masked mean over *valid* views) |
| max-pooling | $z_{i,k}=\max_v m^v_{i,k}$ | 0 | picks the strongest evidence per dimension; brittle to outliers |
| global attention (HAN) | $z_i=\sum_v\beta_v m_i^v$ | $d\,d_a+d_a+d_a$ | one interpretable weight per view; convex combination |
| node-specific attention | $z_i=\sum_v\beta_i^v m_i^v$ | same | per-node weights; masking of missing views; convex |
| gating | $z_i=\sum_v g_i^v m_i^v$, $g_i^v=\mathrm{sigmoid}(u^\top m_i^v+c)$ | $d+1$ | views do **not** compete; weights need not sum to one; can amplify agreement or switch all views off |
| late fusion | $\text{score}=\sum_v w_v f_v(i,j)$ | $P$ | per-view scores are explicit, so the explanation is faithful; no interaction between views before scoring |

**Guidelines.**

- With few views and plenty of data, concatenation is a strong, hard-to-beat baseline (Exercise 11 and the Simple-HGN finding below).
- When some nodes **lack** some views, use masked attention or masked mean. Concatenation feeds zeros, which the model may confuse with "dissimilar to everything".
- When you need **interpretability**, prefer late fusion with additive terms (faithful) over attention weights (descriptive only; see Unit E2, Jain & Wallace 2019).
- When views are **redundant** (two chemical fingerprints), attention will split weight between them somewhat arbitrarily. Do not read the split as "CDK matters 40%, ECFP 60%".

### 2.8 Other heterogeneous GNNs in brief

**HGT: Heterogeneous Graph Transformer (Hu, Dong, Wang & Sun, WWW 2020).** HGT drops hand-picked meta-paths and works on the raw schema. Every edge $(s,e,t)$ has a **meta-relation** $\langle\tau(s),\phi(e),\tau(t)\rangle$ (source type, edge type, target type). Attention is Transformer-style, but every projection depends on types:

$$
\text{ATT}^{k}(s,e,t)=\Big(\text{K}^k_{\tau(s)}(h_s)\;W^{\text{ATT}}_{\phi(e)}\;\text{Q}^k_{\tau(t)}(h_t)^\top\Big)\cdot\frac{\mu_{\langle\tau(s),\phi(e),\tau(t)\rangle}}{\sqrt{d}},
$$

with type-specific key/query/value projections, an edge-type matrix $W^{\text{ATT}}_{\phi(e)}$, and a learned prior $\mu$ per meta-relation. Messages likewise use $W^{\text{MSG}}_{\phi(e)}$. The softmax runs over **all** neighbours of all types at once, so relations compete at the neighbour level instead of in a separate semantic stage. Stacking $L$ layers lets the model learn "soft meta-paths" of length up to $L$. HGT also adds relative temporal encoding and a sampler (HGSampling) for web-scale graphs. *Lesson:* meta-paths can be learned implicitly. *Cost:* many more parameters, which a 600-node graph cannot feed.

**MAGNN: Metapath Aggregated GNN (Fu et al., WWW 2020).** HAN only looks at the *endpoints* of a meta-path instance and forgets the intermediate nodes (the gene in DGD). MAGNN encodes **the whole path instance**: all nodes along it are combined by an encoder (mean, linear, or a "relational rotation" encoder inspired by RotatE). It then applies attention over instances (intra-metapath) and attention over meta-paths (inter-metapath, like HAN). For this project, MAGNN-style encoding is how you would let the model learn *which genes* make two drugs similar.

**HetGNN (Zhang et al., KDD 2019).** HetGNN *samples* a fixed number of strongly correlated neighbours of each type with random walks with restart. It encodes heterogeneous *content* (text, images, attributes) with Bi-LSTMs, aggregates same-type neighbours with another Bi-LSTM, and combines types with attention. It is trained with an unsupervised skip-gram-like objective with negative sampling. *Lesson:* sampling by type keeps hubs from dominating and handles rich node content.

**Simple-HGN and the reality check (Lv et al., KDD 2021, "Are we really making much progress?").** The authors reproduced 12 recent heterogeneous GNNs (including HAN, HGT, MAGNN, HetGNN, GTN and RGCN) with their official code and settings, and compared them against properly configured baselines. Their headline finding: **simple homogeneous GNNs such as GCN and GAT had been "largely underestimated due to improper settings"**, and GAT with proper inputs generally matched or beat all of the specialised models. Each paper had used its own data processing, splits and evaluation set-up, which made fair comparison impossible. They released the **Heterogeneous Graph Benchmark (HGB)** and proposed **Simple-HGN**: a GAT whose attention also sees a learnable **edge-type embedding**,

$$
\alpha_{ij}=\mathrm{softmax}_j\Big(\mathrm{LeakyReLU}\big(a^\top[W h_i\Vert W h_j\Vert W_r\, r_{\psi(\langle i,j\rangle)}]\big)\Big),
$$

plus residual connections (on nodes and on attention scores) and $L_2$ normalisation of the output embeddings. It beat the specialised models on HGB. *Lesson for this project:* compare against strong, fairly tuned baselines on identical splits (the project does this in `04_ablation.py` and with the "benchmark similarities only" variant), and do not assume architectural novelty is what produces gains.

**Others you may meet.** *metapath2vec* (Dong et al. 2017) runs random walks guided by meta-paths plus a heterogeneous skip-gram (the shallow-embedding analogue of HAN). *GTN* (Graph Transformer Networks) learns soft meta-paths by multiplying soft selections of adjacency matrices. *HeCo* (Wang et al. 2021) is self-supervised: contrastive learning between a schema view and a meta-path view.

| Model | Neighbour selection | Meta-paths | Intermediate nodes | Relation weights | Year |
|---|---|---|---|---|---|
| R-GCN | 1-hop per relation | no (stack layers) | — | full / basis / block $W_r$ | 2018 |
| HAN | meta-path neighbours | hand-picked | ignored | per-meta-path attention $a_\Phi$ | 2019 |
| HetGNN | RWR sampling by type | no | — | type-wise Bi-LSTM + attention | 2019 |
| MAGNN | meta-path instances | hand-picked | **encoded** | instance + meta-path attention | 2020 |
| HGT | 1-hop, all types | learned implicitly | via stacking | type-dependent Q/K/V, $W^{ATT}_\phi$, $W^{MSG}_\phi$ | 2020 |
| Simple-HGN | 1-hop | no | — | edge-type embedding in GAT attention | 2021 |
| **MV-HGAT (this project)** | 1-hop per relation; relations = k-NN views + assoc (+ bridge) | gene meta-paths pre-computed | ignored (like HAN) | relation-specific GAT + **node-specific** view attention | — |

### 2.9 When do extra views help, and when do they hurt?

Adding a view is not free. A view **helps** when it brings *complementary signal* for nodes or pairs that the other views handle badly. It **hurts or does nothing** when:

1. **It carries no signal** for the task. Its relation then injects noise into every node's messages, and attention can only partly down-weight it (attention is a convex mix, and softmax weights rarely reach exactly 0).
2. **It is redundant.** A second chemical fingerprint mostly duplicates the first; it adds parameters (one more GAT per layer) without new information.
3. **It covers few nodes.** For uncovered nodes the view is an empty or self-only relation, and the model must learn to ignore it.
4. **It adds parameters** on a small dataset. Each relation adds about 8k parameters per layer at $d=64$ (section 5), learned from the same ~1,500 training links.
5. **It leaks.** A view derived from the labels (for example CTD *inferred* gene–disease links, which partly come from known treatments) can inflate scores. The project deliberately excludes those (`HOW_IT_WORKS.md` §6, items 7–8).

**Project evidence.** The views were first tested *alone* with a non-learned propagation score (`HOW_IT_WORKS.md` §10):

| View alone | Fdataset AUPR | Reading |
|---|---|---|
| cold start: `pheno_mim` | 0.149 | the benchmark's own disease view is strong |
| cold start: `sem_mondo` | 0.106 | weaker alone... |
| cold start: `pheno_mim + sem_mondo` | **0.175** | ...but complementary: together better than either |
| cold start: `gene_d` | 0.057 | weak |
| cold start: `gene_bridge` | 0.011 | random level (positive rate ≈ 0.010); AUC 0.52–0.57 |
| warm: `chem_cdk` | 0.061 | benchmark drug view |
| warm: `chem_ecfp` | **0.091** | a modern fingerprint beats the benchmark's |
| warm: `gene_r` | 0.031 | weak |

The **gene bridge** is the clearest example of a view that cannot help: alone it is at chance level. That is why `use_bridge=False` by default. The ablation (section 5.6) shows that adding it costs about 0.024 AUPR. The project also observed (`HOW_IT_WORKS.md` §10, point 3) that *concatenating* more similarity views into a plain GNN slightly **hurt**, because there were more parameters and no more labels. The views only paid off once the late-fusion propagation head let their signal through additively.

---

## 3. Putting it together: one MV-HGAT layer in equations

Before the code, here is the project's encoder written in the notation of this chapter, so that sections 4 and 5 have a precise target. Node $i$ has type $\phi(i)\in\{\text{drug},\text{disease}\}$. Relation $r$ has a destination type and a source type; $\mathcal{R}_T$ is the set of relations whose destination type is $T$.

**Input projection (HAN's type-specific $M_\phi$).**

$$
h_i^{(0)}=\mathrm{Dropout}\big(\mathrm{ELU}(M_{\phi(i)}x_i)\big),\qquad
x_i = [\,\text{similarity rows of all views}\;\Vert\;\text{visible link row}\,].
$$

**Node-level attention, one GAT per relation, $K=4$ heads, relation-specific weights.** For relation $r$, head $k$:

$$
u_j=W^{r}_{\text{src}}h_j,\quad v_i=W^{r}_{\text{dst}}h_i,\qquad
e^{r,k}_{ij}=\mathrm{LeakyReLU}_{0.2}\big(a^{r,k\top}_{\text{dst}}v_i^{k}+a^{r,k\top}_{\text{src}}u_j^{k}\big),
$$

$$
\alpha^{r,k}_{ij}=\frac{\exp e^{r,k}_{ij}}{\sum_{l\in\mathcal{N}^r(i)}\exp e^{r,k}_{il}},\qquad
m_i^{r}=\big\Vert_{k=1}^{K}\sum_{j\in\mathcal{N}^r(i)}\alpha^{r,k}_{ij}\,u_j^{k},\qquad
\text{valid}_i^r=\big[\mathcal{N}^r(i)\neq\varnothing\big].
$$

**Semantic (view) attention, node-specific and masked.**

$$
s_i^r=q_{\phi(i)}^\top\tanh\big(P_{\phi(i)}m_i^r+p_{\phi(i)}\big),\qquad
\beta_i^r=\frac{\text{valid}_i^r\,\exp s_i^r}{\sum_{r'\in\mathcal{R}_{\phi(i)}}\text{valid}_i^{r'}\exp s_i^{r'}},\qquad
z_i=\sum_{r\in\mathcal{R}_{\phi(i)}}\beta_i^r m_i^r .
$$

**Update with skip connection (R-GCN's $W_0$) and LayerNorm.**

$$
h_i^{(l+1)}=\mathrm{Dropout}\Big(\mathrm{ELU}\big(\mathrm{LayerNorm}_{\phi(i)}(z_i+S_{\phi(i)}h_i^{(l)})\big)\Big).
$$

**Jumping knowledge and decoder** (decoder details in Unit C6):

$$
H_i=[h_i^{(0)}\Vert h_i^{(1)}\Vert h_i^{(2)}]\in\mathbb{R}^{192},\qquad
\text{logit}(i,j)=g(i,j)\,H_i^\top W H_j+\sum_v w_vP_v[i,j]+b .
$$

Compared with HAN, the project:

1. uses **relations** (1-hop typed edges, some of them pre-computed meta-paths) rather than only meta-paths;
2. gives every relation its own projection matrices, which is the **R-GCN** idea;
3. computes $\beta$ **per node**, with a **validity mask**;
4. adds a skip connection, LayerNorm and jumping knowledge;
5. has one relation (`assoc`) that **changes every epoch**, because known links are hidden at random during training (Unit C6).

---

## 4. Code: building it from scratch

All code runs on CPU in a few seconds to about two minutes, using only `numpy` and `torch` (no `torch_geometric`). Run each block from a folder of your choice with the project's interpreter:

```powershell
& "C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe" metapaths.py
```

The outputs shown were produced by exactly this code (PyTorch 2.14, NumPy 2.5, CPU). Different library versions or thread counts can change the last digit of trained results, but not the conclusions.

### 4.1 Meta-paths as matrix products (worked example 1)

Save as `metapaths.py`:

```python
import numpy as np

drugs, diseases, genes = ["d1", "d2", "d3"], ["s1", "s2"], ["g1", "g2", "g3"]
R = np.array([[1, 1, 0],      # drug-gene (d1 hits g1,g2; d2 hits g2; d3 hits g3)
              [0, 1, 0],
              [0, 0, 1]])
Q = np.array([[1, 1, 0],      # disease-gene (s1 involves g1,g2; s2 involves g3)
              [0, 0, 1]])
A = np.array([[1, 0],         # known indications: d1 treats s1, d3 treats s2
              [0, 0],
              [0, 1]])

DGD = R @ R.T        # number of drug-gene-drug path instances
SGS = Q @ Q.T        # disease-gene-disease
DGS = R @ Q.T        # drug-gene-disease (the "gene bridge")
DSD = A @ A.T        # drug-disease-drug (co-indication)
print("DGD =\n", DGD); print("SGS =\n", SGS); print("DGS =\n", DGS); print("DSD =\n", DSD)

def pathsim(M):                      # Sun et al. 2011, symmetric meta-paths only
    d = np.diag(M)
    return 2 * M / (d[:, None] + d[None, :])

def jaccard(X):                      # what the project uses for gene_r / gene_d
    inter = X @ X.T
    size = X.sum(1)
    return inter / (size[:, None] + size[None, :] - inter)

def cosine(X, Y):                    # what the project uses for the gene bridge
    return (X @ Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))

np.set_printoptions(precision=3, suppress=True)
print("PathSim(DGD) =\n", pathsim(DGD))
print("Jaccard(drug gene sets) =\n", jaccard(R))
print("cosine(drug, disease gene profiles) =\n", cosine(R, Q))
```

Expected output:

```text
DGD =
 [[2 1 0]
 [1 1 0]
 [0 0 1]]
SGS =
 [[2 0]
 [0 1]]
DGS =
 [[2 0]
 [1 0]
 [0 1]]
DSD =
 [[1 0 0]
 [0 0 0]
 [0 0 1]]
PathSim(DGD) =
 [[1.    0.667 0.   ]
 [0.667 1.    0.   ]
 [0.    0.    1.   ]]
Jaccard(drug gene sets) =
 [[1.  0.5 0. ]
 [0.5 1.  0. ]
 [0.  0.  1. ]]
cosine(drug, disease gene profiles) =
 [[1.    0.   ]
 [0.707 0.   ]
 [0.    1.   ]]
```

**Read the output.** `DGD`, `DGS` and `DSD` match the hand calculation in section 2.2. PathSim and Jaccard give different values for $(d_1,d_2)$ (0.667 and 0.5) but the same *ranking*. The `DSD` row for $d_2$ is all zeros: $d_2$ has no co-indicated drugs, not even itself. The gene-bridge cosine says that $d_2$, which hits only $g_2$, is 0.707-similar to disease $s_1$. If we trusted that bridge, we would predict "$d_2$ treats $s_1$".

### 4.2 An R-GCN layer with basis and block decompositions (worked example 2)

Save as `rgcn.py`:

```python
import torch
import torch.nn as nn

class RGCNLayer(nn.Module):
    """h_i' = act( W_0 h_i + sum_r sum_{j in N_r(i)} (1/c_{i,r}) W_r h_j )

    adj[r] is a dense (N x N) 0/1 matrix: adj[r][i, j] = 1 means j -> i in relation r.
    mode = "full"  : one free matrix per relation
           "basis" : W_r = sum_b a[r, b] V_b         (B shared bases)
           "block" : W_r = blockdiag(Q_r1, ..., Q_rB) (B blocks per relation)
    """
    def __init__(self, n_rel, d_in, d_out, mode="full", n_bases=2, act=torch.relu):
        super().__init__()
        self.mode, self.n_rel, self.act = mode, n_rel, act
        if mode == "full":
            self.W = nn.Parameter(torch.randn(n_rel, d_in, d_out) * 0.1)
        elif mode == "basis":
            self.V = nn.Parameter(torch.randn(n_bases, d_in, d_out) * 0.1)
            self.a = nn.Parameter(torch.randn(n_rel, n_bases) * 0.1)
        elif mode == "block":
            assert d_in % n_bases == 0 and d_out % n_bases == 0
            self.Qb = nn.Parameter(torch.randn(n_rel, n_bases, d_in // n_bases, d_out // n_bases) * 0.1)
        self.W0 = nn.Parameter(torch.randn(d_in, d_out) * 0.1)   # self-connection

    def rel_weights(self):                                       # (R, d_in, d_out)
        if self.mode == "full":
            return self.W
        if self.mode == "basis":
            return torch.einsum("rb,bio->rio", self.a, self.V)
        return torch.stack([torch.block_diag(*self.Qb[r]) for r in range(self.n_rel)])

    def forward(self, h, adj):
        W = self.rel_weights()
        out = h @ self.W0
        for r in range(self.n_rel):
            c = adj[r].sum(1, keepdim=True).clamp(min=1)          # c_{i,r} = |N_r(i)|
            out = out + (adj[r] / c) @ (h @ W[r])                 # mean over r-neighbours
        return self.act(out)

# ---- the hand-worked example: nodes d1 d2 d3 s1 s2 (indices 0..4) -------------
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 0.], [0., 1.]])
N = 5
adj = torch.zeros(3, N, N)
adj[0, 0, 1] = adj[0, 1, 0] = 1          # r0 "shares-gene": d1 <-> d2
adj[1, 3, 0] = 1; adj[1, 4, 2] = 1       # r1 "treats":     d1 -> s1, d3 -> s2
adj[2, 0, 3] = 1; adj[2, 2, 4] = 1       # r2 "treated-by": s1 -> d1, s2 -> d3

layer = RGCNLayer(n_rel=3, d_in=2, d_out=2, mode="basis", n_bases=2)
with torch.no_grad():                    # set the weights used in the hand calculation
    layer.V[0] = torch.eye(2)                          # V_1 = I
    layer.V[1] = torch.tensor([[0., 1.], [1., 0.]])    # V_2 = swap
    layer.a[:] = torch.tensor([[0.5, 0.0],             # W_shares  = 0.5 I
                               [0.0, 1.0],             # W_treats  = swap
                               [1.0, 0.5]])            # W_treated = I + 0.5 swap
    layer.W0[:] = torch.eye(2)
    print("W_r from bases:\n", layer.rel_weights())
    print("h' =\n", layer(x, adj))

# ---- parameter counts at the project's scale: 8 relations, 64 -> 64 -----------
for mode, B in [("full", None), ("basis", 2), ("basis", 4), ("block", 4), ("block", 16)]:
    l = RGCNLayer(8, 64, 64, mode=mode, n_bases=B or 1)
    n = sum(p.numel() for p in l.parameters()) - 64 * 64   # exclude W0
    print(f"{mode:5s} B={str(B):4s}: {n:6d} relation parameters")
```

Expected output:

```text
W_r from bases:
 tensor([[[0.5000, 0.0000],
         [0.0000, 0.5000]],

        [[0.0000, 1.0000],
         [1.0000, 0.0000]],

        [[1.0000, 0.5000],
         [0.5000, 1.0000]]])
h' =
 tensor([[2.0000, 1.0000],
        [0.5000, 1.0000],
        [1.5000, 2.0000],
        [1.0000, 1.0000],
        [1.0000, 2.0000]])
full  B=None:  32768 relation parameters
basis B=2   :   8208 relation parameters
basis B=4   :  16416 relation parameters
block B=4   :   8192 relation parameters
block B=16  :   2048 relation parameters
```

**Read the output.** The three relation matrices built from two bases are exactly $0.5I$, the swap, and $I+0.5\,\text{swap}$. The layer output reproduces worked example 2 row by row: $d_1\to[2,1]$, $d_2\to[0.5,1]$, $d_3\to[1.5,2]$, $s_1\to[1,1]$, $s_2\to[1,2]$. The parameter counts match the table in section 2.4.

*Try this:* set `mode="full"` and copy the three matrices into `layer.W`. The output is the same, because basis decomposition is a *restriction* of the full model, not a different model.

### 4.3 Semantic attention: global (HAN) versus node-specific (worked example 3)

Save as `semantic.py`:

```python
import torch

# meta-path-specific embeddings z_i^Phi for drugs d1, d2, d3 (output of node-level attention)
Z = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]],     # Phi_1 = drug-gene-drug
                  [[2.0, 1.0], [0.0, 0.0], [1.0, 1.0]]])    # Phi_2 = drug-disease-drug
W, b, q = torch.eye(2), torch.zeros(2), torch.tensor([1.0, 1.0])

s = torch.tanh(Z @ W.T + b) @ q          # (P, N): score of meta-path P for node i
print("per-node scores s[P, i] =\n", s)

# --- HAN (Wang et al. 2019): average scores over nodes, ONE beta per meta-path ---
w = s.mean(1)
beta_global = torch.softmax(w, 0)
print("w_Phi =", w, " beta (global) =", beta_global)
z_global = (beta_global[:, None, None] * Z).sum(0)

# --- node-specific variant (the project's ViewAttention): softmax per node --------
beta_node = torch.softmax(s, 0)
print("beta (node-specific), columns = d1 d2 d3 =\n", beta_node)

# --- with a validity mask: d2 has NO drug-disease-drug neighbours at all ------------
valid = torch.tensor([[True, True, True], [True, False, True]])
beta_masked = torch.softmax(s.masked_fill(~valid, float("-inf")), 0)
print("beta (node-specific + mask) =\n", beta_masked)
print("fused z, global      =\n", z_global)
print("fused z, node+mask   =\n", (beta_masked[..., None] * Z).sum(0))
```

Expected output:

```text
per-node scores s[P, i] =
 tensor([[0.7616, 0.9242, 0.0000],
        [1.7256, 0.0000, 1.5232]])
w_Phi = tensor([0.5619, 1.0829])  beta (global) = tensor([0.3726, 0.6274])
beta (node-specific), columns = d1 d2 d3 =
 tensor([[0.2761, 0.7159, 0.1790],
        [0.7239, 0.2841, 0.8210]])
beta (node-specific + mask) =
 tensor([[0.2761, 1.0000, 0.1790],
        [0.7239, 0.0000, 0.8210]])
fused z, global      =
 tensor([[1.6274, 0.6274],
        [0.1863, 0.1863],
        [0.6274, 0.6274]])
fused z, node+mask   =
 tensor([[1.7239, 0.7239],
        [0.5000, 0.5000],
        [0.8210, 0.8210]])
```

**Read the output.** Global attention gives every drug the same $(0.3726, 0.6274)$ split, which dilutes $d_2$'s only real evidence to $[0.186,0.186]$. Node-specific attention gives $d_2$ $(0.716,0.284)$. Adding the mask gives $(1,0)$ and restores $[0.5,0.5]$. This is the project's `ViewAttention` behaviour in miniature.

### 4.4 A tiny HAN from scratch, and five ways to fuse views

This experiment builds a complete one-layer HAN (node-level GAT per view, then a fusion step, then a classifier). It compares five fusion rules on a synthetic node-classification task designed to need *per-node* view selection:

- 400 nodes, two classes, weak features (one node's features alone are a poor guide to its class);
- **view A** is reliable (same-class neighbours) for *group-0* nodes and **view B** for *group-1* nodes;
- for the other group, the view is **unreliable**. In scenario `"noisy"` it has 2 random neighbours, like a sparse, low-quality similarity row. In scenario `"opposite"` it has 10 neighbours *of the other class*, i.e. it is confidently misleading.

Save as `fusion.py` (it takes about 1.5 minutes on a laptop CPU):

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_views(n=400, deg=10, scenario="noisy", seed=0):
    """Two classes, weak node features, two views (relations).
    View A is reliable (same-class neighbours) for group-0 nodes, view B for group-1.
    For the OTHER group the view is unreliable:
      scenario="noisy"    -> 2 random neighbours (a sparse, noisy similarity row)
      scenario="opposite" -> 10 neighbours of the opposite class (misleading)"""
    rng = np.random.default_rng(seed)
    y = rng.integers(2, size=n)
    group = rng.integers(2, size=n)
    mu = np.zeros(16); mu[:4] = 0.5
    X = (2 * y[:, None] - 1) * mu + rng.normal(scale=1.5, size=(n, 16))
    same = [np.flatnonzero(y == c) for c in (0, 1)]
    def graph(good_group):
        M = np.zeros((n, n), dtype=bool)
        for i in range(n):
            if group[i] == good_group:
                M[i, rng.choice(same[y[i]], deg, replace=False)] = True
            elif scenario == "noisy":
                M[i, rng.choice(n, 2, replace=False)] = True
            else:
                M[i, rng.choice(same[1 - y[i]], deg, replace=False)] = True
        np.fill_diagonal(M, True)                 # self-loop, as knn_mask does
        return M
    views = np.stack([graph(0), graph(1)])
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(y),
            torch.tensor(views), group)

class NodeLevelGAT(nn.Module):
    """Single-head GAT restricted to one view / meta-path (dense mask, dst x src)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(d_out) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(d_out) * 0.1)
    def forward(self, h, mask):
        z = self.W(h)
        e = F.leaky_relu((z @ self.a_dst)[:, None] + (z @ self.a_src)[None, :], 0.2)
        alpha = torch.softmax(e.masked_fill(~mask, float("-inf")), dim=1)
        return F.elu(alpha @ z)

class Fusion(nn.Module):
    def __init__(self, kind, n_views, d, d_att=32):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(d, d_att)                   # HAN: W, b
        self.q = nn.Linear(d_att, 1, bias=False)          # HAN: q
        self.gate = nn.Linear(d, 1)                       # gating
        self.cat = nn.Linear(n_views * d, d)              # concatenation
    def forward(self, M):                                 # M: (P, N, d)
        if self.kind == "mean":
            return M.mean(0), None
        if self.kind == "concat":
            return self.cat(torch.cat(list(M), dim=1)), None
        if self.kind == "gate":                           # independent sigmoid per view
            g = torch.sigmoid(self.gate(M))               # (P, N, 1)
            return (g * M).sum(0), g.squeeze(-1)
        s = self.q(torch.tanh(self.proj(M))).squeeze(-1)  # (P, N) per-node scores
        if self.kind == "global":                         # original HAN: average over nodes
            beta = torch.softmax(s.mean(1), 0)[:, None].expand_as(s)
        else:                                             # "node": per-node softmax
            beta = torch.softmax(s, 0)
        return (beta[..., None] * M).sum(0), beta

class TinyHAN(nn.Module):
    def __init__(self, n_views, kind, d_in=16, d=32):
        super().__init__()
        self.gats = nn.ModuleList(NodeLevelGAT(d_in, d) for _ in range(n_views))
        self.fuse = Fusion(kind, n_views, d)
        self.out = nn.Linear(d, 2)
    def forward(self, X, views):
        M = torch.stack([g(X, v) for g, v in zip(self.gats, views)])
        z, beta = self.fuse(M)
        return self.out(F.elu(z)), beta

def run(kind, scenario, seed, epochs=200):
    X, y, views, group = make_views(scenario=scenario, seed=seed)
    torch.manual_seed(seed)
    train = torch.rand(len(y)) < 0.5
    model = TinyHAN(len(views), kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        logits, _ = model(X, views)
        loss = F.cross_entropy(logits[train], y[train])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits, beta = model(X, views)
    acc = (logits.argmax(1) == y)[~train].float().mean().item()
    return acc, beta, group

if __name__ == "__main__":
    for scenario in ("noisy", "opposite"):
        print(f"--- scenario: {scenario} ---")
        for kind in ("mean", "concat", "gate", "global", "node"):
            accs = [run(kind, scenario, s)[0] for s in range(3)]
            print(f"{kind:7s} test acc = {np.mean(accs):.3f}  (seeds "
                  + " ".join(f"{a:.3f}" for a in accs) + ")")
    for kind in ("global", "node"):
        acc, beta, group = run(kind, "noisy", 0)
        print(f"{kind:6s} beta on view A: group-0 nodes {beta[0, group == 0].mean():.2f}, "
              f"group-1 nodes {beta[0, group == 1].mean():.2f}")
```

Expected output:

```text
--- scenario: noisy ---
mean    test acc = 0.831  (seeds 0.799 0.848 0.845)
concat  test acc = 0.834  (seeds 0.829 0.819 0.856)
gate    test acc = 0.848  (seeds 0.834 0.828 0.881)
global  test acc = 0.824  (seeds 0.809 0.828 0.835)
node    test acc = 0.828  (seeds 0.869 0.784 0.830)
--- scenario: opposite ---
mean    test acc = 0.563  (seeds 0.563 0.544 0.582)
concat  test acc = 0.606  (seeds 0.588 0.544 0.686)
gate    test acc = 0.590  (seeds 0.598 0.539 0.634)
global  test acc = 0.588  (seeds 0.618 0.544 0.603)
node    test acc = 0.564  (seeds 0.598 0.475 0.619)
global beta on view A: group-0 nodes 0.58, group-1 nodes 0.58
node   beta on view A: group-0 nodes 0.65, group-1 nodes 0.38
```

**Read the output carefully; it is more instructive than a clean win would be.**

1. **Node-specific attention learns the right thing.** On the noisy scenario its $\beta$ on view A is 0.65 for group-0 nodes (where A is reliable) and 0.38 for group-1 nodes (where it is not). Global attention gives 0.58 to everyone, and it *cannot* do otherwise.
2. **But learning the right weights did not reliably improve accuracy.** Averaged over three seeds, all five rules land between 0.82 and 0.85. Node-specific attention scored 0.869 on seed 0 (the best single run), but only 0.784 on seed 1. With 200 test nodes, one seed's accuracy has a standard error of about $\sqrt{0.83\cdot0.17/200}\approx0.027$, so differences of 0.02 are noise. Attention weights are learned jointly with everything else and are hard to train well on small data. That matches both the project's ablation (view attention is worth about 0.02 AUPR; section 5.6) and Simple-HGN's warning that simple fusion is a strong baseline.
3. **In the "opposite" scenario every rule fails** (0.56–0.61). A misleading view sends messages that look just as confident as the reliable view's. A scorer that sees only the message cannot tell which one is lying. This is the limitation discussed at the end of section 2.6. Exercise 10 tries a context-aware fix.

### 4.5 Poking the project's real model

You can import the project's classes (read-only) and look at what they return. Save as `poke_project.py`:

```python
import sys
import torch
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import MVHGAT                      # the project's real model (CPU here)

torch.manual_seed(0)
n_r, n_d = 4, 3
relations = {"view:chem": ("drug", "drug"), "view:pheno": ("disease", "disease"),
             "assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")}
A = torch.tensor([[1, 0, 0],                         # drug 3 has NO visible links
                  [1, 1, 0],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=torch.bool)
graphs = {"view:chem": torch.eye(n_r, dtype=torch.bool) | torch.tensor(
              [[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
          "view:pheno": torch.ones(n_d, n_d, dtype=torch.bool),
          "assoc>drug": A, "assoc>disease": A.T}
X = {"drug": torch.randn(n_r, 5), "disease": torch.randn(n_d, 5)}
model = MVHGAT(relations, 5, 5, hidden=8, layers=2, heads=2, dropout=0.0, n_prop=2).eval()

P = torch.rand(2, n_r, n_d)                          # fake propagation slices (2 views)
deg = (A.float().sum(1), A.float().sum(0))
with torch.no_grad():
    logits, betas = model(X, graphs, P=P, deg=deg)
    print("logits shape:", tuple(logits.shape))
    names, beta = betas[0]["drug"]
    print("layer-1 drug relations:", names)
    print("layer-1 drug beta (rows = relations, cols = drugs):\n", beta.numpy().round(3))
    print("view weights w_v = softplus(0) * 5 =", model.view_weights().numpy().round(3))
    print("gate (drugs x diseases) at initialisation:\n", model.gnn_gate(*deg).numpy().round(3))
```

Expected output:

```text
logits shape: (4, 3)
layer-1 drug relations: ['view:chem', 'assoc>drug']
layer-1 drug beta (rows = relations, cols = drugs):
 [[0.46  0.447 0.468 1.   ]
 [0.54  0.553 0.532 0.   ]]
view weights w_v = softplus(0) * 5 = [3.466 3.466]
gate (drugs x diseases) at initialisation:
 [[0.5   0.444 0.444]
 [0.563 0.5   0.5  ]
 [0.5   0.444 0.444]
 [0.375 0.333 0.333]]
```

**Read the output.**

- `betas[0]["drug"]` holds the layer-1 relation names for drugs and a $(R\times n)$ matrix of weights. Each column sums to 1.
- **Drug 3 has no visible links**, so its `assoc>drug` relation is invalid and gets $\beta=0$; its whole weight goes to the chemical view. The other drugs split roughly 45/55 because the model is untrained.
- `view_weights` at initialisation is $\mathrm{softplus}(0)\times5=\ln2\times5=3.466$ for every view.
- The **degree gate** at initialisation is $\mathrm{sigmoid}(\log(1+\deg_i))\cdot\mathrm{sigmoid}(\log(1+\deg_j))$. A useful identity: $\mathrm{sigmoid}(\ln(1+d))=\frac{1+d}{2+d}$, so degrees 0, 1, 2 give 0.5, 0.667, 0.75. For drug 0 (degree 1) and disease 0 (degree 2) the gate is $0.667\times0.75=0.5$, matching the printed matrix. Drug 3 (degree 0) has the smallest gates. The gate is covered properly in Unit C6.

---

## 5. In this project: a line-by-line walkthrough

### 5.1 Building the heterogeneous graph: `methods.py::MVHGATMethod.build`

```python
rv = list(c.drug_views or data.drug_view_names)       # ['chem_cdk', 'chem_ecfp', 'gene_r']
dv = list(c.disease_views or data.disease_view_names) # ['pheno_mim', 'sem_mondo', 'gene_d']
```

These are the view names. The ablations change them, for example `"no_gene_views": {"drug_views": ("chem_cdk","chem_ecfp"), ...}` in `scripts/04_ablation.py`.

```python
X = {"drug": feats([data.drug_view(v) for v in rv]),
     "disease": feats([data.disease_view(v) for v in dv])}
```

**Early fusion.** With `feat_views="all"`, each node's input is the concatenation of its rows in every view (`np.concatenate(mats, 1)`). `fill_missing` turns NaN (uncovered entity) into 0 with a 1 on the diagonal, so an uncovered drug's chemical row is "similar only to itself". Later, `fit_predict.features` appends the visible link row.

```python
for v in rv:
    relations[f"view:{v}"] = ("drug", "drug")
    graphs[f"view:{v}"] = t(knn_mask(data.drug_view(v), c.k), torch.bool)
```

**One relation per view.** `knn_mask` (in `data.py`) keeps each node's top-$k=10$ most similar nodes with similarity $>0$, symmetrises (`M |= M.T`), and adds self-loops (`np.fill_diagonal(M, True)`). The result is a dense boolean `dst x src` mask. The same loop builds the disease views.

```python
if c.use_assoc_edges:
    relations["assoc>drug"] = ("drug", "disease")
    relations["assoc>disease"] = ("disease", "drug")
    A = t(A_train > 0, torch.bool)
    graphs["assoc>drug"], graphs["assoc>disease"] = A, A.T
```

**The known-link relation and its inverse**, built from the *training* matrix only (test links are zero in `A_train`). During training this mask is replaced every epoch by the *visible* links `Am`, and supervision is placed on the hidden ones (Unit C6).

```python
if c.use_bridge:
    B = data.gene_bridge
    relations["gene_bridge>drug"] = ("drug", "disease")
    relations["gene_bridge>disease"] = ("disease", "drug")
    graphs["gene_bridge>drug"] = t(topk_bipartite(B, c.bridge_k), torch.bool)
    graphs["gene_bridge>disease"] = t(topk_bipartite(B.T, c.bridge_k), torch.bool)
```

**The folded drug–gene–disease meta-path.** `data.gene_bridge` is the cosine of gene profiles (section 2.2). `topk_bipartite` keeps the top 10 diseases per drug, and the top 10 drugs per disease for the reverse relation. Note that the reverse relation is *not* the transpose of the forward one: "disease $j$ is among drug $i$'s top-10" is not the same as "drug $i$ is among disease $j$'s top-10". Off by default.

So the default schema has **2 node types and 8 relations**: 3 drug views, 3 disease views, and assoc in both directions. With the bridge there are 10.

### 5.2 Node-level attention: `model.py::DenseGAT`

Unit C4 covered this class in detail. What matters here is **where it sits**:

```python
self.gat = nn.ModuleDict({r: DenseGAT(in_dim, out_dim, heads, dropout) for r in relations})
```

There is **one GAT per relation**, so `W_src`, `W_dst`, `a_src` and `a_dst` are all relation-specific. This is R-GCN's relation-specific weighting, combined with HAN's node-level attention. Two details of `DenseGAT.forward` matter for the semantic level:

```python
att = torch.nan_to_num(att, nan=0.0)                      # nodes w/o neighbours
...
return out, mask.any(1)
```

A node with no neighbours in a relation has an all-$-\infty$ row, so its softmax is NaN and `nan_to_num` zeroes it. Its message is the zero vector, and `mask.any(1)` reports `False` for it. That boolean is the **validity flag** that the view attention uses to mask the relation. For *view* relations this never happens, because `knn_mask` always adds a self-loop. For `assoc` relations it happens for every drug or disease with no visible link. It happens all the time during cold-start practice, and always for the held-out disease in leave-one-disease-out testing.

### 5.3 Semantic-level attention: `model.py::ViewAttention`

```python
class ViewAttention(nn.Module):
    """Node-specific softmax over relation messages (semantic attention, HAN)."""

    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)          # W and b of HAN
        self.q = nn.Linear(att_dim, 1, bias=False)   # the semantic query vector q
```

These are exactly HAN's semantic-attention parameters: $W\in\mathbb{R}^{64\times64}$, $b\in\mathbb{R}^{64}$ and $q\in\mathbb{R}^{64}$. There is one `ViewAttention` per node type (`self.view_att = nn.ModuleDict({t: ViewAttention(out_dim) for t in ("drug", "disease")})`), because drugs and diseases weigh different sets of relations.

```python
    def forward(self, msgs, valid, uniform=False):
        # msgs: R,N,D   valid: R,N (relation has >= 1 neighbour for this node)
        if uniform:
            s = torch.zeros(valid.shape, device=msgs.device)
        else:
            s = self.q(torch.tanh(self.proj(msgs))).squeeze(-1)
```

`msgs` stacks the $R$ relation messages of all $N$ nodes of one type. `self.proj(msgs)` applies $W m+b$ to every (relation, node) pair at once, broadcasting over the first two dimensions. `tanh`, then `self.q`, then `squeeze(-1)` gives the score matrix $s\in\mathbb{R}^{R\times N}$, with entries $s_i^r=q^\top\tanh(Wm_i^r+b)$. **There is no `.mean(1)`.** That single missing line is the difference between HAN's global attention and this project's node-specific attention.

With `uniform=True` (the ablation "w/o view attention") all scores are zero, so after masking the softmax gives **equal weight to every valid relation**. That is a masked mean, not a plain mean.

```python
        s = s.masked_fill(~valid, float("-inf"))
        beta = torch.nan_to_num(torch.softmax(s, dim=0), nan=0.0)  # R,N
        return (beta[..., None] * msgs).sum(0), beta
```

Invalid relations get score $-\infty$, hence weight exactly 0. The softmax runs over `dim=0`, **the relation axis**: each node's weights sum to one across relations. (In `DenseGAT` the softmax runs over neighbours, `dim=1`. Mixing up the two axes is the classic bug.) If *all* of a node's relations were invalid, the softmax would be NaN and `nan_to_num` would turn it into zero weights. With self-looped views that cannot happen. Finally `(beta[..., None] * msgs).sum(0)` is $z_i=\sum_r\beta_i^r m_i^r$, and `beta` is returned for interpretation.

### 5.4 One heterogeneous layer: `model.py::HeteroLayer.forward`

```python
def forward(self, h, graphs, uniform=False, drop_rel=()):
    msgs = {"drug": [], "disease": []}
    valid = {"drug": [], "disease": []}
    names = {"drug": [], "disease": []}
    for rel, (dst, src) in self.relations.items():
        out, has = self.gat[rel](h[dst], h[src], graphs[rel])    # LEVEL 1: node-level attention
        if rel in drop_rel:                      # occlusion for explanations
            has = torch.zeros_like(has)
        msgs[dst].append(out)
        valid[dst].append(has)
        names[dst].append(rel)
```

The loop visits every relation, runs that relation's GAT from source-type states to destination-type states, and files the message under the **destination type**. After the loop, each drug has 4 messages (3 views + `assoc>drug`) and each disease has 4, or 5 each with the bridge. `drop_rel` implements **occlusion**: marking a relation invalid removes it from the view attention, and its weight is redistributed over the remaining relations. `MVHGATMethod.occlusion` uses this to measure how much a prediction depends on one evidence source (Unit E2).

```python
    new, betas = {}, {}
    for t in ("drug", "disease"):
        z, beta = self.view_att[t](torch.stack(msgs[t]), torch.stack(valid[t]), uniform)  # LEVEL 2
        new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
        betas[t] = (names[t], beta)
    return new, betas
```

For each node type: stack the messages into `R x N x D`, apply semantic (view) attention, add the **skip connection** `self.skip[t](h[t])` (R-GCN's $W_0h_i$; it keeps the node's own information, which matters most when the relation messages are weak), then LayerNorm, ELU and dropout. `betas` returns the relation names together with $\beta$, so `MVHGATMethod.view_attention` can report them.

### 5.5 The full encoder: `model.py::MVHGAT.encode` and the explanation helpers

```python
def encode(self, X, graphs, drop_rel=()):
    h = {t: self.drop(F.elu(self.inp[t](X[t]))) for t in ("drug", "disease")}   # type-specific M_phi
    outs = {t: [h[t]] for t in h}
    all_betas = []
    for layer in self.layers:                                                  # 2 HeteroLayers
        h, betas = layer(h, graphs, self.uniform, drop_rel)
        all_betas.append(betas)
        for t in h:
            outs[t].append(h[t])
    return torch.cat(outs["drug"], 1), torch.cat(outs["disease"], 1), all_betas  # jumping knowledge
```

`self.inp` is HAN's type-specific projection. Two layers are stacked, so information travels two hops. For example, disease → drug → disease through `assoc`: "diseases treated by the drugs that treat me". Note that **stacking relation layers composes relations**, which is how a 2-layer model can follow length-2 meta-paths it was never given explicitly (HGT's insight).

`MVHGATMethod.view_attention` re-encodes the graph and averages $\beta$ over the two layers:

```python
B = torch.stack([b[tp][1] for b in betas]).mean(0).T.cpu().numpy()
```

This yields an `n_nodes x R` table, e.g. "disease X: 45% `gene_d`, 30% `pheno_mim`, ...". Remember Unit E2: $\beta$ describes what the encoder *mixed*, not what *caused* a prediction. For a causal reading, use the additive propagation-head shares (`view_weights`) and `occlusion`.

### 5.6 Reading the ablation results

All ablation variants use the **same five folds** (Fdataset, 5-fold CV, one repeat), so differences can be compared **fold by fold** (a paired comparison). Below, $\Delta$ is the mean of the per-fold AUPR differences (variant minus full), and "worse in" counts the folds where the variant was worse. Std is over the 5 folds.

| Variant | AUC | AUPR | $\Delta$AUPR (paired) | worse in | What it tells us about heterogeneity |
|---|---|---|---|---|---|
| **Full MV-HGAT** | 0.943 ± 0.007 | 0.500 ± 0.029 | — | — | reference |
| w/o view attention (masked mean) | 0.939 ± 0.012 | 0.478 ± 0.043 | −0.023 | 4/5 | a small, fairly consistent gain from learned per-node weights |
| w/o gene views | 0.946 ± 0.005 | 0.468 ± 0.024 | −0.033 | 3/5 | gene views help the *top* of the ranking (AUPR) though not the overall ordering (AUC +0.002); mixed evidence |
| + gene bridge | 0.941 ± 0.004 | 0.476 ± 0.022 | −0.024 | 3/5 | adding a chance-level relation **hurts** a little: noise in every message |
| w/o ECFP | 0.939 ± 0.007 | 0.463 ± 0.022 | −0.038 | 4/5 | the largest single-view loss, matching ECFP's standalone strength (AUPR 0.091 vs CDK 0.061) |
| w/o MONDO semantic | 0.941 ± 0.001 | 0.510 ± 0.013 | +0.009 | 2/5 | no benefit in *warm* start; MONDO's value showed up in *cold* start (0.149 → 0.175 together with phenotype) |
| benchmark similarities only | 0.943 ± 0.007 | 0.481 ± 0.018 | −0.019 | 4/5 | the four extra views add about 2 AUPR points in warm start |

How to read this honestly:

1. **Effect sizes are small relative to fold-to-fold spread.** The full model's per-fold AUPR ranges from 0.466 to 0.543. A paired difference of 0.02 that is negative in 4 of 5 folds is suggestive, not conclusive. With 5 paired folds, even "5/5 worse" has a sign-test p-value of only $2\cdot0.5^5=0.0625$ (two-sided). More repeats (the script supports `--repeats`) would sharpen these estimates.
2. **AUC barely moves** (all within ±0.01) while AUPR moves by 0.02–0.04. Views and fusion mainly affect *which* candidates reach the very top of each drug's list, and that is exactly what AUPR measures and AUC hides (Unit B2).
3. **"Benchmark similarities only" is the key control.** It reaches AUPR 0.481, close to the full model's 0.500, and SCMFDD (which uses the same two benchmark similarities) scores 0.495 in the main table. In warm start, therefore, the extra views and the heterogeneous fusion add a modest amount, and the AUPR advantage over the strongest matrix-factorisation baseline is essentially nil (main results: 0.488 vs 0.495), while the AUC advantage is large (0.939 vs 0.893). Say this plainly in the paper. The views' real value is clearer in cold start, where similarity is all the model has (Unit C6).
4. **Heterogeneity-aware fusion matters more than the number of views.** The bridge and, partly, the gene views show that *more* relations do not mean *better*. Per-relation normalisation and per-node masking keep noisy relations from swamping good ones, but they do not make a useless relation useful.

### 5.7 Map: HAN paper → project code

| HAN concept | Equation | Project code |
|---|---|---|
| type-specific projection $M_\phi$ | $h'_i=M_{\phi(i)}h_i$ | `MVHGAT.inp[t]` |
| meta-path neighbours $\mathcal{N}^\Phi(i)$ | $\{j:(M_\Phi)_{ij}>0\}$ + self | `knn_mask(...)` per view; `A` / `A.T` for assoc; `topk_bipartite` for bridge |
| node-level attention $\alpha^\Phi_{ij}$ | GAT within $\Phi$ | `DenseGAT.forward`, one per relation (`HeteroLayer.gat[rel]`) |
| multi-head concat | $\Vert_k$ | `.reshape(h_dst.shape[0], -1)` in `DenseGAT` |
| semantic scorer $q^\top\tanh(Wz+b)$ | $s_i^\Phi$ | `ViewAttention.proj`, `ViewAttention.q` |
| average over nodes | $w_\Phi=\frac1{\lvert V\rvert}\sum_i s_i^\Phi$ | **absent**: node-specific by design |
| softmax over meta-paths | $\beta$ | `torch.softmax(s, dim=0)` after `masked_fill(~valid, -inf)` |
| fused embedding | $z_i=\sum\beta z_i^\Phi$ | `(beta[..., None] * msgs).sum(0)` |
| (not in HAN) skip, LayerNorm, JK | — | `HeteroLayer.skip`, `.norm`; `torch.cat(outs, 1)` in `encode` |

---

## 6. Common mistakes and misconceptions

1. **"Heterogeneous just means the nodes have different features."** No: it means nodes *and edges* carry types that the model uses. Two node types with one shared weight matrix is still a homogeneous model with padded features.
2. **Forgetting inverse relations.** If only `assoc>disease` exists, diseases hear from drugs but drugs never hear from diseases. R-GCN and the project both add the inverse explicitly.
3. **Confusing the two softmax axes.** Node-level attention normalises over *neighbours* (`dim=1` in `DenseGAT`, a `dst x src` matrix). Semantic attention normalises over *relations* (`dim=0` in `ViewAttention`, an `R x N` matrix). Swapping them gives a model that trains but means nothing.
4. **Reading $\beta$ as importance or causation.** $\beta$ is a mixing weight inside the encoder. A relation can have high $\beta$ but carry little predictive information, for example when its messages are near-duplicates of another relation's. Softmax also splits weight between redundant views arbitrarily. Use occlusion or the additive propagation head for faithful attributions (Unit E2).
5. **Thinking a mask is the same as a zero message.** Without the mask, an empty relation contributes a zero vector *and still takes weight*, diluting the others (worked example 3: $d_2$ kept only 37% of its signal under global attention). The mask removes it from the softmax.
6. **Assuming self-looped views are "valid" means "informative".** A drug without SMILES has a chemical view consisting of a self-loop only. It is *valid*, so it is not masked, and its "message" is just a transformed copy of its own features. The attention must *learn* to discount it.
7. **Treating meta-path similarity as neutral.** Raw path counts favour hubs. Normalise (Jaccard, PathSim, cosine) and sparsify (top-$k$), as the project does.
8. **Expecting every extra view to help.** See section 2.9 and the bridge ablation: a chance-level relation costs AUPR.
9. **Believing a specialised heterogeneous architecture is automatically better.** Simple-HGN showed that well-tuned GAT matches many HGNNs. Always include strong simple baselines on identical splits.
10. **Over-reading a single-repeat ablation.** Differences of 0.01–0.02 AUPR are within fold-to-fold noise. Report paired, per-fold differences and use more repeats before drawing conclusions.
11. **Leaking labels through a view.** A "similarity" computed from the association matrix, or from data partly derived from it (CTD inferred links), makes test links visible through a back door.

---

## 7. Exercises

Exercises are graded: **[C]** conceptual, **[M]** mathematical, **[P]** coding/project. Try each one before opening the solution.

**Exercise 1 [C].** Draw the network schema of the project *with* gene nodes, listing every relation and its inverse. Then list the relations that actually exist in the default MV-HGAT graph and say how each gene relation was folded away.

<details><summary>Solution</summary>

With genes: node types {drug, disease, gene}. Relations: drug–drug *chem_cdk* and *chem_ecfp* (symmetric, so each is its own inverse); disease–disease *pheno_mim* and *sem_mondo* (symmetric); drug→disease *treats* with inverse disease→drug *treated-by*; drug→gene *interacts-with* (CTD chemical–gene) with inverse gene→drug; disease→gene *associated-with* (CTD curated) with inverse gene→disease.

Default MV-HGAT graph (2 node types, 8 relations): `view:chem_cdk`, `view:chem_ecfp`, `view:gene_r` (drug←drug); `view:pheno_mim`, `view:sem_mondo`, `view:gene_d` (disease←disease); `assoc>drug` (drug←disease) and `assoc>disease` (disease←drug).

Folding: drug–gene–drug becomes `gene_r` (Jaccard of gene sets, then 10-NN); disease–gene–disease becomes `gene_d` (Jaccard, 10-NN); drug–gene–disease becomes `gene_bridge>drug` / `gene_bridge>disease` (cosine of gene profiles, top-10 per row; off by default, so 10 relations when on).
</details>

**Exercise 2 [M].** Drugs $d_1,d_2,d_3$ have gene sets $\{g_1\}$, $\{g_1,g_2\}$, $\{g_2,g_3\}$. (a) Write $R$ and compute $M_{DGD}$. (b) Compute PathSim and Jaccard for all three pairs. (c) Add a "hub" drug $d_4$ that hits $g_1,g_2,g_3$. What are its raw DGD counts with $d_1,d_2,d_3$, and its Jaccard similarities? What does this show?

<details><summary>Solution</summary>

(a) $R=\begin{bmatrix}1&0&0\\1&1&0\\0&1&1\end{bmatrix}$, $M=RR^\top=\begin{bmatrix}1&1&0\\1&2&1\\0&1&2\end{bmatrix}$.

(b) PathSim $=2M_{ij}/(M_{ii}+M_{jj})$: $(d_1,d_2)=2/3=0.667$; $(d_2,d_3)=2/4=0.5$; $(d_1,d_3)=0$. Jaccard $=M_{ij}/(M_{ii}+M_{jj}-M_{ij})$: $(d_1,d_2)=1/2$; $(d_2,d_3)=1/3$; $(d_1,d_3)=0$.

(c) $d_4$'s counts are $|G_4\cap G_j|$ = 1, 2, 2. These are the *largest* counts in the table: the hub "shares the most" with everyone. Jaccard: $1/3$, $2/3$, $2/3$. Still high, but now bounded by 1 and penalised for $d_4$'s large union with $d_1$. Raw counts reward a node simply for touching many genes (the literature bias of well-studied drugs in CTD); normalisation reduces, but does not remove, that bias. Top-$k$ sparsification then limits how many nodes the hub can connect to.
</details>

**Exercise 3 [M].** An R-GCN layer has $|\mathcal{R}|=20$ relations and $d_\text{in}=d_\text{out}=128$. Count the relation parameters (excluding $W_0$) for (a) full weights, (b) basis decomposition with $B=4$, (c) block-diagonal with $B=4$. (d) For what $B$ does the basis model have as many parameters as the full model?

<details><summary>Solution</summary>

$128^2=16{,}384$. (a) $20\times16{,}384=327{,}680$. (b) $4\times16{,}384+20\times4=65{,}616$. (c) $20\times16{,}384/4=81{,}920$. (d) $B\,d^2+|\mathcal{R}|B=|\mathcal{R}|d^2\Rightarrow B=|\mathcal{R}|d^2/(d^2+|\mathcal{R}|)=20\cdot16384/16404\approx19.98$. So at $B=|\mathcal{R}|=20$ the basis model has *slightly more* parameters (it pays for the coefficients) and can represent any set of $W_r$ (see Exercise 4).
</details>

**Exercise 4 [M].** Show that (a) with $B=|\mathcal{R}|$ the basis decomposition can represent *any* set of relation matrices, and (b) with $B=1$ all relation matrices are scalar multiples of one matrix. What does (b) mean for a model with a *treats* relation and a *similar-to* relation?

<details><summary>Solution</summary>

(a) Given arbitrary $W_1,\dots,W_R$, set $V_b=W_b$ and $a_{rb}=\delta_{rb}$. Then $\sum_b a_{rb}V_b=W_r$.

(b) $W_r=a_{r1}V_1$. Every relation applies the *same* transformation, up to a scale factor and possibly a sign. For *treats* and *similar-to* the model can only say "listen to treated diseases $a$ times as loudly as to similar drugs". It cannot route them into different feature directions. It is barely more expressive than a homogeneous GCN with per-relation normalisation. Small $B$ is a strong regulariser; choose it on validation data.
</details>

**Exercise 5 [M].** Two meta-paths, two nodes. The per-node scores are $s^{\Phi_1}=(1.0,\;0.0)$ and $s^{\Phi_2}=(0.0,\;2.0)$ for nodes $(1,2)$. Compute (a) HAN's global $\beta$ and (b) node-specific $\beta$ for each node.

<details><summary>Solution</summary>

(a) $w_{\Phi_1}=0.5$, $w_{\Phi_2}=1.0$. $\beta=\big(e^{0.5},e^{1}\big)/(e^{0.5}+e^{1})=(1.6487,2.7183)/4.3670=(0.3775,0.6225)$ for both nodes.

(b) Node 1: $(e^1,e^0)/(e+1)=(0.7311,0.2689)$. Node 2: $(e^0,e^2)/(1+e^2)=(0.1192,0.8808)$.

Global attention gives node 1 only 38% of its preferred meta-path; node-specific gives 73%. (These values were checked numerically.)
</details>

**Exercise 6 [M].** (a) Derive $\partial\beta_\Phi/\partial w_\Psi$ for the softmax. (b) Evaluate the Jacobian at $\beta=(0.3726,0.6274)$ from worked example 3. (c) Explain why a view whose $\beta$ has collapsed to $\approx0$ early in training tends to stay there.

<details><summary>Solution</summary>

(a) $\partial\beta_\Phi/\partial w_\Psi=\beta_\Phi(\delta_{\Phi\Psi}-\beta_\Psi)$ (section 2.5).

(b) $\partial\beta_1/\partial w_1=0.3726\times0.6274=0.2338$; $\partial\beta_1/\partial w_2=-0.3726\times0.6274=-0.2338$; symmetrically $\partial\beta_2/\partial w_2=0.2338$ and $\partial\beta_2/\partial w_1=-0.2338$. Each column sums to zero.

(c) The gradient of the loss with respect to that view's score is $\sum_\Phi(\partial L/\partial\beta_\Phi)\,\beta_\Phi(\delta_{\Phi\Psi}-\beta_\Psi)$. Every term carries a factor $\beta_\Psi$ or $\beta_\Phi(\ldots)$ that is tiny when $\beta_\Psi\approx0$. In addition, the view's message receives gradient only through $\beta_\Psi\,m^\Psi$, so the message encoder of that view also stops learning. Both effects freeze the view out: a "rich get richer" dynamic. Remedies: attention dropout, temperature, entropy regularisation, or a uniform warm-up.
</details>

**Exercise 7 [M].** Prove that the fused embedding $z_i=\sum_\Phi\beta_\Phi z_i^\Phi$ satisfies $\lVert z_i\rVert\le\max_\Phi\lVert z_i^\Phi\rVert$. Give a two-view example in which concatenation followed by a linear map can produce a larger output than any single view, and explain why this matters for "agreement" between views.

<details><summary>Solution</summary>

By the triangle inequality and $\beta_\Phi\ge0$, $\sum\beta_\Phi=1$: $\lVert\sum\beta_\Phi z^\Phi\rVert\le\sum\beta_\Phi\lVert z^\Phi\rVert\le\max_\Phi\lVert z^\Phi\rVert$.

Example: $z^1=z^2=[1]$. Attention gives $[1]$ whatever $\beta$ is. Concatenation $[1,1]$ followed by $U=[1,1]$ gives $[2]$. Concatenation (and gating, whose weights need not sum to 1) can turn *agreement* between views into a *stronger* signal, as in evidence accumulation. Attention can only interpolate: two moderately confident, agreeing views stay moderately confident. Later layers can rescale, so this is a bias rather than a hard limit, but it is one reason concatenation is a strong baseline.
</details>

**Exercise 8 [P].** Count the trainable parameters of one project `HeteroLayer` with the default 8 relations, `in_dim = out_dim = 64`, 4 heads. Break the count down by component, then check it in code.

<details><summary>Solution</summary>

- One `DenseGAT`: `W_src` and `W_dst` are $64\times64$ without bias ($2\times4096=8192$); `a_src` and `a_dst` are $4\times16$ each (128). Total **8,320**. Eight relations: **66,560**.
- `view_att` (2 node types): `proj` $64\times64+64=4160$, `q` 64, so 4,224 each, **8,448** total.
- `skip` (2 types): $64\times64+64=4160$ each, **8,320**.
- `norm` (2 LayerNorms): $2\times64$ each, **256**.

Total **83,584**. Check:

```python
import sys; sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import HeteroLayer, DenseGAT
rels = {f"view:{v}": ("drug", "drug") for v in "abc"}
rels.update({f"view:{v}": ("disease", "disease") for v in "xyz"})
rels.update({"assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")})
L = HeteroLayer(rels, 64, 64, 4, 0.2)
cnt = lambda m: sum(p.numel() for p in m.parameters())
print(cnt(DenseGAT(64, 64, 4, 0.2)), cnt(L), cnt(L.view_att), cnt(L.skip), cnt(L.norm))
# 8320 83584 8448 8320 256
```

Note that 80% of the layer is relation-specific GAT weights: each extra view adds 8,320 parameters per layer, which is the "more parameters, same data" cost of section 2.9.
</details>

**Exercise 9 [P].** A biologic drug has no SMILES. Trace exactly what happens to it in `chem_ecfp` through `fill_missing`, `knn_mask`, `DenseGAT` and `ViewAttention`. Is the relation masked? What does the attention have to learn?

<details><summary>Solution</summary>

1. Its `chem_ecfp` row and column are NaN. `fill_missing` sets them to 0 and the diagonal to 1.
2. In `knn_mask`, `work` has $-\infty$ on the diagonal and 0 elsewhere in its row. `argpartition` picks 10 indices, but `keep = vals > 0` discards them all. Other drugs never pick it either, since their similarity to it is 0. Symmetrisation adds nothing, and `fill_diagonal(M, True)` adds the **self-loop**. The row has exactly one `True`.
3. In `DenseGAT`, softmax over a single neighbour gives $\alpha_{ii}=1$, so the message is $W^{\text{ecfp}}_\text{src}h_i$, a transformed copy of its *own* state. `mask.any(1)` is `True`.
4. In `ViewAttention` the relation is **valid**, so it is **not masked**. It competes in the softmax with a self-message.

The attention must learn to give low scores to "self-only" messages. It can, because such messages have a recognisable signature (they correlate with the skip path), but nothing guarantees it. A possible improvement is to mark views with no non-self neighbour as invalid (`mask` minus diagonal `.any(1)`), so that coverage gaps are handled by masking rather than by learning. That would be a good, small ablation.
</details>

**Exercise 10 [P].** The "opposite" scenario of section 4.4 defeats message-only attention. Implement **context-aware** view attention, $s_i^\Phi=q^\top\tanh(P[m_i^\Phi\Vert h_i]+p)$ with $h_i$ an embedding of the node's own features, and test it on both scenarios. Then append a "metadata" feature (the node's group) to the features and test again. Explain the results.

<details><summary>Solution</summary>

Save as `ex_context.py` next to `fusion.py`:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import make_views, NodeLevelGAT

class ContextHAN(nn.Module):
    """View attention whose score also sees the node's OWN state: s = q^T tanh(P [m || h_self])."""
    def __init__(self, n_views, d_in=16, d=32, d_att=32):
        super().__init__()
        self.gats = nn.ModuleList(NodeLevelGAT(d_in, d) for _ in range(n_views))
        self.self_emb = nn.Linear(d_in, d)
        self.proj = nn.Linear(2 * d, d_att)
        self.q = nn.Linear(d_att, 1, bias=False)
        self.out = nn.Linear(d, 2)
    def forward(self, X, views):
        M = torch.stack([g(X, v) for g, v in zip(self.gats, views)])        # (P, N, d)
        h = F.elu(self.self_emb(X)).expand(M.shape[0], -1, -1)              # (P, N, d)
        s = self.q(torch.tanh(self.proj(torch.cat([M, h], -1)))).squeeze(-1)
        beta = torch.softmax(s, 0)
        return self.out(F.elu((beta[..., None] * M).sum(0))), beta

def run(scenario, seed, epochs=200):
    X, y, views, group = make_views(scenario=scenario, seed=seed)
    torch.manual_seed(seed)
    train = torch.rand(len(y)) < 0.5
    model = ContextHAN(len(views))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        logits, _ = model(X, views)
        loss = F.cross_entropy(logits[train], y[train])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits, beta = model(X, views)
    return (logits.argmax(1) == y)[~train].float().mean().item(), beta, group

if __name__ == "__main__":
    for scenario in ("noisy", "opposite"):
        out = [run(scenario, s) for s in range(3)]
        accs = [o[0] for o in out]
        b, g = out[0][1], out[0][2]
        print(f"{scenario:8s} context attention acc = {np.mean(accs):.3f} (seeds "
              + " ".join(f"{a:.3f}" for a in accs) + f") | seed-0 beta_A: group0 "
              f"{b[0, g == 0].mean():.2f}, group1 {b[0, g == 1].mean():.2f}")
```

Output:

```text
noisy    context attention acc = 0.831 (seeds 0.819 0.814 0.861) | seed-0 beta_A: group0 0.76, group1 0.43
opposite context attention acc = 0.623 (seeds 0.613 0.618 0.639) | seed-0 beta_A: group0 0.46, group1 0.44
```

With a metadata column (save as `ex_context2.py`):

```python
import numpy as np
import torch
import torch.nn.functional as F
import fusion
from fusion import make_views, TinyHAN
from ex_context import ContextHAN

def run(model_fn, seed, epochs=200):
    X, y, views, group = make_views(scenario="opposite", seed=seed)
    X = torch.cat([X, torch.tensor(group, dtype=torch.float32)[:, None]], 1)  # metadata column
    torch.manual_seed(seed)
    train = torch.rand(len(y)) < 0.5
    model = model_fn()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        logits, _ = model(X, views)
        loss = F.cross_entropy(logits[train], y[train])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits, beta = model(X, views)
    return (logits.argmax(1) == y)[~train].float().mean().item(), beta, group

for name, fn in [("node-specific (HAN-style)", lambda: TinyHAN(2, "node", d_in=17)),
                 ("context-aware", lambda: ContextHAN(2, d_in=17))]:
    out = [run(fn, s) for s in range(3)]
    b, g = out[0][1], out[0][2]
    print(f"{name:26s} acc = {np.mean([o[0] for o in out]):.3f} | seed-0 beta_A: "
          f"group0 {b[0, g == 0].mean():.2f}, group1 {b[0, g == 1].mean():.2f}")
```

Output:

```text
node-specific (HAN-style)  acc = 0.583 | seed-0 beta_A: group0 0.58, group1 0.48
context-aware              acc = 0.632 | seed-0 beta_A: group0 0.52, group1 0.34
```

*Interpretation.* On the noisy scenario context attention separates the groups even more clearly ($\beta_A$ 0.76 vs 0.43) with similar accuracy (0.831). On the opposite scenario it improves accuracy a little (0.623 vs 0.564 for message-only node attention), but $\beta$ does not separate: the node's own features are too weak a clue to tell which view is lying. When we add a metadata column that *does* reveal the group, context attention starts to separate (0.52 vs 0.34) and reaches 0.632, while message-only attention cannot use the clue (0.583). Both remain far from the ~0.85 a perfect per-node selection would allow, because attention is hard to learn from 200 labelled nodes.

Lesson for the project: if you know *why* a view is unreliable for some entities (no SMILES, few CTD genes, old OMIM entry), give that information to the attention explicitly, through masks or metadata features, rather than hoping it is inferred from messages.
</details>

**Exercise 11 [C].** In which situations would HAN's **global** semantic attention be *preferable* to node-specific attention? Give two reasons and one project-relevant example.

<details><summary>Solution</summary>

(1) **Variance:** with few labels per node, the averaged score is a far more stable estimate; node-specific weights can overfit to noise. (2) **Interpretability at dataset level:** one ranking of meta-paths ("PSP matters more than PAP") is easy to report and compare across datasets. (3) Global weights also need no mask, as long as every node has neighbours in every meta-path.

Project example: when comparing Fdataset and Cdataset, a single global weight per view, such as the propagation head's $w_v$ (`view_weights`), is the natural summary. Node-specific $\beta$ is for per-entity case studies. The project in fact provides both: global late-fusion weights $w_v$ and node-specific $\beta$.
</details>

**Exercise 12 [C].** Using worked example 3, compute the fused embedding of each drug under the project's `uniform=True` setting (masked mean), with $d_2$'s DSD view invalid.

<details><summary>Solution</summary>

$d_1$: both valid, so $\tfrac12([1,0]+[2,1])=[1.5,0.5]$. $d_2$: only DGD is valid, so $[0.5,0.5]$. $d_3$: both valid, so $\tfrac12([0,0]+[1,1])=[0.5,0.5]$. The masked mean already fixes $d_2$'s dilution problem. Learned attention adds the ability to prefer one *valid* view over another, which is the −0.023 AUPR gap in the ablation.
</details>

**Exercise 13 [C].** A colleague reads the ablation table and claims: (a) "The MONDO view is useless and should be removed." (b) "The gene bridge is harmful, so drug–gene–disease biology is irrelevant to repositioning." (c) "View attention improves the model by 0.023 AUPR." Evaluate each claim.

<details><summary>Solution</summary>

(a) Not supported. In warm-start CV, removing MONDO changed AUPR by +0.009 (worse in only 2/5 folds), i.e. no detectable effect. But in cold start MONDO was complementary to phenotype similarity (0.149 → 0.175 standalone). A warm-start ablation cannot assess a cold-start view; run the ablation with the LODO protocol before deciding.

(b) Not supported. The bridge *as constructed* (cosine of CTD gene profiles, top-10) carries chance-level signal on these benchmarks (AUPR 0.011, positive rate 0.010). Likely reasons: curation bias, half the diseases lack curated genes, and CTD "interactions" are mostly expression changes rather than targets (`HOW_IT_WORKS.md` §10). That is a statement about this data source and construction, not about biology.

(c) Roughly right, with caveats: the paired mean difference is −0.023 with the variant worse in 4/5 folds. That is a single repeat, and the fold-to-fold spread is about 0.03–0.04. State it as "a small, fairly consistent gain (≈0.02 AUPR, 4/5 folds), single repeat", and ideally confirm it with more repeats.
</details>

**Exercise 14 [M].** In a homogeneous mean aggregator over the union graph, a drug has 45 similarity neighbours (over 3 views) and 3 indication neighbours. (a) What fraction of the aggregated message comes from indications? (b) Under R-GCN-style per-relation means with equal relation weights over 4 relations (3 views + assoc), what is that fraction? (c) Why does (b) still not guarantee the model *uses* the indication signal well?

<details><summary>Solution</summary>

(a) $3/48=6.25\%$. (b) Each relation is averaged first, then the 4 relation means are combined equally, so 25%. (c) The fraction is a *share of the input*, not of the useful signal. The model still needs suitable $W_r$ (or attention) to turn the indication message into features that the decoder can use. Moreover, during training the visible indication edges must not trivially reveal the supervised links (edge leakage, Unit C6).
</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (C5)

### Q1. Where are the two levels of attention in `HeteroLayer.forward`?

**Level 1, node-level attention**, is inside the loop over relations:

```python
for rel, (dst, src) in self.relations.items():
    out, has = self.gat[rel](h[dst], h[src], graphs[rel])
```

`self.gat[rel]` is the `DenseGAT` belonging to relation `rel`. Inside it, every destination node $i$ scores every source node $j$ with $e_{ij}=\mathrm{LeakyReLU}(a_\text{dst}^\top W_\text{dst}h_i+a_\text{src}^\top W_\text{src}h_j)$ per head, masks non-neighbours with $-\infty$, and normalises **over neighbours** (`torch.softmax(e, dim=1)` on a `dst x src x heads` tensor). The weighted sum gives one message per node per relation (`out`), plus a validity flag (`has`). This is HAN's node-level attention within one "meta-path", with relation-specific weights as in R-GCN.

**Level 2, semantic (view) attention**, comes after the loop:

```python
z, beta = self.view_att[t](torch.stack(msgs[t]), torch.stack(valid[t]), uniform)
```

For each node type, the relation messages are stacked into `R x N x D`. `ViewAttention` scores each $(r,i)$ with $q^\top\tanh(Wm_i^r+b)$, masks invalid relations, and normalises **over relations** (`torch.softmax(s, dim=0)`). The weighted sum $z_i=\sum_r\beta_i^r m_i^r$ is then combined with the skip connection, normalised and activated. So level 1 decides *which neighbours* within a relation matter; level 2 decides *which relations* matter, for each node separately.

### Q2. Our `ViewAttention` is node-specific, unlike the original HAN. What does that buy us?

HAN averages the semantic scores over all nodes and therefore learns **one weight per meta-path for the whole graph**. The project omits that average, so **every drug and every disease gets its own weights**, at no extra parameter cost. This buys three things.

1. **Robustness to uneven coverage.** Entities differ in which views they have: biologics lack chemical structure, about half the diseases lack curated genes, and drugs or diseases with no visible link lack the `assoc` relation (always so for a cold-start disease). Node-specific weights plus the validity mask let each node redistribute weight over the relations it actually has. A global weight would dilute those nodes' real evidence with empty views (worked example 3: 37% vs 100% of the real signal).
2. **Robustness to uneven quality.** Even when present, a view's reliability varies by entity (rich vs sparse CTD gene sets; close vs distant chemical analogues). A per-node score can down-weight a view that is noisy *for this node* while still using it for others (section 4.4: $\beta_A$ 0.65 vs 0.38 by group).
3. **Per-entity interpretability.** `MVHGATMethod.view_attention` returns an `n x R` table, so a case study can say which relations the encoder leaned on for a particular disease. (Remember: attention describes mixing, not causation. Pair it with occlusion and the additive propagation-head shares.)

The costs: per-node weights are noisier (estimated from one node's messages), can overfit on small data, and are only as smart as the scorer's input. A message-only scorer cannot detect a view that is *confidently misleading* for some nodes (section 4.4, "opposite" scenario). In the ablation, replacing learned attention by a masked mean cost about 0.023 AUPR (4/5 folds, one repeat): a real but modest benefit.

---

## 9. Summary and cheat sheet

**Big picture.** Drugs, diseases and genes are different *types*, and "similar chemically", "shares genes" and "treats" are different *relations*. Heterogeneous GNNs (1) normalise and transform per relation, and (2) learn how to weigh relations. MV-HGAT is a HAN-style model with relation-specific GATs, node-specific masked view attention, a skip connection and jumping knowledge, and it folds gene nodes into meta-path relations.

| Concept | One-line formula / fact |
|---|---|
| HIN | $\phi:\mathcal{V}\to\mathcal{T}$, $\psi:\mathcal{E}\to\mathcal{R}$, $\lvert\mathcal{T}\rvert+\lvert\mathcal{R}\rvert>2$ |
| network schema | type-level graph; every edge is an instance of a schema relation |
| meta-path commuting matrix | $M_\Phi=A_{r_1}\cdots A_{r_\ell}$; $(M_\Phi)_{ij}$ = number of path instances |
| PathSim | $2M_{ij}/(M_{ii}+M_{jj})$ (symmetric meta-paths) |
| Jaccard via counts | $M_{ij}/(M_{ii}+M_{jj}-M_{ij})$ |
| R-GCN | $h_i'=\sigma(W_0h_i+\sum_r\sum_{j\in\mathcal{N}^r(i)}c_{i,r}^{-1}W_rh_j)$ |
| basis decomposition | $W_r=\sum_b a_{rb}V_b$; params $Bd^2+\lvert\mathcal{R}\rvert B$ |
| block-diagonal | $W_r=\mathrm{blockdiag}(Q_{1r},\dots,Q_{Br})$; params $\lvert\mathcal{R}\rvert d^2/B$ |
| HAN node level | $\alpha^\Phi_{ij}=\mathrm{softmax}_{j\in\mathcal{N}^\Phi(i)}\mathrm{LeakyReLU}(a_\Phi^\top[h_i'\Vert h_j'])$ |
| HAN semantic level | $w_\Phi=\frac1{\lvert V\rvert}\sum_i q^\top\tanh(Wz_i^\Phi+b)$, $\beta=\mathrm{softmax}(w)$ |
| node-specific (project) | $\beta_i^\Phi=\mathrm{softmax}_\Phi\big(q^\top\tanh(Wz_i^\Phi+b)\big)$ with mask |
| softmax Jacobian | $\partial\beta_\Phi/\partial w_\Psi=\beta_\Phi(\delta_{\Phi\Psi}-\beta_\Psi)$ |
| convexity | attention fusion cannot amplify: $\lVert z\rVert\le\max\lVert z^\Phi\rVert$ |
| fusion menu | concat · mean · max · global attention · node attention · gating · late fusion |
| HGT | type-dependent Q/K/V + $W^{ATT}_{\phi(e)}$, $W^{MSG}_{\phi(e)}$, prior $\mu$; no hand-made meta-paths |
| MAGNN | encodes all nodes along a meta-path instance |
| HetGNN | RWR neighbour sampling by type, Bi-LSTM content and neighbour encoders |
| Simple-HGN | GAT + edge-type embeddings + residuals + $L_2$ norm; well-tuned GAT matches most HGNNs |
| project code | `DenseGAT` (level 1, softmax `dim=1`) → `ViewAttention` (level 2, softmax `dim=0`, mask) → skip + LayerNorm + ELU |
| default graph | 2 node types, 8 relations (+2 with bridge); gene nodes folded into `gene_r`, `gene_d`, `gene_bridge` |
| ablation take-home | attention ≈ +0.02 AUPR; ECFP the most valuable extra view; the bridge hurts; AUC barely moves |

---

## 10. Further resources (all links checked on 2026-10-01)

**Courses and books**
- [Stanford CS224W, *Machine Learning with Graphs*: course home](https://web.stanford.edu/class/cs224w/) (free). Slides, colabs and reading lists.
- [CS224W Fall 2025, Lecture 9 "Heterogeneous graphs" (slides, PDF)](https://snap.stanford.edu/class/cs224w-2025/slides/09-hetero.pdf) (free). R-GCN and HGT, with exactly this unit's framing.
- [CS224W Fall 2021 lecture videos (YouTube playlist)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn) (free). The heterogeneous-graph and R-GCN lectures.
- [W. L. Hamilton, *Graph Representation Learning* (2020), free pre-print](https://www.cs.mcgill.ca/~wlh/grl_book/) (free). Ch. 4 covers multi-relational data; §5.4 covers multi-relational GNNs (R-GCN).

**Primary papers**
- [Wang et al. 2019, "Heterogeneous Graph Attention Network" (WWW), arXiv](https://arxiv.org/abs/1903.07293) (free). HAN; read §4 for the two attention levels.
- [Schlichtkrull et al. 2018, "Modeling Relational Data with Graph Convolutional Networks" (ESWC), arXiv](https://arxiv.org/abs/1703.06103) (free). R-GCN, basis and block decompositions.
- [Hu, Dong, Wang & Sun 2020, "Heterogeneous Graph Transformer" (WWW), arXiv](https://arxiv.org/abs/2003.01332) (free). HGT.
- [Lv et al. 2021, "Are we really making much progress? Revisiting, benchmarking, and refining heterogeneous graph neural networks" (KDD), arXiv](https://arxiv.org/abs/2112.14936) (free). Simple-HGN and the HGB benchmark; essential reading on fair evaluation.
- [Fu et al. 2020, "MAGNN: Metapath Aggregated Graph Neural Network" (WWW), arXiv](https://arxiv.org/abs/2002.01680) (free). Encoding intermediate nodes of meta-paths.
- [Zhang et al. 2019, "Heterogeneous Graph Neural Network" (KDD), DOI](https://doi.org/10.1145/3292500.3330961) (free, open access on ACM DL). HetGNN.
- [Dong, Chawla & Swami 2017, "metapath2vec" (KDD), author PDF](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) (free). Meta-path-guided random-walk embeddings.
- [Sun et al. 2011, "PathSim: Meta Path-Based Top-K Similarity Search in HINs" (PVLDB), PDF](https://www.vldb.org/pvldb/vol4/p992-sun.pdf) (free). Meta-paths and PathSim.
- [Wang, Liu, Han & Shi 2021, "Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning" (KDD), arXiv](https://arxiv.org/abs/2105.09111) (free). HeCo, an optional extension.

**Surveys**
- [Shi, Li, Zhang, Sun & Yu, "A Survey of Heterogeneous Information Network Analysis" (TKDE 2017), arXiv](https://arxiv.org/abs/1511.04854) (free). Definitions of HIN, schema and meta-path.
- [Wang et al., "A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources" (IEEE TBD 2022), arXiv](https://arxiv.org/abs/2011.14867) (free). A broad map of the field.

**Hands-on**
- [PyTorch Geometric: "Heterogeneous Graph Learning" guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html) (free). How a library represents typed graphs (`HeteroData`); useful even though we implement from scratch.
- [DGL tutorial: Relational Graph Convolutional Network](https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/4_rgcn.html) (free). R-GCN step by step.
- [DGL example implementation of HAN (GitHub)](https://github.com/dmlc/dgl/tree/master/examples/pytorch/han) (free). A compact reference implementation.

**Biomedical applications**
- [Zitnik, Agrawal & Leskovec 2018, "Modeling polypharmacy side effects with graph convolutional networks" (Decagon), arXiv](https://arxiv.org/abs/1802.00543) (free). R-GCN-style multi-relational link prediction on a drug–protein graph.
- [Cai et al. 2021, "Drug repositioning based on the heterogeneous information fusion graph convolutional network" (Brief. Bioinform.), DOI](https://doi.org/10.1093/bib/bbab319) (paid). A heterogeneous GCN on the same F/C benchmarks.

---

## 11. Glossary

- **Additive (Bahdanau) attention**: scoring a vector by $q^\top\tanh(Wx+b)$ with a learned query $q$. HAN's semantic attention uses this form.
- **Basis decomposition**: writing every relation matrix as a learned mix of a few shared basis matrices, $W_r=\sum_b a_{rb}V_b$.
- **Block-diagonal decomposition**: constraining each relation matrix to be block-diagonal, so that groups of hidden dimensions do not interact within a relation.
- **Commuting matrix**: the product of adjacency matrices along a meta-path; it counts path instances.
- **Early / intermediate / late fusion**: combining views at the input, inside the encoder, or at the output score.
- **Gating**: weighting each view by an independent sigmoid; weights need not sum to one.
- **Heterogeneous information network (HIN)**: a graph whose nodes and edges carry types, with more than one type in total.
- **HGT**: Heterogeneous Graph Transformer; Transformer-style attention with type-dependent parameters, no hand-designed meta-paths.
- **Inverse relation**: the reverse direction of a directed relation (treats ↔ treated-by), added so messages flow both ways.
- **Jumping knowledge (JK)**: concatenating the outputs of all layers to form the final embedding.
- **Masking (of views)**: setting a relation's score to $-\infty$ for nodes with no neighbours in it, so it receives zero attention weight.
- **Meta-path**: a path of types and relations in the network schema, e.g. drug–gene–drug, describing a composite relation.
- **Meta-path-based neighbours**: nodes reachable from a node by at least one instance of a meta-path.
- **Meta-relation**: in HGT, the triple (source type, edge type, target type).
- **Multi-view learning**: learning from several descriptions (views) of the same objects; guided by consensus and complementarity.
- **Multiplex graph**: one node set with several layers (types) of edges.
- **Network schema**: the type-level blueprint of a HIN (one node per type, one edge per relation).
- **Node-level attention**: attention over the neighbours of a node *within* one relation or meta-path (GAT).
- **Node-specific (view) attention**: semantic attention computed separately for each node, without averaging over nodes (this project).
- **PathSim**: a normalised meta-path similarity, $2M_{ij}/(M_{ii}+M_{jj})$.
- **R-GCN**: Relational GCN; a GCN with one weight matrix per relation plus a self-connection.
- **Relation (edge type)**: a typed set of edges from a source type to a destination type; in this project, a view, the known links, or the gene bridge.
- **Semantic-level attention**: attention *across* meta-paths or relations, producing weights $\beta$.
- **Simple-HGN**: GAT with edge-type embeddings, residuals and output normalisation; a strong, simple heterogeneous baseline.
- **Validity flag**: in the project, `mask.any(1)`, true when a node has at least one neighbour in a relation.
- **View**: one way of measuring similarity between drugs (or diseases), for example `chem_ecfp`; it becomes one relation in the model.
