# Unit C4 — Attention and Graph Attention Networks (GAT)

> **Course:** Drug repositioning with graph neural networks — a self-contained course
> **Chapter:** 12 of 20 · **Track:** C (Graphs) · **Unit code:** C4

| | |
|---|---|
| **Prerequisites** | C3 (message passing, GCN, receptive field, over-smoothing, dense implementations) — essential. B3 (PyTorch modules, autograd, dropout, `nn.Linear`), A2 (dot products, matrix shapes, broadcasting), A3 (probability distributions; softmax as a distribution), C1 (bipartite graphs). |
| **Estimated study time** | 10–12 hours: about 5 h theory and hand calculations, 3 h code, 3–4 h exercises. |
| **Code environment** | The project's `.venv`. All code is dense PyTorch/NumPy, runs on CPU in seconds (one script takes about half a minute), and the printed outputs below come from actual runs. Some scripts import the project's own `DenseGAT` and `ViewAttention` from `src/drepo/model.py` (read-only) to check our from-scratch versions against them. |

## Learning objectives

After this chapter you will be able to:

1. **Explain** attention as a *soft dictionary lookup* with queries, keys and values, and compute "scores → softmax → weighted sum" by hand.
2. **Compare** additive (Bahdanau), multiplicative/dot-product (Luong) and scaled dot-product (Transformer) scoring, and **derive** why scaled dot-product divides by $\sqrt{d}$.
3. **Implement** multi-head attention from scratch and **verify** it against `torch.nn.MultiheadAttention`; **explain** self-attention, masking, and the Transformer block in outline, and why "a Transformer is a GNN on a complete graph".
4. **Write down and compute by hand** a GAT layer (Veličković et al. 2018): $e_{ij} = \text{LeakyReLU}(a^\top[Wh_i \Vert Wh_j])$, masked softmax over neighbours, weighted sum; multi-head concatenation vs averaging; attention dropout.
5. **Prove** that $a^\top[x \Vert y] = a_1^\top x + a_2^\top y$, and **explain** how the split $a = [a_\text{dst}; a_\text{src}]$ turns an $O(N^2F')$ computation into $O(NF') + O(N^2)$.
6. **State and prove** the *static attention* limitation of GAT, **explain** how GATv2 (Brody et al. 2022) fixes it, and **demonstrate** the difference on a task where it matters.
7. **Handle** nodes with no neighbours (masked softmax of an all-$-\infty$ row) correctly, including the gradient, and **explain** what the project does with them.
8. **Analyse** time and memory complexity of dense and sparse attention, and of bipartite ($\text{dst}\neq\text{src}$) attention as used in `DenseGAT`.
9. **Read** `model.py::DenseGAT` line by line and map every line to an equation.
10. **Argue** carefully about what attention weights can and cannot explain.

---

## 1. Motivation: why this matters for *this* project

In Unit C3 a GCN layer combined a drug's neighbours with **fixed** coefficients $1/\sqrt{\tilde d_i \tilde d_j}$, decided entirely by the graph's degrees. Think about what that means in the project's data:

* In the `chem_cdk` kNN graph a drug has between 10 and 94 neighbours (Unit C3, Section 1). Some are chemically similar *and* share its mechanism; others are similar only in some scaffold that is irrelevant for the indication. A GCN treats them alike (up to degree).
* In the `assoc>drug` relation a drug receives messages from the diseases it is known to treat. A drug linked to a very specific disease and to a broad, heterogeneous one probably should not weigh them equally when its embedding is used to predict new indications.
* Similarity views have **holes**: `gene_d` leaves 153 of 313 diseases without any neighbour other than themselves; `gene_r` leaves 81 drugs isolated. A good model must cope when a relation provides no information for a node.

**Attention** lets the model *learn* the coefficients from the nodes' features: each node computes a relevance score for each neighbour, turns the scores into weights with a softmax, and takes the weighted average of the neighbours' messages. The **Graph Attention Network (GAT)** is exactly this, restricted to graph neighbours.

MV-HGAT uses attention at **two levels** (Unit C5 covers the second):

1. **Node-level attention** inside each relation — `model.py::DenseGAT`, "a line-by-line implementation of the GAT equations" (PREREQUISITES.md). With 8 relations and 2 layers, the model contains 16 `DenseGAT` modules, each with 4 heads.
2. **View-level attention** across relations — `ViewAttention`, which decides per node how much to trust each relation's message.

This chapter teaches level 1 completely, plus the general attention machinery both levels share. On Fdataset 5-fold CV the full model reaches AUC 0.939 / AUPR 0.488 against 0.838 / 0.096 for NIMCGCN and 0.833 / 0.133 for LAGCN — the two GCN baselines that use fixed coefficients. (Attention is only one of several differences between these models; Unit C3, Section 12 discusses others. Do not read the gap as "attention alone is worth 0.35 AUPR".)

---

## 2. The attention idea

### 2.1 A soft dictionary lookup

A Python dictionary does a **hard lookup**: given a query, find the key that matches *exactly* and return its value. Attention does a **soft lookup**: compare the query with *every* key, turn the comparisons into weights that sum to 1, and return the weighted average of the values.

* **Query** $q$ — what I am looking for (the node being updated: "drug $i$, with its current features").
* **Keys** $k_1,\dots,k_n$ — what each item advertises about itself, used *only for matching* (each neighbour $j$).
* **Values** $v_1,\dots,v_n$ — what each item hands over if selected (the neighbour's message).

$$
\text{score}_j = s(q, k_j),\qquad
\alpha_j = \frac{\exp(\text{score}_j)}{\sum_{l=1}^n \exp(\text{score}_l)},\qquad
\text{output} = \sum_{j=1}^n \alpha_j\, v_j .
$$

Three steps: **score → softmax → weighted sum.** The output is a *convex combination* of the values ($\alpha_j \ge 0$, $\sum_j\alpha_j = 1$), so it lies "between" them. If one score is much larger than the others, $\alpha$ is nearly one-hot and attention approaches a hard lookup; if all scores are equal it is a plain mean. Everything is differentiable, so the scoring function can be trained by backpropagation.

### 2.2 The softmax, carefully

$\text{softmax}(s)_j = e^{s_j}/\sum_l e^{s_l}$ has four properties you will use repeatedly:

1. **Output is a probability distribution** — positive and summing to 1.
2. **Shift invariance:** $\text{softmax}(s + c\mathbf 1) = \text{softmax}(s)$ for any constant $c$, because $e^{s_j + c} = e^c e^{s_j}$ and $e^c$ cancels. Implementations subtract $\max_l s_l$ before exponentiating to avoid overflow; this changes nothing mathematically. (Shift invariance has a consequence for GAT, Section 5.6.)
3. **Masking with $-\infty$:** setting $s_j = -\infty$ gives $e^{s_j} = 0$, so item $j$ gets weight exactly 0 and the rest renormalise among themselves. This is how attention is restricted to graph neighbours.
4. **Gradient:** $\dfrac{\partial \alpha_j}{\partial s_k} = \alpha_j(\delta_{jk} - \alpha_k)$. When the softmax **saturates** (one $\alpha \approx 1$, the rest $\approx 0$), every entry of this Jacobian is $\approx 0$: $\alpha_j(1-\alpha_j) \approx 0$ for the winner and $\alpha_j\alpha_k \approx 0$ elsewhere. A saturated softmax barely learns. This is the reason for the $\sqrt d$ scaling below.

*Derivation of 4.* $\alpha_j = e^{s_j}/Z$ with $Z = \sum_l e^{s_l}$, $\partial Z/\partial s_k = e^{s_k}$. Quotient rule: $\partial\alpha_j/\partial s_k = (\delta_{jk}e^{s_j}Z - e^{s_j}e^{s_k})/Z^2 = \alpha_j\delta_{jk} - \alpha_j\alpha_k$.

### 2.3 Scoring functions

| Name | Score $s(q, k)$ | Parameters | Origin |
|---|---|---|---|
| **Additive** ("concat", MLP) | $v^\top \tanh(W_q q + W_k k)$ | $W_q, W_k, v$ | Bahdanau, Cho & Bengio (2015), neural machine translation |
| **Multiplicative** ("general") | $q^\top W k$ | $W$ | Luong, Pham & Manning (2015) |
| **Dot product** | $q^\top k$ | none (projections done beforehand) | Luong et al. (2015) |
| **Scaled dot product** | $q^\top k / \sqrt{d_k}$ | none | Vaswani et al. (2017), the Transformer |
| **GAT** | $\text{LeakyReLU}(a^\top [Wh_i \Vert Wh_j])$ | $W, a$ | Veličković et al. (2018) — a variant of additive attention |
| **GATv2** | $a^\top\text{LeakyReLU}(W[h_i \Vert h_j])$ | $W, a$ | Brody, Alon & Yahav (2022) — closer to Bahdanau's form |

*Additive* attention runs query and key through a small one-hidden-layer network: flexible and well-behaved when query and key have different dimensions. *Dot-product* attention is a single matrix multiplication for all pairs — far faster on GPUs — which is why the Transformer uses it. GAT is a cheap relative of additive attention, with a twist we will dissect in Section 6.

### 2.4 Worked example by hand

One query, three keys, three values (all 2-dimensional):
$$
q = (1, 0),\quad K = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad V = \begin{pmatrix}1&0\\0&1\\0.5&0.5\end{pmatrix}.
$$

**Scaled dot product** ($d = 2$):

1. Scores $q^\top k_j/\sqrt2$: $(1, 0, 1)/1.4142 = (0.7071, 0, 0.7071)$.
2. Exponentials: $e^{0.7071} = 2.0281$, $e^0 = 1$, $2.0281$; sum $= 5.0562$.
3. Weights: $\alpha = (0.4011, 0.1978, 0.4011)$.
4. Output: $0.4011(1,0) + 0.1978(0,1) + 0.4011(0.5,0.5) = (0.6017, 0.3983)$.

**Additive** with $W_q = W_k = I$ and $v = (1, 1)$, so the score is $\tanh(q_1 + k_1) + \tanh(q_2 + k_2)$:

1. $k_1$: $\tanh 2 + \tanh 0 = 0.9640$; $k_2$: $\tanh 1 + \tanh 1 = 1.5232$; $k_3$: $\tanh 2 + \tanh 1 = 1.7256$.
2. Softmax: $(2.622, 4.586, 5.616)/12.82 = (0.204, 0.358, 0.438)$.
3. Output: $0.204(1,0) + 0.358(0,1) + 0.438(0.5,0.5) = (0.423, 0.577)$.

Notice that the two score functions disagree about which key is most relevant — a scoring function is a *modelling choice*, and in practice it is learned (here the weights were fixed for illustration).

```python
# file: c4_01_attention_basics.py
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)           # subtract max for numerical safety
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

q = np.array([1., 0.])                                # one query
K = np.array([[1., 0.], [0., 1.], [1., 1.]])          # three keys
V = np.array([[1., 0.], [0., 1.], [.5, .5]])          # three values

# --- scaled dot-product attention -----------------------------------------------
scores = K @ q / np.sqrt(len(q))
alpha = softmax(scores)
print("dot-product scores :", scores)
print("weights alpha      :", alpha, " sum =", alpha.sum())
print("output alpha @ V   :", alpha @ V)

# --- additive (Bahdanau) attention with W_q = W_k = I, v = [1, 1] -------------
v = np.array([1., 1.])
add_scores = np.tanh(q[None, :] + K) @ v
print("additive scores    :", add_scores)
print("additive weights   :", softmax(add_scores))
print("additive output    :", softmax(add_scores) @ V)

# --- why scale by sqrt(d)? variance of q.k grows with d -------------------------
rng = np.random.default_rng(0)
for d in (4, 64, 512):
    q_, K_ = rng.normal(size=(500, d)), rng.normal(size=(500, 10, d))  # 500 trials, 10 keys
    dots = np.einsum("td,tkd->tk", q_, K_)                              # raw scores q.k
    top_raw = softmax(dots).max(1).mean()                               # avg largest weight
    top_scaled = softmax(dots / np.sqrt(d)).max(1).mean()
    print(f"d={d:3d}: var(q.k)={dots.var():6.1f}  var(q.k/sqrt d)={(dots / np.sqrt(d)).var():.2f}  "
          f"avg max weight: unscaled={top_raw:.3f}  scaled={top_scaled:.3f}")
```

Output:

```text
dot-product scores : [0.707 0.    0.707]
weights alpha      : [0.401 0.198 0.401]  sum = 1.0
output alpha @ V   : [0.602 0.398]
additive scores    : [0.964 1.523 1.726]
additive weights   : [0.204 0.358 0.438]
additive output    : [0.423 0.577]
d=  4: var(q.k)=   3.9  var(q.k/sqrt d)=0.97  avg max weight: unscaled=0.487  scaled=0.303
d= 64: var(q.k)=  65.7  var(q.k/sqrt d)=1.03  avg max weight: unscaled=0.851  scaled=0.316
d=512: var(q.k)= 499.8  var(q.k/sqrt d)=0.98  avg max weight: unscaled=0.951  scaled=0.325
```

### 2.5 Why divide by $\sqrt{d_k}$?

Suppose the components of $q$ and $k$ are independent with mean 0 and variance 1 (roughly true at initialisation). Then
$$\mathbb E[q^\top k] = \sum_{c=1}^{d}\mathbb E[q_c]\,\mathbb E[k_c] = 0,\qquad
\operatorname{Var}(q^\top k) = \sum_{c=1}^d \operatorname{Var}(q_c k_c) = \sum_{c=1}^d \mathbb E[q_c^2]\,\mathbb E[k_c^2] = d.$$
The scores' standard deviation grows like $\sqrt d$; with $d = 512$, scores spread over $\pm 40$ or so, the softmax becomes nearly one-hot (average top weight 0.951 in the run above), and by Section 2.2 its gradient nearly vanishes. Dividing by $\sqrt d$ restores unit variance (0.97–1.03 above) and keeps the softmax in its sensitive range (top weight about 0.3 for 10 keys, regardless of $d$).

GAT does not divide by anything: its score is a single learned linear functional of 2·$F'$ numbers, and the learned vector $a$ can absorb any scale. The project's per-head dimension is only 16.

---

## 3. Multi-head attention

One attention distribution can focus on one "kind" of relevance. **Multi-head attention** runs $H$ attention mechanisms in parallel, each with its own projections, and combines their outputs:
$$
\text{head}_h = \text{Attention}(QW_h^Q, KW_h^K, VW_h^V),\qquad
\text{MultiHead}(Q,K,V) = [\text{head}_1 \Vert \dots \Vert \text{head}_H]\,W^O .
$$
With model width $d_\text{model}$ and $d_k = d_\text{model}/H$ per head, the cost and parameter count equal those of one full-width head, but the model can attend to several things at once — in a molecule-similarity graph, one head might follow neighbours that share a scaffold, another neighbours that share a target class. Multiple heads also **stabilise** training: averaging several noisy attention patterns reduces variance (this was the original motivation in GAT).

Implementation trick: compute one big projection of width $d_\text{model}$, then `view` it as $(n, H, d_k)$, so all heads are computed with batched matrix products. The project does exactly this (`.view(-1, self.h, self.dh)` in `DenseGAT.forward`).

```python
# file: c4_02_multihead.py
import torch
import torch.nn as nn

torch.manual_seed(0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.h, self.dk = heads, d_model // heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X, mask=None):                  # X: (n, d_model); mask: (n, n) bool
        n = X.shape[0]
        Q = self.Wq(X).view(n, self.h, self.dk).transpose(0, 1)   # (h, n, dk)
        K = self.Wk(X).view(n, self.h, self.dk).transpose(0, 1)
        V = self.Wv(X).view(n, self.h, self.dk).transpose(0, 1)
        scores = Q @ K.transpose(1, 2) / self.dk ** 0.5           # (h, n, n)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))     # forbid some pairs
        att = scores.softmax(-1)                                  # each row sums to 1
        out = (att @ V).transpose(0, 1).reshape(n, -1)            # concat heads
        return self.Wo(out), att

d_model, heads, n = 8, 2, 5
mine = MultiHeadSelfAttention(d_model, heads)
ref = nn.MultiheadAttention(d_model, heads, batch_first=True)
with torch.no_grad():                                 # copy my weights into PyTorch's module
    ref.in_proj_weight.copy_(torch.cat([mine.Wq.weight, mine.Wk.weight, mine.Wv.weight]))
    ref.in_proj_bias.copy_(torch.cat([mine.Wq.bias, mine.Wk.bias, mine.Wv.bias]))
    ref.out_proj.weight.copy_(mine.Wo.weight); ref.out_proj.bias.copy_(mine.Wo.bias)

X = torch.randn(n, d_model)
out_mine, att = mine(X)
out_ref, att_ref = ref(X[None], X[None], X[None], average_attn_weights=False)
print("matches nn.MultiheadAttention:", torch.allclose(out_mine, out_ref[0], atol=1e-6),
      torch.allclose(att, att_ref[0], atol=1e-6))
print("attention of head 0, row sums:", att[0].sum(-1).detach().numpy().round(4))

# a graph as an attention mask: token i may only look at its graph neighbours
adj = torch.tensor([[1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1]], dtype=torch.bool)
_, att_masked = mine(X, mask=adj)
print("masked attention, head 0:\n", att_masked[0].detach().numpy().round(3))
```

Output:

```text
matches nn.MultiheadAttention: True True
attention of head 0, row sums: [1. 1. 1. 1. 1.]
masked attention, head 0:
 [[0.412 0.253 0.335 0.    0.   ]
 [0.576 0.2   0.224 0.    0.   ]
 [0.092 0.267 0.309 0.332 0.   ]
 [0.    0.    0.361 0.314 0.325]
 [0.    0.    0.    0.446 0.554]]
```

Our 20-line implementation matches PyTorch's built-in module exactly. The last part is the key bridge to graphs: **a boolean mask turns self-attention into attention over graph neighbours** — zeros appear exactly where the adjacency matrix (with self-loops) has zeros, and every row still sums to 1.

---

## 4. Self-attention and Transformers, in brief

**Self-attention**: queries, keys and values are all computed from the *same* set of items — every item attends to every other item (and itself). Given $X \in \mathbb R^{n\times d}$: $Q = XW^Q$, $K = XW^K$, $V = XW^V$ and $\text{softmax}(QK^\top/\sqrt{d_k})V$.

A **Transformer layer** (Vaswani et al. 2017) wraps multi-head self-attention in a standard block:
$$
\begin{aligned}
X' &= \text{LayerNorm}\big(X + \text{MultiHead}(X, X, X)\big)\\
X'' &= \text{LayerNorm}\big(X' + \text{FFN}(X')\big),\qquad \text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2
\end{aligned}
$$
(the original "post-norm" order; many modern models put LayerNorm first). Because self-attention ignores order (it is permutation equivariant — Unit C3's notion!), sequence models add **positional encodings** to $X$. The base model in the paper used $d_\text{model} = 512$, $H = 8$ heads of $d_k = 64$.

**Transformers are GNNs.** Self-attention over $n$ items is a message-passing layer on the **complete graph** with attention-weighted aggregation; restricting attention with a mask (Section 3) gives a GNN on an arbitrary graph. Conversely, a GAT is a Transformer-style attention layer whose mask is the adjacency matrix. The cost of full self-attention is $O(n^2 d)$ time and $O(n^2)$ memory per head; on a graph, only edges cost.

**Echoes in the project.** `HeteroLayer.forward` computes `F.elu(LayerNorm(z + skip(h)))`: an attention sub-layer with a residual connection and LayerNorm, much like a Transformer sub-layer (but with a *learned* skip projection, ELU instead of an FFN sub-block, and graph-masked rather than full attention).

---

## 5. Graph Attention Networks (Veličković et al. 2018)

### 5.1 From fixed to learned coefficients

Recall the GCN layer node by node: $h_i' = \sigma\big(\sum_{j\in\mathcal N(i)\cup\{i\}} c_{ij} W h_j\big)$ with $c_{ij} = 1/\sqrt{\tilde d_i\tilde d_j}$ fixed by the graph. GAT keeps everything except the coefficients, which become **learned, feature-dependent attention weights** $\alpha_{ij}$:
$$ h_i' = \sigma\Big(\sum_{j\in\mathcal N(i)\cup\{i\}} \alpha_{ij}\, W h_j\Big). $$
In message-passing terms (Unit C3, Section 3.1): message $\psi = \alpha_{ij}Wh_j$, aggregation = sum (with weights summing to 1, so effectively a weighted mean), update $= \sigma$.

### 5.2 The GAT equations

Input: node features $h_i \in \mathbb R^F$. Output: $h_i' \in \mathbb R^{F'}$. Parameters: a shared weight matrix $W\in\mathbb R^{F'\times F}$ and an attention vector $a \in \mathbb R^{2F'}$.

1. **Linear projection** (shared by all nodes): $z_i = W h_i$.
2. **Unnormalised attention score** for every edge $j \to i$:
$$ e_{ij} = \text{LeakyReLU}\big(a^\top [\,z_i \,\Vert\, z_j\,]\big), $$
where $\Vert$ is concatenation and LeakyReLU has negative slope 0.2: $\text{LeakyReLU}(x) = x$ for $x > 0$, $0.2x$ otherwise.
3. **Masked softmax over the neighbourhood** (this is what makes it a *graph* attention network — "masked attention" in the paper; $j$ ranges over $\mathcal N(i)$ including $i$ itself):
$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal N(i)\cup\{i\}} \exp(e_{ik})}. $$
4. **Weighted aggregation and nonlinearity:**
$$ h_i' = \sigma\Big(\sum_{j\in\mathcal N(i)\cup\{i\}} \alpha_{ij}\, z_j\Big). $$

In Section 2.1's vocabulary: node $i$'s query is $z_i$ (through the first half of $a$), neighbour $j$'s key is $z_j$ (through the second half of $a$), and its value is also $z_j$. The paper uses ELU as $\sigma$ in hidden layers.

**Properties worth stating explicitly.**

* **Permutation equivariant**, like every message-passing layer: the score of $j$ depends on $j$'s features, not on its position in a list.
* **No dependence on the global graph structure** (unlike GCN's degree normalisation, which uses $\tilde d_j$): the layer needs only each node's neighbour list, so GAT applies directly to inductive settings (the paper's PPI experiment tests on unseen graphs).
* **Different weights for different neighbours, computed from features** — the inductive bias GCN lacks. With all scores equal, GAT reduces to a mean-aggregation GCN.

### 5.3 Multi-head GAT: concatenate or average

With $K$ independent heads (each with its own $W^k$ and $a^k$):

* **Hidden layers — concatenate:**
$$ h_i' = \big\Vert_{k=1}^{K}\ \sigma\Big(\sum_j \alpha_{ij}^{k} W^k h_j\Big) \in \mathbb R^{KF'}. $$
* **Final (prediction) layer — average**, then apply the output nonlinearity (softmax for classes, sigmoid for multi-label):
$$ h_i' = \sigma\Big(\frac1K\sum_{k=1}^K\sum_j \alpha_{ij}^k W^k h_j\Big). $$
Concatenating in the last layer would leave $K$ separate sets of class logits with no sensible single prediction; averaging makes the heads vote.

The paper's settings: on Cora, 8 heads × 8 features (64 hidden units) in layer 1 with ELU, and a single-head output layer; on the inductive PPI benchmark, 3 layers with 4 heads × 256 features, 6 averaged heads in the output layer, and skip connections. The project uses **4 heads × 16 = 64**, concatenated, in every layer (`DenseGAT(in_dim=64, out_dim=64, heads=4)`), because its "prediction layer" is a separate bilinear decoder, not a GAT layer.

### 5.4 Regularisation: attention dropout

GAT applies dropout (p = 0.6 on the small citation datasets) both to the layer inputs **and to the normalised attention coefficients** $\alpha_{ij}$. Dropping $\alpha_{ij}$ means that, in this training step, node $i$ *does not hear from neighbour $j$ at all* — each node aggregates over a random sub-neighbourhood, a form of stochastic neighbour sampling (compare GraphSAGE, Unit C3 Section 6) and close to DropEdge. With PyTorch's *inverted* dropout the kept coefficients are scaled by $1/(1-p)$, so a row no longer sums to exactly 1 during training, but its **expected** value still does; at evaluation time dropout is off and rows sum to 1 again. In the project, `self.drop(att)` in `DenseGAT.forward` is attention dropout with $p$ = `dropout` (0.2 in `MVHGATConfig`).

### 5.5 Worked example by hand

Same 4-node graph as in Unit C3 (edges 0–1, 0–2, 1–2, 2–3) **with self-loops**, one head, $F = F' = 2$:
$$
H = \begin{pmatrix}1&0\\0&1\\1&1\\-1&0\end{pmatrix},\quad
W = \begin{pmatrix}1&0\\1&1\end{pmatrix},\quad
a = [\,a_\text{dst};\,a_\text{src}\,] = [\,(1,\,0);\ (0.5,\,-1)\,].
$$

**Step 1 — project.** $z_j = W h_j$: $z_0 = (1, 1)$, $z_1 = (0, 1)$, $z_2 = (1, 2)$, $z_3 = (-1, -1)$.

**Step 2 — the two halves of the score.** Because $a^\top[z_i \Vert z_j] = a_\text{dst}^\top z_i + a_\text{src}^\top z_j$ (Section 5.7), compute two numbers per node:

| node | $a_\text{dst}^\top z$ ("as receiver") | $a_\text{src}^\top z$ ("as sender") |
|---|---|---|
| 0 | 1 | $0.5 - 1 = -0.5$ |
| 1 | 0 | $-1$ |
| 2 | 1 | $0.5 - 2 = -1.5$ |
| 3 | $-1$ | $-0.5 + 1 = 0.5$ |

**Step 3 — scores, node 2** (neighbours 0, 1, 2, 3): $1 + (-0.5) = 0.5$; $1 + (-1) = 0$; $1 + (-1.5) = -0.5 \to$ LeakyReLU $\to -0.1$; $1 + 0.5 = 1.5$. So $e_{2\cdot} = (0.5, 0, -0.1, 1.5)$.

**Step 4 — softmax.** $e^{0.5} = 1.6487$, $e^{0} = 1$, $e^{-0.1} = 0.9048$, $e^{1.5} = 4.4817$; sum $8.0352$; $\alpha_{2\cdot} = (0.2052, 0.1245, 0.1126, 0.5578)$.

**Step 5 — aggregate.** $h_2' = 0.2052(1,1) + 0.1245(0,1) + 0.1126(1,2) + 0.5578(-1,-1) = (-0.2400, -0.0029)$, and $\text{ELU}(x) = e^x - 1$ for $x<0$ gives $(-0.2133, -0.0029)$.

**Node 3** (neighbours 2, 3): $e_{32} = \text{LReLU}(-1 - 1.5) = -0.5$, $e_{33} = \text{LReLU}(-1 + 0.5) = -0.1$; $\alpha_{3\cdot} = (0.6065, 0.9048)/1.5113 = (0.4013, 0.5987)$; $h_3' = 0.4013(1,2) + 0.5987(-1,-1) = (-0.1974, 0.2039)$.

```python
# file: c4_03_gat_by_hand.py
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# same 4-node graph as in Unit C3, WITH self-loops (each node attends to itself too)
adj = np.array([[1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 1]], dtype=bool)
H = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])     # input features h_j
W = np.array([[1., 0.], [1., 1.]])                           # z_j = W h_j
a_dst = np.array([1.0, 0.0])                                 # first half of a
a_src = np.array([0.5, -1.0])                                # second half of a

Z = H @ W.T                                                  # row j = W h_j
print("Z = W h:\n", Z)
s_dst, s_src = Z @ a_dst, Z @ a_src
print("a_dst.z_i:", s_dst, "  a_src.z_j:", s_src)

leaky = lambda x: np.where(x > 0, x, 0.2 * x)
E = leaky(s_dst[:, None] + s_src[None, :])                   # e_ij for EVERY pair
print("e_ij (all pairs):\n", E)
E_masked = np.where(adj, E, -np.inf)                         # keep only real neighbours
alpha = np.exp(E_masked - E_masked.max(1, keepdims=True))
alpha /= alpha.sum(1, keepdims=True)
print("alpha (masked softmax, rows sum to 1):\n", alpha)
out = alpha @ Z
print("h' = sum_j alpha_ij z_j (before nonlinearity):\n", out)
print("ELU(h'):\n", np.where(out > 0, out, np.exp(out) - 1))

# the concatenation form a^T [z_i || z_j] gives exactly the same scores
a = np.concatenate([a_dst, a_src])
E_concat = np.array([[leaky(a @ np.concatenate([Z[i], Z[j]])) for j in range(4)] for i in range(4)])
print("split form == concat form:", np.allclose(E, E_concat))

# "static" attention: without the mask, every node ranks the keys in the same order
print("argmax_j e_ij for every i (no mask):", E.argmax(1))
```

Output:

```text
Z = W h:
 [[ 1.  1.]
 [ 0.  1.]
 [ 1.  2.]
 [-1. -1.]]
a_dst.z_i: [ 1.  0.  1. -1.]   a_src.z_j: [-0.5 -1.  -1.5  0.5]
e_ij (all pairs):
 [[ 0.5  0.  -0.1  1.5]
 [-0.1 -0.2 -0.3  0.5]
 [ 0.5  0.  -0.1  1.5]
 [-0.3 -0.4 -0.5 -0.1]]
alpha (masked softmax, rows sum to 1):
 [[0.464  0.2814 0.2546 0.    ]
 [0.3672 0.3322 0.3006 0.    ]
 [0.2052 0.1245 0.1126 0.5578]
 [0.     0.     0.4013 0.5987]]
h' = sum_j alpha_ij z_j (before nonlinearity):
 [[ 0.7186  1.2546]
 [ 0.6678  1.3006]
 [-0.24   -0.0029]
 [-0.1974  0.2039]]
ELU(h'):
 [[ 0.7186  1.2546]
 [ 0.6678  1.3006]
 [-0.2133 -0.0029]
 [-0.1791  0.2039]]
split form == concat form: True
argmax_j e_ij for every i (no mask): [3 3 3 3]
```

Two observations to carry forward:

* **Unlike the GCN in Unit C3, nodes 0 and 1 no longer collapse to the same vector** ($(0.72, 1.25)$ vs $(0.67, 1.30)$): they share the same neighbourhood, but their own projected features enter their scores ($a_\text{dst}^\top z_0 = 1$ vs $a_\text{dst}^\top z_1 = 0$), which changes the *sharpness* of their attention (row 0 is more peaked than row 1).
* **Every row of the full score matrix ranks the senders in the same order** — 3, then 0, then 1, then 2 — and the unmasked argmax is node 3 for every receiver. This is GAT's *static attention*, the subject of Section 6.

### 5.6 A subtle point: why the LeakyReLU matters at all

Suppose GAT had no nonlinearity: $e_{ij} = a_\text{dst}^\top z_i + a_\text{src}^\top z_j$. In the softmax over $j$, the term $a_\text{dst}^\top z_i$ is a constant (it does not depend on $j$), and softmax is shift invariant (Section 2.2). It **cancels**:
$$ \alpha_{ij} = \frac{e^{a_\text{dst}^\top z_i}\, e^{a_\text{src}^\top z_j}}{e^{a_\text{dst}^\top z_i}\sum_k e^{a_\text{src}^\top z_k}} = \frac{e^{a_\text{src}^\top z_j}}{\sum_k e^{a_\text{src}^\top z_k}}. $$
The receiver would have no influence whatsoever on how it weighs its neighbours — attention would be a fixed "popularity" of each sender. The LeakyReLU is the only thing that lets $z_i$ matter: depending on whether $a_\text{dst}^\top z_i + a_\text{src}^\top z_j$ is positive or negative, the slope is 1 or 0.2, which changes how *spread out* $i$'s weights are. It still cannot change their *order* (Section 6). Exercise 3 verifies the cancellation numerically.

### 5.7 The split $a = [a_\text{dst}; a_\text{src}]$ and efficient computation

**The identity.** Write $a = [a_1; a_2]$ with $a_1, a_2 \in \mathbb R^{F'}$. For any $x, y \in \mathbb R^{F'}$:
$$ a^\top [x \Vert y] = \sum_{c=1}^{F'} a_c x_c + \sum_{c=1}^{F'} a_{F'+c}\, y_c = a_1^\top x + a_2^\top y. $$
That is all — the dot product of a concatenation splits into the sum of two dot products.

**Why it matters computationally.** Done naively, building $[z_i \Vert z_j]$ for every pair costs $O(N^2 F')$ time and memory ($N^2$ vectors of length $2F'$). With the split:

1. compute $s^\text{dst}_i = a_1^\top z_i$ for all $i$ and $s^\text{src}_j = a_2^\top z_j$ for all $j$: $O(NF')$;
2. form all scores by **broadcasting** an outer sum, $E = \text{LeakyReLU}(s^\text{dst}\mathbf 1^\top + \mathbf 1 (s^\text{src})^\top)$: $O(N^2)$ additions, no $F'$ factor.

In the project this is one line of `DenseGAT.forward`:
```python
e = (zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]
```
`(zd * self.a_dst).sum(-1)` is $a_\text{dst}^\top z_i$ for every destination node and head (shape D × H); `(zs * self.a_src).sum(-1)` is $a_\text{src}^\top z_j$ for every source node and head (S × H); indexing with `[:, None, :]` and `[None, :, :]` broadcasts them to a D × S × H tensor. **This is why the code stores `a` as two parameters**, `a_src` and `a_dst`, each of shape (heads, d_head): they are the two halves of the paper's $a$, one pair per head. The PyTorch Geometric `GATConv` uses the same trick (`att_src`, `att_dst`).

(The names follow the message direction: messages flow from **source** $j$ to **destination** $i$, so $a_\text{dst}$ multiplies the receiver's features and $a_\text{src}$ the sender's.)

### 5.8 GAT layer in PyTorch, and a check against the project's `DenseGAT`

```python
# file: c4_04_gat_layer.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """Dense multi-head GAT layer (Velickovic et al. 2018), from scratch.

    mask[i, j] = True  <=>  node i (destination) may attend to node j (source).
    concat=True : output = [head_1 || ... || head_K]   (hidden layers)
    concat=False: output = mean over heads              (final layer in the paper)
    """
    def __init__(self, d_in, d_head, heads, concat=True, att_drop=0.0, slope=0.2):
        super().__init__()
        self.K, self.d, self.concat, self.slope = heads, d_head, concat, slope
        self.W = nn.Linear(d_in, heads * d_head, bias=False)     # shared by src and dst
        self.a_dst = nn.Parameter(torch.empty(heads, d_head))    # a = [a_dst ; a_src]
        self.a_src = nn.Parameter(torch.empty(heads, d_head))
        nn.init.xavier_uniform_(self.a_dst); nn.init.xavier_uniform_(self.a_src)
        self.att_drop = nn.Dropout(att_drop)

    def forward(self, H, mask, return_att=False):
        N = H.shape[0]
        Z = self.W(H).view(N, self.K, self.d)                     # N, K, d
        s_dst = (Z * self.a_dst).sum(-1)                          # N, K   a_dst . z_i
        s_src = (Z * self.a_src).sum(-1)                          # N, K   a_src . z_j
        e = F.leaky_relu(s_dst[:, None, :] + s_src[None, :, :], self.slope)   # N, N, K
        e = e.masked_fill(~mask[:, :, None], float("-inf"))       # non-neighbours -> -inf
        att = torch.softmax(e, dim=1)                             # normalise over j
        att = torch.nan_to_num(att, nan=0.0)                      # rows with no neighbour
        out = torch.einsum("ijk,jkd->ikd", self.att_drop(att), Z)  # N, K, d
        out = out.reshape(N, -1) if self.concat else out.mean(1)
        return (out, att) if return_att else out

torch.manual_seed(0)
adj = torch.tensor([[1, 1, 1, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                    [0, 0, 1, 1]], dtype=torch.bool)
H = torch.randn(4, 3)

layer = GATLayer(3, d_head=2, heads=4, concat=True)
out, att = layer(H, adj, return_att=True)
print("concat output shape :", tuple(out.shape), "  attention shape (dst, src, heads):", tuple(att.shape))
print("attention rows sum to 1 for every head:", torch.allclose(att.sum(1), torch.ones(4, 4)))
print("zero weight on non-edges:", bool((att[~adj] == 0).all()))
mean_layer = GATLayer(3, d_head=2, heads=4, concat=False)
print("average-heads output shape:", tuple(mean_layer(H, adj).shape))

# ---- compare with the project's DenseGAT ---------------------------------------
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import DenseGAT
proj = DenseGAT(3, 8, heads=4, dropout=0.0)
with torch.no_grad():                    # tie the project's W_src = W_dst = our W
    proj.W_src.weight.copy_(layer.W.weight); proj.W_dst.weight.copy_(layer.W.weight)
    proj.a_src.copy_(layer.a_src); proj.a_dst.copy_(layer.a_dst)
out_proj, has = proj(H, H, adj)          # (h_dst, h_src, mask)
print("project DenseGAT == our GATLayer:", torch.allclose(out_proj, out, atol=1e-6))
print("'has at least one neighbour' flags:", has.tolist())

# ---- a node with NO neighbours --------------------------------------------------
adj2 = adj.clone(); adj2[3] = False                    # node 3 now has no neighbour at all
e = torch.tensor([[0.3, float("-inf")], [float("-inf"), float("-inf")]])
print("softmax of an all -inf row:", torch.softmax(e, 1)[1].tolist())
out2, has2 = proj(H, H, adj2)
print("output row of the isolated node:", out2[3].tolist())
print("has-neighbour flags:", has2.tolist())
```

Output:

```text
concat output shape : (4, 8)   attention shape (dst, src, heads): (4, 4, 4)
attention rows sum to 1 for every head: True
zero weight on non-edges: True
average-heads output shape: (4, 2)
project DenseGAT == our GATLayer: True
'has at least one neighbour' flags: [True, True, True, True]
softmax of an all -inf row: [nan, nan]
output row of the isolated node: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
has-neighbour flags: [True, True, True, False]
```

With the project's two projection matrices tied ($W_\text{src} = W_\text{dst}$), `DenseGAT` computes **exactly** the original GAT layer (no bias, no output nonlinearity — the project applies ELU later, after combining relations). Section 7 explains the last three lines.

### 5.9 GAT vs GCN on a noisier toy graph

We reuse Unit C3's two-community graph but make it **noisier**: cross-community edge probability 0.08 instead of 0.02, so 75 of 269 edges (28 %) connect different classes. Ten labels per class. A 2-layer GAT (4 heads × 8, ELU, dropout 0.5 on inputs and attention, single-head output layer) against a 2-layer GCN (32 hidden).

```python
# file: c4_07_gat_vs_gcn.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):                      # same layer as in Section 5.8 (compact)
    def __init__(self, d_in, d_head, heads, concat=True, att_drop=0.0):
        super().__init__()
        self.K, self.d, self.concat = heads, d_head, concat
        self.W = nn.Linear(d_in, heads * d_head, bias=False)
        self.a_dst = nn.Parameter(nn.init.xavier_uniform_(torch.empty(heads, d_head)))
        self.a_src = nn.Parameter(nn.init.xavier_uniform_(torch.empty(heads, d_head)))
        self.att_drop = nn.Dropout(att_drop)
    def forward(self, H, mask):
        Z = self.W(H).view(H.shape[0], self.K, self.d)
        e = F.leaky_relu((Z * self.a_dst).sum(-1)[:, None] + (Z * self.a_src).sum(-1)[None], 0.2)
        att = torch.nan_to_num(torch.softmax(e.masked_fill(~mask[..., None], float("-inf")), 1))
        self.last_att = att.detach()
        out = torch.einsum("ijk,jkd->ikd", self.att_drop(att), Z)
        return out.reshape(H.shape[0], -1) if self.concat else out.mean(1)

class GAT(nn.Module):
    def __init__(self, d_in, n_cls, hid=8, heads=4, drop=0.5):
        super().__init__()
        self.l1 = GATLayer(d_in, hid, heads, concat=True, att_drop=drop)
        self.l2 = GATLayer(hid * heads, n_cls, 1, concat=False, att_drop=drop)
        self.drop = drop
    def forward(self, X, mask):
        h = F.dropout(X, self.drop, self.training)
        h = F.elu(self.l1(h, mask))
        h = F.dropout(h, self.drop, self.training)
        return self.l2(h, mask)

# a NOISIER toy graph than before: many edges cross the two communities
rng = np.random.default_rng(0)
y = np.repeat([0, 1], 30); N = 60
same = y[:, None] == y[None, :]
up = np.triu(rng.random((N, N)) < np.where(same, 0.20, 0.08), 1)
adj = up | up.T
print(f"edges inside communities: {int((adj & same).sum() // 2)}, across: {int((adj & ~same).sum() // 2)}")
X = rng.normal(size=(N, 8)).astype(np.float32); X[:, 0] += np.where(y == 1, 1.0, -1.0)
mask = torch.tensor(adj | np.eye(N, dtype=bool))              # add self-loops
Xt, yt = torch.tensor(X), torch.tensor(y)
train = np.concatenate([np.where(y == c)[0][:10] for c in (0, 1)])   # 10 labels per class
test = np.setdiff1d(np.arange(N), train)

def gcn_norm(M):
    M = M.float(); d = M.sum(1).rsqrt(); return d[:, None] * M * d[None, :]

class GCN(nn.Module):
    def __init__(self):
        super().__init__(); self.l1, self.l2 = nn.Linear(8, 32), nn.Linear(32, 2)
    def forward(self, X, mask):
        P = gcn_norm(mask)
        h = F.dropout(X, 0.5, self.training)
        h = F.dropout(F.relu(P @ self.l1(h)), 0.5, self.training)
        return P @ self.l2(h)

def run(model, seed):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train()
        loss = F.cross_entropy(model(Xt, mask)[train], yt[train])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(Xt, mask).argmax(1)[test] == yt[test]).float().mean().item()

for name in ("GCN", "GAT"):
    accs = []
    for s in range(5):
        torch.manual_seed(s)
        accs.append(run(GCN() if name == "GCN" else GAT(8, 2), s))
    print(f"{name}: test accuracy {np.mean(accs):.3f} +- {np.std(accs):.3f}")

# where does the trained GAT's first layer put its attention? (last seed's model)
torch.manual_seed(4); model = GAT(8, 2); run(model, 4)
with torch.no_grad():
    model(Xt, mask)
att = model.l1.last_att.mean(-1).numpy()                     # average over heads, (dst, src)
off = adj                                                    # real edges, no self-loops
rel = att * mask.numpy().sum(1, keepdims=True)               # attention / (1/degree): 1 = uniform
print(f"attention relative to uniform, same-community edges : {rel[off & same].mean():.3f}")
print(f"attention relative to uniform, cross-community edges: {rel[off & ~same].mean():.3f}")
```

Output:

```text
edges inside communities: 194, across: 75
GCN: test accuracy 0.885 +- 0.025
GAT: test accuracy 0.920 +- 0.010
attention relative to uniform, same-community edges : 1.019
attention relative to uniform, cross-community edges: 0.929
```

GAT is a little more accurate and more stable across seeds. But look at the attention: relative to uniform weighting ($1/\text{degree}$ = 1.0), same-community edges receive only about 2 % more and cross-community edges 7 % less. **GAT did not learn to "cut" the noisy edges**; most of its gain comes from elsewhere (multi-head averaging, the self-feature path through $a_\text{dst}$, attention dropout as regulariser). Keep this in mind for Section 10: a better model is not evidence that its attention weights carry a crisp, interpretable story.

---

## 6. Static vs dynamic attention: GATv2 (Brody, Alon & Yahav 2022)

### 6.1 The limitation

**Definition.** A family of scoring functions computes **static attention** if, for any set of keys, there is a single key that receives the highest score from *every* query (more generally, the ranking of keys is the same for every query). Attention is **dynamic** if different queries can prefer different keys.

**Theorem (GAT is static).** For a GAT layer, for every receiver $i$, the ranking of candidate senders $j$ by $e_{ij}$ is the ranking by $a_\text{src}^\top z_j$ — the same for all receivers.

*Proof.* $e_{ij} = \text{LeakyReLU}(a_\text{dst}^\top z_i + a_\text{src}^\top z_j)$. For fixed $i$, the argument is $c_i + s_j$ with $c_i = a_\text{dst}^\top z_i$ constant in $j$. LeakyReLU is **strictly increasing** (slopes 1 and 0.2 are both positive), so $e_{ij} > e_{ik} \iff c_i + s_j > c_i + s_k \iff s_j > s_k$. The softmax is also strictly increasing in each score, so the attention weights inherit the same order. $\square$

So in GAT there is a global "attractiveness" $s_j = a_\text{src}^\top Wh_j$ of each node (per head), and every node ranks its neighbours by that one number. The receiver only controls *how peaked* its distribution is (through which side of zero the LeakyReLU argument falls) and, of course, the mask decides *which* nodes are candidates. The worked example showed it: all four rows ranked the senders $3 > 0 > 1 > 2$.

Why does this happen? The paper's score applies the learned vector $a$ and *then* the nonlinearity; since $a^\top[z_i \Vert z_j]$ is linear and the nonlinearity monotone, the composition can never model an **interaction** between $i$ and $j$.

### 6.2 The fix: apply $a$ after the nonlinearity

GATv2 swaps the order:
$$ e_{ij} = a^\top\, \text{LeakyReLU}\big(W [h_i \Vert h_j]\big) = a^\top\,\text{LeakyReLU}\big(W_\text{dst} h_i + W_\text{src} h_j\big), $$
with $W = [W_\text{dst} \mid W_\text{src}]$. Now $i$ and $j$ are mixed *inside* the nonlinearity, and the score is a one-hidden-layer MLP of the pair — Bahdanau's additive attention with LeakyReLU instead of tanh. Brody et al. prove that GATv2 can express **any** selection of a "best" key per query (it is a universal approximator of such scoring functions), i.e. it computes dynamic attention. The parameter count is essentially unchanged.

**Costs.** The split trick no longer applies — the nonlinearity sits between $i$ and $j$ — so a dense implementation must form the $N_\text{dst}\times N_\text{src}\times F'$ tensor $W_\text{dst}h_i + W_\text{src}h_j$: $F'$ times more memory than GAT's $N_\text{dst}\times N_\text{src}$ scores. With sparse edge lists it is $O(|E|F')$, which is fine.

### 6.3 A task that needs dynamic attention: DictionaryLookup

Brody et al. built a minimal benchmark. A complete bipartite graph has $k$ **key** nodes and $k$ **query** nodes. Key $j$ carries a name and a value; query $i$ carries only a name. Query $i$ must output the value of the key with the **same name**. So every query must put its attention on a *different* key — impossible for static attention. The graph is bipartite (destinations = queries, sources = keys), exactly the shape of the project's `assoc>drug` relation.

```python
# file: c4_06_gat_vs_gatv2_lookup.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# DictionaryLookup (Brody et al. 2022): k "key" nodes, k "query" nodes, complete bipartite
# graph query <- key. Key j carries (name_j, value_j); query i carries only a name.
# Query i must output the value of the key with the SAME name -> it has to attend to a
# different key for every query. Bipartite (dst = queries, src = keys), like assoc>drug.
k, B = 8, 128

def batch(B):
    names = torch.arange(k).expand(B, k)                        # key j has name j
    values = torch.stack([torch.randperm(k) for _ in range(B)])  # random values
    q_names = torch.stack([torch.randperm(k) for _ in range(B)]) # queries ask in random order
    x_key = torch.cat([F.one_hot(names, k), F.one_hot(values, k)], -1).float()      # B,k,2k
    x_qry = torch.cat([F.one_hot(q_names, k), torch.zeros(B, k, k)], -1).float()  # B,k,2k
    target = torch.gather(values, 1, q_names)                   # value of the matching key
    return x_qry, x_key, target

class BipartiteAttention(nn.Module):
    def __init__(self, version, d=32):
        super().__init__()
        self.version = version
        self.W_dst, self.W_src = nn.Linear(2 * k, d), nn.Linear(2 * k, d)
        self.a = nn.Linear(d, 1, bias=False)          # GATv2: a^T LeakyReLU(W_dst x_i + W_src x_j)
        self.a_dst = nn.Linear(d, 1, bias=False)      # GAT  : LeakyReLU(a_dst^T z_i + a_src^T z_j)
        self.a_src = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(d, k)
    def forward(self, x_qry, x_key):
        zd, zs = self.W_dst(x_qry), self.W_src(x_key)              # B,k,d
        if self.version == "GAT":
            e = F.leaky_relu(self.a_dst(zd) + self.a_src(zs).transpose(1, 2), 0.2)   # B,kq,kk
        else:
            e = self.a(F.leaky_relu(zd[:, :, None, :] + zs[:, None, :, :], 0.2)).squeeze(-1)
        att = e.softmax(-1)
        return self.out(F.elu(att @ zs)), att

for version in ("GAT", "GATv2"):
    torch.manual_seed(0)
    m = BipartiteAttention(version)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(1500):
        xq, xk, y = batch(B)
        logits, _ = m(xq, xk)
        loss = F.cross_entropy(logits.reshape(-1, k), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xq, xk, y = batch(1000)
        logits, att = m(xq, xk)
        acc = (logits.argmax(-1) == y).float().mean().item()
        # how many DIFFERENT keys are the top choice across the k queries of one graph?
        distinct = torch.tensor([len(set(a.argmax(-1).tolist())) for a in att]).float().mean().item()
    print(f"{version:5s}: test accuracy {acc:.3f}   distinct top-attended keys per graph {distinct:.2f} / {k}")
```

Output (about 30 seconds on a CPU):

```text
GAT  : test accuracy 0.125   distinct top-attended keys per graph 1.00 / 8
GATv2: test accuracy 1.000   distinct top-attended keys per graph 8.00 / 8
```

The result is as stark as the theory predicts. GAT is stuck at $1/k = 0.125$ (chance): in every graph all 8 queries attend most strongly to the **same** key. GATv2 solves the task perfectly, with each query focusing on a different key.

### 6.4 Does static attention matter for the project?

Honestly: we do not know without an ablation, and the project has not run one. Arguments both ways:

* **Probably mild.** In a similarity view the mask already makes attention receiver-specific (each drug chooses among *its own* 10–94 neighbours), and a global per-node "informativeness" score — "drugs with clean, well-annotated profiles are good neighbours", "well-studied diseases are informative" — is plausibly most of what is needed. GAT is also cheaper in memory, which matters for the dense implementation.
* **Possibly limiting.** In `assoc>drug`, whether disease $j$ is relevant to drug $i$ could depend on *which aspect of $i$* we are modelling (an anti-inflammatory used in both arthritis and a cardiac indication). Static attention cannot express "for this drug, disease A matters more; for that drug, disease B" when both diseases are candidates for both drugs.

Swapping `DenseGAT`'s score for the GATv2 form is a natural ablation for the paper (Exercise 7 writes the layer). Run it on the validation split, never the test folds.

---

## 7. Nodes with no neighbours

### 7.1 What goes wrong

If node $i$ has **no** allowed neighbour in a relation, every entry of its score row is masked to $-\infty$. The softmax computes $e^{-\infty}/\sum e^{-\infty} = 0/0 = \text{NaN}$ (the output above: `softmax of an all -inf row: [nan, nan]`). NaN then propagates through the weighted sum into the node's embedding, into the loss and into every gradient: **one empty row can destroy the whole model.**

### 7.2 What the project does

`DenseGAT.forward` handles it in three steps:

```python
e = F.leaky_relu(e, 0.2).masked_fill(~mask[..., None], float("-inf"))
att = torch.softmax(e, dim=1)
att = torch.nan_to_num(att, nan=0.0)                      # nodes w/o neighbours
out = torch.einsum("dsh,shk->dhk", self.drop(att), zs).reshape(h_dst.shape[0], -1)
return out, mask.any(1)
```

1. `nan_to_num(att, nan=0.0)` replaces the NaN row by zeros, so the node's **message from this relation is the zero vector** (the "output row of the isolated node: [0.0, ...]" above).
2. The layer also returns `mask.any(1)`: a boolean per destination node, **"has at least one neighbour in this relation"**.
3. `HeteroLayer` passes these flags to `ViewAttention` as `valid`, which masks the relation's view score with $-\infty$ *before* the softmax over relations — so the empty relation gets view weight $\beta = 0$, and the node's combined message is built only from relations that actually have information. (If a node had no valid relation at all, `ViewAttention` would also produce an all-NaN column, again zeroed by `nan_to_num`; its new state would then come only from the skip connection `self.skip[t](h[t])`. In the project this cannot happen, see below.)

**Which relations can be empty in practice?**

* **Similarity views — never.** `knn_mask` always sets the diagonal to `True`, so every node has at least itself. An *isolated* node in a view (no SMILES, no genes — e.g. 153 diseases in `gene_d`) attends only to itself: $\alpha_{ii} = 1$ and its message is $W_\text{src}h_i$, its own transformed state. That is a valid message, but an uninformative one; the view attention learns to down-weight such views for such nodes. (On a self-only row, $\alpha_{ii} = 1$ regardless of the score, so the attention parameters receive no gradient from that node.)
* **Association relations — often.** `assoc>drug` and `assoc>disease` have no self-loops (they are bipartite). A disease has an empty row whenever all its links are invisible: during training for the 10 % "cold" diseases of each epoch (`cold_frac`), for diseases whose few links were all hidden by the 20 % `drop_edge`, and at evaluation for a held-out disease in leave-one-disease-out testing. These are exactly the cases the `valid` flag and the degree gate are designed for.

The script below runs the project's own classes on a 4-drug × 3-disease toy where drug 3 has no visible link:

```python
# file: c4_09_bipartite_empty_rows.py
import sys
import torch
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import DenseGAT, ViewAttention

torch.manual_seed(0)
# visible known links: 4 drugs x 3 diseases; drug 3 has NO visible link (cold drug)
A = torch.tensor([[1, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 0, 0]], dtype=torch.bool)
h_drug, h_dis = torch.randn(4, 16), torch.randn(3, 16)

gat = DenseGAT(16, 16, heads=4, dropout=0.0)
msg, has = gat(h_drug, h_dis, A)                 # relation assoc>drug: dst = drugs, src = diseases
print("message shape (drugs x hidden):", tuple(msg.shape))
print("has a neighbour in assoc>drug:", has.tolist())
print("message of drug 3 is all zeros:", bool((msg[3] == 0).all()))

msg_T, has_T = gat(h_dis, h_drug, A.T)           # the reverse direction, assoc>disease
print("assoc>disease message shape:", tuple(msg_T.shape), " has:", has_T.tolist())

# View attention for the drugs: relation 0 = a similarity view (everyone valid),
# relation 1 = assoc>drug (drug 3 invalid). Drug 3 must put all its weight on relation 0.
va = ViewAttention(16)
msgs = torch.stack([torch.randn(4, 16), msg])            # R=2, N=4, D=16
valid = torch.stack([torch.ones(4, dtype=torch.bool), has])
z, beta = va(msgs, valid)
print("beta (relations x drugs):\n", beta.detach().numpy().round(3))
```

Output:

```text
message shape (drugs x hidden): (4, 16)
has a neighbour in assoc>drug: [True, True, True, False]
message of drug 3 is all zeros: True
assoc>disease message shape: (3, 16)  has: [True, True, True]
beta (relations x drugs):
 [[0.374 0.466 0.431 1.   ]
 [0.626 0.534 0.569 0.   ]]
```

Drug 3's `assoc>drug` message is zero, flagged invalid, and its view attention puts **all** weight ($\beta = 1$) on the similarity view. (Here the same `gat` module was reused for both directions only to save lines; in the project each relation has its own `DenseGAT`.)

### 7.3 The gradient subtlety: why `masked_fill` and not "add a $-\infty$ mask"

`nan_to_num` repairs the *forward* pass. What about the *backward* pass? The softmax's backward on an all-$-\infty$ row produces NaN gradients with respect to that row's scores. They are harmless only if they are **blocked** before reaching any parameter. `masked_fill` does exactly that: its backward sets the gradient of every masked position to 0 — and in an empty row, *every* position is masked. Writing the mask as an addition (`e + mask_tensor` with $-\infty$ entries) looks equivalent in the forward pass but lets the NaN flow straight through to the parameters:

```python
# file: c4_05_mask_gradients.py
import torch

w = torch.tensor([0.3, -0.2], requires_grad=True)          # pretend these are scores e_ij
mask = torch.tensor([[True, True],
                     [False, False]])                      # row 1: no neighbours at all
weights = torch.tensor([[1., 2.], [3., 4.]])

# (a) masked_fill(-inf) + softmax + nan_to_num   (what drepo/model.py::DenseGAT does)
e = w[None, :].expand(2, 2).masked_fill(~mask, float("-inf"))
att = torch.nan_to_num(torch.softmax(e, 1), nan=0.0)
(att * weights).sum().backward()
print("masked_fill  -> gradient:", [round(g, 4) for g in w.grad.tolist()])

# (b) the same thing written as "add a -inf mask"
w.grad = None
e = w[None, :].expand(2, 2) + torch.where(mask, 0.0, float("-inf"))
att = torch.nan_to_num(torch.softmax(e, 1), nan=0.0)
(att * weights).sum().backward()
print("additive -inf -> gradient:", [round(g, 4) for g in w.grad.tolist()])
```

Output:

```text
masked_fill  -> gradient: [-0.235, 0.235]
additive -inf -> gradient: [nan, nan]
```

Same forward result, but version (b) poisons the parameters with NaN on the first `optimizer.step()`. The project's choice of `masked_fill` is therefore not cosmetic. (Other safe patterns: use a large finite negative number such as $-10^9$ instead of $-\infty$ and multiply the attention by the mask afterwards, or skip empty rows explicitly.)

---

## 8. Bipartite attention ($\text{dst} \neq \text{src}$), as in `DenseGAT`

In a homogeneous graph, senders and receivers are the same set of nodes. In a **relation between two node types** they are not: in `assoc>drug` the 593 drugs receive and the 313 diseases send; in `assoc>disease` the roles are reversed. GAT generalises directly:

$$
z^\text{src}_j = W_\text{src} h^\text{src}_j,\quad z^\text{dst}_i = W_\text{dst} h^\text{dst}_i,\quad
e_{ij} = \text{LeakyReLU}\big(a_\text{dst}^\top z^\text{dst}_i + a_\text{src}^\top z^\text{src}_j\big),
$$
$$
\alpha_{ij} = \operatorname{softmax}_{j:\ M_{ij}=1}(e_{ij}),\qquad m_i = \Big\Vert_{h=1}^{H} \sum_j \alpha^h_{ij}\, z^{\text{src},h}_j .
$$

* The **mask** $M$ is rectangular, $N_\text{dst}\times N_\text{src}$ (for `assoc>drug`: the visible part of $A$, 593 × 313; for `assoc>disease`: its transpose).
* The softmax runs over the **source** axis (`dim=1` in the code) — each destination's weights over *its* senders sum to 1.
* **Two projection matrices.** Drugs and diseases live in different feature spaces (different input features, different meanings of coordinates), so there is no reason for one $W$ to fit both. `DenseGAT` always has separate `W_src` and `W_dst`, also for same-type relations (drug ← drug), where it is a slightly more flexible variant of the original GAT; with tied matrices it *is* the original GAT (verified in Section 5.8).
* **The value is the source's projection** $z^\text{src}_j$, so the message $m_i$ lives in the same space as the other relations' messages to node type "drug" — which is what allows `ViewAttention` to combine them.

**Shapes in the project** (Fdataset, 4 heads, $d_h = 16$): for `assoc>drug`, `zs` is 313 × 4 × 16, `zd` 593 × 4 × 16, `e` and `att` 593 × 313 × 4, and the output 593 × 64. The `einsum("dsh,shk->dhk", att, zs)` contracts over the source index $s$ separately for each head $h$: for each destination $d$, head $h$ and feature $k$, $\text{out}_{dhk} = \sum_s \text{att}_{dsh}\, \text{zs}_{shk}$. `.reshape(D, -1)` then concatenates the 4 heads into 64 features.

---

## 9. Complexity

Let $N_d$, $N_s$ be the numbers of destination and source nodes, $F$ the input width, $H$ heads of width $d_h$ ($F' = Hd_h$), and $|E|$ the number of edges.

| Step | Dense (`DenseGAT`) | Sparse (edge list) |
|---|---|---|
| projections $W_\text{src}h$, $W_\text{dst}h$ | $O((N_d + N_s) F F')$ | same |
| half-scores $a^\top z$ | $O((N_d + N_s)F')$ | same |
| scores $e$, softmax | $O(N_d N_s H)$ time **and memory** | $O(\lvert E\rvert H)$ |
| aggregation | $O(N_d N_s F')$ | $O(\lvert E\rvert F')$ |
| GATv2 scores | $O(N_d N_s F')$ memory | $O(\lvert E\rvert F')$ |

The original paper quotes $O(|V|FF' + |E|F')$ per head for the sparse version — the same as a GCN. Concretely for the project:

```python
# file: c4_08_explanation_and_memory.py
import numpy as np

# 1) Attention weights can be very different while the output is (almost) identical
Z = np.array([[1.00, 2.00],      # neighbour A
              [1.01, 1.99],      # neighbour B: nearly the same value vector as A
              [0.00, 0.00]])     # neighbour C
for alpha in ([0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]):
    print(f"alpha={alpha} -> output {np.round(np.array(alpha) @ Z, 3)}")

# 2) memory of the dense attention tensor e/att in the project (float32, 4 heads)
H = 4
for name, n_dst, n_src in [("drug view (593x593)", 593, 593),
                           ("disease view (313x313)", 313, 313),
                           ("assoc>drug (593x313)", 593, 313)]:
    print(f"{name:24s}: {n_dst * n_src * H:9,d} scores = {n_dst * n_src * H * 4 / 2**20:5.2f} MiB per tensor")
edges = 2 * 4384 + 593                       # chem_cdk kNN graph: directed edges + self-loops
print(f"sparse alternative for chem_cdk: {edges * H:,d} scores")
```

Output:

```text
alpha=[0.9, 0.1, 0.0] -> output [1.001 1.999]
alpha=[0.1, 0.9, 0.0] -> output [1.009 1.991]
alpha=[0.5, 0.5, 0.0] -> output [1.005 1.995]
drug view (593x593)     : 1,406,596 scores =  5.37 MiB per tensor
disease view (313x313)  :   391,876 scores =  1.49 MiB per tensor
assoc>drug (593x313)    :   742,436 scores =  2.83 MiB per tensor
sparse alternative for chem_cdk: 37,444 scores
```

(The first part belongs to Section 10.) A dense score tensor is a few MiB per relation; with several intermediate tensors per relation (`e`, the masked copy, `att`, the dropout output and their gradients), 8 relations and 2 layers, a forward-backward pass needs on the order of a few hundred MiB — fine on the project's GPU, and much simpler than sparse code. But **97 % of those scores are masked away**: the `chem_cdk` graph has only 37,444 real (edge, head) scores out of 1.4 million. For graphs with more than roughly $10^4$ nodes per type, a sparse implementation (edge list + scatter softmax, as in PyTorch Geometric's `GATConv`) is the only option.

---

## 10. Attention weights as (limited) explanations

It is tempting to read $\alpha_{ij}$ as "how much neighbour $j$ mattered for node $i$" and to report it as an explanation ("drug X was predicted for disease Y because it attended to drug Z"). Be careful. The NLP literature debated this intensely: Jain & Wallace (2019) showed that attention weights often correlate poorly with gradient-based importance and that very different attention distributions can produce the same prediction; Wiegreffe & Pinter (2019) replied that this does not make attention *useless*, only that whether it explains depends on the model and must be tested. For GNNs specifically:

1. **Different weights, same output.** If two neighbours carry similar value vectors, the output barely depends on how attention splits between them — the first part of the script above gives three very different $\alpha$ vectors and nearly identical outputs. A high $\alpha_{ij}$ then says little about the *importance* of $j$.
2. **Many paths bypass attention.** In the project, a node's new state is $\text{ELU}(\text{LayerNorm}(z + W_\text{skip}h))$: the skip connection and the other relations contribute too, and the decoder also has the propagation head and the degree gate. A large $\alpha$ inside one relation can be irrelevant to the final score.
3. **Layers and heads mix information.** In layer 2, "attention to node $j$" is attention to $j$'s *layer-1 state*, which already mixes $j$'s neighbours (and $j$'s 4 heads are concatenated and transformed). The weight does not refer to $j$'s original features.
4. **Static attention.** In GAT, rankings are global per head (Section 6): a high $\alpha_{ij}$ may reflect $j$'s general "attractiveness" rather than anything specific to $i$.
5. **Weak alignment with ground truth.** In Section 5.9's toy, GAT beat GCN, yet its attention barely separated useful from noisy edges.

**What the project does instead.** `HOW_IT_WORKS.md` §4.5 ranks its interpretability tools by faithfulness: (1) the **global view weights** $w_v$ of the *additive* propagation head — $w_v P_v[i,j]$ is literally view $v$'s share of the logit; (2) **occlusion** (`MVHGATMethod.occlusion`): switch off one evidence source and measure how much the predicted probability drops — a causal test of the model's dependence; and only then (3) the per-node **view attention** $\beta$ as supporting detail, citing Jain & Wallace. Node-level $\alpha$ from `DenseGAT` is not reported at all. That ordering is the right one: if you want to say "this prediction relied on X", **intervene on X** (remove or perturb it) and measure, rather than reading off an attention weight. Unit E2 develops this further.

---

## 11. In this project: `DenseGAT` line by line

Open `src/drepo/model.py`. The whole class:

```python
class DenseGAT(nn.Module):
    """Multi-head graph attention over a dense boolean adjacency (dst x src)."""

    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()
        assert out_dim % heads == 0
        self.h, self.dh = heads, out_dim // heads
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_dst, h_src, mask):
        zs = self.W_src(h_src).view(-1, self.h, self.dh)          # S,H,dh
        zd = self.W_dst(h_dst).view(-1, self.h, self.dh)          # D,H,dh
        e = (zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]
        e = F.leaky_relu(e, 0.2).masked_fill(~mask[..., None], float("-inf"))
        att = torch.softmax(e, dim=1)
        att = torch.nan_to_num(att, nan=0.0)                      # nodes w/o neighbours
        out = torch.einsum("dsh,shk->dhk", self.drop(att), zs).reshape(h_dst.shape[0], -1)
        return out, mask.any(1)
```

**Constructor**

| Line | Meaning | Equation |
|---|---|---|
| `assert out_dim % heads == 0` | the output is split evenly across heads | $F' = H d_h$ |
| `self.h, self.dh = heads, out_dim // heads` | 4 heads × 16 = 64 in the project | $H = 4$, $d_h = 16$ |
| `self.W_src = nn.Linear(in_dim, out_dim, bias=False)` | projection of senders, all heads at once; no bias, as in GAT | $z^\text{src}_j = W_\text{src}h_j$ |
| `self.W_dst = nn.Linear(...)` | projection of receivers (separate: node types may differ) | $z^\text{dst}_i = W_\text{dst}h_i$ |
| `self.a_src`, `self.a_dst` of shape `(heads, dh)` | the two halves of the attention vector, one pair per head | $a^h = [a^h_\text{dst}; a^h_\text{src}]$ |
| `nn.init.xavier_uniform_` | Glorot initialisation (the GAT paper's choice), keeps initial scores at a sensible scale | — |
| `self.drop = nn.Dropout(dropout)` | attention dropout, $p$ = 0.2 from `MVHGATConfig` | Section 5.4 |

**Forward pass**, with Fdataset `assoc>drug` shapes (D = 593 drugs, S = 313 diseases):

1. `zs = self.W_src(h_src).view(-1, self.h, self.dh)` → 313 × 4 × 16: every disease's projected state, split into heads. These are both the **keys** (through `a_src`) and the **values**.
2. `zd = self.W_dst(h_dst).view(-1, self.h, self.dh)` → 593 × 4 × 16: every drug's projected state — the **queries** (through `a_dst`). Note that `zd` is used *only* for scoring; the message is built from `zs`.
3. `e = (zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]` → 593 × 313 × 4: the split-$a$ trick of Section 5.7, $e_{ijh} = a^{h\top}_\text{dst}z^h_i + a^{h\top}_\text{src}z^h_j$.
4. `F.leaky_relu(e, 0.2)` — the GAT nonlinearity, slope 0.2 as in the paper. `.masked_fill(~mask[..., None], float("-inf"))` — **masked attention**: non-edges get $-\infty$ in every head (`mask[..., None]` broadcasts the D × S mask over the head axis). `masked_fill` also guarantees safe gradients for empty rows (Section 7.3).
5. `att = torch.softmax(e, dim=1)` — normalise over **sources** (dim 1): for each drug and head, the weights over its diseases sum to 1.
6. `att = torch.nan_to_num(att, nan=0.0)` — empty rows (no visible disease for this drug) become all-zero instead of NaN.
7. `torch.einsum("dsh,shk->dhk", self.drop(att), zs)` — attention dropout, then the weighted sum $\sum_s \alpha_{dsh}\,z^h_s$ per drug and head → 593 × 4 × 16. `.reshape(h_dst.shape[0], -1)` concatenates the heads → 593 × 64 (multi-head **concatenation**, Section 5.3).
8. `return out, mask.any(1)` — the message and the "has at least one neighbour" flag (Section 7).

**What is deliberately *not* in `DenseGAT`:** no bias and no nonlinearity on the output. Those come after the relations are combined, in `HeteroLayer.forward`:

```python
for rel, (dst, src) in self.relations.items():
    out, has = self.gat[rel](h[dst], h[src], graphs[rel])
    if rel in drop_rel:                      # occlusion for explanations
        has = torch.zeros_like(has)
    msgs[dst].append(out)
    valid[dst].append(has)
    names[dst].append(rel)
new, betas = {}, {}
for t in ("drug", "disease"):
    z, beta = self.view_att[t](torch.stack(msgs[t]), torch.stack(valid[t]), uniform)
    new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
    betas[t] = (names[t], beta)
```

* `self.gat` is a `ModuleDict` with **one `DenseGAT` per relation** — separate parameters for `view:chem_cdk`, `view:chem_ecfp`, …, `assoc>drug`, `assoc>disease`. Each relation learns its own notion of a relevant neighbour.
* `self.gat[rel](h[dst], h[src], graphs[rel])` — destination and source embeddings by node type, plus the relation's boolean mask (D × S). For a view relation `dst == src`; for `assoc>drug`, `dst = "drug"` and `src = "disease"`: bipartite attention (Section 8).
* `if rel in drop_rel: has = zeros` — for **occlusion** explanations the relation is declared invalid, so the view attention ignores it. This is the interventional explanation tool of Section 10, implemented by reusing the empty-neighbourhood machinery.
* The messages for each node type are stacked (R × N × 64) and combined by `ViewAttention` (Unit C5) using the `valid` flags. Then skip connection, LayerNorm, ELU, dropout (Unit C3, Section 12.5).

**Parameter counts** (verified with the code in Exercise 6): one `DenseGAT(64, 64, 4)` has $2\cdot64\cdot64 + 2\cdot4\cdot16 = 8{,}320$ parameters; one `HeteroLayer` with 8 relations has 83,584, of which 66,560 are in the eight GATs; the whole Fdataset model has 436,108, more than half of it in the two input projections.

**`ViewAttention` in one sentence** (details in Unit C5): it is *additive* attention (Section 2.3) over relations, with score $s^r_i = q^\top\tanh(P m^r_i + b)$, a masked softmax over relations giving $\beta^r_i$, and the weighted sum $z_i = \sum_r \beta^r_i m^r_i$ — the same "score → softmax → weighted sum" pattern, one level up.

**How it is configured.** `MVHGATConfig`: `hidden=64`, `heads=4`, `layers=2`, `dropout=0.2`; `MVHGAT.__init__` builds `HeteroLayer(relations, hidden, hidden, heads, dropout)` for each layer, so every `DenseGAT` maps 64 → 64. The input projection `self.inp` first brings the 2,092- and 1,532-wide raw features to 64.

---

## 12. Common mistakes and misconceptions

1. **Softmax over the wrong axis.** In a D × S score matrix the normalisation must run over *sources* (`dim=1` in `DenseGAT`). Normalising over destinations makes each *sender's* outgoing weights sum to 1 — a different, usually wrong, model. Check: rows of `att` should sum to 1 (or 0 for empty rows).
2. **Forgetting the mask** (or using `0` instead of $-\infty$). A score of 0 still gets $e^0 = 1$ weight; non-edges must be $-\infty$ *before* the softmax.
3. **NaN from empty neighbourhoods.** Any node with no allowed neighbour gives an all-NaN row (Section 7). Fix it in the forward pass (`nan_to_num`) **and** make sure gradients are clean (`masked_fill`, not an additive $-\infty$ mask — Section 7.3).
4. **Forgetting self-loops in a homogeneous GAT.** Without them a node cannot attend to its own features; GAT includes $i \in \mathcal N(i)$. In the project the views have self-loops via `knn_mask`; the bipartite association relations do not need them because the skip connection carries the node's own state.
5. **Concatenating heads in the final prediction layer.** Average instead (Section 5.3) — unless, as in the project, a separate decoder follows.
6. **Believing GAT attention is query-specific.** Its ranking of neighbours is the same for every query (static attention, Section 6). Use GATv2 if the task needs dynamic attention.
7. **Reading attention weights as explanations** without an intervention test (Section 10).
8. **Assuming attention always beats fixed weights.** On small, noisy data attention adds parameters and variance; it can learn near-uniform weights (Section 5.9) — then it is mostly an expensive mean.
9. **Scaling dot-product attention incorrectly** (dividing by $d$ instead of $\sqrt d$, or not at all for large $d$) — saturated softmax, vanishing gradients (Section 2.5).
10. **Memory blow-ups with dense GATv2.** Its score needs an $N_d \times N_s \times F'$ tensor; the dense GAT trick does not apply (Section 6.2).
11. **Expecting attention rows to sum to 1 during training.** With attention dropout they sum to 1 only in expectation.
12. **Confusing the two halves of $a$.** `a_dst` multiplies the *receiver* (query side), `a_src` the *sender* (key side). Swapping them is a different model (and for bipartite relations the shapes may not even match).

---

## 13. Exercises

★ = conceptual / quick, ★★ = requires working, ★★★ = challenging.

**Exercise 1 (★, conceptual).** In `DenseGAT.forward` for relation `assoc>drug`, identify the queries, keys and values (name the tensors), and explain why keys and values coincide.

<details><summary>Solution</summary>

* **Queries:** the drug side, `zd = W_dst(h_dst)`, entering the score only through `(zd * self.a_dst).sum(-1)`, i.e. $a_\text{dst}^\top z^\text{dst}_i$.
* **Keys:** the disease side, `zs`, entering through `(zs * self.a_src).sum(-1)`, i.e. $a_\text{src}^\top z^\text{src}_j$.
* **Values:** also `zs` — the tensor in `einsum("dsh,shk->dhk", att, zs)`.

In GAT, keys and values share one projection ($W$, here `W_src`): the same transformed neighbour state is used to *decide* the weight (via the scalar $a_\text{src}^\top z_j$) and is *sent* as the message. This saves parameters. A Transformer uses separate $W^K$ and $W^V$, which is more flexible; GAT's key is in effect a one-dimensional projection of the value per head.
</details>

**Exercise 2 (★★, math).** Compute scaled dot-product attention by hand for $q = (2, 0)$, $K = \begin{psmallmatrix}1&1\\0&2\end{psmallmatrix}$, $V = \begin{psmallmatrix}4&0\\0&4\end{psmallmatrix}$. Then check with NumPy.

<details><summary>Solution</summary>

Scores: $q^\top k_1 = 2$, $q^\top k_2 = 0$; divide by $\sqrt2$: $(1.4142, 0)$. Softmax: $e^{1.4142} = 4.1133$; weights $(4.1133, 1)/5.1133 = (0.8044, 0.1956)$. Output: $0.8044(4,0) + 0.1956(0,4) = (3.2177, 0.7823)$.

```python
# file: c4_ex02_scaled_dot.py
import numpy as np
np.set_printoptions(precision=4, suppress=True)
q = np.array([2., 0.]); K = np.array([[1., 1.], [0., 2.]]); V = np.array([[4., 0.], [0., 4.]])
s = K @ q / np.sqrt(2); a = np.exp(s) / np.exp(s).sum()
print("scores", s, "weights", a, "output", a @ V)
```

```text
scores [1.4142 0.    ] weights [0.8044 0.1956] output [3.2177 0.7823]
```
</details>

**Exercise 3 (★★, math + code).** Show that if the LeakyReLU were removed from GAT, the attention weights $\alpha_{ij}$ would not depend on $i$ at all (apart from the mask). Verify numerically.

<details><summary>Solution</summary>

Without the nonlinearity $e_{ij} = c_i + s_j$ with $c_i = a_\text{dst}^\top z_i$, $s_j = a_\text{src}^\top z_j$. Softmax over $j$ is shift invariant, so $\alpha_{ij} = e^{c_i + s_j}/\sum_k e^{c_i+s_k} = e^{s_j}/\sum_k e^{s_k}$, independent of $i$. (With a mask, $i$ influences only *which* $j$ are in the sum.)

```python
# file: c4_ex03_no_leaky.py
import numpy as np
rng = np.random.default_rng(0)
s_dst, s_src = rng.normal(size=4), rng.normal(size=4)
lin = s_dst[:, None] + s_src[None, :]                       # e_ij without the nonlinearity
att = np.exp(lin) / np.exp(lin).sum(1, keepdims=True)
print("rows of attention identical for every i:", np.allclose(att, att[0]))
```

```text
rows of attention identical for every i: True
```

This is why the paper's LeakyReLU is essential: it is the only place the receiver enters. Even so, it can only change the sharpness of $i$'s distribution, not the ranking (Exercise 4).
</details>

**Exercise 4 (★★, proof).** Prove that for a GAT layer, if two receivers $i$ and $i'$ both have senders $j$ and $k$ as neighbours, then $\alpha_{ij} > \alpha_{ik} \iff \alpha_{i'j} > \alpha_{i'k}$. Then give a two-sentence intuition for why GATv2 escapes this.

<details><summary>Solution</summary>

For receiver $i$, $\alpha_{ij}/\alpha_{ik} = \exp(e_{ij} - e_{ik})$ (the normalisers cancel), so $\alpha_{ij} > \alpha_{ik} \iff e_{ij} > e_{ik}$. With $f$ = LeakyReLU (strictly increasing) and $c_i = a_\text{dst}^\top z_i$: $e_{ij} > e_{ik} \iff f(c_i + s_j) > f(c_i + s_k) \iff s_j > s_k$. The last condition does not involve $i$, so the same equivalence holds for $i'$. $\square$

Intuition for GATv2: its score $a^\top\text{LeakyReLU}(W_\text{dst}h_i + W_\text{src}h_j)$ applies the nonlinearity *coordinate-wise* to a mixture of $i$ and $j$ before the final linear read-out, so each hidden unit can switch on or off depending on the *combination* of $i$ and $j$. That makes the score a genuine function of the pair (an MLP), which can rank $j$ above $k$ for one receiver and below for another.
</details>

**Exercise 5 (★★, math).** (a) Derive $\operatorname{Var}(q^\top k) = d$ for independent components with mean 0 and variance 1. (b) Using $\partial\alpha_j/\partial s_k = \alpha_j(\delta_{jk} - \alpha_k)$, show that the gradient of a softmax with $\alpha \approx (1, 0, \dots, 0)$ is nearly zero. (c) Connect (a) and (b) to the $1/\sqrt{d_k}$ in the Transformer.

<details><summary>Solution</summary>

(a) $q^\top k = \sum_c q_ck_c$. Each term has mean $\mathbb E[q_c]\mathbb E[k_c] = 0$ and variance $\mathbb E[q_c^2k_c^2] - 0 = \mathbb E[q_c^2]\mathbb E[k_c^2] = 1$. Terms are independent across $c$, so variances add: $\operatorname{Var} = d$.

(b) Jacobian entries: diagonal $\alpha_j(1 - \alpha_j)$; off-diagonal $-\alpha_j\alpha_k$. With $\alpha_1 \approx 1$, others $\approx 0$: $\alpha_1(1-\alpha_1) \approx 0$, $\alpha_j(1-\alpha_j)\approx 0$ for $j > 1$, and every $\alpha_j\alpha_k \approx 0$. Whatever the upstream gradient, almost nothing flows back to the scores.

(c) With $d_k = 512$ the raw scores have standard deviation $\approx 22.6$, so the largest score typically exceeds the second by many units, the softmax saturates (average top weight 0.951 in Section 2.4's run) and (b) says learning stalls. Dividing by $\sqrt{d_k}$ makes the variance 1 for any $d_k$ (top weight ≈ 0.3 for 10 keys).
</details>

**Exercise 6 (★★, math + code).** Count the parameters of (a) one `DenseGAT(64, 64, heads=4)`, (b) one `HeteroLayer` with the project's 8 relations, (c) the full `MVHGAT` on Fdataset (input widths 2,092 and 1,532, 2 layers, propagation head with 6 views). Verify with code.

<details><summary>Solution</summary>

(a) `W_src`, `W_dst`: $2\times64\times64 = 8{,}192$; `a_src`, `a_dst`: $2\times4\times16 = 128$. Total **8,320**.

(b) 8 GATs: $66{,}560$. `ViewAttention` per node type: `proj` $64\cdot64 + 64 = 4{,}160$, `q` $64$ → $4{,}224$; two types: $8{,}448$. `skip`: $2\times(64\cdot64 + 64) = 8{,}320$. `LayerNorm`: $2\times(64 + 64) = 256$. Total **83,584**.

(c) Input projections: $(2092\cdot64 + 64) + (1532\cdot64 + 64) = 133{,}952 + 98{,}112 = 232{,}064$. Two layers: $167{,}168$. Bilinear `W`: $192^2 = 36{,}864$. Propagation head: `prop_w` 6, `prop_scale` 1, `bias` 1, `gate` 4 = 12. Total **436,108**.

```python
# file: c4_ex06_param_counts.py
import sys
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import DenseGAT, HeteroLayer, MVHGAT

rel = {f"view:{v}": ("drug", "drug") for v in ("chem_cdk", "chem_ecfp", "gene_r")}
rel |= {f"view:{v}": ("disease", "disease") for v in ("pheno_mim", "sem_mondo", "gene_d")}
rel |= {"assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")}
count = lambda m: sum(p.numel() for p in m.parameters())
print("one DenseGAT(64 -> 64, 4 heads):", count(DenseGAT(64, 64, 4, 0.2)))
layer = HeteroLayer(rel, 64, 64, 4, 0.2)
print("one HeteroLayer (8 relations):  ", count(layer))
print("  of which GATs:", count(layer.gat), " view attention:", count(layer.view_att),
      " skip:", count(layer.skip), " LayerNorm:", count(layer.norm))
in_drug, in_dis = 3 * 593 + 313, 3 * 313 + 593             # Fdataset feature widths
model = MVHGAT(rel, in_drug, in_dis, 64, 2, 4, 0.2, n_prop=6)
print("whole MVHGAT on Fdataset:       ", count(model))
print("  input projections:", count(model.inp), "  bilinear W:", model.W.numel())
```

```text
one DenseGAT(64 -> 64, 4 heads): 8320
one HeteroLayer (8 relations):   83584
  of which GATs: 66560  view attention: 8448  skip: 8320  LayerNorm: 256
whole MVHGAT on Fdataset:        436108
  input projections: 232064   bilinear W: 36864
```

Observation: the attention vectors are a tiny fraction (128 per relation); almost all attention-layer parameters are the projections. More than half of the model is the input layer — the price of transductive, node-indexed features.
</details>

**Exercise 7 (★★, coding).** Write a dense, bipartite, multi-head **GATv2** layer with masking and empty-row handling (`forward(h_dst, h_src, mask)`), and check that attention rows sum to 1 (or 0 for an empty row).

<details><summary>Solution</summary>

```python
# file: c4_ex07_gatv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATv2Layer(nn.Module):
    def __init__(self, d_in, d_head, heads):
        super().__init__()
        self.K, self.d = heads, d_head
        self.W_dst = nn.Linear(d_in, heads * d_head, bias=False)
        self.W_src = nn.Linear(d_in, heads * d_head, bias=False)
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.empty(heads, d_head)))
    def forward(self, h_dst, h_src, mask):
        zd = self.W_dst(h_dst).view(-1, self.K, self.d)                  # D,K,d
        zs = self.W_src(h_src).view(-1, self.K, self.d)                  # S,K,d
        g = F.leaky_relu(zd[:, None] + zs[None, :], 0.2)                 # D,S,K,d  (big!)
        e = (g * self.a).sum(-1)                                         # D,S,K
        att = torch.nan_to_num(torch.softmax(e.masked_fill(~mask[..., None], float("-inf")), 1))
        return torch.einsum("dsk,skc->dkc", att, zs).reshape(len(h_dst), -1), att

torch.manual_seed(0)
mask = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.bool)
out, att = GATv2Layer(5, 3, 2)(torch.randn(4, 5), torch.randn(3, 5), mask)
print("output", tuple(out.shape), "row sums per head:\n", att.sum(1).detach().numpy())
```

```text
output (4, 6) row sums per head:
 [[1. 1.]
 [1. 1.]
 [0. 0.]
 [1. 1.]]
```

Destination 2 has no neighbour, so its rows sum to 0 and its message is zero. Note the 4-D intermediate `g` of shape D × S × K × d: for `assoc>drug` in the project that would be $593\times313\times4\times16 \approx 11.9$ million floats (45 MiB) per relation per layer, versus 0.74 million for GAT's scores. To use it in the project you would subclass `DenseGAT`, keep its signature and return `(out, mask.any(1))`.
</details>

**Exercise 8 (★★, coding).** The kNN masks are boolean, so `DenseGAT` ignores the *values* of the similarities. Write a subclass `SimGAT` that adds a learned per-head bonus $\gamma_h S_{ij}$ (with $S$ the similarity matrix) to the score before the mask, and check that with $\gamma = 0$ it reproduces `DenseGAT` exactly.

<details><summary>Solution</summary>

```python
# file: c4_ex08_simgat.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import DenseGAT

class SimGAT(DenseGAT):
    """DenseGAT + a learned bonus gamma * s_ij for the similarity value of the edge."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.gamma = nn.Parameter(torch.zeros(self.h))
    def forward(self, h_dst, h_src, mask, S):
        zs = self.W_src(h_src).view(-1, self.h, self.dh)
        zd = self.W_dst(h_dst).view(-1, self.h, self.dh)
        e = (zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]
        e = F.leaky_relu(e, 0.2) + self.gamma * S[..., None]             # edge-value bonus
        e = e.masked_fill(~mask[..., None], float("-inf"))
        att = torch.nan_to_num(torch.softmax(e, dim=1), nan=0.0)
        out = torch.einsum("dsh,shk->dhk", self.drop(att), zs).reshape(h_dst.shape[0], -1)
        return out, mask.any(1)

torch.manual_seed(0)
S = torch.rand(4, 4); S = (S + S.T) / 2
m = SimGAT(3, 8, 4, 0.0)
H = torch.randn(4, 3); adj = torch.ones(4, 4, dtype=torch.bool)
base, _ = m(H, H, adj, S)
plain = DenseGAT(3, 8, 4, 0.0); plain.load_state_dict({k: v for k, v in m.state_dict().items() if k != "gamma"})
print("gamma = 0 reproduces DenseGAT:", torch.allclose(base, plain(H, H, adj)[0], atol=1e-6))
```

```text
gamma = 0 reproduces DenseGAT: True
```

Why add it after the LeakyReLU? Then $\gamma_h S_{ij}$ acts as a log-prior on the attention weight: $\alpha_{ij} \propto e^{\gamma_h S_{ij}}\,e^{\text{GAT score}}$, i.e. a similarity-proportional prior multiplied by learned evidence. With $\gamma_h > 0$ the head prefers more similar neighbours; the model can learn $\gamma_h \approx 0$ if similarity values carry no extra information beyond the kNN selection. (Because it depends on the pair $(i, j)$, this term also makes attention partly dynamic.) Wiring it into the project would require passing the similarity values alongside the masks — an ablation, not a free improvement.
</details>

**Exercise 9 (★, conceptual).** A GAT hidden layer has 8 heads of width 8. What is its output width with concatenation, and with averaging? Why does the GAT paper average in the output layer but the project never averages?

<details><summary>Solution</summary>

Concatenation: $8\times8 = 64$; averaging: 8. In the paper's output layer each head produces class scores, so averaging lets heads vote on one set of class scores. Concatenating would give 8 separate sets of scores with no natural single prediction. In MV-HGAT no `DenseGAT` is a prediction layer: every one feeds into view attention, LayerNorm and eventually the JK concatenation and the bilinear decoder, so concatenation (more features for the next layer) is the natural choice.
</details>

**Exercise 10 (★★, conceptual + math).** With attention dropout $p = 0.2$ (inverted dropout), what is the expected value of a row sum of `self.drop(att)` during training? What is its effect on the neighbourhood a node "sees", and how does it relate to DropEdge and GraphSAGE sampling?

<details><summary>Solution</summary>

Each entry is kept with probability $1-p$ and scaled by $1/(1-p)$, or zeroed: $\mathbb E[\tilde\alpha_{ij}] = (1-p)\cdot\alpha_{ij}/(1-p) = \alpha_{ij}$. By linearity the expected row sum is $\sum_j\alpha_{ij} = 1$ (actual sums fluctuate around 1). In each training step every node aggregates over a random ~80 % subset of its neighbours (per head), with its weights rescaled. That is stochastic neighbourhood sub-sampling — like DropEdge (which removes edges before the normalisation) and like GraphSAGE's neighbour sampling (which samples a fixed number). All three regularise by preventing the node from relying on any single neighbour. Unlike DropEdge, the remaining weights are not renormalised to sum to 1.
</details>

**Exercise 11 (★★, analysis).** Estimate the memory for the dense score tensors of one MV-HGAT layer on Fdataset (float32, 4 heads, 8 relations), and the number of *real* (edge, head) scores in the `chem_cdk` relation. What fraction of the dense work is wasted, and when would you switch to a sparse implementation?

<details><summary>Solution</summary>

Per relation, one D × S × 4 float32 tensor: drug views $593^2\cdot4\cdot4$ B = 5.37 MiB (×3); disease views $313^2\cdot4\cdot4$ B = 1.49 MiB (×3); `assoc>drug` and `assoc>disease` $593\cdot313\cdot4\cdot4$ B = 2.83 MiB (×2). Total ≈ $16.1 + 4.5 + 5.7 \approx 26$ MiB *per intermediate tensor*; with about five intermediates per relation (raw scores, masked, softmax, dropout output, plus gradients) and 2 layers, a few hundred MiB per forward–backward pass — fine on a GPU.

`chem_cdk` has 4,384 undirected edges → 8,768 directed + 593 self-loops = 9,361 entries, × 4 heads = 37,444 real scores out of 1,406,596: about **2.7 % useful, 97 % wasted** (Section 9 output). Switch to sparse (edge lists with a scatter-softmax, e.g. PyTorch Geometric `GATConv`) when $N^2$ memory becomes the bottleneck — roughly beyond $10^4$ nodes per type, or when using GATv2 (dense memory × $d_h$).
</details>

**Exercise 12 (★★★, critical thinking).** A draft of the paper says: *"MV-HGAT predicted drug X for disease Y because, in the `view:chem_ecfp` relation, drug X attended 60 % to drug Z, a known treatment of Y."* Give four reasons why this sentence is not justified, and describe an analysis that would justify a (weaker) causal claim.

<details><summary>Solution</summary>

1. **Attention is not importance.** If drug Z's message is similar to that of X's other neighbours, moving the 60 % elsewhere would barely change the output (Section 10, first script).
2. **Other paths.** The prediction also depends on the other 7 relations, the skip connection, the view attention $\beta$, the second layer, the JK concatenation, the propagation head and the degree gate. One $\alpha$ inside one relation in one layer is a small part of the computation.
3. **Layer-2 attention refers to mixed states.** Which layer? In layer 2, "attending to Z" means attending to Z's layer-1 state, which already mixes Z's own neighbours.
4. **Heads and static attention.** Each of the 4 heads has its own $\alpha$ (60 % in which head? an average?), and GAT rankings are global per head: Z may get high weight from *every* drug that has Z as a neighbour, regardless of disease Y. Also, attention is computed without reference to disease Y at all — the encoder does not know which pair is being scored.

**A justified analysis (interventional):** use `MVHGATMethod.occlusion` to remove the `view:chem_ecfp` relation *and* its propagation-head term and measure the drop in P(X treats Y); for the specific neighbour, delete the X–Z edge from the `chem_ecfp` mask (or replace Z's features) and re-score. Report the probability change, ideally against a baseline distribution of changes from deleting random edges. Even then, the claim is "the model's prediction depends on Z", not "the drug works because of Z". This is the order the project's documentation prescribes (§4.5: lead with the additive view weights and occlusion; attention is supporting detail).
</details>

**Exercise 13 (★, conceptual).** Why does `DenseGAT` have two projection matrices `W_src` and `W_dst`, when the GAT paper has one $W$? Give one case where two are *necessary* and one where they are merely optional.

<details><summary>Solution</summary>

**Necessary** in bipartite relations whose node types have different input widths or different meanings: in general a drug's features and a disease's features need not even have the same dimension, and a single $W$ would force one linear map onto both spaces. (Inside `HeteroLayer` both are 64-wide after the input projection, so it would *run* with one $W$, but the 64 drug coordinates and 64 disease coordinates mean different things.)

**Optional** in same-type relations (drug ← drug in `view:chem_cdk`): the original GAT ties them, and `DenseGAT` with `W_src = W_dst` reproduces it exactly (Section 5.8). Untied matrices give a slightly more expressive "asymmetric" attention (the receiver's features are mapped differently when used as a query), at the cost of $64^2 = 4{,}096$ extra parameters per relation. Note that `W_dst` only affects the *scores* (through `a_dst`); the message itself is always built from `W_src`.
</details>

---

## 14. Answers to the self-check questions (PREREQUISITES.md, unit C4)

### Q1. Why is `a` split into `a_src` and `a_dst` in our code? (Hint: $a^\top[x \Vert y] = a_1^\top x + a_2^\top y$.)

**Mathematically nothing changes.** The paper's score is $e_{ij} = \text{LeakyReLU}(a^\top[Wh_i \Vert Wh_j])$ with $a \in \mathbb R^{2F'}$. Split $a$ into its first and second halves, $a = [a_1; a_2]$; since the dot product of a concatenation is the sum of the two half dot products,
$$a^\top[z_i\Vert z_j] = a_1^\top z_i + a_2^\top z_j.$$
The code names the halves by the direction of the message: `a_dst` $= a_1$ multiplies the **receiver** (destination) $z_i$, `a_src` $= a_2$ the **sender** (source) $z_j$. Both have shape `(heads, dh)`: one pair per head. The worked example in Section 5.5 verified numerically that the split and the concatenated forms give identical scores.

**Computationally it is what makes dense GAT affordable.** Building the concatenated vector $[z_i \Vert z_j]$ for all $N_d\times N_s$ pairs would cost $O(N_dN_sF')$ memory and time (each pair's vector has $2F' = 128$ entries: $593\times593\times128 \approx 45$ million numbers for one drug view). With the split, the code computes one scalar per node and head — `(zd * self.a_dst).sum(-1)` (D × H) and `(zs * self.a_src).sum(-1)` (S × H) — in $O((N_d + N_s)F')$, then forms all pairwise scores with a broadcast addition `[:, None, :] + [None, :, :]` in $O(N_dN_sH)$ (593 × 593 × 4 ≈ 1.4 million numbers). That is a factor $2d_h = 32$ less, and it is a single vectorised GPU operation. (The same decomposition is used by PyTorch Geometric's `GATConv`.)

**Two further reasons it is natural here:** (i) in bipartite relations (`assoc>drug`) the receiver and sender are *different node types* with different projections (`W_dst`, `W_src`), so it is clearer to keep "receiver part" and "sender part" separate; (ii) the split makes GAT's **static-attention** property visible: since only `a_src · z_j` varies across neighbours and LeakyReLU is monotone, every receiver ranks its neighbours by the same number (Section 6). It also explains why GATv2, which puts the nonlinearity *between* the two parts, cannot use this trick.

### Q2. What happens to a node with no neighbours in a relation?

**Mechanically**, in `DenseGAT.forward`:

1. Its whole row of the mask is `False`, so `masked_fill` sets all its scores (in every head) to $-\infty$.
2. `torch.softmax` over that row computes $0/0$ = **NaN** for every entry.
3. `torch.nan_to_num(att, nan=0.0)` replaces them with **0**, so the einsum produces a **zero message** for that node.
4. The layer returns `mask.any(1)`, which is `False` for this node: **"no information from this relation"**.
5. In `HeteroLayer`, the flag goes into `ViewAttention` as `valid`; the relation's view score for this node is set to $-\infty$ before the softmax over relations, so its weight $\beta = 0$ and the node's combined message comes only from relations that do have neighbours. The node's new state is $\text{ELU}(\text{LayerNorm}(z + W_\text{skip}h))$, so its own previous state still flows through the skip connection.
6. **Gradients stay finite** because the mask is applied with `masked_fill`, whose backward zeroes the gradient at every masked position — the NaN gradients of the all-$-\infty$ softmax row never reach a parameter. An additive $-\infty$ mask would give NaN gradients (Section 7.3, demonstrated).

**When it happens in the project.** Never in the similarity views: `knn_mask` always adds the self-loop, so an "isolated" node (153 diseases in `gene_d`, 81 drugs in `gene_r` on Fdataset) still has one neighbour — itself — and its message is its own projected state $W_\text{src}h_i$ (valid but uninformative; view attention can learn to discount it). It happens in the **association relations**, which have no self-loops: for the ~10 % of diseases whose links are all hidden in each training epoch (`cold_frac`, cold-start practice), for nodes whose few links were all hidden by `drop_edge`, and at test time for a held-out disease in leave-one-disease-out evaluation (and for any drug or disease with no visible link in a fold). It is also *used deliberately*: `HeteroLayer`'s `drop_rel` argument sets `has` to all-False to switch a relation off for **occlusion** explanations.

**Why the design is right:** returning zero *and* a validity flag means "no evidence" is represented as *absence* (excluded from the view softmax), not as a fake zero-vector message that would dilute the other relations' messages — which is what a naive mean over relations would do.

---

## 15. Summary and cheat sheet

**One-paragraph summary.** Attention is a differentiable soft lookup: score each key against a query, softmax the scores into weights, and return the weighted average of the values. Additive (Bahdanau) attention scores with a small MLP; dot-product attention with an inner product, scaled by $1/\sqrt{d_k}$ to keep the softmax out of saturation; multi-head attention runs several in parallel. Self-attention over a set is message passing on a complete graph; masking turns it into attention over graph neighbours. GAT computes $e_{ij} = \text{LeakyReLU}(a^\top[Wh_i\Vert Wh_j])$, normalises over each node's neighbours, and averages $Wh_j$ with those weights; heads are concatenated in hidden layers and averaged in the output layer; attention dropout samples neighbourhoods. Splitting $a = [a_\text{dst}; a_\text{src}]$ makes dense scoring cheap but also reveals GAT's static attention (one global ranking of neighbours), which GATv2 fixes by applying $a$ after the nonlinearity. Empty neighbourhoods produce NaN and must be handled in both the forward and the backward pass. Attention weights are not, by themselves, faithful explanations.

| Item | Formula / fact |
|---|---|
| Attention | $\alpha = \text{softmax}(s(q, k_j))$, output $\sum_j\alpha_jv_j$ |
| Softmax | shift invariant; $-\infty$ → weight 0; $\partial\alpha_j/\partial s_k = \alpha_j(\delta_{jk} - \alpha_k)$ |
| Additive (Bahdanau) | $v^\top\tanh(W_qq + W_kk)$ |
| Dot / general (Luong) | $q^\top k$ / $q^\top Wk$ |
| Scaled dot (Transformer) | $\text{softmax}(QK^\top/\sqrt{d_k})V$; $\operatorname{Var}(q^\top k) = d_k$ |
| Multi-head | $[\text{head}_1\Vert\dots\Vert\text{head}_H]W^O$, $d_k = d_\text{model}/H$ |
| Transformer block | $\text{LN}(X + \text{MHA}(X))$, $\text{LN}(X' + \text{FFN}(X'))$, + positional encodings |
| GAT score | $e_{ij} = \text{LeakyReLU}_{0.2}(a_\text{dst}^\top Wh_i + a_\text{src}^\top Wh_j)$ |
| GAT weights / update | $\alpha_{ij} = \text{softmax}_{j\in\mathcal N(i)\cup\{i\}}(e_{ij})$; $h_i' = \sigma(\sum_j\alpha_{ij}Wh_j)$ |
| Heads | hidden: concat ($KF'$); output: average |
| Regularisation | dropout on inputs and on $\alpha$ (p = 0.6 in the paper; 0.2 in the project) |
| Static attention (GAT) | ranking of neighbours by $a_\text{src}^\top Wh_j$, same for every receiver |
| GATv2 | $e_{ij} = a^\top\text{LeakyReLU}(W_\text{dst}h_i + W_\text{src}h_j)$ — dynamic; no split trick |
| Empty row | all $-\infty$ → NaN → `nan_to_num` → 0 message + `valid=False`; use `masked_fill` for clean gradients |
| Complexity | dense $O(N_dN_sH)$ scores; sparse $O(\lvert E\rvert H)$; GATv2 dense $O(N_dN_sF')$ |
| Project | `DenseGAT`: 4 heads × 16, separate `W_src`/`W_dst`, `a_src`/`a_dst` (heads × 16), softmax over sources, concat heads, returns `(msg, mask.any(1))`; one per relation per layer (16 in total) |
| Explanations | attention ≠ importance; prefer interventions (occlusion) and additive terms |

---

## 16. Further resources (all links checked)

**Attention and Transformers**
* [Jay Alammar, "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/) — the best visual walk-through of self-attention, Q/K/V and multi-head attention. *Free.*
* [Jay Alammar, "Visualizing A Neural Machine Translation Model (seq2seq with attention)"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — Bahdanau/Luong attention, animated. *Free.*
* [3Blue1Brown, "Attention in transformers, step-by-step" (video)](https://www.youtube.com/watch?v=eMlx5fFNoYc) — geometric intuition for queries, keys and values. *Free.*
* [Lilian Weng, "Attention? Attention!"](https://lilianweng.github.io/posts/2018-06-24-attention/) — a compact survey of attention variants with formulas. *Free.*
* [*Dive into Deep Learning*, chapter "Attention Mechanisms and Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html) — textbook treatment with runnable code. *Free.*
* [Harvard NLP, "The Annotated Transformer"](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper, line by line, as working PyTorch. *Free.*
* [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — builds self-attention from first principles in code. *Free.*
* [Chaitanya Joshi, "Transformers are Graph Neural Networks" (The Gradient)](https://thegradient.pub/transformers-are-graph-neural-networks/) — the bridge between Sections 4 and 5. *Free.*

**Graph attention**
* [Petar Veličković, "Graph attention networks" (author's blog page)](https://petar-v.com/GAT/) — the first author's own explanation with figures. *Free.*
* [Petar Veličković, "Theoretical Foundations of Graph Neural Networks" (video)](https://www.youtube.com/watch?v=uF53xsT7mjc) — places GAT among convolutional, attentional and message-passing GNNs. *Free.*
* [Stanford CS224W lecture videos (YouTube playlist)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn) — the "GNN design space" lecture covers GAT; course site: [web.stanford.edu/class/cs224w](https://web.stanford.edu/class/cs224w/). *Free.*
* [William L. Hamilton, *Graph Representation Learning*](https://www.cs.mcgill.ca/~wlh/grl_book/) — Ch. 5 covers neighbourhood attention in the GNN framework. *Free pre-print.*

**Primary papers** (free on arXiv)
* [Bahdanau, Cho & Bengio 2015, "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) — additive attention.
* [Luong, Pham & Manning 2015, "Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025) — dot/general/concat scores.
* [Vaswani et al. 2017, "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — scaled dot-product, multi-head, Transformer.
* [Veličković et al. 2018, "Graph Attention Networks"](https://arxiv.org/abs/1710.10903) — read Section 2.1 with this chapter's Section 5 beside it.
* [Brody, Alon & Yahav 2022, "How Attentive are Graph Attention Networks?" (GATv2)](https://arxiv.org/abs/2105.14491) — static vs dynamic attention, DictionaryLookup.
* [Knyazev, Taylor & Amer 2019, "Understanding Attention and Generalization in Graph Neural Networks"](https://arxiv.org/abs/1905.02850) — when graph attention helps, and how hard it is to learn good attention.
* [Wang et al. 2019, "Heterogeneous Graph Attention Network" (HAN)](https://arxiv.org/abs/1903.07293) — the two-level attention that `ViewAttention` adapts (next unit).
* [Bronstein, Bruna, Cohen & Veličković 2021, *Geometric Deep Learning* proto-book](https://arxiv.org/abs/2104.13478) — attentional GNNs as one of three "flavours" of GNN layer.

**Attention as explanation**
* [Jain & Wallace 2019, "Attention is not Explanation"](https://arxiv.org/abs/1902.10186).
* [Wiegreffe & Pinter 2019, "Attention is not not Explanation"](https://arxiv.org/abs/1908.04626) — read both, in this order.

---

## 17. Glossary

* **Additive attention** — score computed by a small MLP of query and key, $v^\top\tanh(W_qq + W_kk)$ (Bahdanau).
* **Attention dropout** — dropout applied to the normalised attention weights; each node aggregates over a random subset of neighbours during training.
* **Attention weight ($\alpha_{ij}$)** — the softmax-normalised importance node $i$ assigns to neighbour $j$; non-negative, summing to 1 over $j$.
* **Bipartite attention** — attention where senders and receivers are different node sets/types (e.g. drugs attending over diseases); rectangular mask.
* **Dot-product attention** — score $q^\top k$; **scaled** version divides by $\sqrt{d_k}$.
* **Dynamic attention** — the ranking of keys can differ between queries (GATv2, Transformer attention).
* **GAT** — Graph Attention Network: $e_{ij} = \text{LeakyReLU}(a^\top[Wh_i\Vert Wh_j])$, masked softmax over neighbours, weighted sum.
* **GATv2** — $e_{ij} = a^\top\text{LeakyReLU}(W[h_i\Vert h_j])$; computes dynamic attention.
* **Head** — one independent attention mechanism; multi-head attention runs several and concatenates or averages.
* **Key** — the representation of an item used to compute its match with a query.
* **LayerNorm** — normalises each vector to zero mean and unit variance across its features, then rescales; used in Transformers and in `HeteroLayer`.
* **LeakyReLU** — $x$ for $x>0$, $0.2x$ otherwise (slope 0.2 in GAT); strictly increasing.
* **Masked attention / masked softmax** — setting disallowed scores to $-\infty$ so they receive zero weight; restricts attention to graph neighbours.
* **Multi-head attention** — several attention heads in parallel with separate projections, combined by concatenation (+ linear map) or averaging.
* **Positional encoding** — vectors added to token embeddings so that a permutation-equivariant Transformer can use order.
* **Query** — the representation of the item doing the looking-up (the receiving node).
* **Saturation (softmax)** — almost one-hot output; gradients nearly vanish.
* **Self-attention** — queries, keys and values all computed from the same set of items.
* **Softmax** — $e^{s_j}/\sum_l e^{s_l}$; turns scores into a probability distribution.
* **Split attention vector** — $a = [a_\text{dst}; a_\text{src}]$, using $a^\top[x\Vert y] = a_\text{dst}^\top x + a_\text{src}^\top y$ to compute scores with broadcasting.
* **Static attention** — every query ranks the keys in the same order (GAT).
* **Transformer** — architecture of stacked multi-head self-attention and feed-forward blocks with residual connections and LayerNorm.
* **Valid flag (`mask.any(1)`)** — `DenseGAT`'s per-node indicator that a relation provided at least one neighbour; used to exclude empty relations in view attention.
* **Value** — the representation an item contributes to the output when attended to.
* **View attention ($\beta$)** — MV-HGAT's node-specific attention over relations (Unit C5).
