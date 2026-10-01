# Unit C6: Link Prediction

> **Track C: Graphs and graph neural networks** · Chapter 14 of the learning track
>
> **Prerequisites:** C5 *Heterogeneous graphs and HAN* (the encoder this unit plugs into), C3 *GCN* (transductive vs inductive), C4 *GAT*, C2 *Random walks and propagation* (Katz, random walk with restart, MBiRW), B4 *Matrix factorisation* (implicit feedback, negative sampling), B2 *Evaluation and imbalanced data* (AUC, AUPR), A3 *Probability* (Bernoulli likelihood, cross-entropy).
>
> **Estimated study time:** 10–12 hours. About 4 h of theory (section 2), 2.5 h running the code (section 4), 1.5 h on the project walkthrough (section 5), and 2–3 h of exercises.

---

## Learning objectives

When you finish this unit you should be able to do the following.

1. **Frame** drug repositioning as link prediction on a bipartite graph: what is observed, what is predicted, what counts as a negative, and why the problem is *positive–unlabelled*.
2. **Compute by hand** common neighbours, Jaccard, Adamic–Adar, resource allocation, preferential attachment and Katz scores, and **explain** why common neighbours is always zero for a drug–disease pair.
3. **Describe** DeepWalk and node2vec in a paragraph each, and **place** them in the encoder–decoder framework.
4. **Write down and compare** dot-product, bilinear and MLP decoders, and the knowledge-graph scoring functions TransE, DistMult, ComplEx and RotatE, including which relation patterns each can represent.
5. **Derive** BCE, BPR and margin losses, **relate** BPR to AUC, and **derive** how negative sampling shifts the model's logits (calibration offset).
6. **Explain** edge leakage (supervising on edges that are also message-passing inputs), **demonstrate** it in code, and **compare** the fixes: disjoint message/supervision edges, DropEdge, and this project's hidden-link supervision.
7. **Distinguish** transductive and inductive splits, warm-start and cold-start evaluation, and **explain** how cold-start practice, the propagation head and the degree gate make the project's model work for unseen diseases.
8. **Summarise** SEAL and the labeling trick, and **say** when pair-level (subgraph) methods beat node-wise GNNs.
9. **Design** an evaluation protocol for link prediction that avoids leakage and inflated metrics, and **audit** `evaluation.py` against it.
10. **Walk through** `MVHGATMethod.fit_predict` and `MVHGAT.forward` line by line, and **interpret** the project's ablation and cold-start numbers honestly.

---

## Notation used in this chapter

| Symbol | Meaning |
|---|---|
| $A\in\{0,1\}^{n_r\times n_d}$ | drug × disease association matrix ($n_r=593$, $n_d=313$ in Fdataset) |
| $A_\text{train}$ | links visible in the current CV fold (test links set to 0) |
| $A_\text{vis}$ (code: `Am`) | links visible to the model *in the current epoch* |
| $E_\text{msg}$, $E_\text{sup}$ | message-passing edges and supervision edges |
| $S_r$, $S_d$ | drug–drug and disease–disease similarity matrices (views) |
| $K_v$ | row-normalised k-NN kernel of view $v$ (`data.knn_kernel`) |
| $h_i$, $H_i$ | node embedding (encoder output); in the project $H_i\in\mathbb{R}^{192}$ |
| $s(i,j)$ or $\ell_{ij}$ | score / logit of pair $(i,j)$ |
| $\sigma(x)$ | logistic sigmoid $1/(1+e^{-x})$ |
| $\mathcal{N}(x)$, $k_x$ | neighbour set and degree of node $x$ |
| $\rho$ | negative sampling rate (fraction of candidate negatives used per step) |

---

## 1. Motivation: why this matters for *this* project

### 1.1 Repositioning *is* link prediction

The project's data is a bipartite graph: 593 drug nodes, 313 disease nodes, and 1,933 "treats" edges (Fdataset). The question "which existing drug might treat disease $j$?" is the question "**which edges are missing** from this graph?". Every method in `methods.py`, from MBiRW's random walk to MV-HGAT, returns a full $593\times313$ score matrix. `evaluation.py` then asks how well those scores rank the hidden true links above the unknown pairs.

So the design decisions of this unit are the design decisions of the project:

| Decision | Where it lives | This unit's section |
|---|---|---|
| how to score a (drug, disease) pair from embeddings | `MVHGAT.forward` (bilinear + propagation head + degree gate) | 2.4, 2.5, 5.3 |
| which pairs count as negatives during training | `fit_predict`: `neg_pool`, `neg_ratio` | 2.6 |
| which loss to use | `F.binary_cross_entropy_with_logits` | 2.7 |
| which links the model may *see* while being trained on them | `fit_predict`: `hide`, `Am`, `sup` | 2.8 |
| how to handle a disease with no known drugs | `cold_frac`, `prop_head`, `degree_gate` | 2.10 |
| how to evaluate fairly | `evaluation.py::kfold_splits`, `run_lodo` | 2.12 |

### 1.2 Two moments in the project's history that this unit explains

**"Training on visible links is a trap."** The first version of the GNN was trained, as most tutorials do, on the known links that were *also edges in its input graph*. Its validation AUC was **0.71**, worse than a simple random walk. Changing one thing, so that each epoch a random share of links is hidden and the loss is computed **only on the hidden ones**, raised it to **0.92** (`HOW_IT_WORKS.md` §10). Section 2.8 explains why; section 4.5 reproduces the effect on a toy graph.

**"Cold start has to be trained for."** A disease with no known drugs (leave-one-disease-out testing) got a validation AUPR of **0.055** from the plain GNN, against a random level of about 0.01 and MBiRW's **0.174**. Cold-start practice alone raised it to about 0.12. Together with a propagation head and a degree gate it reached about **0.15**. Section 2.10 explains each ingredient; section 4.6 reproduces the pattern.

Both lessons are general. They are among the most common ways link-prediction papers go wrong.

---

## 2. Core theory

### 2.1 Framing the task

**Intuition.** You see part of a network and want to guess the rest. In social networks: who will become friends? In biology: which protein pairs interact? Here: which drug–disease pairs are undiscovered indications?

**Definition (link prediction).** Let $\mathcal{G}=(\mathcal{V},\mathcal{E})$ be the true graph and $\mathcal{E}_\text{obs}\subset\mathcal{E}$ the observed edges. A link predictor is a function $s:\mathcal{V}\times\mathcal{V}\to\mathbb{R}$, computed from $\mathcal{G}_\text{obs}=(\mathcal{V},\mathcal{E}_\text{obs})$ and any node features, such that pairs in $\mathcal{E}\setminus\mathcal{E}_\text{obs}$ (the missing true links) score higher than pairs not in $\mathcal{E}$. It is fundamentally a **ranking** problem. We rarely need calibrated probabilities; we need the right pairs at the top of each list.

**Bipartite version.** Here $\mathcal{V}=\mathcal{V}_r\cup\mathcal{V}_d$ and edges only go between the two parts. Candidate pairs are the $n_rn_d=185{,}609$ cells of $A$.

**Three properties that shape everything else.**

1. **Positive–unlabelled (PU) data.** A 1 in $A$ is a verified indication. A 0 means *not known*, not *does not work*. Some zeros are undiscovered positives: precisely the ones we hope to find. Training must treat zeros as "probably negative". Evaluation must accept that some "false positives" are true discoveries, which makes reported metrics **pessimistic** (`HOW_IT_WORKS.md` §6, assumption 1).
2. **Extreme imbalance.** About 1.04% of cells are 1 in Fdataset. A random ranker has AUC 0.5 and AUPR ≈ 0.0104. Any AUPR must be read against that (Unit B2).
3. **The graph is both input and target.** In node classification, labels and graph are separate. In link prediction, the edges we predict are *the same kind of object* as the edges we propagate messages along. This is the root of edge leakage (2.8).

**Missing-at-random or not?** Random edge splits assume test links are a random sample of all links. Real discoveries are not random: new indications tend to involve well-studied drugs, popular disease areas, and recent years. Time-based splits ("train on indications known before 2015, test on later ones") test this more realistically. The project uses random splits (warm start) and leave-one-disease-out (cold start), the two conventions of this benchmark literature, and adds a cross-dataset test (Fdataset → Cdataset) as an external check.

### 2.2 Heuristic methods: scoring pairs from topology alone

Heuristics need no training. They are the baselines every learned method must beat, and several of them reappear inside learned models.

#### Local heuristics (paths of length 2)

For nodes $x,y$ with neighbour sets $\mathcal{N}(x)$, $\mathcal{N}(y)$ and degrees $k_x$, $k_y$:

| Heuristic | Formula | Idea |
|---|---|---|
| **Common neighbours (CN)** | $\lvert\mathcal{N}(x)\cap\mathcal{N}(y)\rvert$ | friends of friends become friends (triadic closure) |
| **Jaccard** | $\dfrac{\lvert\mathcal{N}(x)\cap\mathcal{N}(y)\rvert}{\lvert\mathcal{N}(x)\cup\mathcal{N}(y)\rvert}$ | CN, normalised for how many neighbours the pair has |
| **Adamic–Adar (AA)** | $\displaystyle\sum_{z\in\mathcal{N}(x)\cap\mathcal{N}(y)}\frac{1}{\log k_z}$ | a *rare* shared neighbour is more telling than a hub |
| **Resource allocation (RA)** | $\displaystyle\sum_{z\in\mathcal{N}(x)\cap\mathcal{N}(y)}\frac{1}{k_z}$ | like AA, with a stronger hub penalty |
| **Preferential attachment (PA)** | $k_x\cdot k_y$ | rich get richer: high-degree nodes gain links |

*Why $1/\log k_z$?* Adamic and Adar argued for weighting a shared feature by its rarity. A shared neighbour with degree 2 links only these two nodes, which is specific evidence; a shared neighbour with degree 1,000 links everyone, so it says little. The logarithm softens the penalty. RA can be derived from a resource-flow model: $x$ sends one unit of resource, split equally among its neighbours, and each neighbour $z$ passes its share on equally to its $k_z$ neighbours. The amount reaching $y$ is $\sum_z\frac{1}{k_x}\frac{1}{k_z}$, which is RA up to the constant $1/k_x$.

#### Global heuristics (all path lengths)

**Katz index.** Count *all* paths (walks) between $x$ and $y$, discounting long ones geometrically:

$$
\text{Katz}(x,y)=\sum_{\ell=1}^{\infty}\beta^\ell\,(A^\ell)_{xy}.
$$

*Closed form (derivation).* The matrix series $\sum_{\ell\ge0}(\beta A)^\ell$ is a Neumann series. It converges if and only if the spectral radius satisfies $\beta\lambda_{\max}(A)<1$. Then, for $S=\sum_{\ell=0}^\infty(\beta A)^\ell$, we have $S-\beta AS=I$, so $S=(I-\beta A)^{-1}$, and removing the $\ell=0$ term gives

$$
\boxed{\;\text{Katz}=(I-\beta A)^{-1}-I,\qquad 0<\beta<1/\lambda_{\max}(A).\;}
$$

Small $\beta$ makes Katz behave like CN (length-2 paths dominate); larger $\beta$ brings in long-range structure. **Rooted PageRank** (random walk with restart, Unit C2) and **SimRank** ("two nodes are similar if their neighbours are similar") are other global indices. MBiRW is a bipartite, similarity-weighted relative of random walk with restart.

#### Worked example 1: heuristics by hand

Graph on six nodes with edges A–B, A–C, B–C, B–D, C–D, D–E, E–F:

```
    A
   / \
  B───C
   \ /
    D───E───F
```

Degrees: $k_A=2$, $k_B=3$, $k_C=3$, $k_D=3$, $k_E=2$, $k_F=1$.

*Pair (A, D).* $\mathcal{N}(A)=\{B,C\}$, $\mathcal{N}(D)=\{B,C,E\}$.

- CN $=|\{B,C\}|=2$.
- Jaccard $=2/|\{B,C,E\}|=2/3=0.667$.
- AA $=\frac{1}{\ln3}+\frac{1}{\ln3}=2/1.0986=1.820$.
- RA $=\frac13+\frac13=0.667$.
- PA $=2\times3=6$.
- Katz with $\beta=0.1$. Walks of length 2: A–B–D, A–C–D (2). Length 3: A–B–C–D, A–C–B–D (2). Length 4: 12. So $\text{Katz}\approx 0.01\cdot2+0.001\cdot2+0.0001\cdot12+\dots=0.0232+\dots$; the exact value is $0.02350$. Is $\beta=0.1$ allowed? $\lambda_{\max}=2.655$, so we need $\beta<0.377$. Yes.

*Pair (A, E).* No common neighbour, so CN = Jaccard = AA = RA = 0, but there are 2 walks of length 3 (A–B–D–E, A–C–D–E). Katz $=0.00237>0$. **Local indices cannot see beyond distance 2; Katz can.** For (A, F), at distance 4, Katz is 0.00024.

*Pair (B, E).* One common neighbour, D, with $k_D=3$: CN 1, Jaccard $1/|\{A,C,D,F\}|=0.25$, AA $1/\ln3=0.910$, RA $1/3$, PA 6.

Section 4.1 reproduces this table.

#### The bipartite trap

In a bipartite graph a drug's neighbours are all diseases and a disease's neighbours are all drugs, so **a drug and a disease never share a neighbour**. CN, Jaccard, AA and RA are identically 0 for every (drug, disease) pair. Paths between the two sides have **odd** length. The shortest informative path is drug → disease → drug → disease, and the number of such walks is

$$
(AA^\top A)_{ij}=\sum_{k,l}A_{il}A_{kl}A_{kj},
$$

which reads as "for every disease $l$ that drug $i$ treats, and every drug $k$ that also treats $l$, does $k$ treat $j$?". That is **item-based collaborative filtering**: "drugs that share indications with me also treat $j$". (Note that $AA^\top A$ counts *walks*, which can reuse an edge, so existing links get extra credit from back-and-forth walks.)

The project's **propagation head** replaces one of the $A$ factors by a learned-free *similarity* kernel:

$$
P_v^{\text{drug}}=K_vA\quad(\text{"drugs similar to } i \text{ treat } j\text{"}),\qquad
P_u^{\text{dis}}=(K_uA^\top)^\top=AK_u^\top\quad(\text{"} i \text{ treats diseases similar to } j\text{"}).
$$

These are length-2 paths in the *heterogeneous* graph: drug →(similar)→ drug →(treats)→ disease. They are the guilt-by-association heuristic, one per view, and they are exactly what the MBiRW random walk iterates. In section 4.3 this simple heuristic beats every learned model on AUPR in our toy world, just as it is the strongest single component in the project's ablation.

### 2.3 Embedding methods: DeepWalk and node2vec (briefly)

**Idea.** Learn a vector $z_v\in\mathbb{R}^d$ for every node such that nodes that co-occur on short random walks get similar vectors. Then score a pair by comparing vectors.

- **DeepWalk (Perozzi et al. 2014).** Generate uniform random walks of fixed length from every node. Treat each walk as a "sentence" of node "words", and train the word2vec **skip-gram** model: maximise $\log P(v_{t+c}\mid v_t)$ for context nodes within a window. A softmax over all nodes is expensive, so it is approximated by hierarchical softmax or **negative sampling** (Mikolov et al. 2013): maximise $\log\sigma(z_u^\top z_v)+\sum_{k=1}^{K}\mathbb{E}_{n\sim P_n}\log\sigma(-z_u^\top z_n)$, with noise distribution $P_n(v)\propto\deg(v)^{3/4}$.
- **node2vec (Grover & Leskovec 2016).** Biases the walk with two parameters. Having just moved from $t$ to $v$, the unnormalised probability of stepping to $x$ is $1/p$ if $x=t$ (go back), $1$ if $x$ is a neighbour of $t$ (stay close), and $1/q$ otherwise (move outward). Low $q$ gives DFS-like walks that capture *communities* (homophily); high $q$ gives BFS-like walks that capture *structural roles*. For link prediction node2vec combines the two endpoint vectors with a binary operator (average, **Hadamard** $z_u\odot z_v$, $L_1$ or $L_2$ distance) and trains a logistic regression on the result. The Hadamard product worked best in the paper.

**Limitations relevant here.** These are **shallow, transductive** embeddings: one free vector per node, and no way to embed a node that was not in the training graph or has no edges. A cold-start disease with no links has no random walks through "treats" edges, so its vector is never trained. They also ignore node features (our six similarity views) unless the views are added as edges. Unit B4's matrix factorisation $A\approx UV^\top$ is the same family: it is equivalent to factorising a co-occurrence matrix, which is what skip-gram does implicitly.

### 2.4 The encoder–decoder framework

Hamilton's book organises all of these methods (and GNNs) into one template:

$$
\underbrace{\text{ENC}:\;v\mapsto h_v\in\mathbb{R}^d}_{\text{node} \to \text{vector}},\qquad
\underbrace{\text{DEC}:\;(h_u,h_v)\mapsto s(u,v)\in\mathbb{R}}_{\text{pair of vectors} \to \text{score}},\qquad
\mathcal{L}=\sum_{(u,v)\in\mathcal{D}}\ell\big(s(u,v),\,y_{uv}\big).
$$

| Encoder | What $h_v$ depends on | Inductive? | Example |
|---|---|---|---|
| shallow lookup | a free parameter vector per node | no | MF, DeepWalk, node2vec |
| feature MLP | the node's own features | yes, if features exist | NIMC-style "inductive matrix completion" |
| GNN | the node's features **and** its neighbourhood | partly (needs features + edges) | GCN, GAT, HAN, MV-HGAT |

#### Decoders

**Dot product.** $s(u,v)=h_u^\top h_v$. It is parameter-free and fast: all pairs at once is one matrix product, $H_rH_d^\top$. It is symmetric, $s(u,v)=s(v,u)$, which is harmless for a bipartite graph whose two sides have separate embeddings, but wrong for a directed relation within one node type.

**Bilinear.** $s(u,v)=h_u^\top Wh_v$, with $W\in\mathbb{R}^{d\times d}$ learned. Writing it out,

$$
h_u^\top Wh_v=\sum_{a,b}h_{u,a}\,W_{ab}\,h_{v,b},
$$

so $W_{ab}$ says how much *dimension $a$ of the drug* should match *dimension $b$ of the disease*. That lets the drug and disease spaces differ: "chemical dimension 3 of a drug matches mechanism dimension 7 of a disease" (`HOW_IT_WORKS.md` §4.3). Special cases are $W=I$ (dot product) and diagonal $W$ (DistMult). A non-symmetric $W$ gives an asymmetric score. With one $W_r$ per relation this is **RESCAL** (section 2.5). The project uses a full $192\times192$ $W$ on jumping-knowledge embeddings: 36,864 parameters (`MVHGAT.W`).

**MLP.** $s(u,v)=\text{MLP}([h_u\Vert h_v])$, or on $h_u\odot h_v$. This is the most flexible option in principle. Its costs: all-pairs scoring needs $n_rn_d$ forward passes instead of one matrix product; it is easier to overfit; and, perhaps surprisingly, MLPs are poor at learning a plain dot product. Rendle et al. (RecSys 2020) showed that a well-tuned dot product beats MLP-based "neural collaborative filtering". In section 4.3 the MLP decoder is no better than the dot product.

#### Pair-level features in the decoder

A decoder can also take inputs that are not functions of $h_u$ and $h_v$ alone: heuristic scores such as CN or Katz, or the project's per-view propagation scores $P_v[i,j]$. Node-wise encoders cannot represent some pair-level structure (section 2.11), so adding such features is a cheap and effective fix. The project's decoder is

$$
\ell_{ij}=g(i,j)\cdot H_i^\top WH_j\;+\;\sum_v w_v\,P_v[i,j]\;+\;b ,
$$

a bilinear GNN term plus an additive heuristic head, with a degree-dependent gate $g$ (section 2.10).

### 2.5 Knowledge-graph embeddings (in brief)

A knowledge graph is a set of triples $(h,r,t)$: *(aspirin, treats, headache)*, *(aspirin, inhibits, PTGS2)*. KG embedding methods are shallow encoders plus **relation-specific decoders** $f_r(h,t)$:

| Model | Score $f_r(h,t)$ (higher = more plausible) | Embedding space | Notes |
|---|---|---|---|
| **TransE** (Bordes et al. 2013) | $-\lVert h+r-t\rVert$ | $\mathbb{R}^d$ | relation = translation |
| **RESCAL** (Nickel et al. 2011) | $h^\top M_rt$ | $\mathbb{R}^d$, $M_r\in\mathbb{R}^{d\times d}$ | full bilinear; many parameters |
| **DistMult** (Yang et al. 2015) | $\langle h,r,t\rangle=\sum_k h_kr_kt_k$ | $\mathbb{R}^d$ | diagonal RESCAL; **always symmetric** |
| **ComplEx** (Trouillon et al. 2016) | $\operatorname{Re}\big(\sum_k h_kr_k\bar t_k\big)$ | $\mathbb{C}^d$ | conjugation breaks symmetry |
| **RotatE** (Sun et al. 2019) | $-\lVert h\circ r-t\rVert$, $\lvert r_k\rvert=1$ | $\mathbb{C}^d$ | relation = rotation |

**Relation patterns.** A *symmetric* relation (r(x,y) ⇒ r(y,x), e.g. "is similar to") is natural for DistMult, which can model nothing else. TransE can only model it with $r=0$, which collapses $h$ and $t$. An *antisymmetric* relation ("treats": a drug treats a disease, never the reverse) suits TransE, ComplEx and RotatE, but not DistMult. *Inversion* ("treats" vs "treated-by") and *composition* ("inhibits ∘ is-involved-in ⇒ may-treat") are handled by RotatE (rotations compose by adding angles) and partly by TransE (translations add). This is why RotatE is a sensible default for rich biomedical KGs (Decagon-style multi-relational drug graphs).

**Worked example 2 (checked in section 4.4).** Two-dimensional TransE with $h=[1,0]$, $r=[0,1]$, $t=[1,1]$: $h+r-t=0$, so $f=0$ (perfect), while the reverse triple $(t,r,h)$ has $f=-\lVert[0,2]\rVert=-2$. DistMult with $h=[1,2]$, $r=[0.5,1]$, $t=[2,1]$: $f=1\cdot0.5\cdot2+2\cdot1\cdot1=3$ in *both* directions. DistMult cannot say that diseases do not treat drugs. ComplEx with $h=1+i$, $r=-i$, $t=1-i$: $h\,r\,\bar t=(1+i)(-i)(1+i)=(1-i)(1+i)=2$, so $f=2$; reversed, $t\,r\,\bar h=(1-i)(-i)(1-i)=(-1-i)(1-i)=-2$. RotatE with $h=1$, $r=e^{i\pi/2}=i$, $t=i$: $hr-t=0$, so $f=0$; reversed, $|i\cdot i-1|=2$, so $f=-2$.

**Connection to the project.** The project has one predicted relation (treats), so a single bilinear form ($W$) is RESCAL with one relation. R-GCN's link-prediction model is exactly "R-GCN encoder + DistMult decoder". If you extended the project to predict several relation types (indication, contraindication, side effect), you would give the decoder one $W_r$ per relation, or use a ComplEx/RotatE-style decoder.

### 2.6 Negative sampling

**Why sample?** Positives are the known links. "Negatives" must be drawn from the unknown cells (PU data). With 185k cells you *could* use every unknown cell every step, and the baselines NIMCGCN and LAGCN do: `weighted_bce` up-weights positives by the negative/positive ratio. But for large graphs that is impossible, and even here sampling has advantages: fresh negatives each epoch act as regularisation, and the positive/negative balance becomes a tunable hyper-parameter.

**Strategies.**

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **uniform** | pick unknown cells uniformly | simple; unbiased over pairs | most negatives are trivially easy |
| degree-based (word2vec $\propto k^{3/4}$) | prefer high-degree endpoints | counters popularity bias (the model cannot win by scoring hubs high) | can over-penalise genuinely promiscuous drugs |
| **hard negatives** | pick high-scoring unknowns (by the current model or a heuristic) | sharper decision boundary near the top of the ranking | in PU data, the hardest "negatives" are the most likely *undiscovered positives*: false-negative contamination |
| corruption (KGs) | replace $h$ or $t$ of a true triple | per-positive negatives | may create true triples |
| type-constrained | only pairs of valid types | no nonsense pairs | needs a schema |

**False negatives.** If a fraction $\pi$ of unknown cells are actually true indications, then a uniformly sampled negative is wrong with probability $\pi$, which is small (most drugs do not treat most diseases). A hard negative selected *because* the model scores it highly is wrong with a probability far above $\pi$. That is why hard negatives tend to hurt in drug repositioning, and in section 4 (Exercise 8) they reduce AUPR from 0.251 to 0.195.

**Ratio.** `neg_ratio = 2` in `MVHGATConfig` means two sampled negatives per supervised positive per epoch, re-sampled every epoch (`torch.randint`, with replacement). Higher ratios expose the model to more of the negative space per step but shift the training base rate further from reality, and above a point they add little (Exercise 8: ratios 1–10 give AUPR 0.23–0.25).

**What sampling does to probabilities: a derivation.** Suppose a pair with features $x$ is truly positive with probability $p(x)$. During training, positives enter the loss at rate $\rho_+$ (fraction of positives used per epoch) and negatives at rate $\rho_-$. The pointwise BCE minimiser outputs the *training-distribution* posterior

$$
\sigma(\ell^*(x))=\frac{\rho_+p(x)}{\rho_+p(x)+\rho_-(1-p(x))}
\;\;\Longrightarrow\;\;
\ell^*(x)=\underbrace{\log\frac{p(x)}{1-p(x)}}_{\text{true log-odds}}+\log\frac{\rho_+}{\rho_-}.
$$

The learned logit is the true logit **plus a constant**. Rankings, and therefore AUC and AUPR, are unaffected, but $\sigma(\ell)$ is *not* a calibrated probability. In the project (fold 0 of Fdataset 5-fold CV), about 28% of the 1,546 training positives are supervised per epoch ($\rho_+\approx0.28$), and $2\times433\approx866$ negatives are drawn from a pool of 146,940 ($\rho_-\approx0.0059$). The offset is $\log(0.28/0.0059)\approx3.9$, so the model's odds are inflated about 47-fold. **Read the output of `fit_predict` as a score, not as a probability.**

### 2.7 Loss functions

Let $\ell^+$ be the logit of a positive pair and $\ell^-$ of a negative one.

**Binary cross-entropy (pointwise).** From the Bernoulli likelihood (Unit A3):

$$
\mathcal{L}_\text{BCE}=-\log\sigma(\ell^+)-\log\big(1-\sigma(\ell^-)\big)=\operatorname{softplus}(-\ell^+)+\operatorname{softplus}(\ell^-).
$$

Gradients: $\partial\mathcal{L}/\partial\ell^+=\sigma(\ell^+)-1$ and $\partial\mathcal{L}/\partial\ell^-=\sigma(\ell^-)$. BCE cares about **absolute** levels: it wants positives high *and* negatives low, each on its own. The project uses BCE via `F.binary_cross_entropy_with_logits`, which is numerically stable.

**Bayesian Personalised Ranking, BPR (pairwise; Rendle et al. 2009).** Model the probability that the positive outranks the negative as $\sigma(\ell^+-\ell^-)$, and maximise its log-likelihood:

$$
\mathcal{L}_\text{BPR}=-\log\sigma(\ell^+-\ell^-).
$$

*Relation to AUC.* AUC is the fraction of (positive, negative) pairs ranked correctly, $\frac1{|P||N|}\sum\mathbb{1}[\ell^+>\ell^-]$. Replace the step function by the smooth $\sigma$ and take logs, and you get BPR: **BPR is a differentiable surrogate of AUC**. It is invariant to adding a constant to all scores; it only cares about **differences**.

**Margin / hinge (pairwise; TransE).**

$$
\mathcal{L}_\text{margin}=\max\big(0,\;\gamma-\ell^++\ell^-\big).
$$

It is zero once the positive beats the negative by the margin $\gamma$, after which that pair stops contributing gradient. It yields sparse gradients and is not probabilistic.

**Softmax / InfoNCE (listwise).** $-\log\frac{e^{\ell^+}}{e^{\ell^+}+\sum_k e^{\ell^-_k}}$ treats one positive and $K$ negatives as a $(K{+}1)$-way classification. It is common in contrastive learning and large-scale recommendation.

**Worked example 3 (checked in section 4.4).**

| $\ell^+$ | $\ell^-$ | BCE | BPR | margin ($\gamma=1$) | comment |
|---|---|---|---|---|---|
| 2.0 | 0.5 | 1.1010 | 0.2014 | 0 | correct order; BCE still unhappy because $\ell^-=0.5$ means "62% positive" |
| 0.3 | 0.8 | 1.7255 | 0.9741 | 1.5 | wrong order: every loss penalises |
| 10.0 | 9.0 | 9.0002 | 0.3133 | 0 | correct order, terrible calibration: BCE is huge, BPR is small, margin is zero |

The third row is the essence: **BCE punishes miscalibration, BPR and margin do not**. For a pure ranking task, pairwise losses match the evaluation metric more closely. BCE is simpler to combine with an additive propagation head and a bias (the project's choice), and in practice the two are close (Exercise 8: AUC 0.868 vs 0.877, AUPR 0.251 vs 0.238).

### 2.8 Edge leakage and how to prevent it

This is the most important section of the unit for the project.

**The setting.** A GNN encoder computes $h=\text{ENC}(X,E_\text{msg})$ from the features and the message-passing edges. The decoder is trained on supervision pairs $E_\text{sup}$ (positives) plus sampled negatives. At test time we score pairs $(i,j)\notin E_\text{msg}$: a test link is, by construction, never in the input graph.

**The leak.** The naive recipe uses $E_\text{msg}=E_\text{sup}=$ all training links. Then for every positive $(i,j)$ in the loss, **the edge $i$–$j$ itself is in the input**:

- through message passing, $j$'s state flows directly into $h_i$ and vice versa, so $h_i^\top Wh_j$ can be large simply because $h_i$ *contains* $h_j$;
- through features, if node features include the visible link row (the project's `feat_assoc=True`), then $x_i$ has a 1 in column $j$, and the input layer can learn a weight that fires exactly when "column $j$ of my link row is 1 and I am being paired with $j$";
- through degree and normalisation constants, which are computed from the same edges.

The cheapest way to reduce the loss is to learn **"score high if the edge is already there"**. This is an *edge detector*, not an edge *predictor*. Training AUC goes to ~1. At test time the edge is never there, so the detector is useless and the test inputs come from a different distribution (test positives look like training negatives: no direct edge). Formally, the training distribution of (inputs, label) differs from the test distribution in exactly the feature that the model learned to rely on: a classic **shortcut**.

**Fix 1: disjoint message and supervision edges (CS224W's recipe).** Split the training links once into *message edges* (used for propagation and features) and *supervision edges* (used only in the loss), e.g. 70/30. Then every supervised positive is absent from the input, as at test time. The cost: 30% of links never inform the encoder, and 70% never provide gradient. Validation edges are held out from both.

**Fix 2: DropEdge (Rong et al. 2020).** Each epoch, remove a random fraction $p$ of edges from the message graph (and renormalise). DropEdge was proposed for *node classification*, as data augmentation and to slow over-smoothing. Applied to link prediction while still supervising **all** positives, it only *partly* closes the leak: each positive is hidden in a fraction $p$ of epochs and visible in the rest. In those visible epochs the shortcut is still rewarded.

**Fix 3: hidden-link supervision (this project's choice): a fresh disjoint split every epoch.** Each epoch:

1. hide a random 20% of the training links (plus all links of ~10% of diseases, for cold-start practice);
2. build the message graph, the features, the propagation scores and the degrees from the **visible** links only;
3. compute the loss on the **hidden** links (plus negatives) only.

This is Fix 1 with the split re-drawn every epoch, so over many epochs *every* link serves both as a message edge and as a supervision edge, but **never in the same epoch**. Code (`fit_predict`):

```python
hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge      # 20% hidden
...
Am[pos[~hide]] = True                                            # visible links only
if c.supervise_hidden:
    sup = pos[hide]                                              # supervise ONLY the hidden ones
```

The demo in section 4.5 shows the three regimes on a toy graph:

| Regime | train-edge AUC | test AUC | test AUPR | score drop when own edge removed |
|---|---|---|---|---|
| naive (see and supervise all) | 0.998 | 0.826 | 0.172 | **0.589** |
| DropEdge, supervise all | 0.995 | 0.858 | 0.200 | 0.133 |
| hidden-link supervision | 0.951 | **0.879** | **0.287** | −0.122 |

The last column is a direct **edge-detector test**: take a training link, remove *only that link* from the input, and see how much its score falls. The naive model's score collapses by 0.59: it was looking at the edge itself. With hidden-link supervision the score does not depend on the edge being present (it even rises slightly, because the model has only ever been asked to score links that were hidden; section 6, mistake 4).

**Other leakage channels to close** (Unit E1 goes further):

- *Features:* `features(Am)` builds the link-row features from the **visible** links each epoch, not from `A_full`.
- *Propagation head:* $P_v=K_vA_\text{vis}$, and $K_v$ has a **zero diagonal** (`knn_kernel` does `np.fill_diagonal(W, 0.0)`), so $P_v[i,j]=\sum_{k\neq i}K_v[i,k]A_\text{vis}[k,j]$ never includes $A[i,j]$. The head is leak-free by construction, even without hiding.
- *Degrees:* the degree gate uses `degrees(Am)`, the visible degrees.
- *Negatives:* `neg_pool` excludes test cells (`neg_mask`) and all training positives, so a test link is never used as a negative and never as a positive.
- *Similarity views:* none is computed from $A$ (`HOW_IT_WORKS.md` §6, assumption 7).

**SEAL's version of the same idea.** SEAL (2.11) extracts a subgraph around each target pair and **removes the target edge from its own subgraph** during training, for exactly the same reason.

### 2.9 Transductive and inductive splits

| Split | What is hidden | What the model has seen | Realistic question | Project |
|---|---|---|---|---|
| **transductive edge split** (warm start) | a random subset of links | every node, with most of its links | "what other diseases might this well-studied drug treat?" | `run_kfold` (5- and 10-fold CV) |
| **leave-one-node-out** (cold start, transductive node) | *all* links of one node | the node itself and its features / similarity edges | "a newly characterised disease: which drugs?" | `run_lodo` (leave-one-disease-out) |
| **inductive node split** | entire nodes (features, edges) absent in training | nothing about the test nodes | "a brand-new compound, never in the graph" | not supported: the model must be retrained with the new node added |
| **pair cold start** | both endpoints new | neither | "a new drug for a new disease" | not evaluated |
| **time split** | links discovered after a date | everything before | "will tomorrow's indications be found?" | not evaluated (cross-dataset test is the nearest proxy) |

**Transductive vs inductive models** (Unit C3). MV-HGAT is transductive: its graph and features are fixed matrices over the 593 + 313 known entities. In LODO the held-out disease *is* in the graph with its similarity edges and features; only its links are hidden. This is a "transductive cold start": harder than warm start, easier than truly inductive.

**Why LODO is much harder than k-fold.** In k-fold CV a test pair $(i,j)$ usually has *other* links of both $i$ and $j$ in training (about 80% of them). The collaborative signal ("drugs with indications like $i$'s treat $j$") is available. In LODO disease $j$ has **zero** visible links. Its link row is empty, its `assoc` relation is invalid, its drug-side propagation scores are all zero ($K_rA[:,j]=0$ since column $j$ is empty), and its degree is 0. Only the disease-side similarity views can carry information. That is why Fdataset LODO AUPRs are around 0.15 while warm-start AUPRs are around 0.5.

### 2.10 Cold-start link prediction

**Why a GNN trained in warm start fails on cold nodes.** Two reasons.

1. *Missing input:* the node's collaborative evidence (its edges) is absent, so its embedding is built from features and similarity relations alone.
2. *Distribution shift:* during warm-start training *every* disease has visible links, so the encoder and decoder never see a disease with an empty link row and an invalid `assoc` relation. Facing one at test time is out-of-distribution. Recommender-systems research calls the fix *training with dropped-out preferences*: simulate cold users during training (DropoutNet, Volkovs et al. 2017).

**The project's three remedies** (`MVHGATConfig`):

**(a) Cold-start practice (`cold_frac = 0.1`).** Each epoch, about 10% of diseases lose **all** their links (in the graph, the features, the propagation scores and the degrees), and those links join the supervision set:

```python
cold = torch.rand(n_d, device=DEVICE) < c.cold_frac
hide |= cold[pos % n_d]          # pos indexes the flattened matrix; pos % n_d is the disease column
```

The model is now trained on exactly the LODO situation in every epoch: "score drugs for a disease you cannot see any links of".

**(b) Propagation head (`prop_head = True`).** It adds $\sum_v w_vP_v[i,j]$ to the logit, one term per view, with $w_v=\operatorname{softplus}(\theta_v)\cdot5\ge0$ (`MVHGAT.view_weights`). For a cold disease $j$ the drug-view terms vanish, but the disease-view terms $P_u[i,j]=\sum_lK_u[j,l]A[i,l]$, "does drug $i$ treat diseases similar to $j$?", remain. They need no training and no edges of $j$. This is the guilt-by-association prior, made explicit and **inductive by construction**: it only needs $j$'s similarity row.

**(c) Degree gate (`degree_gate = True`).** The GNN term is multiplied by

$$
g(i,j)=\operatorname{sigmoid}\big(a+b\log(1+\deg_i)\big)\cdot\operatorname{sigmoid}\big(c+d\log(1+\deg_j)\big),
$$

with $a,b,c,d$ learned (initialised to $0,1,0,1$). With visible degree 0 the gate is small and the score leans on the propagation head; with many links it approaches 1 and the collaborative GNN term counts fully. This is a two-expert **mixture with a learned gate on degree**. In the toy demo of section 4.6, the learned disease factor is 0.59 at degree 0, 0.73 at 1, 0.88 at 5 and 0.96 at 20.

**Evidence.** Toy demo (section 4.6), 16 cold diseases, metrics pooled over their columns and averaged over 3 seeds:

| Model | AUC | AUPR |
|---|---|---|
| disease-view propagation only (no learning) | 0.798 | 0.214 |
| GNN with hidden-link supervision | 0.819 | 0.173 |
| + cold-start practice | 0.877 | 0.258 |
| + propagation head (no cold practice) | 0.854 | 0.196 |
| + propagation head + cold practice | 0.881 | 0.261 |
| + degree gate (full recipe) | 0.882 | 0.268 |

The project, on its validation split: plain GNN AUPR 0.055 → about 0.12 with cold-start practice alone → about 0.15 (AUC 0.85) with practice, propagation head and degree gate. **MBiRW, a simple random walk, gets 0.174**: still better in cold start. In both cases **cold-start practice is the biggest single step**, and in both cases a *non-learned* propagation scorer is a strong reference that the learned model barely matches. Report this honestly.

**Other cold-start approaches.** Meta-learning (treat each cold entity as a few-shot task); content-to-embedding mappers (learn $f:\text{features}\to\text{shallow embedding}$ for warm nodes, then apply it to cold ones); **IGMC** (Zhang & Chen 2020), an inductive matrix completion method that scores each pair from its local enclosing subgraph with no node identities, and therefore transfers to unseen users and items.

### 2.11 SEAL and the labeling-trick perspective (briefly)

**SEAL (Zhang & Chen 2018).** To score pair $(x,y)$:

1. extract the $h$-hop **enclosing subgraph** around $x$ and $y$ (with the target edge removed during training);
2. **label** each node by its distances to $x$ and $y$ (Double-Radius Node Labeling, DRNL), so the GNN knows which nodes are the targets and where everyone else sits relative to them;
3. run a GNN on the subgraph and read out a graph-level vector, then classify it.

The paper proves a "$\gamma$-decaying heuristic" theorem: many global heuristics (Katz, rooted PageRank, SimRank) can be approximated well from a small $h$-hop subgraph, with error shrinking exponentially in $h$. So SEAL can *learn* heuristics instead of hand-picking one.

**Why labeling matters: the labeling trick (Zhang et al. 2021).** A node-wise GNN computes one embedding per node and then combines two of them. If nodes $v_2$ and $v_3$ are *automorphic* (symmetric positions in the graph, e.g. two identical leaves), they receive identical embeddings. Then for any $v_1$ the pairs $(v_1,v_2)$ and $(v_1,v_3)$ get identical scores, even when $v_2$ is two hops from $v_1$ and $v_3$ is ten hops away. Node-wise embeddings cannot represent *the relationship between* two nodes; they only represent each node on its own. Labeling the two target nodes before running the GNN breaks the symmetry. The paper proves that, with a sufficiently expressive GNN, this suffices to learn the most expressive structural representations of node pairs.

**Relevance here.** SEAL needs one subgraph and one GNN pass *per candidate pair*: 185,609 pairs per fold for full-matrix evaluation. That is affordable but heavy. The project's cheaper alternative is to feed **pair-level heuristic features** to the decoder (the propagation head): the same spirit of "give the model information about the pair, not just about each node". The ablation shows this pair-level information is the single most valuable component (section 5.6).

### 2.12 Designing an evaluation protocol

A link-prediction result is only as good as its protocol. A checklist:

1. **What is a test positive?** A held-out true link, removed from *every* input (graph, features, propagation, degrees, similarity views).
2. **What is a test negative?** Ideally *all* unknown pairs in the fold (the project's choice: each fold's test set is its 1s plus its share of all 0s, about 36.7k negatives against 387 positives in Fdataset). Sampling a few negatives per positive (e.g. 1:1) inflates AUPR enormously and makes results incomparable. Random negatives are also *easy*. HeaRT (Li et al. 2023) argues for per-positive *hard* negatives as a more realistic test.
3. **Never train on test negatives.** `neg_mask` removes each fold's test zeros from the negative pool, so the model cannot learn "these specific cells are 0".
4. **Metrics.** Report AUPR alongside AUC (Unit B2). For "top-k candidates per disease" use per-disease precision@k, recall@k, hits@k or MRR (as in OGB). State whether metrics are **pooled** (all pairs together) or **averaged per entity**. `run_lodo` reports both pooled AUC/AUPR and the mean per-disease AUC, and they can differ a lot (section 4.7: pooled 0.805 vs per-disease mean 0.729).
5. **Random baselines.** AUPR of a random ranker = positive rate (0.0104 for Fdataset; 0.056 for the cold columns of the toy demo).
6. **Splits.** Same folds for all methods (seeded `kfold_splits`); repeated CV with mean ± std; **paired** comparisons per fold.
7. **Tuning.** Hyper-parameters chosen on a validation split, never on test folds (`05_sensitivity.py`). For full rigour, use nested CV.
8. **Protocol matches the claim.** Warm-start CV supports claims about well-studied entities; LODO supports claims about new diseases. Do not use one to argue the other.
9. **Fair baselines.** Same information (e.g. "benchmark similarities only" for comparisons with methods that use only those) and the same splits. Report re-implementations as such.
10. **An external check.** Train on one dataset and test on links found only in another (`07_cross_dataset.py`), or validate top predictions against independent evidence (CTD curated links, ClinicalTrials.gov).

---

## 3. Putting it together: the project's training loop as an algorithm

Here is `MVHGATMethod.fit_predict` written as an algorithm, so that every piece of section 2 has a place. Inputs: $A_\text{train}$ (test links already zeroed), `neg_mask` (cells allowed as negatives), and the config.

```
P   ← flat indices of training positives          (≈1,546 in an Fdataset 5-fold training fold)
Neg ← flat indices with neg_mask ∧ A_train = 0     (≈146,940; excludes test cells and all positives)
for epoch = 1 … 600:
    hide ← Bernoulli(0.2) for each p ∈ P                         # hidden-link supervision  (2.8)
    cold ← Bernoulli(0.1) for each disease;  hide ← hide ∨ cold[col(p)]   # cold-start practice (2.10)
    A_vis ← training links with hide = False
    graph['assoc'] ← A_vis ;  X ← [similarity rows ‖ A_vis rows]          # inputs from VISIBLE links only
    P_v ← K_v · A_vis for each view;  deg ← row/column sums of A_vis       # propagation head, degree gate
    ℓ ← g(deg) ⊙ (H_r W H_dᵀ) + Σ_v w_v P_v + b                          # decoder (2.4)
    S ← P[hide]                                       # supervise only the hidden positives
    N ← |S| × 2 indices drawn uniformly from Neg      # negative sampling, ratio 2 (2.6)
    loss ← BCE(ℓ[S], 1) + BCE(ℓ[N], 0)  (mean)       # pointwise loss (2.7)
    Adam step
test time: A_vis ← A_train (all training links visible); return σ(ℓ) for every cell
```

Two quantities you should be able to derive (section 2.6): each epoch supervises about $1-0.8\times0.9=28\%$ of the positives (≈433) against ≈866 negatives, so the outputs are scores with a constant logit offset of about $+3.9$ relative to calibrated log-odds.

---

## 4. Code: building it from scratch

All examples use NumPy, PyTorch and scikit-learn only, run on CPU, and finish in seconds to under a minute. Run them with the project interpreter, e.g.

```powershell
& "C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe" heuristics.py
```

Put all files of this section in the same folder: later scripts import `toy_world.py`. The outputs shown were produced by exactly this code (PyTorch 2.14, NumPy 2.5, scikit-learn 1.9, CPU). Other versions or thread counts may change the last digit of trained results, but not the conclusions.

### 4.1 Heuristic link scores (worked example 1)

Save as `heuristics.py`:

```python
import numpy as np

nodes = ["A", "B", "C", "D", "E", "F"]
edges = [("A","B"), ("A","C"), ("B","C"), ("B","D"), ("C","D"), ("D","E"), ("E","F")]
idx = {v: i for i, v in enumerate(nodes)}
Adj = np.zeros((6, 6))
for u, v in edges:
    Adj[idx[u], idx[v]] = Adj[idx[v], idx[u]] = 1
deg = Adj.sum(1)

def common_neighbours(x, y): return Adj[x] @ Adj[y]
def jaccard(x, y):
    inter = Adj[x] @ Adj[y]; union = ((Adj[x] + Adj[y]) > 0).sum()
    return inter / union
def adamic_adar(x, y):
    z = np.flatnonzero(Adj[x] * Adj[y]); return (1 / np.log(deg[z])).sum()
def resource_allocation(x, y):
    z = np.flatnonzero(Adj[x] * Adj[y]); return (1 / deg[z]).sum()
def pref_attachment(x, y): return deg[x] * deg[y]

beta = 0.1                                     # must be < 1 / largest eigenvalue
lam = np.linalg.eigvalsh(Adj).max()
Katz = np.linalg.inv(np.eye(6) - beta * Adj) - np.eye(6)
print(f"largest eigenvalue = {lam:.3f}, so beta must be < {1/lam:.3f}")
print("paths of length 2, 3 between A and D:",
      int(np.linalg.matrix_power(Adj, 2)[0, 3]), int(np.linalg.matrix_power(Adj, 3)[0, 3]))
print("paths of length 2, 3 between A and E:",
      int(np.linalg.matrix_power(Adj, 2)[0, 4]), int(np.linalg.matrix_power(Adj, 3)[0, 4]))
print(f"{'pair':6s}{'CN':>5s}{'Jacc':>7s}{'AA':>7s}{'RA':>7s}{'PA':>5s}{'Katz':>9s}")
for a, b in [("A","D"), ("A","E"), ("B","E"), ("A","F")]:
    x, y = idx[a], idx[b]
    print(f"{a}-{b}   {common_neighbours(x,y):5.0f}{jaccard(x,y):7.3f}{adamic_adar(x,y):7.3f}"
          f"{resource_allocation(x,y):7.3f}{pref_attachment(x,y):5.0f}{Katz[x,y]:9.5f}")

# ---- the bipartite trap: drugs r1..r3, diseases s1..s3 ----------------------------
A = np.array([[1, 1, 0],     # r1 treats s1, s2
              [1, 0, 0],     # r2 treats s1
              [0, 1, 1]])    # r3 treats s2, s3
Big = np.block([[np.zeros((3, 3)), A], [A.T, np.zeros((3, 3))]])
print("common neighbours of (r2, s2) in the bipartite graph:", int(Big[1] @ Big[3 + 1]))
print("3-path counts A A^T A (drug x disease) =\n", (A @ A.T @ A).astype(int))
```

Expected output:

```text
largest eigenvalue = 2.655, so beta must be < 0.377
paths of length 2, 3 between A and D: 2 2
paths of length 2, 3 between A and E: 0 2
pair     CN   Jacc     AA     RA   PA     Katz
A-D       2  0.667  1.820  0.667    6  0.02350
A-E       0  0.000  0.000  0.000    4  0.00237
B-E       1  0.250  0.910  0.333    6  0.01187
A-F       0  0.000  0.000  0.000    2  0.00024
common neighbours of (r2, s2) in the bipartite graph: 0
3-path counts A A^T A (drug x disease) =
 [[3 3 1]
 [2 1 0]
 [1 3 2]]
```

**Read the output.** The table reproduces worked example 1. Katz is the only index that ranks (A, E) above (A, F), because it counts length-3 and length-4 walks. In the bipartite graph, drug r2 and disease s2 share no neighbour. The 3-walk matrix gives r2–s2 a score of 1 through the path r2–s1–r1–s2: "r2 shares an indication with r1, and r1 treats s2".

### 4.2 A toy drug–disease world

All later demos use the same synthetic world: 120 drugs and 80 diseases with a hidden "mechanism" vector each. A drug treats a disease when their mechanisms align, plus noise. Similarity views are noisy cosines of the mechanism vectors, the analogue of chemical and phenotype similarity. Density is 4% (Fdataset: 1%). It is small, so experiments run in seconds. Save as `toy_world.py`:

```python
"""A miniature drug-disease world with a hidden 'mechanism' structure."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def make_world(n_r=120, n_d=80, n_mech=4, density=0.04, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_mech, 8))
    U = centres[rng.integers(n_mech, size=n_r)] + 0.6 * rng.normal(size=(n_r, 8))  # drugs
    V = centres[rng.integers(n_mech, size=n_d)] + 0.6 * rng.normal(size=(n_d, 8))  # diseases
    unit = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True)
    affinity = 10 * unit(U) @ unit(V).T + rng.gumbel(size=(n_r, n_d))  # noisy "does it work?"
    A = (affinity > np.quantile(affinity, 1 - density)).astype(np.float32)
    def cos_sim(Z):                                              # a noisy similarity view
        Z = Z + 0.6 * rng.normal(size=Z.shape)
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return (Z @ Z.T).astype(np.float32)
    return A, cos_sim(U), cos_sim(V)

def split(A, test_frac=0.2, seed=0):
    """Warm-start split like evaluation.kfold_splits: hide 20% of 1s and 20% of 0s."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
    rng.shuffle(pos); rng.shuffle(neg)
    tp, tn = pos[: int(test_frac * len(pos))], neg[: int(test_frac * len(neg))]
    A_tr = A.copy().ravel(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
    neg_mask = (A == 0).ravel(); neg_mask[tn] = False             # test 0s never used
    return A_tr, neg_mask.reshape(A.shape), tp, tn

def knn_kernel(S, k=10):
    """Row-normalised weights of each node's k most similar OTHER nodes (as data.py)."""
    W = S.copy(); np.fill_diagonal(W, -np.inf)
    keep = np.argsort(-W, axis=1)[:, :k]
    K = np.zeros_like(S); rows = np.arange(len(S))[:, None]
    K[rows, keep] = np.clip(S[rows, keep], 0, None)
    return K / np.clip(K.sum(1, keepdims=True), 1e-12, None)

def evaluate(scores, tp, tn):
    y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    s = scores.ravel()[np.r_[tp, tn]]
    return roc_auc_score(y, s), average_precision_score(y, s)

if __name__ == "__main__":
    A, Sr, Sd = make_world()
    print("A:", A.shape, "links:", int(A.sum()), f"density: {A.mean():.3f}")
    print("diseases with no links:", int((A.sum(0) == 0).sum()))
    A_tr, neg_mask, tp, tn = split(A)
    print("test positives:", len(tp), " test negatives:", len(tn))
    print(f"random ranker: AUC ~0.5, AUPR ~ positive rate = {len(tp)/(len(tp)+len(tn)):.3f}")
```

Expected output:

```text
A: (120, 80) links: 384 density: 0.040
diseases with no links: 6
test positives: 76  test negatives: 1843
random ranker: AUC ~0.5, AUPR ~ positive rate = 0.040
```

`split` mirrors `evaluation.kfold_splits`: 20% of the 1s **and** 20% of the 0s form the test set, and `neg_mask` forbids the test 0s as training negatives. `knn_kernel` mirrors `data.knn_kernel`: each row keeps the 10 most similar *other* nodes, row-normalised, with a zero diagonal.

### 4.3 Encoder–decoder link prediction with negative sampling

Two encoders (shallow free embeddings vs. an MLP over each node's similarity row) × three decoders (dot, bilinear, MLP). All are trained with BCE and two uniformly sampled negatives per positive, re-sampled each epoch, and compared with two heuristics. Save as `encoder_decoder.py`:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from toy_world import make_world, split, knn_kernel, evaluate

A, Sr, Sd = make_world()
A_tr, neg_mask, tp, tn = split(A)
n_r, n_d = A.shape

# ---------------- encoders ------------------------------------------------------
class Shallow(nn.Module):                     # one free vector per node (= matrix factorisation)
    def __init__(self, d=16):
        super().__init__()
        self.r = nn.Parameter(0.1 * torch.randn(n_r, d))
        self.s = nn.Parameter(0.1 * torch.randn(n_d, d))
    def forward(self):
        return self.r, self.s

class FeatureMLP(nn.Module):                  # embed each node from its similarity row
    def __init__(self, d=16):
        super().__init__()
        self.Xr, self.Xd = torch.tensor(Sr), torch.tensor(Sd)
        self.fr = nn.Sequential(nn.Linear(n_r, 32), nn.ReLU(), nn.Linear(32, d))
        self.fd = nn.Sequential(nn.Linear(n_d, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self):
        return self.fr(self.Xr), self.fd(self.Xd)

# ---------------- decoders: all-pairs logits (n_r x n_d) -------------------------
class Dot(nn.Module):
    def forward(self, Hr, Hd): return Hr @ Hd.T
class Bilinear(nn.Module):
    def __init__(self, d=16):
        super().__init__(); self.W = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
    def forward(self, Hr, Hd): return Hr @ self.W @ Hd.T
class MLPDec(nn.Module):
    def __init__(self, d=16):
        super().__init__(); self.f = nn.Sequential(nn.Linear(2 * d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, Hr, Hd):
        pairs = torch.cat([Hr[:, None, :].expand(-1, n_d, -1), Hd[None, :, :].expand(n_r, -1, -1)], -1)
        return self.f(pairs).squeeze(-1)

def train(enc, dec, neg_ratio=2, epochs=300, seed=0):
    torch.manual_seed(seed)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=0.01, weight_decay=1e-4)
    pos = torch.tensor(np.flatnonzero(A_tr.ravel() > 0))
    pool = torch.tensor(np.flatnonzero((neg_mask & (A_tr == 0)).ravel()))
    for _ in range(epochs):
        logits = dec(*enc()).reshape(-1)
        neg = pool[torch.randint(len(pool), (len(pos) * neg_ratio,))]    # uniform negatives
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        loss = F.binary_cross_entropy_with_logits(torch.cat([logits[pos], logits[neg]]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(dec(*enc())).numpy()

if __name__ == "__main__":
    # ---------------- heuristic baselines (no training) ---------------------------
    walk3 = A_tr @ A_tr.T @ A_tr                              # drug-disease-drug-disease walks
    prop = knn_kernel(Sr) @ A_tr + (knn_kernel(Sd) @ A_tr.T).T  # similarity propagation
    for name, S in [("3-walk count", walk3), ("kNN propagation", prop)]:
        print(f"{name:28s} AUC {evaluate(S, tp, tn)[0]:.3f}  AUPR {evaluate(S, tp, tn)[1]:.3f}")

    for enc_name, Enc in [("shallow", Shallow), ("similarity-MLP", FeatureMLP)]:
        for dec_name, Dec in [("dot", Dot), ("bilinear", Bilinear), ("MLP", MLPDec)]:
            torch.manual_seed(0)
            auc, aupr = evaluate(train(Enc(), Dec()), tp, tn)
            print(f"{enc_name + ' + ' + dec_name:28s} AUC {auc:.3f}  AUPR {aupr:.3f}")
```

Expected output:

```text
3-walk count                 AUC 0.770  AUPR 0.276
kNN propagation              AUC 0.858  AUPR 0.347
shallow + dot                AUC 0.737  AUPR 0.154
shallow + bilinear           AUC 0.700  AUPR 0.133
shallow + MLP                AUC 0.746  AUPR 0.147
similarity-MLP + dot         AUC 0.879  AUPR 0.277
similarity-MLP + bilinear    AUC 0.864  AUPR 0.205
similarity-MLP + MLP         AUC 0.881  AUPR 0.228
```

**Read the output.**

1. **Side information beats free embeddings.** Shallow embeddings (matrix factorisation) only see 308 training links for 200 nodes, and AUPR stays around 0.15. Encoding each node from its similarity row reaches AUC 0.86–0.88. This is why every serious drug-repositioning model uses similarity views.
2. **The decoder matters less than the encoder.** Dot, bilinear and MLP land within a few points of each other, and the MLP decoder is not better than the dot product (compare Rendle et al. 2020).
3. **The non-learned propagation heuristic has the best AUPR (0.347).** It puts the true links at the very top more reliably than any of the learned models, even though "similarity-MLP + MLP" has a slightly higher AUC. This is the same phenomenon that led the project to add the propagation head to its decoder: a learned model should be built *on top of* this signal, not be asked to rediscover it.

### 4.4 Losses and knowledge-graph scoring functions (worked examples 2 and 3)

Save as `losses_kg.py`:

```python
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))
def bce(s_pos, s_neg):    return -np.log(sig(s_pos)) - np.log(1 - sig(s_neg))
def bpr(s_pos, s_neg):    return -np.log(sig(s_pos - s_neg))
def margin(s_pos, s_neg, gamma=1.0): return max(0.0, gamma - s_pos + s_neg)

for s_pos, s_neg in [(2.0, 0.5), (0.3, 0.8), (10.0, 9.0)]:
    print(f"s+={s_pos:5.1f} s-={s_neg:4.1f}  BCE {bce(s_pos, s_neg):.4f}  "
          f"BPR {bpr(s_pos, s_neg):.4f}  margin {margin(s_pos, s_neg):.4f}")

# ---- knowledge-graph scoring functions (higher = more plausible) ------------------
def transe(h, r, t):   return 0.0 - np.linalg.norm(h + r - t)
def distmult(h, r, t): return np.sum(h * r * t)
def complex_(h, r, t): return np.real(np.sum(h * r * np.conj(t)))
def rotate(h, r, t):   return 0.0 - np.linalg.norm(h * r - t)

tests = {
    "TransE":   (transe,   np.array([1., 0.]), np.array([0., 1.]), np.array([1., 1.])),
    "DistMult": (distmult, np.array([1., 2.]), np.array([.5, 1.]), np.array([2., 1.])),
    "ComplEx":  (complex_, np.array([1 + 1j]), np.array([-1j]),   np.array([1 - 1j])),
    "RotatE":   (rotate,   np.array([1 + 0j]), np.array([1j]), np.array([1j])),
}
for name, (f, h, r, t) in tests.items():
    print(f"{name:8s} score(drug, treats, disease) = {f(h, r, t):6.3f}   "
          f"score(disease, treats, drug) = {f(t, r, h):6.3f}")
```

Expected output:

```text
s+=  2.0 s-= 0.5  BCE 1.1010  BPR 0.2014  margin 0.0000
s+=  0.3 s-= 0.8  BCE 1.7255  BPR 0.9741  margin 1.5000
s+= 10.0 s-= 9.0  BCE 9.0002  BPR 0.3133  margin 0.0000
TransE   score(drug, treats, disease) =  0.000   score(disease, treats, drug) = -2.000
DistMult score(drug, treats, disease) =  3.000   score(disease, treats, drug) =  3.000
ComplEx  score(drug, treats, disease) =  2.000   score(disease, treats, drug) = -2.000
RotatE   score(drug, treats, disease) =  0.000   score(disease, treats, drug) = -2.000
```

The first three lines are the loss table of section 2.7: the $(10,9)$ row shows BCE punishing miscalibration (9.0) while BPR (0.31) and margin (0) only care about the order. The last four lines are worked example 2: only DistMult gives the same score in both directions.

### 4.5 Edge leakage, and how hidden-link supervision fixes it

This is the central demo of the unit. The model mimics the project's design on the toy world. Input features are `[similarity row ‖ visible link row]`, there is one round of message passing over the **visible** drug–disease edges, and the decoder is bilinear. Three training regimes are compared, each averaged over three seeds:

- **naive**: all training links visible and all supervised (the textbook recipe);
- **DropEdge, supervise all**: hide 20% of links from the input each epoch but still supervise all positives (equivalent to the project's `supervise_hidden=False` setting);
- **hidden-link supervision**: hide 20% each epoch and supervise only those (the project's default).

The extra diagnostic removes *only* a training link from the input and measures how much that link's score drops. A large drop means the model is detecting the edge rather than predicting it. Save as `leakage.py`:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from toy_world import make_world, split, evaluate

A, Sr, Sd = make_world()
A_tr, neg_mask, tp, tn = split(A)
n_r, n_d = A.shape
Sr_t, Sd_t = torch.tensor(Sr), torch.tensor(Sd)

class LinkGNN(nn.Module):
    """Input = [similarity row, VISIBLE link row] (like feat_assoc in the project),
    then one round of message passing over the VISIBLE drug-disease edges,
    then a bilinear decoder."""
    def __init__(self, d=32):
        super().__init__()
        self.inp_r = nn.Linear(n_r + n_d, d)
        self.inp_d = nn.Linear(n_d + n_r, d)
        self.self_r, self.msg_r = nn.Linear(d, d), nn.Linear(d, d)
        self.self_d, self.msg_d = nn.Linear(d, d), nn.Linear(d, d)
        self.W = nn.Parameter(torch.eye(d))
    def forward(self, Avis):
        hr = F.relu(self.inp_r(torch.cat([Sr_t, Avis], 1)))
        hd = F.relu(self.inp_d(torch.cat([Sd_t, Avis.T], 1)))
        mr = (Avis / Avis.sum(1, keepdim=True).clamp(min=1)) @ hd       # mean over visible diseases
        md = (Avis.T / Avis.sum(0)[:, None].clamp(min=1)) @ hr         # mean over visible drugs
        hr, hd = F.relu(self.self_r(hr) + self.msg_r(mr)), F.relu(self.self_d(hd) + self.msg_d(md))
        return hr @ self.W @ hd.T

def fit(mode, epochs=300, drop=0.2, neg_ratio=2, seed=0):
    torch.manual_seed(seed)
    model = LinkGNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    A_full = torch.tensor(A_tr)
    pos = torch.tensor(np.flatnonzero(A_tr.ravel() > 0))
    pool = torch.tensor(np.flatnonzero((neg_mask & (A_tr == 0)).ravel()))
    for _ in range(epochs):
        if mode == "naive":                       # every training link visible AND supervised
            Avis, sup = A_full, pos
        else:                                     # hide a random 20% of links this epoch
            hide = torch.rand(len(pos)) < drop
            Avis = torch.zeros(n_r * n_d); Avis[pos[~hide]] = 1; Avis = Avis.view(n_r, n_d)
            sup = pos[hide] if mode == "hidden" else pos
        logits = model(Avis).reshape(-1)
        neg = pool[torch.randint(len(pool), (len(sup) * neg_ratio,))]
        y = torch.cat([torch.ones(len(sup)), torch.zeros(len(neg))])
        loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        S = torch.sigmoid(model(A_full)).numpy()     # test time: all training links visible
        drops = []                                   # "edge detector" diagnostic
        for e in pos[:100].tolist():
            A_minus = A_full.clone().view(-1); A_minus[e] = 0
            drops.append(S.ravel()[e] - torch.sigmoid(model(A_minus.view(n_r, n_d))).view(-1)[e].item())
    y = np.r_[np.ones(len(pos)), np.zeros(len(pool))]
    train_auc = roc_auc_score(y, S.ravel()[np.r_[pos.numpy(), pool.numpy()]])
    return train_auc, *evaluate(S, tp, tn), np.mean(drops)

for mode, label in [("naive", "naive (see & supervise all)"),
                    ("dropedge", "DropEdge, supervise all"),
                    ("hidden", "hidden-link supervision")]:
    res = np.array([fit(mode, seed=s) for s in range(3)]).mean(0)
    print(f"{label:28s} train-edge AUC {res[0]:.3f} | test AUC {res[1]:.3f}  AUPR {res[2]:.3f}"
          f" | score drop when own edge removed {res[3]:.3f}")
```

Expected output:

```text
naive (see & supervise all)  train-edge AUC 0.998 | test AUC 0.826  AUPR 0.172 | score drop when own edge removed 0.589
DropEdge, supervise all      train-edge AUC 0.995 | test AUC 0.858  AUPR 0.200 | score drop when own edge removed 0.133
hidden-link supervision      train-edge AUC 0.951 | test AUC 0.879  AUPR 0.287 | score drop when own edge removed -0.122
```

**Read the output.**

- The naive model is nearly perfect on its training links (AUC 0.998) and the worst on test links (AUC 0.826, AUPR 0.172). Removing a link from the input drops its predicted probability by 0.59 on average: **the model learned to look at the edge**.
- DropEdge with full supervision reduces the dependence (drop 0.13) and helps (AUPR 0.200), but each positive is still visible in 80% of the epochs in which it is supervised, so the shortcut is still partly rewarded.
- Hidden-link supervision gives the best test results (AUC 0.879, AUPR 0.287, i.e. +67% AUPR over naive). Its training-link AUC is *lower* (0.951) because it no longer memorises. The score no longer depends on the link's own presence; it slightly *rises* when the link is removed (−0.122). The model has only ever been asked to score links that were hidden, so "this pair's link is missing from the input" has become, mildly, part of what a positive looks like. At test time that is exactly the situation of every test link, so the habit is harmless there. It does mean that **the scores of *training* links are not comparable to the scores of unknown pairs**, which matters if you rank "all pairs including known ones".

This reproduces, in miniature, the project's 0.71 → 0.92 validation AUC jump.

### 4.6 Cold start: practice, propagation head, degree gate

Sixteen diseases (each with at least 3 known drugs) are held out completely, as in leave-one-disease-out but for a group. Their columns are zero in training and excluded from the negatives. Metrics are pooled over those 16 columns, as in `run_lodo`. The model is the leakage demo's GNN with hidden-link supervision, plus the project's optional decoder parts. Save as `coldstart.py`:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from toy_world import make_world, knn_kernel

A, Sr, Sd = make_world()
n_r, n_d = A.shape
rng = np.random.default_rng(1)
cold = rng.choice(np.flatnonzero(A.sum(0) >= 3), 16, replace=False)   # 16 "new" diseases
A_tr = A.copy(); A_tr[:, cold] = 0                                    # all their links hidden
neg_mask = A == 0; neg_mask[:, cold] = False                          # never train on them
Sr_t, Sd_t = torch.tensor(Sr), torch.tensor(Sd)
Kr, Kd = torch.tensor(knn_kernel(Sr)), torch.tensor(knn_kernel(Sd))

def cold_metrics(S):
    y, s = A[:, cold].ravel(), S[:, cold].ravel()            # pooled, like run_lodo
    return roc_auc_score(y, s), average_precision_score(y, s)

class Model(nn.Module):
    def __init__(self, prop_head, gate, d=32):
        super().__init__()
        self.prop_head, self.use_gate = prop_head, gate
        self.inp_r, self.inp_d = nn.Linear(n_r + n_d, d), nn.Linear(n_d + n_r, d)
        self.self_r, self.msg_r = nn.Linear(d, d), nn.Linear(d, d)
        self.self_d, self.msg_d = nn.Linear(d, d), nn.Linear(d, d)
        self.W = nn.Parameter(torch.eye(d))
        self.prop_w = nn.Parameter(torch.zeros(2))                 # one weight per view
        self.bias = nn.Parameter(torch.tensor(-3.0))
        self.gate = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    def forward(self, Avis):
        hr = F.relu(self.inp_r(torch.cat([Sr_t, Avis], 1)))
        hd = F.relu(self.inp_d(torch.cat([Sd_t, Avis.T], 1)))
        mr = (Avis / Avis.sum(1, keepdim=True).clamp(min=1)) @ hd
        md = (Avis.T / Avis.sum(0)[:, None].clamp(min=1)) @ hr
        hr, hd = F.relu(self.self_r(hr) + self.msg_r(mr)), F.relu(self.self_d(hd) + self.msg_d(md))
        logits = hr @ self.W @ hd.T                                # GNN (bilinear) term
        if self.use_gate:                                          # degree gate
            g = self.gate
            logits = logits * (torch.sigmoid(g[0] + g[1] * torch.log1p(Avis.sum(1)))[:, None]
                               * torch.sigmoid(g[2] + g[3] * torch.log1p(Avis.sum(0)))[None, :])
        if self.prop_head:                                         # propagation head
            P = torch.stack([Kr @ Avis, (Kd @ Avis.T).T])          # drug view, disease view
            w = F.softplus(self.prop_w) * 5.0                      # non-negative view weights
            logits = logits + (w[:, None, None] * P).sum(0) + self.bias
        return logits

def fit(prop_head=False, gate=False, cold_frac=0.0, epochs=300, seed=0):
    torch.manual_seed(seed)
    model = Model(prop_head, gate)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    pos = torch.tensor(np.flatnonzero(A_tr.ravel() > 0))
    pool = torch.tensor(np.flatnonzero((neg_mask & (A_tr == 0)).ravel()))
    for _ in range(epochs):
        hide = torch.rand(len(pos)) < 0.2                         # hidden-link supervision
        if cold_frac > 0:                                         # cold-start practice
            hide |= (torch.rand(n_d) < cold_frac)[pos % n_d]
        Avis = torch.zeros(n_r * n_d); Avis[pos[~hide]] = 1; Avis = Avis.view(n_r, n_d)
        sup = pos[hide]
        logits = model(Avis).reshape(-1)
        neg = pool[torch.randint(len(pool), (len(sup) * 2,))]
        y = torch.cat([torch.ones(len(sup)), torch.zeros(len(neg))])
        loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(A_tr))).numpy(), model

print(f"cold diseases: {len(cold)}, their true links: {int(A[:, cold].sum())}, "
      f"random AUPR ~ {A[:, cold].mean():.3f}")
P_dis = (knn_kernel(Sd) @ A_tr.T).T                               # no learning at all
print(f"{'disease-view propagation only':34s} AUC {cold_metrics(P_dis)[0]:.3f}  AUPR {cold_metrics(P_dis)[1]:.3f}")
variants = [("GNN (hidden-link sup.)", {}),
            ("GNN + cold-start practice", {"cold_frac": 0.1}),
            ("GNN + prop head", {"prop_head": True}),
            ("GNN + prop head + cold practice", {"prop_head": True, "cold_frac": 0.1}),
            ("  ... + degree gate (full recipe)", {"prop_head": True, "cold_frac": 0.1, "gate": True})]
for label, kw in variants:
    res = np.array([cold_metrics(fit(seed=s, **kw)[0]) for s in range(3)])
    print(f"{label:34s} AUC {res[:, 0].mean():.3f}  AUPR {res[:, 1].mean():.3f}")
_, m = fit(prop_head=True, cold_frac=0.1, gate=True)
g = m.gate.detach()
for k in (0, 1, 5, 20):
    print(f"disease gate factor with {k:2d} visible links: "
          f"{torch.sigmoid(g[2] + g[3] * torch.log1p(torch.tensor(float(k)))).item():.3f}")
```

Expected output:

```text
cold diseases: 16, their true links: 108, random AUPR ~ 0.056
disease-view propagation only      AUC 0.798  AUPR 0.214
GNN (hidden-link sup.)             AUC 0.819  AUPR 0.173
GNN + cold-start practice          AUC 0.877  AUPR 0.258
GNN + prop head                    AUC 0.854  AUPR 0.196
GNN + prop head + cold practice    AUC 0.881  AUPR 0.261
  ... + degree gate (full recipe)  AUC 0.882  AUPR 0.268
disease gate factor with  0 visible links: 0.590
disease gate factor with  1 visible links: 0.731
disease gate factor with  5 visible links: 0.881
disease gate factor with 20 visible links: 0.959
```

**Read the output.**

1. Even a plain GNN with hidden-link supervision beats random (0.056) by a wide margin, because the input features include each disease's similarity row. But it is *worse* on AUPR (0.173) than the non-learned disease-view propagation (0.214). It has never seen a disease with no links during training.
2. **Cold-start practice is the biggest single gain** (AUPR 0.173 → 0.258, AUC 0.819 → 0.877). The project saw the same: 0.055 → ~0.12 from practice alone.
3. The propagation head helps without practice (0.196) and a little on top of it (0.261). The degree gate adds a little more (0.268). The learned gate rises with visible degree (0.59 → 0.96), so the GNN term is trusted less for diseases with no links.
4. Effects of 0.01 between the last three rows are within seed noise. Treat them as "no worse", not "better".

### 4.7 Evaluation protocols: k-fold versus leave-one-disease-out

The same scorer (two-view propagation, no training, so the comparison is about the *protocol*) evaluated under both protocols. Save as `eval_protocols.py`:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from toy_world import make_world, knn_kernel

A, Sr, Sd = make_world()
Kr, Kd = knn_kernel(Sr), knn_kernel(Sd)
def propagate(A_tr):                       # the scorer being evaluated (no training needed)
    return Kr @ A_tr + (Kd @ A_tr.T).T

# ---- warm start: 5-fold CV over cells (1s and 0s each split into 5 folds) -----------
rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
rng.shuffle(pos); rng.shuffle(neg)
res = []
for tp, tn in zip(np.array_split(pos, 5), np.array_split(neg, 5)):
    A_tr = A.copy().ravel(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
    s = propagate(A_tr).ravel()[np.r_[tp, tn]]
    y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    res.append((roc_auc_score(y, s), average_precision_score(y, s)))
res = np.array(res)
print(f"5-fold CV : AUC {res[:,0].mean():.3f} +/- {res[:,0].std():.3f}   "
      f"AUPR {res[:,1].mean():.3f} +/- {res[:,1].std():.3f}")

# ---- cold start: leave-one-disease-out (like evaluation.run_lodo) -------------------
ys, ss, per = [], [], []
for j in range(A.shape[1]):
    A_tr = A.copy(); A_tr[:, j] = 0
    s = propagate(A_tr)[:, j]
    ys.append(A[:, j]); ss.append(s)
    if 0 < A[:, j].sum() < len(A):
        per.append(roc_auc_score(A[:, j], s))
y, s = np.concatenate(ys), np.concatenate(ss)
print(f"LODO      : pooled AUC {roc_auc_score(y, s):.3f}  pooled AUPR {average_precision_score(y, s):.3f}"
      f"  mean per-disease AUC {np.mean(per):.3f} over {len(per)} diseases")
print(f"random AUPR baseline = positive rate = {A.mean():.3f}")
```

Expected output:

```text
5-fold CV : AUC 0.859 +/- 0.014   AUPR 0.289 +/- 0.042
LODO      : pooled AUC 0.805  pooled AUPR 0.202  mean per-disease AUC 0.729 over 74 diseases
random AUPR baseline = positive rate = 0.040
```

**Read the output.** The same scorer drops from AUPR 0.289 (warm) to 0.202 (cold, pooled): cold start is harder even for a method that does not learn. Pooled AUC (0.805) and the mean per-disease AUC (0.729) differ because pooling also rewards getting the *between-disease* scale right, while the per-disease mean only asks "within each disease, are its drugs ranked first?". Six diseases with no links are excluded from the per-disease mean (74 of 80 remain), as in `run_lodo`'s `if 0 < y.sum() < len(y)`. Always say which aggregation you report.

---

## 5. In this project: a line-by-line walkthrough

### 5.1 The configuration: `methods.py::MVHGATConfig`

```python
neg_ratio: int = 2                    # sampled negatives per positive per epoch
drop_edge: float = 0.2                # fraction of assoc edges hidden each epoch
supervise_hidden: bool = True         # loss only on the links hidden this epoch
feat_assoc: bool = True               # append (visible) association rows to features
cold_frac: float = 0.1                # share of diseases fully hidden per epoch
feat_prop: bool = False               # per-view neighbour association profiles as features
prop_head: bool = True                # add the multi-view propagation head to the decoder
degree_gate: bool = True              # scale the GNN term by a learned link-count gate
```

> **Note on the self-check wording.** `PREREQUISITES.md` mentions hiding "~30%" of links and a negative ratio of 5. In the current code, `drop_edge = 0.2` and `cold_frac = 0.1` together hide on average $1-0.8\times0.9=28\%$ of links per epoch ("~30%"), and `neg_ratio = 2` (`HOW_IT_WORKS.md` §4.4 agrees: "2 per positive"). The ratio-5 wording refers to an earlier setting. The reasoning is the same either way.

### 5.2 Setting up: `fit_predict`, before the loop

```python
relations, X0, graphs = self.build(data, A_train)
A_full = t(A_train > 0)
```

`A_train` already has this fold's test links set to 0 (`run_kfold`: `A_tr[test_pos] = 0`), so nothing about test links can enter.

```python
def features(Am):
    # everything derived from links uses only the VISIBLE links Am
    Am = Am.float()
    parts = {"drug": [X0["drug"]], "disease": [X0["disease"]]}
    if c.feat_assoc:
        parts["drug"].append(Am)
        parts["disease"].append(Am.T)
```

**Leak channel closed: features.** The link-row features come from `Am`, the links visible *this epoch*, never from `A_full` during training.

```python
def propagation(Am):
    """One (drugs x diseases) score slice per view, from VISIBLE links only."""
    if not c.prop_head:
        return None
    Am = Am.float()
    return torch.stack([K @ Am for K in self._prop["drug"]] +
                       [(K @ Am.T).T for K in self._prop["disease"]])
```

**The propagation head's inputs**: one $593\times313$ slice per view. Drug views give $K_vA_\text{vis}$ ("similar drugs treat $j$"); disease views give $(K_uA_\text{vis}^\top)^\top$ ("$i$ treats diseases similar to $j$"). `self._prop` holds `knn_kernel` matrices whose diagonal is zero, so **a pair's own link never enters its own propagation score**.

```python
def degrees(Am):
    Am = Am.float()
    return (Am.sum(1), Am.sum(0)) if c.degree_gate else None
```

**Visible degrees** for the gate: per drug (row sums) and per disease (column sums).

```python
n_r, n_d = A_train.shape
pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)
neg_pool = t(np.flatnonzero((neg_mask & (A_train == 0)).ravel()), torch.long)
full_assoc = graphs.get("assoc>drug")
```

Positives and the negative pool are **flat indices** into the $n_r\times n_d$ matrix. Index $p$ is cell $(p\,//\,n_d,\;p\bmod n_d)$, which is why `pos % n_d` gives the disease column below. The pool excludes this fold's test zeros (via `neg_mask`) and every training positive.

### 5.3 The training loop

```python
for _ in range(c.epochs):
    model.train()
    sup = pos
    if c.drop_edge > 0 or c.cold_frac > 0:
        hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge
        if c.cold_frac > 0:
            cold = torch.rand(n_d, device=DEVICE) < c.cold_frac
            hide |= cold[pos % n_d]
```

**Hidden links:** 20% of positives at random, plus *all* positives of the cold diseases this epoch.

```python
        Am = torch.zeros(n_r * n_d, dtype=torch.bool, device=DEVICE)
        Am[pos[~hide]] = True
        Am = Am.view(n_r, n_d)
        if c.supervise_hidden:
            sup = pos[hide]
    else:
        Am = A_full.bool()
```

`Am` is the visible-link matrix. With `supervise_hidden=True` the supervision set is **only the hidden positives** (the fix of section 2.8). With `False` (the "w/o hidden-link supervision" ablation), `sup` stays equal to all positives while `Am` still hides some: "DropEdge, supervise all".

```python
    if full_assoc is not None:
        graphs["assoc>drug"], graphs["assoc>disease"] = Am, Am.T
    logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
```

**Leak channels closed: message passing, features, propagation and degrees** all use `Am`. One forward pass scores *all* $593\times313$ pairs at once (the bilinear decoder is a matrix product).

```python
    logits = logits.reshape(-1)
    n_neg = min(neg_pool.numel(), sup.numel() * c.neg_ratio) if c.neg_ratio > 0 \
        else neg_pool.numel()
    neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]
```

**Negative sampling:** `neg_ratio` × (number of supervised positives), drawn **uniformly with replacement** from the pool, fresh every epoch. (`neg_ratio = 0` would mean "use the whole pool".)

```python
    y = torch.cat([torch.ones(sup.numel(), device=DEVICE),
                   torch.zeros(n_neg, device=DEVICE)])
    loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

**BCE** on hidden positives and sampled negatives, mean-reduced, then one Adam step (lr 2e-3, weight decay 5e-4).

```python
if full_assoc is not None:
    graphs["assoc>drug"], graphs["assoc>disease"] = full_assoc, full_assoc.T
model.eval()
P, deg = propagation(A_full), degrees(A_full)
with torch.no_grad():
    logits, _ = model(X, graphs, P=P, deg=deg)
self.last = (model, X, graphs, P, deg)
return torch.sigmoid(logits).cpu().numpy()
```

**Prediction:** restore the full training graph (all training links visible, test links still absent), switch off dropout (`model.eval()`), turn off autograd, and return $\sigma(\text{logits})$ for every cell. These are **scores**, with the sampling offset of section 2.6, not calibrated probabilities.

### 5.4 The decoder: `model.py::MVHGAT.forward`, `view_weights`, `gnn_gate`

```python
def forward(self, X, graphs, drop_rel=(), P=None, deg=None):
    Hr, Hd, betas = self.encode(X, graphs, drop_rel)
    logits = Hr @ self.W @ Hd.T
```

**Bilinear decoder** over the 192-dimensional jumping-knowledge embeddings: all pairs in one product of shapes $(593\times192)(192\times192)(192\times313)$.

```python
    if self.n_prop and P is not None:
        if deg is not None:
            logits = self.gnn_gate(*deg) * logits
        logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
    return logits, betas
```

**Gate, then add the propagation head and bias.** `P` has shape (views, drugs, diseases). `view_weights()[:, None, None]` broadcasts one weight per view, and `.sum(0)` adds the per-view terms. Because the head is *additive*, $w_vP_v[i,j]$ is literally view $v$'s contribution to the logit of $(i,j)$. That is a faithful explanation (Unit E2).

```python
def view_weights(self):
    return F.softplus(self.prop_w) * self.prop_scale
```

$w_v=\operatorname{softplus}(\theta_v)\cdot s\ge0$. Initialised with $\theta_v=0$ and $s=5$, so $w_v=5\ln2\approx3.47$. Non-negativity encodes the prior that "neighbours treat it" can only *raise* a score.

```python
def gnn_gate(self, deg_r, deg_d):
    g = self.gate
    return torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))[:, None] * \
           torch.sigmoid(g[2] + g[3] * torch.log1p(deg_d))[None, :]
```

The degree gate of section 2.10 as an outer product: a drug factor (column vector) times a disease factor (row vector) gives a $593\times313$ gate. At initialisation $g=(0,1,0,1)$, so each factor is $\operatorname{sigmoid}(\ln(1+d))=\frac{1+d}{2+d}$: 0.5 at degree 0, 0.667 at 1, 0.917 at 10. `self.bias` starts at −3, so an "empty" pair begins with $\sigma(-3)\approx0.05$.

### 5.5 Evaluation: `evaluation.py`

**`kfold_splits` and `run_kfold` (warm start).** The 1s and the 0s are shuffled **separately** into $k$ folds (stratification: each fold has ~1% positives). For fold $f$: `A_tr[test_pos] = 0` hides the test links from *everything*. `neg_mask[test_neg] = False` keeps the test zeros out of the negative pool. The test set is `test_pos ∪ test_neg`: **all** of the fold's unknown cells, not a sample. Metrics are computed per fold, and `summarise` reports mean ± std over folds and repeats. The seed `seed * 1000 + r * 100 + f` differs per fold but is **the same for every method and ablation variant**, which is what makes the paired comparisons below valid.

**`run_lodo` (cold start).** For each disease $j$: `A_tr[:, j] = 0` and `neg_mask[:, j] = False`, so the model is trained without any of $j$'s links and without using any of $j$'s cells as negatives. It then scores all drugs for $j$. Predictions for all diseases are pooled before computing AUC/AUPR, and the mean per-disease AUC is reported too (diseases with 0 or all positives are skipped). Note that the model is **retrained for each held-out disease**, which is why full LODO takes hours and `run_all.ps1` uses a seeded subset of 100 diseases.

### 5.6 Interpreting the ablation (Fdataset, 5-fold CV, 1 repeat, paired folds)

| Variant | AUC | AUPR | $\Delta$AUPR (paired) | worse in | Reading |
|---|---|---|---|---|---|
| **Full MV-HGAT** | 0.943 ± 0.007 | 0.500 ± 0.029 | — | — | reference |
| w/o propagation head | 0.934 ± 0.005 | 0.433 ± 0.034 | **−0.067** | 4/5 | the largest drop: the pair-level heuristic head is the most valuable component; the GNN alone does not rediscover it (cf. section 4.3) |
| w/o degree gate | 0.941 ± 0.004 | 0.479 ± 0.022 | −0.022 | 4/5 | a small, fairly consistent warm-start gain; mixing experts by degree helps even when every disease has links |
| w/o cold-start practice | 0.944 ± 0.004 | 0.509 ± 0.033 | +0.009 | 1/5 | no warm-start cost or benefit, as expected: practice targets cold start (validation: 0.055 → ~0.12 there) |
| w/o hidden-link supervision | 0.944 ± 0.004 | **0.546 ± 0.020** | **+0.046** | **0/5** | *better in every fold*: see below |

**The surprising row.** For the plain GNN, hidden-link supervision was essential (validation AUC 0.71 → 0.92), and the toy demo reproduces that. Yet in the *full* model, switching it off (keeping DropEdge-style hiding but supervising all positives) **improved** warm-start AUPR in all five folds. Possible explanations, none yet verified:

1. **The propagation head is leak-free by construction** (zero-diagonal kernels), and the gated GNN term is now a smaller part of the score. Once the main pathway cannot see the edge, the shortcut is much less dangerous.
2. **3.5× more supervision per epoch.** Supervising all ~1,546 positives instead of ~433 gives lower-variance gradients, which matters with so few labels.
3. **Partial hiding still regularises.** Each positive is hidden in 28% of epochs, which may be enough to stop the GNN term from relying purely on the edge (in the toy, DropEdge already cut the score drop from 0.59 to 0.13).
4. **Training/test mismatch in degrees.** With hidden-only supervision, supervised pairs always have the pair's own link missing, so their degrees are one lower than at test time. That is a small but systematic shift, which the degree gate may amplify.

**What to do** (and what the project is doing): re-check on the validation split, which is *not* the test folds, ideally with several repeats and also under LODO. Warm-start AUPR alone should not decide a setting that was introduced for robustness. If `supervise_hidden=False` also wins on validation and does not hurt cold start, switch the default and say in the paper that the leakage fix mattered for the plain GNN but became unnecessary once the decoder had a leak-free propagation head. That is a genuinely interesting finding. **Do not** switch it because it looks better on the test folds: that would be tuning on test data (Unit E1).

**Headline numbers in context.** Main results (5 repeats × 5 folds): MV-HGAT 0.939 / 0.488, SCMFDD 0.893 / 0.495, MBiRW 0.883 / 0.311. MV-HGAT has the clearly best AUC, while its AUPR is on par with SCMFDD. In cold start (validation proxy) MV-HGAT reaches ~0.15 AUPR against MBiRW's 0.174. A fair summary: *the learned model is competitive with the strongest classical methods, clearly better in overall ranking (AUC), and its components each contribute measurably. The simplest propagation methods remain hard to beat at the very top of the ranking, especially in cold start.*

---

## 6. Common mistakes and misconceptions

1. **Supervising on edges that are in the input graph** (edge leakage). Symptom: training AUC ≈ 1 with poor test AUC. Fix: disjoint message and supervision edges, re-drawn per epoch (section 2.8).
2. **Closing only the graph channel.** Hiding a link from message passing but leaving it in the *features* (link rows), in a propagation feature with a non-zero diagonal, or in the degrees still leaks. Audit every input derived from links.
3. **Evaluating against a handful of sampled negatives.** 1:1 positive/negative test sets inflate AUPR and make results incomparable across papers. Use all unknowns, or a documented hard-negative protocol.
4. **Ranking known and unknown pairs together.** Training links' scores come from a different regime (seen or memorised, or, under hidden-only supervision, slightly inflated or deflated). For candidate lists, rank *unknown* pairs only (`06_case_study.py` lists *new* drugs).
5. **Treating sigmoid outputs as probabilities.** Negative sampling adds a constant logit offset (section 2.6). Calibrate on a validation set if you need probabilities.
6. **Hard negatives in PU data without care.** The highest-scoring unknowns are the most likely undiscovered positives. Here hard negatives cut AUPR from 0.251 to 0.195 (Exercise 8).
7. **Using common neighbours on a bipartite graph.** It is always zero across the two sides. Use odd-length paths or similarity-augmented paths.
8. **Expecting a warm-start-trained model to handle cold start.** Without training on cold situations, the model meets an out-of-distribution input at test time (plain GNN: 0.055 AUPR).
9. **Confusing LODO with inductive learning.** In LODO the disease node and its similarity edges are in the graph; only its links are hidden. A brand-new disease must be *added* to the graph (with its similarity rows) and the model retrained.
10. **Comparing pooled and per-entity metrics.** They answer different questions (section 4.7: 0.805 vs 0.729 AUC).
11. **Choosing a symmetric decoder for an asymmetric relation.** DistMult or dot product cannot express "drug treats disease but not vice versa" within one node type. With separate drug and disease embeddings this is harmless; in a single-type KG it is not.
12. **Tuning on test folds** because a variant "looks better there" (the `no_hidden_sup` temptation). Use a validation split.

---

## 7. Exercises

**[C]** conceptual, **[M]** mathematical, **[P]** coding/project.

**Exercise 1 [M].** For the six-node graph of worked example 1, compute CN, Jaccard, AA, RA and PA for the pair (D, F). Which index ranks (D, F) above (B, E), and why?

<details><summary>Solution</summary>

$\mathcal{N}(D)=\{B,C,E\}$, $\mathcal{N}(F)=\{E\}$, common $\{E\}$ with $k_E=2$. CN = 1; Jaccard $=1/|\{B,C,E\}|=1/3=0.333$; AA $=1/\ln2=1.443$; RA $=1/2=0.5$; PA $=3\times1=3$.

(B, E) had CN 1, Jaccard 0.25, AA 0.910, RA 0.333, PA 6. **AA and RA** rank (D, F) higher, because the shared neighbour E has degree 2 (rare, specific evidence) while (B, E)'s shared neighbour D has degree 3. Jaccard also ranks (D, F) higher, because F has so few neighbours. PA ranks (B, E) higher (6 vs 3), because it only looks at degrees. CN ties. (Values checked numerically.)
</details>

**Exercise 2 [M].** (a) Derive the closed form of the Katz index and its convergence condition. (b) For pair (A, D) with $\beta=0.1$, compute the Katz sum truncated after length 3, given walk counts 0, 2, 2 for lengths 1–3, and compare it with the exact value 0.02350. (c) What happens as $\beta\to1/\lambda_{\max}$?

<details><summary>Solution</summary>

(a) $\sum_{\ell\ge1}\beta^\ell A^\ell=(I-\beta A)^{-1}-I$, valid when the spectral radius of $\beta A$ is below 1, i.e. $\beta<1/\lambda_{\max}$ (for a non-negative symmetric $A$ the spectral radius is $\lambda_{\max}$). Proof: $S=\sum_{\ell\ge0}(\beta A)^\ell$ converges, and $S(I-\beta A)=I$.

(b) $0.1\cdot0+0.01\cdot2+0.001\cdot2=0.022$. The exact 0.02350 adds $0.0001\cdot12=0.0012$ from length 4 and smaller terms. The truncation error is about 6%.

(c) The series diverges. Near the limit the scores become dominated by the leading eigenvector: $\text{Katz}\approx\frac{\beta\lambda_1}{1-\beta\lambda_1}u_1u_1^\top$, so every pair's score is roughly proportional to the product of the two nodes' eigenvector centralities, a global popularity score that ignores locality.
</details>

**Exercise 3 [M/P].** (a) Prove that common neighbours is identically zero for every (drug, disease) pair in a bipartite graph. (b) Show that in leave-one-disease-out, the 3-walk score $(A_\text{train}A_\text{train}^\top A_\text{train})_{:,j}$ is zero for every drug, so its AUC is exactly 0.5. Verify in code, and compare with disease-view propagation.

<details><summary>Solution</summary>

(a) $\mathcal{N}(\text{drug})\subseteq\mathcal{V}_d$ and $\mathcal{N}(\text{disease})\subseteq\mathcal{V}_r$, which are disjoint, so the intersection is empty.

(b) $(A A^\top A)_{ij}=\sum_{k,l}A_{il}A_{kl}A_{kj}$, and $A_{kj}=0$ for all $k$ when column $j$ is hidden. Every drug gets the same score, so every positive–negative pair is a tie, and ties count ½: AUC = 0.5. Save as `ex_walk_lodo.py`:

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from toy_world import make_world, knn_kernel

A, Sr, Sd = make_world()
j = int(np.argmax(A.sum(0)))                 # the disease with the most known drugs
A_tr = A.copy(); A_tr[:, j] = 0              # leave this disease out
walk3 = (A_tr @ A_tr.T @ A_tr)[:, j]
prop_dis = (knn_kernel(Sd) @ A_tr.T).T[:, j]
print("disease", j, "has", int(A[:, j].sum()), "true drugs")
print("3-walk scores in the held-out column:", np.unique(walk3), " AUC", roc_auc_score(A[:, j], walk3))
print(f"disease-view propagation AUC {roc_auc_score(A[:, j], prop_dis):.3f}")
```

Output:

```text
disease 47 has 19 true drugs
3-walk scores in the held-out column: [0.]  AUC 0.5
disease-view propagation AUC 0.934
```

Pure-topology heuristics are blind in cold start. Similarity-based propagation needs only $j$'s similarity row (AUC 0.934 for this disease). This is the reason for the project's propagation head and disease views.
</details>

**Exercise 4 [M].** Using the calibration result of section 2.6, compute the logit offset in the project (fold 0) for `neg_ratio = 2` and for `neg_ratio = 5`. Positives: 1,546, of which ~28% are supervised per epoch; negative pool 146,940. Does the ratio change AUC?

<details><summary>Solution</summary>

Supervised positives ≈ $0.28\times1546\approx433$.

Ratio 2: 866 negatives, $\rho_-=866/146940=0.00589$; offset $=\ln(0.28/0.00589)=\ln47.5\approx3.86$.

Ratio 5: 2,165 negatives, $\rho_-=0.01473$; offset $=\ln(0.28/0.01473)=\ln19.0\approx2.94$.

Under the idealised result the offset is a constant added to every logit, a monotone transformation, so AUC and AUPR are unchanged. In practice the ratio still changes results somewhat (Exercise 8), because it changes optimisation: the gradient balance between positives and negatives, and how much of the negative space is seen per step. The ideal result assumes a perfectly flexible model at its optimum.
</details>

**Exercise 5 [M].** (a) Derive $\partial\mathcal{L}_\text{BPR}/\partial\ell^+$ and $\partial\mathcal{L}_\text{BPR}/\partial\ell^-$. (b) Show that BPR is invariant to adding a constant $c$ to all scores, while BCE is not. (c) Using the table in section 2.7, explain why a model trained with BPR could not be used directly with the project's additive propagation head and bias.

<details><summary>Solution</summary>

(a) Let $\Delta=\ell^+-\ell^-$. $\mathcal{L}=-\log\sigma(\Delta)$ and $d(-\log\sigma(\Delta))/d\Delta=-(1-\sigma(\Delta))$. So $\partial\mathcal{L}/\partial\ell^+=-(1-\sigma(\Delta))$ and $\partial\mathcal{L}/\partial\ell^-=+(1-\sigma(\Delta))$: equal and opposite, and large only when the pair is mis-ordered or close.

(b) $\Delta$ is unchanged by adding $c$ to both scores. BCE changes, e.g. $(2,0.5)\to(12,10.5)$ changes BCE from 1.10 to about 10.5.

(c) It *could* be used, but the bias $b$ would receive zero gradient (any constant cancels in $\Delta$) and would stay at its initial value. The scores would be identifiable only up to a constant, so their scale would be arbitrary. Rankings would be fine, but thresholds and probability-like interpretations would be meaningless. BCE pins the absolute level, which keeps the additive decomposition $w_vP_v+b$ interpretable on a common scale.
</details>

**Exercise 6 [M].** (a) Prove that DistMult is symmetric: $f_r(h,t)=f_r(t,h)$. (b) Show that TransE can model a symmetric relation only with $r=0$. (c) Show that RotatE models a symmetric relation when each $r_k\in\{+1,-1\}$.

<details><summary>Solution</summary>

(a) $\sum_kh_kr_kt_k=\sum_kt_kr_kh_k$, since multiplication commutes.

(b) Perfect symmetric triples need $h+r=t$ and $t+r=h$. Adding gives $2r=0$, so $r=0$, and then $h=t$: all related entities collapse to one point.

(c) Symmetry needs $h\circ r=t$ and $t\circ r=h$, so $h\circ r\circ r=h$, i.e. $r_k^2=1$ for every $k$, giving $r_k=\pm1$ (rotation by 0 or π). Unlike TransE this does not force $h=t$ (for example $r_k=-1$ maps $h_k$ to $-h_k$).
</details>

**Exercise 7 [C].** For each situation, choose a decoder (dot, bilinear, MLP, KG-style) and justify: (a) drug–disease indications with separate drug and disease encoders; (b) drug–drug interactions where "A increases B's toxicity" is directional; (c) several relation types (indication, contraindication, side effect) between drugs and diseases. (d) How many parameters does the project's bilinear $W$ have?

<details><summary>Solution</summary>

(a) Dot or bilinear. The two sides have different embeddings, so symmetry is not an issue; bilinear lets the two spaces differ (the project's choice). (b) Same node type with a directional relation: an asymmetric decoder such as a non-symmetric bilinear $h_a^\top Wh_b$, ComplEx or RotatE. A dot product would give A→B and B→A the same score. (c) One $W_r$ per relation (RESCAL, or DistMult/ComplEx-style diagonals to save parameters), sharing the encoder: R-GCN + DistMult is the classic recipe. (d) $192\times192=36{,}864$, since the jumping-knowledge embedding is $64\times3=192$-dimensional.
</details>

**Exercise 8 [P].** On the toy world with the similarity-MLP encoder and dot decoder, compare (a) BCE vs BPR, (b) negative ratios 1, 2, 5, 10, and (c) uniform vs hard negatives (after 50 warm-up epochs, keep the highest-scoring of 10× uniformly drawn candidates). Average over 3 seeds and explain the results.

<details><summary>Solution</summary>

Save as `ex_losses_negs.py` next to `encoder_decoder.py` and `toy_world.py`:

```python
import numpy as np
import torch
import torch.nn.functional as F
from encoder_decoder import FeatureMLP, Dot, A_tr, neg_mask, tp, tn
from toy_world import evaluate

pos = torch.tensor(np.flatnonzero(A_tr.ravel() > 0))
pool = torch.tensor(np.flatnonzero((neg_mask & (A_tr == 0)).ravel()))

def fit(loss="bce", neg_ratio=2, hard=False, epochs=300, seed=0):
    torch.manual_seed(seed)
    enc, dec = FeatureMLP(), Dot()
    opt = torch.optim.Adam(enc.parameters(), lr=0.01, weight_decay=1e-4)
    for ep in range(epochs):
        logits = dec(*enc()).reshape(-1)
        if hard and ep >= 50:          # after warm-up: keep the highest-scoring of 10x candidates
            cand = pool[torch.randint(len(pool), (len(pos) * neg_ratio * 10,))]
            top = logits[cand].detach().topk(len(pos) * neg_ratio).indices
            neg = cand[top]
        else:
            neg = pool[torch.randint(len(pool), (len(pos) * neg_ratio,))]
        if loss == "bce":
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
            L = F.binary_cross_entropy_with_logits(torch.cat([logits[pos], logits[neg]]), y)
        else:                          # BPR: each positive paired with neg_ratio negatives
            diff = logits[pos].repeat(neg_ratio) - logits[neg]
            L = -F.logsigmoid(diff).mean()
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        return evaluate(torch.sigmoid(dec(*enc())).numpy(), tp, tn)

def avg(**kw):
    return np.array([fit(seed=s, **kw) for s in range(3)]).mean(0)

for l in ("bce", "bpr"):
    auc, aupr = avg(loss=l)
    print(f"(a) loss {l}: AUC {auc:.3f}  AUPR {aupr:.3f}")
for r in (1, 2, 5, 10):
    auc, aupr = avg(neg_ratio=r)
    print(f"(b) neg_ratio {r:2d}: AUC {auc:.3f}  AUPR {aupr:.3f}")
for hard in (False, True):
    auc, aupr = avg(hard=hard)
    print(f"(c) {'hard' if hard else 'uniform':7s} negatives: AUC {auc:.3f}  AUPR {aupr:.3f}")
```

Output:

```text
(a) loss bce: AUC 0.868  AUPR 0.251
(a) loss bpr: AUC 0.877  AUPR 0.238
(b) neg_ratio  1: AUC 0.882  AUPR 0.233
(b) neg_ratio  2: AUC 0.868  AUPR 0.251
(b) neg_ratio  5: AUC 0.863  AUPR 0.237
(b) neg_ratio 10: AUC 0.855  AUPR 0.226
(c) uniform negatives: AUC 0.868  AUPR 0.251
(c) hard    negatives: AUC 0.849  AUPR 0.195
```

(Numbers differ slightly from section 4.3's single-seed run because they average 3 seeds.)

(a) BPR and BCE are close: BPR has slightly higher AUC (it optimises an AUC surrogate), BCE slightly higher AUPR. (b) The ratio matters little; 1–2 is enough here, and large ratios slowly hurt (more easy negatives dominate the loss). This is consistent with the project's choice of 2. (c) Hard negatives **hurt** (AUPR 0.251 → 0.195). Selected for high scores, they are disproportionately pairs that resemble positives: in a PU setting many are undiscovered or plausible links. Training the model to push them down damages the top of the ranking. If you want hard negatives in drug repositioning, filter them (e.g. exclude pairs with high propagation scores) or down-weight them.
</details>

**Exercise 9 [M].** In the project's default training (600 epochs, 28% of positives hidden per epoch), (a) how many epochs is a given training link supervised on average? (b) What is the probability that a given link is *never* supervised? (c) How many negatives does the model see per epoch, and what fraction of the pool is that?

<details><summary>Solution</summary>

(a) $600\times0.28=168$ epochs. (b) $0.72^{600}=e^{600\ln0.72}=e^{-197}\approx0$: every link is supervised many times. (c) ≈866 per epoch, $866/146{,}940\approx0.59\%$ of the pool. Over 600 epochs ≈520k draws with replacement, so each pool cell is drawn about 3.5 times on average, and the probability that a given cell is never drawn is about $e^{-3.5}\approx3\%$.
</details>

**Exercise 10 [C].** Audit `fit_predict` for leakage of a **test** link $(i,j)$ in warm-start CV. List every channel through which information about it could reach the model, and the line(s) that close each.

<details><summary>Solution</summary>

1. *Message passing* (`assoc` edges): `A_train` has the test link zeroed (`run_kfold`: `A_tr[test_pos] = 0`), and `build` uses `A_train`. 2. *Link-row features*: `features(Am)` and `features(A_full)` both derive from `A_train`. 3. *Propagation head*: `propagation(...)` uses `A_train`-derived matrices only. 4. *Degrees*: same. 5. *Supervision*: `pos` is computed from `A_train`, so the test link is never a positive. 6. *Negatives*: `neg_pool` requires `neg_mask`, which excludes the fold's test zeros. The test *positive* is not in `neg_pool` either, because `neg_mask` starts from `A == 0`. So the test link is never shown as a negative (which would also be a kind of leak, teaching the model that cell). 7. *Similarity views*: not derived from $A$ (`HOW_IT_WORKS.md` §6). 8. *Hyper-parameters*: tuned on a validation split, not on the test folds (`05_sensitivity.py`), with the caveat that the validation split overlaps the CV data (nested CV would be stricter).

(The *training* links are a separate matter: there the leak is between message edges and supervision edges, handled by `hide`/`sup`.)
</details>

**Exercise 11 [C].** In the leakage demo, hidden-link supervision produced a *negative* score drop (−0.122): removing a training link from the input *raised* its score. Explain why, and say when this could cause problems in the project.

<details><summary>Solution</summary>

Under hidden-only supervision, every positive the model is trained on has its own link absent from the input; links that are present are never labelled. The model can therefore learn mild cues associated with "a hidden positive", e.g. a drug whose visible link row is missing one of its usual diseases, or a node whose visible degree is one lower than "expected" from its similarity profile. Removing a training link creates exactly that situation, so its score rises a little.

This is harmless for test links (they are absent from the input, just like training positives). It matters when ranking **all** pairs including known training links, or when comparing training-link scores with unknown-pair scores, e.g. "is this known indication scored above the new candidates?". Rank only unknown pairs for candidate lists (as `06_case_study.py` does with *new* drugs), and do not interpret training-link scores.
</details>

**Exercise 12 [C].** A reviewer asks: "Can your model propose indications for a drug approved after the benchmark was built?" Design an evaluation that answers this, and say what must change in the code to support it.

<details><summary>Solution</summary>

This is an **inductive drug cold start**: the drug is not in the graph. Protocol: choose a set of drugs; for each, remove the drug's *node entirely* during training (its links, its similarity rows/columns, its features), then add it back at test time with its similarity rows only (and no links), and rank all diseases for it. Pool AUC/AUPR and the per-drug mean; compare with similarity propagation on the drug side ($K_rA$) and MBiRW under the same protocol. A cheaper variant is **leave-one-drug-out**, the mirror of `run_lodo` (`A_tr[i, :] = 0`): the drug stays in the graph with its similarity edges but no links. That is the transductive cold start, closest to what the code supports now (add `run_lodro` to `evaluation.py`, and set `cold_frac` analogously for drugs during training, e.g. hide all links of ~10% of *drugs* per epoch). For a true inductive test the model must be retrained with the new node added, because it is transductive. A time-split evaluation (indications approved after the benchmark date, e.g. from ClinicalTrials.gov or newer DrugBank) is the most convincing version.
</details>

**Exercise 13 [C].** Give a small example in which a node-wise GNN *must* assign the same score to two candidate links that should be scored differently, and explain how SEAL's labeling fixes it.

<details><summary>Solution</summary>

Take a cycle of 6 nodes $c_0\dots c_5$ with identical features. All nodes are automorphic, so a node-wise GNN gives all of them the same embedding, and every candidate pair, $(c_0,c_2)$ at distance 2 and $(c_0,c_3)$ at distance 3, gets the same score. Yet the two pairs differ structurally (common-neighbour count 1 vs 0). SEAL labels each node of the enclosing subgraph by its distances to the *two targets*. The subgraph of $(c_0,c_2)$ has a node at distance (1,1) (their common neighbour); that of $(c_0,c_3)$ does not. The GNN therefore sees different labelled subgraphs and can score them differently. In general, the labeling trick makes the representation a function of the *pair*, not of two separately computed node embeddings.
</details>

**Exercise 14 [C].** The ablation shows "w/o propagation head" is the largest drop (−0.067 AUPR), while section 4.3 shows the propagation heuristic alone has the best AUPR on the toy. Is the GNN useless? What experiment would quantify what the GNN adds on top of propagation?

<details><summary>Solution</summary>

Not necessarily. The full model (GNN + head) beats the GNN alone by 0.067, but we have not shown what the head *alone* achieves on the real data. Add an ablation **"propagation head only"**: set the GNN term to zero (e.g. a variant that skips `Hr @ self.W @ Hd.T`), keep the learned $w_v$ and bias, and train the same way. If it scores, say, 0.46 AUPR, the GNN adds ~0.04. If it scores 0.50, the GNN adds nothing in warm start. Run it on the same folds (paired), with several repeats, and also under LODO, where the gate makes the GNN term small anyway. This is the honest "how much does the deep part buy?" experiment that reviewers will ask for. MBiRW (0.311 AUPR) is a weaker, non-learned propagation reference; the learned per-view weights are already one improvement over it.
</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (C6)

### Q1. Why do we hide ~30% of known links from the message-passing graph each epoch?

*(In the current code: 20% at random plus all links of ~10% of diseases, i.e. about 28% per epoch.)*

Because **the links we supervise on must not be visible in the model's input**. That is the only way training looks like testing. At test time, the link to be predicted is never in the graph, never in the features, never in the propagation scores and never in the degrees. If we trained on links that *are* visible, the cheapest way to lower the loss would be to detect them: a drug's link row has a 1 in column $j$, and through the `assoc` relation disease $j$'s state flows straight into the drug's embedding. The model would learn "score high when the edge is already there" (edge leakage). That gives near-perfect training scores and is useless on test pairs, which never have their edge. The project's plain GNN showed exactly this: validation AUC 0.71 when trained on visible links, 0.92 with hidden-link supervision. Our toy demo shows the same pattern: naive training AUC 0.998 vs test AUC 0.826, and removing a link from the input dropped its score by 0.59.

Hiding a fresh random subset **every epoch** (rather than one fixed message/supervision split) means every link serves as a message edge in most epochs and as a supervision target in about 28% of them (≈168 of 600 epochs), never both at once. So no information is permanently thrown away. The extra "cold" part, hiding *all* links of ~10% of diseases, additionally trains the model for leave-one-disease-out testing, where a disease has no visible links at all. Note the nuance from the ablation (section 5.6): once the decoder has the leak-free propagation head, supervising *all* positives while still hiding some from the input did better in warm-start CV. Whether to keep hidden-only supervision is being re-checked on the validation split.

### Q2. In LODO the held-out disease has no `assoc` edges at all. Which relations can still inform its embedding?

Encoder side (relations into the disease node):

1. **The three disease similarity views**, `view:pheno_mim`, `view:sem_mondo`, `view:gene_d`. These are k-NN edges from similar diseases, computed from phenotype text, the MONDO hierarchy and CTD genes, none of which depend on $A$. They remain valid (each also has a self-loop).
2. **Two-hop information through those views.** In layer 2, the held-out disease receives its similar diseases' layer-1 states, and *those* diseases did receive messages from their drugs through `assoc>disease` in layer 1. So "drugs that treat diseases like me" reaches the embedding indirectly.
3. **The gene bridge** (`gene_bridge>disease`), if enabled: drugs whose gene profiles overlap the disease's. It is off by default, and it carried almost no signal.
4. **Its own input features and the skip path**: its similarity rows in all disease views. Its link row is all zeros.

`assoc>disease` itself is **invalid** for this node (no visible neighbours), so `ViewAttention` masks it with $-\infty$ and its weight is redistributed over the valid views.

Decoder side: the drug-view propagation terms are zero for its column ($K_rA[:,j]=0$), but the **disease-view propagation terms** $P_u[i,j]=\sum_lK_u[j,l]A[i,l]$, "does drug $i$ treat diseases similar to $j$?", remain, with learned weights $w_u$. The **degree gate** sees $\deg_j=0$ and shrinks the GNN term, so the prediction leans on those propagation terms. Cold-start practice during training is what taught the gate and the weights to behave this way.

---

## 9. Summary and cheat sheet

**Big picture.** Drug repositioning is link prediction on a sparse, positive-unlabelled bipartite graph. A link predictor = encoder (node → vector) + decoder (pair → score) + negatives + loss + a protocol that never lets the predicted link be seen.

| Concept | Formula / fact |
|---|---|
| CN, Jaccard | $\lvert\mathcal{N}(x)\cap\mathcal{N}(y)\rvert$; divided by $\lvert\mathcal{N}(x)\cup\mathcal{N}(y)\rvert$ |
| Adamic–Adar, RA | $\sum_z1/\log k_z$; $\sum_z1/k_z$ over common neighbours |
| Preferential attachment | $k_xk_y$ |
| Katz | $(I-\beta A)^{-1}-I$, $\beta<1/\lambda_{\max}$ |
| bipartite | CN ≡ 0 across sides; use $AA^\top A$ (item-CF) or similarity paths $K_vA$, $AK_u^\top$ |
| DeepWalk / node2vec | random walks + skip-gram with negative sampling; node2vec's $p,q$ bias BFS/DFS; transductive |
| decoders | dot $h_u^\top h_v$; bilinear $h_u^\top Wh_v$; MLP$([h_u\Vert h_v])$ |
| KG | TransE $-\lVert h+r-t\rVert$; DistMult $\langle h,r,t\rangle$ (symmetric); ComplEx $\operatorname{Re}\langle h,r,\bar t\rangle$; RotatE $-\lVert h\circ r-t\rVert$ |
| BCE | $\operatorname{softplus}(-\ell^+)+\operatorname{softplus}(\ell^-)$; punishes miscalibration |
| BPR | $-\log\sigma(\ell^+-\ell^-)$; smooth AUC surrogate; shift-invariant |
| margin | $\max(0,\gamma-\ell^++\ell^-)$ |
| sampling offset | $\ell^*=\text{true log-odds}+\log(\rho_+/\rho_-)$; ranks unchanged; project ≈ +3.9 |
| hard negatives | risky in PU data (false negatives); toy AUPR 0.251 → 0.195 |
| edge leakage | supervised edge ∈ input ⇒ edge detector; train AUC ≈ 1, test poor |
| fixes | disjoint message/supervision edges; DropEdge (partial); **hidden-link supervision** (re-split every epoch) |
| other leak channels | features, propagation (zero-diagonal kernel), degrees, negatives, label-derived views |
| warm vs cold | k-fold edge split vs leave-one-disease-out; LODO = transductive cold start |
| cold-start recipe | cold-start practice + propagation head + degree gate $\sigma(a+b\log(1{+}\deg))$ |
| SEAL | enclosing subgraph + DRNL labels + GNN; target edge removed; labeling trick fixes automorphic pairs |
| protocol | all unknowns as test negatives; test zeros excluded from training negatives; AUPR + AUC; paired folds; tune on validation |
| project numbers | full 0.943/0.500; w/o prop head −0.067 AUPR; w/o gate −0.022; w/o hidden sup +0.046 (5/5, re-checking); cold: 0.055 → ~0.15 (MBiRW 0.174) |

---

## 10. Further resources (all links checked on 2026-10-01)

**Courses and books**
- [Stanford CS224W, course home](https://web.stanford.edu/class/cs224w/) (free).
- [CS224W Fall 2025, Lecture 5 "GNN augmentation and training" (slides, PDF)](https://snap.stanford.edu/class/cs224w-2025/slides/05-GNN3.pdf) (free). Link-prediction splits, **message vs supervision edges**, transductive vs inductive; the canonical source for section 2.8.
- [CS224W Fall 2025, Lecture 10 "Knowledge graphs" (slides, PDF)](https://snap.stanford.edu/class/cs224w-2025/slides/10-kg.pdf) (free). TransE, DistMult, ComplEx, RotatE and relation patterns.
- [CS224W Fall 2025, Lecture 11 "GNNs for recommender systems" (slides, PDF)](https://snap.stanford.edu/class/cs224w-2025/slides/11-recsys.pdf) (free). BPR loss, negative sampling, LightGCN; the closest analogue to drug–disease recommendation.
- [CS224W Fall 2021 videos (YouTube playlist)](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn) (free).
- [W. L. Hamilton, *Graph Representation Learning*, free pre-print](https://www.cs.mcgill.ca/~wlh/grl_book/) (free). Ch. 2 §2.2 heuristics; ch. 3 encoder–decoder and shallow embeddings; ch. 4 multi-relational data and KGs; §6.1.3 GNNs for relation prediction.

**Heuristics and embeddings**
- [Liben-Nowell & Kleinberg 2007, "The Link Prediction Problem for Social Networks" (JASIST), PDF](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf) (free). The classic comparison of CN, Jaccard, AA, Katz and more.
- [Lü & Zhou 2011, "Link Prediction in Complex Networks: A Survey" (Physica A), arXiv](https://arxiv.org/abs/1010.0725) (free). RA, local path and global indices, with derivations.
- [Perozzi et al. 2014, "DeepWalk" (KDD), arXiv](https://arxiv.org/abs/1403.6652) (free).
- [Grover & Leskovec 2016, "node2vec" (KDD), arXiv](https://arxiv.org/abs/1607.00653) (free). See §4.4 for link prediction with binary operators.
- [Mikolov et al. 2013, "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS), arXiv](https://arxiv.org/abs/1310.4546) (free). The origin of negative sampling.

**Decoders, knowledge graphs and losses**
- [Bordes et al. 2013, "Translating Embeddings for Modeling Multi-relational Data" (TransE, NeurIPS)](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) (free).
- [Nickel, Tresp & Kriegel 2011, "A Three-Way Model for Collective Learning on Multi-Relational Data" (RESCAL, ICML), PDF](https://icml.cc/2011/papers/438_icmlpaper.pdf) (free).
- [Yang et al. 2015, "Embedding Entities and Relations for Learning and Inference in Knowledge Bases" (DistMult, ICLR), arXiv](https://arxiv.org/abs/1412.6575) (free).
- [Trouillon et al. 2016, "Complex Embeddings for Simple Link Prediction" (ComplEx, ICML), arXiv](https://arxiv.org/abs/1606.06357) (free).
- [Sun et al. 2019, "RotatE" (ICLR), arXiv](https://arxiv.org/abs/1902.10197) (free).
- [Rendle et al. 2009, "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI), arXiv](https://arxiv.org/abs/1205.2618) (free).
- [Kipf & Welling 2016, "Variational Graph Auto-Encoders", arXiv](https://arxiv.org/abs/1611.07308) (free). The GNN-encoder + dot-decoder template for link prediction.
- [KGE tutorial "Knowledge Graph Embeddings: From Theory to Practice" (ECAI 2020)](https://kge-tutorial-ecai2020.github.io/) (free). Slides and notebooks.

**Leakage, subgraph methods, cold start**
- [Rong et al. 2020, "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification" (ICLR), arXiv](https://arxiv.org/abs/1907.10903) (free).
- [Zhang & Chen 2018, "Link Prediction Based on Graph Neural Networks" (SEAL, NeurIPS), arXiv](https://arxiv.org/abs/1802.09691) (free).
- [Zhang et al. 2021, "Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning" (NeurIPS), arXiv](https://arxiv.org/abs/2010.16103) (free).
- [Zhang & Chen 2020, "Inductive Matrix Completion Based on Graph Neural Networks" (IGMC, ICLR), arXiv](https://arxiv.org/abs/1904.12058) (free). Inductive, cold-start-friendly pair scoring.

**Evaluation**
- [Li et al. 2023, "Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking" (NeurIPS D&B), arXiv](https://arxiv.org/abs/2306.10453) (free). Easy random negatives, inconsistent protocols, and the HeaRT fix.
- [Hu et al. 2020, "Open Graph Benchmark" (NeurIPS), arXiv](https://arxiv.org/abs/2005.00687) (free). Standardised link-prediction splits and Hits@K / MRR metrics.

**Hands-on and biomedical**
- [PyG `RandomLinkSplit` documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RandomLinkSplit.html) (free). See `disjoint_train_ratio`: the library form of message/supervision edge splitting.
- [PyG example `link_pred.py` (GitHub)](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py) (free).
- [Zitnik, Agrawal & Leskovec 2018, "Modeling polypharmacy side effects with graph convolutional networks" (Decagon), arXiv](https://arxiv.org/abs/1802.00543) (free). Multi-relational link prediction for drugs.
- [Yu et al. 2021, "Predicting drug–disease associations through layer attention graph convolutional network" (LAGCN, Brief. Bioinform.), DOI](https://doi.org/10.1093/bib/bbaa243) (paid). One of the project's baselines.

---

## 11. Glossary

- **Adamic–Adar (AA)**: common-neighbour count weighted by $1/\log(\text{degree})$ of each shared neighbour.
- **AUPR / AUC**: area under the precision–recall / ROC curve (Unit B2). AUPR's random baseline equals the positive rate.
- **Bilinear decoder**: $s(u,v)=h_u^\top Wh_v$ with a learned matrix $W$.
- **BPR**: Bayesian Personalised Ranking loss, $-\log\sigma(\ell^+-\ell^-)$; a smooth AUC surrogate.
- **Calibration offset**: the constant shift of learned logits caused by sampling positives and negatives at different rates.
- **Cold start**: predicting links for a node with no observed links (here: a disease with no known drugs).
- **Cold-start practice**: hiding all links of a random subset of nodes during training, to simulate cold start (`cold_frac`).
- **Common neighbours (CN)**: number of shared neighbours of two nodes; always 0 across the sides of a bipartite graph.
- **Degree gate**: the learned factor $\operatorname{sigmoid}(a+b\log(1+\deg))$ that scales the GNN term by how many visible links a node has.
- **DistMult / ComplEx / RotatE / TransE / RESCAL**: knowledge-graph scoring functions (section 2.5).
- **DropEdge**: randomly removing a fraction of edges from the message graph each epoch.
- **Edge leakage**: supervising on edges that are also inputs to the encoder, so that the model learns to detect rather than predict edges.
- **Enclosing subgraph**: the $h$-hop neighbourhood around a target pair, used by SEAL.
- **Encoder–decoder framework**: node → vector (encoder) and pair of vectors → score (decoder), trained with a pairwise loss.
- **Hard negative**: a sampled negative chosen because the model, or a heuristic, scores it highly.
- **Hidden-link supervision**: each epoch, hiding a random subset of links from all inputs and computing the loss only on them (`supervise_hidden`).
- **Katz index**: discounted count of all walks between two nodes, $(I-\beta A)^{-1}-I$.
- **Labeling trick**: marking target nodes before running a GNN, so that it can represent pair-level structure.
- **Leave-one-disease-out (LODO)**: hide all links of one disease, train, rank all drugs for it; repeat for every disease.
- **Message edges / supervision edges**: edges used as encoder input vs edges used as training targets.
- **Negative sampling**: drawing a subset of unknown pairs as negatives each training step.
- **node2vec / DeepWalk**: shallow node embeddings learned from (biased / uniform) random walks with skip-gram.
- **Positive–unlabelled (PU) learning**: learning from known positives and unlabelled examples, some of which are positive.
- **Preferential attachment (PA)**: the product of degrees.
- **Propagation head**: the project's additive per-view guilt-by-association term $\sum_vw_vP_v[i,j]$.
- **Resource allocation (RA)**: common neighbours weighted by $1/\text{degree}$.
- **SEAL**: link prediction by classifying labelled enclosing subgraphs with a GNN.
- **Transductive / inductive**: whether the model can only handle nodes present during training, or also new ones.
- **Warm start**: evaluation where test pairs involve nodes that still have other known links in training (k-fold CV).
