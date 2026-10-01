# Prerequisites module - what to study to own this project

This is a study plan, not a textbook. Each unit says **why it matters for this
project**, **what exactly to learn**, **where in the code you will meet it**, a
few **self-check questions**, and **resources**. The units are ordered so that
each one only depends on the ones before it.

If you can answer every self-check question at the end of a unit without looking,
move on. If you cannot, that is the part to revisit.

```
 Track A - Foundations          Track B - Machine learning        Track C - Graphs
 A1 Python / NumPy / pandas     B1 Supervised learning basics     C1 Graph theory basics
 A2 Linear algebra              B2 Evaluation & imbalanced data   C2 Random walks & propagation
 A3 Probability & statistics    B3 Neural nets & PyTorch          C3 Graph neural networks (GCN)
                                B4 Matrix factorisation &         C4 Attention & GAT
                                   recommender systems            C5 Heterogeneous graphs & HAN
                                                                  C6 Link prediction
 Track D - The biology / chemistry                 Track E - Research practice
 D1 Pharmacology & drug repositioning             E1 Data leakage & experimental design
 D2 Cheminformatics (SMILES, fingerprints)        E2 Interpretability & its limits
 D3 Biomedical ontologies & semantic similarity   E3 Reproducibility & writing results
 D4 Biomedical databases (DrugBank, OMIM, CTD...)
```

Suggested order if you are starting from scratch (roughly 8-10 weeks part-time):
**A1 → A2 → A3 → B1 → B2 → D1 → B3 → C1 → C2 → B4 → C3 → C4 → C5 → C6 → D2 → D3 → D4 → E1 → E2 → E3**.
If you already know ML well, start at D1, then C1.

---

## Track A - Foundations

### A1. Python for data work: NumPy, pandas, scripts
**Why here:** every file in `scripts/` and `src/drepo/` is NumPy/pandas code. The
association matrix is a NumPy array; masks, fancy indexing and `np.flatnonzero`
are used everywhere to build folds.

**Learn:** array shapes and broadcasting; boolean masks; `ravel`/`reshape`;
`np.argsort`, `np.argpartition`; pandas `read_csv`, `groupby`, `merge`,
chunked reading of large files; virtual environments and `pip`.

**Meet it in:** `evaluation.py::kfold_splits`, `data.py::knn_mask`,
`02_build_features.py` (chunked CTD reading).

**Self-check**
1. `A` is 593x313. What does `A.ravel()[k]` refer to in terms of (drug, disease)?
2. Why does `knn_mask` use `argpartition` instead of `argsort`?
3. Why read `CTD_chemicals_diseases.tsv.gz` in chunks?

**Resources:** *Python for Data Analysis* (Wes McKinney, free online, ch. 4-5, 7-8);
NumPy "absolute beginners" guide (numpy.org).

### A2. Linear algebra
**Why here:** a graph *is* a matrix. GCN layers are matrix products; SVT/DRRS is a
singular-value decomposition; the bilinear decoder is `H_r W H_dᵀ`.

**Learn:** matrix multiplication as "mixing rows"; transpose; symmetric matrices;
eigen-decomposition and SVD; rank and low-rank approximation; norms (Frobenius,
nuclear); the graph Laplacian `L = I - D^-1/2 S D^-1/2`.

**Meet it in:** `methods.py` - `sym_norm`, `DRRS` (SVD thresholding), `SCMFDD`
(Laplacian regulariser), `MVHGAT.forward` (bilinear decoder).

**Self-check**
1. What does multiplying a similarity matrix `S` (n x n) by `A` (n x m) do to each row of `A`?
2. Why does a low-rank assumption make sense for drug-disease matrices?
3. What does `tr(Uᵀ L U)` penalise?

**Resources:** 3Blue1Brown "Essence of Linear Algebra" (YouTube);
Gilbert Strang, *Linear Algebra and Learning from Data* (ch. I.1-I.9).

### A3. Probability and statistics
**Why here:** the model outputs probabilities (sigmoid), is trained with
cross-entropy, and results are reported as mean ± standard deviation.

**Learn:** Bernoulli distribution; likelihood and log-likelihood; binary
cross-entropy; mean/variance/standard deviation; what a random baseline looks
like; base rates (why 1% positives changes everything).

**Self-check**
1. Derive binary cross-entropy from the Bernoulli likelihood.
2. If 1% of pairs are positive, what AUPR does a random ranker get? What AUC?

**Resources:** *Seeing Theory* (seeing-theory.brown.edu); StatQuest (YouTube) on
likelihood, cross-entropy, ROC/AUC.

---

## Track B - Machine learning

### B1. Supervised learning basics
**Learn:** train/validation/test; over-fitting and regularisation (weight decay,
dropout); gradient descent and Adam; hyper-parameters vs parameters; k-fold
cross-validation.

**Meet it in:** `MVHGATConfig` (every field is a hyper-parameter),
`05_sensitivity.py` (tuning on a validation split, not the test folds).

**Self-check**
1. Why must hyper-parameters be tuned on a validation split and not on the test folds?
2. What do dropout and weight decay each do?

**Resources:** Andrew Ng, *Machine Learning Specialization* (Coursera) - courses 1-2;
Google's *Machine Learning Crash Course*.

### B2. Evaluation and imbalanced data  ← very important for this project
**Why here:** only ~1% of drug-disease pairs are known links. The headline
numbers (AUC, AUPR) and the protocol (warm-start CV vs cold-start LODO)
decide whether a result is meaningful.

**Learn:** confusion matrix; TPR/FPR; ROC curve and AUC (and its probabilistic
meaning); precision-recall curve, average precision/AUPR; why ROC looks
optimistic under heavy imbalance; ranking metrics (precision@k, recall@k).

**Meet it in:** `evaluation.py` (docstring explains both protocols),
`08_make_figures.py` (curves).

**Self-check**
1. A model has AUC 0.93 but AUPR 0.30. Is it good? Compared with what?
2. Why is leave-one-disease-out harder than 5-fold CV?
3. What does "the test set contains all unknown pairs" assume about unknowns?

**Resources:** Saito & Rehmsmeier (2015), "The precision-recall plot is more
informative than the ROC plot when evaluating binary classifiers on imbalanced
datasets", *PLOS ONE*; Fawcett (2006), "An introduction to ROC analysis".

### B3. Neural networks and PyTorch
**Learn:** tensors and autograd; `nn.Module`, `nn.Linear`, activations (ReLU,
ELU, LeakyReLU); softmax; `LayerNorm`; residual/skip connections; the training
loop (forward → loss → backward → step); moving tensors to the GPU;
`model.train()` vs `model.eval()`.

**Meet it in:** `model.py` (the whole file), `methods.py::MVHGATMethod.fit_predict`.

**Self-check**
1. Why is `torch.no_grad()` used at prediction time?
2. What does `masked_fill(~mask, -inf)` followed by softmax achieve?

**Resources:** official PyTorch "Learn the Basics" tutorials; Andrej Karpathy
"Neural Networks: Zero to Hero" (first 3 videos); *Dive into Deep Learning* (d2l.ai) ch. 2-5.

### B4. Matrix factorisation and recommender systems
**Why here:** drug repositioning *is* a recommendation problem: diseases are
"users", drugs are "items", known indications are "ratings". SCMFDD and DRRS
are matrix-completion methods; our bilinear decoder is a learned factorisation.

**Learn:** collaborative filtering; matrix factorisation `A ≈ U Vᵀ`; implicit
feedback (unknown ≠ negative); negative sampling; matrix completion and nuclear
norm / singular value thresholding.

**Self-check**
1. Why is "unknown" not the same as "does not treat"? How does that affect training labels?
2. What does negative sampling ratio 5 mean in `MVHGATConfig`?

**Resources:** Koren, Bell & Volinsky (2009), "Matrix factorization techniques
for recommender systems", *IEEE Computer*; Google's *Recommendation Systems* course.

---

## Track C - Graphs and graph neural networks

### C1. Graph theory basics
**Learn:** nodes, edges, weighted/unweighted, directed/undirected; adjacency
matrix; degree; bipartite and k-partite graphs; heterogeneous graphs (node and
edge types); k-nearest-neighbour graphs; meta-paths (e.g. drug-gene-disease).

**Meet it in:** `data.py::knn_mask`, `methods.py::MVHGATMethod.build` (all relations).

**Self-check**
1. Our graph has drug, disease and gene nodes. Which edge types exist?
2. How is a drug-gene-disease meta-path turned into a direct drug-disease relation?

**Resources:** Stanford **CS224W** (Jure Leskovec), lectures 1-2 (free on YouTube + slides);
William Hamilton, *Graph Representation Learning* (free book), ch. 1-2.

### C2. Random walks and label propagation
**Why here:** MBiRW (a baseline) and many classic methods are random walks on
the heterogeneous network; GNNs can be seen as learned propagation.

**Learn:** transition matrices; random walk with restart; label propagation;
the bi-random walk of MBiRW.

**Meet it in:** `MBiRW` in `methods.py`, and the *multi-view propagation head*
of our own model (`propagation()` in `MVHGATMethod.fit_predict`): one step of
similarity-weighted propagation per view.

**Self-check:** In `MBiRW`, what does `alpha` trade off? What does "left walk" vs
"right walk" mean? Why is propagation so strong for diseases with no known drugs?

**Resources:** CS224W lecture on PageRank/random walks; Luo et al. 2016 (MBiRW) Methods section.

### C3. Graph neural networks - message passing and GCN
**Learn:** the message-passing template (aggregate neighbours → update);
GCN (Kipf & Welling 2017) and its normalisation `D^-1/2 A D^-1/2`;
over-smoothing and why depth is limited; GraphSAGE; jumping-knowledge
(concatenating layers); transductive vs inductive learning.

**Meet it in:** `_GCN`, `NIMCGCN`, `LAGCN` in `methods.py`; the JK concat in `model.py`.

**Self-check**
1. Why can a 2-layer GNN "see" 2-hop neighbours?
2. What is over-smoothing and how does the kNN sparsification help?
3. Why is our model *transductive*, and what does that imply for brand-new drugs?

**Resources:** Kipf & Welling (2017), "Semi-supervised classification with graph
convolutional networks"; Distill.pub "A Gentle Introduction to Graph Neural
Networks" and "Understanding Convolutions on Graphs"; CS224W lectures 6-8.

### C4. Attention and Graph Attention Networks (GAT)
**Learn:** the attention idea (score → softmax → weighted sum); multi-head
attention; GAT (Veličković et al. 2018): `e_ij = LeakyReLU(aᵀ[Wh_i || Wh_j])`,
masked softmax over neighbours.

**Meet it in:** `model.py::DenseGAT` - it is a line-by-line implementation of the
GAT equations (the two `a_src`/`a_dst` vectors are the two halves of `a`).

**Self-check**
1. Why is `a` split into `a_src` and `a_dst` in our code? (Hint: `aᵀ[x||y] = a1ᵀx + a2ᵀy`.)
2. What happens to a node with no neighbours in a relation?

**Resources:** Veličković et al. (2018), "Graph Attention Networks" (ICLR);
Jay Alammar "The Illustrated Transformer" (for attention intuition).

### C5. Heterogeneous graphs and HAN
**Why here:** this is the core of the proposed model: several relations
(similarity views, known links, gene bridge) combined with a *second*
attention over relations.

**Learn:** relational GNNs (R-GCN); Heterogeneous Graph Attention Network (HAN):
node-level attention *within* each meta-path + semantic-level attention
*across* meta-paths; multi-view learning.

**Meet it in:** `model.py::ViewAttention`, `HeteroLayer`.

**Self-check**
1. Where are the two levels of attention in `HeteroLayer.forward`?
2. Our `ViewAttention` is node-specific, unlike the original HAN. What does that buy us?

**Resources:** Wang et al. (2019), "Heterogeneous Graph Attention Network" (WWW);
Schlichtkrull et al. (2018), "Modeling relational data with graph convolutional networks";
Hamilton's book ch. 5.

### C6. Link prediction
**Learn:** framing a task as predicting missing edges; encoders + decoders
(dot product, bilinear, MLP); negative sampling; *edge leakage* (supervising
on edges that are also message-passing inputs) and DropEdge; cold-start.

**Meet it in:** `MVHGAT.forward` (bilinear decoder), `fit_predict` (negative
sampling + DropEdge), `evaluation.py::run_lodo` (cold-start).

**Self-check**
1. Why do we hide ~30% of known links from the message-passing graph each epoch?
2. In LODO the held-out disease has no `assoc` edges at all. Which relations can still inform its embedding?

**Resources:** CS224W lectures on link prediction/knowledge graphs; Zhang & Chen (2018),
"Link prediction based on graph neural networks" (SEAL).

---

## Track D - The biology and chemistry

### D1. Pharmacology and drug repositioning
**Learn:** drug, target, mechanism of action; indication vs off-label use;
what repositioning is and famous examples (sildenafil, thalidomide, metformin
in oncology trials); approval phases and clinical trials; why computational
predictions are *hypotheses*, not recommendations.

**Self-check**
1. Why do similar drugs tend to treat similar diseases (the "guilt-by-association" assumption)? When does it fail?
2. What would it take to turn a top-10 prediction into a real repurposed drug?

**Resources:** Pushpakom et al. (2019), "Drug repurposing: progress, challenges and
recommendations", *Nature Reviews Drug Discovery*; Gottlieb et al. (2011) PREDICT - Introduction.

### D2. Cheminformatics: SMILES, fingerprints, Tanimoto
**Learn:** molecular graphs; SMILES strings; canonicalisation; salts and
counter-ions; Morgan/ECFP circular fingerprints (radius 2 ≈ ECFP4);
Tanimoto coefficient `|a∩b| / |a∪b|`; limits (biologics have no SMILES).

**Meet it in:** `similarity.py::morgan_tanimoto`; `02_build_features.py` (PubChem lookup).

**Self-check**
1. Why do we keep only the largest fragment of a molecule before fingerprinting?
2. Why is chemical similarity missing for some drugs, and how does the model cope?

**Resources:** RDKit "Getting Started in Python" docs; Rogers & Hahn (2010),
"Extended-connectivity fingerprints"; TeachOpenCADD (free notebooks, talktorials T001-T004).

### D3. Biomedical ontologies and semantic similarity
**Learn:** ontologies as DAGs (is_a); OBO format; MeSH trees; Disease Ontology;
MONDO (a merged disease ontology) and cross-references (xrefs); Wang et al.
(2007) semantic similarity; Resnik/Lin information-content similarity.

**Meet it in:** `similarity.py::wang_similarity`, `02_build_features.py::parse_mondo`.

**Self-check**
1. In Wang's method, why does a close common ancestor contribute more than the root?
2. Why can one OMIM disease map to several MONDO terms, and how do we handle that?

**Resources:** Wang et al. (2007), "A new method to measure the semantic similarity
of GO terms", *Bioinformatics*; Vasilevsky et al. (2022), "Mondo: unifying diseases
for the world", *medRxiv*; the MeSH browser (meshb.nlm.nih.gov).

### D4. Biomedical databases
**Learn:** what each source contains and its identifiers:
DrugBank (DBxxxxx), PubChem (CID/SID, InChIKey), OMIM (6-digit MIM numbers),
MeSH (Dxxxxxx / Cxxxxxx), CTD (curated vs inferred links, DirectEvidence),
MedGen (UMLS CUIs), ClinicalTrials.gov; licensing (DrugBank academic licence);
entity resolution across databases.

**Meet it in:** `02_build_features.py` - the whole file is entity resolution.

**Self-check**
1. Why is InChIKey a good key to join chemical databases?
2. Why did we exclude CTD *inferred* and *therapeutic* gene-disease links from the model's inputs?

**Resources:** each database's "About/Downloads" page; Davis et al. (2023),
"Comparative Toxicogenomics Database (CTD): update 2023", *Nucleic Acids Research*.

---

## Track E - Research practice

### E1. Data leakage and experimental design
**Learn:** train/test contamination; tuning on test data; transductive leakage
in graphs (test links as message-passing edges); "similarity computed from
labels" leakage; fair baselines (same splits, same information);
ablation design (change one thing at a time); repeated runs and seeds.

**Self-check**
1. List three ways information about a test link could leak into training in this project, and how the code prevents each.
2. Why do we also run an ablation "benchmark similarities only"?

**Resources:** Kapoor & Narayanan (2023), "Leakage and the reproducibility crisis in
machine-learning-based science", *Patterns*.

### E2. Interpretability and its limits
**Learn:** attention weights vs faithful explanations; occlusion/perturbation
attribution; global vs local explanations.

**Meet it in:** `MVHGATMethod.view_attention` (attention) and `.occlusion`
(perturbation) - we report both because attention alone is not proof.

**Self-check:** Why might a relation get high attention but low occlusion impact?

**Resources:** Jain & Wallace (2019), "Attention is not Explanation";
Wiegreffe & Pinter (2019), "Attention is not not Explanation"; Ying et al. (2019), "GNNExplainer".

### E3. Reproducibility and writing up results
**Learn:** fixed seeds; recording versions of data (the date you downloaded CTD);
reporting mean ± std; honest comparison with baselines (reimplementations vs
published numbers); writing a methods section that someone else can re-run.

**Resources:** the DRAGNN (Meng et al. 2024) and HEDDI-Net (Su et al. 2025) papers in
your proposal are good templates for the structure of the Results section.

---

## Map: file → units to study first

| File | Units |
|---|---|
| `scripts/01_download_data.py` | A1, D4 |
| `scripts/02_build_features.py` | A1, D2, D3, D4 |
| `src/drepo/similarity.py` | A2, D2, D3 |
| `src/drepo/data.py` | A1, C1 |
| `src/drepo/model.py` | B3, C3, C4, C5, C6 |
| `src/drepo/methods.py` (MV-HGAT) | B1, B3, C6 |
| `src/drepo/methods.py` (baselines) | A2, B4, C2, C3 |
| `src/drepo/evaluation.py` | B1, B2, E1 |
| `scripts/04_ablation.py`, `05_sensitivity.py` | B1, E1 |
| `scripts/06_case_study.py` | D1, D4, E2 |
| `scripts/07_cross_dataset.py` | B2, E1 |
