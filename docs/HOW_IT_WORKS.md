# How this project works

*A plain-language walkthrough of the whole pipeline, the model, the evaluation,
and every assumption it makes. Read it top to bottom once; afterwards use the
headings as a reference. Terms you may not know are explained in
[PREREQUISITES.md](PREREQUISITES.md).*

---

## 1. The problem in one paragraph

We have a table: rows are **drugs**, columns are **diseases**, and a cell is 1
if the drug is a known treatment (an *indication*) for that disease. Almost all
cells are 0, but a 0 means **"not known"**, not **"does not work"**. *Drug
repositioning* asks: which of those 0s are probably 1s we have not discovered yet?
That is exactly the problem Netflix solves when it guesses which films you would
like: diseases are the "users", drugs are the "items". We solve it by building a
**graph** of drugs and diseases (plus genes), learning a vector ("embedding") for
every drug and disease with a **graph neural network**, and scoring each
drug-disease pair by how well their vectors match.

The key biological intuition (the *guilt-by-association* assumption):
**similar drugs tend to treat similar diseases.** "Similar" can mean similar
chemical structure, similar target genes, similar symptoms, or a nearby position
in a disease classification. The model's job is to learn *which* kinds of
similarity to trust, for which drugs and diseases.

---

## 2. The data

### 2.1 The two benchmarks (what you downloaded)

| | Fdataset | Cdataset |
|---|---|---|
| Drugs (DrugBank IDs) | **593** | **663** |
| Diseases (OMIM IDs) | **313** | **409** |
| Known drug-disease links | **1,933** | **2,532** |
| Density (share of cells that are 1) | 1.04% | 0.93% |
| Origin | Gottlieb et al. 2011 (PREDICT) | Luo et al. 2016 (MBiRW) |

> **Correction to the proposal:** the proposal's table says Fdataset has
> "593 associations, 313 drugs, 313 diseases" and Cdataset "663 associations,
> 409 drugs, 663 diseases". The true numbers are in the table above (verified
> from the files). Please fix this in the proposal.

Each benchmark has three matrices:

* `DiDrA.csv` - the association matrix. **Careful: in the files, rows are
  diseases and columns are drugs** (313 x 593), even though the Fdataset README
  says the opposite. The code transposes it to drugs x diseases.
* `DrugSim.csv` - drug-drug **chemical** similarity (computed by the original
  authors from DrugBank structures with CDK fingerprints + Tanimoto).
* `DiseaseSim.csv` - disease-disease **phenotype** similarity from MimMiner
  (text-mining of OMIM disease descriptions).

Your CSV copies did **not** contain the drug and disease IDs. Without IDs we
could not link to any other database or name the drugs in the case studies,
so we downloaded the original MATLAB files (`Fdataset.mat`, `Cdataset.mat`)
from the DRHGCN GitHub repository. We verified they contain *exactly* the same
matrices as your CSVs (maximum difference 0 for Fdataset, 5e-6 rounding for
Cdataset) plus the IDs (`Wrname` = DrugBank, `Wdname` = OMIM).

### 2.2 What we downloaded, and how it replaces what was unavailable

| Proposal source | Status | What we used instead / in addition |
|---|---|---|
| **DrugBank v5.x** (SMILES, targets) | Full XML needs an approved academic account and was unavailable | **PubChem**: DrugBank deposits its own records in PubChem, so `DrugBank ID → PubChem substance → compound` gives the same structure (SMILES, InChIKey) and name. Targets come from **CTD** chemical-gene interactions instead. |
| **CTD** | Available (`ctdbase.org/reports/`), you just need the direct file links | Chemical vocabulary, MEDIC disease vocabulary, chemical-gene interactions, curated gene-disease links, curated chemical-disease links (validation only). The 3.2 GB *inferred* gene-disease file was **not** used. |
| **MeSH / Disease Ontology** | Both available | We used **MONDO** (`mondo.obo`), the ontology that merges Disease Ontology, OMIM, Orphanet and MeSH. It covers ~99% of our OMIM diseases; DO alone covers only ~59%. `doid.obo` is downloaded too. |
| (not in proposal) **NCBI MedGen** | Free | Gives official names for OMIM IDs (OMIM's own name file needs a licence key) and extra OMIM → MeSH links. |
| **CMap / LINCS L1000** | Optional in proposal | **Not used.** The files are tens of GB, need heavy signature processing, and cover only part of our drugs. It is left as future work (section 9). |
| **ClinicalTrials.gov** | Public API | Queried live in the case studies to check predictions. |

Everything is fetched by `scripts/01_download_data.py` (plus PubChem calls in
`02_build_features.py`). None of it needs a login.

### 2.3 Entity resolution - joining the databases

Different databases name the same thing differently, so the first real step is
to translate identifiers:

```
Drugs:     DrugBank ID ──PubChem (DrugBank's deposited substance)──► PubChem CID, SMILES, InChIKey, name
           PubChem CID / InChIKey / name ──CTD chemical vocabulary──► CTD (MeSH) chemical ID ──► genes

Diseases:  OMIM ID ──MedGen──► name, MeSH ID
           OMIM ID ──MONDO xrefs──► MONDO term(s) ──► position in disease hierarchy
           OMIM ID / MeSH ID ──CTD MEDIC──► CTD disease ID ──► genes
```

The readable results are in `data/interim/drugs_all.csv` and
`data/interim/diseases_all.csv`. Open them in Excel to check any mapping.

**How much could be mapped** (682 unique drugs and 415 unique diseases across both benchmarks):

| | Fdataset | Cdataset |
|---|---|---|
| Drugs with a chemical structure (PubChem SMILES) | 97.5% | 98.9% |
| Drugs with CTD human gene interactions | 84.5% | 83.9% |
| Diseases placed in the MONDO hierarchy | 97.1% | 97.3% |
| Diseases with CTD curated (marker/mechanism) genes | 51.1% | 54.0% |

The ~2% of drugs without a structure are biologics (proteins, heparins,
conjugated estrogens) or retired DrugBank IDs (DB00510, DB01258, DB01402, which
keep their ID as their name). Disease gene coverage is only about half because
many 2011-era OMIM entries are rare syndromes with no curated genes in CTD.

---

## 3. From data to a graph

### 3.1 Six similarity "views"

A *view* is one way of saying how similar two drugs (or two diseases) are.

| View | Node type | How it is computed | What it captures |
|---|---|---|---|
| `chem_cdk` | drug | given in the benchmark (CDK fingerprints, Tanimoto) | chemical structure |
| `chem_ecfp` | drug | **we compute**: RDKit Morgan fingerprint (radius 2 = ECFP4, 2048 bits) on PubChem SMILES, Tanimoto | chemical structure, a modern fingerprint |
| `gene_r` | drug | **we compute**: Jaccard overlap of the drugs' CTD human gene sets | shared targets / biological effect |
| `pheno_mim` | disease | given in the benchmark (MimMiner) | similar symptoms |
| `sem_mondo` | disease | **we compute**: Wang semantic similarity on the MONDO hierarchy | similar place in disease classification |
| `gene_d` | disease | **we compute**: Jaccard overlap of CTD curated disease genes | shared molecular mechanism |

**Formulas, in words:**

* **Tanimoto**: each molecule becomes a 2048-bit fingerprint (bit = "this small
  substructure is present"). Similarity = (bits both have) / (bits either has).
* **Jaccard**: the same formula on gene sets: shared genes / all genes of the two.
* **Wang semantic similarity**: every disease sits in a tree-like hierarchy
  (e.g. *Alzheimer disease* → *dementia* → *neurodegenerative disease* → ...).
  Each ancestor gets a weight that halves at every step up (1, 0.5, 0.25, ...).
  Two diseases are similar if they share ancestors, especially *close* ones:
  similarity = (weights of shared ancestors, from both sides) / (all weights of both).

**Missing values.** A biologic drug (a protein) has no SMILES, an old OMIM
entry may have no genes in CTD, and so on. In such a view that entity simply has
**no neighbours**. The model still has its other views, and the view attention
learns to rely on those. We never invent a similarity value.

### 3.2 The heterogeneous graph

```
            chem_cdk / chem_ecfp / gene_r                pheno_mim / sem_mondo / gene_d
              (k-nearest-neighbour edges)                  (k-nearest-neighbour edges)
                ┌───────────┐                                  ┌───────────┐
                │  DRUG  ●──┼──●   known indication (assoc)  ●──┼──●  DISEASE│
                │  nodes ●──┼──┼────────────────────────────────┼──● nodes  │
                │        ●  │  └─── gene bridge (drug→gene→disease) ──●     │
                └───────────┘                                  └───────────┘
                          \                                      /
                           \────────  GENE nodes (CTD)  ────────/
```

* **k-nearest-neighbour (kNN) sparsification.** A similarity matrix links every
  drug to every other drug with some weight. That is too dense: message passing
  would average everything into mush. For each view we keep each node's
  **k = 10 most similar** neighbours (and itself).
* **Known indications** (training fold only!) are drug-disease edges.
* **Gene nodes** are handled as *meta-paths*, the standard trick for an
  intermediate node type (as in the HAN model):
  drug-gene-drug = the `gene_r` view, disease-gene-disease = the `gene_d` view,
  and **drug-gene-disease** = a direct "gene bridge" edge between a drug and the
  top-10 diseases whose gene profile overlaps most with the drug's (cosine
  similarity of gene profiles). In principle this gives even a disease with no
  known drugs a path to plausible drugs. **In practice the bridge carried almost
  no signal** (AUC 0.52-0.57 on its own, section 10), so it is **off by default**
  and tested as "+ bridge" in the ablation.

So the default graph has 2 node types and 8 relation types:
3 drug views + 3 disease views + known links (2 directions); the optional
gene bridge adds 2 more.

---

## 4. The model: MV-HGAT

*Multi-View Heterogeneous Graph Attention Network.* Code: `src/drepo/model.py`.

### 4.1 Input features

Each drug starts with a long vector: its rows in all drug similarity matrices
plus its row of *visible* known links. Each disease likewise. A linear layer
compresses this to 64 numbers. This is the starting embedding `h⁰`.

### 4.2 One layer = two levels of attention

**Level 1: node-level attention (GAT), within each relation.**
For a drug *i* and relation *r* (say `chem_ecfp`), look at its neighbours *j*
in that relation and ask "how relevant is *j* to *i*?":

```
e_ij = LeakyReLU( a_dstᵀ W h_i  +  a_srcᵀ W h_j )      relevance score
α_ij = softmax over neighbours j of e_ij               attention weights (sum to 1)
m_i^r = Σ_j α_ij · W h_j                               message from relation r
```

This is done with 4 independent "heads" whose outputs are concatenated, as in
the original GAT paper. Result: one message per relation, e.g. a drug gets 5
messages (3 drug views, known links, gene bridge).

**Level 2: view-level attention, across relations.** This is the interpretability hook.
The node now decides how much to trust each message:

```
s_i^r = qᵀ tanh(P m_i^r + b)        score of relation r for node i
β_i^r = softmax over r of s_i^r      VIEW WEIGHTS: how much node i relies on relation r
z_i   = Σ_r β_i^r · m_i^r
h_i'  = ELU( LayerNorm( z_i + skip(h_i) ) )
```

Unlike the original HAN, **β is computed per node**, so every drug and disease
has its own profile, e.g. "this disease relies 45% on its gene view and 30% on
phenotype similarity".

We stack **2 layers** (each node sees neighbours of neighbours) and concatenate
`h⁰, h¹, h²` (*jumping knowledge*: keeps both local and wider information).

### 4.3 Decoder: GNN term + multi-view propagation head

The final score has two parts:

```
logit(i, j) = gate(i, j) · h_iᵀ W h_j            ← GNN term (bilinear)
            + Σ_v  w_v · P_v[i, j]  +  b          ← multi-view propagation head
score(i, j) = sigmoid(logit)
```

**Propagation term `P_v`.** For a drug view v (say `chem_ecfp`), `P_v[i, j]` is
"what share of drug i's 10 most similar drugs are known to treat disease j"
(similarity-weighted). For a disease view u (say `sem_mondo`), `P_u[i, j]` is
"how strongly do disease j's 10 most similar diseases use drug i". This is
the guilt-by-association idea written out directly, one term per view. Each
view gets a learned weight `w_v ≥ 0`.

**Why add it?** While building the model we found that this simple
neighbour propagation is *very* strong on these benchmarks (it is essentially
what MBiRW does), and a pure GNN struggled to match it, especially in cold
start. With the head, the GNN only has to learn what propagation misses.

**Degree gate.** `gate(i, j) = σ(a + b·log(1 + #links of i)) · σ(c + d·log(1 + #links of j))`,
with a, b, c, d learned. A disease with no known drugs (cold start) gets a small
gate, so its score relies on the similarity views. A well-studied disease gets
a large gate, so the GNN's collaborative signal counts.

`W` is a learned matrix, so the GNN term can learn that, say, "chemical
dimension 3 of a drug matches mechanism dimension 7 of a disease".

### 4.4 Training: the tricks that matter most

* **Loss:** binary cross-entropy. Positives = known links; negatives = randomly
  sampled unknown cells (2 per positive, re-sampled each epoch).
* **Hidden-link supervision (crucial).** Each epoch we randomly **hide 20% of
  the training links** from the graph, the input features *and* the
  propagation terms, and compute the loss **only on those hidden links** (plus
  negatives). Why: at test time, the link we must predict is never visible. If
  we trained on links the model can *see*, it would learn the shortcut "score
  high if the edge is already there", which is useless on test pairs. In our
  tuning this single change raised validation AUC from 0.71 to 0.92.
* **Cold-start practice.** Each epoch, 10% of diseases additionally lose **all**
  their links, as the held-out disease does in leave-one-disease-out testing.
  This teaches the degree gate and the view weights what to do when a disease
  has no known drugs (cold-start AUPR on validation: 0.055 → 0.12 from this alone).
* **Optimiser:** Adam, learning rate 0.002, weight decay 5e-4, dropout 0.2,
  600 epochs. These were chosen on a *validation split*, never on test folds
  (`scripts/05_sensitivity.py`).

### 4.5 Interpretability: three complementary tools

1. **Global view weights `w_v`** (`MVHGATMethod.view_weights`): how much each
   similarity view contributes to scores overall. Because the propagation head
   is additive, `w_v · P_v[i, j]` is *literally* view v's share of the logit for
   pair (i, j). This is a faithful explanation, not an approximation.
2. **Per-node view attention β** (`MVHGATMethod.view_attention`): inside the GNN,
   which relations each individual drug/disease leaned on.
3. **Occlusion** (`MVHGATMethod.occlusion`): for a specific prediction, switch
   off one evidence source (its graph relation *and* its propagation term) and
   measure how much the probability drops. The case-study tables report the
   top-2 sources for every predicted drug.

Attention weights alone are not proof of causation (Jain & Wallace 2019), so
the paper should lead with (1) and (3) and use (2) as supporting detail.

---

## 5. How we evaluate

Code: `src/drepo/evaluation.py`.

### 5.1 Warm-start k-fold cross-validation (5-fold and 10-fold)

1. Shuffle the known links (1s) into k folds, **and** the unknown cells (0s) into k folds.
2. For fold f: hide fold f's 1s, train on the rest, using negatives only from the other folds' 0s.
3. Test set = fold f's 1s + fold f's 0s. Compute AUC and AUPR.
4. Repeat with several random shuffles; report **mean ± standard deviation** over all folds.

"Warm start" means every test drug and disease still has other known links in training.

### 5.2 Cold-start: leave-one-disease-out (LODO)

For each disease, hide **all** of its known drugs and ask the model to rank all
drugs for it. The disease looks brand-new to the model. Only the disease
similarity views (and the gene bridge, if enabled) can help. This is the realistic scenario for a newly
characterised disease and is much harder. We pool all diseases' predictions,
then compute AUC/AUPR; we also report the average per-disease AUC.

### 5.3 Metrics, and why AUPR is the honest one

* **AUC (ROC)**: the probability that a random true link is scored above a random
  non-link. 0.5 = random, 1 = perfect. With 99% negatives, AUC can look great
  even when the top of the ranking is mediocre.
* **AUPR** (average precision): how precise the ranking is as you go down the
  list. Random ≈ 0.01 here (the positive rate). An AUPR of 0.5 is therefore
  ~50x better than random. **When comparing methods, look at AUPR first.**

### 5.4 Baselines

All five are re-implemented in `src/drepo/methods.py`, use **only the two
benchmark similarity matrices** (as in their papers), are evaluated on
**exactly the same splits**, and had their key hyper-parameters tuned on the
same validation split as our model.

| Baseline | Idea in one line |
|---|---|
| MBiRW (Luo 2016) | random walks that alternate between the drug and disease similarity networks |
| DRRS (Luo 2018) | stack all matrices into one big matrix and fill its gaps assuming it is low-rank (singular value thresholding) |
| SCMFDD (Zhang 2018) | matrix factorisation `A ≈ UVᵀ` where similar drugs/diseases are forced to have similar factors |
| NIMCGCN (Li 2020) | separate GCNs on the drug and disease similarity graphs + inductive matrix completion |
| LAGCN (Yu 2021) | GCN on the combined drug-disease graph with attention over layers |

> **Be honest in the paper:** these are *our re-implementations*, not the
> authors' code. Published numbers can differ because of different splits,
> preprocessing and tuning. If you also quote published numbers, say so and
> keep them in a separate table.

### 5.5 The rest of the Results section

* **Ablation** (`04_ablation.py`): remove one component at a time (propagation
  head, degree gate, cold-start practice, hidden-link supervision, view
  attention, gene views, each new view) or add the gene bridge, and re-run CV.
  The variant "benchmark similarities only" is important: it shows how much
  comes from the *architecture* vs. from the *extra data*.
* **Sensitivity** (`05_sensitivity.py`): vary embedding size, layers, negative
  ratio, k, hiding rate, one at a time, on the validation split.
* **Case studies** (`06_case_study.py`): train on all links, list the top-10
  *new* drugs for chosen diseases, and check each one against CTD curated
  evidence, the other benchmark, and ClinicalTrials.gov.
* **Cross-dataset** (`07_cross_dataset.py`): train on all of Fdataset and test
  whether it recovers the links that only Cdataset has (for shared drugs and
  diseases), and vice versa. This is a genuinely external test set.

All numbers are collected in `results/RESULTS.md`; figures in `results/figures/`.

---

## 6. The assumptions (read this before writing the paper)

**About the data**

1. **Unknown = negative for training and testing.** Every 0 is treated as
   "does not treat", although some are undiscovered true links. This makes all
   reported scores *pessimistic*, and is the universal convention in this
   field (PREDICT, MBiRW, DRRS, LAGCN all do it).
2. **The benchmarks are correct and complete enough.** Fdataset/Cdataset were
   built 2011-2016; indications approved since then appear as 0s.
3. **Guilt by association.** Similar drugs treat similar diseases. This holds
   on average, but not for drugs whose effect depends on something the views do
   not capture (dose, delivery, pharmacokinetics).
4. **PubChem's copy of a DrugBank record has the same structure** as DrugBank
   itself. It is DrugBank's own deposit, so this is safe, but the PubChem
   compound is the *parent* compound (salts are stripped).
5. **CTD gene sets are a fair proxy for targets / disease mechanism.** CTD
   chemical-gene interactions include expression changes, not only direct
   binding, so `gene_r` measures *shared biological effect* more than *shared
   targets*. Well-studied drugs have many more genes (literature bias).
6. **Mapping OMIM → MONDO/MeSH/CTD is correct.** Old OMIM entries can be
   retired or merged; when one OMIM ID maps to several terms we take the
   maximum similarity.

**About leakage (what we deliberately avoided)**

7. No similarity view is computed from the association matrix, so test links
   cannot leak through similarities.
8. We **did not** use CTD *inferred* gene-disease links, nor gene-disease links
   with *therapeutic* evidence, as model input. Those are partly derived from
   known drug-disease treatments, i.e. from the answer.
9. CTD curated chemical-disease links are used **only** to check case-study
   predictions, never for training.
10. Hyper-parameters were tuned on a validation split carved from the data,
    not on the test folds. (Strictly, the validation split overlaps with the
    data later used in CV folds. This is common practice, but a fully nested CV
    would be stricter; mention it as a limitation.)

**About the model**

11. **Transductive.** The model embeds the drugs and diseases in the graph. A
    completely new drug can only be scored after adding it (with its
    similarities) to the graph and retraining.
12. **kNN with k = 10** keeps the strongest similarities (for both the graph and
    the propagation head); weaker but real relations are dropped.
13. **Attention ≠ explanation.** β weights describe what the GNN used, not
    what is biologically true. The propagation-head shares and occlusion are
    faithful *to the model*, but still not proof of biology.
14. **A high score is a hypothesis, not a recommendation.** Case-study
    predictions need literature, clinical-trial and experimental follow-up.

---

## 7. Running everything

```powershell
cd C:\Users\Abhineet Anand\Desktop\DrugRepositioning
.venv\Scripts\activate

python scripts/01_download_data.py      # ~5 min, ~260 MB
python scripts/02_build_features.py     # ~30 min first time (PubChem), ~3 min after (cached)

# everything else, in order (~4 h on an RTX 3050); logs go to results/logs/
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

Or run any single step, e.g.

```powershell
python scripts/03_evaluate.py --dataset F --protocol cv5 --repeats 5
python scripts/03_evaluate.py --dataset F --protocol lodo                   # ALL 313 diseases (~3 h)
python scripts/03_evaluate.py --dataset F --protocol lodo --lodo-subset 100 # what run_all uses
python scripts/04_ablation.py --dataset C --repeats 1
python scripts/06_case_study.py --dataset C --omim 104300,114480,176807
python scripts/08_make_figures.py                                           # tables + figures
```

`run_all.ps1` evaluates LODO on the same random 100 diseases for every method
(seeded), to keep run time reasonable. For the final paper, run the full LODO
overnight by dropping `--lodo-subset`.

Find a disease's OMIM ID for a case study:
`python scripts/06_case_study.py --dataset C --search parkinson`

---

## 8. Where each proposal item lives

| Proposal | Implemented in |
|---|---|
| Obj. 1 / Step 1-3: heterogeneous network | `01_download_data.py`, `02_build_features.py`, `similarity.py`, `MVHGATMethod.build` |
| Obj. 2 / Step 4: GNN + link-prediction decoder | `model.py` (GAT encoder, bilinear decoder + propagation head) |
| Obj. 3: interpretability over views | `view_weights` (w_v), `ViewAttention` (β), `MVHGATMethod.occlusion` |
| Obj. 4 / Step 5: 5-fold, 10-fold, cold-start | `evaluation.py`, `03_evaluate.py` |
| Step 6: baselines + ablation | `methods.py` (5 baselines), `04_ablation.py` |
| Results 5.1-5.3 | `03_evaluate.py` + `08_make_figures.py` |
| Results 5.4 | `04_ablation.py` |
| Results 5.5 | `05_sensitivity.py` |
| Results 5.6 / Step 7 | `06_case_study.py` |
| Results 5.7 | `07_cross_dataset.py` |

## 9. Deviations from the proposal, and future work

* **MONDO instead of MeSH/DO** for semantic similarity: much better coverage
  (MONDO integrates both). MeSH IDs are still mapped and used for CTD genes.
* **PubChem + CTD instead of the DrugBank XML.** Same structures; targets come
  from CTD. If you later get DrugBank access, add a `target_r` view (Jaccard of
  DrugBank targets) in `02_build_features.py`. The model needs no change; a new
  view is just one more relation.
* **Gene nodes as meta-paths** rather than explicit gene embeddings. This keeps
  the graph small (no 20,000 gene nodes) and is the standard HAN approach. An
  explicit tripartite variant is possible future work.
* **Gene bridge off by default.** It is implemented and tested in the
  ablation, but on these benchmarks it adds almost no signal (section 10).
* **Decoder extended** with a multi-view propagation head and a degree gate
  (section 4.3). The proposal said "bilinear or MLP decoder"; this is a
  bilinear decoder plus an interpretable additive term. Describe it as a
  contribution, and use the ablation to justify it.
* **LINCS L1000 not included** (optional in the proposal). To add it later,
  compute a drug-drug correlation of L1000 consensus signatures and save it as a
  fourth drug view in `02_build_features.py`; nothing else changes.
* **Baselines re-implemented**, not the authors' original code.

## 10. What we learned while building it (useful for the Discussion section)

These findings came from tuning on a **validation split** (20% of links held
out, never the test folds), on Fdataset unless stated.

1. **Training on visible links is a trap.** A GNN supervised on links it can see
   in its own input graph learns to detect edges, not to predict them
   (validation AUC 0.71). Supervising only on links hidden each epoch fixed it
   (0.92, then 0.945 after tuning).
2. **The new views carry real signal** when tested alone (no learning; score a
   pair by the links of the 10 nearest neighbours in that view):

   | View (alone) | Fdataset AUPR | Cdataset AUPR |
   |---|---|---|
   | Cold start, disease side: `pheno_mim` (benchmark) | 0.149 | 0.318 |
   | Cold start: `sem_mondo` (new, MONDO) | 0.106 | 0.222 |
   | Cold start: `pheno_mim + sem_mondo` | **0.175** | **0.322** |
   | Cold start: `gene_d` (new, CTD) | 0.057 | 0.126 |
   | Cold start: `gene_bridge` (new, CTD) | 0.011 (≈ random) | 0.014 (≈ random) |
   | Warm, drug side: `chem_cdk` (benchmark) | 0.061 | 0.047 |
   | Warm: `chem_ecfp` (new, RDKit) | **0.091** | **0.068** |
   | Warm: `gene_r` (new, CTD) | 0.031 | 0.024 |

   So the **MONDO semantic view complements phenotype similarity**, the **modern
   ECFP fingerprint beats the benchmark's own CDK fingerprint**, and **CTD gene
   evidence is weak** for these indication benchmarks (curation bias, half the
   diseases have no genes, and CTD "interactions" are mostly expression changes
   rather than drug targets).
3. **A plain GNN does not exploit extra views automatically.** Concatenating more
   similarity views into a plain GNN slightly *hurt* (more parameters, same
   data). The propagation head is what lets the views' signal through.
4. **Cold start has to be trained for.** Without cold-start practice the model
   had validation cold-start AUPR 0.055; with practice, the propagation head and
   the degree gate, about 0.15 (AUC 0.85). MBiRW, a simple random walk, remains a
   strong cold-start baseline: compare against it honestly.
