# Unit E1 — Data Leakage and Experimental Design

*How to make sure that a number like "AUPR 0.50" measures what the model can really do, and how to compare models and variants so that the comparison means something.*

---

## 0. About this chapter

**Prerequisites.**
- **B1 (Supervised learning)**: training/validation/test sets, k-fold cross-validation, over-fitting, hyper-parameters.
- **B2 (Evaluation and imbalanced data)**: ROC AUC, average precision (AUPR), warm-start CV versus leave-one-disease-out (LODO), mean ± std over folds, the "unknown = negative" assumption. This chapter builds directly on B2's Section 3.14.
- **A3 (Probability and statistics)**: expectation, variance, the normal and $t$ distributions, hypothesis tests and $p$-values.
- **C6 (Link prediction)**: training a model to score drug–disease pairs, negative sampling.
- **D4 (Biomedical databases)**: what CTD curated/inferred and therapeutic links are.

**Estimated study time.** 9–11 hours: about 4 h on the theory (Sections 1–3), 2.5 h running and modifying the simulations (Section 4), 1.5 h on the project walk-through (Section 5), 2–3 h on the exercises.

**How to run the code.** Every block in Section 4 and in the exercise solutions is a self-contained simulation that runs on the CPU in seconds to about a minute, with the project environment (`.venv\Scripts\python.exe`). The outputs shown are the real outputs. Nothing touches the GPU or the project's files.

**Learning objectives.** After this unit you will be able to:

1. Define data leakage precisely (learn–predict separation and feature legitimacy) and explain why it almost always makes results look *better*.
2. Classify any leakage scenario using Kapoor & Narayanan's eight-way taxonomy (L1.1–L1.4, L2, L3.1–L3.3) and give a drug-repositioning example of each.
3. Explain the leakage channels specific to graph link prediction — test edges in the message-passing graph, features and similarities computed from the association matrix, negatives sampled from test cells, normalizations computed with test edges — and demonstrate each with a simulation.
4. Distinguish leakage from train/test *mismatch* (the "edge-detector" shortcut) and explain how hidden-link supervision fixes the latter.
5. Explain why tuning on test data inflates results, quantify the inflation with the expected-maximum formula, and design a nested cross-validation.
6. Design fair baseline comparisons (same splits, same information, same tuning budget) and ablation studies (one change at a time, positive and negative controls, common random numbers).
7. Quantify run-to-run variance from seeds and splits, and report it correctly.
8. Compare two methods statistically across folds with paired tests, the Nadeau–Bengio correction, the Wilcoxon signed-rank test and effect sizes, and correct for multiple comparisons (Holm).
9. Write a reporting checklist (model info sheet) for the project.
10. Audit this project's code for every leakage channel, name the code line that prevents each one, and explain its acknowledged limitation and its open question (the hidden-link-supervision ablation).

---

## 1. Motivation: a perfect score on pure noise

Here is a result from Section 4.3. Take a drug × disease matrix of 300 × 150 cells in which every cell is a link with probability 3%, **completely at random**. There is nothing to learn: no method can honestly beat AUC 0.5. Now evaluate three link predictors with 5-fold cross-validation, exactly like the project's `run_kfold`:

```text
  clean (train links only)   AUC 0.494  AUPR 0.030
  similarity from full A     AUC 0.993  AUPR 0.933
  test edges in graph        AUC 1.000  AUPR 0.996
```

The second method computes drug–drug similarity from the *whole* association matrix before splitting; the third lets its graph propagation run over a graph that still contains the test edges. Both are one-line mistakes, and both turn noise into a near-perfect predictor. In real papers the inflation is usually smaller and therefore harder to spot — a few points of AUPR that make a new method look better than the baselines.

The project contains a subtler, honest-to-goodness experimental-design question as well. The ablation study (`scripts/04_ablation.py`, Fdataset, 5 folds) produced:

| Variant | AUPR (mean of 5 test folds) |
|---|---|
| Full MV-HGAT (hidden-link supervision **on**) | 0.500 |
| w/o hidden-link supervision | **0.546** |

The variant *without* one of the model's design features scored better, on all five folds. Should we switch the feature off? If we do, we will have chosen a design **because it scored best on the test folds** — which is tuning on test data, exactly the error Section 2.4 quantifies. The correct response is to settle the question on a *validation* split, with several seeds, and only then (if the result holds) change the default and re-run the test evaluation once. Section 5.6 walks through that protocol.

These two examples are the subject of this unit: **leakage**, which makes evaluation numbers lie, and **experimental design**, which decides whether a comparison between numbers means anything.

---

## 2. Core theory

### 2.1 What we are trying to estimate

When we report "MV-HGAT has AUPR 0.49 on Fdataset", we are making a claim about a *procedure* $P$ (preprocess, build graph, train, predict) applied to data like ours:

$$\theta(P) \;=\; \mathbb{E}\big[\,\text{metric}\big(P(\mathcal{D}_\text{train}),\ \mathcal{D}_\text{new}\big)\big],$$

the expected performance of the model that $P$ builds from the information available at prediction time, when it is used on **new** cases from the population we care about. Cross-validation estimates $\theta(P)$ by repeatedly pretending that part of the data is new. The estimate is only valid if the pretence is perfect: during training, nothing may depend on the held-out part, and the held-out part must look like the new cases of the claim.

**Definition (data leakage).** *Leakage* is any flow of information into the model-building procedure that would not be available when the model is actually used, or any mismatch between the evaluation data and the population of the scientific claim, such that the evaluation estimate becomes biased. Kaufman et al. (2012) phrase the first part as two principles:

- **Learn–predict separation:** the procedure that builds the model must not see the target values of the examples it will be evaluated on.
- **Legitimacy:** every feature must be something that would legitimately be known, at prediction time, for the cases we want to predict; a feature that is a proxy of the target (or is derived from it) is illegitimate even if it is "in the data".

**Why the bias is almost always optimistic.** Training is an optimization: the procedure uses whatever information correlates with the labels. If information about the test labels is available, it will be used, because it helps the training objective — and it helps the test score even more, because it *is* the test labels. Leakage rarely makes results worse; that is why it survives review. A good habit: **be suspicious of results that are too good, and also of improvements that are too easy.**

**Prediction time in drug repositioning.** In our setting, a "new case" is a drug–disease pair whose link is unknown today. At prediction time we legitimately know: all *currently known* indications (the training links), drug structures, disease ontologies, curated biology (genes, pathways), the literature up to today. We do **not** know the link we are predicting. So in cross-validation, every quantity derived from links must be derived from the *training* links of that fold only, and no feature may be a disguised indication.

### 2.2 A taxonomy of leakage (Kapoor & Narayanan, 2023)

Kapoor and Narayanan surveyed fields that adopted machine learning and found leakage reported in 17 fields, affecting 294 papers; in their own reproduction of civil-war-prediction studies, correcting leakage removed the apparent advantage of complex models over logistic regression. They organized the errors into eight types. Here is each, with what it would look like in drug repositioning.

**[L1] No clean separation of training and test data.**

| Type | Meaning | Drug-repositioning example |
|---|---|---|
| **L1.1 No test set** | Evaluating on the training data | Training on all links (as `06_case_study.py` does — legitimately, for case studies) and then reporting the AUC of the known links as "performance" |
| **L1.2 Pre-processing on train + test** | Any data-dependent preprocessing fitted on all data: imputation, normalization, oversampling, kernels | Computing a Gaussian-interaction-profile (GIP) kernel, a degree normalization or a random-walk matrix from the **full** association matrix before splitting (Exercise 10 shows it on noise) |
| **L1.3 Feature selection on train + test** | Choosing features using their association with labels on all data | Selecting the genes "most associated with indications" on the full matrix, then cross-validating (Section 4.1 shows this turns noise into AUC 0.95) |
| **L1.4 Duplicates** | The same unit in train and test | Two DrugBank IDs for the same active compound; disease entries that are the same disease under two OMIM numbers; identical drug–disease pairs present in both of two benchmarks used as train and test |

**[L2] Illegitimate features.** A feature that would not be available at prediction time, or that is a proxy of the outcome. Example: CTD *therapeutic* chemical–disease links, ChEMBL or DrugBank indication tables, clinical-trial counts, literature co-mention counts of a drug and a disease — all of these partly *are* the indication. Kapoor & Narayanan deliberately give no sub-types: deciding legitimacy needs domain knowledge (Unit D4 Section 8 is such an argument).

**[L3] The test set is not drawn from the distribution of scientific interest.**

| Type | Meaning | Drug-repositioning example |
|---|---|---|
| **L3.1 Temporal leakage** | Test data from *before* (or contemporaneous with) the training data, when the claim is about predicting the future | Using CTD gene data curated in 2026 — including papers written *because* a drug was found to treat a disease — as features to "predict" indications known since 2011 |
| **L3.2 Non-independence between train and test** | Test samples related to training samples (same patient, same unit) when the claim is about new units | Enantiomer pairs (omeprazole/esomeprazole) or disease subtypes (Alzheimer types 1–4) split across train and test; random pair-wise splits when the claim is about *new drugs* |
| **L3.3 Sampling bias in the test set** | The test distribution differs from the target population | Benchmarks favour well-studied drugs and diseases; random "negatives" are mostly trivially easy pairs; unknown pairs that are actually true indications are scored as negatives |

Two remarks. First, L3 errors are *not* about information flowing from test to training; they are about the test set answering a different question from the one claimed. A model can be evaluated with perfect separation and still overclaim. Second, the remedy for L3 is usually **a different split** (temporal split, group split, cold-start split) or **a narrower claim** ("on these benchmarks, in the warm-start setting").

Kapoor & Narayanan propose **model info sheets**: for every model, the authors argue in writing (1) why train and test are cleanly separated in every step, (2) why every feature is legitimate, and (3) why the test set represents the claimed population. Section 5.7 writes one for this project.

### 2.3 Leakage specific to graph link prediction

Link prediction has a feature that ordinary classification lacks: **the labels and the input are the same object.** The known drug–disease links are the targets we predict *and* the edges of the graph the model reads. Most link-prediction leakage is a version of "the graph used as input still contains (or reflects) the edges we are testing".

**Transductive setting.** Our models are *transductive*: every drug and every disease is a node during training, including the ones whose links are tested. That is legitimate — at prediction time we do know all the drugs and diseases we want to score, and their similarities (computed from structures and ontologies, not from links). What must be hidden in each fold is the **test links** and anything computed from them. "Transductive" describes which *nodes* are present; it does not license using test *edges*.

**Channel 1 — test edges in the message-passing (or propagation) graph.** A graph neural network computes a drug's embedding by aggregating messages from its neighbours. If the test edge $(i, j)$ is in the graph, drug $i$'s embedding contains disease $j$'s features, and a dot-product decoder scores $(i,j)$ highly *because the edge is there*. The same is true for any propagation: a 3-step random walk from drug $i$ through the graph returns to disease $j$ via the path $i \to j \to i \to j$ that uses the test edge itself. Section 4.3 shows AUC 1.000 on noise.

*Defence:* build every graph, every propagation matrix and every normalization (degrees, $D^{-1/2}AD^{-1/2}$) from `A_train` only, in every fold.

**Channel 2 — features computed from labels.** Node features that are functions of the association matrix: the association row itself, the degree (number of known indications), node2vec/DeepWalk embeddings of the drug–disease graph, PageRank, GIP kernels. All are legitimate **if computed from the training links of the fold**, and leaky if computed once from the full matrix.

**Channel 3 — similarity computed from the association matrix.** A special case of channel 2 that is extremely common in drug-repositioning papers: "two drugs are similar if they treat similar diseases" (cosine/Jaccard of indication profiles, the GIP kernel). If computed from the full $A$, the similarity of drug $i$ to drug $k$ is raised by their shared *test* indications, and then "similar drugs treat similar diseases" propagates the test link back to drug $i$. Section 4.3: AUC 0.993 on noise. The project avoids the whole issue by computing **no view from the association matrix** (chemical structures, MimMiner phenotypes, MONDO semantics and CTD genes only).

**Channel 4 — negatives sampled from test cells.** Models trained with sampled negatives need a pool of "training negatives". There are three tempting pools, and they behave differently (Section 4.4):

| Pool | Contains | Effect |
|---|---|---|
| (a) zeros of `A_train` | training zeros + test negatives + **hidden test positives** | No leakage: the model cannot tell hidden positives from test negatives, so it pushes both down equally (a few true positives are trained as negatives — a pessimistic bias, not an optimistic one) |
| (b) zeros of the **full** `A` | training zeros + test negatives (test positives excluded) | **Leakage:** choosing the pool with the full matrix reveals *which* test cells are negatives; a flexible model memorizes them as low, so test negatives score lower than test positives — AUC 0.605 on noise |
| (c) zeros of `A` minus the test cells | training zeros only | Clean: the project's `neg_mask` |

Pool (b) is the subtle one: it uses the label of every test cell to *exclude* the test positives. Pool (c) removes both test positives and test negatives, so the test cells play no role at all in training.

**Channel 5 — early stopping, checkpoint selection or threshold choice on the test fold.** Choosing the epoch with the best *test* AUPR is tuning on test data (Section 2.4).

**Not leakage, but related: the "edge-detector" shortcut.** If a model is trained to score pairs whose edges it can *see* in its input graph, it can learn the trivial rule "score high if the edge is present". At test time the edge is never present (it was hidden), so the rule is useless and the model has learned nothing transferable. This is a *train/test mismatch* rather than leakage — it makes results worse, not better — but it is created by the same fact (labels = input edges). **Hidden-link supervision** fixes it: in each epoch, hide a random subset of training links from the graph and the features and compute the loss only on those hidden links, so that training pairs look like test pairs. Section 4.5 shows a gradient-boosting model collapse to AUC 0.500 with visible-link supervision and recover 0.735 with hidden-link supervision.

**Cold start (LODO).** When all links of disease $j$ are hidden, *every* link-derived quantity must also lose them: the graph, the features, the degrees, the propagation inputs and the negative pool. A forgotten degree feature that still counts disease $j$'s links tells the model "this disease has many drugs", which is information about the hidden labels.

**Non-independent pairs.** In pair-input problems (drug–target, drug–disease, protein–protein), Park & Marcotte (2012) and Pahikkala et al. (2015) showed that performance depends dramatically on whether test pairs share a drug or a disease with training pairs. A random pair-wise split (warm start; "S1" in Pahikkala's notation) always lets the model use other links of the same drug and disease; a drug-wise split ("S2", new drugs) or disease-wise split ("S3", our LODO) does not. Guney (2017) showed that similarity-based repositioning performance "drops sharply" when the drugs in training and test folds are disjoint. Neither split is wrong — they answer different questions — but **the claim must match the split** (L3). And even a drug-wise split can leak if near-duplicate drugs land on opposite sides (Exercise 12: AUC 0.875 versus 0.502).

### 2.4 Model selection and tuning on test data

**The mechanism.** Suppose you try $m$ configurations (hyper-parameters, architectures, seeds) and each one's CV score is its true performance plus noise: $\hat{\theta}_c = \theta_c + \varepsilon_c$, with $\varepsilon_c \sim \mathcal{N}(0, \sigma^2)$. If all configurations are truly equal ($\theta_c = \theta$), the *best* observed score is

$$\mathbb{E}\Big[\max_{c} \hat\theta_c\Big] = \theta + \sigma\, \mathbb{E}\Big[\max_{c\le m} Z_c\Big], \qquad Z_c \sim \mathcal{N}(0,1)\ \text{i.i.d.}$$

The expected maximum of $m$ standard normals is $0.564$ for $m=2$, $1.163$ for $m=5$, $1.539$ for $m=10$, $1.867$ for $m=20$ and $2.508$ for $m=100$ (computed in Section 4.9; the familiar approximation $\sqrt{2\ln m}$ overestimates at these small $m$). So reporting the best of 20 configurations whose evaluation noise has $\sigma = 0.02$ inflates the result by about $1.87 \times 0.02 \approx 0.037$ — the size of a typical "improvement over the state of the art". This is the **winner's curse**, or optimistic **selection bias**. Correlated configurations (e.g. neighbouring $k$ in kNN) behave like a smaller effective $m$, but the bias never disappears.

**The rule.** The data used to *choose* anything — hyper-parameters, number of epochs, which variant to keep, which seed to report, even which architecture idea to pursue — must be different from the data used to *report* performance. This gives the classic three-way split:

```
 all labelled data
 ├── training data     → fit parameters (weights)
 ├── validation data   → choose hyper-parameters, variants, epochs
 └── test data         → touched ONCE, to report the final number
```

**Nested cross-validation** applies this inside CV, so that every reported test fold is untouched by selection:

```
for each outer fold o = 1..K_out:
    outer_train, outer_test = split(data, o)
    for each configuration c in grid:
        for each inner fold i = 1..K_in:          # split outer_train only
            train on inner_train, score on inner_val
        inner_score[c] = mean over i
    c* = argmax_c inner_score[c]
    refit with c* on all of outer_train
    report score on outer_test                     # never used to choose c*
final estimate = mean over outer folds
```

Cost: $K_\text{out} \times (K_\text{in} \times |\text{grid}| + 1)$ fits, e.g. $5 \times (5 \times 20 + 1) = 505$ fits for a 20-point grid. Cheaper variants: one validation split inside each outer training fold instead of an inner CV ($5 \times (20+1) = 105$ fits).

Varma & Simon (2006) showed on null microarray data (no real signal) that tuning a classifier with CV and reporting the same CV error gave strongly optimistic error estimates, while nested CV was nearly unbiased. Cawley & Talbot (2010) showed that over-fitting the *model-selection criterion* is a real phenomenon whose effect on performance estimates can be as large as the differences between learning algorithms being compared. Section 4.6 reproduces both points in a minute of CPU time.

**Researcher degrees of freedom.** Nested CV protects against *automated* selection. It does not protect against the human loop: running many experiments, looking at the test numbers, and changing the method until it wins. Every architectural decision made after looking at test results is selection on the test set. The defences are procedural: make design decisions on validation data; touch the test folds only for final runs; keep a log of everything you tried; and, ideally, confirm the final model on data nobody looked at during development (a held-out benchmark, a temporal split, prospective case studies).

### 2.5 Fair baselines

A comparison "our method vs baseline X" is fair only if the baseline had the same chances. Check four parities:

1. **Same splits.** Every method is evaluated on identical folds, so the per-fold scores can be *paired*. In the project, `kfold_splits(A, k, seed)` depends only on the seed, so all methods see the same 25 splits (5 folds × 5 repeats).
2. **Same information.** If the new method uses six similarity views and the baselines use two, a win may come from the *data*, not the method. Either give the baselines the same inputs where their design allows, or add an ablation of your method restricted to the baselines' inputs (the project's "benchmark similarities only").
3. **Same tuning budget.** If your method's hyper-parameters were tuned over 20 configurations × 3 seeds and the baselines ran with defaults, part of your margin is the tuning. Give each method a comparable budget on the same validation split and report it. Li et al. (2023) found, for GNN link prediction, that many reported gains vanished when baselines were tuned properly.
4. **Same evaluation details.** Same negative-sampling policy, same metric implementation, same handling of ties and missing values, same number of repeats.

Also: **re-implementations versus reported numbers.** A re-implemented baseline can be weaker than the original because of details the paper did not report. Say "our re-implementation", keep published numbers in a separate table, and be most cautious about baselines that perform far below their published values (a sign of a bug or of poor tuning). Finally, include at least one **simple, strong baseline** (a nearest-neighbour propagation, a random walk): if a complex model barely beats it, that is the headline.

### 2.6 Ablation design

An **ablation study** removes (or replaces) one component at a time to measure its contribution. Principles:

- **One change at a time** (one-factor-at-a-time, OFAT): each variant differs from the full model in exactly one respect, so a difference can be attributed to that component. Changing two things at once confounds them.
- **Common random numbers:** run every variant on the *same* splits with the *same* seeds, so that differences are not split or initialization noise. The project does this: in `run_kfold`, both the folds (`kfold_splits(A, k, seed + r)`) and the model seed (`seed * 1000 + r * 100 + f`) depend only on the repeat and fold. (A nice check: the full model's five ablation folds reproduce, to the fourth decimal, the first repeat of its main 5-fold CV run.)
- **Controls.** A *negative control* is a variant that should **not** help, e.g. replacing a similarity view by a randomly permuted copy (same distribution, no information); if the permuted view "helps", the improvement comes from extra capacity or noise, not from the view's content. A *positive control* is a variant that **must** hurt, e.g. removing all similarity views; if it doesn't, the evaluation is insensitive. A *sanity check* — shuffling the labels — must give random performance.
- **Interactions.** OFAT misses interactions (component A only helps when B is present). If a few components are central, a small factorial design ($2^k$ variants) reveals interactions.
- **Additive versus subtractive.** "Full minus X" measures what X contributes *given everything else*; "baseline plus X" measures what X contributes *alone*. They can disagree; report the one that matches your claim.
- **Uncertainty and multiplicity.** Ten variants on five folds yield ten noisy differences; some will look large by chance. Report paired differences with their spread, and correct for multiple comparisons (Section 2.8.5).
- **Ablations explain; they do not select.** An ablation run on test folds tells the reader what each component contributes. It must not be used to change the model — that would make the test folds a validation set (the hidden-link-supervision example of Section 1).

Lipton & Steinhardt (2018) list "failure to identify the sources of empirical gains" — for example, attributing to an architectural novelty a gain that really comes from tuning or extra data — as one of four troubling trends in ML scholarship. Ablations and same-information baselines are the cure.

### 2.7 Variance, seeds and repeated runs

A deep model's test score varies from run to run because of:

| Source | Controlled by |
|---|---|
| which cells fall in which fold | the split seed |
| weight initialization | the model seed |
| negative sampling, dropout, edge hiding each epoch | the model seed |
| order of floating-point operations on the GPU (non-deterministic kernels) | partly not controllable |
| data version (a newer CTD release) | the manifest (Unit D4) |

Setting seeds makes a run *reproducible* (you get the same number again); it does not make the number *representative*. To estimate the variability, repeat over several split seeds and several model seeds and report the mean and the standard deviation, saying what was varied.

Three distinctions:
- **Standard deviation** of fold scores ($s$): how much one run varies. This is what `summarise()` reports (with NumPy's default `ddof=0`).
- **Standard error** of the mean ($s/\sqrt{J}$ naively): how precisely the *mean* is known. With overlapping training sets this naive formula is too small (Section 2.8.2).
- **Confidence interval:** mean ± $t_{0.975,J-1}\times$SE, again with the corrected SE.

And one rule: **never report the best seed.** The best of 10 seeds of a model whose seed-to-seed standard deviation is 0.02 is, on average, 0.031 better than its typical run (Section 4.9) — a free, fake improvement.

### 2.8 Comparing two methods statistically

#### 2.8.1 Pairing

Let $a_r$ and $b_r$ be the scores of methods A and B on split $r = 1,\dots,J$ (same splits for both). The paired differences $d_r = a_r - b_r$ remove the variation that both methods share (some folds are simply easier). If $a$ and $b$ have standard deviations $s_a, s_b$ and correlation $\rho_{ab}$ across folds,

$$\mathrm{Var}(d) = s_a^2 + s_b^2 - 2\rho_{ab}\, s_a s_b,$$

so pairing helps a lot when the methods' fold scores are strongly correlated and little when they are not. On Fdataset (Section 4.7), MV-HGAT's and SCMFDD's fold AUCs correlate at +0.59, but their fold AUPRs only at +0.09.

#### 2.8.2 The paired $t$-test and why folds are not independent

The paired $t$ statistic is

$$t = \frac{\bar d}{s_d/\sqrt{J}}, \qquad \bar d = \frac1J\sum_r d_r,\quad s_d^2 = \frac{1}{J-1}\sum_r (d_r-\bar d)^2,$$

compared with a $t$ distribution with $J-1$ degrees of freedom. It assumes the $d_r$ are independent. They are not: in 5-fold CV any two training sets share 75% of their data, and repeated CV reuses the same data again. The $d_r$ are positively correlated, $s_d^2/J$ underestimates the true variance of $\bar d$, and the test rejects too often. Dietterich (1998) showed this inflated type-I error for resampled $t$-tests; Bengio & Grandvalet (2004) proved that no unbiased estimator of the variance of the k-fold CV estimate exists.

#### 2.8.3 The Nadeau–Bengio corrected resampled $t$-test

Nadeau & Bengio (2003) model the correlation and inflate the variance:

$$t_\text{corr} = \frac{\bar d}{\sqrt{\left(\dfrac1J + \dfrac{n_\text{test}}{n_\text{train}}\right)s_d^2}}, \qquad \text{df} = J-1.$$

For k-fold CV, $n_\text{test}/n_\text{train} = 1/(k-1)$, i.e. $1/4$ for 5-fold. Bouckaert & Frank (2004) studied the *replicability* of tests (does the conclusion survive a different random partition?) and recommended repeated CV (e.g. 10 × 10) with this corrected variance. Notice what the correction implies: as $J \to \infty$ the standard error tends to $s_d\sqrt{n_\text{test}/n_\text{train}} = s_d/2$, **not** to zero. Repeating CV more often cannot make a small difference significant on one dataset (Exercise 13 computes the power ceiling).

**Worked example (by hand).** The ablation folds of Section 1 give differences (w/o hidden-link supervision − full) of
$d = (0.0415,\ 0.0933,\ 0.0217,\ 0.0292,\ 0.0431)$.
- $\bar d = 0.2288/5 = 0.04576$.
- Deviations: $-0.00426, 0.04754, -0.02406, -0.01656, -0.00266$; squares sum to $0.0031384$; $s_d^2 = 0.0031384/4 = 0.0007846$; $s_d = 0.02801$.
- Naive: $\text{SE} = 0.02801/\sqrt5 = 0.01253$, $t = 3.65$, df $=4$, two-sided $p \approx 0.022$.
- Corrected: $\text{SE} = \sqrt{(1/5 + 1/4)\times 0.0007846} = \sqrt{0.0003531} = 0.01879$, $t = 2.44$, $p \approx 0.072$.

So the naive test says "significant at 5%", the corrected test says "not significant". Section 4.7 confirms these numbers in code.

#### 2.8.4 Other tests

- **Wilcoxon signed-rank test:** ranks the $|d_r|$ and compares the rank sums of positive and negative differences; no normality assumption. With only 5 pairs the smallest attainable two-sided $p$ is $2/2^5 = 0.0625$ — it *cannot* reach 0.05 even if one method wins every fold. It also ignores the fold dependence.
- **Sign test / win count:** "A beats B on 25 of 25 folds" is a vivid summary but, like the others, treats folds as independent.
- **5×2cv paired $t$-test** (Dietterich 1998): five repetitions of 2-fold CV with a specially constructed statistic having 5 df; it has acceptable type-I error, at the cost of training on only half the data.
- **McNemar's test:** for classifiers evaluated once on a single test set, compares the counts of examples that one classifier gets right and the other wrong.
- **Across several datasets** (Demšar 2006): use the Wilcoxon signed-rank test over datasets to compare two methods, and the Friedman test followed by the Nemenyi post-hoc test (displayed as a critical-difference diagram) to compare many methods. Datasets, unlike folds, are independent samples of "problems".
- **Bayesian alternatives** (Benavoli et al. 2017): the Bayesian correlated $t$-test gives the posterior probability that A is better, worse or practically equivalent (within a *region of practical equivalence*, e.g. ±0.01 AUPR) — often a more useful answer than a $p$-value.

#### 2.8.5 Effect sizes and multiple comparisons

A $p$-value answers "could this difference be noise?", not "is it large?". Always report an **effect size**:
- the raw mean difference with a (corrected) confidence interval — the most interpretable;
- the standardized paired effect $d_z = \bar d / s_d$ (Lakens 2013);
- the win rate (probability of superiority).

**Multiple comparisons.** Testing 10 ablation variants at $\alpha = 0.05$ gives a $1 - 0.95^{10} \approx 40\%$ chance of at least one false positive if nothing matters. Control the **family-wise error rate** with **Holm's step-down procedure**: sort the $m$ $p$-values $p_{(1)} \le \dots \le p_{(m)}$; the adjusted values are

$$\tilde p_{(i)} = \max_{j \le i}\ \min\big(1,\ (m - j + 1)\, p_{(j)}\big),$$

and you reject where $\tilde p_{(i)} \le \alpha$. Holm is uniformly more powerful than Bonferroni ($m\,p_{(i)}$) with the same guarantee.

**Statistical versus practical significance.** With enough folds a difference of 0.001 AUC can be "significant" and worthless; with five folds a difference of 0.05 AUPR can be "non-significant" and real. Decide in advance what difference would matter (a region of practical equivalence), and read the evidence against it.

### 2.9 Reporting standards

What a reader needs, at minimum, to judge an ML result (synthesized from Kapoor & Narayanan's model info sheets, the NeurIPS paper checklist and Pineau et al. 2021):

1. **Data:** sources, versions/download dates, licences, preprocessing, and the exact entity-mapping coverage.
2. **Splits:** protocol (warm k-fold, LODO, temporal…), $k$, number of repeats, seeds, how negatives are defined, what is hidden in each fold.
3. **Leakage argument:** separation of train/test in every step; legitimacy of every feature; match between test set and claim.
4. **Tuning:** search space, budget, the data used for tuning (validation, never test), the same for baselines.
5. **Results:** mean ± std (say over what, and the `ddof`), the random baseline of each metric, paired comparisons with effect sizes and corrected tests, all variants tried (not only the best).
6. **Baselines:** re-implemented or published, inputs given, tuning budget.
7. **Ablations:** one change at a time, same splits, controls.
8. **Limitations:** stated plainly — including the ones you would rather not mention.
9. **Code and data availability**, compute used (hardware, time), and enough detail to re-run.

The NeurIPS checklist asks, among other things, whether error bars are reported, what variability they capture and how they were computed, whether the training details (splits, hyper-parameters and how they were chosen) are given, whether the limitations are discussed, and whether the licences of existing assets are respected — every item above has a counterpart there.

---

## 3. Worked examples by hand

### Worked example 1 — auditing a pipeline line by line

A (hypothetical) drug-repositioning paper describes this pipeline. Mark each step *clean* or *leaky*, and name the taxonomy type.

| # | Step | Verdict |
|---|---|---|
| 1 | Drug similarity = Tanimoto of ECFP4 fingerprints | Clean: computed from structures only |
| 2 | Drug similarity 2 = GIP kernel of the drug rows of the association matrix, computed once | **Leaky** (L1.2, channel 3): uses test links |
| 3 | Diseases with fewer than 2 known drugs are removed before CV | **Leaky in a mild way** (L1.2/L3.3): the filter uses labels, including test labels; also changes the population. Filter on information available at prediction time, or report it as part of the population definition |
| 4 | Node features = degree of each node in the drug–disease graph | Leaky if computed from the full $A$ (channel 2); clean if recomputed from `A_train` per fold |
| 5 | Negatives = random cells where $A = 0$, re-sampled each epoch | **Leaky** if "$A$" is the full matrix (pool (b), channel 4) |
| 6 | Number of epochs chosen where the test AUPR peaks | **Leaky** (channel 5, Section 2.4) |
| 7 | Five seeds; the table reports the best | **Leaky** in effect (selection on test; Section 2.7) |
| 8 | Baselines run with their papers' default hyper-parameters; ours tuned | Not leakage, but an **unfair comparison** (Section 2.5) |

### Worked example 2 — how much can selection inflate a result?

The project's sensitivity log (`results/logs/F_sens.log`, validation split, 2 seeds) shows AUPR standard deviations across seeds between 0.002 and 0.043 for the same configuration, typically around 0.01–0.02. The grid has 4 + 3 + 4 + 4 + 4 = 19 settings (15 distinct configurations, because each hyper-parameter's default appears once in each row). Taking $\sigma \approx 0.015$ and $m = 15$ equally good configurations, $\mathbb{E}[\max Z] \approx 1.74$ (between the values for $m=10$ and $m=20$), so the best *validation* score would be inflated by roughly $1.74 \times 0.015 \approx 0.026$ even if the hyper-parameters did not matter at all. That is why (i) the validation score of the chosen configuration (0.521) must not be reported as the model's performance, and (ii) the test evaluation must be separate.

### Worked example 3 — Holm correction for the ablation table

Section 4.8 computes paired $t$-test $p$-values for the 10 ablation variants against the full model. Sorted: 0.0217 (w/o hidden-link supervision), 0.0651 (w/o ECFP), 0.0666 (w/o propagation head), 0.0800, 0.0946, 0.1701, 0.1784, 0.3114, 0.4158, 0.4643.

| rank $i$ | $p_{(i)}$ | $(m-i+1)\,p_{(i)}$ | Holm $\tilde p_{(i)}$ (running max) |
|---|---|---|---|
| 1 | 0.0217 | 10 × 0.0217 = 0.217 | 0.217 |
| 2 | 0.0651 | 9 × 0.0651 = 0.586 | 0.586 |
| 3 | 0.0666 | 8 × 0.0666 = 0.533 | 0.586 |
| 4 | 0.0800 | 7 × 0.0800 = 0.560 | 0.586 |
| 5 | 0.0946 | 6 × 0.0946 = 0.568 | 0.586 |
| 6 | 0.1701 | 5 × 0.1701 = 0.851 | 0.851 |
| … | … | … | … |

Nothing survives. With five folds, an ablation can show the *direction and size* of effects (removing the propagation head costs 0.067 AUPR; removing ECFP 0.038), but it cannot establish any of them statistically. The honest write-up gives the paired differences with their spread, says that five folds give low power, and avoids the word "significant".

### Worked example 4 — how much does the validation split overlap the test folds?

`05_sensitivity.py` holds out a random 20% of the positives (and of the negatives) as a validation set, *once*, before CV. Each CV test fold is another random 20% of the positives. For any positive cell, $P(\text{in validation}) = 0.2$ independently of its fold, so on average **20% of every test fold's positives were validation positives** — cells on which the hyper-parameters were chosen to score well. The hyper-parameters were never *trained* on those cells, so this is selection, not training contamination; with 15 configurations and the small effects seen in the sensitivity log, the resulting bias is probably small (well under Worked example 2's 0.026, because only a fifth of each test fold is involved). But it is not zero, and it is unmeasured. A nested scheme (tuning inside each outer training fold) or a final untouched hold-out set would remove it.

### Worked example 5 — the power ceiling of cross-validation

Suppose a new component truly improves AUPR by $\delta = 0.02$ and the per-fold differences have $s_d = 0.03$. With the corrected test, the expected statistic is
$$\frac{\delta}{s_d\sqrt{1/J + 1/4}} \;\xrightarrow{J\to\infty}\; \frac{0.02}{0.03 \times 0.5} = 1.33,$$
below the critical value 1.96 even with infinitely many repeats. On a single dataset of this size, CV cannot "prove" a 0.02 improvement; you need more datasets (Demšar's setting), external validation, or a larger effect. Exercise 13 turns this into a power table.

---

## 4. Code: leakage you can watch happen

Each block is a self-contained simulation. Most use **random labels**, so that the honest answer is known to be AUC 0.5: anything above it is leakage by construction. Run them, then change the seeds and sizes and watch which numbers move.

### 4.1 Feature selection before cross-validation (L1.3)

Eighty samples, 5,000 features of pure noise, labels unrelated to the features. The wrong way selects the 20 features most correlated with the labels *using all 80 samples*, then cross-validates a classifier on them. The right way puts the selection inside a `Pipeline`, so it is refitted on each training fold.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
n, p, k = 80, 5000, 20
X = rng.standard_normal((n, p))          # pure noise features
y = np.repeat([0, 1], n // 2)             # labels unrelated to X
cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = LogisticRegression(max_iter=1000)

# WRONG: choose the 20 "best" features using ALL 80 samples, then cross-validate
X_sel = SelectKBest(f_classif, k=k).fit_transform(X, y)
wrong = cross_val_score(clf, X_sel, y, cv=cv, scoring="roc_auc")

# RIGHT: feature selection is a step INSIDE the model, refit on each training fold
pipe = make_pipeline(SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000))
right = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

print(f"selection on all data : AUC {wrong.mean():.3f} +/- {wrong.std():.3f}")
print(f"selection inside CV   : AUC {right.mean():.3f} +/- {right.std():.3f}")
```

Output:

```text
selection on all data : AUC 0.947 +/- 0.035
selection inside CV   : AUC 0.487 +/- 0.063
```

With 5,000 noise features, about 20 will correlate with the labels by chance — in the full data, including the test folds. Selecting them on all data hands the classifier features that "predict" the test labels. This is the error that Hastie, Tibshirani & Friedman call "the wrong way to do cross-validation" (ESL, Section 7.10.2). In drug repositioning, the equivalent is choosing genes, pathways or views by their association with the *full* indication matrix.

### 4.2 Duplicates across the split (L1.4 / L3.2)

Each of 150 "parent compounds" appears three times (think free base and two salts, with almost identical features); labels are noise. A random row split puts copies of the same parent on both sides; a group split keeps all copies together.

```python
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(3)
n_parent = 150
X_parent = rng.standard_normal((n_parent, 10))
y_parent = rng.integers(0, 2, n_parent)                 # labels are pure noise
# every "parent compound" appears 3 times (e.g. free base + two salt forms):
X = np.repeat(X_parent, 3, axis=0) + 0.01 * rng.standard_normal((3 * n_parent, 10))
y = np.repeat(y_parent, 3)
groups = np.repeat(np.arange(n_parent), 3)              # which parent each row belongs to

def cv_auc(splitter):
    aucs = []
    for tr, te in splitter.split(X, y, groups=groups if isinstance(splitter, GroupKFold) else None):
        clf = KNeighborsClassifier(n_neighbors=3).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return np.mean(aucs)

print(f"random row split (duplicates leak) : AUC {cv_auc(KFold(5, shuffle=True, random_state=0)):.3f}")
print(f"group split (parents kept together): AUC {cv_auc(GroupKFold(5)):.3f}")
```

Output:

```text
random row split (duplicates leak) : AUC 0.903
group split (parents kept together): AUC 0.472
```

A 3-nearest-neighbour classifier finds the test compound's twin in the training set and copies its label. The fix is to split by *group* (`GroupKFold`), with the group defined by what should count as "the same unit" for your claim — here the parent compound (the InChIKey of the parent, Unit D4). Over-sampling the minority class *before* splitting (Kapoor & Narayanan's L1.2 example) creates exactly this situation artificially.

### 4.3 Link prediction: similarity from labels and test edges in the graph

This is the simulation quoted in Section 1, extended with a matrix that does contain real structure. All three scorers are evaluated with the project's protocol (positives and negatives split into 5 folds; test positives hidden from `A_train`).

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def cosine(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    S = (M / n) @ (M / n).T
    np.fill_diagonal(S, 0)          # never let a drug "vote" for itself
    return S

def kfold_cells(A, k, rng):
    pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
    rng.shuffle(pos); rng.shuffle(neg)
    return list(zip(np.array_split(pos, k), np.array_split(neg, k)))

def evaluate(A, scorer, k=5, seed=0):
    rng = np.random.default_rng(seed)
    aucs, auprs = [], []
    for tp, tn in kfold_cells(A, k, rng):
        A_tr = A.copy().ravel(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
        S = scorer(A_tr, A)                       # A (full) is passed so leaky scorers can misuse it
        idx = np.concatenate([tp, tn]); y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
        aucs.append(roc_auc_score(y, S.ravel()[idx])); auprs.append(average_precision_score(y, S.ravel()[idx]))
    return np.mean(aucs), np.mean(auprs)

# --- scorers: each returns a drugs x diseases score matrix -------------------
def clean_cf(A_tr, A_full):      # similarity AND propagation from training links only
    return cosine(A_tr) @ A_tr
def sim_from_labels(A_tr, A_full):  # similarity computed from the FULL association matrix
    return cosine(A_full) @ A_tr
def test_edges_in_graph(A_tr, A_full):  # 3-hop paths on a graph that still has test edges
    return A_full @ A_full.T @ A_full

rng = np.random.default_rng(42)
A_noise = (rng.random((300, 150)) < 0.03).astype(float)   # NO real signal at all
print(f"random matrix: {int(A_noise.sum())} links, density {A_noise.mean():.3f}")
for name, f in [("clean (train links only)", clean_cf),
                ("similarity from full A", sim_from_labels),
                ("test edges in graph", test_edges_in_graph)]:
    auc, ap = evaluate(A_noise, f)
    print(f"  {name:26s} AUC {auc:.3f}  AUPR {ap:.3f}")

# --- the same three scorers on a matrix WITH real structure -----------------
gr, gd = rng.integers(0, 10, 300), rng.integers(0, 8, 150)     # hidden drug / disease groups
compatible = rng.random((10, 8)) < 0.2                          # which group pairs "treat"
P = np.where(compatible[gr][:, gd], 0.25, 0.01)
A_sig = (rng.random(P.shape) < P).astype(float)
print(f"structured matrix: {int(A_sig.sum())} links, density {A_sig.mean():.3f}")
for name, f in [("clean (train links only)", clean_cf),
                ("similarity from full A", sim_from_labels),
                ("test edges in graph", test_edges_in_graph)]:
    auc, ap = evaluate(A_sig, f)
    print(f"  {name:26s} AUC {auc:.3f}  AUPR {ap:.3f}")
```

Output:

```text
random matrix: 1357 links, density 0.030
  clean (train links only)   AUC 0.494  AUPR 0.030
  similarity from full A     AUC 0.993  AUPR 0.933
  test edges in graph        AUC 1.000  AUPR 0.996
structured matrix: 3178 links, density 0.071
  clean (train links only)   AUC 0.781  AUPR 0.195
  similarity from full A     AUC 0.915  AUPR 0.616
  test edges in graph        AUC 0.939  AUPR 0.620
```

- `clean`: similarity (cosine of indication profiles) and propagation both from `A_train`. On noise it gives 0.494; on the structured matrix 0.781 — the honest value.
- `similarity from full A`: only the *similarity* uses the full matrix; propagation still uses training links. On noise it reaches 0.993: drugs that share the hidden test link look similar, and each "votes" for the other's hidden link.
- `test edges in graph`: three-step propagation on the full graph. The path drug → disease → drug → disease includes the test edge itself.

On the structured matrix, the leaky versions add 0.13–0.16 AUC and roughly triple the AUPR *on top of real signal*. In a paper this would look like "our method beats the baseline by a wide margin" — and the baseline would be the honest one.

### 4.4 Where do the training negatives come from? (channel 4)

A high-rank logistic matrix factorization (able to memorize individual cells) is trained with negatives re-sampled each epoch from three different pools. Labels are random.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def train_mf(A_tr, neg_pool, rank=64, epochs=400, lr=2.0, neg_ratio=5, seed=0):
    """Logistic matrix factorisation (score = U V^T) trained with BCE on all training
    positives plus negatives re-sampled from `neg_pool` every epoch. High rank = it can memorise."""
    r = np.random.default_rng(seed)
    n, m = A_tr.shape
    U, V = 0.1 * r.standard_normal((n, rank)), 0.1 * r.standard_normal((m, rank))
    pos = np.flatnonzero(A_tr.ravel() > 0)
    for _ in range(epochs):
        neg = r.choice(neg_pool, len(pos) * neg_ratio)
        cells = np.r_[pos, neg]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        i, j = np.divmod(cells, m)
        p = 1 / (1 + np.exp(-(U[i] * V[j]).sum(1)))
        g = (p - y)[:, None]                                  # dLoss/dlogit per sampled cell
        gU, gV = np.zeros_like(U), np.zeros_like(V)
        np.add.at(gU, i, g * V[j])
        np.add.at(gV, j, g * U[i])
        U -= lr / n * gU
        V -= lr / m * gV
    return U @ V.T

rng = np.random.default_rng(7)
A = (rng.random((150, 80)) < 0.05).astype(float)              # random links: honest AUC = 0.5
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
rng.shuffle(pos); rng.shuffle(neg)
pools = {
    "(a) zeros of A_train (test pos + neg)":   lambda A_tr, tn: np.flatnonzero(A_tr.ravel() == 0),
    "(b) zeros of FULL A (test negatives)":     lambda A_tr, tn: np.flatnonzero(A.ravel() == 0),
    "(c) zeros of A minus test cells (project)": lambda A_tr, tn: np.setdiff1d(
        np.flatnonzero(A.ravel() == 0), tn),
}
res = {k: [] for k in pools}
for tp, tn in zip(np.array_split(pos, 5), np.array_split(neg, 5)):
    A_tr = A.ravel().copy(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
    idx, y = np.r_[tp, tn], np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    for key, pool in pools.items():
        S = train_mf(A_tr, pool(A_tr, tn))
        res[key].append(roc_auc_score(y, S.ravel()[idx]))
for k, v in res.items():
    print(f"{k:42s} AUC {np.mean(v):.3f} +/- {np.std(v):.3f}")
```

Output (about a minute on a laptop CPU):

```text
(a) zeros of A_train (test pos + neg)      AUC 0.473 +/- 0.011
(b) zeros of FULL A (test negatives)       AUC 0.605 +/- 0.010
(c) zeros of A minus test cells (project)  AUC 0.476 +/- 0.013
```

Pool (b) leaks: it was built from the full matrix, so it contains the test negatives but *not* the test positives. The model memorizes the test negatives as "0" and leaves the test positives untouched, so it ranks test positives above test negatives without any real signal — AUC 0.605 on noise. Pools (a) and (c) do not leak. Pool (a) treats test positives and test negatives identically (both are zeros of `A_train`), and pool (c) — the project's `neg_mask` — excludes both. (Both sit slightly below 0.5: 0.47–0.49 here and in a rerun with another random matrix, where pool (b) again gave 0.610. Whatever the cause of that small dip, its direction is pessimistic, so it cannot make a method look better than it is.)

### 4.5 The edge-detector shortcut and hidden-link supervision

This simulation isolates the reason the project uses **hidden-link supervision**. Each drug–disease pair gets two features: `f1` = "is this edge visible in the input graph?" and `f2` = a propagation score from the visible graph (similar drugs' links). A gradient-boosting classifier is trained either on visible links or, as in `MVHGATMethod.fit_predict`, on links hidden from the graph in each "epoch".

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(1)
gr, gd = rng.integers(0, 10, 300), rng.integers(0, 8, 150)
P = np.where((rng.random((10, 8)) < 0.2)[gr][:, gd], 0.25, 0.01)
A = (rng.random(P.shape) < P).astype(float)            # structured drug x disease matrix

def cosine(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    S = (M / n) @ (M / n).T; np.fill_diagonal(S, 0); return S

def pair_features(A_vis):
    """f1 = is the edge itself visible?  f2 = neighbour-propagation score."""
    return np.c_[A_vis.ravel(), (cosine(A_vis) @ A_vis).ravel()]

# one CV fold: hide 20% of positives and 20% of negatives as the test set
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
test_pos = rng.choice(pos, len(pos) // 5, replace=False)
test_neg = rng.choice(neg, len(neg) // 5, replace=False)
train_pos, train_neg = np.setdiff1d(pos, test_pos), np.setdiff1d(neg, test_neg)
A_tr = A.ravel().copy(); A_tr[test_pos] = 0; A_tr = A_tr.reshape(A.shape)
idx, y_test = np.r_[test_pos, test_neg], np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]
X_test = pair_features(A_tr)[idx]                        # at test time the test edge is invisible

def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(X, y)

# (A) supervise on VISIBLE links: every positive has f1 = 1, every negative f1 = 0
negs = rng.choice(train_neg, 2 * len(train_pos), replace=False)
F = pair_features(A_tr)
m_vis = fit(F[np.r_[train_pos, negs]], np.r_[np.ones(len(train_pos)), np.zeros(len(negs))])

# (B) hidden-link supervision: hide 20% of training links, supervise ONLY on those
Xs, ys = [], []
for _ in range(10):                                      # 10 "epochs" of random hiding
    hide = rng.random(len(train_pos)) < 0.2
    A_vis = A_tr.ravel().copy(); A_vis[train_pos[hide]] = 0; A_vis = A_vis.reshape(A.shape)
    F = pair_features(A_vis)
    sup = train_pos[hide]; negs = rng.choice(train_neg, 2 * len(sup), replace=False)
    Xs.append(F[np.r_[sup, negs]]); ys.append(np.r_[np.ones(len(sup)), np.zeros(len(negs))])
m_hid = fit(np.vstack(Xs), np.concatenate(ys))

for name, m in [("supervise on visible links", m_vis), ("hidden-link supervision", m_hid)]:
    s = m.predict_proba(X_test)[:, 1]
    print(f"{name:27s} test AUC {roc_auc_score(y_test, s):.3f}   "
          f"distinct test scores: {len(np.unique(s))}")
print(f"reference: f2 alone (no learning)  test AUC {roc_auc_score(y_test, X_test[:, 1]):.3f}")
```

Output:

```text
supervise on visible links  test AUC 0.500   distinct test scores: 1
hidden-link supervision     test AUC 0.735   distinct test scores: 163
reference: f2 alone (no learning)  test AUC 0.754
```

Trained on visible links, the model discovers that `f1` separates the classes perfectly and ignores `f2`. At test time `f1 = 0` for every test pair (the test edges are hidden), so every test pair gets the **same** score — AUC exactly 0.5, with one distinct value. Trained on hidden links (where `f1 = 0` for the supervised positives, as it will be at test time), the model must learn from `f2` and recovers almost all of its signal (0.735 vs 0.754 for `f2` alone). A GNN with the visible graph as input can fall into the same trap more subtly: it can learn to recognize "an edge to this node is present in my neighbourhood". HOW_IT_WORKS reports that, during development, switching to hidden-link supervision raised validation AUC from 0.71 to 0.92.

### 4.6 Tuning on the test folds versus nested cross-validation

Twenty candidate values of $k$ for a $k$-nearest-neighbour classifier (20 "configurations"), 100 samples, 30 simulated studies. For each study we compare: the best configuration's CV score on the same folds used to choose it; a nested CV estimate; and the true AUC of the chosen model, measured on 2,000 fresh samples. The AUC uses the rank formula (a fast equivalent of `roc_auc_score`).

```python
import numpy as np
from scipy.stats import rankdata

def roc_auc_score(y, s):
    """Fast AUC via the Mann-Whitney rank formula (ties get average ranks)."""
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

KS = list(range(1, 40, 2))                     # 20 candidate values of k = 20 "configurations"

def knn_scores(Xtr, ytr, Xte):
    """Score = fraction of positive labels among the k nearest training points, for every k."""
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    nn = ytr[np.argsort(d, axis=1)]                     # neighbour labels, nearest first
    csum = np.cumsum(nn, axis=1)
    return {k: csum[:, k - 1] / k for k in KS}

def folds(n, k, rng):
    return np.array_split(rng.permutation(n), k)

def cv_auc_per_k(X, y, rng, n_folds=5):
    res = {k: [] for k in KS}
    for te in folds(len(y), n_folds, rng):
        tr = np.setdiff1d(np.arange(len(y)), te)
        S = knn_scores(X[tr], y[tr], X[te])
        for k in KS:
            res[k].append(roc_auc_score(y[te], S[k]))
    return {k: np.mean(v) for k, v in res.items()}

def make_data(n, rng, signal):
    X = rng.standard_normal((n, 20))
    logit = signal * (X[:, 0] - X[:, 1])                # signal = 0 -> labels are pure noise
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

for signal in (0.0, 0.7):
    rows = []
    for rep in range(30):
        rng = np.random.default_rng(rep)
        X, y = make_data(100, rng, signal)
        X_new, y_new = make_data(2000, rng, signal)     # fresh data from the same source = truth
        # (1) "tune on test": evaluate all 20 k on the same 5 folds and report the best score
        cv = cv_auc_per_k(X, y, np.random.default_rng(1000 + rep))
        k_best = max(cv, key=cv.get)
        tuned_on_test = cv[k_best]
        # (2) nested CV: inside each outer training fold, choose k by an inner 5-fold CV
        outer = []
        for te in folds(len(y), 5, np.random.default_rng(2000 + rep)):
            tr = np.setdiff1d(np.arange(len(y)), te)
            inner = cv_auc_per_k(X[tr], y[tr], np.random.default_rng(3000 + rep))
            k_in = max(inner, key=inner.get)
            outer.append(roc_auc_score(y[te], knn_scores(X[tr], y[tr], X[te])[k_in]))
        # (3) the truth: the model chosen in (1), trained on all 100 points, scored on new data
        truth = roc_auc_score(y_new, knn_scores(X, y, X_new)[k_best])
        rows.append((tuned_on_test, np.mean(outer), truth))
    R = np.array(rows); r, se = R.mean(0), R.std(0, ddof=1) / np.sqrt(len(R))
    print(f"signal={signal}:  tuned-on-test {r[0]:.3f} +/- {se[0]:.3f} | "
          f"nested CV {r[1]:.3f} +/- {se[1]:.3f} | true AUC {r[2]:.3f} +/- {se[2]:.3f}")
```

Output:

```text
signal=0.0:  tuned-on-test 0.571 +/- 0.009 | nested CV 0.522 +/- 0.012 | true AUC 0.500 +/- 0.002
signal=0.7:  tuned-on-test 0.647 +/- 0.014 | nested CV 0.591 +/- 0.013 | true AUC 0.605 +/- 0.006
```

With no signal at all (`signal=0.0`), choosing the best of 20 configurations on the evaluation folds reports AUC 0.571 for a model whose true AUC is 0.500 — an optimism of 0.07, many standard errors away from zero. Nested CV gives 0.522, within two standard errors of the truth. With real signal the pattern persists: tuned-on-test 0.647 versus a true 0.605. Nested CV (0.591) is slightly *pessimistic* here, because each inner model trains on only 64 samples instead of 100 — the usual, acceptable direction of error for nested CV.

### 4.7 Paired comparison of two methods across folds (real project numbers)

The per-fold AUC and AUPR of MV-HGAT and SCMFDD on Fdataset (25 folds; copied from `results/Fdataset/cv5/*.json` as they were on 1 October 2026) and the five ablation folds of the full model versus the variant without hidden-link supervision (from `results/Fdataset/ablation_cv5/`).

```python
import numpy as np
from scipy import stats

def compare(a, b, k, label):
    """Paired comparison of two methods evaluated on the SAME folds (a, b = per-fold scores)."""
    d = a - b
    n = len(d)
    t_p, p_p = stats.ttest_rel(a, b)                       # paired t-test
    corr = 1 / n + (1 / k) / (1 - 1 / k)                   # Nadeau-Bengio: 1/n + n_test/n_train
    t_c = d.mean() / np.sqrt(corr * d.var(ddof=1))
    p_c = 2 * stats.t.sf(abs(t_c), df=n - 1)
    p_w = stats.wilcoxon(a, b).pvalue                      # signed-rank test
    print(f"{label}")
    print(f"   means {a.mean():.4f} vs {b.mean():.4f}  diff {d.mean():+.4f}  "
          f"sd(diff) {d.std(ddof=1):.4f}  corr(a,b) {np.corrcoef(a, b)[0, 1]:+.2f}")
    print(f"   paired t p={p_p:.3g} | corrected t p={p_c:.3g} | Wilcoxon p={p_w:.3g} | "
          f"d_z {d.mean() / d.std(ddof=1):+.2f} | wins {np.sum(d > 0)}/{n}")

# Per-fold results on Fdataset (5-fold CV x 5 repeats = 25 folds, identical splits for
# every method), copied from results/Fdataset/cv5/mvhgat.json and scmfdd.json.
mv_aupr = np.array([.5092, .4659, .5119, .5432, .4715, .4561, .4883, .5084, .5059, .4730, .5066, .4732,
                    .5019, .4804, .4779, .5050, .4923, .4734, .5000, .4771, .4798, .5323, .4936, .4012, .4664])
sc_aupr = np.array([.4596, .5157, .4951, .4845, .4848, .4711, .5072, .5063, .4643, .4731, .4537, .4882,
                    .5148, .5309, .5008, .5081, .4996, .4911, .4928, .5053, .5215, .5154, .5125, .4728, .4967])
mv_auc = np.array([.9402, .9395, .9349, .9557, .9467, .9344, .9366, .9499, .9421, .9307, .9402, .9438,
                   .9282, .9426, .9382, .9377, .9420, .9436, .9364, .9445, .9287, .9486, .9279, .9263, .9417])
sc_auc = np.array([.8971, .8707, .8931, .9067, .9044, .8837, .8947, .9002, .8856, .8892, .8998, .8940,
                   .8775, .9047, .8838, .9078, .8996, .8939, .8777, .9031, .8944, .9109, .8918, .8807, .8911])
compare(mv_auc, sc_auc, 5, "AUC:  MV-HGAT vs SCMFDD (25 folds)")
compare(mv_aupr, sc_aupr, 5, "AUPR: MV-HGAT vs SCMFDD (25 folds)")

# Ablation on Fdataset, 5 folds (results/Fdataset/ablation_cv5): full model vs. the
# variant WITHOUT hidden-link supervision. Same 5 test folds for both.
full = np.array([.5092, .4659, .5119, .5432, .4715])
no_hidden = np.array([.5507, .5592, .5336, .5724, .5146])
compare(no_hidden, full, 5, "AUPR: w/o hidden-link supervision vs full (5 folds)")
```

Output:

```text
AUC:  MV-HGAT vs SCMFDD (25 folds)
   means 0.9392 vs 0.8934  diff +0.0458  sd(diff) 0.0085  corr(a,b) +0.59
   paired t p=2.08e-19 | corrected t p=5.27e-10 | Wilcoxon p=1.23e-05 | d_z +5.37 | wins 25/25
AUPR: MV-HGAT vs SCMFDD (25 folds)
   means 0.4878 vs 0.4946  diff -0.0069  sd(diff) 0.0329  corr(a,b) +0.09
   paired t p=0.307 | corrected t p=0.702 | Wilcoxon p=0.174 | d_z -0.21 | wins 8/25
AUPR: w/o hidden-link supervision vs full (5 folds)
   means 0.5461 vs 0.5003  diff +0.0458  sd(diff) 0.0280  corr(a,b) +0.51
   paired t p=0.0217 | corrected t p=0.0716 | Wilcoxon p=0.0625 | d_z +1.63 | wins 5/5
```

Reading the three comparisons:
1. **AUC, MV-HGAT vs SCMFDD:** a large, consistent difference (+0.046; 25/25 wins; $d_z = 5.4$). Even the corrected test gives $p \approx 5\times10^{-10}$. A defensible claim.
2. **AUPR, MV-HGAT vs SCMFDD:** −0.007, with fold differences scattered both ways (8/25 wins), $d_z = -0.21$, all tests non-significant. The defensible claim is "no detectable difference in AUPR" — not "MV-HGAT is better". (Chapter 05, Unit B2, explains why AUC and AUPR can disagree.) Note how weakly the two methods' AUPRs correlate across folds (+0.09): pairing gains little here.
3. **Hidden-link supervision ablation:** +0.046 AUPR in favour of switching it off, 5/5 folds. Naive paired $t$: $p = 0.022$. Corrected: $p = 0.072$. Wilcoxon: $p = 0.0625$, its floor for 5 pairs. Suggestive, not conclusive, and measured on test folds — Section 5.6 explains what to do.

### 4.8 Ten ablations at once: effect sizes and the Holm correction

```python
import numpy as np
from scipy import stats

# Per-fold AUPR, Fdataset ablation (results/Fdataset/ablation_cv5/*.json), 5 identical folds.
R = {
    "full":             [.5092, .4659, .5119, .5432, .4715],
    "no_prop_head":     [.4076, .4907, .3915, .4421, .4333],
    "no_degree_gate":   [.4671, .4794, .4838, .5141, .4481],
    "no_cold_practice": [.5216, .5034, .5191, .5503, .4499],
    "no_hidden_sup":    [.5507, .5592, .5336, .5724, .5146],
    "no_attention":     [.5021, .4525, .4687, .5448, .4200],
    "no_gene_views":    [.4293, .4838, .4495, .4915, .4839],
    "with_bridge":      [.5121, .4905, .4682, .4495, .4618],
    "no_ecfp":          [.4614, .4476, .4316, .4947, .4787],
    "no_semantic":      [.5164, .5150, .5096, .5222, .4851],
    "bench_only":       [.4804, .4875, .4658, .5120, .4594],
}
full = np.array(R.pop("full"))
rows = []
for name, v in R.items():
    d = np.array(v) - full
    p = stats.ttest_rel(v, full).pvalue
    rows.append((name, d.mean(), d.mean() / d.std(ddof=1), p))
# Holm step-down correction for 10 simultaneous comparisons
order = np.argsort([r[3] for r in rows])
m, running, holm = len(rows), 0.0, {}
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * rows[i][3]))
    holm[i] = running
print(f"{'variant':17s} {'dAUPR':>7s} {'d_z':>6s} {'p (paired t)':>12s} {'p (Holm)':>9s}")
for i, (name, dm, dz, p) in enumerate(rows):
    print(f"{name:17s} {dm:+7.4f} {dz:+6.2f} {p:12.4f} {holm[i]:9.4f}")
```

Output:

```text
variant             dAUPR    d_z p (paired t)  p (Holm)
no_prop_head      -0.0673  -1.12       0.0666    0.5860
no_degree_gate    -0.0218  -1.04       0.0800    0.5860
no_cold_practice  +0.0085  +0.41       0.4158    0.9341
no_hidden_sup     +0.0458  +1.63       0.0217    0.2171
no_attention      -0.0227  -0.98       0.0946    0.5860
no_gene_views     -0.0327  -0.73       0.1784    0.8506
with_bridge       -0.0239  -0.52       0.3114    0.9341
no_ecfp           -0.0375  -1.13       0.0651    0.5860
no_semantic       +0.0093  +0.36       0.4643    0.9341
bench_only        -0.0193  -0.75       0.1701    0.8506
```

The largest effects are removing the propagation head (−0.067), removing the ECFP view (−0.038), removing the gene views (−0.033) and removing hidden-link supervision (+0.046). None survives the Holm correction with five folds. This is not a reason to hide the ablation; it is a reason to present it as effect sizes with uncertainty, to run more repeats for the components you want to make claims about, and to check the key ones on the second dataset.

### 4.9 The winner's curse in numbers

```python
import numpy as np
from scipy import stats

# Expected maximum of m independent N(0,1) draws, exact (numerical) and the classic approximation
for m in [2, 5, 10, 20, 100]:
    x = np.linspace(-10, 10, 200001)
    pdf = m * stats.norm.pdf(x) * stats.norm.cdf(x) ** (m - 1)      # density of the maximum
    exact = np.trapezoid(x * pdf, x)
    print(f"m={m:3d}  E[max] = {exact:.3f} sigma   sqrt(2 ln m) = {np.sqrt(2 * np.log(m)):.3f}")

# The "best seed" trap: a model whose AUPR is 0.50 on average, seed-to-seed sd 0.02
rng = np.random.default_rng(0)
runs = 0.50 + 0.02 * rng.standard_normal((100_000, 10))            # 10 seeds, many replications
print(f"\nreport the mean of 10 seeds : {runs.mean(1).mean():.4f}")
print(f"report the best of 10 seeds : {runs.max(1).mean():.4f}   (inflation {runs.max(1).mean() - 0.5:+.4f})")
```

Output:

```text
m=  2  E[max] = 0.564 sigma   sqrt(2 ln m) = 1.177
m=  5  E[max] = 1.163 sigma   sqrt(2 ln m) = 1.794
m= 10  E[max] = 1.539 sigma   sqrt(2 ln m) = 2.146
m= 20  E[max] = 1.867 sigma   sqrt(2 ln m) = 2.448
m=100  E[max] = 2.508 sigma   sqrt(2 ln m) = 3.035

report the mean of 10 seeds : 0.5000
report the best of 10 seeds : 0.5308   (inflation +0.0308)
```

The first table is the $\mathbb{E}[\max]$ used in Section 2.4 (computed by integrating the density of the maximum, $m\,\varphi(x)\,\Phi(x)^{m-1}$). The second part shows the "best seed" trap: with 10 seeds and a seed-to-seed standard deviation of 0.02, the best seed is on average 0.031 above the model's real level — an improvement of the kind papers celebrate, produced by nothing.

---

## 5. In this project

### 5.1 Leakage audit: every channel, and the line that closes it

| Channel | Where it could enter | How the project prevents it | Code |
|---|---|---|---|
| Test links in the input graph | association edges of the GNN | the graph is built from the fold's `A_train` | `run_kfold`: `A_tr[test_pos] = 0`; `MVHGATMethod.build(data, A_train)` |
| Test links in features | association rows used as node features | features recomputed from the *visible* links only | `fit_predict.features(Am)` |
| Test links in the propagation head / degrees | `K @ A`, link counts | computed from visible links only | `propagation(Am)`, `degrees(Am)` |
| Similarity computed from labels | any drug–drug or disease–disease view | **no view uses the association matrix**: CDK, ECFP (structure), MimMiner (text), MONDO (ontology), CTD genes | `02_build_features.py` |
| Labels disguised as features | CTD therapeutic/inferred links | inferred and therapeutic-only gene–disease links excluded; curated chemical–disease links used only for validation | `ctd_disease_genes`, `ctd_curated_chem_disease` |
| Negatives from test cells | training negative pool | `neg_mask` excludes the fold's test negatives; test positives are excluded because they are 1 in `A` | `run_kfold`: `neg_mask[test_neg] = False`; `neg_pool = neg_mask & (A_train == 0)` |
| Cold-start leakage | the held-out disease's links anywhere | whole column hidden from `A_train` and from `neg_mask` | `run_lodo` |
| Tuning on test | hyper-parameters, variants | tuned on a separate validation split | `05_sensitivity.py` (limitation: Section 5.5) |
| Train/test mismatch (edge detector) | supervising on visible links | hidden-link supervision | `fit_predict`, `supervise_hidden=True` |

### 5.2 `src/drepo/evaluation.py` — what is hidden, and where negatives may come from

```python
def kfold_splits(A, k, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(A.ravel() > 0)
    neg = np.flatnonzero(A.ravel() == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pf, nf = np.array_split(pos, k), np.array_split(neg, k)
    for f in range(k):
        yield f, pf[f], nf[f]
```

Positives **and** negatives are split into $k$ folds separately (a stratified split over cells), so each test fold has about $1/k$ of the links and $1/k$ of the unknown cells, and every cell is tested exactly once per repeat. The split depends only on `seed`, so every method and every ablation variant sees identical folds — the precondition for paired comparisons.

```python
for f, test_pos, test_neg in kfold_splits(A, k, seed + r):
    A_tr = A.copy().ravel()
    A_tr[test_pos] = 0
    A_tr = A_tr.reshape(shape)
    neg_mask = (A == 0).ravel()
    neg_mask[test_neg] = False
    neg_mask = neg_mask.reshape(shape)

    S = method.fit_predict(data, A_tr, neg_mask, seed=seed * 1000 + r * 100 + f)
```

`A_tr` is the only view of the links that a method receives. `neg_mask` is pool (c) of Section 4.4: it starts from the zeros of the *full* matrix (which would be pool (b), leaky) and then removes the fold's test negatives, so test cells of both classes are out of reach. Note the subtlety: `neg_mask` on its own *would* leak if a method sampled from it without removing the test negatives — the second line is what makes it clean. The model seed is a deterministic function of repeat and fold: common random numbers across methods and variants.

```python
def run_lodo(method, data, diseases=None, seed=0, verbose=True):
    ...
        A_tr = A.copy()
        A_tr[:, j] = 0
        neg_mask = A == 0
        neg_mask[:, j] = False
```

For cold start the whole column $j$ disappears from the links *and* from the negative pool.

### 5.3 `src/drepo/methods.py::MVHGATMethod.fit_predict` — visible links only, hidden-link supervision

```python
relations, X0, graphs = self.build(data, A_train)
A_full = t(A_train > 0)

def features(Am):
    # everything derived from links uses only the VISIBLE links Am
    ...
def propagation(Am):
    """One (drugs x diseases) score slice per view, from VISIBLE links only."""
    ...
pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)
neg_pool = t(np.flatnonzero((neg_mask & (A_train == 0)).ravel()), torch.long)
```

Read the names carefully: `A_full` here means "all **training** links of this fold" — it is built from `A_train`, never from the benchmark matrix. Everything link-derived (features, propagation, degrees, the association edges of the graph) is a function of a visible-link matrix `Am`. `neg_pool` intersects the evaluation's `neg_mask` with the zeros of `A_train`: belt and braces.

```python
for _ in range(c.epochs):
    sup = pos
    if c.drop_edge > 0 or c.cold_frac > 0:
        hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge
        if c.cold_frac > 0:
            cold = torch.rand(n_d, device=DEVICE) < c.cold_frac
            hide |= cold[pos % n_d]
        Am = torch.zeros(n_r * n_d, dtype=torch.bool, device=DEVICE)
        Am[pos[~hide]] = True
        Am = Am.view(n_r, n_d)
        if c.supervise_hidden:
            sup = pos[hide]
    ...
    logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
    ...
    neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]
    loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
```

Each epoch hides 20% of the training links (`drop_edge`) plus all links of about 10% of diseases (`cold_frac`), rebuilds the graph, features and propagation from what remains, and — with `supervise_hidden=True` — computes the positive part of the loss only on the hidden links. Training pairs therefore look like test pairs: their own edge is absent from the input (Section 4.5). Negatives are re-sampled from the clean pool every epoch. At the end, the model predicts with all training links visible, exactly as it will see the graph at test time.

### 5.4 Baselines: same splits, same information?

```python
def bench_sims(data):
    return fill_missing(data.drug_views[0]), fill_missing(data.disease_views[0])
```

All five baselines receive `A_train` and use only the two benchmark similarity matrices, as in their papers. Their leakage status:
- **MBiRW, DRRS, SCMFDD** fit `A_train` directly, treating *all* its zeros (including the fold's test cells, positive and negative alike) as zeros — pool (a) of Section 4.4: no leakage, because test positives and test negatives are indistinguishable to them. DRRS's observation mask `Om = (T != 0)` similarly treats every zero as unobserved.
- **NIMCGCN, LAGCN** use `weighted_bce(logits, A, neg_mask)`, whose mask `(A > 0) | neg_mask` covers training positives and clean negatives only — pool (c).

Fairness, item by item (Section 2.5):
- *Same splits:* yes (`kfold_splits` with the same seed).
- *Same information:* **no, by design** — MV-HGAT uses six views, the baselines two. That is why the ablation **"benchmark similarities only"** exists (Section 8, Q2).
- *Same tuning budget:* HOW_IT_WORKS states that the baselines' key hyper-parameters were tuned on the same validation split, but the repository contains no script that does this (the constructors simply hold default values, e.g. `MBiRW(alpha=0.3, l=2, r=2)`, `DRRS(tau_rel=0.005, …)`). The tuning cannot be re-run or audited from code. Add a `05b_tune_baselines.py` that gives each baseline a stated budget on the same validation split, and report that budget.
- *Re-implementation risk:* NIMCGCN and LAGCN reach only 0.096 and 0.133 AUPR on Fdataset, a third or less of the other baselines (0.31–0.49), although both are neural models of comparable or greater capacity. (NIMCGCN was also designed for miRNA–disease rather than drug–disease data.) A gap that large is a warning sign: check these two re-implementations for bugs and under-tuning, say in the paper that they may be weaker than the originals, and base the main claims on the strong baselines (SCMFDD, DRRS, MBiRW).

### 5.5 Hyper-parameter tuning and the acknowledged limitation

```python
rng = np.random.default_rng(12345)
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
val_pos = rng.choice(pos, len(pos) // 5, replace=False)
val_neg = rng.choice(neg, len(neg) // 5, replace=False)
A_tr = A.copy().ravel()
A_tr[val_pos] = 0
...
neg_mask[val_neg] = False
```

`05_sensitivity.py` builds a **validation split** with the same hiding logic as a CV fold, varies one hyper-parameter at a time around the defaults, and averages over seeds. This is the right *kind* of procedure: tuning never looks at the CV test folds' scores. HOW_IT_WORKS (assumption 10) states its limitation honestly: **the validation cells are not removed from the later CV**, so about 20% of every CV test fold's positives were validation positives during tuning (Section 3, Worked example 4). And the architectural decisions made during development (propagation head, degree gate, cold-start practice) were made on the same validation split. The resulting optimism is selection bias rather than training contamination and is probably small, but it is real. Two remedies, in increasing cost: (1) carve a **final hold-out** (e.g. 10% of links) that is excluded from both tuning and CV and evaluated once at the very end; (2) **nested CV**: repeat the tuning inside each outer training fold (with 5 outer folds and the current 19-setting grid × 2 seeds, roughly 5 × 38 = 190 extra training runs).

### 5.6 The open question: should hidden-link supervision be switched off?

The ablation found AUPR 0.546 without hidden-link supervision versus 0.500 with it (5/5 folds; corrected $p = 0.07$; Holm-adjusted $p = 0.22$). What would an experienced researcher do?

1. **Not change the default on the basis of this result.** It comes from test folds. Choosing the variant that scored best on them makes those folds a validation set and biases every number subsequently reported for "the model".
2. **Re-run the comparison where decisions belong: on the validation split**, e.g. `MVHGATMethod(supervise_hidden=False)` versus the default in `05_sensitivity.py`'s split, with at least 5 seeds, paired by seed, on both datasets, for both warm-start and LODO-style validation (hidden-link supervision may matter more for cold start).
3. **Decide the rule before looking:** e.g. "switch only if validation AUPR improves by more than 0.01 on both datasets and cold-start validation does not degrade".
4. **Form hypotheses, label them as such:** HOW_IT_WORKS reports that hidden-link supervision raised validation AUC from 0.71 to 0.92 in an *earlier* version of the model, before the propagation head and degree gate were added. The propagation head scores pairs from neighbours' links without relying on the pair's own edge, so the edge-detector shortcut may be less dangerous now; meanwhile hidden-link supervision supervises only about 20% of the positives per epoch, which may leave the model under-trained at 600 epochs. These are testable on the validation split.
5. **If the rule says switch:** change the default, re-run the main CV once, and write in the paper that the setting was chosen on the validation split after an ablation suggested it. Keep the original ablation in the supplement.

This is what "the test set is touched once" means in practice: not that you never look at test results, but that what you learn from them does not flow back into the model without going through validation data first.

### 5.7 Feature construction: residual risks worth stating

`02_build_features.py` closes the big channels, but a careful methods section also mentions the smaller ones:

- **Temporal (L3.1).** The labels date from 2011 (Fdataset) and 2016 (Cdataset); the CTD gene data were downloaded in 2026. Some chemical–gene interactions were published *because* a drug was being studied for a disease that is one of its indications. Such features carry a faint echo of the labels. The marker/mechanism filter reduces this for disease genes, but not for drug genes. Unit D4's Exercise 14 shows that gene overlaps are enriched among known links (2.1× for mechanism genes) — that is the signal the gene views exploit, and some of it may be of this kind.
- **Phenotype text.** The benchmark's disease similarity (MimMiner) is text-mined from OMIM records; OMIM entries sometimes mention treatment response. This is inherited from the benchmark and shared by all baselines, so it does not bias comparisons, but it qualifies claims of "label-free" disease similarity.
- **Near-duplicate entities (L3.2).** Twelve groups of our drugs share an InChIKey skeleton (enantiomers, diastereomers; Unit D4 Section 4.6) and often share indications; Alzheimer-type and similar numbered disease subtypes share drugs. In warm-start CV this is legitimate (the claim is about completing a matrix whose other entries are known), but a claim about *new* drugs would need group-aware splits (Exercise 12).
- **Transductive, label-free preprocessing.** The gene vocabulary keeps genes touching at least two drugs or diseases of the dataset, counted over *all* entities. This uses no links, so it is not leakage; it is a transductive choice, legitimate because all entities are known at prediction time.

### 5.8 Cross-dataset validation and non-independent datasets

`07_cross_dataset.py` trains on all links of one benchmark and tests on the links that only the other benchmark has, among shared drugs and diseases. The logs (`results/logs/cross_FC.log`, then `cross_CF.log`) begin:

```text
shared drugs 574/593, shared diseases 307/313
source links among shared entities 1888, target links 1945, NEW target links to recover 57 out of 174330 candidate pairs

shared drugs 574/663, shared diseases 307/409
source links among shared entities 1945, target links 1888, NEW target links to recover 0 out of 174273 candidate pairs
```

Among the shared entities, **Fdataset's 1,888 links are a strict subset of Cdataset's 1,945**: every F link is also a C link. So C → F has nothing to test (the script detects this and stops), and the two benchmarks are not independent samples — results "on two datasets" are not two independent confirmations, and Demšar-style tests across them would overstate the evidence. F → C remains a genuinely external test of 57 links that were never visible; its random-baseline AUPR is 57/174,330 ≈ 0.0003, and MV-HGAT's AUPR of 0.0086 is about 26 times that, with MBiRW (0.0136) ahead — another reason to keep strong simple baselines in every table.

### 5.9 A model info sheet for this project (draft)

> **L1 — clean separation.** In every fold, all link-derived quantities (graph edges, node features, propagation inputs, degree gate, training positives and the negative pool) are computed from the fold's training links; test positives and test negatives are excluded from training (`evaluation.py`, `methods.py`). No similarity is computed from the association matrix. Hyper-parameters were chosen on a validation split, not on test folds; this split overlaps the cells later used in CV (limitation). Ablations were run on test folds and are reported as explanations; no design decision was changed on their basis.
> **L2 — legitimate features.** Drug views: CDK and ECFP4 fingerprints (structures), CTD human chemical–gene interactions. Disease views: MimMiner phenotype similarity, MONDO semantic similarity, CTD curated marker/mechanism gene–disease links. Excluded as label-derived: CTD inferred and therapeutic-only gene–disease links, all chemical–disease links (used only to check case-study predictions).
> **L3 — test distribution.** Warm-start CV estimates matrix completion for drugs and diseases with other known links; LODO estimates performance for diseases with no known drugs. Unknown pairs are treated as negatives (pessimistic). Benchmarks favour well-studied drugs; near-duplicate drugs and disease subtypes exist; F ⊂ C among shared entities. Claims about entirely new drugs are not supported by these protocols.

---

## 6. Common mistakes and misconceptions

1. **"We used a separate test set, so there is no leakage."** Separation must hold in *every* step: preprocessing, feature construction, similarity computation, negative sampling, early stopping, model choice.
2. **"Transductive means the test edges may be in the graph."** Transductive means the test *nodes* are present; the test *edges* never are.
3. **Building the negative pool from the full matrix's zeros.** It silently excludes the test positives and leaks (Section 4.4).
4. **Computing GIP kernels, degrees, node2vec embeddings or normalizations once, before CV.** Recompute them per fold from `A_train`.
5. **"Leakage only matters for deep models."** A cosine similarity and a matrix product reach AUC 0.99 on noise when the similarity uses the full matrix (Section 4.3).
6. **Changing the model after looking at test ablations.** That is tuning on test; do it on validation data.
7. **Reporting the best seed, the best epoch or the best of several runs.** Report the mean and spread over all of them.
8. **Unpaired tests on paired data,** or naive paired $t$-tests on overlapping CV folds. Use paired, corrected tests and report effect sizes.
9. **Equating "not significant" with "no effect" or "significant" with "important".** Five folds have little power; thousands of folds make trivial differences significant.
10. **Ignoring multiple comparisons** in ablation tables with many variants.
11. **Comparing against baselines with less information or no tuning** and attributing the whole margin to the architecture.
12. **Quoting published baseline numbers next to your own numbers** obtained with different splits, preprocessing and negatives.
13. **Treating two related benchmarks as independent confirmations** (F ⊂ C).
14. **Believing that seeds make results representative.** Seeds make them repeatable; variability needs repeats.
15. **Claiming "new drug" performance from warm-start CV** (L3: the split must match the claim).

---

## 7. Exercises

Difficulty: ★ conceptual, ★★ practical, ★★★ coding. Coding solutions were executed with the project environment; outputs are real.

**Exercise 1 (★).** Classify each scenario with Kapoor & Narayanan's taxonomy: (a) imputing missing similarity rows with the mean similarity computed over all drugs, then CV; (b) using "number of clinical trials testing drug X in disease Y" as a feature; (c) a temporal claim ("we predict future approvals") evaluated with random CV; (d) the same drug under two DrugBank IDs in train and test; (e) reporting training-set AUC of a model trained on all links; (f) choosing the 50 genes most correlated with indications on the full matrix; (g) testing only on popular diseases with more than 10 drugs while claiming performance for rare diseases; (h) evaluating a drug-disjoint split where enantiomers are split across folds.

<details><summary>Solution</summary>

(a) L1.2 — but only if the imputation uses labels or test information; imputing a *similarity* with a mean over drugs uses no labels, so it is transductive preprocessing and essentially harmless; if the imputation used association data (e.g. mean of drugs sharing indications), it would leak. (b) L2 — illegitimate feature: trial counts are partly the outcome. (c) L3.1 — temporal leakage: random CV lets the model learn from "future" links. (d) L1.4 — duplicates. (e) L1.1 — no test set. (f) L1.3 — feature selection on train + test. (g) L3.3 — sampling bias: the test distribution differs from the claim. (h) L3.2 — non-independence between train and test.

</details>

**Exercise 2 (★).** Leaky or clean? (a) A disease–disease similarity computed from shared known drugs, recomputed inside each fold from `A_train`. (b) The symmetric normalization $D^{-1/2}AD^{-1/2}$ of the drug–disease graph computed once before CV. (c) Early stopping on a validation split carved from the training links of each fold. (d) The project's `fill_missing` (filling unknown similarity rows with zeros and a diagonal of 1). (e) Using LODO but keeping the held-out disease's links in the degree feature.

<details><summary>Solution</summary>

(a) Clean: link-derived, but from training links only. (b) Leaky: degrees include test edges (channel 2). (c) Clean: the stopping rule never sees test cells. (d) Clean: no labels involved. (e) Leaky: the degree reveals how many drugs the "new" disease has (cold-start leakage).

</details>

**Exercise 3 (★).** Explain in your own words why sampling negatives from the zeros of `A_train` (pool a) does not leak, while sampling from the zeros of the full `A` (pool b) does. Which is what `neg_mask` alone would give, and what line of `run_kfold` makes it clean?

<details><summary>Solution</summary>

A model can only exploit information that distinguishes test positives from test negatives. In pool (a), both kinds of test cell are zeros of `A_train` and are sampled as negatives with the same probability, so training pushes both down equally; the ranking between them is unaffected (a few true links are trained as negatives — a pessimistic effect). In pool (b), the pool is defined by the full matrix, which knows that test positives are 1: they are excluded, test negatives are included, so the model learns "these specific test cells are 0" and ranks the untouched test positives above them. `(A == 0)` alone is pool (b). `neg_mask[test_neg] = False` removes the test negatives, giving pool (c), in which neither class of test cell is used.

</details>

**Exercise 4 (★★).** You compare 10 hyper-parameter settings, each evaluated once on the test folds; the seed-to-seed standard deviation of AUPR is 0.025, and the settings are in truth equally good. What inflation do you expect if you report the best? What if you then also report the best of 5 seeds for that setting?

<details><summary>Solution</summary>

Best of 10: $\mathbb{E}[\max] = 1.539\sigma \approx 1.539 \times 0.025 = 0.038$. Then best of 5 seeds of the chosen setting adds roughly $1.163 \times 0.025 \approx 0.029$ (if seed noise is independent of the first selection), for a total optimism of about 0.067 — larger than most reported improvements in this literature.

</details>

**Exercise 5 (★★).** On the five ablation folds, the "benchmark similarities only" variant has AUCs (0.9385, 0.9432, 0.9392, 0.9563, 0.9384) and SCMFDD (first repeat of its CV, same folds) has (0.8971, 0.8707, 0.8931, 0.9067, 0.9044). Compute $\bar d$, $s_d$, the naive and the corrected $t$.

<details><summary>Solution</summary>

$d = (0.0414, 0.0725, 0.0461, 0.0496, 0.0340)$, $\bar d = 0.2436/5 = 0.04872$. Deviations: $-0.00732, 0.02378, -0.00262, 0.00088, -0.01472$; squares: $5.358\times10^{-5}, 5.655\times10^{-4}, 6.86\times10^{-6}, 7.7\times10^{-7}, 2.167\times10^{-4}$; sum $8.433\times10^{-4}$; $s_d^2 = 2.108\times10^{-4}$, $s_d = 0.01452$. Naive: $t = 0.04872/(0.01452/\sqrt5) = 0.04872/0.006494 = 7.50$ ($p \approx 0.002$, df 4). Corrected: $\text{SE} = \sqrt{0.45 \times 2.108\times10^{-4}} = 0.00974$, $t = 5.00$ ($p \approx 0.007$). The architecture alone beats SCMFDD in AUC even with identical inputs, and the difference survives the correction.

</details>

**Exercise 6 (★★).** Apply Holm's procedure at $\alpha = 0.05$ to $p = (0.004, 0.020, 0.030, 0.25)$. Compare with Bonferroni.

<details><summary>Solution</summary>

Sorted already. Holm: $4\times0.004 = 0.016$; $3\times0.020 = 0.060$; $2\times0.030 = 0.060$ (running max 0.060); $1\times0.25 = 0.25$. Adjusted: $(0.016, 0.060, 0.060, 0.25)$ → only the first is rejected. Bonferroni: $(0.016, 0.080, 0.120, 1.0)$ → also only the first. Holm is never worse; here the decisions coincide, but Holm's adjusted values are smaller.

</details>

**Exercise 7 (★★).** Design a nested CV for MV-HGAT on Fdataset with 5 outer folds, tuning `hidden`, `k` and `drop_edge` (4 values each, one at a time around the defaults as in `05_sensitivity.py`), 2 seeds, and a single inner validation split per outer fold. How many training runs does it cost, and what exactly is reported?

<details><summary>Solution</summary>

One-at-a-time grid: $4+4+4 = 12$ settings, of which the default appears three times ⇒ 10 distinct configurations. Per outer fold: $10 \times 2$ seeds on the inner split $= 20$ runs, plus 1 (or 2, one per seed) refit with the chosen configuration on the whole outer training part ⇒ 21–22 runs. Total $5 \times 22 = 110$ runs. Reported: the mean ± sd of the 5 outer-fold scores, each obtained by a configuration chosen without looking at that fold; also the chosen configuration per fold (if it varies a lot, the hyper-parameter barely matters). The inner split must hide links from the outer *training* part only, with its own `neg_mask`.

</details>

**Exercise 8 (★★).** You add a `target_r` drug view (Jaccard of DrugBank targets). Design the ablation with controls, and state which comparisons answer which question.

<details><summary>Solution</summary>

Variants, all on identical splits and seeds: (1) full model; (2) full + `target_r`; (3) full + *permuted* `target_r` (rows and columns permuted by the same random drug permutation — same distribution, wrong drugs) as a **negative control**; (4) `target_r` alone in the propagation head (no other views) versus (5) permuted `target_r` alone, to measure the view's standalone signal *above popularity*; (6) a positive control, e.g. all views removed, to confirm the evaluation is sensitive. Questions: (2) − (1) = the view's added value; (2) − (3) = whether that value comes from the view's *content* rather than from extra parameters or capacity; (4) − (5) = signal of the view alone. Run in warm start and LODO; report paired differences with corrected intervals; decide whether to keep the view using the **validation** split, not these test results.

</details>

**Exercise 9 (★★).** List what you would need to add or change for the baseline comparison in this project to satisfy all four parities of Section 2.5.

<details><summary>Solution</summary>

Same splits: already satisfied. Same information: keep the "benchmark similarities only" ablation in the main results table next to the baselines; optionally run baselines that can accept more views (e.g. SCMFDD with averaged similarity views) with the extra views. Same tuning budget: write and commit a baseline-tuning script on the same validation split with a stated budget (e.g. 10 configurations × 2 seeds per baseline, like MV-HGAT); report the chosen values. Same evaluation details: already shared (`metrics`, folds, negatives). Plus: label baselines as re-implementations, investigate NIMCGCN/LAGCN's low AUPR (bug or tuning), and keep published numbers in a separate table.

</details>

**Exercise 10 (★★★).** Many drug-repositioning papers use the Gaussian interaction profile (GIP) kernel computed from the association matrix as a similarity view. Demonstrate on random labels what happens when it is computed once from the full matrix versus inside each fold.

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def gip_kernel(A):
    """Gaussian interaction profile kernel between ROWS of A (van Laarhoven et al. 2011)."""
    sq = (A ** 2).sum(1)
    gamma = 1.0 / max(sq.mean(), 1e-12)                 # bandwidth normalised by mean row norm
    D = sq[:, None] + sq[None, :] - 2 * A @ A.T
    K = np.exp(-gamma * D)
    np.fill_diagonal(K, 0)
    return K

rng = np.random.default_rng(5)
A = (rng.random((300, 150)) < 0.03).astype(float)       # random labels: honest AUC = 0.5
pos, neg = np.flatnonzero(A.ravel()), np.flatnonzero(A.ravel() == 0)
rng.shuffle(pos); rng.shuffle(neg)
res = {"GIP from full A (leaky)": [], "GIP from A_train (clean)": []}
for tp, tn in zip(np.array_split(pos, 5), np.array_split(neg, 5)):
    A_tr = A.ravel().copy(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
    idx, y = np.r_[tp, tn], np.r_[np.ones(len(tp)), np.zeros(len(tn))]
    for key, K in [("GIP from full A (leaky)", gip_kernel(A)),
                   ("GIP from A_train (clean)", gip_kernel(A_tr))]:
        S = K @ A_tr                                    # propagate TRAINING links only
        res[key].append(roc_auc_score(y, S.ravel()[idx]))
for k, v in res.items():
    print(f"{k:26s} AUC {np.mean(v):.3f}")
```

Output:

```text
GIP from full A (leaky)    AUC 0.589
GIP from A_train (clean)   AUC 0.490
```

On pure noise the leaky GIP gives AUC 0.589; per-fold GIP gives chance level. The inflation is smaller than for cosine similarity in Section 4.3 because the GIP kernel with a mean-normalized bandwidth is very flat for sparse rows, but it is unmistakably there. Any similarity "from shared indications" must be recomputed from `A_train` in each fold.

</details>

**Exercise 11 (★★★).** Build a negative control for a similarity view: compare a real view, the same view with its drugs permuted, and no view. What does the permuted view alone score, and why is it not 0.5?

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(2)
gr, gd = rng.integers(0, 10, 300), rng.integers(0, 8, 150)
P = np.where((rng.random((10, 8)) < 0.2)[gr][:, gd], 0.25, 0.01)
A = (rng.random(P.shape) < P).astype(float)

def cosine(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1
    S = (M / n) @ (M / n).T; np.fill_diagonal(S, 0); return np.clip(S, 0, None)

# a "chemistry" view that genuinely reflects the drug groups, plus noise
F = np.eye(10)[gr] + 0.5 * rng.standard_normal((300, 10))
def knn(S, k=10):                            # keep each drug's k strongest neighbours (as the project does)
    keep = np.zeros_like(S, dtype=bool)
    keep[np.arange(len(S))[:, None], np.argsort(-S, 1)[:, :k]] = True
    return S * (keep | keep.T)
S_view = knn(cosine(F))
perm = rng.permutation(300)
S_perm = S_view[np.ix_(perm, perm)]          # negative control: same values, wrong drugs

def cv(score_fn, k=5):
    pos, neg = np.flatnonzero(A.ravel()), np.flatnonzero(A.ravel() == 0)
    r = np.random.default_rng(0); r.shuffle(pos); r.shuffle(neg); out = []
    for tp, tn in zip(np.array_split(pos, k), np.array_split(neg, k)):
        A_tr = A.ravel().copy(); A_tr[tp] = 0; A_tr = A_tr.reshape(A.shape)
        s = score_fn(A_tr).ravel()[np.r_[tp, tn]]
        y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
        out.append((roc_auc_score(y, s), average_precision_score(y, s)))
    return np.mean(out, 0)

variants = {
    "real chemistry view alone":           lambda A_tr: S_view @ A_tr,
    "permuted view alone (neg. control)":  lambda A_tr: S_perm @ A_tr,
    "profiles only (baseline)":            lambda A_tr: cosine(A_tr) @ A_tr,
    "profiles + 0.3 x real view":          lambda A_tr: (cosine(A_tr) + 0.3 * S_view) @ A_tr,
    "profiles + 0.3 x permuted view":      lambda A_tr: (cosine(A_tr) + 0.3 * S_perm) @ A_tr,
}
for name, f in variants.items():
    auc, ap = cv(f)
    print(f"{name:36s} AUC {auc:.3f}  AUPR {ap:.3f}")
```

Output:

```text
real chemistry view alone            AUC 0.629  AUPR 0.114
permuted view alone (neg. control)   AUC 0.546  AUPR 0.083
profiles only (baseline)             AUC 0.807  AUPR 0.211
profiles + 0.3 x real view           AUC 0.808  AUPR 0.210
profiles + 0.3 x permuted view       AUC 0.803  AUPR 0.209
```

The permuted view alone scores AUC 0.546, not 0.5: any score of the form $S A_\text{train}$ favours diseases with many training links (popular diseases are more often linked to *any* drug), so a view with no information still exploits **popularity**. The real view's own contribution is therefore $0.629 - 0.546$, not $0.629 - 0.5$. In combination with indication profiles the view adds almost nothing in this warm-start setting — profiles already capture the group structure — which mirrors the project's experience that extra views matter most in cold start. Without the control you would over-credit the view.

</details>

**Exercise 12 (★★★).** Show that even a *drug-disjoint* split leaks when near-duplicate drugs (e.g. enantiomers) can fall on both sides.

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(11)
n_pairs, n_dis = 100, 120
X = rng.standard_normal((n_pairs, 64))                   # "chemistry" of 100 parent drugs
X = np.vstack([X, X + 0.05 * rng.standard_normal(X.shape)])   # each drug has a near-identical twin
base = (rng.random((n_pairs, n_dis)) < 0.04).astype(float)    # indications unrelated to chemistry
twin = np.where(rng.random(base.shape) < 0.9, base, rng.random(base.shape) < 0.04)
A = np.vstack([base, twin])                              # twins share ~90% of indications
group = np.r_[np.arange(n_pairs), np.arange(n_pairs)]

Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
S = Xn @ Xn.T; np.fill_diagonal(S, 0); S = np.clip(S, 0, None)   # chemical similarity view

def auc_hiding(rows_folds):
    aucs = []
    for rows in rows_folds:
        A_tr = A.copy(); A_tr[rows] = 0                  # hide every link of the test drugs
        scores = S @ A_tr                                # score by similar TRAINING drugs
        aucs.append(roc_auc_score(A[rows].ravel(), scores[rows].ravel()))
    return np.mean(aucs)

drugs = rng.permutation(2 * n_pairs)
naive = [np.isin(np.arange(2 * n_pairs), f) for f in np.array_split(drugs, 5)]
grp = rng.permutation(n_pairs)
grouped = [np.isin(group, f) for f in np.array_split(grp, 5)]
print(f"drug-wise CV, twins may be split across folds : AUC {auc_hiding(naive):.3f}")
print(f"drug-wise CV, twins kept in the same fold     : AUC {auc_hiding(grouped):.3f}")
```

Output:

```text
drug-wise CV, twins may be split across folds : AUC 0.875
drug-wise CV, twins kept in the same fold     : AUC 0.502
```

Indications are random with respect to chemistry, so the honest answer is 0.5 — obtained only when twins stay in the same fold. When they are split, the test drug's near-identical twin in the training set "predicts" its indications (AUC 0.875). For a claim about new drugs, group by parent compound (InChIKey block 1, Unit D4).

</details>

**Exercise 13 (★★★).** Compute how the power to detect a true AUPR improvement of 0.02 (per-fold difference sd 0.03) grows with the number of folds $J$ under the corrected test.

<details><summary>Solution</summary>

```python
import numpy as np
from scipy import stats

delta, sd, rho = 0.02, 0.03, 1 / 4          # true AUPR gain, sd of per-fold differences, n_test/n_train
print(" J folds | naive SE | corrected SE | expected corrected t | power (alpha=0.05)")
for J in [5, 10, 25, 50, 100, 1000]:
    se_naive = sd / np.sqrt(J)
    se_corr = sd * np.sqrt(1 / J + rho)
    ncp = delta / se_corr                     # non-centrality of the corrected t statistic
    crit = stats.t.ppf(0.975, J - 1)
    power = stats.nct.sf(crit, J - 1, ncp) + stats.nct.cdf(-crit, J - 1, ncp)
    print(f"{J:8d} | {se_naive:8.4f} | {se_corr:12.4f} | {ncp:20.2f} | {power:6.2f}")
```

Output:

```text
 J folds | naive SE | corrected SE | expected corrected t | power (alpha=0.05)
       5 |   0.0134 |       0.0201 |                 0.99 |   0.12
      10 |   0.0095 |       0.0177 |                 1.13 |   0.17
      25 |   0.0060 |       0.0162 |                 1.24 |   0.22
      50 |   0.0042 |       0.0156 |                 1.28 |   0.24
     100 |   0.0030 |       0.0153 |                 1.31 |   0.25
    1000 |   0.0009 |       0.0150 |                 1.33 |   0.26
```

(Power is computed by treating the corrected statistic as non-central $t$ — an approximation.) The naive standard error shrinks like $1/\sqrt J$, but the corrected one levels off at $0.03/2 = 0.015$, so the expected statistic never exceeds 1.33 and power stays below about 26%. More repeats cannot rescue a small effect on one dataset; more datasets, larger effects or external validation can.

</details>

**Exercise 14 (★★).** Write a short pre-registration (decision protocol) for resolving the hidden-link-supervision question before running anything.

<details><summary>Solution</summary>

*Question:* does `supervise_hidden=False` improve MV-HGAT? *Data:* the validation split of `05_sensitivity.py` on Fdataset and Cdataset; test folds not used. *Runs:* default vs `supervise_hidden=False`, seeds 0–4, paired by seed; warm-start validation and a cold-start validation (hide all links of 10% of diseases). *Metric:* AUPR (primary), AUC (secondary). *Decision rule:* switch if mean paired validation AUPR gain > 0.01 on both datasets and cold-start AUPR does not drop by more than 0.005; otherwise keep. *Reporting:* whatever the outcome, report the validation comparison and the earlier test-fold ablation in the supplement; if switching, re-run the main CV once and state that the default was changed on validation evidence.

</details>

**Exercise 15 (★).** Sketch a temporal evaluation for this project and say which leakage type it addresses.

<details><summary>Solution</summary>

Use the indication approval dates (e.g. from DrugBank or FDA labels) to split links: train on indications known before year $T$ (say 2012), test on indications first approved after $T$, among the same drugs and diseases. Features must also be restricted to information available before $T$ (CTD interactions from papers published before $T$, which requires the PubMed IDs and dates). This addresses L3.1 (temporal leakage) and gives the most realistic estimate of prospective repositioning performance. Cdataset's extra links relative to Fdataset are a crude version of this idea (Section 5.8).

</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (Unit E1)

### Q1. List three ways information about a test link could leak into training in this project, and how the code prevents each.

A test link is a known drug–disease pair $(i,j)$ placed in the current fold's test set. Five routes, with the prevention in the code:

1. **Through the graph, the features and the propagation inputs.** If $(i,j)$ were an edge of the message-passing graph, drug $i$'s embedding would aggregate disease $j$, and the decoder would score the pair high because the edge is there; the same holds for association rows used as features, the propagation head $K A$ and the degree gate. *Prevention:* `run_kfold` zeroes the fold's test positives in `A_tr` before calling the method (`A_tr[test_pos] = 0`), and `MVHGATMethod.fit_predict` derives every link-based quantity — `build(data, A_train)`, `features(Am)`, `propagation(Am)`, `degrees(Am)` — from `A_train` (and, during training, from its visible subset `Am`). Section 4.3 shows what the alternative does: AUC 1.000 on noise.
2. **Through similarity views computed from the association matrix.** A "drugs that treat the same diseases are similar" view computed from the full matrix would make drug $i$ similar to other drugs treating $j$, and propagation would return the hidden link (AUC 0.993 on noise, Section 4.3). *Prevention:* **no view is computed from the association matrix** — `02_build_features.py` builds views from chemical structures (CDK, ECFP), OMIM text (MimMiner), the MONDO ontology and CTD genes. And the CTD gene views exclude inferred and therapeutic-only gene–disease links, which would carry treatment knowledge, i.e. labels (Unit D4, Q2); curated chemical–disease links are used only to check case studies.
3. **Through the negative pool.** If training negatives were sampled from the zeros of the full matrix, the pool would contain the fold's test negatives but not its test positives; a flexible model memorizes the former as low and ranks the latter above them (AUC 0.605 on noise, Section 4.4). *Prevention:* `neg_mask[test_neg] = False` in `run_kfold`, and `neg_pool = neg_mask & (A_train == 0)` in `fit_predict`; the baselines that use a mask (`weighted_bce`) receive the same `neg_mask`.
4. **Through hyper-parameter and design choices.** Choosing settings by their test score lets the test labels shape the model (winner's curse, Section 4.6). *Prevention:* `05_sensitivity.py` tunes on a separate validation split; the default configuration was set there. *Limitation:* the validation split overlaps the CV data (≈20% of each test fold's positives), and the open hidden-link question must be settled on validation, not on the test ablation (Section 5.6).
5. **In cold start.** For LODO, all of disease $j$'s links are test links. *Prevention:* `run_lodo` zeroes the whole column in `A_tr` and in `neg_mask`, so not even the disease's degree is visible.

(Any three, explained with the mechanism and the code line, answer the question; the full list is the audit table of Section 5.1.)

### Q2. Why do we also run an ablation "benchmark similarities only"?

Because the comparison with the baselines is **not same-information**. MV-HGAT uses six similarity views (two from the benchmark plus ECFP, CTD drug genes, MONDO semantics and CTD disease genes); the five baselines use only the two benchmark matrices, as in their papers. If MV-HGAT wins, the win could come from the **architecture** (attention over relations, propagation head, degree gate, training scheme) or from the **extra data**. These are different scientific claims — "a better model" versus "better data integration" — and Lipton & Steinhardt's "failure to identify the sources of empirical gains" is exactly the error of not distinguishing them.

The variant restricted to `chem_cdk` and `pheno_mim` gives MV-HGAT **the same information as the baselines**. Comparing it:
- **with the baselines** (same splits, same inputs) isolates the architecture's contribution. On Fdataset's first repeat (same five folds), benchmark-only MV-HGAT reaches AUC 0.943 versus SCMFDD's 0.894 (paired difference +0.049, corrected $p \approx 0.007$; Exercise 5) but AUPR 0.481 versus 0.488 — so in AUPR the architecture alone is on a par with the best baseline;
- **with the full model** isolates the extra data's contribution: AUPR 0.500 versus 0.481 (+0.019, paired $p = 0.17$ on five folds, not significant), with the ablations of individual views (w/o ECFP −0.038, w/o gene views −0.033) indicating where the gain comes from.

It is also a fairness and honesty device: it lets a reader compare like with like, it guards against attributing to a new architecture what the new data did, and it tells future users whether they need the extra data sources (with their licences and mapping effort, Unit D4) to get the benefit. Finally, like every ablation, it was run on test folds and serves to *explain* the result, not to choose the model.

---

## 9. Summary and cheat sheet

**Definition.** Leakage = information unavailable at prediction time reaches model building, or the test set does not represent the claim. It inflates results; it does not crash.

**Taxonomy (Kapoor & Narayanan 2023).** L1.1 no test set · L1.2 preprocessing on train+test · L1.3 feature selection on train+test · L1.4 duplicates · L2 illegitimate features · L3.1 temporal · L3.2 non-independence · L3.3 sampling bias.

**Graph link prediction — build everything per fold from `A_train`:** graph edges, normalizations, degrees, features, embeddings, similarities from links, propagation inputs. Negatives: zeros of `A` minus the fold's test cells (pool c), never the full matrix's zeros (pool b). LODO: hide the whole column everywhere. Supervise on hidden links to avoid the edge-detector shortcut.

**Selection.** Validation for choices, test once for reporting. $\mathbb{E}[\max_m Z] = 0.56, 1.16, 1.54, 1.87, 2.51$ for $m = 2, 5, 10, 20, 100$. Nested CV: outer folds for estimating, inner splits for choosing. Never report the best seed.

**Fair baselines.** Same splits · same information · same tuning budget · same evaluation details · strong simple baselines · re-implementations labelled.

**Ablations.** One change at a time · same splits and seeds · positive and negative (permuted) controls · effect sizes with uncertainty · Holm for many variants · explain, don't select.

**Statistics.**
$$t_\text{paired} = \frac{\bar d}{s_d/\sqrt J}, \qquad t_\text{corr} = \frac{\bar d}{s_d\sqrt{1/J + n_\text{test}/n_\text{train}}}\ (\text{df } J-1;\ \tfrac14 \text{ for 5-fold}).$$
Wilcoxon with 5 pairs: $p \ge 0.0625$. Effect sizes: $\bar d$ with CI, $d_z = \bar d/s_d$, win rate. Holm: $\tilde p_{(i)} = \max_{j\le i}\min(1,(m-j+1)p_{(j)})$. Across datasets: Wilcoxon (2 methods), Friedman + Nemenyi (many).

**Report.** Data versions · splits and seeds · leakage argument (model info sheet) · tuning protocol and budget for all methods · mean ± sd (say over what) · random baselines · paired corrected comparisons with effect sizes · all variants tried · limitations.

---

## 10. Curated further resources

All links checked on 1 October 2026. Free unless marked.

**Leakage**
- Kapoor S. & Narayanan A. (2023) *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4:100804. [doi:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804) — the taxonomy and model info sheets used in this unit. *Free.*
- [Reproducibility in ML-based science (Princeton)](https://reproducible.cs.princeton.edu/) — the authors' page with the survey table, model info sheet template and follow-ups. *Free.*
- Kaufman S., Rosset S., Perlich C. & Stitelman O. (2012) *Leakage in data mining: formulation, detection, and avoidance.* ACM TKDD 6(4):15. [doi:10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579) — learn–predict separation and legitimacy. *Paywalled (preprints circulate).*
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — data leakage with pipelines, randomness and seeds, with code. *Free.*
- Hastie T., Tibshirani R. & Friedman J. *The Elements of Statistical Learning*, 2nd ed., Section 7.10.2 "The wrong and right way to do cross-validation". [Book page (free PDF)](https://hastie.su.domains/ElemStatLearn/) — *Free.*

**Leakage in pair-input and link prediction**
- Park Y. & Marcotte E.M. (2012) *Flaws in evaluation schemes for pair-input computational predictions.* Nature Methods 9:1134. [doi:10.1038/nmeth.2259](https://doi.org/10.1038/nmeth.2259) — why test pairs sharing components with training pairs inflate results. *Paywalled.*
- Pahikkala T. et al. (2015) *Toward more realistic drug–target interaction predictions.* Briefings in Bioinformatics 16:325. [doi:10.1093/bib/bbu010](https://doi.org/10.1093/bib/bbu010) — the S1–S4 settings (warm, new drug, new target, both new). *Free.*
- Guney E. (2017) *Reproducible drug repurposing: when similarity does not suffice.* PSB 2017. [PDF](https://psb.stanford.edu/psb-online/proceedings/psb17/guney.pdf) — similarity-based repurposing collapses under drug-disjoint CV. *Free.*
- Yang Y., Lichtenwalter R.N. & Chawla N.V. (2015) *Evaluating link prediction methods.* Knowledge and Information Systems 45:751. [doi:10.1007/s10115-014-0789-0](https://doi.org/10.1007/s10115-014-0789-0) — sampling, metrics and temporal issues in link-prediction evaluation. *Paywalled (preprint on arXiv).*
- Li J. et al. (2023) *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking.* NeurIPS Datasets & Benchmarks. [arXiv:2306.10453](https://arxiv.org/abs/2306.10453) — under-tuned baselines, inconsistent splits, unrealistically easy negatives. *Free.*

**Model selection and nested CV**
- Varma S. & Simon R. (2006) *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics 7:91. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91) — *Free.*
- Cawley G.C. & Talbot N.L.C. (2010) *On over-fitting in model selection and subsequent selection bias in performance evaluation.* JMLR 11:2079–2107. [JMLR page](https://jmlr.org/papers/v11/cawley10a.html) — *Free.*
- [scikit-learn: nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) — a runnable illustration. *Free.*

**Comparing models statistically**
- Dietterich T.G. (1998) *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation 10:1895–1923. [doi:10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197) — McNemar, resampled $t$, 5×2cv. *Paywalled (author preprint widely available).*
- Nadeau C. & Bengio Y. (2003) *Inference for the generalization error.* Machine Learning 52:239–281. [doi:10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366) — the corrected resampled $t$-test. *Paywalled.*
- Bengio Y. & Grandvalet Y. (2004) *No unbiased estimator of the variance of K-fold cross-validation.* JMLR 5:1089–1105. [JMLR page](https://jmlr.org/papers/v5/grandvalet04a.html) — *Free.*
- Bouckaert R.R. & Frank E. (2004) *Evaluating the replicability of significance tests for comparing learning algorithms.* PAKDD, LNCS 3056. [doi:10.1007/978-3-540-24775-3_3](https://doi.org/10.1007/978-3-540-24775-3_3) — *Paywalled.*
- Demšar J. (2006) *Statistical comparisons of classifiers over multiple data sets.* JMLR 7:1–30. [JMLR page](https://jmlr.org/papers/v7/demsar06a.html) — Wilcoxon, Friedman, Nemenyi, critical-difference diagrams. *Free.*
- Benavoli A. et al. (2017) *Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis.* JMLR 18(77). [JMLR page](https://jmlr.org/papers/v18/16-305.html) — Bayesian correlated $t$-test, ROPE. *Free.*
- Lakens D. (2013) *Calculating and reporting effect sizes to facilitate cumulative science.* Frontiers in Psychology 4:863. [doi:10.3389/fpsyg.2013.00863](https://doi.org/10.3389/fpsyg.2013.00863) — $d_z$ and friends. *Free.*
- Bouthillier X. et al. (2021) *Accounting for variance in machine learning benchmarks.* MLSys. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098) — which sources of variance matter and how many runs you need. *Free.*

**Research practice and reporting**
- Lipton Z.C. & Steinhardt J. (2018) *Troubling trends in machine learning scholarship.* [arXiv:1807.03341](https://arxiv.org/abs/1807.03341) — including failure to identify the sources of empirical gains. *Free.*
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist) — claims, limitations, reproducibility, error bars, licences. *Free.*
- Pineau J. et al. (2021) *Improving reproducibility in machine learning research.* JMLR 22(164). [JMLR page](https://jmlr.org/papers/v22/20-303.html) — the NeurIPS 2019 reproducibility programme and checklist. *Free.*

---

## 11. Glossary

- **Ablation** — removing or replacing one component of a model to measure its contribution.
- **Common random numbers** — using the same splits and seeds for all compared variants so that differences are not noise.
- **Cold start** — evaluating entities (here, diseases) with no known links in training; LODO.
- **Corrected resampled $t$-test** — Nadeau & Bengio's paired $t$-test with variance inflated by $1/J + n_\text{test}/n_\text{train}$.
- **Data leakage** — information unavailable at prediction time entering model building, or a test set unrepresentative of the claim, biasing evaluation.
- **Edge-detector shortcut** — a link predictor that learns to recognize visible input edges instead of predicting hidden ones.
- **Effect size** — the magnitude of a difference (raw mean difference, $d_z$, win rate), as opposed to its $p$-value.
- **Family-wise error rate** — probability of at least one false positive among several tests.
- **GIP kernel** — Gaussian interaction profile kernel, a similarity computed from rows of the association matrix.
- **Hidden-link supervision** — training on links hidden from the model's input in that epoch, so that training pairs resemble test pairs.
- **Holm procedure** — step-down multiple-comparison correction controlling the family-wise error rate.
- **Illegitimate feature** — a feature unavailable at prediction time or derived from the outcome.
- **Learn–predict separation** — the principle that the model-building procedure must not see the targets it is evaluated on.
- **LODO** — leave-one-disease-out cross-validation.
- **Model info sheet** — a written argument that a model is free of each leakage type (Kapoor & Narayanan).
- **Negative control** — a variant that should not help (e.g. a permuted view); any gain it shows reveals an artefact.
- **Negative pool** — the set of cells from which training negatives are sampled.
- **Nested cross-validation** — CV in which hyper-parameters are chosen inside each outer training fold.
- **Non-independence** — test samples related to training samples (same compound, twin drugs, disease subtypes).
- **Paired test** — a test on per-split differences between two methods evaluated on the same splits.
- **Positive control** — a variant that must hurt (e.g. removing all inputs), confirming the evaluation is sensitive.
- **Power** — the probability that a test detects a true effect of a given size.
- **Researcher degrees of freedom** — the many choices made while developing a method, each a potential channel of selection on test data.
- **Selection bias (winner's curse)** — the optimism of the best of several noisy estimates.
- **Temporal leakage** — using information from after the prediction time.
- **Transductive** — all nodes (but not the test edges) are present during training.
- **Validation set** — data used for choices, distinct from the test data used for reporting.
- **Warm start** — test pairs whose drug and disease have other known links in training.
- **Wilcoxon signed-rank test** — non-parametric paired test based on ranks of differences.
