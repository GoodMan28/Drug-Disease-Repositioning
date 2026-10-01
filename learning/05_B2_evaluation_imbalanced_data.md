# Unit B2 — Evaluation and Imbalanced Data

*How to tell whether a drug-repositioning model is actually good, when 99 % of the answers are "no".*

---

## 0. About this chapter

**Prerequisites.**
- **A1 (Python / NumPy / pandas)**: you should be comfortable with boolean masks, `np.argsort`, `np.concatenate`, and reading `.npz` / `.json` files.
- **A3 (Probability and statistics)**: Bernoulli variables, expectation, variance, standard deviation, conditional probability, and the idea of a *base rate*.
- **B1 (Supervised learning basics)**: training / validation / test sets, k-fold cross-validation, over-fitting, hyper-parameters.
- Helpful but not required: A2 (linear algebra) for the code in Section 4.9.

**Estimated study time.** 12–16 hours: about 6 h reading the theory (Sections 1–3), 3 h running and modifying the code (Section 4), 1 h on the project walk-through (Section 5), and 3–5 h on the exercises.

**Learning objectives.** After this unit you will be able to:

1. Build a confusion matrix from scores, labels and a threshold, and compute TPR, FPR, precision, recall, specificity, accuracy and F1 by hand.
2. Derive the formula that links precision to the positive rate, and use it to predict how precision collapses when positives are rare.
3. Construct a ROC curve and a precision–recall (PR) curve by hand from a ranked list, and compute their areas.
4. State and prove the probabilistic meaning of ROC AUC (the probability that a random positive outranks a random negative) and its equivalence to the Mann–Whitney *U* statistic.
5. Explain the difference between *average precision* (AP) and the *trapezoidal* area under the PR curve, and say which one `sklearn` computes.
6. Explain, with numbers, why ROC AUC looks optimistic when positives are about 1 % of the data, and give the score a random ranker would get under both AUC and AUPR.
7. Compute and interpret ranking metrics: precision@k, recall@k, hits@k and mean reciprocal rank (MRR).
8. Describe warm-start k-fold cross-validation and leave-one-disease-out (cold-start) evaluation, and explain which kinds of model each one rewards.
9. Distinguish pooled, per-fold and per-disease averaging, and predict when they disagree.
10. Quantify how the "unknown = negative" assumption biases measured AUC and AUPR.
11. Compare two methods fairly: paired splits, mean ± standard deviation, why fold scores are not independent, and a corrected paired test.
12. Read `src/drepo/evaluation.py` and the project's result files and explain every number they produce.

---

## 1. Motivation: the 99 % problem in this project

Our association matrix $A$ for **Fdataset** has 593 drugs × 313 diseases $= 185{,}609$ cells, and only **1,933** of them are 1 (known indications). That is a positive rate of

$$\pi = \frac{1933}{185609} = 0.0104 \approx 1\%.$$

For **Cdataset** it is $2532 / (663 \times 409) = 2532/271167 = 0.0093$.

Three facts follow immediately, and the rest of this chapter makes each of them precise.

**Fact 1 — accuracy is useless.** A "model" that predicts *no drug treats any disease* is right on $1 - \pi = 98.96\%$ of Fdataset cells. Any metric that rewards that model is not measuring what we care about.

**Fact 2 — two good-looking metrics can tell different stories.** Here are two real rows from our 5-fold cross-validation on Fdataset (file `results/RESULTS.md`):

| Method | AUC | AUPR |
|---|---|---|
| MV-HGAT (ours) | 0.9392 ± 0.0072 | 0.4878 ± 0.0273 |
| SCMFDD | 0.8934 ± 0.0102 | 0.4946 ± 0.0199 |

By AUC, MV-HGAT is clearly better (it wins on all 25 of 25 paired splits, as we will compute in Section 4.11). By AUPR, the two are statistically indistinguishable, and SCMFDD is even slightly ahead on average. Which is "better" depends on *what part of the ranking you care about* — and in drug repositioning, a biologist will only ever look at the top few predictions per disease. Section 3.8 shows that SCMFDD is slightly better at the very top of the pooled list, MV-HGAT is better everywhere below it, and Section 4.10 shows that MV-HGAT is much better *per disease*.

**Fact 3 — a high AUC can coexist with a tiny AUPR.** In the cross-dataset experiment (`results/cross_Fdataset_to_Cdataset.json`) a model trained on Fdataset must find 57 links that only Cdataset knows, hidden among 174,330 candidate pairs. MV-HGAT scores **AUC 0.944** but **AUPR 0.0086**. Is that a failure? The random baseline for AUPR here is $57/174330 = 0.00033$, so 0.0086 is **26× better than random** — real signal, but it means that among the top-ranked candidates, far fewer than 1 in 10 are confirmed. Meanwhile MBiRW, with a *lower* AUC (0.898), has the *highest* AUPR (0.0136, 42× random).

If you cannot explain these three facts to a reviewer, you cannot defend the project's results. By the end of this chapter you will be able to.

---

## 2. Setting and notation

Every method in this project, from the random walk MBiRW to our MV-HGAT, outputs a **score matrix** $S$ of the same shape as $A$: $S_{ij}$ is a real number, larger meaning "drug $i$ is more likely to treat disease $j$". The evaluation code never needs to know how $S$ was produced.

To evaluate, we take a **test set** of cells $\mathcal{T}$ (for example, one fold of cross-validation). For each test cell $t \in \mathcal{T}$ we have

- a **label** $y_t \in \{0, 1\}$ (1 = known indication, called a *positive*; 0 = not known, called a *negative*),
- a **score** $s_t \in \mathbb{R}$.

Let $P = \sum_t y_t$ be the number of positives, $N = |\mathcal{T}| - P$ the number of negatives, and $\pi = P/(P+N)$ the **positive rate** (also called *prevalence* or *base rate*).

Two kinds of evaluation exist:

- **Threshold metrics** need a cut-off $\tau$: predict "positive" iff $s_t \ge \tau$. They answer "if I act on every prediction above $\tau$, how well do I do?"
- **Ranking (threshold-free) metrics** look only at the *order* of the scores. They answer "how good is the sorted list?" ROC AUC, average precision, precision@k and MRR are all ranking metrics.

**Key property.** Ranking metrics are *invariant to any strictly increasing transformation* of the scores. Applying a sigmoid, a logarithm, or multiplying by 7 changes nothing. This is why a model that outputs logits and one that outputs probabilities can be compared directly by AUC, and also why AUC says *nothing* about whether a score of 0.9 really means "90 % likely" (that is *calibration*, a separate property; Section 2.3.3).

---

## 3. Core theory

### 3.1 The confusion matrix

Fix a threshold $\tau$. Every test item falls in exactly one of four boxes:

```
                         actual positive (y=1)     actual negative (y=0)
                       +-------------------------+-------------------------+
 predicted positive    |  TP  true positive      |  FP  false positive     |
   (s >= tau)          |  "a hit"                |  "a false alarm"        |
                       +-------------------------+-------------------------+
 predicted negative    |  FN  false negative     |  TN  true negative      |
   (s <  tau)          |  "a miss"               |  "a correct rejection"  |
                       +-------------------------+-------------------------+
                         column total = P          column total = N
```

Note that $TP + FN = P$ and $FP + TN = N$ — the columns are fixed by the data, the threshold only moves items up or down within each column.

> **Watch out:** scikit-learn's `confusion_matrix(y, pred)` returns `[[TN, FP], [FN, TP]]` — rows are *actual* class, columns are *predicted* class, and class 0 comes first. That is the transpose of the layout above and of many textbooks. Always check.

### 3.2 Rates derived from the confusion matrix

Each rate divides one box by a row or column total. The column rates condition on the **truth**; the row rates condition on the **prediction**.

| Name(s) | Formula | Question it answers |
|---|---|---|
| **True positive rate (TPR)**, recall, sensitivity, hit rate | $\dfrac{TP}{TP+FN} = \dfrac{TP}{P}$ | Of the real indications, what fraction did we catch? |
| **False positive rate (FPR)**, fall-out, $1-$specificity | $\dfrac{FP}{FP+TN} = \dfrac{FP}{N}$ | Of the non-indications, what fraction did we wrongly flag? |
| **Specificity**, true negative rate (TNR) | $\dfrac{TN}{N} = 1 - \text{FPR}$ | Of the non-indications, what fraction did we correctly reject? |
| **Precision**, positive predictive value (PPV) | $\dfrac{TP}{TP+FP}$ | Of the pairs we flagged, what fraction are real? |
| **Negative predictive value (NPV)** | $\dfrac{TN}{TN+FN}$ | Of the pairs we rejected, what fraction are really negative? |
| **Accuracy** | $\dfrac{TP+TN}{P+N}$ | What fraction of all decisions were right? |
| **F1 score** | $\dfrac{2\,\text{prec}\cdot\text{rec}}{\text{prec}+\text{rec}} = \dfrac{2TP}{2TP+FP+FN}$ | Harmonic mean of precision and recall |
| **F$_\beta$ score** | $\dfrac{(1+\beta^2)\,\text{prec}\cdot\text{rec}}{\beta^2\,\text{prec}+\text{rec}}$ | Like F1 but recall counts $\beta$ times as much |

**Why the harmonic mean in F1?** The harmonic mean of two numbers is dominated by the smaller one. If precision is 0.01 and recall is 1.0 (flag everything), the arithmetic mean would be a flattering 0.505, but F1 is $2(0.01)(1)/(1.01) = 0.0198$. F1 punishes lopsided trade-offs.

**Why accuracy fails under imbalance.** Accuracy $= \pi \cdot \text{TPR} + (1-\pi)\cdot\text{TNR}$ (a weighted average of the two column rates, weighted by class sizes). With $\pi = 0.01$, accuracy is 99 % specificity and 1 % sensitivity: it barely notices whether you found any positives at all.

#### Worked example 1 — ten candidate pairs by hand

A model scores ten drug–disease pairs. Sorted by score:

| rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| score $s$ | 0.95 | 0.90 | 0.80 | 0.70 | 0.65 | 0.60 | 0.40 | 0.30 | 0.20 | 0.10 |
| label $y$ | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |

So $P = 4$, $N = 6$, $\pi = 0.4$. Take $\tau = 0.5$: the top six are predicted positive.

- Among the top six, labels are 1,1,0,1,0,0 → $TP = 3$, $FP = 3$.
- Among the bottom four, labels are 1,0,0,0 → $FN = 1$, $TN = 3$.

Then TPR $= 3/4 = 0.75$; FPR $= 3/6 = 0.5$; specificity $= 0.5$; precision $= 3/6 = 0.5$; accuracy $= 6/10 = 0.6$; F1 $= 2(0.5)(0.75)/(1.25) = 0.6$. Section 4.1 reproduces each number with code and with scikit-learn.

### 3.3 Precision depends on the base rate; TPR and FPR do not

This is the single most important formula in the chapter. TPR and FPR are *column* rates: each is computed within one class, so if you duplicate every negative ten times, FPR does not change. Precision mixes the two columns, so it *does* change.

**Derivation.** Write $TP = \text{TPR}\cdot P$ and $FP = \text{FPR}\cdot N$. Divide numerator and denominator by $P + N$ and use $P/(P+N) = \pi$:

$$
\text{precision} = \frac{TP}{TP + FP} = \frac{\text{TPR}\cdot P}{\text{TPR}\cdot P + \text{FPR}\cdot N}
= \frac{\pi\,\text{TPR}}{\pi\,\text{TPR} + (1-\pi)\,\text{FPR}}.
$$

This is just Bayes' rule: precision $= \Pr(y=1 \mid \text{flagged})$, while TPR $= \Pr(\text{flagged}\mid y=1)$ and FPR $= \Pr(\text{flagged}\mid y=0)$.

#### Worked example 2 — what "FPR = 1 %" means on Fdataset

Suppose a classifier achieves TPR $= 0.60$ at FPR $= 0.01$. On a balanced dataset ($\pi = 0.5$):

$$\text{precision} = \frac{0.5 \times 0.6}{0.5\times0.6 + 0.5\times0.01} = \frac{0.30}{0.305} = 0.984.$$

On Fdataset ($\pi = 0.0104$):

$$\text{precision} = \frac{0.0104 \times 0.6}{0.0104\times0.6 + 0.9896\times0.01} = \frac{0.00624}{0.00624 + 0.009896} = 0.387.$$

Same classifier, same TPR/FPR, but precision drops from 98 % to 39 %. In counts: there are $N = 185{,}609 - 1{,}933 = 183{,}676$ negatives, so FPR = 1 % means about **1,837 false alarms**, against $0.6 \times 1933 \approx 1{,}160$ hits.

This is not hypothetical. On the real pooled cross-validation predictions of MV-HGAT, the operating point at FPR = 1.00 % has TPR = 0.594, i.e. 1,148 true positives and 1,831 false positives, precision **0.385** (computed in Section 4.10). The formula predicts it almost exactly.

### 3.4 Thresholds, scores, and calibration

#### 3.4.1 A threshold is a decision, not a property of the model

Moving $\tau$ trades one error for the other: lowering $\tau$ flags more pairs, so TP and FP both grow (TPR and FPR both rise, precision usually falls). There is no single "right" threshold; it depends on costs. In repositioning, the "cost" of a false positive is a wasted wet-lab experiment or literature search; the cost of a false negative is a missed drug. Because experiments are expensive and the list is long, practitioners look at the *top* of the ranking, which is why ranking metrics dominate this field.

#### 3.4.2 Why we prefer threshold-free metrics here

Different methods produce scores on different scales (MBiRW produces random-walk probabilities, DRRS produces matrix-completion values that can be negative, MV-HGAT produces sigmoid outputs). A fixed threshold like 0.5 is meaningless across them. Ranking metrics avoid the issue entirely.

#### 3.4.3 Calibration (in one paragraph)

A model is *calibrated* if, among pairs it scores 0.8, about 80 % are positive. AUC and AP are blind to calibration: multiply all scores by 0.1 and nothing changes. Calibration matters if someone will read a score as a probability. MV-HGAT's sigmoid outputs are *not* calibrated probabilities: it is trained with 2 sampled negatives per positive (Section 4.4 of `HOW_IT_WORKS.md`), so its scores reflect a 1:2 world, not the true 1:95 world. Never report "the model is 99 % sure amantadine treats Alzheimer's" from a score of 0.99. Calibration also has a subtle effect on *pooled* metrics (Section 3.12).

### 3.5 The ROC curve

**Definition.** The receiver operating characteristic (ROC) curve plots TPR (y-axis) against FPR (x-axis) for every possible threshold. (The name comes from World War II radar engineering — "receiver operating characteristic" — and was later adopted by psychology, medicine and machine learning; Fawcett 2006 is the classic tutorial.)

**Construction by hand.** Sort the items by decreasing score. Start at $(0,0)$: threshold above every score, nothing flagged. Lower the threshold past one item at a time:

- if the item is a positive, move **up** by $1/P$;
- if it is a negative, move **right** by $1/N$.

After all items you reach $(1,1)$: everything flagged. If several items share the same score, they cross the threshold together, and you move diagonally by $(\#\text{neg}/N, \#\text{pos}/P)$.

For Worked Example 1 ($P = 4$, $N = 6$), the label sequence 1,1,0,1,0,0,1,0,0,0 gives the path

```
TPR
1.00 |                         *----*----*----*
     |                         |
0.75 |          *----*----*----+
     |          |
0.50 |*----*----+
     ||
0.25 |*
     ||
0.00 *-----------------------------------------> FPR
     0   1/6  2/6  3/6  4/6  5/6  1
     (moves: up, up, right, up, right, right, up, right, right, right)
```

**Properties worth remembering.**

1. *Diagonal = random.* A ranker that ignores the data flags positives and negatives at the same rate, so TPR ≈ FPR at every threshold.
2. *Top-left corner = perfect.* All positives ranked above all negatives: the curve goes straight up to (0,1), then right.
3. *Below the diagonal = worse than random*, but flipping the scores would make it better than random.
4. *Class-ratio invariance.* Because both axes are column rates, duplicating negatives does not change the curve. This is a strength (comparable across datasets) and, under heavy imbalance, a weakness (Section 3.8).
5. *Monotone-transform invariance.* Only the order of scores matters.

### 3.6 Area under the ROC curve (AUC) and its probabilistic meaning

**Definition.** AUC (also AUROC, ROC AUC) is the area under the ROC curve, between 0 and 1. With the step construction above, it can be computed exactly by the trapezoidal rule.

**Theorem (probabilistic meaning).** Draw one positive $X^+$ and one negative $X^-$ uniformly at random from the test set, independently. Then

$$
\text{AUC} = \Pr\!\left(s(X^+) > s(X^-)\right) + \tfrac{1}{2}\Pr\!\left(s(X^+) = s(X^-)\right)
= \frac{1}{PN}\sum_{a=1}^{P}\sum_{b=1}^{N}\left[\mathbb{1}(s^+_a > s^-_b) + \tfrac{1}{2}\mathbb{1}(s^+_a = s^-_b)\right].
$$

In words: **AUC is the fraction of (positive, negative) pairs that the model orders correctly**, with ties counting as half.

**Proof (no ties, then ties).** Walk the ROC path as in Section 3.5. Each rightward step (a negative) has width $1/N$. The area of the vertical strip under that step is (width) × (height), and the height is the current TPR, i.e. the fraction of positives *already passed*, which are exactly the positives scored **above** this negative. So

$$
\text{AUC} = \sum_{b=1}^{N} \frac{1}{N} \cdot \frac{\#\{a: s^+_a > s^-_b\}}{P} = \frac{1}{PN}\sum_{b}\#\{a: s^+_a > s^-_b\},
$$

the fraction of correctly ordered pairs. With ties, a group of $p$ positives and $n$ negatives sharing one score produces a diagonal segment; the trapezoid under it contributes $\frac{1}{N}\cdot n \cdot$ (height before) $+ \frac{1}{2}\cdot\frac{n}{N}\cdot\frac{p}{P}$, and the extra triangle is exactly "half credit for the $p \times n$ tied pairs". $\blacksquare$

**Connection to the Mann–Whitney U test.** The Mann–Whitney (Wilcoxon rank-sum) statistic of sample 1 against sample 2 is defined as $U = \sum_{a,b}[\mathbb{1}(x_a > z_b) + \frac12\mathbb{1}(x_a = z_b)]$. So

$$\boxed{\text{AUC} = \frac{U}{P\,N}}$$

with $U$ computed for the positives against the negatives. This was pointed out for medical diagnosis by Hanley & McNeil (1982). It has two practical consequences:

- **A fast formula.** Rank all $P+N$ scores from lowest (rank 1) to highest, averaging ranks for ties. Let $R^+$ be the sum of the positives' ranks. Then $U = R^+ - P(P+1)/2$, so $\text{AUC} = \dfrac{R^+ - P(P+1)/2}{PN}$. That is an $O(n\log n)$ computation, which is what libraries do.
- **Statistics for free.** Everything known about the U statistic (its variance, confidence intervals, tests) applies to AUC.

**Worked example 1, continued.** For each positive, count the negatives ranked below it. The positives are at ranks 1, 2, 4, 7; the negatives are at ranks 3, 5, 6, 8, 9, 10.

- rank 1 positive: 6 negatives below;
- rank 2 positive: 6 negatives below;
- rank 4 positive: negatives at 5, 6, 8, 9, 10 → 5;
- rank 7 positive: negatives at 8, 9, 10 → 3.

Total correctly ordered pairs $= 6+6+5+3 = 20$ out of $P N = 24$, so **AUC = 20/24 = 0.8333**.

With the rank-sum formula: ranking from the *bottom*, the positives have ascending ranks $10, 9, 7, 4$, so $R^+ = 30$ and $U = 30 - 4\cdot5/2 = 20$. Same answer.

**How to interpret AUC values.** 0.5 = random; 1.0 = perfect. An AUC of 0.94 means "pick a random true indication and a random unknown pair: 94 % of the time the indication scores higher". Notice what the statement does *not* say: it says nothing about the top of the list specifically. A positive moved from rank 50,000 to rank 40,000 raises AUC exactly as much as one moved from rank 10,001 to rank 1 — both pass 10,000 negatives.

**Variance and confidence intervals (for reference).** Hanley & McNeil (1982) give an approximate standard error,

$$
\text{SE}(\text{AUC}) \approx \sqrt{\frac{A(1-A) + (P-1)(Q_1 - A^2) + (N-1)(Q_2-A^2)}{PN}},\qquad Q_1 = \frac{A}{2-A},\; Q_2 = \frac{2A^2}{1+A},
$$

where $A$ is the AUC. For our test folds ($P \approx 387$, $N \approx 36{,}735$, $A = 0.94$) this gives about 0.008 — the same order as the fold-to-fold standard deviation we observe (0.007). In practice, with cross-validation we use the spread across folds instead (Section 3.14).

**Partial AUC.** Sometimes only the low-FPR region matters (we will only test the top few predictions). The *partial AUC* integrates the ROC curve only over $\text{FPR} \in [0, f_0]$, e.g. $f_0 = 0.01$; scikit-learn offers a standardised version via `roc_auc_score(y, s, max_fpr=0.01)`. It is one fix for ROC's optimism; the PR curve is another.

### 3.7 The precision–recall curve, average precision, and the trapezoid trap

#### 3.7.1 Construction

The PR curve plots precision (y-axis) against recall = TPR (x-axis) as the threshold is lowered. Walking down the ranked list, after the top $k$ items:

$$\text{recall}(k) = \frac{TP(k)}{P},\qquad \text{precision}(k) = \frac{TP(k)}{k}.$$

Recall never decreases. Precision jumps **up** each time a positive is passed and decays **down** while negatives are passed, giving the characteristic *saw-tooth* shape. The curve starts near (0, 1) if the top item is a positive and ends at $(1, \pi)$, because when everything is flagged, precision equals the positive rate.

For Worked Example 1:

| top $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| item is positive? | ✔ | ✔ | | ✔ | | | ✔ | | | |
| precision | 1.000 | 1.000 | 0.667 | **0.750** | 0.600 | 0.500 | **0.571** | 0.500 | 0.444 | 0.400 |
| recall | 0.25 | 0.50 | 0.50 | 0.75 | 0.75 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |

**The random baseline of the PR curve is a horizontal line at $\pi$.** A random ranker has expected precision $\pi$ at every depth. So unlike ROC (baseline area 0.5 always), the PR baseline *moves with the data*: 0.4 in the toy example, 0.0104 on Fdataset, 0.00033 in the cross-dataset test.

#### 3.7.2 Average precision (AP)

**Definition.** Average precision is the mean of the precision values measured at the rank of each positive:

$$
\text{AP} = \frac{1}{P}\sum_{a=1}^{P} \text{precision}(r_a) = \sum_{k=1}^{n} \left(\text{recall}(k) - \text{recall}(k-1)\right)\cdot \text{precision}(k),
$$

where $r_a$ is the rank of the $a$-th positive. The second form shows that AP is a **right-endpoint (step-function) Riemann sum** of the PR curve: each time recall increases by $1/P$, we multiply by the precision at that point. No interpolation between points is used.

**Worked example 1.** Positives at ranks 1, 2, 4, 7 have precision 1, 1, 3/4, 4/7:

$$\text{AP} = \frac{1 + 1 + 0.75 + 0.5714}{4} = \frac{3.3214}{4} = 0.8304.$$

This is exactly what `sklearn.metrics.average_precision_score` returns. In this project, `evaluation.py::metrics` defines "AUPR" as `average_precision_score`, so **every AUPR number in `results/` is AP**.

**Interpretation.** AP rewards putting positives early, and it rewards the first ones most: in the toy example, the positive at rank 1 contributes precision 1.0, the one at rank 7 contributes only 0.571. A model that places half the positives at the very top and the rest anywhere can have a large AP (Section 4.6).

#### 3.7.3 The trapezoidal AUPR and why it can mislead

A tempting alternative is to compute the area under the PR points with the trapezoidal rule, `sklearn.metrics.auc(recall, precision)`. In the toy example this gives **0.8110**, not 0.8304 (Section 4.3). Why the difference?

- The trapezoidal rule *linearly interpolates* between consecutive PR points. Davis & Goadrich (2006) proved that the achievable curve between two PR points is **not** a straight line: interpolating between two operating points in ROC space (which *is* linear) maps to a *curved* path in PR space, because precision is a nonlinear function of TP and FP. Linear interpolation in PR space therefore over- or under-states performance, and tends to be **optimistic** when a big jump in recall happens between a high-precision point and a lower one.
- sklearn's `precision_recall_curve` also adds an artificial starting point (recall 0, precision 1), and the trapezoid area depends on such conventions.

When the curve has many points (thousands of positives), AP and the trapezoid area converge: on our Fdataset predictions they are 0.4917 vs 0.4916 (Section 4.10). With few positives — for example **per-disease** curves, where many diseases have 1–5 known drugs — they can differ substantially. **Rule: report AP (= `average_precision_score`) and call it AP or AUPR consistently. Never mix AP from one paper with trapezoidal AUPR from another.**

Other variants exist: the PASCAL VOC "11-point interpolated AP" takes, at recall levels 0, 0.1, …, 1, the maximum precision at any recall ≥ that level. It is usually larger than AP because it uses the best precision achievable at or beyond each recall level. You will meet it in object-detection papers, not in ours.

#### 3.7.4 The ROC–PR correspondence

Davis & Goadrich (2006) also proved: for a fixed dataset, each ROC point corresponds to exactly one PR point, and **one curve dominates another in ROC space if and only if it dominates in PR space**. So if MV-HGAT's ROC curve were above SCMFDD's *everywhere*, it would also be above in PR space. AUC and AP can disagree only when **curves cross** — and they do cross, as the next section shows with real data.

### 3.8 Why ROC looks optimistic under heavy imbalance

Put Sections 3.3 and 3.6 together. ROC's x-axis is FPR $= FP/N$. When $N$ is huge, a visually tiny FPR is a large *number* of false positives:

| FPR | false positives on Fdataset ($N = 183{,}676$) |
|---|---|
| 0.1 % | 184 |
| 1 % | 1,837 |
| 5 % | 9,184 |
| 10 % | 18,368 |

Compare with only $P = 1{,}933$ positives in total. The whole "interesting" region for a biologist — the first few hundred predictions — is squeezed into the leftmost 0.1 % of the ROC plot, a sliver you cannot even see. AUC integrates over FPR from 0 to 1, so it is dominated by how the model orders the *bulk* of the list, which nobody will ever read.

Here are the real pooled operating points of the two methods from Fact 2 (computed from `results/Fdataset/cv5/*_preds.npz`, Section 4.10):

| FPR | MV-HGAT TPR | MV-HGAT precision | SCMFDD TPR | SCMFDD precision |
|---|---|---|---|---|
| 0.10 % | 0.289 | 0.756 | **0.324** | **0.774** |
| 1 % | **0.594** | **0.385** | 0.562 | 0.375 |
| 5 % | **0.788** | **0.144** | 0.706 | 0.130 |
| 10 % | **0.857** | **0.083** | 0.785 | 0.077 |

The curves **cross**: SCMFDD is better at the extreme top of the pooled list (its first 100 predictions are 100 % correct, versus 98 % for MV-HGAT); MV-HGAT is better everywhere after that. AUC weighs the long stretch from FPR 1 % to 100 % where MV-HGAT is far ahead, so MV-HGAT wins clearly. AP weighs the high-precision start heavily, where SCMFDD is slightly ahead, so they tie.

**A controlled demonstration.** Keep the score distributions fixed — positives $\sim \mathcal{N}(1.5, 1)$, negatives $\sim \mathcal{N}(0,1)$ — and only change how many negatives there are. The theoretical AUC is $\Phi(1.5/\sqrt2) = 0.856$ regardless (a standard result for this *binormal* model, since $s^+ - s^- \sim \mathcal{N}(1.5, 2)$). Section 4.4 simulates it:

| positive rate | AUC | AP | precision at FPR = 5 % |
|---|---|---|---|
| 50 % | 0.851 | 0.847 | 0.894 |
| 10 % | 0.861 | 0.490 | 0.503 |
| 1 % | 0.858 | 0.118 | 0.083 |
| 0.1 % | 0.858 | 0.017 | 0.009 |

**AUC is flat; AP collapses.** The model has not changed; only the haystack grew. This is the core argument of Saito & Rehmsmeier (2015): *on imbalanced data, the PR plot tells you what the ROC plot hides*.

**A balanced view.** "AUPR is always better" is too strong. McDermott et al. (2024) show that AUPRC preferentially rewards improvements to high-scoring items and to high-prevalence subgroups; AUROC weighs all mis-orderings equally and is comparable across datasets with different prevalences. The right question is *what will the scores be used for?* In repositioning, they are used to choose a short list of candidates per disease → precision at the top matters → AP and precision@k are the headline numbers, AUC is a useful secondary number, and both are always reported. That is exactly what this project does.

### 3.9 Random baselines

You cannot judge a score without its baseline.

**ROC AUC of a random ranker is 0.5.** If scores are independent of labels (and continuous, so no ties), then by symmetry $\Pr(s^+ > s^-) = \tfrac12$. This holds for *any* positive rate.

**AP of a random ranker is approximately the positive rate $\pi$.** Intuition: at every depth, expected precision is $\pi$. Exact derivation: a positive at rank $r$ has, among the $r-1$ items above it, a hypergeometric number of other positives with mean $(r-1)(P-1)/(N_{\text{tot}}-1)$, where $N_{\text{tot}} = P+N$. Under a random permutation each positive is equally likely to be at any rank $r = 1,\dots,N_{\text{tot}}$, so

$$
\mathbb{E}[\text{AP}] = \frac{1}{N_{\text{tot}}}\sum_{r=1}^{N_{\text{tot}}} \frac{1 + (r-1)\frac{P-1}{N_{\text{tot}}-1}}{r}
= \frac{P-1}{N_{\text{tot}}-1} + \frac{H_{N_{\text{tot}}}}{N_{\text{tot}}}\cdot\frac{N_{\text{tot}}-P}{N_{\text{tot}}-1},
$$

where $H_n = 1 + \tfrac12 + \dots + \tfrac1n \approx \ln n + 0.577$ is the harmonic number. The second term is tiny for large $N_{\text{tot}}$, so $\mathbb{E}[\text{AP}] \approx \pi$. For Fdataset: $\pi = 0.01041$ and the exact expectation is $0.01048$; a simulation of 20 random rankers gives $0.01045 \pm 0.00019$ (Section 4.5).

**Baselines in this project.**

| Setting | positives / candidates | random AUC | positive rate π | exact random E[AP] |
|---|---|---|---|---|
| Fdataset, 5-fold CV (pooled) | 1,933 / 185,609 | 0.5 | 0.0104 | 0.0105 |
| Cdataset, 5-fold CV (pooled) | 2,532 / 271,167 | 0.5 | 0.0093 | 0.0094 |
| Cross-dataset F → C | 57 / 174,330 | 0.5 | 0.00033 | 0.00039 |
| One disease with 5 drugs (LODO, Fdataset) | 5 / 593 | 0.5 | 0.0084 | **0.0184** |

The last row is a warning: when a query has only a handful of positives, the $H_N/N$ term is no longer negligible and the random AP is about **twice** the positive rate (Exercise 7 works out a 3-drug example). Per-disease AP must be compared with its own baseline.

**Lift.** It is often clearer to report AP *relative* to the baseline: MV-HGAT's 0.488 on Fdataset is $0.488/0.0104 \approx 47\times$ random; its cross-dataset 0.0086 is 26× random. An AUC "improvement from 0.88 to 0.94" sounds small; an AP improvement from 30× to 47× random tells the reader much more.

> **Comparing AP across datasets is dangerous.** Because the baseline moves, Cdataset's AP of 0.586 is not directly comparable to Fdataset's 0.488. Compare *methods on the same test set*, or compare lifts.

### 3.10 Ranking metrics: precision@k, recall@k, hits@k, MRR

ROC and PR summarise the *whole* list. A user of a repositioning tool asks a narrower question: *"For my disease, are the top 10 suggestions any good?"* Ranking metrics answer it directly. They are computed **per query** — here, per disease (rank all drugs for disease $j$) — and then averaged over queries.

For one disease with ranked drug list and $P_j$ known (held-out) drugs:

- **Precision@k** $= \dfrac{\#\text{true drugs in the top } k}{k}$.
- **Recall@k** $= \dfrac{\#\text{true drugs in the top } k}{P_j}$. Bounded above by $k/P_j$, so with $k=10$ and $P_j=25$ the best possible recall@10 is 0.4.
- **Hits@k** $= 1$ if at least one true drug is in the top $k$, else 0 (sometimes called *success@k*). Averaged over diseases: "for what fraction of diseases does the top-$k$ contain something correct?"
- **Reciprocal rank (RR)** $= 1/(\text{rank of the first true drug})$; **mean reciprocal rank (MRR)** is its average over diseases. RR is 1 if the top drug is correct, 0.5 if the first hit is second, 0.1 if tenth.
- **Per-disease AP**, averaged over diseases, is called *mean average precision* (MAP) in information retrieval.
- **nDCG@k** (normalised discounted cumulative gain) weights a hit at rank $r$ by $1/\log_2(r+1)$ and normalises by the best possible value. It is common in recommender systems; we mention it for completeness.

#### Worked example 3 — three diseases, eight drugs, $k = 3$

| | first true drug at rank | true drugs | in top 3 | P@3 | R@3 | Hits@3 | RR | AP |
|---|---|---|---|---|---|---|---|---|
| disease 0 | 2 | ranks 2, 4 | 1 | 1/3 | 1/2 | 1 | 1/2 | (1/2 + 2/4)/2 = 0.500 |
| disease 1 | 1 | ranks 1, 4 | 1 | 1/3 | 1/2 | 1 | 1 | (1/1 + 2/4)/2 = 0.750 |
| disease 2 | 7 | rank 7 | 0 | 0 | 0 | 0 | 1/7 | 1/7 = 0.143 |
| **mean** | | | | 0.222 | 0.333 | 0.667 | **MRR 0.548** | **MAP 0.464** |

Section 4.7 computes the same table in code.

**How ranking metrics relate to the case studies.** `scripts/06_case_study.py` produces, for each of three diseases, the top 10 drugs that are *not already known*, and counts how many have independent support: prostate cancer 10/10, breast cancer 6/10, Alzheimer's 5/10. Those are *precision@10 against an external, noisy reference* — the most practically meaningful number in the project, measured on only three diseases. (Section 5.6 shows that one of the Alzheimer "supports" does not survive inspection.)

### 3.11 Evaluation protocols: warm start versus cold start

How the test set is formed matters at least as much as which metric is used. Park & Marcotte (2012) showed for protein–protein interaction prediction that test pairs whose *both* members appear in training are far easier than pairs with one or two unseen members, and that mixing them inflates results; Pahikkala et al. (2015) made the same point for drug–target prediction. The same logic applies to drug–disease pairs.

#### 3.11.1 Warm-start k-fold cross-validation over cells (our `cv5`, `cv10`)

```
   A (drugs x diseases)               fold f = 1/k of the 1s  +  1/k of the 0s
 +--------------------------+
 | . . 1 . . . . 1 . . . .  |      train: A with fold-f 1s set to 0
 | . 1 . . . . . . . 1 . .  |      negatives for training drawn only
 | . . . . 1 . . . . . . .  |        from the OTHER folds' 0s
 | 1 . . . . . 1 . . . . .  |      test:  fold-f 1s (positives)
 +--------------------------+             + fold-f 0s (negatives)
```

Every cell is assigned to exactly one fold, so over $k$ folds every pair is tested exactly once. The 1s and 0s are split *separately* (stratification), so each fold has the same positive rate. Crucially, a test pair $(i,j)$ usually has drug $i$ linked to *other* diseases and disease $j$ linked to *other* drugs in training: both ends are **warm**. That is why collaborative signals ("drugs that treat similar diseases", "diseases treated by similar drugs") work well here.

#### 3.11.2 Leave-one-disease-out (LODO; our `lodo`)

For each disease $j$, hide **its whole column**: all its known drugs. Train, score all drugs for $j$, and record the column. Disease $j$ looks brand-new to the model — a **cold-start** disease. Any information about $j$ must come from *side information*: its similarity to other diseases (phenotype, ontology, genes).

**Why LODO is harder — a precise argument.** Consider a purely drug-side guilt-by-association scorer: $\hat{S}_{ij} = \sum_{i'} W_{ii'} A^{\text{train}}_{i'j}$ ("do drugs similar to $i$ treat $j$?"). Under LODO, column $j$ of $A^{\text{train}}$ is all zeros, so $\hat{S}_{\cdot j} = 0$ for every drug: the scorer can only output ties, and AUC is exactly 0.5. In warm CV, 80 % of column $j$ is still visible, and the same scorer works. Section 4.9 runs this on real Fdataset data:

| scorer | warm 5-fold CV AUC / AUPR | LODO AUC / AUPR |
|---|---|---|
| drug side (ECFP chemistry, 10 neighbours) | 0.735 / 0.214 | **0.500 / 0.010** (exactly random) |
| disease side (MimMiner phenotype, 10 neighbours) | 0.714 / 0.158 | 0.738 / 0.174 |

Two lessons:

1. LODO destroys every signal that flows *through the held-out disease's own links* — drug-side similarity, matrix factorisation's disease factor, the GNN's message passing along `assoc` edges, and the degree of the disease.
2. LODO is not harder for *everything*: a disease-side scorer does not use column $j$ at all, so it is unaffected (it even does slightly better than in warm CV, because in LODO all *other* columns are fully visible, whereas warm CV hides 20 % of links everywhere). This is why MBiRW's disease-side walk is a strong cold-start baseline, and why MV-HGAT needed explicit "cold-start practice" and a degree gate (Section 4.4 of `HOW_IT_WORKS.md`).

**Other cold-start variants.** Leave-one-*drug*-out (a new drug: only chemical/gene similarity can help); leave-pair-out with both ends unseen (hardest); and *temporal* splits (train on indications approved before year Y, test on later ones), which are the most realistic but need dated labels.

**Pooling in LODO.** Each disease's column yields 593 scores. `run_lodo` concatenates all columns and computes a single pooled AUC/AP, and *also* the mean per-disease AUC. Our project runs LODO on a seeded random subset of 100 diseases to save time; the paper should state that.

### 3.12 Pooled, per-fold and per-disease averaging

Given test predictions from several folds or diseases, there are three common ways to summarise:

1. **Per-fold (macro over folds):** compute AUC/AP on each fold, then report mean ± std over folds. This is what `summarise()` does for CV.
2. **Pooled (micro):** concatenate all folds' $(y, s)$ and compute *one* AUC/AP. This is what the ROC/PR **figures** use, and what LODO's headline number uses.
3. **Per-query (macro over diseases):** for each disease, rank all drugs, compute the metric, and average over diseases (only diseases with at least one positive). This matches how a user consumes the predictions.

They answer different questions and can disagree sharply.

**Pooled vs per-fold: the calibration trap.** A pooled metric compares scores produced by *different models* (one per fold). If fold 2's model happens to output systematically lower numbers, its positives get buried under fold 1's negatives even though each model ranks its own fold perfectly. Toy example (Section 4.12): two folds, each with per-fold AUC 1.000; pooled AUC **0.781**. In our project the effect is small but visible. Comparing the pooled curves (repeat 0) with the mean of the same five folds: SCMFDD's AP is 0.4879 pooled vs 0.4880 per fold (identical), while MV-HGAT's is 0.4917 pooled vs 0.5004 per fold. MV-HGAT's five fold-models are neural networks trained from different random initialisations, so their score scales differ slightly, and pooling costs them about 0.009 AP. A deterministic method like SCMFDD does not pay this price.

**Pooled vs per-disease: popularity.** Pooled metrics let a method win by **ranking whole diseases** — e.g. putting every drug for well-studied cancers above every drug for a rare syndrome. That inflates pooled AP without helping any individual disease's list. Per-disease metrics remove this. On real Fdataset CV predictions (Section 4.10):

| | pooled AUC | pooled AP | mean per-disease AUC | mean per-disease AP | MRR | Hits@10 |
|---|---|---|---|---|---|---|
| MV-HGAT | 0.942 | 0.492 | **0.882** | **0.360** | **0.495** | **0.668** |
| SCMFDD | 0.894 | 0.488 | 0.795 | 0.278 | 0.390 | 0.546 |

Pooled AP says "tie". Per disease, MV-HGAT puts a correct drug in the top 10 for 67 % of diseases versus 55 % for SCMFDD, and its first correct drug is on average much higher. **For a repositioning paper, per-disease ranking metrics are the most faithful to the use case** and are worth adding to the results table.

(Technical note: in that analysis each cell's score comes from the fold in which it was tested, so one disease's ranking mixes scores from 5 fold-models. That is standard but slightly favours well-calibrated methods; a purist alternative is per-disease metrics within each fold, averaged.)

### 3.13 The "unknown = negative" assumption

Every 0 in $A$ means "**not known**", not "**does not treat**". Some 0s are real indications that were approved after the benchmark was built (Fdataset dates from 2011), or used off-label, or simply never curated. Treating all 0s as negatives is the universal convention (PREDICT, MBiRW, DRRS, LAGCN…), but you must understand its effect. In machine-learning language, this is **positive–unlabelled (PU) learning** (Elkan & Noto 2008; Bekker & Davis 2020): we observe some positives; the rest of the data is a mixture of negatives and unobserved positives.

**Effect on measured AUC.** Let a fraction $q$ of the items labelled 0 actually be *hidden positives*, and assume (strongly) that hidden positives get scores distributed like the known positives. A labelled pair (known positive, labelled negative) is then either (positive, true negative), correctly ordered with probability $\text{AUC}_\text{true}$, or (positive, hidden positive), "correctly" ordered with probability 1/2. So

$$\text{AUC}_\text{measured} = (1-q)\,\text{AUC}_\text{true} + q\cdot\tfrac12 .$$

Because $q$ is small (if 1,000 indications are missing from Fdataset, $q = 1000/183676 = 0.5\%$), **AUC is barely affected**.

**Effect on measured precision and AP.** Every hidden positive the model ranks highly is counted as a false positive. If the model is good, hidden positives sit near the top — exactly where AP and precision@k look. So **AP and precision@k are biased downward, sometimes strongly.** Section 4.8 simulates a model with true AUC 0.919 and AP 0.369; hiding 25 % of the positives drops measured AP to 0.275 and P@100 from 0.85 to 0.64, while AUC moves by only 0.004.

**Consequences for this project.**

1. Absolute AP numbers are **pessimistic**: the true precision of the top predictions is higher than measured. The case studies are the evidence: many "false positives" at the top turn out to have literature or trial support.
2. **Comparisons can be distorted**, not just shifted. The bias hits methods that are good at finding *undiscovered* links hardest. If the missing indications are not random — e.g. they are concentrated in well-studied drugs (missing-not-at-random) — a method with a popularity bias may be rewarded or penalised unfairly.
3. Training is affected too: sampled "negatives" sometimes are positives, teaching the model to push down real indications. Negative sampling from a large pool keeps this rare.
4. The honest wording for a paper: "unknown pairs are treated as negatives; reported precision-type metrics are therefore conservative estimates".

### 3.14 Comparing methods fairly: mean ± std and beyond

#### 3.14.1 Same splits, same information

A fair comparison holds everything fixed except the method: identical folds (our `kfold_splits` depends only on the seed, so all methods see the same 25 splits), identical training data and side information (the project's ablation "benchmark similarities only" exists precisely to separate the *architecture* from the *extra data*), and hyper-parameters tuned on a **validation** split for every method, never on the test folds (tuning on test folds leaks information and inflates results; Varma & Simon 2006 quantify this for cross-validation).

#### 3.14.2 What "mean ± std" means — and does not mean

`summarise()` reports the mean and the standard deviation of the 25 fold scores (5 folds × 5 repeats). Three subtleties:

1. **Population vs sample std.** NumPy's `np.std` defaults to `ddof=0` (divide by $n$); the unbiased sample estimate uses $n-1$. For $n=25$ the difference is a factor $\sqrt{25/24} = 1.02$ (0.0072 vs 0.0073 for MV-HGAT's AUC). Say which one you report.
2. **Std is not standard error.** The std describes how much one fold's score varies. The uncertainty of the *mean* is smaller — naively $\text{std}/\sqrt{n}$ — but…
3. **Fold scores are not independent.** Any two training sets in 5-fold CV share 75 % of their data, and repeats re-use the same data. Bengio & Grandvalet (2004) proved there is *no unbiased estimator* of the variance of k-fold CV. Naive $t$-tests on fold scores are therefore over-confident (Dietterich 1998).

#### 3.14.3 Paired comparison and the corrected resampled t-test

Because all methods see the same splits, compare them **pairwise per split**: $d_r = \text{metric}_A(r) - \text{metric}_B(r)$. Pairing removes the variation due to "easy" or "hard" folds, which both methods share.

Nadeau & Bengio (2003) proposed inflating the variance to account for overlapping training sets:

$$
t = \frac{\bar d}{\sqrt{\left(\dfrac{1}{J} + \dfrac{n_\text{test}}{n_\text{train}}\right)\hat\sigma^2_d}},\qquad \text{df} = J - 1,
$$

where $J$ is the number of splits (25), $\hat\sigma^2_d$ the sample variance of the differences, and $n_\text{test}/n_\text{train} = 1/4$ for 5-fold CV. Compared with the naive $\bar d/\sqrt{\hat\sigma_d^2/J}$, the standard error grows by a factor $\sqrt{(1/25 + 1/4)/(1/25)} = \sqrt{7.25} \approx 2.7$.

**Applied to our results (Section 4.11):**

| | mean paired difference (MV-HGAT − SCMFDD) | MV-HGAT wins | naive $t$ ($p$) | corrected $t$ ($p$) |
|---|---|---|---|---|
| AUC | +0.0458 ± 0.0085 | 25 / 25 | 26.9 ($2\times10^{-19}$) | 10.0 ($5\times10^{-10}$) |
| AUPR | −0.0068 ± 0.0329 | 8 / 25 | −1.04 (0.31) | −0.39 (0.70) |

Conclusion you can defend: MV-HGAT has a clearly higher AUC; on pooled-per-fold AUPR the two are **not distinguishable**. (And per-disease ranking metrics favour MV-HGAT, Section 3.12.)

Other tools: the **Wilcoxon signed-rank test** on paired differences (no normality assumption, but it ignores fold dependence too), **5×2 cross-validation** tests (Dietterich 1998), and, for many methods over many datasets, Friedman tests with post-hoc analysis (Demšar 2006). With six methods compared on two datasets, beware multiple comparisons: some "significant" differences will appear by chance.

#### 3.14.4 What to put in the paper

- Mean ± std over all folds of all repeats, stating $k$, number of repeats, and ddof.
- Paired win counts or a corrected paired test for the key comparisons.
- Both AUC and AP, plus the random baseline for AP.
- At least one per-disease ranking metric (MRR or Hits@10).
- The warm/cold protocol, and LODO's disease subset if used.
- The "unknown = negative" caveat.

---

## 4. Code: every idea, computed by hand and with scikit-learn

All code below was run with the project's virtual environment
(`"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe"`, CPU only), and the output shown is the real output. Blocks 4.1–4.8 and 4.12 are self-contained; blocks 4.9–4.11 read the project's data and result files (read-only). Save any block as a `.py` file and run it.

### 4.1 Confusion matrix and threshold metrics (Worked Example 1)

```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

scores = np.array([0.95, 0.90, 0.80, 0.70, 0.65, 0.60, 0.40, 0.30, 0.20, 0.10])
labels = np.array([1,    1,    0,    1,    0,    0,    1,    0,    0,    0])
t = 0.5
pred = (scores >= t).astype(int)

TP = int(((pred == 1) & (labels == 1)).sum())
FP = int(((pred == 1) & (labels == 0)).sum())
FN = int(((pred == 0) & (labels == 1)).sum())
TN = int(((pred == 0) & (labels == 0)).sum())
TPR = TP / (TP + FN); FPR = FP / (FP + TN); TNR = TN / (TN + FP)
PREC = TP / (TP + FP); F1 = 2 * PREC * TPR / (PREC + TPR); ACC = (TP + TN) / len(labels)
print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
print(f"by hand : TPR=recall={TPR:.4f} FPR={FPR:.4f} specificity={TNR:.4f} "
      f"precision={PREC:.4f} F1={F1:.4f} accuracy={ACC:.4f}")
# sklearn's confusion matrix is [[TN, FP], [FN, TP]] for labels (0, 1)
print("sklearn confusion_matrix:\n", confusion_matrix(labels, pred))
print(f"sklearn : recall={recall_score(labels, pred):.4f} precision={precision_score(labels, pred):.4f} "
      f"F1={f1_score(labels, pred):.4f} accuracy={accuracy_score(labels, pred):.4f}")
```

Output:

```text
TP=3 FP=3 FN=1 TN=3
by hand : TPR=recall=0.7500 FPR=0.5000 specificity=0.5000 precision=0.5000 F1=0.6000 accuracy=0.6000
sklearn confusion_matrix:
 [[3 3]
 [1 3]]
sklearn : recall=0.7500 precision=0.5000 F1=0.6000 accuracy=0.6000
```

Check: these match Section 3.2 exactly, and the sklearn matrix is `[[TN, FP], [FN, TP]]`.

### 4.2 ROC curve by hand, AUC three ways

The script builds the ROC staircase item by item, then computes AUC (a) as a trapezoid area, (b) by counting correctly ordered (positive, negative) pairs, (c) from SciPy's Mann–Whitney U statistic, and compares with scikit-learn.

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy.stats import mannwhitneyu

scores = np.array([0.95, 0.90, 0.80, 0.70, 0.65, 0.60, 0.40, 0.30, 0.20, 0.10])
labels = np.array([1,    1,    0,    1,    0,    0,    1,    0,    0,    0])
P, N = labels.sum(), (1 - labels).sum()

# 1) ROC by hand: lower the threshold one item at a time (no ties here)
order = np.argsort(-scores)
tp = np.concatenate([[0], np.cumsum(labels[order])])
fp = np.concatenate([[0], np.cumsum(1 - labels[order])])
tpr, fpr = tp / P, fp / N
for k in range(len(tpr)):
    print(f"top-{k:2d}: FPR={fpr[k]:.3f} TPR={tpr[k]:.3f}")
auc_trapz = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)

# 2) AUC as a probability: fraction of (positive, negative) pairs ranked correctly
pos, neg = scores[labels == 1], scores[labels == 0]
diff = pos[:, None] - neg[None, :]
auc_pairs = ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (P * N)

# 3) Mann-Whitney U statistic of the positives
U = mannwhitneyu(pos, neg, alternative="greater").statistic

fpr_s, tpr_s, thr = roc_curve(labels, scores)
print(f"AUC trapezoid by hand = {auc_trapz:.4f}")
print(f"AUC pair counting      = {auc_pairs:.4f}  ({int((diff > 0).sum())} of {P * N} pairs)")
print(f"U / (P*N)              = {U / (P * N):.4f}  (U = {U})")
print(f"sklearn roc_auc_score  = {roc_auc_score(labels, scores):.4f}")
print(f"sklearn auc(roc_curve) = {auc(fpr_s, tpr_s):.4f}")
```

Output:

```text
top- 0: FPR=0.000 TPR=0.000
top- 1: FPR=0.000 TPR=0.250
top- 2: FPR=0.000 TPR=0.500
top- 3: FPR=0.167 TPR=0.500
top- 4: FPR=0.167 TPR=0.750
top- 5: FPR=0.333 TPR=0.750
top- 6: FPR=0.500 TPR=0.750
top- 7: FPR=0.500 TPR=1.000
top- 8: FPR=0.667 TPR=1.000
top- 9: FPR=0.833 TPR=1.000
top-10: FPR=1.000 TPR=1.000
AUC trapezoid by hand = 0.8333
AUC pair counting      = 0.8333  (20 of 24 pairs)
U / (P*N)              = 0.8333  (U = 20.0)
sklearn roc_auc_score  = 0.8333
sklearn auc(roc_curve) = 0.8333
```

All four methods agree on 0.8333 = 20/24, which is the theorem of Section 3.6 in action.

### 4.3 Precision–recall, average precision, and the trapezoid

```python
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

scores = np.array([0.95, 0.90, 0.80, 0.70, 0.65, 0.60, 0.40, 0.30, 0.20, 0.10])
labels = np.array([1,    1,    0,    1,    0,    0,    1,    0,    0,    0])
P = labels.sum()
order = np.argsort(-scores)
y = labels[order]
tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
precision, recall = tp / k, tp / P
for i in range(len(y)):
    flag = "<- positive" if y[i] else ""
    print(f"rank {k[i]:2d}: precision={precision[i]:.4f} recall={recall[i]:.2f} {flag}")

ap_hand = precision[y == 1].sum() / P                      # mean precision at each positive
pr, rc, _ = precision_recall_curve(labels, scores)
print(f"AP by hand                 = {ap_hand:.4f}")
print(f"sklearn average_precision  = {average_precision_score(labels, scores):.4f}")
print(f"trapezoid auc(recall, prec)= {auc(rc, pr):.4f}   <- linear interpolation, not the same number")
print(f"random-ranker baseline     = positive rate = {labels.mean():.2f}")
```

Output:

```text
rank  1: precision=1.0000 recall=0.25 <- positive
rank  2: precision=1.0000 recall=0.50 <- positive
rank  3: precision=0.6667 recall=0.50 
rank  4: precision=0.7500 recall=0.75 <- positive
rank  5: precision=0.6000 recall=0.75 
rank  6: precision=0.5000 recall=0.75 
rank  7: precision=0.5714 recall=1.00 <- positive
rank  8: precision=0.5000 recall=1.00 
rank  9: precision=0.4444 recall=1.00 
rank 10: precision=0.4000 recall=1.00 
AP by hand                 = 0.8304
sklearn average_precision  = 0.8304
trapezoid auc(recall, prec)= 0.8110   <- linear interpolation, not the same number
random-ranker baseline     = positive rate = 0.40
```

AP is the mean of the four precisions marked "positive" (1, 1, 0.75, 0.5714). The trapezoidal area is a *different number* — the reason Section 3.7.3 tells you never to mix the two.

### 4.4 ROC optimism under imbalance: same model, bigger haystack

We keep the score distributions fixed (positives $\mathcal{N}(1.5,1)$, negatives $\mathcal{N}(0,1)$, 2,000 positives) and add more and more negatives.

```python
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(0)
n_pos = 2000
print(" pos.rate  n_neg     AUC    AP   prec@FPR=5%  #FP@FPR=5%  #TP@FPR=5%")
for rate in [0.5, 0.1, 0.01, 0.001]:
    n_neg = int(round(n_pos * (1 - rate) / rate))
    s_pos = rng.normal(1.5, 1.0, n_pos)        # positives score a bit higher...
    s_neg = rng.normal(0.0, 1.0, n_neg)        # ...than negatives: same distributions every time
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    s = np.r_[s_pos, s_neg]
    thr = np.quantile(s_neg, 0.95)             # threshold that gives FPR = 5%
    tp, fp = (s_pos >= thr).sum(), (s_neg >= thr).sum()
    print(f"{rate:8.3f} {n_neg:7d}  {roc_auc_score(y, s):.3f}  {average_precision_score(y, s):.3f}"
          f"     {tp / (tp + fp):.3f}     {fp:8d}   {tp:8d}")
print("theoretical AUC = Phi(1.5/sqrt(2)) =", round(norm.cdf(1.5 / np.sqrt(2)), 4))
```

Output:

```text
 pos.rate  n_neg     AUC    AP   prec@FPR=5%  #FP@FPR=5%  #TP@FPR=5%
   0.500    2000  0.851  0.847     0.894          100        841
   0.100   18000  0.861  0.490     0.503          900        911
   0.010  198000  0.858  0.118     0.083         9900        892
   0.001 1998000  0.858  0.017     0.009        99900        911
theoretical AUC = Phi(1.5/sqrt(2)) = 0.8556
```

Read the table row by row: AUC hovers around its theoretical value 0.856, while AP falls from 0.85 to 0.02. At a fixed FPR of 5 % the model always catches about 45 % of positives (≈ 900 of 2,000), but the number of false alarms grows from 100 to almost 100,000.

### 4.5 Random baselines on a Fdataset-sized problem

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Fdataset-sized random ranker: 1,933 positives among 185,609 pairs
N, P = 593 * 313, 1933
y = np.zeros(N); y[:P] = 1
rng = np.random.default_rng(1)
aucs, aps = [], []
for _ in range(20):
    s = rng.random(N)                          # scores that ignore the data
    aucs.append(roc_auc_score(y, s)); aps.append(average_precision_score(y, s))
H = np.sum(1.0 / np.arange(1, N + 1))          # harmonic number H_N
exact = (P - 1) / (N - 1) + (H / N) * (N - P) / (N - 1)
print(f"random AUC : mean {np.mean(aucs):.4f}  sd {np.std(aucs):.4f}")
print(f"random AP  : mean {np.mean(aps):.5f} sd {np.std(aps):.5f}")
print(f"positive rate P/N = {P / N:.5f};  exact E[AP] = {exact:.5f}")
print(f"MV-HGAT AUPR 0.488 is {0.488 / (P / N):.0f}x the random baseline")
```

Output:

```text
random AUC : mean 0.5002  sd 0.0059
random AP  : mean 0.01045 sd 0.00019
positive rate P/N = 0.01041;  exact E[AP] = 0.01048
MV-HGAT AUPR 0.488 is 47x the random baseline
```

The simulation, the exact formula of Section 3.9, and the positive rate all agree to three significant figures.

### 4.6 Same AUC, very different AP

AUC treats every mis-ordered pair equally; AP cares where the positives are. Two hand-made rankings of 1,000 items with 10 positives:

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

n = 1000                                        # 10 positives + 990 negatives

def ranking(pos_ranks):
    """Labels in ranked order (rank 1 = highest score) and matching scores."""
    y = np.zeros(n); y[np.array(pos_ranks) - 1] = 1
    s = np.arange(n, 0, -1, dtype=float)       # strictly decreasing scores
    return y, s

# Model A: 5 positives right at the top, 5 buried at ranks 204-208
yA, sA = ranking([1, 2, 3, 4, 5, 204, 205, 206, 207, 208])
# Model B: all 10 positives in a block at ranks 100-109
yB, sB = ranking(list(range(100, 110)))
for name, y, s in [("A (top-heavy)", yA, sA), ("B (uniformly mediocre)", yB, sB)]:
    top10 = y[:10].sum()
    print(f"{name:24s} AUC={roc_auc_score(y, s):.4f}  AP={average_precision_score(y, s):.4f}  "
          f"hits in top 10 = {int(top10)}")
```

Output:

```text
A (top-heavy)            AUC=0.9000  AP=0.5194  hits in top 10 = 5
B (uniformly mediocre)   AUC=0.9000  AP=0.0519  hits in top 10 = 0
```

Verify model A by hand: its first five positives pass 0 negatives, its last five each pass 198 negatives, so the average is 99 negatives above a positive out of 990: AUC $= 1 - 99/990 = 0.9$. Model B's positives each have 99 negatives above them: also 0.9. But only model A would give a biologist anything useful in the top 10.

### 4.7 Ranking metrics per disease (Worked Example 3)

```python
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# scores for 8 drugs (rows) x 3 diseases (columns); 1 = held-out true indication
S = np.array([[.91, .20, .55], [.85, .75, .10], [.40, .70, .35], [.33, .65, .30],
              [.30, .10, .25], [.22, .60, .20], [.15, .05, .15], [.05, .02, .12]])
Y = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0],
              [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])

def ranking_metrics(y, s, k):
    order = np.argsort(-s)
    hits = y[order]
    first = np.flatnonzero(hits)[0] + 1         # rank of the first true drug
    return {"P@k": hits[:k].sum() / k, "R@k": hits[:k].sum() / y.sum(),
            "Hits@k": float(hits[:k].sum() > 0), "RR": 1 / first,
            "AP": average_precision_score(y, s)}

k = 3
rows = {f"disease {j}": ranking_metrics(Y[:, j], S[:, j], k) for j in range(3)}
df = pd.DataFrame(rows).T
print(df.round(4))
print("mean over diseases (MRR is the mean of RR):")
print(df.mean().round(4).to_string())
```

Output:

```text
              P@k  R@k  Hits@k      RR      AP
disease 0  0.3333  0.5     1.0  0.5000  0.5000
disease 1  0.3333  0.5     1.0  1.0000  0.7500
disease 2  0.0000  0.0     0.0  0.1429  0.1429
mean over diseases (MRR is the mean of RR):
P@k       0.2222
R@k       0.3333
Hits@k    0.6667
RR        0.5476
AP        0.4643
```

Note that MRR (0.548) and MAP (0.464) differ: MRR only looks at the first hit, MAP at all of them.

### 4.8 What "unknown = negative" does to the measured numbers

A fixed model's scores are evaluated twice: once against the full truth, then against labels where some true indications have been "forgotten" (labelled 0), as in a real benchmark.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
n_true_pos, n_true_neg = 4000, 180000
s_pos = rng.normal(2.0, 1.0, n_true_pos)       # one model, fixed scores
s_neg = rng.normal(0.0, 1.0, n_true_neg)
s = np.r_[s_pos, s_neg]
y_true = np.r_[np.ones(n_true_pos), np.zeros(n_true_neg)]
auc_true = roc_auc_score(y_true, s)
print(f"truth (all {n_true_pos} indications known): AUC={auc_true:.4f} AP={average_precision_score(y_true, s):.4f}")
print(" hidden  q=hidden/labelled-neg  measured AUC  predicted (1-q)AUC+q/2  measured AP  P@100")
for frac_hidden in [0.0, 0.25, 0.5]:
    y_obs = y_true.copy()
    hidden = rng.choice(n_true_pos, int(frac_hidden * n_true_pos), replace=False)
    y_obs[hidden] = 0                          # undiscovered indications are labelled 0
    q = len(hidden) / (y_obs == 0).sum()
    top100 = np.argsort(-s)[:100]
    print(f"  {frac_hidden:4.0%}        {q:.4f}            {roc_auc_score(y_obs, s):.4f}"
          f"              {(1 - q) * auc_true + q / 2:.4f}            {average_precision_score(y_obs, s):.4f}"
          f"     {y_obs[top100].mean():.2f}")
```

Output:

```text
truth (all 4000 indications known): AUC=0.9186 AP=0.3693
 hidden  q=hidden/labelled-neg  measured AUC  predicted (1-q)AUC+q/2  measured AP  P@100
    0%        0.0000            0.9186              0.9186            0.3693     0.85
   25%        0.0055            0.9149              0.9163            0.2746     0.64
   50%        0.0110            0.9148              0.9140            0.1894     0.38
```

AUC moves by about 0.004 and follows the formula $(1-q)\text{AUC}_\text{true} + q/2$ closely (the small mismatch is sampling noise: which positives get hidden is random). AP and precision@100 fall sharply, because the hidden positives sit near the top of the ranking and are now counted as false alarms. The model did not get worse — the answer key did.

### 4.9 Warm start versus cold start on the real Fdataset

Two simple "guilt-by-association" scorers that use no learning at all: one propagates links from a drug's 10 most chemically similar drugs (ECFP Tanimoto), the other from a disease's 10 most phenotypically similar diseases (MimMiner). Each is evaluated with warm 5-fold CV and with leave-one-disease-out.

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

z = np.load(r"C:/Users/Abhineet Anand/Desktop/DrugRepositioning/data/processed/Fdataset.npz")
A = z["A"].astype(float)                                   # 593 drugs x 313 diseases

def knn_weights(S, k=10):
    S = np.nan_to_num(S.copy()); np.fill_diagonal(S, 0)    # an entity is not its own neighbour
    W = np.zeros_like(S); nn = np.argsort(-S, axis=1)[:, :k]
    r = np.arange(len(S))[:, None]; W[r, nn] = S[r, nn]
    return W / (W.sum(1, keepdims=True) + 1e-12)           # rows sum to 1

Wr = knn_weights(z["drug_views"][list(z["drug_view_names"]).index("chem_ecfp")])
Wd = knn_weights(z["disease_views"][list(z["disease_view_names"]).index("pheno_mim")])
scorers = {
    "drug side (chem_ecfp)":    lambda At: Wr @ At,       # do drugs similar to i treat j?
    "disease side (pheno_mim)": lambda At: At @ Wd.T,     # does i treat diseases similar to j?
}

rng = np.random.default_rng(0)
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
rng.shuffle(pos); rng.shuffle(neg)
folds = list(zip(np.array_split(pos, 5), np.array_split(neg, 5)))
for name, f in scorers.items():
    res = []                                           # warm start: 5-fold CV over cells
    for tp, tn in folds:
        At = A.ravel().copy(); At[tp] = 0; S = f(At.reshape(A.shape)).ravel()
        idx = np.r_[tp, tn]; y = np.r_[np.ones(len(tp)), np.zeros(len(tn))]
        res.append((roc_auc_score(y, S[idx]), average_precision_score(y, S[idx])))
    ys, ss = [], []                                    # cold start: leave-one-disease-out
    for j in range(A.shape[1]):
        At = A.copy(); At[:, j] = 0
        ys.append(A[:, j]); ss.append(f(At)[:, j])
    y, s = np.concatenate(ys), np.concatenate(ss)
    print(f"{name:25s} warm CV: AUC {np.mean(res, 0)[0]:.4f} AUPR {np.mean(res, 0)[1]:.4f} | "
          f"LODO: AUC {roc_auc_score(y, s):.4f} AUPR {average_precision_score(y, s):.4f}")
```

Output:

```text
drug side (chem_ecfp)     warm CV: AUC 0.7348 AUPR 0.2138 | LODO: AUC 0.5000 AUPR 0.0104
disease side (pheno_mim)  warm CV: AUC 0.7144 AUPR 0.1579 | LODO: AUC 0.7377 AUPR 0.1739
```

The drug-side scorer is exactly random under LODO (every drug gets score 0 for the held-out disease, so all are tied), while the disease-side scorer is untouched. This is the precise reason cold start is hard: it removes the signals that pass through the held-out disease's own links.

### 4.10 Pooled versus per-disease metrics on the project's own predictions

`run_kfold` saves the pooled $(y, s)$ of repeat 0 in `results/Fdataset/cv5/<method>_preds.npz`, in the order `[fold-0 positives, fold-0 negatives, fold-1 positives, …]`. Because `kfold_splits` is deterministic given the seed, we can recover which cell each score belongs to, rebuild a full $593 \times 313$ score matrix, and evaluate it disease by disease.

```python
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

ROOT = "C:/Users/Abhineet Anand/Desktop/DrugRepositioning"
sys.path.insert(0, ROOT + "/src")
from drepo.evaluation import kfold_splits                 # read-only import of project code

A = np.load(ROOT + "/data/processed/Fdataset.npz")["A"].astype(np.float32)
# run_kfold concatenates [test_pos, test_neg] of folds 0..4 of repeat 0 (seed 0):
idx = np.concatenate([np.r_[p, n] for _, p, n in kfold_splits(A, 5, 0)])

for m in ["mvhgat", "scmfdd"]:
    z = np.load(f"{ROOT}/results/Fdataset/cv5/{m}_preds.npz"); y, s = z["y"], z["s"]
    assert (A.ravel()[idx] == y).all()                     # reconstruction is exact
    order = np.argsort(-s)
    pr, rc, _ = precision_recall_curve(y, s)
    print(f"{m:7s} POOLED  AUC {roc_auc_score(y, s):.4f}  AP {average_precision_score(y, s):.4f}"
          f"  trapz-AUPR {auc(rc, pr):.4f}  P@100 {y[order[:100]].mean():.2f}"
          f"  P@1000 {y[order[:1000]].mean():.3f}")
    S = np.empty(A.size); S[idx] = s; S = S.reshape(A.shape)   # each cell scored by its test fold
    per = []
    for j in range(A.shape[1]):                            # one ranking of all drugs per disease
        yj, sj = A[:, j], S[:, j]
        o = np.argsort(-sj); first = np.flatnonzero(yj[o])[0] + 1
        per.append((roc_auc_score(yj, sj), average_precision_score(yj, sj), 1 / first,
                    yj[o[:10]].sum() > 0, yj[o[:10]].sum() / yj.sum()))
    a = np.mean(per, 0)
    print(f"{'':7s} PER-DISEASE mean AUC {a[0]:.4f}  mean AP {a[1]:.4f}  MRR {a[2]:.4f}"
          f"  Hits@10 {a[3]:.4f}  Recall@10 {a[4]:.4f}")
```

Output:

```text
mvhgat  POOLED  AUC 0.9420  AP 0.4917  trapz-AUPR 0.4916  P@100 0.98  P@1000 0.683
        PER-DISEASE mean AUC 0.8815  mean AP 0.3603  MRR 0.4947  Hits@10 0.6677  Recall@10 0.4132
scmfdd  POOLED  AUC 0.8943  AP 0.4879  trapz-AUPR 0.4879  P@100 1.00  P@1000 0.708
        PER-DISEASE mean AUC 0.7951  mean AP 0.2778  MRR 0.3902  Hits@10 0.5463  Recall@10 0.3055
```

Pooled AP: a tie. Per disease: MV-HGAT is ahead on every metric (mean AUC +0.09, MAP +0.08, MRR +0.10, Hits@10 +0.12). Also note that the pooled trapezoidal AUPR equals AP to three decimals here, because the curve has 1,933 steps.

### 4.11 Comparing two methods: paired splits and a corrected t-test

```python
import json
import numpy as np
from scipy import stats

R = "C:/Users/Abhineet Anand/Desktop/DrugRepositioning/results/Fdataset/cv5/"
folds = {m: json.load(open(R + m + ".json"))["folds"] for m in ["mvhgat", "scmfdd"]}
for metric in ["AUC", "AUPR"]:
    x = np.array([f[metric] for f in folds["mvhgat"]])
    y = np.array([f[metric] for f in folds["scmfdd"]])     # same 25 splits (same seeds)
    d = x - y
    J = len(d)
    print(f"{metric}: MV-HGAT {x.mean():.4f} +/- {x.std():.4f} (ddof=0) / +/- {x.std(ddof=1):.4f} (ddof=1);"
          f" SCMFDD {y.mean():.4f} +/- {y.std():.4f}")
    print(f"   paired diff {d.mean():+.4f} +/- {d.std(ddof=1):.4f}; MV-HGAT better in {(d > 0).sum()}/{J} splits")
    t_naive = d.mean() / np.sqrt(d.var(ddof=1) / J)
    t_nb = d.mean() / np.sqrt((1 / J + 1 / 4) * d.var(ddof=1))   # n_test/n_train = 1/4 for 5-fold
    print(f"   naive paired t = {t_naive:6.2f} (p = {2 * stats.t.sf(abs(t_naive), J - 1):.1e});"
          f"  Nadeau-Bengio corrected t = {t_nb:5.2f} (p = {2 * stats.t.sf(abs(t_nb), J - 1):.1e})")
```

Output:

```text
AUC: MV-HGAT 0.9392 +/- 0.0072 (ddof=0) / +/- 0.0073 (ddof=1); SCMFDD 0.8934 +/- 0.0102
   paired diff +0.0458 +/- 0.0085; MV-HGAT better in 25/25 splits
   naive paired t =  26.87 (p = 2.0e-19);  Nadeau-Bengio corrected t =  9.98 (p = 5.1e-10)
AUPR: MV-HGAT 0.4878 +/- 0.0273 (ddof=0) / +/- 0.0278 (ddof=1); SCMFDD 0.4946 +/- 0.0199
   paired diff -0.0068 +/- 0.0329; MV-HGAT better in 8/25 splits
   naive paired t =  -1.04 (p = 3.1e-01);  Nadeau-Bengio corrected t = -0.39 (p = 7.0e-01)
```

The "naive" p-value treats the 25 folds as independent experiments, which they are not; the Nadeau–Bengio correction widens the standard error by a factor of about 2.7. For AUC the conclusion survives easily; for AUPR there is no evidence of a difference either way.

### 4.12 Pooling folds can hurt a perfect ranker

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Two CV folds, each with its own trained model. Each model ranks its own fold PERFECTLY,
# but model 2 outputs systematically lower numbers (a calibration offset).
y1 = np.array([1, 1, 0, 0, 0, 0]); s1 = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40])
y2 = np.array([1, 1, 0, 0, 0, 0]); s2 = np.array([0.45, 0.35, 0.30, 0.20, 0.10, 0.05])
f1, f2 = roc_auc_score(y1, s1), roc_auc_score(y2, s2)
print(f"fold AUCs: {f1:.3f}, {f2:.3f} -> per-fold mean {np.mean([f1, f2]):.3f}")
y, s = np.r_[y1, y2], np.r_[s1, s2]
print(f"pooled AUC = {roc_auc_score(y, s):.3f}, pooled AP = {average_precision_score(y, s):.3f}")
```

Output:

```text
fold AUCs: 1.000, 1.000 -> per-fold mean 1.000
pooled AUC = 0.781, pooled AP = 0.750
```

Check the pooled AUC by hand: the four positives (0.90, 0.80, 0.45, 0.35) beat 8, 8, 5 and 4 of the 8 negatives respectively: $25/32 = 0.781$.

---

## 5. In this project: where every concept lives

### 5.1 `src/drepo/evaluation.py::metrics` — the two headline numbers

```python
def metrics(y, s):
    return {"AUC": float(roc_auc_score(y, s)), "AUPR": float(average_precision_score(y, s))}
```

- `roc_auc_score` is the area under the ROC staircase, i.e. the Mann–Whitney probability of Section 3.6 (ties count half).
- `average_precision_score` is **AP** (Section 3.7.2), the step-wise sum, *not* a trapezoid. Wherever the project or its paper says "AUPR", it means AP. Write that sentence in the Methods section.
- Both are ranking metrics, so it does not matter that MBiRW outputs walk probabilities, DRRS outputs completed matrix entries, and MV-HGAT outputs sigmoids.

### 5.2 `kfold_splits` — stratified folds over cells

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

- `A.ravel()` turns the 593 × 313 matrix into 185,609 cells; index $k$ means drug $k \,//\, 313$, disease $k \bmod 313$.
- Positives and negatives are shuffled and split **separately**, so every fold has the same positive rate (≈ 1.04 %): stratification.
- The split depends only on `seed`. Every method run with the same seed sees **exactly the same 25 splits**, which is what makes the paired comparison of Section 3.14.3 valid.

### 5.3 `run_kfold` — what is hidden, and where negatives may come from

```python
A_tr = A.copy().ravel()
A_tr[test_pos] = 0                       # hide this fold's known links
A_tr = A_tr.reshape(shape)
neg_mask = (A == 0).ravel()
neg_mask[test_neg] = False               # test negatives may NOT be used as training negatives
neg_mask = neg_mask.reshape(shape)
S = method.fit_predict(data, A_tr, neg_mask, seed=...)
idx = np.concatenate([test_pos, test_neg])
y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
s = S.ravel()[idx]
m = metrics(y, s)
```

Two leakage guards in five lines: the held-out positives are zeroed in the training matrix, and the held-out negatives are removed from the pool of sampled training negatives (otherwise a model could learn "these particular cells are 0" and score them low at test time). Note also the definition of `neg_mask`: the training negatives are exactly the "unknown = negative" assumption of Section 3.13.

Pooled curves: `if r == 0: pooled_y.append(y); pooled_s.append(s)` keeps **only repeat 0** for the ROC/PR figures. The table numbers use all 25 folds; the figures use 5. That is fine, but the legend values in the figures come from the table (mean over 25 folds), not from the plotted pooled curve, so they will not equal the area under the plotted curve exactly (e.g. MV-HGAT: table AP 0.4878, pooled-curve AP 0.4917).

### 5.4 `summarise` — mean ± std

```python
auc = np.array([f["AUC"] for f in folds])
...
"AUC": float(auc.mean()), "AUC_std": float(auc.std()), ...
```

`np.std` with default `ddof=0`: the population standard deviation of the 25 fold scores (Section 3.14.2). It describes the spread of single-fold results, not the uncertainty of the mean.

### 5.5 `run_lodo` — cold start, pooled + per-disease

```python
A_tr = A.copy()
A_tr[:, j] = 0                           # hide ALL known drugs of disease j
neg_mask = A == 0
neg_mask[:, j] = False                   # never train on the held-out column
S = method.fit_predict(data, A_tr, neg_mask, seed=int(seed + j))
y, s = A[:, j], S[:, j]
...
if 0 < y.sum() < len(y):
    per.append(roc_auc_score(y, s))
...
res = metrics(y, s)                      # pooled over all evaluated diseases
res["mean_per_disease_AUC"] = float(np.mean(per))
```

- The model is re-trained once per held-out disease: expensive (≈ 24 s per disease for MV-HGAT on the RTX 3050 in `results/logs/F_lodo.log`), which is why `runner.run_protocol` evaluates a seeded random subset (`--lodo-subset 100`).
- Both a **pooled** AUC/AP and a **mean per-disease AUC** are reported (Section 3.12). A disease with no known drugs is skipped for the per-disease mean (its AUC is undefined).
- The partial log shows running pooled AUCs of 0.868 → 0.833 → 0.853 after 25/50/75 diseases: with few diseases, pooled numbers are noisy, another reason to report the per-disease mean and its spread.

### 5.6 `scripts/06_case_study.py` — precision@10 against external evidence

```python
order = [i for i in np.argsort(-S[:, j]) if data.A[i, j] == 0][: args.top]
...
confirmed = (df.CTD_curated != "-") | (df[f"known_in_{other.name}"] == "yes") | \
            (df.clinical_trials.fillna(0).astype(float) > 0)
print(f"    {confirmed.sum()}/{len(df)} of the top {args.top} have independent support")
```

This is precision@10 per disease, computed against a *different* answer key (CTD curated links, the other benchmark, ClinicalTrials.gov) — a partial remedy for "unknown = negative". Real results (`results/logs/C_case.log`): prostate cancer 10/10, breast cancer 6/10, Alzheimer's 5/10.

**A measurement-validity lesson from our own results.** The Alzheimer's list is topped by amantadine with "49 clinical trials". We re-queried the ClinicalTrials.gov API and inspected every one of those 49 studies: **none lists amantadine as an intervention; 48 list memantine.** ClinicalTrials.gov expands search terms through the MeSH vocabulary, and memantine is filed under amantadine there (memantine is a dimethyl derivative of amantadine). The same happens for daunorubicin × breast cancer: 34 hits, but daunorubicin is literally named in only 2; 25 are doxorubicin trials. So the external "support" counter itself has false positives: Alzheimer's should read at most **4/10**, not 5/10 (and 3/10 if one also discounts a trial of R(+)-pramipexole, a different enantiomer from the marketed drug; D1 Section 5.5), while breast cancer stays 6/10 because daunorubicin keeps 2 genuine trials. Unit D1, Section 4.5 contains the audit code. The general principle: *an evaluation is only as good as its labels — check them.*

### 5.7 `scripts/07_cross_dataset.py` — a genuinely external test

```python
cand = A_src == 0                    # pairs unseen in the source
y = A_tgt[cand]                      # 1 if the OTHER benchmark knows this link
...
print(f"... (random AUPR = {y.mean():.4f})")
```

The script prints the random-AP baseline right next to every result — exactly the habit Section 3.9 recommends. With 57 new links in 174,330 candidates, the baseline is 0.00033; MV-HGAT's AP 0.0086 is a 26× lift, MBiRW's 0.0136 a 42× lift, despite MBiRW's lower AUC (0.898 vs 0.944). Another case of crossing curves.

### 5.8 `scripts/08_make_figures.py` — drawing the curves

```python
fpr, tpr, _ = roc_curve(y, s)
pr, rc, _ = precision_recall_curve(y, s)
a1.plot(fpr, tpr, ..., label=f"{name} ({summ['AUC']:.3f})")
a2.plot(rc, pr, ..., label=f"{name} ({summ['AUPR']:.3f})")
a1.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
```

The ROC panel draws the random diagonal; the PR panel does **not** draw its random baseline (a horizontal line at $\pi = 0.0104$). On a 0–1 axis it would be almost invisible, which is itself the point; a reader of the figure should be told the value in the caption. A useful extra figure would be the ROC curve with a **logarithmic FPR axis**, which makes the early-retrieval region (FPR < 1 %) visible.

### 5.9 Reading the results table like a reviewer

From `results/RESULTS.md` (Fdataset, 5-fold, mean ± std over 25 folds):

| Method | AUC | AUPR | AUPR lift over random (÷ 0.0104) |
|---|---|---|---|
| MV-HGAT (ours) | 0.9392 ± 0.0072 | 0.4878 ± 0.0273 | 47× |
| SCMFDD | 0.8934 ± 0.0102 | 0.4946 ± 0.0199 | 48× |
| DRRS | 0.8787 ± 0.0123 | 0.3893 ± 0.0174 | 37× |
| MBiRW | 0.8831 ± 0.0095 | 0.3105 ± 0.0187 | 30× |
| LAGCN | 0.8333 ± 0.0152 | 0.1328 ± 0.0181 | 13× |
| NIMCGCN | 0.8384 ± 0.0089 | 0.0956 ± 0.0137 | 9× |

Observations a careful reader makes:

1. The AUC ordering (MV-HGAT ≫ SCMFDD > MBiRW ≈ DRRS > NIMCGCN ≈ LAGCN) and the AUPR ordering (SCMFDD ≈ MV-HGAT > DRRS > MBiRW ≫ LAGCN > NIMCGCN) differ — curves cross.
2. NIMCGCN and LAGCN have respectable AUCs (0.83–0.84) but AUPRs of only 9–13× random: they order the bulk of the list sensibly but are poor at the top. AUC alone would hide that.
3. The AUPR standard deviations (≈ 0.02–0.03) are larger than the MV-HGAT–SCMFDD gap: Section 3.14.3's paired test confirms no significant difference.
4. 10-fold CV numbers (`F_cv10.log`) are all higher than 5-fold (MV-HGAT 0.954 / 0.524): with 10 folds, 90 % of links are visible during training instead of 80 %, so the task is easier. **Never compare a 10-fold number from one paper with a 5-fold number from another.**

---

## 6. Common mistakes and misconceptions

1. **"Accuracy 99 % — great model."** Predicting all zeros gives 98.96 % on Fdataset. Use AP, precision@k, AUC.
2. **"AUC 0.94 means 94 % of predictions are right."** No: it means 94 % of (positive, negative) *pairs* are ordered correctly. At a threshold that catches 59 % of indications, MV-HGAT's flagged list is only 39 % correct.
3. **"AUPR of 0.49 is mediocre; it is below 0.5."** AP's random baseline is the positive rate, here 0.0104; 0.49 is 47× random. AP has no fixed "chance" value of 0.5.
4. **Comparing AP across datasets or protocols.** Baselines differ (0.0104 vs 0.0093 vs 0.00033). Compare methods on the same test set, or compare lifts.
5. **Mixing AP and trapezoidal AUPR.** `average_precision_score` ≠ `auc(recall, precision)`. Differences can be large for small test sets (per-disease curves).
6. **Using `sklearn.metrics.auc` on unsorted points or on PR curves with linear interpolation** and calling it "AUPR" — see Section 3.7.3.
7. **Treating `confusion_matrix` output as `[[TP, FP], [FN, TN]]`.** It is `[[TN, FP], [FN, TP]]`.
8. **Reading model scores as probabilities.** Negative sampling (1 positive : 2 negatives) shifts the scores; they are not calibrated to the 1 : 95 reality.
9. **Comparing 10-fold results with 5-fold results**, or warm-start results with cold-start results. Different protocols, different difficulty.
10. **Reporting only pooled metrics.** A method can win pooled AP by ranking popular diseases above rare ones. Add per-disease MRR / Hits@10.
11. **Treating the ± std as a confidence interval for the mean,** or running a naive t-test on CV folds. Folds overlap; use paired comparisons and a corrected test, and state ddof.
12. **Choosing hyper-parameters on the test folds.** Inflates every metric. The project tunes on a separate validation split (`05_sensitivity.py`).
13. **Forgetting that unknown ≠ negative.** Measured precision is a lower bound; "false positives" at the top may be undiscovered indications — which is the entire point of repositioning.
14. **Trusting an external validation source blindly.** ClinicalTrials.gov's synonym expansion turned 0 amantadine trials into "49"; always inspect a sample of the evidence.
15. **Reporting a cold-start result without saying which diseases were evaluated.** LODO on a 100-disease subset is fine if stated; it is not comparable to full LODO.
16. **Leaving ties unhandled.** Methods that output many identical scores (e.g. all zeros in cold start) get AUC 0.5 from tied pairs counted as half; if your own code breaks ties by index order, you can get a spuriously high or low AUC.

---

## 7. Exercises

Exercises are graded: ★ conceptual, ★★ mathematical, ★★★ coding / research-level. Try each before opening the solution.

**Exercise 1 (★).** On Cdataset (663 × 409 cells, 2,532 known links), what accuracy does the constant prediction "no drug treats any disease" achieve? What are its TPR, FPR, precision and F1?

<details><summary>Solution</summary>

Cells $= 663 \times 409 = 271{,}167$; negatives $= 268{,}635$. Accuracy $= 268635/271167 = 0.99066$, i.e. 99.07 %. It flags nothing, so $TP = FP = 0$: TPR $= 0$, FPR $= 0$; precision $= 0/0$ is undefined (scikit-learn returns 0 with a warning); F1 $= 0$. A metric that gives this useless rule 99 % is the wrong metric.
</details>

**Exercise 2 (★★).** A classifier on Cdataset has TPR 0.80 at FPR 0.02. Compute its precision, the expected number of false positives, and its F1. What FPR would be needed for precision 0.5 at the same TPR?

<details><summary>Solution</summary>

$\pi = 2532/271167 = 0.009337$.

$$\text{precision} = \frac{0.009337 \times 0.8}{0.009337\times0.8 + 0.990663\times0.02} = \frac{0.007470}{0.007470 + 0.019813} = 0.2738.$$

False positives $= 0.02 \times 268635 = 5{,}373$; true positives $= 0.8 \times 2532 = 2{,}026$. F1 $= 2(0.2738)(0.8)/(1.0738) = 0.408$.

For precision 0.5 we need $\pi\,\text{TPR} = (1-\pi)\,\text{FPR}$, so FPR $= 0.007470/0.990663 = 0.00754$, i.e. about 0.75 %: roughly 2,026 false positives, matching the 2,026 true positives.
</details>

**Exercise 3 (★★).** Compute the AUC by pair counting for this ranking, where two items tie:

| item | a | b | c | d | e | f | g |
|---|---|---|---|---|---|---|---|
| score | 0.9 | 0.8 | 0.7 | 0.7 | 0.5 | 0.3 | 0.1 |
| label | 1 | 0 | 1 | 0 | 1 | 0 | 0 |

<details><summary>Solution</summary>

$P = 3$ (a, c, e), $N = 4$ (b, d, f, g), $PN = 12$ pairs.
- a (0.9) beats b, d, f, g → 4.
- c (0.7) loses to b (0.8), **ties** d (0.7) → ½, beats f, g → 2.5.
- e (0.5) loses to b, d; beats f, g → 2.

Total $= 4 + 2.5 + 2 = 8.5$; AUC $= 8.5/12 = 0.7083$. `roc_auc_score([1,0,1,0,1,0,0], [.9,.8,.7,.7,.5,.3,.1])` returns 0.7083.
</details>

**Exercise 4 (★★).** For the ranking in Exercise 3 (break the c/d tie with c first), compute AP, precision@3, recall@3 and the reciprocal rank.

<details><summary>Solution</summary>

Order a(1), b(0), c(1), d(0), e(1), f(0), g(0). Positives at ranks 1, 3, 5 with precisions 1/1, 2/3, 3/5. AP $= (1 + 0.6667 + 0.6)/3 = 0.7556$. Top 3 = a, b, c → 2 hits: P@3 $= 2/3$, R@3 $= 2/3$. First positive at rank 1 → RR $= 1$.

(If the tie is broken the other way, d before c, the precisions become 1, 2/4, 3/5 and AP $= 0.7$. scikit-learn's `average_precision_score` handles the tie by treating c and d as one threshold step: at that step recall rises by 1/3 and precision is 2/4, so it returns 0.7 as well — the pessimistic value. Ties matter for AP.)
</details>

**Exercise 5 (★★).** Prove that if you negate all scores ($s \mapsto -s$), the AUC becomes $1 - \text{AUC}$ (assume no ties). What happens with ties?

<details><summary>Solution</summary>

Without ties, each (positive, negative) pair is ordered either correctly or incorrectly, so (#correct) + (#incorrect) $= PN$. Negating the scores swaps "correct" and "incorrect" for every pair. Hence $\text{AUC}' = \#\text{incorrect}/PN = 1 - \text{AUC}$. With ties, tied pairs contribute ½ before and after negation, so $\text{AUC}' = 1 - \text{AUC}$ still holds: $\text{AUC} = (C + T/2)/PN$ and $\text{AUC}' = (I + T/2)/PN$ with $C + I + T = PN$, so $\text{AUC} + \text{AUC}' = (C+I+T)/PN = 1$. Consequence: an AUC of 0.2 is "very informative, but backwards".
</details>

**Exercise 6 (★).** Model A: AUC 0.95, AP 0.20. Model B: AUC 0.90, AP 0.35. Both evaluated on the same Fdataset folds. A collaborator will test the top 20 predictions per disease in the lab. Which model should you give them, and what would you check first?

<details><summary>Solution</summary>

Probably model B: AP weights the top of the ranking, which is what the lab will see; A's higher AUC comes from ordering the bulk of the list better. The curves must cross (Davis & Goadrich: if one ROC curve dominated, it would also dominate in PR space). But check what is actually used: compute **per-disease** precision@20 / Hits@20 / MRR for both, because the lab works per disease and pooled AP can be won by ranking popular diseases first (Section 3.12). Also check paired significance across folds.
</details>

**Exercise 7 (★★).** In LODO, a disease has 3 known drugs among 593. What AP does a random ranker get on this disease, using the exact formula of Section 3.9? Compare with the positive rate and explain why they differ.

<details><summary>Solution</summary>

$$\mathbb{E}[\text{AP}] = \frac{P-1}{N-1} + \frac{H_N}{N}\cdot\frac{N-P}{N-1},\quad N = 593,\ P = 3.$$

$\frac{2}{592} = 0.003378$; $H_{593} \approx \ln 593 + 0.5772 + \frac{1}{1186} = 6.9633$; $\frac{6.9633}{593}\cdot\frac{590}{592} = 0.011703$. Sum $= 0.01508$, versus $\pi = 3/593 = 0.00506$. Code:

```python
import numpy as np
from sklearn.metrics import average_precision_score
N, P = 593, 3
H = np.sum(1 / np.arange(1, N + 1))
exact = (P - 1) / (N - 1) + H / N * (N - P) / (N - 1)
rng = np.random.default_rng(0)
y = np.zeros(N); y[:P] = 1
sim = np.mean([average_precision_score(y, rng.random(N)) for _ in range(20000)])
print(f"positive rate {P / N:.5f}  exact E[AP] {exact:.5f}  simulated {sim:.5f}")
```
```text
positive rate 0.00506  exact E[AP] 0.01508  simulated 0.01505
```

With few positives, the term $H_N/N$ (the chance that a positive lands very near the top by luck, where precision is high) is no longer negligible: the random baseline for a single small query is **3× the positive rate**. This is why per-disease AP should be compared with its own baseline, and why per-disease metrics are noisy for diseases with 1–3 drugs.
</details>

**Exercise 8 (★★).** Suppose 2 % of the pairs labelled 0 in a test set are actually undiscovered indications, and those behave like known positives. A model's true AUC is 0.95. What AUC will you measure? If the true precision among the top 100 is 0.90 and 30 of those 100 are hidden positives, what precision@100 will you measure?

<details><summary>Solution</summary>

$\text{AUC}_\text{meas} = (1-0.02)(0.95) + 0.02(0.5) = 0.931 + 0.010 = 0.941$.

True hits in the top 100: 90, of which 30 are hidden (labelled 0). Measured hits: 60 → measured precision@100 $= 0.60$. AUC dropped by 0.009; precision@100 dropped by 0.30. The precision-type metrics carry almost all the bias.
</details>

**Exercise 9 (★).** In LODO, which parts of MV-HGAT can still provide information about the held-out disease $j$, and which cannot? Use `HOW_IT_WORKS.md` Section 4.

<details><summary>Solution</summary>

**Cannot:** the `assoc` edges of disease $j$ (all hidden), its input features from visible links (all zeros), drug-side propagation terms $P_v[i,j]$ for drug views (they propagate column $j$, which is zero), and the bilinear GNN term's collaborative signal is down-weighted by the **degree gate** (degree of $j$ is 0, so the gate is small).

**Can:** the disease similarity views (`pheno_mim`, `sem_mondo`, `gene_d`) — both as GAT relations giving $j$ an embedding from similar diseases, and as disease-side propagation terms ("drugs used for diseases similar to $j$"); and the optional gene bridge. The model's explicit "cold-start practice" during training (10 % of diseases lose all links each epoch) is what teaches it to rely on these.
</details>

**Exercise 10 (★★★, coding).** Implement AUC from the rank-sum formula with average ranks for ties, and verify it against scikit-learn on 1,000 random items with many ties.

<details><summary>Solution</summary>

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def auc_ranksum(y, s):
    """AUC from the Mann-Whitney rank-sum formula, with average ranks for ties."""
    y = np.asarray(y); r = rankdata(s)             # rank 1 = lowest score; ties get the mean rank
    P = y.sum(); N = len(y) - P
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)

rng = np.random.default_rng(3)
y = rng.integers(0, 2, 1000)
s = np.round(rng.normal(y * 0.8, 1.0), 1)          # rounding creates many ties
print(f"rank-sum AUC {auc_ranksum(y, s):.6f}   sklearn {roc_auc_score(y, s):.6f}")
```
```text
rank-sum AUC 0.715458   sklearn 0.715458
```

Average ranks give each tied (positive, negative) pair exactly half credit, which is why the formula matches scikit-learn's tie convention.
</details>

**Exercise 11 (★★★, coding).** Compute a 95 % bootstrap interval for MV-HGAT's pooled AP on Fdataset (repeat 0). What does the interval capture, and what does it miss?

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import average_precision_score

z = np.load("C:/Users/Abhineet Anand/Desktop/DrugRepositioning/results/Fdataset/cv5/mvhgat_preds.npz")
y, s = z["y"], z["s"]
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
rng = np.random.default_rng(0)
boot = []
for _ in range(200):                               # stratified bootstrap: resample each class
    idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
    boot.append(average_precision_score(y[idx], s[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"AP {average_precision_score(y, s):.4f}, bootstrap 95% interval [{lo:.4f}, {hi:.4f}]")
```
```text
AP 0.4917, bootstrap 95% interval [0.4727, 0.5159]
```

It captures **test-set sampling** uncertainty: if we had drawn a different set of test pairs from the same distribution, how much would AP move? It **misses** training variability (different folds, different random initialisations of the GNN) — that is what the fold-to-fold std of 0.027 reflects — and it ignores the dependence between pairs that share a drug or disease (a cluster bootstrap that resamples whole diseases would be more honest). 200 resamples is the minimum for a rough interval; use 1,000+ for a paper.
</details>

**Exercise 12 (★★).** Five paired fold differences in AP between two methods are 0.012, −0.004, 0.020, 0.008, 0.014 (5-fold CV, one repeat). Compute the naive paired t statistic and the Nadeau–Bengio corrected one, and their two-sided p-values.

<details><summary>Solution</summary>

Mean $\bar d = 0.050/5 = 0.0100$. Deviations: 0.002, −0.014, 0.010, −0.002, 0.004; squares sum to $0.00032$; sample variance $= 0.00032/4 = 0.00008$, sd $= 0.00894$.

Naive: $t = 0.0100/(0.00894/\sqrt5) = 2.500$, df = 4, $p = 0.067$.
Corrected: $t = 0.0100/\sqrt{(1/5 + 1/4)\times0.00008} = 0.0100/0.006 = 1.667$, $p = 0.171$.

```python
import numpy as np
from scipy import stats
d = np.array([0.012, -0.004, 0.020, 0.008, 0.014])   # paired AP differences, 5 folds
J = len(d); sd = d.std(ddof=1)
t_naive = d.mean() / (sd / np.sqrt(J))
t_nb = d.mean() / np.sqrt((1 / J + 1 / 4) * sd**2)
print(f"mean {d.mean():.4f} sd {sd:.4f}")
print(f"naive t {t_naive:.3f} p {2 * stats.t.sf(t_naive, J - 1):.3f}")
print(f"Nadeau-Bengio t {t_nb:.3f} p {2 * stats.t.sf(t_nb, J - 1):.3f}")
```
```text
mean 0.0100 sd 0.0089
naive t 2.500 p 0.067
Nadeau-Bengio t 1.667 p 0.171
```

Neither is significant at 0.05; with one repeat of 5 folds you have very little power. Repeating CV (the project uses 5 repeats → 25 differences) helps.
</details>

**Exercise 13 (★).** `run_kfold` splits the 0s into folds too, so each fold's test set contains only one fifth of the unknown pairs. Why not test each fold's positives against *all* unknown pairs? And why are the fold's test negatives removed from `neg_mask`?

<details><summary>Solution</summary>

(1) Each unknown pair should be tested exactly once, by a model that did not train on it as a negative. If every fold tested against all 0s, the 0s used as sampled training negatives in fold $f$ would also be test negatives in fold $f$ — the model would have been explicitly taught to score them low, inflating AUC/AP. Splitting the 0s keeps test negatives disjoint from training negatives. (2) Removing them from `neg_mask` is the enforcement of exactly that disjointness. A side benefit: every fold has the same positive rate, so the per-fold metrics are comparable.
</details>

**Exercise 14 (★★★, coding).** Compute the standardised partial AUC for FPR ≤ 1 % and ≤ 0.1 % for MV-HGAT and SCMFDD on the pooled Fdataset predictions. Relate the result to Section 3.8.

<details><summary>Solution</summary>

```python
import numpy as np
from sklearn.metrics import roc_auc_score
R = "C:/Users/Abhineet Anand/Desktop/DrugRepositioning/results/Fdataset/cv5/"
for m in ["mvhgat", "scmfdd"]:
    z = np.load(R + m + "_preds.npz")
    print(f"{m:7s} AUC {roc_auc_score(z['y'], z['s']):.4f}   "
          f"standardised partial AUC (FPR <= 1%) {roc_auc_score(z['y'], z['s'], max_fpr=0.01):.4f}   "
          f"(FPR <= 0.1%) {roc_auc_score(z['y'], z['s'], max_fpr=0.001):.4f}")
```
```text
mvhgat  AUC 0.9420   standardised partial AUC (FPR <= 1%) 0.7276   (FPR <= 0.1%) 0.5977
scmfdd  AUC 0.8943   standardised partial AUC (FPR <= 1%) 0.7296   (FPR <= 0.1%) 0.6256
```

Restricted to the early-retrieval region, the ranking flips: SCMFDD is level at FPR ≤ 1 % and ahead at FPR ≤ 0.1 %. This is the crossing of ROC curves seen in the operating-point table of Section 3.8, and it explains why full AUC favours MV-HGAT while AP ties. (The standardised partial AUC rescales so that 0.5 = random and 1 = perfect within the region; McClish correction.)
</details>

**Exercise 15 (★★, research thinking).** Design an evaluation that would convince a sceptical pharmacologist that MV-HGAT finds *new* indications, not just re-discovers benchmark links. Name at least three components.

<details><summary>Solution</summary>

1. **Temporal hold-out:** train on indications known up to year Y (e.g. using the 2011 Fdataset labels), test on indications approved *after* Y (from DrugBank/FDA labels or the newer Cdataset links) — the cross-dataset experiment is a first step.
2. **Cold-start protocols** (LODO, and leave-one-drug-out) so that the model cannot rely on the held-out entity's own links.
3. **Per-disease ranking metrics** with their random baselines (MRR, Hits@10), because that is how predictions will be used.
4. **Blinded external validation** of the top-k list against literature / trials, with the validator not knowing which model produced which list, and with a random or baseline-method list as control (to estimate the background rate of "support" — many drugs have *some* trial in cancer).
5. **Audit of the evidence source** (Section 5.6) and pre-registration of which diseases will be case-studied, to avoid picking the diseases where the model happens to look good.
</details>

---

## 8. Answers to the PREREQUISITES.md self-check questions (Unit B2)

### Q1. "A model has AUC 0.93 but AUPR 0.30. Is it good? Compared with what?"

It depends entirely on the baselines and on the competition, and the question "compared with what?" is the heart of the answer.

**Against random.** AUC's random value is always 0.5, so 0.93 is far from random. AUPR's random value is the positive rate. On Fdataset (π = 0.0104), 0.30 is about **29× better than random** — strong signal. If the same AUPR were obtained on a balanced dataset (π = 0.5), it would be *worse than random*. So the AUPR number is uninterpretable until you state π.

**What the pair of numbers says about the ranking.** AUC 0.93: a random known indication outranks a random unknown pair 93 % of the time. AUPR 0.30: averaged over the positions of the true indications, about 30 % of the pairs ranked at or above them are true. Using the precision formula, at a threshold where TPR ≈ 0.6 and FPR ≈ 1 %, precision on Fdataset is ≈ 0.39; the gap between a high AUC and a modest AUPR is simply what 99 : 1 imbalance does (Section 3.8). The model is useful for **prioritising** candidates, but a long list taken at face value will contain many false positives (some of which may be undiscovered positives — Section 3.13).

**Against other methods on the same test set.** In our Fdataset table, AUPR 0.30 would sit near MBiRW (0.31) — below MV-HGAT and SCMFDD (≈ 0.49), well above NIMCGCN (0.10). So it would be a decent but not state-of-the-art model. And because AUC and AUPR can rank methods differently when curves cross (MV-HGAT vs SCMFDD), you would also want per-disease metrics (MRR, Hits@10) and a paired significance test across folds.

**Against the use case.** If a lab will test 10 drugs per disease, the relevant number is per-disease precision@10 / Hits@10, which neither AUC nor pooled AUPR directly measures.

*Short answer:* "Yes, probably good — ~29× better than random on AUPR if π ≈ 1 % — but 'good' only means something relative to the positive rate, to competing methods evaluated on identical splits, and to the per-disease top-k performance that the application needs."

### Q2. "Why is leave-one-disease-out harder than 5-fold CV?"

1. **Information removed.** In 5-fold CV a test pair's disease still has ~80 % of its known drugs visible; in LODO it has **none**. Every signal that flows through the disease's own links disappears: collaborative filtering ("diseases treated by the same drugs"), drug-side similarity propagation ("drugs similar to drug $i$ treat disease $j$" — exactly zero for an empty column; Section 4.9 shows such a scorer drops to AUC 0.500), matrix-factorisation disease factors learned from the column, the GNN's message passing along `assoc` edges, and the disease's degree/popularity.
2. **Only side information remains.** The model must rely on disease–disease similarity (phenotype, ontology, genes), which is noisier and weaker. Our disease-side scorer alone reaches AP ≈ 0.17 on Fdataset LODO, versus ≈ 0.49 for the full model in warm CV.
3. **Distribution shift.** Models trained mostly on warm examples learn to lean on collaborative signals; at test time those are absent. MV-HGAT counters this with cold-start practice and the degree gate.
4. **It matches the real question.** For a newly characterised disease or one with no approved drug, LODO is the realistic scenario; warm CV mostly measures how well a method fills gaps in well-studied regions of the matrix.

(Subtlety: LODO is not harder for *every* signal — a pure disease-side method is unaffected, Section 4.9.)

### Q3. "What does 'the test set contains all unknown pairs' assume about unknowns?"

It assumes that **every unknown pair is a true negative** — that if a drug–disease pair is not in the benchmark, the drug does not treat the disease. That is false in general: benchmarks are incomplete (Fdataset dates from 2011; later approvals, off-label uses and uncurated indications are all 0s). The consequences, made precise in Section 3.13:

- Some "negatives" in the test set are hidden positives. When the model ranks them highly — which is exactly what a good repositioning model *should* do — they are counted as **false positives**.
- **AUC is barely affected** ($\text{AUC}_\text{meas} = (1-q)\text{AUC}_\text{true} + q/2$ with small $q$), but **AP and precision@k are biased downward**, potentially by a lot, because hidden positives concentrate at the top.
- Reported scores are therefore **conservative** (pessimistic) estimates of true performance, and comparisons between methods can be distorted if methods differ in how they rank the undiscovered links or if the missing links are not random (e.g. concentrated in well-studied drugs).
- It also affects training (sampled negatives may be positives), which is the positive–unlabelled learning problem.
- The partial remedy is external validation of top predictions (case studies, cross-dataset tests), and the right wording in the paper: "unknown pairs are treated as negatives; precision-type metrics are lower bounds."

---

## 9. Summary and cheat sheet

```
CONFUSION MATRIX (threshold tau)        sklearn: confusion_matrix -> [[TN, FP], [FN, TP]]
  TPR = recall = sensitivity = TP/P     FPR = FP/N        specificity = TN/N = 1 - FPR
  precision = TP/(TP+FP)                F1 = 2TP/(2TP+FP+FN)       accuracy = (TP+TN)/(P+N)

PRECISION vs BASE RATE (Bayes)
  precision = pi*TPR / (pi*TPR + (1-pi)*FPR)          TPR, FPR do not depend on pi; precision does

ROC / AUC                                             random = 0.5 (any pi)
  AUC = P(s+ > s-) + 0.5 P(s+ = s-)  = U / (P N)      (Mann-Whitney)
  rank-sum: AUC = (R+ - P(P+1)/2) / (P N)
  invariant to class ratio and to monotone score transforms -> optimistic when N >> P

PR / AP                                               random ~ pi  (exact: (P-1)/(n-1) + H_n/n (n-P)/(n-1))
  AP = (1/P) * sum over positives of precision@(its rank)   = sklearn average_precision_score
  trapezoid auc(recall, precision) != AP; never mix
  ROC dominance <=> PR dominance; AUC and AP disagree only when curves cross

RANKING (per disease, then average)
  P@k = hits_in_top_k / k     R@k = hits_in_top_k / P_j     Hits@k = 1[hits_in_top_k > 0]
  RR = 1/rank of first hit    MRR = mean RR                 MAP = mean per-disease AP

PROTOCOLS
  warm k-fold over cells: both ends of a test pair have other links in training
  LODO (cold): whole disease column hidden; only disease-side information survives
  10-fold > 5-fold (more training data); never compare across protocols

AVERAGING
  per-fold mean +/- std | pooled (mixes fold-models; calibration matters) | per-disease (use case)

UNKNOWN = NEGATIVE (PU)
  AUC_meas = (1-q) AUC_true + q/2   (q = hidden positives / labelled negatives)
  AP, P@k biased DOWN (hidden positives sit at the top) -> reported precision is a lower bound

FAIR COMPARISON
  same splits, same data, tuned on validation; paired differences; state ddof
  folds are dependent: Nadeau-Bengio t = dbar / sqrt((1/J + n_test/n_train) * s_d^2)

PROJECT NUMBERS (Fdataset cv5): pi = 0.0104 -> random AP 0.0104
  MV-HGAT AUC 0.939 / AP 0.488 (47x)    SCMFDD 0.893 / 0.495 (48x)   -> AUC differs, AP ties
  per-disease: MV-HGAT MRR 0.495, Hits@10 0.668  vs  SCMFDD 0.390, 0.546
```

---

## 10. Curated further resources

All links were checked on 1 October 2026 (journal pages that block automated checks were verified through their DOI). "Free" = readable without a subscription.

**Core papers on ROC / PR**
- Fawcett, T. (2006). *An introduction to ROC analysis.* Pattern Recognition Letters 27:861–874. [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010) — the standard tutorial: construction, convex hull, averaging ROC curves. *Paid (preprints circulate).*
- Saito, T. & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLOS ONE 10:e0118432. [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) — the clearest argument for PR under imbalance, with simulations like Section 4.4. *Free.*
- Davis, J. & Goadrich, M. (2006). *The relationship between precision-recall and ROC curves.* ICML. [PDF](https://www.biostat.wisc.edu/~page/rocpr.pdf) · [doi:10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874) — proves ROC/PR dominance equivalence and why linear PR interpolation is wrong. *Free PDF.*
- Hanley, J. A. & McNeil, B. J. (1982). *The meaning and use of the area under a ROC curve.* Radiology 143:29–36. [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747) — AUC = Wilcoxon statistic; standard-error formula. *Paid.*
- Boyd, K., Eng, K. H. & Page, C. D. (2013). *Area under the precision-recall curve: point estimates and confidence intervals.* ECML-PKDD. [doi:10.1007/978-3-642-40994-3_29](https://doi.org/10.1007/978-3-642-40994-3_29) — compares AUPR estimators (AP vs trapezoid vs interpolated) and their CIs. *Paid.*
- McDermott, M. et al. (2024). *A closer look at AUROC and AUPRC under class imbalance.* NeurIPS 2024. [arXiv:2401.06091](https://arxiv.org/abs/2401.06091) — the important counterpoint: when AUPRC is *not* the better choice. *Free.*
- Flach, P. & Kull, M. (2015). *Precision-Recall-Gain curves: PR analysis done right.* NeurIPS. [link](https://papers.nips.cc/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html) — a principled fix for PR's baseline problems. *Free.*

**Evaluation protocols for pair prediction**
- Park, Y. & Marcotte, E. M. (2012). *Flaws in evaluation schemes for pair-input computational predictions.* Nature Methods 9:1134–1136. [doi:10.1038/nmeth.2259](https://doi.org/10.1038/nmeth.2259) — why warm pairs inflate results; the origin of "C1/C2/C3" test classes. *Paid.*
- Pahikkala, T. et al. (2015). *Toward more realistic drug–target interaction predictions.* Briefings in Bioinformatics 16:325–337. [doi:10.1093/bib/bbu010](https://doi.org/10.1093/bib/bbu010) — the same lesson for drug–target prediction; four CV settings. *Paid.*
- Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4:100804. [doi:10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804) — taxonomy of leakage; read before Unit E1. *Free.*
- Varma, S. & Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics 7:91. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91) — why tuning on test folds inflates scores; nested CV. *Free.*

**Comparing methods statistically**
- Dietterich, T. G. (1998). *Approximate statistical tests for comparing supervised classification learning algorithms.* Neural Computation 10:1895–1923. [doi:10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197) — why naive CV t-tests are anti-conservative; 5×2cv test. *Paid.*
- Nadeau, C. & Bengio, Y. (2003). *Inference for the generalization error.* Machine Learning 52:239–281. [link](https://link.springer.com/article/10.1023/A:1024068626366) — the corrected resampled t-test of Section 3.14.3. *Paid.*
- Bengio, Y. & Grandvalet, Y. (2004). *No unbiased estimator of the variance of K-fold cross-validation.* JMLR 5:1089–1105. [link](https://jmlr.org/papers/v5/grandvalet04a.html) — the theorem behind "folds are not independent". *Free.*
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets.* JMLR 7:1–30. [link](https://jmlr.org/papers/v7/demsar06a.html) — Wilcoxon, Friedman and post-hoc tests for many methods × datasets. *Free.*

**Positive–unlabelled learning ("unknown = negative")**
- Elkan, C. & Noto, K. (2008). *Learning classifiers from only positive and unlabeled data.* KDD. [doi:10.1145/1401890.1401920](https://doi.org/10.1145/1401890.1401920) — the classic PU paper. *Paid.*
- Bekker, J. & Davis, J. (2020). *Learning from positive and unlabeled data: a survey.* Machine Learning 109:719–760. [link](https://link.springer.com/article/10.1007/s10994-020-05877-5) — modern survey including evaluation under PU. *Free (open access).*

**Textbooks, courses and documentation**
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 8 "Evaluation in information retrieval". [online](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — precision@k, MAP, interpolated AP, explained for ranked lists. *Free.*
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 7 (model assessment, cross-validation). [online](https://hastie.su.domains/ElemStatLearn/) — the rigorous treatment of CV. *Free PDF.*
- scikit-learn User Guide, "Metrics and scoring". [link](https://scikit-learn.org/stable/modules/model_evaluation.html) — exact definitions used by the project code (AP, ROC, partial AUC, tie handling). *Free.*
- scikit-learn example, "Precision-Recall". [link](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) — shows AP vs PR curve visually. *Free.*
- Google Machine Learning Crash Course, "ROC and AUC". [link](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) — interactive, very gentle intuition. *Free.*
- StatQuest (Josh Starmer), "ROC and AUC, Clearly Explained!" [video](https://www.youtube.com/watch?v=4jRBRDbJemM) — 16-minute visual walk-through. *Free.*

---

## 11. Glossary

- **AP (average precision)** — mean of the precision values at the ranks of the positives; a step-wise area under the PR curve. What `average_precision_score` and this project's "AUPR" compute.
- **AUC / AUROC** — area under the ROC curve; the probability a random positive outranks a random negative (ties ½).
- **AUPR / AUPRC** — area under the precision–recall curve; estimated by AP (preferred) or by a trapezoid (biased).
- **Base rate / prevalence / positive rate (π)** — fraction of test items that are positive.
- **Calibration** — agreement between scores and observed frequencies; independent of ranking quality.
- **Cold start** — predicting for an entity (disease or drug) with no known links in training.
- **Confusion matrix** — 2×2 table of TP, FP, FN, TN at a given threshold.
- **Cross-dataset validation** — testing a model trained on one benchmark against links found only in another.
- **F1 / F$_\beta$** — harmonic-type means of precision and recall.
- **FPR (false positive rate)** — FP/N; fraction of negatives flagged.
- **Hits@k** — 1 if any true item is in the top k (averaged over queries).
- **LODO** — leave-one-disease-out; hide every known drug of one disease, predict them, repeat.
- **Lift** — metric value divided by its random baseline (e.g. AP/π).
- **Mann–Whitney U** — rank statistic counting (positive, negative) pairs ordered correctly; AUC = U/(PN).
- **MAP** — mean over queries of per-query AP.
- **MRR** — mean over queries of 1/(rank of first true item).
- **Nadeau–Bengio correction** — variance inflation $(1/J + n_\text{test}/n_\text{train})$ for t-tests on resampled CV scores.
- **Paired comparison** — comparing two methods split by split on identical test sets.
- **Partial AUC** — ROC area restricted to a low-FPR range, optionally standardised.
- **Pooled (micro) metric** — one metric over the concatenation of all folds' predictions.
- **Per-query (macro) metric** — metric computed per disease (or fold), then averaged.
- **Precision (PPV)** — TP/(TP+FP); fraction of flagged items that are positive.
- **Precision@k / Recall@k** — precision / recall among the top k of a ranking.
- **PR curve** — precision versus recall as the threshold is lowered; random baseline = π.
- **PU learning** — learning from positive and unlabelled data; the formal name for "unknown = negative".
- **Random baseline** — the expected metric of a ranker that ignores the data (AUC 0.5; AP ≈ π).
- **ROC curve** — TPR versus FPR as the threshold is lowered; random baseline = diagonal.
- **Specificity (TNR)** — TN/N.
- **Stratified split** — folds built so each has the same class proportions.
- **Threshold (τ)** — score cut-off turning a ranking into yes/no decisions.
- **TPR / recall / sensitivity** — TP/P; fraction of positives found.
- **Warm start** — test pairs whose drug and disease both have other links in training.
