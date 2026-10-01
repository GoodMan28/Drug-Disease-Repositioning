# Unit A3 - Probability and Statistics for Link Prediction

> **Course:** Drug Repositioning with Graph Neural Networks - Track A (Foundations)
> **Prerequisites:** Unit A1 (Python, NumPy, pandas) - you should be able to read array code such as `A.ravel()`, boolean masks and `np.flatnonzero`. Unit A2 (Linear algebra) is helpful but only used lightly (dot products, vectors). High-school calculus is assumed: derivatives, the chain rule, and the facts that $\frac{d}{dx}\ln x = 1/x$ and $\frac{d}{dx}e^x = e^x$.
> **Estimated study time:** 12-16 hours (about 6 hours of reading, 4 hours of running and modifying the code labs, 3-5 hours of exercises).

---

## Learning objectives

After this unit you will be able to:

1. **Define** a random variable, its distribution, expectation, variance and standard deviation, and **compute** them by hand for small discrete examples.
2. **State** the Bernoulli and binomial distributions, **derive** their means and variances, and **use** them to model "is drug *i* indicated for disease *j*?".
3. **Convert** between probabilities, odds and log-odds (logits), and **explain** why the model's last step is a sigmoid.
4. **Write down** the likelihood and log-likelihood of a data set under a Bernoulli model, and **derive** the maximum-likelihood estimate.
5. **Derive** binary cross-entropy (BCE) from the Bernoulli likelihood, **derive** its gradient with respect to the logit, and **explain** why `binary_cross_entropy_with_logits` is numerically safer than applying `sigmoid` then `log`.
6. **Explain** how negative sampling (2 negatives per positive) shifts the model's probabilities, and why it does not change AUC or AUPR.
7. **Distinguish** population vs sample quantities, standard deviation vs standard error, `ddof=0` vs `ddof=1`, and **report** results correctly as mean ± SD with the number of runs.
8. **Construct and interpret** a confidence interval (t-interval, bootstrap), and **explain** why folds of cross-validation are not independent.
9. **Compute** the performance of a random ranker under a 1% base rate (AUC ≈ 0.5, AUPR ≈ 0.0104) and use it as a reference point.
10. **Carry out and interpret** paired comparisons of two models across the same folds (paired t-test, Wilcoxon, sign test, permutation test, corrected resampled t-test, Holm correction).

---

## 1. Motivation: why this unit matters for this project

Open `results/RESULTS.md`. The first table reads:

| Method | AUC | AUPR |
|---|---|---|
| MV-HGAT (ours) | 0.9392 ± 0.0072 | 0.4878 ± 0.0273 |
| SCMFDD | 0.8934 ± 0.0102 | 0.4946 ± 0.0199 |

Every symbol in that table is a probability or statistics concept:

* Each number in a cell is a **mean** over 25 runs (5 repeats × 5 folds), and the number after ± is a **standard deviation**. Is it the right kind of standard deviation? (It is computed with `ddof=0`; section 7 explains what that means.)
* AUPR = 0.4878 sounds mediocre ("less than half"), but a model that guesses at random would score about **0.0104**, because only 1.04% of drug-disease pairs are known links. MV-HGAT is therefore about **47 times better than chance** on this metric. Without the idea of a **base rate** (section 9) you cannot read the table.
* SCMFDD has a slightly *higher* mean AUPR than MV-HGAT. Is that a real difference or noise? Answering this needs **hypothesis testing with paired data** (section 10). Spoiler: on AUPR the two are statistically indistinguishable on Fdataset, while on AUC MV-HGAT is clearly better. That is the kind of honest statement a good paper makes.

The model itself is built on probability too. In `src/drepo/methods.py`, `MVHGATMethod.fit_predict` ends with

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict)
loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
...
return torch.sigmoid(logits).cpu().numpy()
```

The model produces a **logit** (a real number) for each (drug, disease) pair, the **sigmoid** turns it into a number between 0 and 1, and training minimises **binary cross-entropy**, which is nothing but the negative **log-likelihood** of a **Bernoulli** model. By the end of this unit you will be able to derive that loss on a blank sheet of paper, explain each term, and say exactly what the output "probabilities" do and do not mean (because of negative sampling they are *not* calibrated probabilities of a true indication - section 6.6).

---

## 2. Probability: the language of uncertainty

### 2.1 Intuition

Probability is a bookkeeping system for uncertainty. When we say "the probability that metformin treats disease X is 0.3", we mean: given everything we know, if we considered many situations that look like this one, about 30% of them would turn out to be true indications. Probability lets us combine uncertain pieces of evidence consistently.

There are two common interpretations, and you will meet both:

* **Frequentist:** probability is the long-run frequency of an outcome in repeated experiments. "The probability a random cell of `A` is 1 is 0.0104" means if you pick cells at random over and over, about 1.04% of picks are 1s.
* **Bayesian:** probability is a degree of belief that is updated by evidence. "I'm 30% sure this drug works" is a Bayesian statement about a single case.

The mathematics is the same for both; only the interpretation differs. In this unit, model outputs are best read in the Bayesian way ("how confident is the model?") and evaluation statistics in the frequentist way ("what would happen across repeated splits?").

### 2.2 Sample spaces, events and the axioms

* An **experiment** is any process with an uncertain outcome (draw a random cell of `A`; shuffle links into folds; initialise a network's weights).
* The **sample space** $\Omega$ is the set of all possible outcomes. For "draw one cell of the 593 × 313 Fdataset matrix", $\Omega$ has $593 \times 313 = 185{,}609$ elements.
* An **event** is a subset of $\Omega$, e.g. $E$ = "the drawn cell is a known link" (1,933 elements).
* A **probability** $P$ assigns a number to each event and satisfies three axioms (Kolmogorov):
  1. $P(E) \ge 0$ for every event $E$;
  2. $P(\Omega) = 1$;
  3. if $E_1, E_2, \dots$ are mutually exclusive (no two can happen together), $P(E_1 \cup E_2 \cup \dots) = P(E_1) + P(E_2) + \dots$

Everything else follows. For example, $P(\text{not } E) = 1 - P(E)$, because $E$ and "not $E$" are mutually exclusive and together make up $\Omega$.

If all outcomes are equally likely (a uniformly random cell), $P(E) = |E| / |\Omega|$. So

$$P(\text{random cell is a known link}) = \frac{1933}{185{,}609} = 0.010414\ldots$$

We will call this number the **base rate** or **prevalence** and write it $\pi$.

### 2.3 Conditional probability and independence

The **conditional probability** of $A$ given $B$ is

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad P(B) > 0.$$

Read it as: "restrict attention to the outcomes where $B$ happened, and ask what fraction of those also have $A$." Rearranging gives the **product rule** $P(A \cap B) = P(A \mid B)\,P(B)$.

Two events are **independent** if $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$: knowing $B$ tells you nothing about $A$.

*Project example.* Let $A$ = "cell $(i, j)$ is a known link" and $B$ = "drug $i$ has at least 10 known indications". These are **not** independent: drugs with many indications (e.g. broad anti-inflammatories) are more likely to have a link in any given column. This dependence is exactly why the degree gate in `model.py` (`MVHGAT.gnn_gate`) uses the number of visible links as an input.

### 2.4 Bayes' theorem, and your first meeting with base rates

From the product rule written both ways, $P(A \mid B)P(B) = P(B \mid A)P(A)$, so

$$\boxed{P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}}, \qquad P(B) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A).$$

**Worked example (by hand).** Suppose a model flags a drug-disease pair as "predicted indication" with these properties: it flags 90% of true links (sensitivity, $P(\text{flag} \mid \text{link}) = 0.9$) and wrongly flags 5% of non-links ($P(\text{flag} \mid \text{no link}) = 0.05$). That sounds excellent. If the model flags a pair, what is the probability it really is a link?

With base rate $\pi = 0.010414$:

$$P(\text{flag}) = 0.9 \times 0.010414 + 0.05 \times 0.989586 = 0.009373 + 0.049479 = 0.058852$$

$$P(\text{link} \mid \text{flag}) = \frac{0.009373}{0.058852} = 0.159.$$

Only **16%** of the flagged pairs are real links, even though the model catches 90% of links and has a "low" 5% false-alarm rate. The reason: there are 95 times more non-links than links, so 5% of a huge number dwarfs 90% of a small number. This quantity, $P(\text{link} \mid \text{flag})$, is called **precision**, and it is what AUPR summarises. Keep this example in mind; it is the single most important intuition for evaluating this project (Unit B2 develops it further).

### 2.5 Random variables and distributions

A **random variable** (r.v.) is a function that attaches a number to each outcome. Examples:

* $Y$ = 1 if a randomly drawn cell is a link, 0 otherwise (a **discrete** r.v. taking values in $\{0, 1\}$).
* $X$ = the number of links in a random test fold (discrete, values $0, 1, 2, \dots$).
* $S$ = the AUPR you get from one run of 5-fold CV with a random seed (continuous, values in $[0, 1]$).

A discrete r.v. is described by its **probability mass function** (PMF) $p(x) = P(X = x)$, with $\sum_x p(x) = 1$. A continuous r.v. is described by a **probability density function** (PDF) $f(x)$, where probabilities are areas: $P(a \le X \le b) = \int_a^b f(x)\,dx$. Both have a **cumulative distribution function** (CDF) $F(x) = P(X \le x)$.

Important subtlety: for a continuous r.v., $f(x)$ is *not* a probability; it can be larger than 1. Only areas under $f$ are probabilities.

### 2.6 Expectation

The **expectation** (mean) of an r.v. is its probability-weighted average:

$$E[X] = \sum_x x\,p(x) \quad \text{(discrete)}, \qquad E[X] = \int x f(x)\,dx \quad \text{(continuous)}.$$

Write $\mu = E[X]$. Intuitively, it is the balance point of the distribution, and the long-run average of many independent draws (law of large numbers, section 7.4).

**Linearity of expectation** - the most useful fact in probability: for any r.v.s $X, Y$ (independent or not) and constants $a, b$,

$$E[aX + bY] = aE[X] + bE[Y].$$

*Proof (discrete case):* $E[aX+bY] = \sum_{x,y}(ax + by)p(x,y) = a\sum_x x \sum_y p(x,y) + b \sum_y y \sum_x p(x,y) = a\sum_x x p(x) + b \sum_y y p(y)$. ∎

Also, for a function $g$: $E[g(X)] = \sum_x g(x) p(x)$. In general $E[g(X)] \ne g(E[X])$ (for example $E[X^2] \ne (E[X])^2$).

### 2.7 Variance, standard deviation, covariance

The **variance** measures spread around the mean:

$$\mathrm{Var}(X) = E\big[(X - \mu)^2\big] = E[X^2] - \mu^2.$$

*Proof of the second form:* $E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$. ∎

The **standard deviation** is $\mathrm{SD}(X) = \sigma = \sqrt{\mathrm{Var}(X)}$. It has the same units as $X$ (variance has squared units), which is why we report SD.

Rules (for constants $a, b$):

* $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$ - shifting does not change spread; scaling by $a$ scales SD by $|a|$.
* $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$,
* $\mathrm{Var}(X - Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) - 2\,\mathrm{Cov}(X, Y)$,

where the **covariance** is $\mathrm{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$ and the **correlation** is $\rho = \mathrm{Cov}(X,Y) / (\sigma_X \sigma_Y) \in [-1, 1]$. If $X$ and $Y$ are independent, $\mathrm{Cov}(X, Y) = 0$ and variances simply add.

The minus-covariance formula is the mathematical heart of **paired comparisons** (section 10): if two models' scores on the same fold are positively correlated (both do well on "easy" folds), the variance of their *difference* is smaller than the sum of their variances, so a paired test is more sensitive.

---

## 3. The Bernoulli and binomial distributions

### 3.1 Bernoulli: one yes/no outcome

A random variable $Y \in \{0, 1\}$ has a **Bernoulli distribution** with parameter $p \in [0, 1]$, written $Y \sim \mathrm{Bernoulli}(p)$, if $P(Y = 1) = p$ and $P(Y = 0) = 1 - p$.

A compact way to write the PMF, which will be crucial for the likelihood:

$$\boxed{P(Y = y) = p^{y}(1 - p)^{1 - y}, \qquad y \in \{0, 1\}.}$$

Check: for $y=1$ this is $p^1 (1-p)^0 = p$; for $y = 0$ it is $p^0(1-p)^1 = 1 - p$. The exponent acts as a switch that selects the right factor.

**Mean and variance.**

$$E[Y] = 0\cdot(1-p) + 1\cdot p = p.$$

Since $Y \in \{0,1\}$, $Y^2 = Y$, so $E[Y^2] = p$ and

$$\mathrm{Var}(Y) = E[Y^2] - (E[Y])^2 = p - p^2 = p(1 - p).$$

The variance is largest at $p = 0.5$ (value 0.25) and tiny when $p$ is near 0 or 1. At the base rate $\pi = 0.0104$, $\mathrm{Var} = 0.0104 \times 0.9896 = 0.0103$.

**Modelling choice in this project.** For each drug $i$ and disease $j$ we treat the label $A_{ij}$ as a Bernoulli r.v. whose parameter $p_{ij}$ depends on the pair. The model's job is to output a good estimate $\hat p_{ij}$. A crucial caveat (see Unit D1 and HOW_IT_WORKS §6): $A_{ij} = 0$ means "not known", not "known not to work". So the label we model is really "is this pair a *recorded* indication", which is a positive-unlabelled (PU) problem. The statistics in this unit are still the right tools, but remember that some "negatives" are undiscovered positives.

### 3.2 Binomial: counting successes

If $Y_1, \dots, Y_n$ are **independent** Bernoulli($p$) r.v.s, their sum $X = \sum_{t=1}^n Y_t$ counts the number of 1s. $X$ has a **binomial distribution**, $X \sim \mathrm{Binomial}(n, p)$, with PMF

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, \dots, n,$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ counts the ways to choose which $k$ of the $n$ trials are successes, and $p^k(1-p)^{n-k}$ is the probability of any one such arrangement.

By linearity of expectation and independence:

$$E[X] = \sum_t E[Y_t] = np, \qquad \mathrm{Var}(X) = \sum_t \mathrm{Var}(Y_t) = np(1-p).$$

**Worked example.** One test fold of 5-fold CV on Fdataset contains 387 positive cells and 36,735 negative cells, 37,122 in total. Suppose instead we had drawn 37,122 cells uniformly at random (not stratified). How many positives would we expect, and how much would that number vary?

$$E[X] = 37{,}122 \times 0.010414 = 386.6, \qquad \mathrm{SD}(X) = \sqrt{37{,}122 \times 0.010414 \times 0.989586} = 19.56.$$

So a random (unstratified) fold would typically contain between about 347 and 426 positives ($\pm 2$ SD). `evaluation.py::kfold_splits` avoids this variation by splitting the positives and the negatives **separately** (`np.array_split(pos, k)` and `np.array_split(neg, k)`), which guarantees 386 or 387 positives in every fold. This is called **stratification** (Unit B1 returns to it). Lab 1 checks these numbers by simulation.

*(Technical footnote: when you draw without replacement from a finite matrix the exact distribution is hypergeometric, whose variance is smaller by the factor $(N - n)/(N - 1)$. With $N = 185{,}609$ and $n = 37{,}122$ that factor is 0.8, so the SD would be about 17.5. The binomial is the simpler, standard approximation.)*

### 3.3 Other distributions you will meet

* **Categorical / multinomial:** the generalisation of Bernoulli/binomial to more than two outcomes. The softmax in the attention layers produces categorical distributions over neighbours or views.
* **Normal (Gaussian)** $\mathcal N(\mu, \sigma^2)$: density $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}$. About 68% of the mass is within $\mu \pm \sigma$ and 95% within $\mu \pm 1.96\sigma$. It appears through the central limit theorem (section 7.4).
* **Student's t** with $\nu$ degrees of freedom: like the normal but with heavier tails; it describes the standardised sample mean when the SD is estimated from a small sample (section 8).
* **Uniform** $U(0,1)$: `rng.random()`. A "random ranker" scores each pair with an independent uniform draw (section 9).

---

## 4. Odds, logits and the sigmoid

### 4.1 Why not output a probability directly?

A neural network's last linear layer outputs an unconstrained real number $z \in (-\infty, \infty)$. A probability must live in $[0, 1]$. We need a smooth, monotone map from the real line onto $(0, 1)$, preferably one with a meaningful interpretation and nice derivatives. The standard choice is the **logistic sigmoid**.

### 4.2 Odds and log-odds

The **odds** of an event with probability $p$ are $\frac{p}{1-p}$. Probability 0.5 is odds 1 ("one to one"); probability 0.2 is odds 0.25 ("one to four"); probability 0.0104 is odds 0.0105.

Odds live in $[0, \infty)$. Taking the logarithm gives the **log-odds**, or **logit**:

$$\mathrm{logit}(p) = \ln\frac{p}{1-p} \in (-\infty, \infty).$$

Log-odds are symmetric: $\mathrm{logit}(1 - p) = -\mathrm{logit}(p)$. $p = 0.5$ maps to 0, probabilities above 0.5 to positive numbers, below 0.5 to negative numbers.

### 4.3 The sigmoid is the inverse of the logit

Solve $z = \ln\frac{p}{1-p}$ for $p$: $e^z = \frac{p}{1-p} \Rightarrow e^z - p e^z = p \Rightarrow p = \frac{e^z}{1 + e^z}$, i.e.

$$\boxed{\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}.}$$

So "the model outputs a logit" means: the number $z$ the network computes is interpreted as the **log-odds** that the pair is a link, and $\sigma(z)$ converts it to a probability. A logit of 0 means 50%; each +1 multiplies the odds by $e \approx 2.718$.

**Key properties (prove each - they are short):**

1. $\sigma(-z) = 1 - \sigma(z)$. *Proof:* $1 - \frac{1}{1+e^{-z}} = \frac{e^{-z}}{1+e^{-z}} = \frac{1}{e^{z}+1} = \sigma(-z)$.
2. $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$. *Proof:* $\frac{d}{dz}(1 + e^{-z})^{-1} = (1+e^{-z})^{-2} e^{-z} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1 - \sigma(z))$.
3. $\ln \sigma(z) = -\ln(1 + e^{-z}) = -\mathrm{softplus}(-z)$, where $\mathrm{softplus}(u) = \ln(1 + e^u)$. And $\ln(1 - \sigma(z)) = \ln\sigma(-z) = -\mathrm{softplus}(z)$.

Property 2 says the slope is largest (0.25) at $z = 0$ and vanishes for large $|z|$: the sigmoid saturates. Property 3 is what makes the loss numerically stable (section 6.5).

**Worked values (Lab 2 reproduces them).**

| $z$ | $\sigma(z)$ | odds $e^z$ |
|---|---|---|
| -6 | 0.0025 | 0.0025 |
| -3 | 0.0474 | 0.0498 |
| -1 | 0.2689 | 0.3679 |
| 0 | 0.5000 | 1 |
| +1 | 0.7311 | 2.718 |
| +3 | 0.9526 | 20.09 |

*Project connection:* in `src/drepo/model.py`, `MVHGAT.__init__` sets `self.bias = nn.Parameter(torch.tensor(-3.0))`. At initialisation, before the network has learned anything, a pair with zero propagation evidence gets logit about $-3$, i.e. probability $\sigma(-3) = 0.047$ - a sensible "most pairs are not links" starting point. The same file uses sigmoids inside `gnn_gate`: `torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))` is a number between 0 and 1 that grows with a drug's degree, acting as a soft on/off switch.

### 4.4 Logistic regression: the simplest model that outputs a logit

If the logit is a linear function of features $x$, $z = w^\top x + b$, and $P(Y=1 \mid x) = \sigma(z)$, the model is called **logistic regression**. MV-HGAT is "logistic regression on top of learned features": its decoder computes $z_{ij} = h_i^\top W h_j + \sum_v w_v P_v[i,j] + b$ and the final probability is $\sigma(z_{ij})$. Everything we derive for logistic regression's loss applies to it unchanged.

---

## 5. Likelihood and maximum likelihood

### 5.1 Probability versus likelihood

Consider the Bernoulli PMF $P(Y = y \mid p) = p^y (1-p)^{1-y}$. It is a function of two things: the data $y$ and the parameter $p$.

* Fix $p$ and vary $y$: you get a **probability distribution** over possible data. It sums to 1 over $y$.
* Fix the observed data $y$ and vary $p$: you get the **likelihood function** $L(p) = P(\text{data} \mid p)$. It tells you how well each candidate parameter value explains the data you actually saw. It is *not* a probability distribution over $p$ and need not integrate to 1.

So "likelihood" answers *"which parameter values make my observed data look plausible?"*

### 5.2 Likelihood of many independent observations

Suppose we observe labels $y_1, \dots, y_n$ that we model as independent Bernoulli($p$). Independence means the joint probability is a product:

$$L(p) = \prod_{t=1}^{n} p^{y_t}(1-p)^{1-y_t} = p^{k}(1-p)^{n-k}, \qquad k = \sum_t y_t.$$

Only the count $k$ matters - it is a **sufficient statistic**.

### 5.3 Why we take logs

Products of many numbers below 1 become astronomically small. Fdataset has $n = 185{,}609$ cells; the likelihood at the best $p$ is about $e^{-10{,}746}$, far below the smallest positive double-precision number ($\approx 10^{-308} \approx e^{-709}$). A computer evaluates it as exactly **0.0** (Lab 3 demonstrates this). The **log-likelihood** turns products into sums:

$$\ell(p) = \ln L(p) = \sum_{t=1}^n \big[y_t \ln p + (1-y_t)\ln(1-p)\big] = k\ln p + (n-k)\ln(1-p).$$

Because $\ln$ is strictly increasing, the $p$ that maximises $\ell$ also maximises $L$. Logs also make derivatives easy (the derivative of a sum is the sum of derivatives).

### 5.4 Maximum likelihood estimation (MLE)

The **maximum-likelihood estimate** is the parameter value that makes the observed data most probable:

$$\hat p_{\text{MLE}} = \arg\max_p \ell(p).$$

**Derivation for the Bernoulli.** Differentiate and set to zero:

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n - k}{1 - p} = 0 \;\Rightarrow\; k(1-p) = (n-k)p \;\Rightarrow\; k = np \;\Rightarrow\; \boxed{\hat p = \frac{k}{n}.}$$

Second derivative: $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$, so this stationary point is a maximum. The MLE is the sample proportion - the answer your intuition gave all along, now justified.

**Worked example (by hand).** In a toy fold, 3 of 10 cells are links. Then $\hat p = 0.3$ and
$\ell(0.3) = 3\ln 0.3 + 7 \ln 0.7 = 3(-1.2040) + 7(-0.3567) = -6.1086$.
Compare $\ell(0.5) = 10 \ln 0.5 = -6.9315$ and $\ell(0.2) = 3\ln 0.2 + 7 \ln 0.8 = -6.3903$. The value 0.3 indeed gives the highest (least negative) log-likelihood.

**Project example.** For Fdataset, $k = 1933$, $n = 185{,}609$, so $\hat p = 0.010414$ - the base rate. Lab 3 confirms by grid search that the log-likelihood peaks exactly there.

### 5.5 Properties of the MLE (what a statistics course would add)

Under mild regularity conditions and a correctly specified model:

* **Consistency:** $\hat\theta \to \theta$ as $n \to \infty$.
* **Asymptotic normality:** $\hat\theta$ is approximately normal with variance $1/I(\theta)$, where the **Fisher information** $I(\theta) = -E[\ell''(\theta)]$. For the Bernoulli, $I(p) = n/(p(1-p))$, so $\mathrm{SD}(\hat p) \approx \sqrt{p(1-p)/n}$ - the familiar standard error of a proportion.
* **Invariance:** the MLE of $g(\theta)$ is $g(\hat\theta)$. The MLE of the log-odds is $\mathrm{logit}(\hat p)$.
* **Efficiency:** asymptotically no unbiased estimator has smaller variance.

For neural networks the model has millions of parameters, the log-likelihood is not concave, and these guarantees no longer hold exactly - but the *principle* (choose parameters that make the observed labels most probable) is still what training does.

### 5.6 Conditional likelihood: when $p$ depends on the input

In supervised learning each example has its own probability, $p_t = \sigma(z_t)$ where $z_t = f_\theta(x_t)$ is computed by the model from the input $x_t$ with parameters $\theta$. The likelihood of all the labels given the inputs is

$$L(\theta) = \prod_t p_t^{y_t}(1-p_t)^{1-y_t}, \qquad \ell(\theta) = \sum_t \big[y_t\ln p_t + (1-y_t)\ln(1 - p_t)\big].$$

Maximising this over $\theta$ is **maximum conditional likelihood** estimation. There is no closed-form solution any more, so we use gradient methods (Unit B1). Lab 6 fits a logistic regression this way with PyTorch and shows it lands exactly where scikit-learn's solver lands.

### 5.7 A glimpse of MAP: where weight decay comes from

If we also put a **prior** distribution on the parameters and maximise the posterior $P(\theta \mid \text{data}) \propto P(\text{data} \mid \theta)P(\theta)$, we get **maximum a posteriori** (MAP) estimation. With a Gaussian prior $\theta \sim \mathcal N(0, \tau^2 I)$, $\ln P(\theta) = -\frac{\|\theta\|^2}{2\tau^2} + \text{const}$, so MAP minimises

$$-\ell(\theta) + \frac{1}{2\tau^2}\|\theta\|^2,$$

which is the negative log-likelihood plus an **L2 penalty**. That is the probabilistic reading of `weight_decay=5e-4` in `MVHGATConfig`. Unit B1 develops this.

---

## 6. Binary cross-entropy, derived from the Bernoulli likelihood

### 6.1 The derivation

We want a **loss** to *minimise*. Take the conditional log-likelihood from 5.6, negate it (maximising $\ell$ is minimising $-\ell$), and divide by the number of examples $N$ so that the loss does not grow with the data set size:

$$\boxed{\mathcal L_{\text{BCE}} = -\frac{1}{N}\sum_{t=1}^N \Big[y_t \ln \hat p_t + (1 - y_t)\ln(1 - \hat p_t)\Big], \qquad \hat p_t = \sigma(z_t).}$$

That is **binary cross-entropy** (also called **log loss** or the **negative log-likelihood** of a Bernoulli model). Every step is reversible, so:

> **Minimising BCE is exactly the same as maximum (conditional) likelihood estimation under the assumption that each label is an independent Bernoulli draw with probability $\sigma(z_t)$.**

Step by step:

1. Model: $Y_t \mid x_t \sim \mathrm{Bernoulli}(\hat p_t)$, independent across $t$.
2. One example's probability: $\hat p_t^{y_t}(1-\hat p_t)^{1-y_t}$.
3. All examples (independence): product over $t$.
4. Take $\ln$: sum of $y_t\ln\hat p_t + (1-y_t)\ln(1-\hat p_t)$.
5. Negate and average: BCE.

### 6.2 Reading the loss term by term

For a single example only one term survives:

* If $y = 1$: loss $= -\ln \hat p$. Confident and right ($\hat p = 0.9$): $-\ln 0.9 = 0.105$. Unsure ($\hat p = 0.5$): $0.693$. Confident and wrong ($\hat p = 0.1$): $2.303$. Very confident and wrong ($\hat p = 0.01$): $4.605$. As $\hat p \to 0$, loss $\to \infty$.
* If $y = 0$: loss $= -\ln(1 - \hat p)$, the mirror image.

BCE punishes confident mistakes very heavily. This is a feature: it forces the model to be cautious when the evidence is weak.

**Worked example (by hand; Lab 4 checks it).** Labels $y = (1, 1, 0, 0, 0, 0)$ (two positives and four sampled negatives, mimicking `neg_ratio=2`), logits $z = (2.0, -0.5, -1.0, 0.3, -3.0, -2.0)$.

| $t$ | $y$ | $z$ | $\hat p = \sigma(z)$ | term |
|---|---|---|---|---|
| 1 | 1 | 2.0 | 0.8808 | $-\ln 0.8808 = 0.1269$ |
| 2 | 1 | -0.5 | 0.3775 | $-\ln 0.3775 = 0.9741$ |
| 3 | 0 | -1.0 | 0.2689 | $-\ln 0.7311 = 0.3133$ |
| 4 | 0 | 0.3 | 0.5744 | $-\ln 0.4256 = 0.8544$ |
| 5 | 0 | -3.0 | 0.0474 | $-\ln 0.9526 = 0.0486$ |
| 6 | 0 | -2.0 | 0.1192 | $-\ln 0.8808 = 0.1269$ |

Mean $= (0.1269 + 0.9741 + 0.3133 + 0.8544 + 0.0486 + 0.1269)/6 = 2.4442/6 = 0.4074$. The two biggest contributions come from example 2 (a positive the model scores below 0.5) and example 4 (a negative scored above 0.5).

### 6.3 The information-theory view: entropy, cross-entropy, KL divergence

Why the name "cross-entropy"? For two distributions $q$ (the truth) and $r$ (the model) over outcomes $\{0, 1\}$:

* **Entropy** $H(q) = -\sum_y q(y)\ln q(y)$: the average "surprise" of outcomes drawn from $q$. Measured in nats (natural log) or bits ($\log_2$).
* **Cross-entropy** $H(q, r) = -\sum_y q(y) \ln r(y)$: the average surprise when outcomes come from $q$ but you predict with $r$.
* **Kullback-Leibler divergence** $\mathrm{KL}(q\|r) = H(q,r) - H(q) = \sum_y q(y)\ln\frac{q(y)}{r(y)} \ge 0$, with equality iff $q = r$ (Gibbs' inequality).

For one example with label $y$, the "true" distribution is the point mass $q = (1-y, y)$ and the model's is $r = (1-\hat p, \hat p)$. Then $H(q) = 0$ and

$$H(q, r) = -\big[y\ln\hat p + (1-y)\ln(1-\hat p)\big],$$

exactly the BCE term. So minimising BCE = minimising cross-entropy = minimising KL divergence from the empirical labels to the model.

**BCE is a proper scoring rule.** Suppose the true probability that a pair is a link is $q$. If the model predicts $r$, the expected loss is $-[q\ln r + (1-q)\ln(1-r)]$. Differentiating in $r$: $-q/r + (1-q)/(1-r) = 0 \Rightarrow r = q$. So the expected BCE is minimised by predicting the true probability. That is why a model trained with BCE on representative data tends to produce **calibrated** probabilities - *if* the training distribution matches the deployment distribution (see 6.6 for why ours does not).

### 6.4 The gradient: "prediction minus label"

For one example, with $\hat p = \sigma(z)$ and $\mathcal L = -[y\ln\hat p + (1-y)\ln(1-\hat p)]$, use the chain rule and $\sigma' = \sigma(1-\sigma)$:

$$\frac{\partial\mathcal L}{\partial z} = -\Big[\frac{y}{\hat p} - \frac{1-y}{1-\hat p}\Big]\hat p(1-\hat p) = -\big[y(1-\hat p) - (1-y)\hat p\big] = \boxed{\hat p - y.}$$

This is one of the most beautiful results in machine learning. The gradient with respect to the logit is simply **"predicted probability minus true label"**:

* a positive scored 0.38 gets gradient $-0.62$: gradient descent pushes its logit **up**;
* a negative scored 0.57 gets gradient $+0.57$: its logit is pushed **down**;
* a correctly confident prediction gets a gradient near 0: it is left alone.

The sigmoid's derivative cancelled. If you had used squared error on the probabilities instead, the factor $\sigma'(z)$ would remain, and for a confidently wrong prediction (large $|z|$, $\sigma' \approx 0$) learning would stall. That cancellation is a big reason BCE is the standard loss for binary outputs. For the mean over $N$ examples, each gradient is divided by $N$ (Lab 4 verifies with autograd).

### 6.5 Numerical stability: why the code uses `binary_cross_entropy_with_logits`

Computing $\sigma(z)$ first and then $\ln$ is dangerous in floating point. For $z = 800$, $\sigma(z)$ rounds to exactly 1.0, so $\ln(1 - \sigma(z)) = \ln 0 = -\infty$ and the loss is infinite (or NaN), even though the true value is a perfectly finite 800.

Using property 3 from 4.3, write the per-example loss directly in terms of $z$:

$$\mathcal L(z, y) = y\,\mathrm{softplus}(-z) + (1-y)\,\mathrm{softplus}(z) = \ln(1 + e^{z}) - yz.$$

*Check:* $y\ln(1+e^{-z}) + (1-y)\ln(1+e^{z})$; use $\ln(1+e^{-z}) = \ln(1+e^{z}) - z$, which gives $\ln(1+e^z) - yz$. ∎

But $e^{z}$ itself overflows for $z > 709$. The standard trick is $\ln(1 + e^{z}) = \max(z, 0) + \ln(1 + e^{-|z|})$, giving

$$\boxed{\mathcal L(z, y) = \max(z, 0) - zy + \ln\big(1 + e^{-|z|}\big),}$$

which never overflows because $e^{-|z|} \le 1$. PyTorch's `F.binary_cross_entropy_with_logits` and `nn.BCEWithLogitsLoss` implement this (the latter's documentation notes that combining the sigmoid and the BCE in one layer is more numerically stable than a sigmoid followed by `BCELoss`). This is why the project's training loop passes **logits**, and applies `torch.sigmoid` only at the very end, for reporting. Lab 5 shows the naive version producing `inf` where the stable one gives 800.

### 6.6 Negative sampling, weighting, and what the output "probability" means

MV-HGAT does not use all ~183,000 negative cells per epoch. In `fit_predict`:

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict)
n_neg = min(neg_pool.numel(), sup.numel() * c.neg_ratio) if c.neg_ratio > 0 \
    else neg_pool.numel()
neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]
```

With `neg_ratio = 2`, each epoch's loss sees one positive for every two negatives - a training base rate of $1/3$ instead of the true $\approx 1\%$.

**What does that do to the learned probabilities?** Use Bayes' theorem. Suppose we keep every positive and each negative independently with probability $s$ (here $s \approx 2 \times 1933 / 183{,}676$). For an input $x$, the odds of a link among the *kept* examples are

$$\frac{P(y=1 \mid x, \text{kept})}{P(y=0 \mid x, \text{kept})} = \frac{P(y=1\mid x)\cdot 1}{P(y=0\mid x)\cdot s} = \frac{1}{s}\times\text{true odds}.$$

Taking logs: **logit under sampling = true logit + $\ln(1/s)$**. Negative sampling adds a constant to every logit. Consequences:

* The raw outputs are **not** calibrated probabilities of a true link; they are inflated. For Fdataset the shift is about $\ln\big(0.5 / (1933/183{,}676)\big) = \ln(47.5) \approx 3.86$ nats: odds about 47× too high. (This is approximate for the project, because hidden-link supervision and the cold-start mask change which positives are used, but the 1:2 ratio is maintained.)
* Adding a constant does not change the **order** of scores, so ranking metrics - AUC, AUPR, precision@k - are unaffected. All of the project's metrics are ranking metrics, so this is harmless for evaluation.
* If you ever needed real probabilities (e.g., "what is the chance this prediction is right?"), subtract $\ln(1/s)$ from the logits, or recalibrate on held-out data (Platt scaling / isotonic regression).

Lab 7 demonstrates this on synthetic data: the fitted intercept shifts by $\ln(1/s)$ almost exactly, the slope is unchanged, and AUC/AUPR are identical.

**Weighted BCE.** The baselines NIMCGCN and LAGCN use `weighted_bce` in `methods.py`, which keeps *all* allowed cells but multiplies each positive's loss by `pw` = (number of negatives)/(number of positives). Probabilistically this is a **weighted likelihood** in which each positive counts as `pw` observations. Its effect on the logits is the same kind of constant shift (by $\ln \texttt{pw}$), so again rankings are unaffected while calibration is.

---

## 7. From populations to samples: estimators, variance and repeated runs

### 7.1 Population vs sample

So far $\mu$ and $\sigma^2$ were properties of a **distribution** (the "population"). In practice we see a **sample** of $n$ values $x_1, \dots, x_n$ and estimate:

* the **sample mean** $\bar x = \frac{1}{n}\sum_t x_t$;
* the **sample variance** $s^2 = \frac{1}{n-1}\sum_t (x_t - \bar x)^2$ and the sample SD $s = \sqrt{s^2}$.

An **estimator** is a rule that turns data into a guess; the guess is itself a random variable, because a different sample would give a different value. Its distribution is the **sampling distribution**.

An estimator $\hat\theta$ is **unbiased** if $E[\hat\theta] = \theta$. The sample mean is unbiased: $E[\bar x] = \frac1n\sum E[x_t] = \mu$.

### 7.2 Why divide by $n - 1$? (Bessel's correction)

**Claim:** for i.i.d. $x_t$ with variance $\sigma^2$, $E\big[\sum_t (x_t - \bar x)^2\big] = (n-1)\sigma^2$.

*Proof.* Write $x_t - \bar x = (x_t - \mu) - (\bar x - \mu)$. Then

$$\sum_t (x_t - \bar x)^2 = \sum_t (x_t-\mu)^2 - 2(\bar x - \mu)\sum_t(x_t - \mu) + n(\bar x - \mu)^2 = \sum_t (x_t-\mu)^2 - n(\bar x-\mu)^2,$$

using $\sum_t (x_t - \mu) = n(\bar x - \mu)$. Taking expectations: $E\sum_t(x_t-\mu)^2 = n\sigma^2$ and $E[n(\bar x - \mu)^2] = n\,\mathrm{Var}(\bar x) = n\cdot\sigma^2/n = \sigma^2$ (see 7.3). So the expectation is $n\sigma^2 - \sigma^2 = (n-1)\sigma^2$. ∎

Intuition: the deviations are measured from $\bar x$, which was fitted to the same data and is therefore "closer" to the data than the true $\mu$; the sum of squares is a bit too small, and dividing by $n - 1$ instead of $n$ compensates. We "used up" one degree of freedom estimating the mean.

**NumPy trap.** `np.std(x)` and `x.std()` divide by $n$ (`ddof=0`, "delta degrees of freedom" = 0). `np.std(x, ddof=1)` divides by $n - 1$. pandas' `Series.std()` defaults to `ddof=1`. The project's `evaluation.py::summarise` uses `auc.std()`, i.e. `ddof=0`. With $n = 25$ folds the difference is a factor $\sqrt{25/24} = 1.0206$: the reported 0.0273 would become 0.0278. Small, but you should state which you used (Lab 8). With $n = 3$ seeds (as in `05_sensitivity.py`), the `ddof=0` value is $\sqrt{2/3} = 0.816$ times the `ddof=1` value - about 18% smaller.

*(Even with $n-1$, $s$ is a slightly biased estimator of $\sigma$ because the square root is concave; this is usually ignored.)*

### 7.3 The standard error: how precise is a mean?

If $x_1, \dots, x_n$ are independent with variance $\sigma^2$:

$$\mathrm{Var}(\bar x) = \frac{1}{n^2}\sum_t \mathrm{Var}(x_t) = \frac{\sigma^2}{n}, \qquad \mathrm{SE}(\bar x) = \frac{\sigma}{\sqrt n} \approx \frac{s}{\sqrt n}.$$

Two different questions, two different numbers:

| | Standard deviation $s$ | Standard error $s/\sqrt n$ |
|---|---|---|
| Describes | spread of individual runs | uncertainty of the *mean* |
| As $n$ grows | stabilises near $\sigma$ | shrinks like $1/\sqrt n$ |
| Answers | "how much does one fold vary?" | "how precisely do we know the average?" |

The project reports **SD** (spread across folds), which is the convention in drug-repositioning papers. Always say which one you report. Reporting SE while calling it SD (or vice versa) is a common and misleading error.

**Important caveat for CV:** the formula $\sigma^2/n$ needs independence. CV folds share most of their training data, so fold results are positively correlated and the true variance of the mean is *larger* than $s^2/n$. Section 8.4 quantifies this.

### 7.4 Law of large numbers and central limit theorem

* **Law of large numbers (LLN):** $\bar x \to \mu$ as $n \to \infty$. Averages of many runs settle down.
* **Central limit theorem (CLT):** for i.i.d. draws with finite variance, the distribution of $\bar x$ approaches $\mathcal N(\mu, \sigma^2/n)$ as $n$ grows, *whatever the shape of the original distribution*.

Lab 9 draws from a heavily skewed (exponential) distribution: with $n = 2$ the sample means are still skewed (skewness 1.44), by $n = 25$ much less (0.41), and their SD matches $\sigma/\sqrt n$ at every $n$. This is why confidence intervals based on the normal or t distribution work reasonably for means of 25 fold scores, but are shakier for means of 3 seeds.

### 7.5 Sources of randomness in this project, and repeated runs

Running the same code twice can give different numbers. In this project randomness enters through:

| Source | Where | Controlled by |
|---|---|---|
| Which links/non-links go into which fold | `kfold_splits(A, k, seed)` | `seed + r` for repeat `r` |
| Weight initialisation | `nn.Linear`, `xavier_uniform_` | `set_seed(seed)` in `fit_predict` |
| Negative sampling each epoch | `torch.randint(...)` | same |
| DropEdge / hidden-link mask and cold-start mask | `torch.rand(...) < c.drop_edge` | same |
| Dropout masks | `nn.Dropout` | same |
| Non-deterministic GPU kernels | CUDA atomics in sums | not fully controllable |

Reporting a single run hides this variability; it is like measuring one patient. The protocol in `evaluation.py::run_kfold` therefore **repeats** CV with different split seeds (`repeats=5` gives 25 fold scores) and reports mean ± SD. Two distinct kinds of variability are mixed in that SD: **split variability** (different test sets) and **training variability** (different initialisations and sampling). `05_sensitivity.py` isolates the second kind: the validation split is fixed (`default_rng(12345)`) and only the training seed changes (`--seeds 3`).

A useful habit: whenever you compare two numbers, ask "what would the spread be if I re-ran this with another seed?" If the difference is smaller than that spread, you cannot claim one is better.

---

## 8. Reporting mean ± SD and confidence intervals

### 8.1 Mean ± SD, done right

**Worked example (by hand).** The first repeat of 5-fold CV for MV-HGAT on Fdataset gave fold AUCs 0.9402, 0.9395, 0.9349, 0.9557, 0.9467.

* Mean: $(0.9402 + 0.9395 + 0.9349 + 0.9557 + 0.9467)/5 = 4.7170 / 5 = 0.9434$.
* Deviations: $-0.0032, -0.0039, -0.0085, +0.0123, +0.0033$ (they sum to 0, as deviations from a mean always do).
* Squares: $1.024, 1.521, 7.225, 15.129, 1.089$ (all ×$10^{-5}$); sum $= 25.988 \times 10^{-5}$.
* $s^2 = 25.988\times 10^{-5}/4 = 6.497\times10^{-5}$, so $s = 0.0081$ (`ddof=1`). With `ddof=0`: $\sqrt{25.988\times10^{-5}/5} = 0.0072$.

Report: "AUC = 0.943 ± 0.008 (mean ± SD over 5 folds, one repeat)". A good report always states (a) what the ± means, (b) over what (folds? seeds? both?), and (c) how many.

**How many decimals?** Report to the precision the SD supports. With SD ≈ 0.008, the third decimal of the mean is already uncertain, so "0.943 ± 0.008" is honest; "0.94340 ± 0.00806" implies precision you do not have. (The project's tables print 4 decimals for uniformity with prior papers; that is acceptable when the SD is shown.)

### 8.2 Confidence intervals: definition and correct interpretation

A **95% confidence interval** (CI) for a parameter $\theta$ is a random interval $[L, U]$, computed from the data by a procedure that, over repeated samples, contains the true $\theta$ 95% of the time.

The correct reading is about the *procedure*: "if we repeated the whole experiment many times, 95% of the intervals built this way would cover the true mean". It is **not** "there is a 95% probability that $\theta$ is in this particular interval" (in the frequentist view $\theta$ is fixed; the interval is what varies). A Bayesian **credible interval** does have the latter meaning, but requires a prior.

### 8.3 The t-interval

If $x_1,\dots,x_n$ are i.i.d. normal, then $T = \frac{\bar x - \mu}{s/\sqrt n}$ follows a **Student t distribution with $n - 1$ degrees of freedom** (W. S. Gosset, 1908). Since $P(-t^* \le T \le t^*) = 0.95$ with $t^* = t_{0.975,\,n-1}$, rearranging gives the 95% CI

$$\bar x \pm t_{0.975,\,n-1}\,\frac{s}{\sqrt n}.$$

Critical values: $t_{0.975,4} = 2.776$, $t_{0.975,9} = 2.262$, $t_{0.975,24} = 2.064$, and $z_{0.975} = 1.960$ for the normal. Small samples need wider intervals because $s$ itself is uncertain.

**Worked example (continuing 8.1).** $\mathrm{SE} = 0.00806/\sqrt5 = 0.00360$; half-width $= 2.776 \times 0.00360 = 0.0100$; 95% CI $= [0.9334, 0.9534]$.

Lab 10 simulates 20,000 experiments with $n = 5$: the t-interval covers the true mean 95.0% of the time, while the naive "mean ± 1.96 SE" covers it only 87.9% of the time. With few runs, use t.

### 8.4 The CV correlation problem and the corrected interval

In $k$-fold CV, any two training sets share a fraction $(k-2)/(k-1)$ of their data (for $k = 5$: 75%), and the repeats reuse the same data set. Fold scores are therefore **positively correlated**, so $s^2/n$ **underestimates** the variance of the mean. Nadeau & Bengio (2003) proposed replacing the factor $1/n$ by $1/n + n_{\text{test}}/n_{\text{train}}$, and Bouckaert & Frank (2004) applied it to $r$-times-repeated $k$-fold CV:

$$\widehat{\mathrm{Var}}(\bar x) = \Big(\frac{1}{kr} + \frac{1}{k - 1}\Big)s^2,$$

since $n_{\text{test}}/n_{\text{train}} = (N/k)/(N(k-1)/k) = 1/(k-1)$. Note what this implies: the second term does not shrink as you add repeats. **Repeating CV many times does not make the uncertainty vanish**, because you are still re-using the same 1,933 links. Only more data does that.

Lab 11 applies three intervals to MV-HGAT's 25 AUPR values: naive t-interval [0.476, 0.499], bootstrap [0.477, 0.498], corrected [0.457, 0.519]. The corrected interval is about 2.7 times wider. The honest summary: "AUPR is about 0.49, plausibly anywhere between 0.46 and 0.52 on data like this."

The correction is a heuristic (it assumes a particular correlation structure), but it is far better calibrated than treating folds as independent (Bouckaert & Frank found the uncorrected test to have badly inflated Type I error).

### 8.5 The bootstrap

The **bootstrap** (Efron, 1979) estimates a sampling distribution by resampling the observed data **with replacement**: draw $n$ values from your $n$ values, compute the statistic, repeat thousands of times, and use the 2.5th and 97.5th percentiles as a 95% **percentile interval**. It needs no normality assumption, which makes it attractive for skewed metrics. But it still treats the resampled units as independent - bootstrapping CV folds inherits the same correlation problem (Lab 11).

A more meaningful bootstrap for this project resamples **test pairs** (or diseases) within one fold to get a CI for that fold's AUPR; it captures test-set sampling noise but not training variability.

### 8.6 Confidence interval for a proportion (precision@k)

For precision@k (the fraction of the top-$k$ predictions that are confirmed), with $\hat p = x/k$:

* **Wald interval:** $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/k}$ - simple but poor for small $k$ or $\hat p$ near 0/1 (it can even extend below 0).
* **Wilson score interval:** $\dfrac{\hat p + \frac{z^2}{2k} \pm z\sqrt{\frac{\hat p(1-\hat p)}{k} + \frac{z^2}{4k^2}}}{1 + z^2/k}$ with $z = 1.96$ - much better behaved; preferred for case-study tables like "7 of the top 10 predictions have literature support".

Example: 7 of 10 confirmed. Wald: $0.7 \pm 0.284 = [0.416, 0.984]$. Wilson: $[0.397, 0.892]$. Exercise 9 asks you to verify the Wilson numbers.

---

## 9. Base rates and random baselines under 1% positives

### 9.1 The base rate

Fdataset: $\pi = 1933/185{,}609 = 1.04\%$. Cdataset: $\pi = 2532/(663 \times 409) = 2532/271{,}167 = 0.93\%$. Roughly 1 in 100 pairs is a known link, 99 in 100 are not. This **class imbalance** changes how every metric behaves.

**The accuracy trap.** A "model" that predicts "no link" for every pair is correct on 98.96% of Fdataset cells. Accuracy is useless here. That is why the project never reports accuracy.

### 9.2 What does a random ranker achieve?

A **random ranker** assigns each pair an independent random score (e.g. Uniform(0, 1)), so the ranking is a uniformly random permutation that ignores the data. It is the reference point "no skill".

**AUC of a random ranker = 0.5.** AUC equals the probability that a randomly chosen positive gets a higher score than a randomly chosen negative (Unit B2 proves this; it is the Mann-Whitney U statistic divided by $n_1 n_0$). For a random ranker, the positive's and negative's scores are i.i.d. continuous, so by symmetry $P(S_+ > S_-) = P(S_- > S_+)$, and ties have probability 0; hence each is $1/2$. Note that **this does not depend on the base rate.**

How much does a random AUC fluctuate? Under the null, the variance of the normalised Mann-Whitney statistic is

$$\mathrm{Var}(\mathrm{AUC}) = \frac{n_1 + n_0 + 1}{12\,n_1 n_0}.$$

For one 5-fold test fold ($n_1 = 387$, $n_0 = 36{,}735$): SD $= 0.0148$. For the whole matrix: SD $= 0.0066$. So a random AUC of 0.53 in one fold would not be surprising; an AUC of 0.94 is about 30 null-SDs above chance.

**AUPR of a random ranker ≈ the base rate $\pi$.** AUPR (average precision) averages the precision at the rank of each positive. For a random permutation, the items above any cutoff are a random subset, so the *expected* fraction of positives among them - the expected precision - is $\pi$ at every cutoff. Averaging gives AUPR ≈ $\pi$. (For finite samples the expectation is slightly above $\pi$, because the precision at a positive's own rank includes that positive itself; with thousands of items the excess is negligible.) Unlike AUC, **the random AUPR depends entirely on the base rate**:

| Base rate $\pi$ | Random AUC | Random AUPR |
|---|---|---|
| 50% (balanced) | 0.5 | 0.50 |
| 10% | 0.5 | 0.10 |
| 1.04% (Fdataset) | 0.5 | 0.0104 |
| 0.93% (Cdataset) | 0.5 | 0.0093 |

**Precision@k of a random ranker** is $\pi$ in expectation (each of the top $k$ is a link with probability $\pi$), so random precision@100 ≈ 0.0104: about one hit in the top 100.

Lab 12 runs 200 random rankers on a matrix with Fdataset's base rate: mean AUC 0.4996 (SD 0.0065, matching the 0.0066 formula), mean AUPR 0.0105, mean precision@100 0.0105.

### 9.3 Reading results against the baseline

* **Lift** = metric / random value. MV-HGAT AUPR 0.4878 / 0.0104 ≈ **47×** random. NIMCGCN AUPR 0.0956 is ≈ 9× random - better than chance, but far behind.
* AUC 0.94 vs 0.5 means 94% of (positive, negative) pairs are ordered correctly. But with 95 negatives per positive, even 6% mis-ordered pairs means each positive is outranked on average by about $0.06 \times 36{,}735 \approx 2{,}200$ negatives in a test fold - which is why AUPR (focused on the top of the list) is the more honest headline.
* When the base rate differs between data sets (Fdataset 1.04% vs Cdataset 0.93%), raw AUPR values are not directly comparable; lift or comparison with the same baselines is.

### 9.4 Base rates in cold start

In leave-one-disease-out, the test column for a disease with 5 known drugs among 593 has base rate $5/593 = 0.84\%$, while for a disease with 40 known drugs it is $6.7\%$. Random per-disease AUPR therefore varies from disease to disease, which is one reason `run_lodo` pools predictions and also reports the mean per-disease **AUC** (whose random value is 0.5 for every disease).

---

## 10. Hypothesis testing and paired comparisons of models

### 10.1 The logic of a significance test

We want to know whether model A is really better than model B, or whether the observed difference could be due to chance (which folds, which seeds).

1. **Null hypothesis** $H_0$: no true difference (e.g., the mean of the per-fold differences is 0).
2. **Alternative** $H_1$: there is a difference (two-sided), or A is better (one-sided).
3. Choose a **test statistic** that is large when the data disagree with $H_0$.
4. Compute its distribution **assuming $H_0$ is true** (the null distribution).
5. The **p-value** = probability, under $H_0$, of a statistic at least as extreme as the one observed.
6. If $p < \alpha$ (commonly 0.05), reject $H_0$ ("statistically significant").

What a p-value is **not**: it is not the probability that $H_0$ is true; it is not the probability that the result is a fluke; and a large p-value is not evidence that the models are equal (absence of evidence ≠ evidence of absence). Always report the **effect size** (the mean difference, with a CI) alongside the p-value: a difference can be statistically significant yet practically trivial, or practically large but not significant because of too few runs.

**Errors.** A **Type I error** is rejecting a true $H_0$ (false discovery; rate $\alpha$). A **Type II error** is failing to reject a false $H_0$ (missed effect; rate $\beta$). **Power** $= 1 - \beta$ grows with the true effect size, the number of runs, and lower noise - and pairing is one of the cheapest ways to lower noise.

### 10.2 Why pairing matters

In `evaluation.py::run_kfold`, every method gets the folds from `kfold_splits(A, k, seed + r)`, and `03_evaluate.py` passes the same `--seed` (default 0) for every method. Therefore **fold f of repeat r has exactly the same test cells for all six methods**. The results are **paired**: we can compare models fold by fold.

Let $a_t, b_t$ be the two models' scores on fold $t$ and $d_t = a_t - b_t$. Then

$$\mathrm{Var}(d) = \mathrm{Var}(a) + \mathrm{Var}(b) - 2\,\mathrm{Cov}(a, b).$$

If some folds are intrinsically harder for *both* models (positive covariance), differencing cancels that shared difficulty and the test sees the model difference more clearly. An **unpaired** test (e.g. Welch's two-sample t-test) ignores the pairing and wastes this information.

### 10.3 The paired t-test

Treat $d_1, \dots, d_n$ as i.i.d. from a distribution with mean $\mu_d$. Test $H_0: \mu_d = 0$ with

$$t = \frac{\bar d}{s_d/\sqrt n}, \qquad \text{df} = n - 1.$$

**Worked example (by hand).** AUPR, repeat 0 only, MV-HGAT vs SCMFDD:

| fold | MV-HGAT | SCMFDD | $d$ |
|---|---|---|---|
| 0 | 0.5092 | 0.4596 | +0.0496 |
| 1 | 0.4659 | 0.5157 | -0.0498 |
| 2 | 0.5119 | 0.4951 | +0.0168 |
| 3 | 0.5432 | 0.4845 | +0.0587 |
| 4 | 0.4715 | 0.4848 | -0.0133 |

$\bar d = 0.0620/5 = 0.0124$, $s_d = 0.0449$, $t = 0.0124/(0.0449/\sqrt5) = 0.617$, df = 4, two-sided $p = 0.57$. No evidence of a difference.

### 10.4 Non-parametric alternatives

The t-test assumes the differences are roughly normal. Alternatives with weaker assumptions:

* **Sign test:** count how many $d_t > 0$. Under $H_0$ (median difference 0) that count is Binomial($n$, 0.5). Very robust, low power.
* **Wilcoxon signed-rank test:** rank $|d_t|$, sum the ranks of the positive differences; uses magnitudes and signs. Assumes the differences are symmetric about their median under $H_0$. Recommended by Demšar (2006) for comparing two classifiers over multiple *data sets*.
* **Permutation (sign-flip) test:** under $H_0$ each difference is equally likely to be $+d_t$ or $-d_t$. Randomly flip signs many times, recompute $\bar d$, and see how often the flipped mean is as extreme as the observed one. Assumption-light and intuitive.

### 10.5 The correct test for repeated cross-validation

All the tests above assume the $n$ differences are **independent**. CV folds are not (8.4), so these tests have an inflated false-positive rate: they find "significant" differences too often. Options:

* **Corrected repeated k-fold CV t-test** (Nadeau & Bengio 2003; Bouckaert & Frank 2004):
  $$t = \frac{\bar d}{\sqrt{\big(\frac{1}{kr} + \frac{1}{k-1}\big)s_d^2}}, \qquad \text{df} = kr - 1.$$
* **5×2 cv paired t-test** (Dietterich 1998): 5 repeats of 2-fold CV with a special variance estimate. Well-calibrated, but 2-fold training sets are half-size, which is a problem for our small data.
* Report differences with their corrected CIs and let the reader judge.

### 10.6 The project's real comparison (Labs 13 and 14)

Using all 25 paired folds of `results/Fdataset/cv5`:

**AUPR, MV-HGAT vs SCMFDD.** Mean difference $-0.0069$ (SCMFDD slightly higher); MV-HGAT wins 8 of 25 folds. Paired t-test $p = 0.31$; Wilcoxon $p = 0.17$; sign test $p = 0.11$; permutation $p = 0.30$; corrected test $p = 0.70$. Conclusion: **no detectable difference in AUPR on Fdataset.** Note also that the correlation between the two models' fold scores is only 0.09: their errors are largely unrelated across folds, so pairing gains little here.

**AUC, MV-HGAT vs MBiRW / DRRS / SCMFDD.** Mean differences +0.056, +0.061, +0.046; corrected p-values all below $10^{-9}$ even after Holm correction. Conclusion: **MV-HGAT's AUC advantage is large and robust.**

That is a nuanced, publishable finding: MV-HGAT ranks known links much better overall (AUC), while at the very top of the list (AUPR) it is on par with the strongest matrix-factorisation baseline on Fdataset. Unit E3 discusses how to write it up.

### 10.7 Multiple comparisons

Comparing MV-HGAT to 5 baselines on 2 metrics and 2 data sets is 20 tests. At $\alpha = 0.05$, even if nothing were truly different, you would expect about one "significant" result by chance. Corrections:

* **Bonferroni:** multiply each p-value by the number of tests $m$ (or use $\alpha/m$). Simple, conservative.
* **Holm (step-down):** sort the p-values $p_{(1)} \le \dots \le p_{(m)}$; adjusted $p_{(i)} = \max_{j \le i} \min\big(1, (m - j + 1)\,p_{(j)}\big)$. Controls the same family-wise error rate as Bonferroni but is uniformly more powerful. Lab 14 implements it.
* For many methods across many data sets, Demšar (2006) recommends the **Friedman test** followed by post-hoc tests, visualised as a critical-difference diagram.

---

## 11. Code laboratory

All labs are self-contained, run on the CPU in a few seconds, and were executed with the project's interpreter (`.venv\Scripts\python.exe`: NumPy 2.5, SciPy 1.18, scikit-learn 1.9, PyTorch 2.14). The output shown under each block is the real output. Save a block as `lab.py` and run `python lab.py`.

### Lab 1 - Bernoulli and binomial by simulation (sections 3.1-3.2)

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
p = 1933 / (593 * 313)           # Fdataset base rate: P(a random cell is a known link)
print(f"p = {p:.6f}")

# 1. Bernoulli: simulate one million cells
x = rng.random(1_000_000) < p     # True with probability p
print(f"simulated mean {x.mean():.6f}   theory p        {p:.6f}")
print(f"simulated var  {x.var():.6f}   theory p(1-p)   {p*(1-p):.6f}")

# 2. Binomial: number of positives among the 37,122 cells of one 5-fold test fold
n = 37_122
X = stats.binom(n, p)
print(f"E[X] = {X.mean():.1f}, SD[X] = {X.std():.2f}")
print(f"P(X <= 350) = {X.cdf(350):.4f},  P(X >= 420) = {X.sf(419):.4f}")
draws = rng.binomial(n, p, size=100_000)
print(f"simulated mean {draws.mean():.1f}, simulated SD {draws.std():.2f}")
```

```text
p = 0.010414
simulated mean 0.010411   theory p        0.010414
simulated var  0.010303   theory p(1-p)   0.010306
E[X] = 386.6, SD[X] = 19.56
P(X <= 350) = 0.0310,  P(X >= 420) = 0.0477
simulated mean 386.6, simulated SD 19.57
```

*What to notice:* the simulated mean and variance match $p$ and $p(1-p)$; an unstratified random fold would have fewer than 351 positives about 3% of the time and 420 or more about 5% of the time. `X.sf(419)` is $P(X > 419) = P(X \ge 420)$ - "sf" is the survival function $1 - F$.

### Lab 2 - Sigmoid, logit and odds (section 4)

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logit(p):
    return np.log(p / (1 - p))

z = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
p = sigmoid(z)
for zi, pi in zip(z, p):
    print(f"z = {zi:+.1f}  ->  sigma(z) = {pi:.4f}   odds = {pi/(1-pi):8.4f}")

print("logit(sigmoid(z)) == z :", np.allclose(logit(p), z))
print("sigma(-z) == 1 - sigma(z):", np.allclose(sigmoid(-z), 1 - p))

# derivative check: sigma'(z) = sigma(z)(1 - sigma(z))
h = 1e-6
numeric = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
print("max |numeric - analytic| derivative:", f"{np.max(np.abs(numeric - p * (1 - p))):.2e}")
```

```text
z = -6.0  ->  sigma(z) = 0.0025   odds =   0.0025
z = -3.0  ->  sigma(z) = 0.0474   odds =   0.0498
z = -1.0  ->  sigma(z) = 0.2689   odds =   0.3679
z = +0.0  ->  sigma(z) = 0.5000   odds =   1.0000
z = +1.0  ->  sigma(z) = 0.7311   odds =   2.7183
z = +3.0  ->  sigma(z) = 0.9526   odds =  20.0855
z = +6.0  ->  sigma(z) = 0.9975   odds = 403.4288
logit(sigmoid(z)) == z : True
sigma(-z) == 1 - sigma(z): True
max |numeric - analytic| derivative: 7.23e-11
```

*What to notice:* odds equal $e^z$ exactly (e.g. $e^1 = 2.7183$, $e^3 = 20.0855$); each unit of logit multiplies the odds by $e$. The central-difference derivative agrees with $\sigma(1-\sigma)$ to about $10^{-10}$.

### Lab 3 - Likelihood, log-likelihood and the MLE (section 5)

```python
import numpy as np

k, n = 1933, 593 * 313                   # successes and trials in Fdataset
ps = np.linspace(0.005, 0.02, 7)
loglik = k * np.log(ps) + (n - k) * np.log(1 - ps)
for p_, ll in zip(ps, loglik):
    print(f"p = {p_:.4f}   log L = {ll:12.2f}")

# finer grid search vs the closed form k/n
grid = np.linspace(0.001, 0.05, 200_001)
ll = k * np.log(grid) + (n - k) * np.log(1 - grid)
print(f"grid argmax  p = {grid[np.argmax(ll)]:.6f}")
print(f"closed form k/n = {k / n:.6f}")

# why we take logs: the likelihood itself underflows to zero
p_hat = k / n
direct = np.prod(np.where(np.arange(n) < k, p_hat, 1 - p_hat))
print("product of probabilities:", direct)
print("sum of log-probabilities:", round(k * np.log(p_hat) + (n - k) * np.log(1 - p_hat), 2))
```

```text
p = 0.0050   log L =    -11162.33
p = 0.0075   log L =    -10840.65
p = 0.0100   log L =    -10747.80
p = 0.0125   log L =    -10780.88
p = 0.0150   log L =    -10894.04
p = 0.0175   log L =    -11062.84
p = 0.0200   log L =    -11272.69
grid argmax  p = 0.010414
closed form k/n = 0.010414
product of probabilities: 0.0
sum of log-probabilities: -10746.21
```

*What to notice:* the log-likelihood is a smooth hill peaking at $k/n$; the raw product of 185,609 probabilities is exactly 0.0 in floating point, while its logarithm is a perfectly ordinary number, $-10{,}746.21$.

### Lab 4 - BCE three ways, and its gradient (sections 6.1-6.4)

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss

y = np.array([1, 1, 0, 0, 0, 0])                 # 2 positives, 4 sampled negatives
z = np.array([2.0, -0.5, -1.0, 0.3, -3.0, -2.0]) # logits from a model
p = 1 / (1 + np.exp(-z))

by_hand = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
torch_bce = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y, dtype=torch.float64))
print(f"by hand          : {by_hand:.6f}")
print(f"torch (logits)   : {torch_bce.item():.6f}")
print(f"sklearn log_loss : {log_loss(y, p):.6f}")

# gradient of the mean BCE with respect to each logit is (sigma(z) - y) / N
zt = torch.tensor(z, requires_grad=True)
F.binary_cross_entropy_with_logits(zt, torch.tensor(y, dtype=torch.float64)).backward()
print("autograd  :", np.round(zt.grad.numpy(), 4))
print("formula   :", np.round((p - y) / len(y), 4))
```

```text
by hand          : 0.407356
torch (logits)   : 0.407356
sklearn log_loss : 0.407356
autograd  : [-0.0199 -0.1037  0.0448  0.0957  0.0079  0.0199]
formula   : [-0.0199 -0.1037  0.0448  0.0957  0.0079  0.0199]
```

*What to notice:* all three implementations agree with the hand computation in 6.2 (0.4074). The gradient for the positive with logit −0.5 is the largest negative number (push up), and the negative with logit +0.3 gets the largest positive gradient (push down).

### Lab 5 - Why the code passes logits (section 6.5)

```python
import numpy as np
import torch
import torch.nn.functional as F

z = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
y = np.array([1.0, 1.0, 1.0, 0.0, 0.0])          # every one is badly wrong except z=0

with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
    p = 1 / (1 + np.exp(-z))                      # sigmoid first ...
    naive = -(y * np.log(p) + (1 - y) * np.log(1 - p))   # ... then log: breaks
stable = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
torch_v = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y), reduction="none")

np.set_printoptions(suppress=True, precision=4)
print("naive  :", naive)
print("stable :", stable)
print("torch  :", torch_v.numpy())
```

```text
naive  : [    inf 40.      0.6931     inf     inf]
stable : [800.      40.       0.6931  40.     800.    ]
torch  : [800.      40.       0.6931  40.     800.    ]
```

*What to notice:* at $z = 40$ with $y = 0$, $\sigma(40)$ rounds to exactly 1.0 in double precision, so the naive loss is $-\ln 0 = \infty$ even though the true loss is 40. In float32 (what GPUs use) this happens already at $|z| \approx 17$. An infinite loss produces NaN gradients and silently destroys training.

### Lab 6 - BCE minimisation *is* maximum likelihood (section 5.6)

```python
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
rng = np.random.default_rng(0)

# Toy data: 2,000 "pairs", one feature, true model p = sigmoid(-1 + 2x)
x = rng.normal(size=2000)
y = (rng.random(2000) < 1 / (1 + np.exp(-(-1 + 2 * x)))).astype(np.float32)
X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.5)
for step in range(2000):
    loss = F.binary_cross_entropy_with_logits(b + w * X, Y)   # mean negative log-likelihood
    opt.zero_grad(); loss.backward(); opt.step()
print(f"MLE by gradient descent    : b = {b.item():.3f}, w = {w.item():.3f}, mean NLL = {loss.item():.4f}")

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=np.inf).fit(x[:, None], y)
print(f"sklearn (C=inf, no penalty): b = {lr.intercept_[0]:.3f}, w = {lr.coef_[0, 0]:.3f}")
```

```text
MLE by gradient descent    : b = -0.952, w = 2.038, mean NLL = 0.4282
sklearn (C=inf, no penalty): b = -0.952, w = 2.038
```

*What to notice:* minimising BCE with gradient descent and scikit-learn's maximum-likelihood solver give identical parameters. Both are close to, but not exactly, the true values (−1, 2): with finite data the MLE has sampling error (section 5.5). `C=np.inf` turns off scikit-learn's default L2 penalty.

### Lab 7 - Negative sampling shifts logits but not rankings (section 6.6)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(1)
n = 200_000
x = rng.normal(size=(n, 1))
true_logit = -5.7 + 1.5 * x[:, 0]                # gives roughly a 1% positive rate
y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
print(f"positive rate in the full data: {y.mean():.4f}")

# Model A: trained on all cells.  Model B: all positives + 2 sampled negatives per positive
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
keep = np.concatenate([pos, rng.choice(neg, 2 * len(pos), replace=False)])
A = LogisticRegression(C=np.inf).fit(x, y)
B = LogisticRegression(C=np.inf).fit(x[keep], y[keep])
print(f"intercepts: full {A.intercept_[0]:.3f}   neg-sampled {B.intercept_[0]:.3f}")
print(f"slopes    : full {A.coef_[0,0]:.3f}    neg-sampled {B.coef_[0,0]:.3f}")

r = (len(neg) / len(pos)) / 2                    # how much the negatives were thinned
print(f"predicted intercept shift log(r) = {np.log(r):.3f}; "
      f"observed = {B.intercept_[0] - A.intercept_[0]:.3f}")

pA, pB = A.predict_proba(x)[:, 1], B.predict_proba(x)[:, 1]
print(f"mean predicted prob: full {pA.mean():.4f}   neg-sampled {pB.mean():.4f}")
print(f"AUC  full {roc_auc_score(y, pA):.4f}   neg-sampled {roc_auc_score(y, pB):.4f}")
print(f"AUPR full {average_precision_score(y, pA):.4f}   neg-sampled {average_precision_score(y, pB):.4f}")
```

```text
positive rate in the full data: 0.0097
intercepts: full -5.732   neg-sampled -1.790
slopes    : full 1.531    neg-sampled 1.506
predicted intercept shift log(r) = 3.937; observed = 3.942
mean predicted prob: full 0.0097   neg-sampled 0.2154
AUC  full 0.8558   neg-sampled 0.8558
AUPR full 0.1113   neg-sampled 0.1113
```

*What to notice:* the slope (what the feature "means") is essentially unchanged; the intercept moves up by $\ln(1/s)$ as predicted by Bayes' theorem; the average predicted probability becomes ~0.22 instead of the true ~0.01; AUC and AUPR are identical to four decimals, because a constant shift of the logit is a monotone transformation and preserves the ranking.

### Lab 8 - Mean, SD (ddof), SE on the project's own numbers (section 7)

```python
import numpy as np

# Per-fold AUPR of MV-HGAT, Fdataset, 5 repeats x 5 folds (results/Fdataset/cv5/mvhgat.json, rounded)
aupr = np.array([0.5092, 0.4659, 0.5119, 0.5432, 0.4715, 0.4561, 0.4883, 0.5084, 0.5059, 0.4730,
                 0.5066, 0.4732, 0.5019, 0.4804, 0.4779, 0.5050, 0.4923, 0.4734, 0.5000, 0.4771,
                 0.4798, 0.5323, 0.4936, 0.4012, 0.4664])
n = len(aupr)
mean = aupr.sum() / n
var_pop = ((aupr - mean) ** 2).sum() / n          # divide by n   (numpy default, ddof=0)
var_smp = ((aupr - mean) ** 2).sum() / (n - 1)    # divide by n-1 (Bessel, ddof=1)
print(f"n = {n}, mean = {mean:.4f}")
print(f"std ddof=0 = {np.sqrt(var_pop):.4f}  (np.std: {aupr.std():.4f})")
print(f"std ddof=1 = {np.sqrt(var_smp):.4f}  (np.std(ddof=1): {aupr.std(ddof=1):.4f})")
print(f"standard error of the mean = {aupr.std(ddof=1) / np.sqrt(n):.4f}")
print(f"report: AUPR = {mean:.3f} ± {aupr.std(ddof=1):.3f} (mean ± SD over {n} folds)")

# per-repeat means: each repeat is one complete 5-fold CV
per_repeat = aupr.reshape(5, 5).mean(axis=1)
print("per-repeat means:", np.round(per_repeat, 4), " SD of those:", round(per_repeat.std(ddof=1), 4))
```

```text
n = 25, mean = 0.4878
std ddof=0 = 0.0273  (np.std: 0.0273)
std ddof=1 = 0.0278  (np.std(ddof=1): 0.0278)
standard error of the mean = 0.0056
report: AUPR = 0.488 ± 0.028 (mean ± SD over 25 folds)
per-repeat means: [0.5003 0.4863 0.488  0.4896 0.4747]  SD of those: 0.0092
```

*What to notice:* the reported 0.0273 in `RESULTS.md` is the `ddof=0` value. The SD of the five per-repeat means (0.0092) is much smaller than the per-fold SD, because averaging 5 folds cancels some noise - but it is not $0.0278/\sqrt5 = 0.0124$ either; the numbers depend on how the noise is structured.

### Lab 9 - The central limit theorem in action (section 7.4)

```python
import numpy as np

rng = np.random.default_rng(0)
# A skewed "population": per-fold metric values cannot be assumed normal
population = rng.exponential(scale=1.0, size=1_000_000)   # true mean 1, true SD 1
for n in [2, 5, 25, 100]:
    means = rng.choice(population, size=(20_000, n)).mean(axis=1)
    print(f"n = {n:3d}: SD of sample means = {means.std():.4f}   "
          f"sigma/sqrt(n) = {1/np.sqrt(n):.4f}   skewness = {((means-means.mean())**3).mean()/means.std()**3:.2f}")
```

```text
n =   2: SD of sample means = 0.7102   sigma/sqrt(n) = 0.7071   skewness = 1.44
n =   5: SD of sample means = 0.4487   sigma/sqrt(n) = 0.4472   skewness = 0.92
n =  25: SD of sample means = 0.2015   sigma/sqrt(n) = 0.2000   skewness = 0.41
n = 100: SD of sample means = 0.1012   sigma/sqrt(n) = 0.1000   skewness = 0.21
```

*What to notice:* the SD of the mean follows $\sigma/\sqrt n$ exactly (independence holds here by construction), and the skewness of the sampling distribution decays roughly like $2/\sqrt n$ toward the normal's 0.

### Lab 10 - Confidence-interval coverage: t versus z (section 8.3)

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
mu, sigma, n, trials = 0.49, 0.03, 5, 20_000
cover_t = cover_z = 0
for _ in range(trials):
    s = rng.normal(mu, sigma, n)
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    cover_t += (m - t_crit * se <= mu <= m + t_crit * se)
    cover_z += (m - 1.96 * se <= mu <= m + 1.96 * se)
print(f"t critical value for n={n}: {stats.t.ppf(0.975, n-1):.3f} (vs 1.960 for the normal)")
print(f"coverage of 95% t-interval : {cover_t / trials:.3f}")
print(f"coverage of 'mean ± 1.96 SE': {cover_z / trials:.3f}")
```

```text
t critical value for n=5: 2.776 (vs 1.960 for the normal)
coverage of 95% t-interval : 0.950
coverage of 'mean ± 1.96 SE': 0.879
```

### Lab 11 - Three confidence intervals for MV-HGAT's AUPR (sections 8.3-8.5)

```python
import numpy as np
from scipy import stats

aupr = np.array([0.5092, 0.4659, 0.5119, 0.5432, 0.4715, 0.4561, 0.4883, 0.5084, 0.5059, 0.4730,
                 0.5066, 0.4732, 0.5019, 0.4804, 0.4779, 0.5050, 0.4923, 0.4734, 0.5000, 0.4771,
                 0.4798, 0.5323, 0.4936, 0.4012, 0.4664])
n, m, s = len(aupr), aupr.mean(), aupr.std(ddof=1)

# (a) naive t-interval treating the 25 folds as independent
t = stats.t.ppf(0.975, n - 1)
print(f"naive 95% CI        : [{m - t*s/np.sqrt(n):.4f}, {m + t*s/np.sqrt(n):.4f}]")

# (b) corrected for overlapping training sets (Nadeau & Bengio; Bouckaert & Frank)
k, r = 5, 5
se_corr = s * np.sqrt(1 / (k * r) + 1 / (k - 1))
print(f"corrected 95% CI    : [{m - t*se_corr:.4f}, {m + t*se_corr:.4f}]")

# (c) percentile bootstrap of the mean (resampling folds, also assumes independence)
rng = np.random.default_rng(0)
boot = rng.choice(aupr, size=(10_000, n)).mean(axis=1)
print(f"bootstrap 95% CI    : [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
```

```text
naive 95% CI        : [0.4763, 0.4993]
corrected 95% CI    : [0.4569, 0.5187]
bootstrap 95% CI    : [0.4767, 0.4983]
```

### Lab 12 - The random baseline at a 1% base rate (section 9)

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(0)
A = np.zeros(593 * 313)
A[rng.choice(A.size, 1933, replace=False)] = 1    # a matrix with Fdataset's base rate
pi = A.mean()

aucs, aps, p_at_100 = [], [], []
for _ in range(200):
    s = rng.random(A.size)                        # a ranker that knows nothing
    aucs.append(roc_auc_score(A, s))
    aps.append(average_precision_score(A, s))
    p_at_100.append(A[np.argsort(-s)[:100]].mean())
print(f"base rate pi = {pi:.4f}")
print(f"random AUC : mean {np.mean(aucs):.4f}  SD {np.std(aucs):.4f}")
print(f"random AUPR: mean {np.mean(aps):.4f}  SD {np.std(aps):.4f}")
print(f"random precision@100: mean {np.mean(p_at_100):.4f}")
print(f"'always say no' accuracy: {1 - pi:.4f}")

# theoretical SD of a random AUC (Mann-Whitney null) for one 5-fold test fold
n1, n0 = 387, 36_735
print(f"null SD of AUC, whole matrix: {np.sqrt((1933 + 183_676 + 1) / (12 * 1933 * 183_676)):.4f}")
print(f"null SD of AUC in one fold: {np.sqrt((n1 + n0 + 1) / (12 * n1 * n0)):.4f}")
print(f"MV-HGAT AUPR 0.4878 is {0.4878 / pi:.0f}x the random baseline")
```

```text
base rate pi = 0.0104
random AUC : mean 0.4996  SD 0.0065
random AUPR: mean 0.0105  SD 0.0003
random precision@100: mean 0.0105
'always say no' accuracy: 0.9896
null SD of AUC, whole matrix: 0.0066
null SD of AUC in one fold: 0.0148
MV-HGAT AUPR 0.4878 is 47x the random baseline
```

### Lab 13 - Paired comparison on the real fold results (section 10)

```python
import numpy as np
from scipy import stats

mv = np.array([0.5092, 0.4659, 0.5119, 0.5432, 0.4715, 0.4561, 0.4883, 0.5084, 0.5059, 0.4730,
               0.5066, 0.4732, 0.5019, 0.4804, 0.4779, 0.5050, 0.4923, 0.4734, 0.5000, 0.4771,
               0.4798, 0.5323, 0.4936, 0.4012, 0.4664])
sc = np.array([0.4596, 0.5157, 0.4951, 0.4845, 0.4848, 0.4711, 0.5072, 0.5063, 0.4643, 0.4731,
               0.4537, 0.4882, 0.5148, 0.5309, 0.5008, 0.5081, 0.4996, 0.4911, 0.4928, 0.5053,
               0.5215, 0.5154, 0.5125, 0.4728, 0.4967])
d = mv - sc
n, k, r = len(d), 5, 5
print(f"mean AUPR  MV-HGAT {mv.mean():.4f}   SCMFDD {sc.mean():.4f}   mean diff {d.mean():+.4f}")
print(f"MV-HGAT wins {np.sum(d > 0)} of {n} folds; corr(mv, sc) = {np.corrcoef(mv, sc)[0, 1]:.3f}")

print("unpaired Welch t-test    p =", round(stats.ttest_ind(mv, sc, equal_var=False).pvalue, 4))
print("paired t-test            p =", round(stats.ttest_rel(mv, sc).pvalue, 4))
print("Wilcoxon signed-rank     p =", round(stats.wilcoxon(mv, sc).pvalue, 4))
print("sign test (binomial)     p =", round(stats.binomtest(int(np.sum(d > 0)), n, 0.5).pvalue, 4))

# corrected repeated k-fold t-test (Bouckaert & Frank 2004, after Nadeau & Bengio 2003)
t_c = d.mean() / np.sqrt((1 / n + 1 / (k - 1)) * d.var(ddof=1))
p_c = 2 * stats.t.sf(abs(t_c), df=n - 1)
print(f"corrected repeated CV t  t = {t_c:.3f}, p = {p_c:.4f}")

# paired permutation (sign-flip) test, exact enough with 100k random flips
rng = np.random.default_rng(0)
flips = rng.choice([-1, 1], size=(100_000, n))
null = (flips * d).mean(axis=1)
print("sign-flip permutation    p =", round(np.mean(np.abs(null) >= abs(d.mean())), 4))
```

```text
mean AUPR  MV-HGAT 0.4878   SCMFDD 0.4946   mean diff -0.0069
MV-HGAT wins 8 of 25 folds; corr(mv, sc) = 0.094
unpaired Welch t-test    p = 0.3248
paired t-test            p = 0.3071
Wilcoxon signed-rank     p = 0.1742
sign test (binomial)     p = 0.1078
corrected repeated CV t  t = -0.388, p = 0.7018
sign-flip permutation    p = 0.3045
```

*What to notice:* every test agrees there is no significant AUPR difference; the corrected test is the most conservative (p = 0.70), as it should be. Because the two models' fold scores are almost uncorrelated (0.094), the paired and unpaired tests give similar p-values here; pairing helps most when per-fold difficulty is shared.

### Lab 14 - Several baselines at once with Holm correction (section 10.7)

```python
import numpy as np
from scipy import stats

auc = {
 "MV-HGAT": [0.9402,0.9395,0.9349,0.9557,0.9467,0.9344,0.9366,0.9499,0.9421,0.9307,0.9402,0.9438,0.9282,
             0.9426,0.9382,0.9377,0.9420,0.9436,0.9364,0.9445,0.9287,0.9486,0.9279,0.9263,0.9417],
 "MBiRW":   [0.8756,0.8818,0.8847,0.9030,0.8848,0.8647,0.8863,0.8822,0.8888,0.8885,0.8956,0.8907,0.8716,
             0.8846,0.8775,0.8899,0.8864,0.8768,0.8741,0.8833,0.8769,0.9019,0.8887,0.8645,0.8751],
 "DRRS":    [0.8842,0.8811,0.8498,0.8941,0.8687,0.8818,0.8824,0.8962,0.8630,0.8782,0.8853,0.8757,0.8738,
             0.8860,0.8874,0.8816,0.8632,0.8843,0.8754,0.8873,0.8799,0.8982,0.8795,0.8472,0.8841],
 "SCMFDD":  [0.8971,0.8707,0.8931,0.9067,0.9044,0.8837,0.8947,0.9002,0.8856,0.8892,0.8998,0.8940,0.8775,
             0.9047,0.8838,0.9078,0.8996,0.8939,0.8777,0.9031,0.8944,0.9109,0.8918,0.8807,0.8911],
}
ours = np.array(auc["MV-HGAT"])
k, r = 5, 5
rows = []
for name in ["MBiRW", "DRRS", "SCMFDD"]:
    d = ours - np.array(auc[name])
    t_c = d.mean() / np.sqrt((1 / (k * r) + 1 / (k - 1)) * d.var(ddof=1))
    rows.append((name, d.mean(), 2 * stats.t.sf(abs(t_c), k * r - 1)))

# Holm correction for 3 comparisons: sort p-values, multiply the i-th smallest by (m - i)
m = len(rows)
order = sorted(range(m), key=lambda i: rows[i][2])
running = 0.0
adj = [None] * m
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * rows[i][2]))
    adj[i] = running
for (name, md, p), pa in zip(rows, adj):
    print(f"MV-HGAT vs {name:7s}: mean AUC diff {md:+.4f}  corrected p = {p:.2e}  Holm-adjusted p = {pa:.2e}")
```

```text
MV-HGAT vs MBiRW  : mean AUC diff +0.0561  corrected p = 5.02e-12  Holm-adjusted p = 1.51e-11
MV-HGAT vs DRRS   : mean AUC diff +0.0605  corrected p = 2.58e-10  Holm-adjusted p = 5.16e-10
MV-HGAT vs SCMFDD : mean AUC diff +0.0458  corrected p = 5.27e-10  Holm-adjusted p = 5.27e-10
```

---

## 12. In this project: the concepts in the real code

### 12.1 The Bernoulli model and BCE in the training loop

`src/drepo/methods.py`, `MVHGATMethod.fit_predict` (inside the epoch loop):

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict)
logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
logits = logits.reshape(-1)
n_neg = min(neg_pool.numel(), sup.numel() * c.neg_ratio) if c.neg_ratio > 0 \
    else neg_pool.numel()
neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]
y = torch.cat([torch.ones(sup.numel(), device=DEVICE),
               torch.zeros(n_neg, device=DEVICE)])
loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
```

Line by line:

1. `model(...)` returns a drugs × diseases matrix of **logits** $z_{ij}$ (log-odds). `reshape(-1)` flattens it so that index `i * n_diseases + j` addresses pair $(i, j)$ - the same flat indexing as `A.ravel()` (Unit A1).
2. `sup` holds the flat indices of the positives supervised this epoch (the hidden links). `n_neg = sup.numel() * c.neg_ratio` asks for 2 negatives per positive, capped by the size of the pool.
3. `neg = neg_pool[torch.randint(...)]` draws negatives **uniformly with replacement** from the allowed unknown cells (`neg_pool` excludes test cells, Unit E1). Fresh negatives every epoch means that over 600 epochs the model sees most of the pool.
4. `y` is the label vector: ones for positives, zeros for negatives. Each element is one Bernoulli observation.
5. `F.binary_cross_entropy_with_logits(z, y)` computes $\frac1N\sum_t[\max(z_t,0) - z_t y_t + \ln(1+e^{-|z_t|})]$, i.e. the mean negative Bernoulli log-likelihood, stably (section 6.5). Its gradient with respect to each selected logit is $(\sigma(z_t) - y_t)/N$ (section 6.4), which backpropagates into the GAT layers.

At the end:

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict)
model.eval()
P, deg = propagation(A_full), degrees(A_full)
with torch.no_grad():
    logits, _ = model(X, graphs, P=P, deg=deg)
self.last = (model, X, graphs, P, deg)
return torch.sigmoid(logits).cpu().numpy()
```

`torch.sigmoid(logits)` maps log-odds to $(0, 1)$. Because of negative sampling (section 6.6) these values are inflated relative to the true ~1% link rate; they are **scores for ranking**, which is all `evaluation.metrics` needs. Applying the sigmoid does not change AUC or AUPR (it is monotone); it only makes the numbers human-readable.

### 12.2 Weighted BCE in the baselines

```python
# (quoted from src/drepo/methods.py, weighted_bce)
def weighted_bce(logits, A, neg_mask):
    """BCE over all allowed cells, positives up-weighted to balance classes."""
    m = (A > 0) | neg_mask
    pw = (m & (A == 0)).sum() / max((A > 0).sum(), 1)
    w = torch.where(A > 0, pw, torch.ones_like(A)) * m
    return (F.binary_cross_entropy_with_logits(logits, A, reduction="none") * w).sum() / w.sum()
```

* `m` marks cells allowed in the loss: training positives and allowed negatives.
* `pw` = number of allowed negatives / number of positives ≈ 95 in a 5-fold training fold (1,546 positives vs 146,940 allowed negatives). Each positive's log-likelihood term is multiplied by `pw`, so positives and negatives contribute equal total weight.
* `reduction="none"` returns the per-cell BCE; multiplying by `w` and dividing by `w.sum()` gives a **weighted average negative log-likelihood**.

Statistically this is a weighted likelihood; like negative sampling, it shifts logits by roughly $\ln(\texttt{pw})$ and leaves ranking intact.

### 12.3 The sigmoid inside the model

`src/drepo/model.py`, `MVHGAT.gnn_gate`:

```python
# (quoted from src/drepo/model.py, MVHGAT.gnn_gate; line wrapped for readability)
g = self.gate
return torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))[:, None] * \
       torch.sigmoid(g[2] + g[3] * torch.log1p(deg_d))[None, :]
```

Each factor is a logistic function of $\ln(1 + \text{degree})$: a learned soft switch in $(0, 1)$. With the initial values `gate = [0, 1, 0, 1]`, a disease with 0 visible links gets $\sigma(0 + 1\cdot\ln 1) = \sigma(0) = 0.5$, while one with 10 links gets $\sigma(\ln 11) = 11/12 = 0.917$ (because $\sigma(\ln x) = x/(1+x)$). Here the sigmoid is not a probability of anything; it is used as a smooth gate. Same function, different role.

### 12.4 Mean ± SD in `evaluation.py::summarise`

```python
# (quoted from src/drepo/evaluation.py, summarise)
auc = np.array([f["AUC"] for f in folds])
aupr = np.array([f["AUPR"] for f in folds])
return {"summary": {"AUC": float(auc.mean()), "AUC_std": float(auc.std()),
                    "AUPR": float(aupr.mean()), "AUPR_std": float(aupr.std()),
                    "n_runs": len(folds)}, ...
```

* `folds` holds one dictionary per (repeat, fold); with `--repeats 5` and `cv5` there are 25.
* `auc.mean()` is the sample mean over all 25.
* `auc.std()` is the **population** formula (`ddof=0`). When you write the paper, say "mean ± SD (population formula) over 25 folds from 5 repeats of 5-fold CV", or change it to `ddof=1` and say "sample SD". Either is fine if stated.
* `n_runs` is stored so you can always report how many values the SD was computed from.
* The `run_kfold` loop also saves per-fold results (`m.update(repeat=r, fold=f)`), which is what makes the **paired** tests in Labs 13-14 possible. Keep them; never report only the summary.

### 12.5 Repeated runs in `05_sensitivity.py`

```python
# (quoted from scripts/05_sensitivity.py)
for s in range(args.seeds):
    S = MVHGATMethod(**{p: v}).fit_predict(data, A_tr, neg_mask, seed=s)
    runs.append(metrics(y, S.ravel()[idx]))
r = {"value": v,
     "AUC": float(np.mean([x["AUC"] for x in runs])),
     "AUC_std": float(np.std([x["AUC"] for x in runs])), ...
```

Here the validation split is fixed and only the training seed varies (default 3 seeds), so the SD measures **training variability only**. With 3 values and `ddof=0` the SD is underestimated by about 18% relative to `ddof=1` ($\sqrt{2/3} = 0.816$), and any SD from 3 numbers is itself very uncertain. Example from `results/Fdataset/sensitivity.json`: `hidden=64` gave AUPR 0.5209 ± 0.0075 and `hidden=128` gave 0.4651 ± 0.0167. The gap (0.056) is several SDs, so it is probably real; but `hidden=32` (0.4792 ± 0.0182) vs `hidden=16` (0.4748 ± 0.0066) is well within the noise.

### 12.6 The metrics and the base rate in `evaluation.py`

The module docstring says: "AUPR = average precision ... With ~1% positives this is the harder, more honest number; random = positive rate." Section 9 proved that statement. `metrics(y, s)` calls scikit-learn's `roc_auc_score` and `average_precision_score`, which depend only on the **ordering** of `s` - consistent with section 6.6's point that the inflated probabilities do no harm.

---

## 13. Common mistakes and misconceptions

1. **"The model's output is the probability that the drug works."** No. It is a score from a model trained on *recorded* indications with negative sampling; it is shifted (section 6.6), and "unknown" ≠ "does not work". Use it to rank.
2. **Applying `sigmoid` and then `BCELoss` (or `log`) yourself.** Numerically unstable (Lab 5). Pass logits to `binary_cross_entropy_with_logits`.
3. **Applying the sigmoid twice.** E.g. returning `torch.sigmoid(logits)` from the model and then calling `BCEWithLogitsLoss` on it. The loss then sees values in (0, 1) as logits; training still "runs" but learns badly. A silent bug.
4. **Confusing likelihood with probability of the parameter.** $L(p)$ is $P(\text{data}\mid p)$, not $P(p \mid \text{data})$.
5. **Reporting accuracy on a 1%-positive problem.** "Always no" scores 98.96%.
6. **Comparing AUPR across data sets with different base rates** without noting the different random baselines.
7. **Not saying what ± means.** SD or SE? Over folds, repeats or seeds? How many? `ddof` 0 or 1?
8. **Treating CV folds as independent** in a t-test or CI. Leads to overconfident claims. Use the corrected test.
9. **Using an unpaired test on paired results**, or comparing methods evaluated on *different* splits (e.g. your numbers vs numbers copied from a paper) as if paired.
10. **"p > 0.05, so the models are equally good."** No - you merely failed to detect a difference; look at the CI of the difference.
11. **"p = 0.001, so the improvement is important."** Significance ≠ size. A 0.001 AUC gain can be significant with enough runs and still be irrelevant.
12. **Running many comparisons and reporting the significant ones** without correction (multiple comparisons; also called "p-hacking" when done selectively).
13. **Believing more repeats remove all uncertainty.** Repeats reduce the split/seed noise but not the uncertainty from having only 1,933 links (section 8.4).
14. **Thinking a random ranker has AUPR 0.5.** Only under a 50% base rate. Here it is ~0.01.
15. **Interpreting a 95% CI as "95% probability the true value is inside".** That is the Bayesian credible-interval reading; the frequentist CI is a statement about the procedure.

---

## 14. Exercises

Difficulty: (C) conceptual, (M) mathematical, (P) programming. Stars indicate difficulty.

**Exercise 1 (C, ★).** A colleague says: "Our model reaches 99% accuracy on Fdataset, so it's excellent." What single number do you ask for, and why?

<details><summary>Solution</summary>

Ask for the base rate (or, equivalently, the accuracy of the trivial model that always predicts "no link"). On Fdataset, 1 − π = 98.96%, so 99% accuracy is only 0.04 percentage points better than a model that knows nothing. Better still, ask for AUPR together with its random baseline (π ≈ 0.0104), or precision@k. Accuracy is dominated by the 99% negatives and says nothing about whether the model finds the rare positives.
</details>

**Exercise 2 (M, ★).** Show that for $Y \sim \mathrm{Bernoulli}(p)$, $\mathrm{Var}(Y)$ is maximised at $p = 1/2$, and compute the variance at $p = 0.0104$ and at $p = 1/3$ (the training base rate under `neg_ratio=2`).

<details><summary>Solution</summary>

$v(p) = p(1-p) = p - p^2$; $v'(p) = 1 - 2p = 0 \Rightarrow p = 1/2$; $v''(p) = -2 < 0$, so it is a maximum, $v(1/2) = 1/4$.
At $p = 0.0104$: $0.0104 \times 0.9896 = 0.01029$. At $p = 1/3$: $\frac13\cdot\frac23 = 2/9 = 0.2222$. Negative sampling makes the training labels much "more informative per example" (higher variance), which is one intuition for why it speeds up learning.
</details>

**Exercise 3 (M, ★★).** Derive BCE from the Bernoulli likelihood for $N$ independent examples with predicted probabilities $\hat p_t = \sigma(z_t)$, then show $\partial \mathcal L / \partial z_t = (\hat p_t - y_t)/N$ for the mean loss.

<details><summary>Solution</summary>

Likelihood: $L = \prod_t \hat p_t^{y_t}(1-\hat p_t)^{1-y_t}$. Log: $\ell = \sum_t [y_t\ln\hat p_t + (1-y_t)\ln(1-\hat p_t)]$. Mean negative log-likelihood: $\mathcal L = -\ell/N$ - this is BCE.
Only the $t$-th term depends on $z_t$. With $\frac{d\hat p_t}{dz_t} = \hat p_t(1-\hat p_t)$:
$\frac{\partial \mathcal L}{\partial z_t} = -\frac1N\Big[\frac{y_t}{\hat p_t} - \frac{1-y_t}{1-\hat p_t}\Big]\hat p_t(1-\hat p_t) = -\frac1N[y_t(1-\hat p_t) - (1-y_t)\hat p_t] = -\frac1N[y_t - \hat p_t] = \frac{\hat p_t - y_t}{N}.$
</details>

**Exercise 4 (M, ★★).** Show that $\ln(1 + e^z) - yz$ equals the BCE term $-[y\ln\sigma(z) + (1-y)\ln(1-\sigma(z))]$, and then that $\ln(1 + e^z) = \max(z, 0) + \ln(1 + e^{-|z|})$. Why does the second identity matter?

<details><summary>Solution</summary>

$\ln\sigma(z) = -\ln(1 + e^{-z})$ and $\ln(1-\sigma(z)) = \ln\sigma(-z) = -\ln(1+e^{z})$. So the term is $y\ln(1+e^{-z}) + (1-y)\ln(1+e^{z})$. Using $\ln(1 + e^{-z}) = \ln\big(e^{-z}(e^{z}+1)\big) = -z + \ln(1+e^z)$, it becomes $-yz + y\ln(1+e^z) + (1-y)\ln(1+e^z) = \ln(1+e^z) - yz$.
For the second identity: if $z \ge 0$, $\ln(1+e^z) = \ln(e^z(e^{-z}+1)) = z + \ln(1+e^{-z}) = \max(z,0) + \ln(1+e^{-|z|})$. If $z < 0$, $\max(z,0) = 0$ and $|z| = -z$, so the right side is $\ln(1 + e^{z})$. ∎
It matters because $e^z$ overflows for $z > 709$ in float64 (and $z > 88$ in float32), whereas $e^{-|z|} \le 1$ never overflows. This is the formula used by `binary_cross_entropy_with_logits`.
</details>

**Exercise 5 (M, ★★).** In a training set, positives are kept with probability 1 and negatives with probability $s$. Prove that, for any input $x$, $\mathrm{logit}\,P(y=1\mid x,\text{kept}) = \mathrm{logit}\,P(y=1\mid x) - \ln s$. For MV-HGAT on Fdataset, estimate $\ln(1/s)$ and interpret it.

<details><summary>Solution</summary>

By Bayes, $P(y=1\mid x,\text{kept}) \propto P(\text{kept}\mid y=1)P(y=1\mid x) = P(y=1\mid x)$ and $P(y=0\mid x,\text{kept}) \propto s\,P(y=0\mid x)$, with the same proportionality constant $1/P(\text{kept}\mid x)$. Their ratio (the odds) is $\frac{1}{s}\cdot\frac{P(y=1\mid x)}{P(y=0\mid x)}$; taking logs gives the result.
For Fdataset the training ratio is 1 positive : 2 negatives (odds 0.5) while the full-data odds are $1933/183{,}676 = 0.01052$. So $1/s \approx 0.5/0.01052 = 47.5$ and $\ln(1/s) \approx 3.86$. The model's logits are about 3.86 too high relative to calibrated log-odds of a recorded link; probabilities are inflated (e.g. a calibrated 1% would display as about 32%). Rankings are unchanged.
</details>

**Exercise 6 (P, ★★).** Write code that confirms numerically that $\sigma(\ln x) = x/(1+x)$ for $x \in \{1, 4, 11\}$ and use it to state the initial degree-gate value of `MVHGAT.gnn_gate` for a drug with 3 visible links and a disease with 10 visible links.

<details><summary>Solution</summary>

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
for x in [1, 4, 11]:
    print(x, round(sig(np.log(x)), 4), round(x / (1 + x), 4))
deg_r, deg_d = 3, 10
gate = sig(0 + 1 * np.log1p(deg_r)) * sig(0 + 1 * np.log1p(deg_d))
print("initial gate:", round(gate, 4))
```

```text
1 0.5 0.5
4 0.8 0.8
11 0.9167 0.9167
initial gate: 0.7333
```

With `gate=[0,1,0,1]`, each factor is $\sigma(\ln(1+d)) = (1+d)/(2+d)$: drug $4/5 = 0.8$, disease $11/12 = 0.9167$, product $0.7333$. The GNN term is multiplied by 0.73 at initialisation; training then adjusts the four gate parameters.
</details>

**Exercise 7 (M, ★★).** Five folds give AUCs 0.90, 0.92, 0.91, 0.93, 0.89. Compute the mean, the SD with `ddof=0` and `ddof=1`, the SE, and a 95% t-interval ($t_{0.975,4} = 2.776$).

<details><summary>Solution</summary>

Mean $= 4.55/5 = 0.910$. Deviations: $-0.01, +0.01, 0, +0.02, -0.02$; squares sum $= 0.0001+0.0001+0+0.0004+0.0004 = 0.0010$.
`ddof=0`: $\sqrt{0.0010/5} = \sqrt{0.0002} = 0.01414$. `ddof=1`: $\sqrt{0.0010/4} = \sqrt{0.00025} = 0.01581$.
SE $= 0.01581/\sqrt5 = 0.00707$. Half-width $= 2.776 \times 0.00707 = 0.01963$. CI $= [0.890, 0.930]$. (If these are CV folds, the corrected half-width is $2.776 \times 0.01581\times\sqrt{1/5 + 1/4} = 0.0294$, giving [0.881, 0.939].)
</details>

**Exercise 8 (C/M, ★★).** Explain without code why a random ranker's AUC is 0.5 regardless of the base rate, while its AUPR equals the base rate. What are the random AUPRs for Fdataset and Cdataset?

<details><summary>Solution</summary>

AUC = P(score of random positive > score of random negative). For a random ranker, both scores are i.i.d. draws from the same continuous distribution; by symmetry each is equally likely to be the larger, so the probability is 1/2. The class proportions never enter.
AUPR averages precision over the positions of the positives. In a random permutation, any top-$m$ set is a random subset of the items, whose expected positive fraction is the overall fraction $\pi$. So expected precision is $\pi$ at every cutoff and AUPR ≈ $\pi$.
Fdataset: $1933/185{,}609 = 0.0104$. Cdataset: $2532/271{,}167 = 0.0093$.
</details>

**Exercise 9 (M, ★★).** A case study finds that 7 of the top 10 predicted drugs for a disease have supporting evidence. Compute the Wald and Wilson 95% intervals for the true "hit rate" and say which you would report.

<details><summary>Solution</summary>

$\hat p = 0.7$, $k = 10$, $z = 1.96$, $z^2 = 3.8416$.
Wald: $0.7 \pm 1.96\sqrt{0.21/10} = 0.7 \pm 1.96 \times 0.1449 = 0.7 \pm 0.284 = [0.416, 0.984]$.
Wilson: centre $= (0.7 + 3.8416/20)/(1 + 0.38416) = 0.89208/1.38416 = 0.6445$; half-width $= 1.96\sqrt{0.021 + 3.8416/400}/1.38416 = 1.96\sqrt{0.030604}/1.38416 = 1.96 \times 0.17494/1.38416 = 0.2477$. Interval $[0.397, 0.892]$.
Report Wilson: Wald is known to undercover for small $k$, and its upper end (0.984) is unrealistically close to 1. Either way, the message is that 10 predictions only pin the hit rate down to roughly 40-90%.
</details>

**Exercise 10 (P, ★★).** Using the per-fold **AUC** values of MV-HGAT and SCMFDD (Lab 14), compute the paired t-test p-value, the corrected repeated-CV p-value, and the fraction of folds MV-HGAT wins.

<details><summary>Solution</summary>

```python
import numpy as np
from scipy import stats
mv = np.array([0.9402,0.9395,0.9349,0.9557,0.9467,0.9344,0.9366,0.9499,0.9421,0.9307,0.9402,0.9438,0.9282,
               0.9426,0.9382,0.9377,0.9420,0.9436,0.9364,0.9445,0.9287,0.9486,0.9279,0.9263,0.9417])
sc = np.array([0.8971,0.8707,0.8931,0.9067,0.9044,0.8837,0.8947,0.9002,0.8856,0.8892,0.8998,0.8940,0.8775,
               0.9047,0.8838,0.9078,0.8996,0.8939,0.8777,0.9031,0.8944,0.9109,0.8918,0.8807,0.8911])
d = mv - sc
print("wins:", int((d > 0).sum()), "of", len(d))
print("paired t p =", f"{stats.ttest_rel(mv, sc).pvalue:.2e}")
t_c = d.mean() / np.sqrt((1/25 + 1/4) * d.var(ddof=1))
print("corrected p =", f"{2 * stats.t.sf(abs(t_c), 24):.2e}")
```

```text
wins: 25 of 25
paired t p = 2.08e-19
corrected p = 5.27e-10
```

MV-HGAT wins every fold. The naive paired test gives an absurdly small p-value; the corrected test is much larger but still overwhelming. The AUC advantage is real.
</details>

**Exercise 11 (C, ★★★).** A student runs 5 repeats of 5-fold CV and gets 25 AUPR values; they then run 50 repeats (250 values) "to make the CI ten times narrower than before". Explain what goes wrong, using the corrected variance formula.

<details><summary>Solution</summary>

The naive SE $s/\sqrt{n}$ shrinks by $\sqrt{10} \approx 3.2$ (not 10) when $n$ goes from 25 to 250. But the honest variance factor is $1/(kr) + 1/(k-1)$. For $k = 5$: with $r = 5$, $1/25 + 1/4 = 0.29$; with $r = 50$, $1/250 + 1/4 = 0.254$. The corrected SE shrinks only by $\sqrt{0.29/0.254} = 1.07$, i.e. about 7%. Repeats average out *split and seed* noise but cannot remove the uncertainty due to having a single finite data set; they all reuse the same 1,933 links. The student's narrow naive CI would be badly overconfident.
</details>

**Exercise 12 (P, ★★★).** Simulate the Type I error rate (false-positive rate) of the naive paired t-test on repeated CV folds. Create data sets in which two models have *exactly the same true performance*: the label depends equally on two features, model A uses only feature 0 and model B only feature 1. On each simulated data set run 5 repeats of 5-fold CV, compute the 25 per-fold AUC differences, and test them with (a) the naive one-sample t-test and (b) the corrected repeated-CV t-test. Repeat for 200 data sets. What fraction of data sets give "significant" differences at 0.05?

<details><summary>Solution</summary>

```python
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
trials, naive_rej, corr_rej = 200, 0, 0
for trial in range(trials):
    # two features that are EQUALLY informative in the population
    n = 200
    X = rng.normal(size=(n, 2))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))).astype(int)
    d = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=trial)
    for tr, te in cv.split(X, y):
        a = LogisticRegression().fit(X[tr][:, [0]], y[tr])     # model A uses feature 0
        b = LogisticRegression().fit(X[tr][:, [1]], y[tr])     # model B uses feature 1
        d.append(roc_auc_score(y[te], a.decision_function(X[te][:, [0]]))
                 - roc_auc_score(y[te], b.decision_function(X[te][:, [1]])))
    d = np.array(d)
    naive_rej += stats.ttest_1samp(d, 0).pvalue < 0.05
    t_c = d.mean() / np.sqrt((1 / 25 + 1 / 4) * d.var(ddof=1))
    corr_rej += 2 * stats.t.sf(abs(t_c), 24) < 0.05
print(f"naive paired t-test false-positive rate : {naive_rej / trials:.3f}")
print(f"corrected test false-positive rate      : {corr_rej / trials:.3f}")
```

```text
naive paired t-test false-positive rate : 0.385
corrected test false-positive rate      : 0.055
```

Because the models are equally good in the population, every rejection is a false positive and a well-calibrated test should reject about 5% of the time. The naive test rejects in about 38% of the data sets: on any *particular* finite data set one feature happens to look a bit better, that advantage shows up in all 25 folds (they reuse the same 200 examples), and the naive test mistakes this data-set-level luck for a real difference. The corrected test, which inflates the variance by $1/(k-1)$ to account for the overlap, rejects at close to the nominal 5%. This is exactly the phenomenon described by Dietterich (1998) and Bouckaert & Frank (2004), and the reason the project's comparisons should use the corrected test. (Runtime: about one minute.)
</details>

**Exercise 13 (M, ★★).** Prove that $E[\bar x] = \mu$ and $\mathrm{Var}(\bar x) = \sigma^2/n$ for i.i.d. $x_t$, and show where the proof breaks if the $x_t$ have pairwise correlation $\rho > 0$. Derive $\mathrm{Var}(\bar x)$ in that case.

<details><summary>Solution</summary>

$E[\bar x] = \frac1n\sum E[x_t] = \mu$ (linearity; no independence needed).
$\mathrm{Var}(\bar x) = \frac{1}{n^2}\mathrm{Var}(\sum x_t) = \frac{1}{n^2}\big[\sum_t \mathrm{Var}(x_t) + \sum_{s\ne t}\mathrm{Cov}(x_s, x_t)\big]$. With independence the covariances vanish: $\frac{n\sigma^2}{n^2} = \sigma^2/n$.
With pairwise correlation $\rho$, each of the $n(n-1)$ ordered pairs contributes $\rho\sigma^2$:
$\mathrm{Var}(\bar x) = \frac{1}{n^2}[n\sigma^2 + n(n-1)\rho\sigma^2] = \sigma^2\Big[\frac1n + \frac{n-1}{n}\rho\Big]$.
As $n \to \infty$ this tends to $\rho\sigma^2$, not 0: correlated runs can never average away their shared component. Nadeau & Bengio's correction has exactly this form, with $\rho \approx n_{\text{test}}/n_{\text{train}}$ standing in for the correlation between fold estimates.
</details>

---

## 15. Answers to the PREREQUISITES.md self-check questions (Unit A3)

### Self-check 1: "Derive binary cross-entropy from the Bernoulli likelihood."

**Setting.** We have $N$ training examples (drug-disease pairs) with labels $y_t \in \{0, 1\}$ (1 = known link, 0 = sampled unknown). The model outputs a logit $z_t$ for each, and we interpret $\hat p_t = \sigma(z_t) = 1/(1+e^{-z_t})$ as the probability that $y_t = 1$.

**Step 1 - one example.** Under the model, $Y_t \sim \mathrm{Bernoulli}(\hat p_t)$, whose PMF can be written as a single expression: $P(Y_t = y_t) = \hat p_t^{\,y_t}(1-\hat p_t)^{1-y_t}$. (For $y_t = 1$ it gives $\hat p_t$, for $y_t = 0$ it gives $1-\hat p_t$.)

**Step 2 - all examples.** Assume the labels are conditionally independent given the model's outputs. Then the probability of the whole label vector - the likelihood of the parameters $\theta$ that produced the $\hat p_t$ - is the product:
$$L(\theta) = \prod_{t=1}^N \hat p_t^{\,y_t}(1-\hat p_t)^{1-y_t}.$$

**Step 3 - take logs.** The logarithm is strictly increasing, so it has the same maximiser; it turns the product into a sum (avoiding underflow - the Fdataset likelihood is about $e^{-10{,}746}$, which is 0 in floating point):
$$\ell(\theta) = \sum_{t=1}^N \big[y_t\ln\hat p_t + (1-y_t)\ln(1-\hat p_t)\big].$$

**Step 4 - turn maximisation into a loss.** Optimisers minimise, so negate; and divide by $N$ to make the scale independent of the batch size:
$$\mathcal L_{\text{BCE}}(\theta) = -\frac1N\sum_{t=1}^N\big[y_t\ln\hat p_t + (1-y_t)\ln(1-\hat p_t)\big].$$

That is binary cross-entropy. Therefore **minimising BCE ≡ maximum-likelihood estimation of a Bernoulli model** whose success probability is $\sigma(z_t)$.

**Why it is called cross-entropy.** For each example, $-[y\ln\hat p + (1-y)\ln(1-\hat p)]$ is the cross-entropy $H(q, r)$ between the "true" distribution $q = (1-y, y)$ and the model's $r = (1-\hat p, \hat p)$; since $H(q) = 0$ for a point mass, it also equals $\mathrm{KL}(q\|r)$.

**Consequences worth knowing.**
* Gradient with respect to the logit: $\partial\mathcal L/\partial z_t = (\hat p_t - y_t)/N$ - prediction minus label.
* Stable form used by PyTorch: $\max(z,0) - zy + \ln(1+e^{-|z|})$, which is why `fit_predict` passes logits to `F.binary_cross_entropy_with_logits` instead of probabilities.
* BCE is a proper scoring rule: expected loss is minimised by predicting the true probability, so BCE training produces calibrated probabilities *for the training distribution*. In this project the training distribution has 1 positive per 2 negatives, so the outputs are shifted by about $+3.86$ in logit space relative to the true 1% rate; ranking metrics are unaffected.
* If we add a Gaussian prior on the weights and maximise the posterior instead, we get BCE + L2 penalty: the `weight_decay` term.

### Self-check 2: "If 1% of pairs are positive, what AUPR does a random ranker get? What AUC?"

**AUC ≈ 0.5.** AUC is the probability that a randomly chosen positive is scored above a randomly chosen negative. A random ranker gives every pair an independent score from the same distribution, so for any (positive, negative) pair, each is equally likely to come out on top: probability 1/2. The base rate plays no role. In finite samples the observed AUC fluctuates around 0.5 with SD $\sqrt{(n_1+n_0+1)/(12 n_1 n_0)}$: about 0.015 for one Fdataset test fold, 0.0066 for the whole matrix (Lab 12 measured 0.4996 ± 0.0065).

**AUPR ≈ 0.01 (= the base rate).** AUPR (average precision) is the mean of the precision values at the positions where positives appear in the ranking. For a random ranking, whatever cutoff you choose, the items above it are a random sample, so their expected fraction of positives is the overall fraction, 1%. Averaging precisions that each have expectation 1% gives ≈ 1%. More precisely, for Fdataset the random AUPR is 0.0104 and for Cdataset 0.0093 (Lab 12 measured 0.0105 ± 0.0003).

**Why "1% changes everything".**
* Accuracy becomes meaningless ("always no" = 99%).
* AUC can look excellent (0.94) while the top of the ranking still contains many false positives, because there are ~95 negatives per positive: a 6% mis-ordering rate means thousands of negatives above a typical positive.
* AUPR's floor moves from 0.5 (balanced) to 0.01. An AUPR of 0.49 - which would be *worse than random* on a balanced problem - is 47× better than random here. You must always quote AUPR next to its baseline.
* Even a classifier with 90% sensitivity and a 5% false-positive rate has only ~16% precision at a 1% base rate (Bayes' theorem, section 2.4). Most "hits" of a screening model are false alarms unless its false-positive rate is tiny - which is why case-study validation (Unit D1) matters.

---

## 16. Summary and cheat sheet

**Probability basics**
* $P(A\mid B) = P(A\cap B)/P(B)$; Bayes: $P(A\mid B) = P(B\mid A)P(A)/P(B)$.
* $E[aX+bY] = aE[X]+bE[Y]$ always. $\mathrm{Var}(X) = E[X^2]-E[X]^2$. $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$.
* $\mathrm{Var}(X - Y) = \mathrm{Var}X + \mathrm{Var}Y - 2\mathrm{Cov}(X,Y)$ ← why pairing works.

**Distributions**
* Bernoulli($p$): $P(y) = p^y(1-p)^{1-y}$, mean $p$, variance $p(1-p)$.
* Binomial($n,p$): $\binom nk p^k(1-p)^{n-k}$, mean $np$, variance $np(1-p)$.

**Sigmoid and logits**
* $\sigma(z) = 1/(1+e^{-z})$, $\mathrm{logit}(p) = \ln\frac{p}{1-p}$, inverses of each other.
* $\sigma(-z) = 1-\sigma(z)$; $\sigma'(z) = \sigma(z)(1-\sigma(z))$; odds $= e^{z}$.

**Likelihood**
* $L(p) = \prod p^{y_t}(1-p)^{1-y_t}$; $\ell = k\ln p + (n-k)\ln(1-p)$; MLE $\hat p = k/n$.
* MAP with Gaussian prior = NLL + L2 penalty (weight decay).

**BCE**
* $\mathcal L = -\frac1N\sum[y\ln\hat p + (1-y)\ln(1-\hat p)]$ = mean Bernoulli NLL = cross-entropy.
* $\partial\mathcal L/\partial z = (\hat p - y)/N$.
* Stable: $\max(z,0) - zy + \ln(1+e^{-|z|})$ → use `binary_cross_entropy_with_logits`.
* Negative sampling at rate $s$ adds $\ln(1/s)$ to logits; rankings unchanged. Fdataset with 1:2 sampling: ≈ +3.86.

**Estimation and reporting**
* $\bar x$; $s^2 = \frac{1}{n-1}\sum(x-\bar x)^2$; NumPy `std()` is `ddof=0`, pandas `std()` is `ddof=1`.
* SE $= s/\sqrt n$ (independent data only). SD describes runs; SE describes the mean.
* 95% t-interval: $\bar x \pm t_{0.975,n-1}\,s/\sqrt n$; $t_{0.975,4}=2.776$, $t_{0.975,24}=2.064$.
* Repeated $k$-fold CV corrected variance: $(\frac{1}{kr} + \frac{1}{k-1})s^2$.
* Always state: mean ± SD (or SE), over what, how many, which ddof.

**Base rates**
* Fdataset π = 0.0104, Cdataset π = 0.0093.
* Random ranker: AUC = 0.5 (any π); AUPR ≈ π; precision@k ≈ π.
* Null SD of AUC $= \sqrt{(n_1+n_0+1)/(12n_1n_0)}$.

**Testing**
* p-value = P(result at least this extreme | $H_0$). Report effect size + CI too.
* Paired designs: same folds for all methods (true in `run_kfold`).
* Tests: paired t, Wilcoxon signed-rank, sign, permutation; for CV use the corrected repeated k-fold t-test: $t = \bar d/\sqrt{(\frac{1}{kr}+\frac{1}{k-1})s_d^2}$, df $= kr-1$.
* Multiple comparisons: Holm (step-down Bonferroni).
* Project result: AUC gain over all baselines is significant; AUPR vs SCMFDD on Fdataset is not.

---

## 17. Curated further resources

All links were checked and resolved on 2026-10-01 (a few publisher pages block automated checks but are standard DOI/publisher links).

**Courses and lecture notes**
* [MIT 18.05 Introduction to Probability and Statistics (OCW, Spring 2022)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) - **Free.** Best single course for this unit: complete reading notes, problem sets with solutions, covering random variables through MLE, CIs, hypothesis tests and the bootstrap.
* [Harvard Stat 110: Probability (lecture videos playlist)](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) - **Free.** Joe Blitzstein's legendary lectures; best for deep intuition about random variables, expectation and conditioning.
* [Introduction to Probability, Blitzstein & Hwang (free online edition)](http://probabilitybook.net) - **Free online / paid print.** The Stat 110 textbook; superb worked examples and exercises.
* [Stanford CS109: Probability for Computer Scientists](https://web.stanford.edu/class/cs109/) and its [course reader](https://chrispiech.github.io/probabilityForComputerScientists/en/) - **Free.** Probability taught for programmers, including MLE, logistic regression and bootstrapping with code.
* [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) - **Free.** Interactive visualisations; ideal for building intuition on distributions, the CLT and [frequentist inference / confidence intervals](https://seeing-theory.brown.edu/frequentist-inference/index.html).
* [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability) - **Free.** Gentle refresher if any of the basics (variance, normal distribution, t-tests) feel shaky.

**Videos (StatQuest, Josh Starmer - all free)**
* [In Statistics, Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) - the cleanest visual explanation of the probability-vs-likelihood distinction.
* [Maximum Likelihood, clearly explained](https://www.youtube.com/watch?v=XepXtl9YKwc) - the idea of MLE in 6 minutes.
* [Odds Ratios and Log(Odds Ratios), Clearly Explained](https://www.youtube.com/watch?v=8nm0G-1uJzA) - builds intuition for odds and log-odds before logistic regression.
* [Logistic Regression Details Pt 2: Maximum Likelihood](https://www.youtube.com/watch?v=BfKanl1aSG0) - how the log-likelihood of logistic regression is computed and maximised.
* [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) - cross-entropy as a loss for neural networks.
* [ROC and AUC, Clearly Explained](https://www.youtube.com/watch?v=4jRBRDbJemM) - preparation for Unit B2.
* [StatQuest video index](https://statquest.org/video_index.html) - find related videos (t-tests, p-values, bootstrapping).

**Books**
* [Murphy, *Probabilistic Machine Learning: An Introduction* (2022)](https://probml.github.io/pml-book/book1.html) - **Free PDF.** Ch. 2-4 (probability, statistics, MLE/MAP) and ch. 10 (logistic regression) are the best ML-oriented treatment of everything in this unit.
* [Bishop, *Pattern Recognition and Machine Learning* (2006)](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) - **Free PDF from Microsoft Research.** Ch. 1-2 (probability, distributions, information theory) and §4.3 (logistic regression, cross-entropy).
* [Wasserman, *All of Statistics*](https://www.stat.cmu.edu/~larry/all-of-statistics/) - **Paid book (companion site free).** Concise, rigorous coverage of estimation, CIs, bootstrap and testing for people coming from CS.
* [MacKay, *Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itila/book.html) - **Free PDF.** For the entropy / cross-entropy / KL view.
* [Downey, *Think Stats* (3rd ed.)](https://greenteapress.com/wp/think-stats-3e/) - **Free online.** Statistics through Python code; great for practising the computational side.

**Papers on comparing models (read these before writing the Results section)**
* [Nadeau & Bengio (2003), "Inference for the Generalization Error", *Machine Learning* 52](https://link.springer.com/article/10.1023/A:1024068626366) - **Paywalled (preprints exist).** Origin of the corrected resampled t-test.
* [Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", *Neural Computation* 10(7)](https://doi.org/10.1162/089976698300017197) - **Paywalled (preprints exist).** Classic study showing naive CV t-tests have inflated Type I error; proposes 5×2cv.
* [Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets", *JMLR* 7](https://jmlr.org/papers/v7/demsar06a.html) - **Free.** Wilcoxon and Friedman tests, critical-difference diagrams.
* [Raschka (2018), "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning"](https://arxiv.org/abs/1811.12808) - **Free.** Very readable survey of CIs, bootstrap and statistical tests for ML; bridges this unit and B1.
* [Bouthillier et al. (2021), "Accounting for Variance in Machine Learning Benchmarks"](https://arxiv.org/abs/2103.03098) - **Free.** Why you must randomise seeds *and* splits, and how many runs you need.
* [Saito & Rehmsmeier (2015), "The Precision-Recall Plot Is More Informative than the ROC Plot ... on Imbalanced Datasets", *PLOS ONE*](https://doi.org/10.1371/journal.pone.0118432) - **Free.** The base-rate argument for AUPR.

**Documentation**
* [PyTorch `BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) - **Free.** Exact formula, `pos_weight`, and the stability note.
* [SciPy `ttest_rel`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) and [`wilcoxon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) - **Free.** Arguments and assumptions of the paired tests used in Labs 13-14.

---

## 18. Glossary

* **AUC (ROC AUC):** probability that a random positive is scored above a random negative; 0.5 = random.
* **AUPR / average precision:** average of the precision values at the ranks of the positives; random ≈ base rate.
* **Base rate (prevalence) π:** fraction of positives in the population; 1.04% for Fdataset.
* **Bayes' theorem:** $P(A\mid B) = P(B\mid A)P(A)/P(B)$.
* **Bernoulli distribution:** distribution of a single 0/1 outcome with $P(1) = p$.
* **Bessel's correction:** dividing by $n-1$ rather than $n$ in the sample variance (`ddof=1`).
* **Binary cross-entropy (BCE, log loss):** mean negative Bernoulli log-likelihood.
* **Binomial distribution:** number of successes in $n$ independent Bernoulli trials.
* **Bootstrap:** estimating a sampling distribution by resampling the data with replacement.
* **Calibration:** agreement between predicted probabilities and observed frequencies.
* **Central limit theorem (CLT):** sample means are approximately normal for large $n$.
* **Conditional probability:** probability of an event given that another occurred.
* **Confidence interval (CI):** interval from a procedure that covers the true parameter in a stated fraction of repeated experiments.
* **Corrected resampled t-test:** t-test for CV results with variance inflated by $1/(k-1)$ to account for overlapping training sets.
* **Covariance / correlation:** measures of how two random variables vary together.
* **Cross-entropy $H(q,r)$:** expected surprise under model $r$ when outcomes come from $q$.
* **ddof:** "delta degrees of freedom" in NumPy; the divisor is $n - \text{ddof}$.
* **Effect size:** the magnitude of a difference (e.g. mean AUPR difference), as opposed to its significance.
* **Entropy $H(q)$:** expected surprise of a distribution.
* **Estimator / unbiased:** a rule mapping data to a parameter guess; unbiased if its expectation equals the truth.
* **Event / sample space:** a set of outcomes / the set of all outcomes.
* **Expectation (mean) $E[X]$:** probability-weighted average of a random variable.
* **Fisher information:** curvature of the log-likelihood; inverse gives the MLE's asymptotic variance.
* **Holm correction:** step-down multiple-comparison correction, more powerful than Bonferroni.
* **Independence:** $P(A\cap B) = P(A)P(B)$.
* **KL divergence:** $\mathrm{KL}(q\|r) = H(q,r) - H(q) \ge 0$.
* **Law of large numbers:** sample averages converge to the expectation.
* **Lift:** metric divided by its random-baseline value.
* **Likelihood $L(\theta)$:** probability of the observed data as a function of the parameters.
* **Logit / log-odds:** $\ln\frac{p}{1-p}$; the raw output of the model before the sigmoid.
* **MAP estimation:** maximising likelihood × prior; with a Gaussian prior gives L2 regularisation.
* **Maximum likelihood estimate (MLE):** parameter value that maximises the likelihood.
* **Negative sampling:** training with a random subset of negatives (2 per positive here).
* **Null hypothesis $H_0$:** the "no effect" hypothesis a test tries to reject.
* **Odds:** $p/(1-p)$.
* **Paired test:** a test on per-unit differences between two methods evaluated on the same units (folds).
* **Permutation / sign-flip test:** test whose null distribution is generated by randomly flipping or permuting labels.
* **Power:** probability that a test detects a real effect ($1 - \beta$).
* **Precision:** fraction of predicted positives that are true positives.
* **Proper scoring rule:** a loss whose expectation is minimised by reporting the true probability.
* **p-value:** probability under $H_0$ of a statistic at least as extreme as observed.
* **Random ranker:** a scorer that assigns i.i.d. random scores; the "no skill" baseline.
* **Random variable:** a numerical function of a random outcome.
* **Sigmoid $\sigma(z)$:** $1/(1+e^{-z})$, maps logits to (0, 1).
* **Sign test:** test based only on the count of positive differences.
* **Softplus:** $\ln(1+e^u)$.
* **Standard deviation (SD):** square root of the variance; spread of individual values.
* **Standard error (SE):** SD of an estimator such as the mean; $s/\sqrt n$ under independence.
* **Stratification:** splitting so that each fold has the same class proportions.
* **Student's t distribution:** distribution of the standardised mean when the SD is estimated.
* **Sufficient statistic:** a summary of the data that carries all the information about a parameter (e.g. $k$ for the Bernoulli).
* **Type I / Type II error:** false positive / false negative of a statistical test.
* **Variance:** expected squared deviation from the mean.
* **Wilcoxon signed-rank test:** non-parametric paired test using ranks of absolute differences.
* **Wilson interval:** a well-behaved CI for a proportion.
