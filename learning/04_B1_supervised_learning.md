# Unit B1 - Supervised Learning Basics

> **Course:** Drug Repositioning with Graph Neural Networks - Track B (Machine learning)
> **Prerequisites:** Unit A1 (Python/NumPy), Unit A2 (Linear algebra: vectors, matrix products, norms), Unit A3 (Probability and statistics: likelihood, BCE, mean/SD, confidence intervals). Calculus: derivatives, partial derivatives, the chain rule, the idea of a gradient as the vector of partial derivatives.
> **Estimated study time:** 14-18 hours (7 hours reading, 5 hours code labs, 4-6 hours exercises).

---

## Learning objectives

After this unit you will be able to:

1. **Formulate** a supervised learning problem precisely: inputs, labels, hypothesis class, parameters, loss, empirical risk and true risk, and **map** each onto the drug-disease link-prediction task.
2. **Explain** generalisation, the generalisation gap, and why a score measured on data used for any decision is optimistic; **quantify** this with the winner's-curse argument.
3. **Design** a correct train/validation/test protocol, and **identify** leakage in a given protocol (including a subtle one in this project).
4. **Derive** the bias-variance decomposition of the squared error and **use** it to diagnose under- and over-fitting.
5. **Explain mechanistically** what L2 regularisation/weight decay, dropout and early stopping do, derive the weight-decay update, and **state** the difference between L2-in-Adam and AdamW.
6. **Choose** a loss function for a task and **justify** it from a likelihood (MSE ↔ Gaussian, BCE ↔ Bernoulli).
7. **Derive and implement** gradient descent, SGD, momentum and Adam; **predict** the effect of the learning rate on convergence for a quadratic; **compute** an Adam step by hand.
8. **Distinguish** parameters from hyper-parameters and **classify** every field of `MVHGATConfig`.
9. **Implement** k-fold, stratified, repeated and grouped cross-validation, and **explain** what CV estimates.
10. **Design** a hyper-parameter search (grid, random, one-at-a-time) on a validation split and **explain** why nested CV is needed for an unbiased estimate of a tuned procedure.

---

## 1. Motivation: why this unit matters for this project

Open `src/drepo/methods.py` and look at `MVHGATConfig`. It has 23 fields: `k=10`, `hidden=64`, `layers=2`, `heads=4`, `dropout=0.2`, `lr=2e-3`, `weight_decay=5e-4`, `epochs=600`, `neg_ratio=2`, `drop_edge=0.2`, `cold_frac=0.1`, and more. **None of these is learned by the model.** Somebody had to choose them. The model's ~436,000 *parameters* (weights) are learned by Adam; these 23 *hyper-parameters* are chosen by you. How you choose them decides whether the numbers you publish are honest.

Here is the trap. Suppose you try 50 combinations of hyper-parameters, evaluate each with 5-fold CV on Fdataset, and report the best one's AUPR. Even if all 50 were equally good in truth, the best of 50 noisy estimates is higher than the truth (section 10.2 shows by simulation that picking the best of 30 equally good configurations with noise SD 0.02 inflates the score by about 0.04 - six times the AUPR gap between MV-HGAT and SCMFDD (0.007) and almost as large as MV-HGAT's AUC lead over the best baseline (0.046)). Your "result" would partly be luck you selected for. That is why `scripts/05_sensitivity.py` begins:

```python
# (quoted from scripts/05_sensitivity.py, module docstring)
"""
To avoid tuning on the test data, we carve a VALIDATION split out of the data:
20% of the known links and 20% of the unknown pairs are hidden, the model
trains on the rest, and we score on the hidden 20%. One hyper-parameter is
varied at a time around the defaults in MVHGATConfig.
"""
```

The project's most important engineering lesson is also a supervised-learning lesson. `docs/HOW_IT_WORKS.md` §10 reports: "A GNN supervised on links it can see in its own input graph learns to detect edges, not to predict them (validation AUC 0.71). Supervising only on links hidden each epoch fixed it (0.92...)". That is a story about **generalisation**: a model that does brilliantly on its training signal (spotting edges that are in its input) and poorly on what we actually care about (predicting edges that are *not* in its input). Recognising this pattern - training performance that does not transfer - is the core skill of this unit.

Finally, the training loop itself - `torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)`, `opt.zero_grad()`, `loss.backward()`, `opt.step()`, `model.train()`/`model.eval()` - is pure B1 material. After this unit you will be able to explain every one of those lines, and what would happen if you changed each number.

---

## 2. The supervised learning problem

### 2.1 Intuition

Supervised learning means learning a function from examples where the right answer is given. You show the algorithm many (input, answer) pairs and it adjusts itself so that its answers match; then you hope it gives good answers for inputs it has never seen. That hope - performance on *new* inputs - is the whole point. Memorising the training examples is easy; generalising is hard.

### 2.2 Formal setup

* **Input space** $\mathcal X$ and **label space** $\mathcal Y$. An **example** is a pair $(x, y)$.
* The examples come from an unknown **data-generating distribution** $\mathcal D$ over $\mathcal X \times \mathcal Y$. The standard assumption is that training and test examples are **i.i.d.** (independent and identically distributed) draws from $\mathcal D$.
* A **model** or **hypothesis** $f_\theta: \mathcal X \to \mathbb R$ with **parameters** $\theta$ (the weights). The set of all functions the model can represent as $\theta$ varies is the **hypothesis class** $\mathcal F$.
* A **loss function** $\ell(f_\theta(x), y) \ge 0$ measures how bad a prediction is.
* The **true risk** (expected loss, generalisation error) is $R(\theta) = \mathbb E_{(x,y)\sim\mathcal D}\big[\ell(f_\theta(x), y)\big]$. This is what we want small, but we cannot compute it because $\mathcal D$ is unknown.
* The **empirical risk** on a training set $S = \{(x_t, y_t)\}_{t=1}^n$ is $\hat R_S(\theta) = \frac1n\sum_t \ell(f_\theta(x_t), y_t)$.
* **Empirical risk minimisation (ERM):** choose $\hat\theta = \arg\min_\theta \hat R_S(\theta)$, usually plus a regulariser: $\hat\theta = \arg\min_\theta \hat R_S(\theta) + \lambda\Omega(\theta)$.

Two flavours: **regression** ($\mathcal Y = \mathbb R$, e.g. predict a drug's IC50) and **classification** ($\mathcal Y$ finite, e.g. link / no link). With probabilistic outputs, binary classification is usually treated as predicting a probability $\sigma(f_\theta(x))$ (Unit A3).

### 2.3 Mapping onto this project

| Concept | In drug repositioning |
|---|---|
| Example $(x, y)$ | a (drug $i$, disease $j$) pair with label $A_{ij}$ |
| Input $x$ | everything the model can see about $i$ and $j$: similarity rows in 6 views, visible association rows, graph neighbourhoods |
| Label $y$ | 1 = recorded indication, 0 = not recorded (unknown!) |
| Model $f_\theta$ | MV-HGAT: GAT encoder + bilinear decoder + propagation head, outputs a logit |
| Parameters $\theta$ | ~436k weights (Lab 12) |
| Loss | BCE on hidden positives + 2 sampled negatives per positive |
| Training set | the visible links of the training folds, and negatives from `neg_mask` |
| Test set | one fold's positives + one fold's negatives |

Two features make this problem unusual, and you should keep them in mind throughout:

1. **The examples are not independent.** Pairs share drugs and diseases; the model's input for pair $(i, j)$ includes *other* links of $i$ and $j$ (transductive learning on a graph). Classical i.i.d. theory is a guide, not a guarantee.
2. **Negatives are unlabelled, not negative.** A 0 may be an undiscovered indication (positive-unlabelled learning). Some "errors" on 0s are actually discoveries.

---

## 3. Generalisation

### 3.1 Training error is optimistic

The training error $\hat R_S(\hat\theta)$ is computed on the same data that chose $\hat\theta$. Since $\hat\theta$ was selected to make exactly this number small, it is biased downward: $\mathbb E[\hat R_S(\hat\theta)] \le R(\hat\theta)$ typically. The **generalisation gap** is $R(\hat\theta) - \hat R_S(\hat\theta)$. A model with enough capacity can drive the training error to zero (Lab 6: training BCE 0.000) while the test error is large.

### 3.2 A fixed model's test error is unbiased - with a concentration guarantee

If a model $f$ is **fixed before** you look at a test set $T$ of $m$ i.i.d. examples, its test error $\hat R_T(f)$ is an *unbiased* estimate of $R(f)$, and for a loss bounded in $[0, 1]$ (e.g. 0-1 error) **Hoeffding's inequality** gives

$$P\big(|\hat R_T(f) - R(f)| > \varepsilon\big) \le 2e^{-2m\varepsilon^2}.$$

Example: $m = 1000$ test examples, $\varepsilon = 0.05$: probability at most $2e^{-5} = 0.013$ that the test error is off by more than 5 points.

### 3.3 Selecting among many models breaks the guarantee

Now suppose you evaluate $M$ models on the same test set and pick the best. The union bound gives $P(\text{any of them is off by} > \varepsilon) \le 2Me^{-2m\varepsilon^2}$ - the guarantee weakens by a factor $M$. Worse, the *winner* is preferentially one whose estimate was lucky. If all $M$ have the same true score $\mu$ and independent estimation noise with SD $\tau$, the winner's estimated score is on average $\mu + \tau\,\mathbb E[\max_{1..M} Z_i]$ where $Z_i \sim \mathcal N(0, 1)$: about $\mu + 1.16\tau$ for $M = 5$, $\mu + 1.87\tau$ for $M = 20$, $\mu + 2.04\tau$ for $M = 30$. This is the **winner's curse** (or selection bias, or "overfitting the validation set").

**The consequence:** any data used to make a decision (choose hyper-parameters, choose an epoch, choose an architecture, choose which result to report) can no longer give an unbiased estimate of the chosen model's performance. You need data that was **not** used for the decision. That is what the test set is for.

### 3.4 Capacity, under-fitting and over-fitting

* **Capacity** (complexity) is how rich the hypothesis class is: how many different patterns it can fit. More parameters, more layers, larger hidden size, fewer constraints → more capacity.
* **Under-fitting:** the model is too simple to capture the real pattern: both training and test errors are high. (E.g. a degree-1 polynomial for a sine wave; Lab 5.)
* **Over-fitting:** the model fits the noise and accidents of the training set: training error low, test error high. (Degree 9 with 30 points; or a 64-unit MLP on 300 examples, Lab 6.)

The classical picture is a U-shaped test-error curve as capacity grows: first decreasing (less under-fitting), then increasing (more over-fitting). Modern over-parameterised networks sometimes show a second descent past the point where they interpolate the training data ("double descent", Belkin et al. 2019); regularisation and the implicit bias of the optimiser play a role. For the moderate-size models here, the classical picture plus regularisation is the right mental model.

### 3.5 Learning curves: the diagnostic tool

Plot training and validation loss (or metric) against **epochs** or against **training-set size**:

```
loss                                     loss
 |  val                                    | val ------_______
 |   \        ____----  <- over-fitting    |                  ------  <- under-fitting:
 |    \___---                              | train _______-------         both high, gap small
 |  train \___                             |
 |            \_______                     |
 +-----------------------> epochs          +-----------------------> epochs
   gap grows: high variance                  gap small, error high: high bias
```

* Validation loss turning upward while training loss keeps falling → over-fitting: regularise, get more data, reduce capacity, or stop earlier.
* Both high and close → under-fitting: increase capacity, train longer, better features, reduce regularisation.

---

## 4. Train, validation and test

### 4.1 Three roles, three sets

| Set | Used for | Must not be used for |
|---|---|---|
| **Training** | fitting parameters by gradient descent | measuring final performance |
| **Validation** (development) | choosing hyper-parameters, architecture, epoch, features; comparing variants | reporting the final number |
| **Test** | one final, unbiased estimate of the chosen procedure | any decision whatsoever |

A common split is 60/20/20 or 80/10/10, but with small data (1,933 positives) a single fixed split wastes data and gives noisy estimates - hence cross-validation (section 9).

### 4.2 The golden rule

> **The test data must not influence any choice.** If you look at test performance and then change anything - a hyper-parameter, a feature, a random seed, which baseline to include - the test set has become a validation set, and its numbers are optimistic.

This includes "soft" leakage: running the full CV, seeing the result, adjusting the model, and running again. Each iteration leaks a little information. Professional practice is to finish all development on validation data, then run the test protocol once.

### 4.3 Leakage in link prediction

**Data leakage** is any way information about the test labels reaches training. In this project the subtle channels are:

1. **The test link in the input graph.** If the edge $(i, j)$ is present in the association graph or the feature rows while we predict $(i, j)$, the model can simply read it. `run_kfold` zeroes the test positives in `A_tr` before calling `fit_predict`, and inside `fit_predict` everything derived from links (`features`, `propagation`, `degrees`, the `assoc` graphs) is computed from the visible matrix only.
2. **Test negatives used as training negatives.** If a test 0 is also used as a training negative, the model is told its label. `neg_mask[test_neg] = False` prevents that, and `neg_pool` is built from `neg_mask`.
3. **Hyper-parameters tuned on test folds** (sections 10-11).
4. **Similarity computed from labels.** If a similarity view were computed from the association matrix itself (e.g. Gaussian interaction-profile kernels, used by some papers), it must be recomputed per fold from training links only. The project's views come from chemistry, genes, phenotypes and ontologies, which do not depend on `A`, so they are safe.

Unit E1 covers leakage in depth.

### 4.4 Hidden-link supervision: a train/test mismatch fixed inside training

Each epoch, `fit_predict` hides 20% of the training links (`drop_edge = 0.2`) from the graph and features and computes the loss **only on those hidden links** (`supervise_hidden = True`). Why does that help? At test time, the link being predicted is never visible. If training supervised links that *are* visible, the training task ("is this edge in my input?") differs from the test task ("is this missing edge real?"). The model learns a shortcut that does not generalise. Hiding the supervised links makes every training example look like a test example. This is the general principle **"train the way you test"** - make the training distribution of inputs match the test distribution. The cold-start practice (`cold_frac = 0.1`) applies the same principle to leave-one-disease-out testing.

---

## 5. The bias-variance decomposition

### 5.1 Setting

Regression with squared loss. Data: $y = f(x) + \varepsilon$, with $\mathbb E[\varepsilon] = 0$, $\mathrm{Var}(\varepsilon) = \sigma^2$, noise independent of everything else. A learning algorithm trained on a random training set $S$ produces a predictor $\hat f_S$. Because $S$ is random, $\hat f_S(x)$ at a fixed $x$ is a random variable. Define the **average predictor** $\bar f(x) = \mathbb E_S[\hat f_S(x)]$.

### 5.2 The theorem and its proof

$$\boxed{\mathbb E_{S,\varepsilon}\big[(y - \hat f_S(x))^2\big] = \underbrace{\sigma^2}_{\text{irreducible noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb E_S\big[(\hat f_S(x) - \bar f(x))^2\big]}_{\text{variance}}.}$$

*Proof.* Write $y - \hat f_S = (y - f) + (f - \bar f) + (\bar f - \hat f_S)$ (all at the same $x$). Call the three terms $a = \varepsilon$, $b = f - \bar f$ (a constant), $c = \bar f - \hat f_S$. Then

$$\mathbb E[(a + b + c)^2] = \mathbb E[a^2] + b^2 + \mathbb E[c^2] + 2b\,\mathbb E[a] + 2\,\mathbb E[ac] + 2b\,\mathbb E[c].$$

* $\mathbb E[a] = \mathbb E[\varepsilon] = 0$.
* $\mathbb E[c] = \bar f - \mathbb E_S[\hat f_S] = 0$ by definition of $\bar f$.
* $\mathbb E[ac] = \mathbb E[\varepsilon]\,\mathbb E[c] = 0$, because the test noise $\varepsilon$ is independent of the training set.

So all cross terms vanish, leaving $\sigma^2 + b^2 + \mathbb E[c^2]$. ∎

### 5.3 Interpretation

* **Bias** is systematic error: even averaged over infinitely many training sets, the model misses the truth because the hypothesis class cannot represent it (or regularisation pulls it away). Simple models: high bias.
* **Variance** is sensitivity to the particular training sample: different training sets give very different predictors. Flexible models on small data: high variance.
* **Noise** $\sigma^2$ is a floor no model can beat.

Lab 5 measures each term for polynomials of degree 1, 3, 5, 9 fitted to 30 noisy points of a sine wave: degree 1 has bias² 0.155 (under-fits), degree 3 has bias² 0.003 and variance 0.013 (best total, 0.106), degree 9 has tiny bias but variance 11.8 (catastrophic over-fitting). The sum bias² + variance + noise matches the measured test MSE to within simulation error in every row.

### 5.4 Bias-variance for classification and for this project

For 0-1 loss or BCE the decomposition is not as clean, but the qualitative trade-off holds. Project translations:

* `hidden=128` vs `hidden=64`: more capacity. Sensitivity results (`results/Fdataset/sensitivity.json`): AUPR 0.465 vs 0.521 - the larger model did worse on validation, a variance symptom.
* `layers=3` vs `2`: AUPR 0.479 vs 0.521, the same story (and deeper GNNs also over-smooth; Unit C3).
* `k=40` neighbours vs `k=10`: AUPR 0.475 vs 0.521 - not a capacity change but a noisier graph (more weak edges): more bias from blurring.
* The ensemble effect of **repeating** training with different seeds and averaging predictions reduces variance; reporting several seeds measures it.
* Regularisers (weight decay, dropout, DropEdge) deliberately add a little bias to remove a lot of variance.

---

## 6. Regularisation

**Regularisation** is any modification to the learning procedure intended to reduce test error, possibly at the cost of higher training error. It works by restricting or softening the model's capacity.

### 6.1 L2 regularisation (ridge, Tikhonov)

Add a penalty on the squared size of the weights:

$$J(\theta) = \hat R_S(\theta) + \frac{\lambda}{2}\|\theta\|_2^2.$$

**Why it helps.** Large weights make the function change sharply in response to small input changes; they are how a network fits noise. Penalising them favours smoother functions. In the bias-variance language: higher bias, much lower variance.

**The closed form for linear regression (ridge).** With squared loss $\frac12\|Xw - y\|^2 + \frac\lambda2\|w\|^2$, setting the gradient $X^\top(Xw - y) + \lambda w$ to zero gives

$$w_\lambda = (X^\top X + \lambda I)^{-1}X^\top y.$$

Adding $\lambda I$ also makes the matrix invertible even when $X^\top X$ is singular (more features than examples). In the eigenbasis of $X^\top X$ (eigenvalues $d_j$), ridge shrinks each component of the least-squares solution by the factor $d_j/(d_j + \lambda)$: directions well supported by the data ($d_j \gg \lambda$) are barely touched, poorly supported ones ($d_j \ll \lambda$) are shrunk towards zero. Lab 8 shows $\|w\|$ falling from 2.20 to 0.45 as $\lambda$ goes from 0 to 100, while training error rises.

**Worked example (1-D, by hand).** Data $x = (1, 2)$, $y = (1, 3)$, no intercept. Least squares: $w = \sum xy/\sum x^2 = (1 + 6)/(1 + 4) = 1.4$. Ridge with $\lambda = 5$: $w = \sum xy/(\sum x^2 + \lambda) = 7/10 = 0.7$. Halved.

**Probabilistic reading (from A3 §5.7).** L2 regularisation = MAP estimation with a Gaussian prior $\theta \sim \mathcal N(0, \tau^2 I)$, with $\lambda = 1/\tau^2$ when the loss is the summed negative log-likelihood (for a mean loss, $\lambda = 1/(n\tau^2)$). A strong penalty = a confident prior that weights are small.

**L1 regularisation** $\lambda\|\theta\|_1$ (lasso) instead pushes many weights to exactly zero (sparsity); it corresponds to a Laplace prior. Elastic net mixes both.

### 6.2 Weight decay, and the gradient step

The gradient of $\frac\lambda2\|\theta\|^2$ is $\lambda\theta$, so one gradient-descent step on $J$ is

$$\theta \leftarrow \theta - \eta\big(\nabla\hat R_S(\theta) + \lambda\theta\big) = \underbrace{(1 - \eta\lambda)}_{\text{shrink}}\theta - \eta\nabla\hat R_S(\theta).$$

Each step first **decays** the weights by the factor $(1 - \eta\lambda)$ and then takes the usual data step. Hence the name **weight decay**. In the project, $\eta = 0.002$ and $\lambda = 5\times10^{-4}$, so the per-step factor would be $1 - 10^{-6}$ under plain SGD - tiny per step, but it acts on every step and is opposed only by gradients that genuinely help the loss.

**For plain SGD, L2 regularisation and weight decay are the same thing. For Adam they are not.** PyTorch's `torch.optim.Adam(..., weight_decay=λ)` implements L2: it adds $\lambda\theta$ to the gradient *before* Adam's per-parameter normalisation. The decay term is therefore divided by $\sqrt{\hat v}$ like everything else: parameters with large historical gradients are decayed less, parameters with small gradients more, and the decay is no longer a simple shrink-by-a-constant. Loshchilov & Hutter (2019) showed this coupling makes L2 less effective in Adam and proposed **AdamW**, which applies decay directly: $\theta \leftarrow \theta - \eta\lambda\theta - \eta\cdot\text{AdamStep}$. Lab 4 shows the difference in its starkest form: with no data gradient at all, Adam's normalised L2 term moves every weight towards 0 by about $\eta$ per step (both a large and a tiny weight are driven to ~0 in 1,000 steps), whereas AdamW shrinks each by exactly $(1 - \eta\lambda)^{1000} = 0.999$.

The project uses `torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)` - L2-in-Adam. That is a legitimate choice (and the one used by most GNN reference implementations), and because `weight_decay` was tuned with this optimiser, its value is meaningful *for Adam*; just do not call it "decoupled weight decay", and do not expect the same $\lambda$ to behave the same way with AdamW.

### 6.3 Dropout

**Mechanism.** During training, each unit's output is independently set to zero with probability $p$ (the dropout rate), and the survivors are multiplied by $1/(1-p)$ ("inverted dropout"). At evaluation time dropout does nothing.

$$\tilde h = \frac{m \odot h}{1 - p}, \qquad m_j \sim \mathrm{Bernoulli}(1 - p) \text{ independently.}$$

**Why scale by $1/(1-p)$?** So the expected activation is unchanged: $\mathbb E[\tilde h_j] = \frac{(1-p)h_j + p\cdot 0}{1-p} = h_j$. Then the network at test time (no dropout, no scaling) sees activations of the same average magnitude as during training. Lab 7 confirms: with $p = 0.2$, survivors become 1.25, about 20% are zeroed, and the mean stays ≈ 1.

**Why it regularises.** Several complementary explanations (Srivastava et al. 2014):

1. **Prevents co-adaptation.** A unit cannot rely on any specific other unit being present, so it must learn features that are useful on their own, which are more robust.
2. **Implicit ensemble.** Each mini-batch trains a different random "thinned" sub-network; there are $2^{\#\text{units}}$ of them, sharing weights. Test-time use of the full network with scaled weights approximates averaging their predictions. Averaging reduces variance.
3. **Noise injection.** Like data augmentation, it makes the training problem harder and the solution smoother. For linear models, dropout on inputs is approximately an adaptive L2 penalty.

**Train vs eval mode.** In PyTorch, `model.train()` turns dropout on and `model.eval()` turns it off (it also switches BatchNorm behaviour; LayerNorm, used in this project, behaves the same in both modes). Forgetting `model.eval()` before evaluation gives noisy, worse predictions; forgetting `model.train()` before training silently disables dropout. `fit_predict` calls `model.train()` at the start of every epoch and `model.eval()` before the final prediction.

**Where the project uses dropout** (`dropout = 0.2`):
* `MVHGAT.encode`: `self.drop(F.elu(self.inp[t](X[t])))` - on the input projection of every node.
* `HeteroLayer.forward`: `self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))` - on every layer's output embeddings.
* `DenseGAT.forward`: `self.drop(att)` - on the **attention coefficients**, so each node randomly ignores some neighbours each epoch (attention dropout, as in the original GAT paper).

A related graph regulariser is **DropEdge** (Rong et al. 2020): randomly remove edges each epoch. The project's `drop_edge` parameter hides association edges, serving both as regularisation and as the hidden-link supervision mechanism.

### 6.4 Early stopping

Train while monitoring a validation metric; stop when it has not improved for a number of epochs (the **patience**) and keep the best checkpoint. Since an over-parameterised model moves from simple to complex functions as training proceeds (weights start small and grow), the number of training steps is itself a capacity control. For a quadratic loss with gradient descent, early stopping is approximately equivalent to L2 regularisation with $\lambda \approx 1/(\eta \cdot \text{steps})$ (Goodfellow et al. 2016, §7.8).

Three rules:

1. **Monitor the metric you care about.** Lab 6 shows that validation BCE reaches its minimum at epoch 32, where validation AUC is only 0.705, while validation AUC peaks at 0.761 at epoch 72. BCE punishes over-confidence; ranking can keep improving after BCE starts rising. For this project, monitor AUPR or AUC.
2. **The stopping epoch is a hyper-parameter**, chosen on validation data. Choosing it on the test fold is leakage.
3. **The best validation score is optimistic** (winner's curse over epochs).

The project does not early-stop inside each fold; it uses a fixed `epochs = 600` chosen on the validation split. That is the simplest leak-free option: the stopping point is decided once, before any test fold is touched.

### 6.5 Other regularisers you will meet

* **Data augmentation:** transform inputs in label-preserving ways. Graph analogues: DropEdge, feature masking. Hidden-link supervision and cold-start practice are task-specific augmentations.
* **Negative re-sampling every epoch:** the model never sees the same set of negatives twice, which prevents memorising specific negatives.
* **Architectural constraints:** the k-nearest-neighbour graphs (`k = 10`) cut weak edges; the softplus that keeps view weights non-negative (`MVHGAT.view_weights`); the residual skip and LayerNorm (stabilise training).
* **Low rank and similarity smoothness:** SCMFDD's objective $\|A - UV^\top\|^2 + \mu(\|U\|^2 + \|V\|^2) + \lambda(\mathrm{tr}\,U^\top L_r U + \mathrm{tr}\,V^\top L_d V)$ uses an L2 penalty ($\mu$) *and* a graph-Laplacian penalty that forces similar drugs to have similar factors (Unit A2, B4).

---

## 7. Loss functions

### 7.1 What a loss should do

A loss turns "how wrong was this prediction?" into a number that gradient descent can minimise. A good loss is (1) aligned with the evaluation goal, (2) differentiable (or at least sub-differentiable) with informative gradients, and (3) ideally derived from a sensible probabilistic model so that its minimiser has a clear meaning.

### 7.2 The common losses

| Loss | Formula (one example) | Probabilistic model | Typical use |
|---|---|---|---|
| Squared error (MSE) | $(y - \hat y)^2$ | Gaussian noise: $y \sim \mathcal N(\hat y, \sigma^2)$ | regression; SCMFDD reconstructs $A$ |
| Absolute error (MAE) | $\lvert y - \hat y\rvert$ | Laplace noise | robust regression |
| 0-1 loss | $\mathbb 1[\text{sign}(z) \ne y]$ | - | evaluation only (not differentiable) |
| Binary cross-entropy | $-[y\ln\sigma(z) + (1-y)\ln(1 - \sigma(z))]$ | Bernoulli | binary classification; MV-HGAT |
| Weighted BCE | $w_y \times \text{BCE}$ | weighted likelihood | imbalanced data; NIMCGCN, LAGCN |
| Hinge | $\max(0, 1 - y'z)$, $y' \in \{\pm1\}$ | - (max-margin) | SVMs |
| BPR (pairwise ranking) | $-\ln\sigma(z_{\text{pos}} - z_{\text{neg}})$ | P(positive ranked above negative) | recommender systems |

**MSE from a Gaussian likelihood.** If $y \sim \mathcal N(\hat y, \sigma^2)$, the negative log-likelihood is $\frac{(y - \hat y)^2}{2\sigma^2} + \ln(\sigma\sqrt{2\pi})$; with $\sigma$ fixed, minimising it = minimising squared error. So **MSE is to Gaussian what BCE is to Bernoulli** (Unit A3 derived the latter).

**Why not MSE on probabilities for classification?** With $\hat y = \sigma(z)$, the MSE gradient is $2(\sigma(z) - y)\sigma'(z)$; the extra $\sigma'(z)$ vanishes when the model is confidently wrong, so learning stalls exactly when it should be fastest. BCE's gradient $\sigma(z) - y$ has no such factor (A3 §6.4).

**Surrogate losses.** What we really care about here is a *ranking* metric (AUPR), which is piecewise constant in the scores and has zero gradient almost everywhere. We optimise a smooth **surrogate** (BCE) whose minimisation tends to improve the ranking. Pairwise losses such as BPR target ranking more directly; they are an interesting extension for this project (Unit B4).

### 7.3 The loss in the full objective

The objective actually minimised is

$$J(\theta) = \underbrace{\frac{1}{|\mathcal B|}\sum_{t\in\mathcal B}\ell(f_\theta(x_t), y_t)}_{\text{data term on this epoch's batch}} + \underbrace{\frac\lambda2\|\theta\|^2}_{\text{weight decay}},$$

where in MV-HGAT the batch $\mathcal B$ is this epoch's hidden positives plus sampled negatives, and dropout/DropEdge make $f_\theta$ itself stochastic during training.

---

## 8. Optimisation: gradient descent, SGD, momentum, Adam

### 8.1 Gradient descent from first principles

We want $\min_\theta J(\theta)$ where $J$ is differentiable. Near the current point, a first-order Taylor expansion says

$$J(\theta + \Delta) \approx J(\theta) + \nabla J(\theta)^\top\Delta.$$

Among all steps of a fixed small length $\|\Delta\| = r$, the decrease $-\nabla J^\top\Delta$ is largest when $\Delta$ points opposite to the gradient (Cauchy-Schwarz). So the update

$$\boxed{\theta_{t+1} = \theta_t - \eta\,\nabla J(\theta_t)}$$

with a **learning rate** (step size) $\eta > 0$ decreases $J$ for small enough $\eta$. Repeat until the gradient is near zero.

**Worked example (by hand; Lab 1 reproduces it).** $J(w) = (w - 3)^2$, $\nabla J = 2(w - 3)$, start at $w_0 = 0$, $\eta = 0.1$:

| $t$ | $w_t$ | gradient $2(w_t - 3)$ | $w_{t+1} = w_t - 0.1 \times$ gradient |
|---|---|---|---|
| 0 | 0.000 | -6.000 | 0.600 |
| 1 | 0.600 | -4.800 | 1.080 |
| 2 | 1.080 | -3.840 | 1.464 |
| 3 | 1.464 | -3.072 | 1.771 |

The error $w_t - 3$ is multiplied by $(1 - 2\eta) = 0.8$ every step: $-3, -2.4, -1.92, -1.536, \dots$ Linear (geometric) convergence.

### 8.2 The learning rate and curvature

For the quadratic $J(w) = \frac L2(w - w^\star)^2$ (curvature $L$; here $L = 2$), the update gives $w_{t+1} - w^\star = (1 - \eta L)(w_t - w^\star)$. Therefore:

* $0 < \eta < 1/L$: monotone convergence (Lab 1, $\eta = 0.1$).
* $\eta = 1/L$: converges in one step ($\eta = 0.5$ jumps straight to 3).
* $1/L < \eta < 2/L$: converges while oscillating around the minimum ($\eta = 0.9$).
* $\eta > 2/L$: $|1 - \eta L| > 1$, the iterates **diverge** ($\eta = 1.1$: $0, 6.6, -1.3, 8.2, -3.2, \dots$).

In many dimensions, $J(\theta) \approx \frac12(\theta - \theta^\star)^\top H(\theta - \theta^\star)$ with Hessian $H$ whose eigenvalues range from $\mu$ (flattest direction) to $L$ (steepest). Stability needs $\eta < 2/L$, but progress along the flattest direction is then only a factor $(1 - \eta\mu) \ge 1 - 2\mu/L$ per step. The **condition number** $\kappa = L/\mu$ governs speed: the best fixed step $\eta = 2/(L + \mu)$ contracts the error by $\frac{\kappa - 1}{\kappa + 1}$ per step. For $\kappa = 50$ that is $0.96$ per step - slow. Ill-conditioning (long, narrow valleys) is the main enemy of plain gradient descent, and neural network losses are badly conditioned.

**Choosing $\eta$ in practice:** try a log-spaced range (e.g. $10^{-4}, 3\times10^{-4}, 10^{-3}, 3\times10^{-3}, 10^{-2}$) on the validation split; too small → slow, plateaus; too large → loss spikes or NaN. Learning-rate **schedules** (step decay, cosine annealing, warm-up) lower $\eta$ over time to settle into a minimum. The project uses a constant $\eta = 0.002$.

### 8.3 Stochastic and mini-batch gradient descent

The full gradient $\nabla\hat R_S = \frac1n\sum_t\nabla\ell_t$ costs a pass over all data. **Stochastic gradient descent (SGD)** uses one random example, and **mini-batch SGD** a random batch $\mathcal B$ of size $b$:

$$g_t = \frac1b\sum_{s\in\mathcal B}\nabla\ell_s(\theta_t), \qquad \mathbb E[g_t] = \nabla\hat R_S(\theta_t).$$

The mini-batch gradient is an **unbiased but noisy** estimate of the full gradient, with variance shrinking like $1/b$. Benefits: far cheaper steps (many updates per pass), and the noise helps escape sharp minima and saddle points, often improving generalisation. Cost: the noise prevents exact convergence with a constant $\eta$ (iterates jitter around the minimum), so the learning rate is often decayed. One full pass over the data is an **epoch**.

**In this project** every epoch is **one optimiser step** on one "batch": the whole graph is encoded (the GNN needs all nodes' messages), and the loss is taken over the hidden positives and their sampled negatives. So `epochs = 600` means exactly 600 Adam steps. It is still stochastic: the negatives, the hidden-link mask, the cold-start mask and the dropout masks change every step, so each step's gradient is a noisy estimate of the expected objective.

### 8.4 Momentum

**Intuition.** A heavy ball rolling down a narrow valley: gradient components that keep pointing the same way (along the valley floor) accumulate speed, while components that flip sign (across the valley walls) cancel out.

**Update (heavy ball, PyTorch's convention).**

$$v_{t+1} = \beta v_t + \nabla J(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\,v_{t+1},$$

with $\beta \in [0, 1)$ (typically 0.9). Unrolling, $v_{t+1} = \sum_{s \le t}\beta^{t-s}\nabla J(\theta_s)$: an exponentially weighted sum of past gradients, with effective memory about $1/(1-\beta)$ steps (10 for 0.9). In a direction where the gradient is constant, the effective step is amplified up to $\eta/(1-\beta)$.

**Worked example (by hand).** $J(w) = (w - 3)^2$, $\eta = 0.1$, $\beta = 0.9$, $w_0 = 0$, $v_0 = 0$:

| $t$ | gradient | $v_{t+1} = 0.9v_t + g$ | $w_{t+1} = w_t - 0.1\,v_{t+1}$ |
|---|---|---|---|
| 0 | -6.00 | -6.00 | 0.600 |
| 1 | -4.80 | -10.20 | 1.620 |
| 2 | -2.76 | -11.94 | 2.814 |

After 3 steps momentum is at 2.814, plain GD at 1.464. (On this 1-D problem momentum will overshoot 3 next and oscillate; its advantage is largest in ill-conditioned valleys.)

With optimally tuned $\eta$ and $\beta$ on a quadratic, heavy-ball momentum contracts the error by $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}$ per step instead of $\frac{\kappa - 1}{\kappa + 1}$: a square-root improvement in the dependence on the condition number. For $\kappa = 50$: $0.75$ instead of $0.96$. Lab 2 measures it: on a bowl with $\kappa = 50$, the best plain GD needs 227 steps, tuned momentum 46. **Nesterov momentum** evaluates the gradient at the look-ahead point $\theta_t - \eta\beta v_t$ and has slightly better theory.

### 8.5 Adaptive methods and Adam

**Problem:** different parameters need different step sizes (the condition-number issue, parameter by parameter). Adaptive methods scale each coordinate's step by an estimate of its gradient magnitude. AdaGrad accumulates all past squared gradients; RMSProp uses an exponential moving average of them; **Adam** (Kingma & Ba 2015) combines RMSProp with momentum.

**Adam's update** for each parameter coordinate, with gradient $g_t$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad\text{(1st moment: mean of gradients)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \qquad\text{(2nd moment: mean of squared gradients)}$$
$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t} \qquad\text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Defaults (PyTorch and the paper): $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The project uses these defaults with $\eta = 0.002$.

**Why bias correction?** $m_0 = v_0 = 0$, so early averages are biased towards zero. If the gradients had constant expectation, $\mathbb E[m_t] = (1 - \beta_1^t)\mathbb E[g]$, because $m_t = (1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}g_s$ and $(1-\beta_1)\sum_{s=1}^t\beta_1^{t-s} = 1 - \beta_1^t$ (geometric series). Dividing by $1 - \beta_1^t$ removes the bias. Same for $v_t$. Without it, the first steps would be tiny (for $\beta_2 = 0.999$, $v_1 = 0.001g_1^2$).

**The first step, by hand.** At $t = 1$: $m_1 = 0.1g$, $v_1 = 0.001g^2$, $\hat m_1 = g$, $\hat v_1 = g^2$, so the step is $\eta\,g/(|g| + \epsilon) \approx \eta\,\mathrm{sign}(g)$. **Every parameter moves by about $\eta$ on the first step, regardless of its gradient's size.** Lab 3 checks this: with gradients 100 and 0.01 and $\eta = 0.002$, both parameters move by exactly 0.002, whereas plain SGD would move them by 0.2 and 0.00002.

**Worked 3-step example.** $J(w) = (w-3)^2$, $\eta = 0.1$, $w_0 = 0$:

| $t$ | $g_t$ | $m_t$ | $v_t$ | $\hat m_t$ | $\hat v_t$ | $w_t$ |
|---|---|---|---|---|---|---|
| 1 | -6.000 | -0.600 | 0.0360 | -6.000 | 36.00 | 0.100 |
| 2 | -5.800 | -1.120 | 0.0696 | -5.895 | 34.82 | 0.200 |
| 3 | -5.600 | -1.568 | 0.1009 | -5.786 | 33.67 | 0.300 |

Adam moves almost exactly $\eta = 0.1$ per step while the gradient is consistent: $|\hat m|/\sqrt{\hat v} \approx 1$. Its step size is set by $\eta$, not by the gradient scale - a crucial practical property: **in Adam, $\eta$ is roughly the maximum distance any parameter moves per step.** With $\eta = 0.002$ and 600 steps, no parameter of MV-HGAT can move by much more than about $600 \times 0.002 = 1.2$ from its initial value (a useful sanity check when you change `lr` or `epochs`).

**Strengths and caveats.** Adam is robust to gradient scale, needs little tuning, and works well on noisy, sparse, badly scaled problems such as GNNs - which is why nearly every model in this project (MV-HGAT, SCMFDD, NIMCGCN, LAGCN) uses it. On some problems well-tuned SGD with momentum generalises slightly better; and remember the L2-vs-AdamW distinction (section 6.2).

### 8.6 The PyTorch training loop, line by line

```python
# (generic PyTorch loop; same structure as MVHGATMethod.fit_predict)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)
for epoch in range(epochs):
    model.train()                 # enable dropout
    logits = model(inputs)        # forward pass: builds the computation graph
    loss = loss_fn(logits, y)     # scalar objective
    opt.zero_grad()               # clear gradients left from the previous step
    loss.backward()               # autograd: d loss / d parameter, stored in p.grad
    opt.step()                    # Adam update of every parameter using p.grad
model.eval()                      # disable dropout
with torch.no_grad():             # no graph building: faster, less memory
    scores = torch.sigmoid(model(inputs))
```

The order matters: gradients **accumulate** in `p.grad` across `backward()` calls, so without `zero_grad()` each step would use the sum of all previous gradients.

---

## 9. Parameters vs hyper-parameters

### 9.1 Definitions

* **Parameters** $\theta$: quantities **learned from the training data** by the optimiser - weights and biases. They are what `model.parameters()` returns.
* **Hyper-parameters**: quantities **set before training** that control the model's form, the learning procedure, or the data pipeline. They are chosen by the experimenter, ideally by validation. They cannot sensibly be learned by minimising the training loss (training loss would always prefer more capacity and no regularisation).

A quick test: *"Does gradient descent update it?"* Yes → parameter. No, I set it → hyper-parameter.

Sometimes a choice could be either: MV-HGAT **learns** its view weights $w_v$ and degree-gate coefficients (parameters), where a simpler model would fix them by hand (hyper-parameters). Turning hyper-parameters into learned parameters, with suitable constraints, is a common modelling move.

### 9.2 MV-HGAT's parameters (Lab 12)

| Parameter group | Shape / count | Role |
|---|---|---|
| `inp` (input projections) | $2092\times64 + 64$ (drug) and $1532\times64 + 64$ (disease) = 232,064 | map raw feature rows to 64-d |
| `layers` (2 × HeteroLayer) | 167,168 | GAT weights per relation, attention vectors, view attention, skips, LayerNorms |
| `W` (bilinear decoder) | $192\times192$ = 36,864 | $h_i^\top W h_j$ (192 = 64 × 3 jumping-knowledge) |
| `prop_w`, `prop_scale`, `bias`, `gate` | 6 + 1 + 1 + 4 | propagation-head view weights, scale, intercept, degree gate |
| **Total** | **436,108** | |

The input width 2092 = three drug similarity rows (3 × 593) + the drug's association row (313). With only 1,546 training positives per fold, 436k parameters is a lot - which is why regularisation, the kNN sparsification and the strong inductive bias of the propagation head matter.

### 9.3 MV-HGAT's hyper-parameters

Every field of `MVHGATConfig` is a hyper-parameter. Grouped by what it controls:

| Group | Fields | Notes |
|---|---|---|
| Data / input | `drug_views`, `disease_views`, `feat_assoc`, `feat_views`, `feat_prop`, `use_bridge` | which information the model sees |
| Graph construction | `k`, `bridge_k`, `use_assoc_edges` | sparsity of the kNN similarity graphs |
| Architecture (capacity) | `hidden`, `layers`, `heads`, `uniform_attention`, `prop_head`, `degree_gate` | size and form of $\mathcal F$ |
| Regularisation | `dropout`, `weight_decay`, `drop_edge` | variance control |
| Optimisation | `lr`, `epochs` | how the minimiser is found (and implicit early stopping) |
| Training task design | `neg_ratio`, `supervise_hidden`, `cold_frac` | what the loss is computed on |

The baselines have their own (`SCMFDD(k=128, mu=0.05, lam=2.0, epochs=400, lr=0.02)`, `MBiRW(alpha=0.3, l=2, r=2)`, `DRRS(tau_rel=0.005, iters=200, rank=200)`...). A fair comparison tunes every method's key hyper-parameters with the **same** validation protocol and comparable budget; HOW_IT_WORKS §5.4 says this was done.

---

## 10. Cross-validation

### 10.1 k-fold CV

Split the data into $k$ disjoint **folds** of (roughly) equal size. For $f = 1..k$: train on all folds except $f$, evaluate on fold $f$. Report the mean (and SD) of the $k$ scores.

```
data:  [ F1 | F2 | F3 | F4 | F5 ]
run 1: [TEST| tr | tr | tr | tr ]
run 2: [ tr |TEST| tr | tr | tr ]
 ...
run 5: [ tr | tr | tr | tr |TEST]     every example is tested exactly once
```

**Why:** with small data, a single train/test split wastes data (the test part is never trained on) and gives a high-variance estimate (it depends on which examples landed in the test part). CV uses every example for testing once and for training $k-1$ times.

### 10.2 What CV estimates, and its bias/variance

* Each training set has a fraction $(k-1)/k$ of the data, so CV estimates the performance of the **learning procedure** trained on slightly less data than the final model. The estimate is slightly **pessimistic** (biased down), less so for larger $k$.
* The $k$ fold scores are **correlated** (overlapping training sets; Unit A3 §8.4), so their SD understates the uncertainty of the mean.
* Bates, Hastie & Tibshirani (2021) show that CV estimates the *average* performance of the procedure over training sets drawn like ours - not the error of the specific model fitted on our data - and that naive CV confidence intervals are too narrow.
* **Choosing $k$:** 5 or 10 is the standard compromise. Leave-one-out ($k = n$) has low bias but high variance and high cost. Larger $k$ → each training set closer to the full data (less bias), more compute.
* **Repeated CV:** re-shuffle and repeat $r$ times to average out the luck of one particular partition; the project's `--repeats 5` gives 25 scores.

### 10.3 Stratified CV

**Stratification** keeps the class proportions equal in every fold. With 1% positives, a random fold could easily get noticeably more or fewer positives than average (A3 §3.2), changing both training and the AUPR baseline. `kfold_splits` stratifies by shuffling and splitting positives and negatives **separately** (`np.array_split(pos, k)` and `np.array_split(neg, k)`): each Fdataset 5-fold test fold has 386 or 387 positives and 36,735 or 36,736 negatives. Lab 9 runs the same function on a toy matrix.

### 10.4 Group CV and leave-one-group-out

If examples come in **groups** that should not be split across train and test (all pairs of the same disease; all images of the same patient), use **group k-fold**: whole groups go to a fold. **Leave-one-disease-out** (`evaluation.py::run_lodo`) is leave-one-group-out with diseases as groups: all of disease $j$'s links are hidden at once, which tests a genuinely different question (cold start: can we predict for a disease with no known drugs?) than pair-level CV (warm start: can we fill in missing links of known diseases?). Choosing the CV scheme = choosing the deployment scenario you claim to model.

### 10.5 CV for link prediction: the project's specifics

`run_kfold` differs from textbook CV in instructive ways:

1. The "examples" are matrix cells; folds are made of cells, not of drugs or diseases.
2. Training on fold $f$'s complement means **zeroing** fold $f$'s positives in the matrix (they become unknown), not deleting rows.
3. Test negatives are excluded from the training negative pool (`neg_mask[test_neg] = False`).
4. The test set includes **all** of the fold's negatives (~36,735 cells), so test metrics reflect the real 1% base rate, while training uses sampled negatives.
5. The same seeds give the same folds for every method → paired comparison (A3 §10).

---

## 11. Hyper-parameter tuning

### 11.1 Search strategies

* **Grid search:** evaluate every combination of a few values per hyper-parameter. Cost grows exponentially with the number of hyper-parameters (5 values for each of 6 hyper-parameters = 15,625 runs).
* **Random search** (Bergstra & Bengio 2012): sample combinations at random. With the same budget it explores many more distinct values of each hyper-parameter, which matters because usually only a few hyper-parameters are important. Usually beats grid search.
* **One-at-a-time (OAT) / sensitivity analysis:** vary one hyper-parameter around a default while holding the others fixed. Cheap and very interpretable ("how sensitive is the model to $k$?"), but misses interactions (the best `dropout` may depend on `hidden`). `05_sensitivity.py` does OAT over 19 configurations (`hidden` 4 values, `layers` 3, `neg_ratio` 4, `k` 4, `drop_edge` 4), 3 seeds each.
* **Bayesian optimisation / successive halving / Hyperband:** model the score surface or stop bad configurations early. Worth it when each run is expensive and the space is large (e.g. Optuna).
* **Scale:** search learning rate, weight decay and similar on a **log scale** (1e-4, 3e-4, 1e-3, ...).

### 11.2 Tuning on a validation split

The procedure:

1. Split off a validation set from the data available for development.
2. For each candidate configuration: train on the remaining training data, score on validation (several seeds if the result is noisy).
3. Pick the configuration with the best validation score (or a simpler one within noise of the best).
4. Retrain on all development data with the chosen configuration.
5. Evaluate once on the test data.

Lab 10 shows why step 5 must use fresh data: selecting the best of 30 equally good configurations on validation inflates its validation score by 0.041, while its test score is unbiased (0.7996 vs the true 0.80).

**Dealing with noise when picking.** In `sensitivity.json`, `hidden=64` scores AUPR 0.5209 ± 0.0075 and `layers=1` scores 0.5257 ± 0.0255. Should you switch to one layer? The difference (0.005) is far smaller than the seed SD; a sensible rule is the **one-standard-error rule**: prefer the simplest / default configuration whose score is within one SE of the best. Do not chase noise.

### 11.3 Nested cross-validation

If you tune with CV and report the best CV score, the reported number has the winner's-curse bias (the same folds chose and scored the configuration). **Nested CV** fixes this:

```
OUTER loop (estimates performance of the WHOLE procedure, tuning included)
  for each outer fold o:
      outer-train = all data except fold o;  outer-test = fold o
      INNER loop (selects hyper-parameters using outer-train ONLY)
          for each configuration c:
              run k-fold CV inside outer-train -> mean inner score of c
          c* = best configuration
      retrain with c* on all of outer-train
      score on outer-test   -> one unbiased outer score
report mean ± SD of the outer scores
```

The outer test fold never influences the choice of $c^*$. Note that different outer folds may pick different $c^*$: nested CV evaluates the **procedure** "tune then train", which is what you would actually do on new data. Cawley & Talbot (2010) show that the optimistic bias of non-nested evaluation can be as large as the differences between learning algorithms; Varma & Simon (2006) demonstrated it in bioinformatics. Lab 11 reproduces it on a small noisy problem: non-nested CV AUC 0.668 vs nested 0.609, an optimism of +0.059.

**Cost:** (outer folds) × (configurations) × (inner folds) trainings. For this project, 5 outer folds × 19 configurations × 3 seeds = 285 trainings instead of 57 - expensive but feasible for a final paper run.

### 11.4 What the project does, and its one subtlety

`05_sensitivity.py` carves the validation split like this:

```python
# (quoted from scripts/05_sensitivity.py)
rng = np.random.default_rng(12345)
pos, neg = np.flatnonzero(A.ravel() > 0), np.flatnonzero(A.ravel() == 0)
val_pos = rng.choice(pos, len(pos) // 5, replace=False)
val_neg = rng.choice(neg, len(neg) // 5, replace=False)
A_tr = A.copy().ravel()
A_tr[val_pos] = 0
A_tr = A_tr.reshape(A.shape)
neg_mask = (A == 0).ravel()
neg_mask[val_neg] = False
neg_mask = neg_mask.reshape(A.shape)
```

This is a correct, leak-free **validation** protocol for choosing hyper-parameters: validation positives are hidden from training, validation negatives are excluded from the negative pool, and tuning never reads the CV test-fold results.

The subtlety: `val_pos` and `val_neg` are drawn from the **full** matrix `A`, and the CV in `03_evaluate.py` later partitions the **same full matrix** into test folds. So about 20% of every CV test fold's cells were also validation cells during tuning. The test folds were never used to *score* the tuning, but the cells overlap; strictly, the CV estimate is therefore not fully independent of the tuning decisions. Three mitigating facts: only ~19 configurations were compared, one at a time around defaults (small winner's-curse effect, compared with Lab 11's 42-way search); the baselines were tuned on the same split, so the *comparison* is symmetric; and most settings differ by less than the seed noise. The rigorous options, in increasing cost, are: (a) state this in the paper's limitations; (b) hold out a final test set (e.g., a random 20% of links) that is excluded from both tuning and development CV; (c) nested CV with tuning inside each outer training fold. Knowing exactly where your protocol is and is not airtight is part of owning the project.

---

## 12. Code laboratory

All labs are self-contained, run on the CPU in seconds (Lab 11 takes about 40 seconds), and were executed with the project's interpreter (`.venv\Scripts\python.exe`: NumPy 2.5, scikit-learn 1.9, PyTorch 2.14). The output shown under each block is the real output.

### Lab 1 - Gradient descent and the learning rate (section 8.1-8.2)

```python
# Gradient descent on f(w) = (w - 3)^2, whose gradient is 2(w - 3) and curvature L = 2
def gd(lr, steps=8, w=0.0):
    path = [w]
    for _ in range(steps):
        w = w - lr * 2 * (w - 3)
        path.append(w)
    return path

for lr in [0.1, 0.5, 0.9, 1.1]:
    print(f"lr = {lr:<4}", " ".join(f"{w:7.3f}" for w in gd(lr)))
```

```text
lr = 0.1    0.000   0.600   1.080   1.464   1.771   2.017   2.214   2.371   2.497
lr = 0.5    0.000   3.000   3.000   3.000   3.000   3.000   3.000   3.000   3.000
lr = 0.9    0.000   5.400   1.080   4.536   1.771   3.983   2.214   3.629   2.497
lr = 1.1    0.000   6.600  -1.320   8.184  -3.221  10.465  -5.958  13.750  -9.899
```

*What to notice:* the four regimes of section 8.2: monotone ($\eta = 0.1$), one-step ($\eta = 1/L = 0.5$), oscillating but converging ($\eta = 0.9$), divergent ($\eta = 1.1 > 2/L$).

### Lab 2 - Ill-conditioning: GD vs momentum vs Adam (sections 8.2-8.5)

```python
import numpy as np

# Ill-conditioned bowl: f(x, y) = 0.5 * (x^2 + 50 y^2); minimum at (0, 0); condition number 50
H = np.array([1.0, 50.0])
grad = lambda w: H * w
f = lambda w: 0.5 * np.sum(H * w ** 2)

def run(update, steps=1000):
    """Number of steps until f < 1e-6, starting from (10, 1)."""
    w, state = np.array([10.0, 1.0]), {}
    for t in range(1, steps + 1):
        w = update(w, grad(w), state, t)
        if f(w) < 1e-6:
            return t
    return steps

def gd(lr):
    return lambda w, g, s, t: w - lr * g

def momentum(lr, beta):
    def step(w, g, s, t):
        s["v"] = beta * s.get("v", 0) + g          # velocity = decaying sum of gradients
        return w - lr * s["v"]
    return step

def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def step(w, g, s, t):
        s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
        s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
        m_hat, v_hat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)
        return w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return step

kappa = 50.0
for name, upd in [("GD, lr = 0.02", gd(0.02)),
                  ("GD, lr = 2/(L+mu) = 0.039", gd(2 / 51)),
                  ("GD, lr = 0.041 (> 2/L)", gd(0.041)),
                  ("momentum 0.9, lr = 0.02", momentum(0.02, 0.9)),
                  ("momentum, tuned", momentum(4 / (np.sqrt(50) + 1) ** 2,
                                               ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2)),
                  ("Adam, lr = 0.1", adam(0.1)),
                  ("Adam, lr = 0.5", adam(0.5))]:
    print(f"{name:27s}: {run(upd):4d} steps")
```

```text
GD, lr = 0.02              :  439 steps
GD, lr = 2/(L+mu) = 0.039  :  227 steps
GD, lr = 0.041 (> 2/L)     : 1000 steps
momentum 0.9, lr = 0.02    :  153 steps
momentum, tuned            :   46 steps
Adam, lr = 0.1             :  305 steps
Adam, lr = 0.5             :  151 steps
```

*What to notice:* plain GD is limited by the steep direction: its best stable rate still needs 227 steps, and a step just above $2/L = 0.04$ never converges (1000 = the cap). Momentum with the textbook-optimal $\beta = \big(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\big)^2$ and $\eta = \frac{4}{(\sqrt L + \sqrt\mu)^2}$ needs only 46. Adam rescales each coordinate and is competitive without knowing the curvature, but with a constant learning rate it jitters near the optimum rather than converging exactly; its speed here depends strongly on $\eta$.

### Lab 3 - Adam's first step by hand (section 8.5)

```python
import torch

# One Adam step by hand vs torch.optim.Adam, for two parameters with very different gradients
w = torch.tensor([1.0, 1.0], requires_grad=True)
opt = torch.optim.Adam([w], lr=0.002)          # the project's learning rate
loss = 100.0 * w[0] + 0.01 * w[1]              # gradients: 100 and 0.01
opt.zero_grad(); loss.backward(); opt.step()
print("torch Adam after one step :", w.detach().numpy())

g = torch.tensor([100.0, 0.01])
m = 0.1 * g; v = 0.001 * g ** 2                # beta1 = 0.9, beta2 = 0.999, from m0 = v0 = 0
m_hat, v_hat = m / (1 - 0.9), v / (1 - 0.999)
print("by hand                   :", (torch.tensor([1.0, 1.0]) - 0.002 * m_hat / (v_hat.sqrt() + 1e-8)).numpy())
print("plain SGD with lr=0.002   :", (torch.tensor([1.0, 1.0]) - 0.002 * g).numpy())
```

```text
torch Adam after one step : [0.998 0.998]
by hand                   : [0.998 0.998]
plain SGD with lr=0.002   : [0.8     0.99998]
```

### Lab 4 - L2 in Adam vs decoupled weight decay (AdamW) (section 6.2)

```python
import torch

# How torch.optim.Adam(weight_decay=...) differs from AdamW: zero data gradient, only decay acts
for name, Opt in [("Adam  (L2 added to grad)", torch.optim.Adam), ("AdamW (decoupled decay)", torch.optim.AdamW)]:
    w = torch.tensor([1.0, 0.001], requires_grad=True)       # a big and a tiny weight
    opt = Opt([w], lr=0.002, weight_decay=5e-4)
    for step in range(1000):
        opt.zero_grad()
        (0.0 * w).sum().backward()                           # data loss contributes nothing
        opt.step()
    print(f"{name}: after 1000 steps w = {w.detach().numpy()}")
print("pure exponential decay (1 - lr*wd)^1000 =", round((1 - 0.002 * 5e-4) ** 1000, 6))
```

```text
Adam  (L2 added to grad): after 1000 steps w = [2.7071258e-02 8.3169051e-27]
AdamW (decoupled decay): after 1000 steps w = [9.989867e-01 9.989523e-04]
pure exponential decay (1 - lr*wd)^1000 = 0.999
```

*What to notice:* in Adam the gradient is just $\lambda w$; Adam normalises it to $\approx \mathrm{sign}(w)$, so each weight moves ~0.002 per step towards zero *regardless of $\lambda$*: both are driven to ≈ 0 (the large one is still oscillating near zero). AdamW decays multiplicatively by $(1 - \eta\lambda)$ per step: 0.999 after 1,000 steps. This is an extreme case (no data gradient); in real training the data gradients dominate $\hat v$ and the L2 term is just one small contributor, but the coupling is real.

### Lab 5 - Measuring bias and variance (section 5)

```python
import numpy as np

rng = np.random.default_rng(0)
f_true = lambda x: np.sin(2 * np.pi * x)       # the unknown truth
sigma = 0.3                                    # label noise SD -> irreducible error 0.09
x_test = np.linspace(0.05, 0.95, 50)

print("degree  bias^2   variance  noise   sum     measured test MSE")
for degree in [1, 3, 5, 9]:
    preds = []
    for _ in range(500):                       # 500 independent training sets of 30 points
        x = rng.random(30)
        y = f_true(x) + rng.normal(0, sigma, 30)
        coef = np.polyfit(x, y, degree)
        preds.append(np.polyval(coef, x_test))
    preds = np.array(preds)                    # 500 x 50
    bias2 = np.mean((preds.mean(0) - f_true(x_test)) ** 2)
    var = np.mean(preds.var(0))
    y_new = f_true(x_test) + rng.normal(0, sigma, preds.shape)   # fresh test labels
    mse = np.mean((preds - y_new) ** 2)
    print(f"{degree:4d}   {bias2:7.4f}  {var:8.4f}  {sigma**2:.4f}  {bias2 + var + sigma**2:7.4f}  {mse:7.4f}")
```

```text
degree  bias^2   variance  noise   sum     measured test MSE
   1    0.1550    0.0215  0.0900   0.2665   0.2672
   3    0.0029    0.0133  0.0900   0.1061   0.1062
   5    0.0001    0.0257  0.0900   0.1157   0.1169
   9    0.0367   11.8363  0.0900  11.9631  11.9535
```

*What to notice:* the decomposition adds up (column "sum" ≈ measured MSE). Moving from degree 1 to 3 trades a big bias reduction for a small variance increase; beyond that, variance explodes. Degree 9 has occasional wild fits (some training sets of 30 random points leave gaps that a degree-9 polynomial fills with huge swings), which dominate its variance.

### Lab 6 - Over-fitting, weight decay, dropout and early stopping in PyTorch (sections 3, 6)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
# Small, noisy data set: 300 training examples, 40 features, only 3 of them informative
def make(n):
    X = rng.normal(size=(n, 40)).astype(np.float32)
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)
Xtr, ytr = make(300)
Xva, yva = make(2000)

def train(weight_decay=0.0, dropout=0.0, epochs=300):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.002, weight_decay=weight_decay)
    hist = []                                   # (train BCE, val BCE, val AUC) per epoch
    for ep in range(epochs):
        net.train()                             # dropout ON
        loss = F.binary_cross_entropy_with_logits(net(Xtr).squeeze(1), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        net.eval()                              # dropout OFF for measuring
        with torch.no_grad():
            s_tr, s_va = net(Xtr).squeeze(1), net(Xva).squeeze(1)
            hist.append((F.binary_cross_entropy_with_logits(s_tr, ytr).item(),
                         F.binary_cross_entropy_with_logits(s_va, yva).item(),
                         roc_auc_score(yva.numpy(), s_va.numpy())))
    return np.array(hist)

print("setting               train BCE  val BCE  val AUC | best val AUC (epoch) | val-BCE minimum at epoch")
for label, kw in [("no regularisation", {}),
                  ("weight decay 1e-2", {"weight_decay": 1e-2}),
                  ("weight decay 5e-2", {"weight_decay": 5e-2}),
                  ("dropout 0.5", {"dropout": 0.5}),
                  ("dropout 0.5 + wd 5e-2", {"dropout": 0.5, "weight_decay": 5e-2})]:
    h = train(**kw)
    e_auc, e_bce = h[:, 2].argmax(), h[:, 1].argmin()
    print(f"{label:22s} {h[-1,0]:8.3f} {h[-1,1]:8.3f} {h[-1,2]:8.4f} | {h[e_auc,2]:.4f} ({e_auc+1:3d})      "
          f"| {e_bce+1:3d} (val AUC there {h[e_bce,2]:.4f})")
```

```text
setting               train BCE  val BCE  val AUC | best val AUC (epoch) | val-BCE minimum at epoch
no regularisation         0.000    1.606   0.7544 | 0.7608 ( 72)      |  32 (val AUC there 0.7046)
weight decay 1e-2         0.023    0.683   0.7804 | 0.7843 (138)      |  39 (val AUC there 0.7469)
weight decay 5e-2         0.156    0.492   0.7910 | 0.7989 ( 71)      | 154 (val AUC there 0.7943)
dropout 0.5               0.001    2.196   0.7711 | 0.7741 (251)      |  40 (val AUC there 0.7319)
dropout 0.5 + wd 5e-2     0.191    0.487   0.7929 | 0.7981 ( 78)      | 214 (val AUC there 0.7945)
```

*What to notice:*
* **Over-fitting:** without regularisation the training BCE reaches 0.000 (memorised) while the validation BCE climbs to 1.6.
* **Weight decay** keeps the training loss away from zero and gives the best final validation AUC (0.791 at $\lambda = 0.05$ vs 0.754 without).
* **Dropout** improves the validation AUC (0.771) even though its validation BCE is *worse* - the network is still over-confident on the examples it gets wrong, which BCE punishes heavily. Different metrics can disagree about which model is better; judge by the metric you will report.
* **Early stopping:** the validation-BCE minimum (epoch 32 without regularisation) is *not* where AUC peaks (epoch 72). Early-stop on the metric you care about. And the "best val AUC" column is itself optimistic - it was selected on the same validation set.
* Note the weight-decay values that help here (1e-2 to 5e-2) are much larger than the project's 5e-4; the right $\lambda$ depends on the model, data size and optimiser, which is why it is tuned.

### Lab 7 - Dropout mechanics (section 6.3)

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
drop = nn.Dropout(p=0.2)                      # the project's dropout rate
x = torch.ones(10)

drop.train()
print("train mode:", drop(x).numpy())        # survivors are scaled by 1/(1 - 0.2) = 1.25
big = drop(torch.ones(1_000_000))
print(f"train mode: fraction zeroed {(big == 0).float().mean():.4f}, mean output {big.mean():.4f}")

drop.eval()
print("eval mode :", drop(x).numpy())        # identity at test time
```

```text
train mode: [0.   1.25 1.25 0.   1.25 1.25 1.25 1.25 1.25 1.25]
train mode: fraction zeroed 0.1996, mean output 1.0004
eval mode : [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

### Lab 8 - Ridge regression: L2 shrinks weights (section 6.1)

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 30, 10
X = rng.normal(size=(n, d))
w_true = np.zeros(d); w_true[:3] = [2.0, -1.0, 0.5]
y = X @ w_true + rng.normal(0, 1.0, n)

print("lambda   ||w||    first 3 weights           train MSE")
for lam in [0.0, 1.0, 10.0, 100.0]:
    w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)   # ridge closed form
    mse = np.mean((X @ w - y) ** 2)
    print(f"{lam:6.1f}  {np.linalg.norm(w):6.3f}   {np.round(w[:3], 3)}   {mse:.3f}")
```

```text
lambda   ||w||    first 3 weights           train MSE
   0.0   2.198   [ 1.954 -0.764  0.533]   0.674
   1.0   2.077   [ 1.839 -0.719  0.5  ]   0.682
  10.0   1.458   [ 1.241 -0.472  0.332]   1.015
 100.0   0.448   [ 0.329 -0.117  0.093]   2.930
```

*What to notice:* as $\lambda$ grows the weight norm shrinks and the training error rises - regularisation always costs training fit; it pays off only on new data.

### Lab 9 - The project's stratified fold construction (section 10.3)

```python
import numpy as np

# The project's split logic (evaluation.py::kfold_splits) on a toy 20 x 10 matrix
rng0 = np.random.default_rng(0)
A = (rng0.random((20, 10)) < 0.1).astype(float)

def kfold_splits(A, k, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(A.ravel() > 0)
    neg = np.flatnonzero(A.ravel() == 0)
    rng.shuffle(pos); rng.shuffle(neg)
    pf, nf = np.array_split(pos, k), np.array_split(neg, k)
    for f in range(k):
        yield f, pf[f], nf[f]

print(f"matrix has {int(A.sum())} positives and {int((A == 0).sum())} negatives")
seen = []
for f, tp, tn in kfold_splits(A, 5, seed=0):
    seen += list(tp) + list(tn)
    print(f"fold {f}: {len(tp)} test positives, {len(tn)} test negatives, "
          f"positive rate {len(tp) / (len(tp) + len(tn)):.3f}")
print("every cell is tested exactly once:", sorted(seen) == list(range(A.size)))
```

```text
matrix has 21 positives and 179 negatives
fold 0: 5 test positives, 36 test negatives, positive rate 0.122
fold 1: 4 test positives, 36 test negatives, positive rate 0.100
fold 2: 4 test positives, 36 test negatives, positive rate 0.100
fold 3: 4 test positives, 36 test negatives, positive rate 0.100
fold 4: 4 test positives, 35 test negatives, positive rate 0.103
every cell is tested exactly once: True
```

### Lab 10 - The winner's curse in hyper-parameter selection (sections 3.3, 11.2)

```python
import numpy as np

# Winner's curse: 30 configurations that are ALL truly equal (true score 0.80);
# each validation estimate has noise SD 0.02. Pick the best on validation, then test it.
rng = np.random.default_rng(0)
true, noise, n_configs = 0.80, 0.02, 30
picked_val, picked_test = [], []
for _ in range(10_000):
    val = true + rng.normal(0, noise, n_configs)
    best = val.argmax()
    picked_val.append(val[best])
    picked_test.append(true + rng.normal(0, noise))      # independent test estimate
print(f"mean validation score of the winner : {np.mean(picked_val):.4f}")
print(f"mean test score of the winner       : {np.mean(picked_test):.4f}")
print(f"optimism from selection             : {np.mean(picked_val) - np.mean(picked_test):+.4f}")
```

```text
mean validation score of the winner : 0.8409
mean test score of the winner       : 0.7996
optimism from selection             : +0.0413
```

*What to notice:* the optimism, 0.041, is $0.02 \times 2.04$, i.e. noise SD × expected maximum of 30 standard normals (section 3.3). It grows with the number of configurations tried and with the noise of each estimate.

### Lab 11 - Nested vs non-nested cross-validation (section 11.3)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

X, y = make_classification(n_samples=100, n_features=50, n_informative=3,
                           flip_y=0.2, random_state=0)      # small, noisy: easy to overfit the CV
grid = {"C": np.logspace(-3, 3, 7), "gamma": np.logspace(-4, 1, 6)}   # 42 configurations
non_nested, nested = [], []
for trial in range(5):
    inner = StratifiedKFold(5, shuffle=True, random_state=trial)
    outer = StratifiedKFold(5, shuffle=True, random_state=100 + trial)
    search = GridSearchCV(SVC(), grid, cv=inner, scoring="roc_auc")
    search.fit(X, y)
    non_nested.append(search.best_score_)                  # tuned and scored on the same folds
    nested.append(cross_val_score(search, X, y, cv=outer, scoring="roc_auc").mean())
non_nested, nested = np.array(non_nested), np.array(nested)
print(f"non-nested CV AUC: {non_nested.mean():.4f}")
print(f"nested CV AUC    : {nested.mean():.4f}")
print(f"optimistic bias  : {np.mean(non_nested - nested):+.4f}")
```

```text
non-nested CV AUC: 0.6676
nested CV AUC    : 0.6091
optimistic bias  : +0.0585
```

*What to notice:* `GridSearchCV.best_score_` is the best of 42 CV scores - selected and reported on the same folds - and overstates the AUC by about 0.06. Passing the whole `GridSearchCV` object to `cross_val_score` makes the tuning happen *inside* each outer training fold: that is nested CV in two lines. With little, noisy data and many configurations, the gap is large; with plenty of data and few configurations it shrinks.

### Lab 12 - Counting MV-HGAT's parameters (section 9.2)

```python
import sys
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
import torch
from drepo.model import MVHGAT

# Rebuild MV-HGAT with Fdataset's shapes (no data needed) and count its PARAMETERS
relations = {f"view:{v}": ("drug", "drug") for v in ["chem_cdk", "chem_ecfp", "gene_r"]}
relations.update({f"view:{v}": ("disease", "disease") for v in ["pheno_mim", "sem_mondo", "gene_d"]})
relations.update({"assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")})
in_drug = 3 * 593 + 313          # three drug similarity rows + the drug's association row
in_dis = 3 * 313 + 593
model = MVHGAT(relations, in_drug, in_dis, hidden=64, layers=2, heads=4, dropout=0.2, n_prop=6)

total = sum(p.numel() for p in model.parameters())
print(f"trainable parameters: {total:,}")
groups = {}
for name, p in model.named_parameters():
    key = name.split(".")[0] if not name.startswith("layers") else "layers (GAT + view attention)"
    groups[key] = groups.get(key, 0) + p.numel()
for k, v in groups.items():
    print(f"  {k:32s} {v:>9,}")
print("hyper-parameters in MVHGATConfig: 23 fields (set by you, not learned)")
```

```text
trainable parameters: 436,108
  prop_w                                   6
  prop_scale                               1
  bias                                     1
  gate                                     4
  W                                   36,864
  inp                                232,064
  layers (GAT + view attention)      167,168
hyper-parameters in MVHGATConfig: 23 fields (set by you, not learned)
```

---

## 13. In this project: the concepts in the real code

### 13.1 The hyper-parameter container: `MVHGATConfig`

```python
# (quoted from src/drepo/methods.py, MVHGATConfig - excerpt)
@dataclass
class MVHGATConfig:
    k: int = 10                           # neighbours kept per similarity view
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 5e-4
    epochs: int = 600
    neg_ratio: int = 2                    # sampled negatives per positive per epoch
    drop_edge: float = 0.2                # fraction of assoc edges hidden each epoch
    supervise_hidden: bool = True         # loss only on the links hidden this epoch
    cold_frac: float = 0.1                # share of diseases fully hidden per epoch
    ...
```

A `@dataclass` gives a typed record with defaults. `MVHGATMethod.__init__(self, cfg=None, **overrides)` copies the defaults and applies overrides with `setattr`, so `MVHGATMethod(hidden=128)` changes one hyper-parameter: exactly what `05_sensitivity.py` (`MVHGATMethod(**{p: v})`) and `04_ablation.py` (`MVHGATMethod(**kw)` with e.g. `{"prop_head": False}`) need. Keeping all hyper-parameters in one object is good practice: it can be saved with each result to make experiments reproducible (Unit E3).

### 13.2 The optimiser and the training loop

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict - abridged)
model = MVHGAT(relations, X["drug"].shape[1], X["disease"].shape[1], c.hidden, c.layers,
               c.heads, c.dropout, c.uniform_attention,
               n_prop=len(self.prop_names) if c.prop_head else 0).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
...
for _ in range(c.epochs):
    model.train()
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
    loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Line by line, in B1 vocabulary:

* `MVHGAT(..., c.hidden, c.layers, c.heads, c.dropout, ...)` - **architecture hyper-parameters** fix the hypothesis class and the number of **parameters**.
* `torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)` - Adam over all parameters; $\eta = 0.002$; L2 penalty $\lambda = 5\times10^{-4}$ added to the gradients (section 6.2: coupled L2, not AdamW).
* `for _ in range(c.epochs)` - 600 full-graph steps; the fixed epoch count is the project's (implicit) early-stopping choice, tuned on validation.
* `model.train()` - dropout on (in the GAT attention, input projections and layer outputs).
* `hide = torch.rand(...) < c.drop_edge` - each training link is hidden with probability 0.2 (a Bernoulli mask; Unit A3). **DropEdge-style regularisation** and the mechanism behind **hidden-link supervision**.
* `cold = torch.rand(n_d) < c.cold_frac`; `hide |= cold[pos % n_d]` - pick ~10% of diseases and hide all their links. `pos % n_d` converts a flat index to its disease (column) index (Unit A1). This is **"train the way you test"** for LODO.
* `Am[pos[~hide]] = True` - the visible association matrix for this epoch; every link-derived input is computed from `Am`.
* `sup = pos[hide]` - **supervise only the hidden links**, so the training task matches the test task.
* `logits, _ = model(...)` - forward pass.
* `loss = F.binary_cross_entropy_with_logits(...)` - the BCE **loss** (Bernoulli NLL) on hidden positives and sampled negatives.
* `opt.zero_grad(); loss.backward(); opt.step()` - clear old gradients, backpropagate, take one Adam step (section 8.6).

After the loop:

```python
# (quoted from src/drepo/methods.py, MVHGATMethod.fit_predict)
if full_assoc is not None:
    graphs["assoc>drug"], graphs["assoc>disease"] = full_assoc, full_assoc.T
model.eval()
P, deg = propagation(A_full), degrees(A_full)
with torch.no_grad():
    logits, _ = model(X, graphs, P=P, deg=deg)
```

* Restores the full **training** association graph (all training links visible - at test time there is nothing left to hide; the test links were already removed by `run_kfold`).
* `model.eval()` - dropout off; predictions become deterministic.
* `torch.no_grad()` - no gradient bookkeeping during inference.

### 13.3 Dropout inside the model

```python
# (quoted from src/drepo/model.py)
# DenseGAT.forward - attention dropout: randomly ignore some neighbours
out = torch.einsum("dsh,shk->dhk", self.drop(att), zs).reshape(h_dst.shape[0], -1)
# HeteroLayer.forward - dropout on each layer's output embeddings
new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
# MVHGAT.encode - dropout on the projected input features
h = {t: self.drop(F.elu(self.inp[t](X[t]))) for t in ("drug", "disease")}
```

All three `self.drop` are `nn.Dropout(dropout)` modules created in `__init__`, so they obey `model.train()` / `model.eval()` automatically.

### 13.4 Baselines: the same ideas, different settings

* `SCMFDD.fit_predict`: loss $= \|A - UV^\top\|^2$ (squared error) $+ \mu(\|U\|^2 + \|V\|^2)$ (**explicit L2** written into the loss, `mu=0.05`) $+$ Laplacian smoothness; optimiser `torch.optim.Adam([U, V], lr=self.lr)` with **no** `weight_decay` because the L2 term is already explicit. 400 epochs.
* `NIMCGCN.fit_predict`: `torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)`, dropout 0.3, `weighted_bce` loss, 1,500 epochs.
* `LAGCN.fit_predict`: `weight_decay=1e-4`, `dropout=0.0`, optional `drop_edge`, 1,000 epochs; also `gcn.train()` / `gcn.eval()`.
* `MBiRW` and `DRRS` have no gradient training at all: their hyper-parameters (`alpha`, `l`, `r`; `tau_rel`, `iters`, `rank`) still need choosing, but there are no learned weights - the "model" is an algorithm applied to the training matrix.

### 13.5 Cross-validation: `evaluation.py::run_kfold`

```python
# (quoted from src/drepo/evaluation.py, run_kfold - abridged)
for r in range(repeats):
    for f, test_pos, test_neg in kfold_splits(A, k, seed + r):
        A_tr = A.copy().ravel()
        A_tr[test_pos] = 0
        A_tr = A_tr.reshape(shape)
        neg_mask = (A == 0).ravel()
        neg_mask[test_neg] = False
        neg_mask = neg_mask.reshape(shape)

        S = method.fit_predict(data, A_tr, neg_mask, seed=seed * 1000 + r * 100 + f)
        idx = np.concatenate([test_pos, test_neg])
        y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
        s = S.ravel()[idx]
        m = metrics(y, s)
```

* `for r in range(repeats)` - **repeated** CV; `seed + r` gives each repeat a different stratified partition, the same for all methods (paired).
* `A_tr[test_pos] = 0` - training matrix with the test links hidden (they become "unknown").
* `neg_mask[test_neg] = False` - test negatives may not be used as training negatives.
* `seed=seed * 1000 + r * 100 + f` - a distinct, reproducible training seed for every (repeat, fold).
* `S.ravel()[idx]` - scores for exactly the test cells; `metrics` computes AUC/AUPR on them.
* No hyper-parameter is chosen inside this loop - the method arrives with its configuration fixed. That is what keeps the test folds clean.

### 13.6 The validation protocol: `scripts/05_sensitivity.py`

Section 11.4 quoted the split. The search loop:

```python
# (quoted from scripts/05_sensitivity.py)
GRID = {
    "hidden": [16, 32, 64, 128],
    "layers": [1, 2, 3],
    "neg_ratio": [1, 2, 5, 10],
    "k": [5, 10, 20, 40],
    "drop_edge": [0.1, 0.2, 0.4, 0.6],
}
...
for p in args.params.split(","):
    for v in GRID[p]:
        runs = []
        for s in range(args.seeds):
            S = MVHGATMethod(**{p: v}).fit_predict(data, A_tr, neg_mask, seed=s)
            runs.append(metrics(y, S.ravel()[idx]))
```

* `GRID` - a **one-at-a-time** design: each list is swept with all other hyper-parameters at their defaults.
* `MVHGATMethod(**{p: v})` - override one field of `MVHGATConfig`.
* `fit_predict(data, A_tr, neg_mask, seed=s)` - train on the training part of the validation split; several seeds to measure training noise.
* `metrics(y, S.ravel()[idx])` - score on the **validation** cells only.

The results (Fdataset, AUPR on validation, mean ± SD over 3 seeds) show both sensible capacity effects and how noisy single comparisons are:

| Hyper-parameter | values → AUPR |
|---|---|
| `hidden` | 16: 0.475, 32: 0.479, **64: 0.521**, 128: 0.465 |
| `layers` | 1: 0.526, **2: 0.521**, 3: 0.479 |
| `k` | 5: 0.522, **10: 0.521**, 20: 0.506, 40: 0.475 |
| `drop_edge` | 0.1: 0.505, **0.2: 0.521**, 0.4: 0.522, 0.6: 0.483 |
| `neg_ratio` | 1: 0.463, **2: 0.521**, 5: 0.523, 10: 0.487 |

(Bold = default.) Typical pattern: too little capacity or regularisation and too much both hurt, with a broad plateau in between; and several neighbours of the default are within noise of it - which is exactly when the one-standard-error rule says "keep the default".

---

## 14. Common mistakes and misconceptions

1. **Tuning on the test folds** (directly, or by re-running CV after looking at the results). Every decision must be made on validation data.
2. **Reporting the best validation score as the result.** It is optimistic (winner's curse). Report a separate test estimate or use nested CV.
3. **Forgetting `model.eval()`** before prediction (dropout still on → noisy, worse scores), or forgetting `model.train()` (dropout silently off).
4. **Forgetting `opt.zero_grad()`** → gradients accumulate across steps.
5. **Thinking weight decay in `torch.optim.Adam` is AdamW.** It is L2 added to the gradient; AdamW decouples it.
6. **Early stopping on the test fold**, or on a metric different from the one you report.
7. **Assuming more capacity is always better** because training loss improves. Watch validation.
8. **Comparing models tuned with different budgets** (your model tuned over 100 configurations, baselines run with defaults). The comparison then measures tuning effort, not method quality.
9. **Leaking test links into inputs** (features, graphs, similarity computed from `A`, degree counts). In link prediction, everything derived from the association matrix must be recomputed from training links.
10. **Using unstratified folds with 1% positives**, giving folds with different base rates and noisier AUPR.
11. **Confusing a hyper-parameter's validation sensitivity with its importance in general.** One-at-a-time results hold around one default and on one data set.
12. **Learning rate too high** → loss spikes, NaN; too low → apparent "convergence" that is actually a plateau. Always plot the training loss.
13. **Ignoring seed noise** when declaring one configuration better. Use several seeds and the one-SE rule.
14. **Treating CV fold SD as the uncertainty of the mean.** Folds are correlated (Unit A3 §8.4).
15. **Believing regularisation reduces training error.** It increases it; the benefit is on unseen data.

---

## 15. Exercises

(C) conceptual, (M) mathematical, (P) programming; stars indicate difficulty.

**Exercise 1 (C, ★).** Classify each as parameter or hyper-parameter: (a) the matrix `W` in the bilinear decoder; (b) `k = 10`; (c) `prop_w` (view weights); (d) `epochs = 600`; (e) the attention vectors `a_src`, `a_dst` of `DenseGAT`; (f) `neg_ratio`; (g) `MBiRW.alpha`; (h) the Adam moment estimates $m_t, v_t$.

<details><summary>Solution</summary>

(a) parameter (learned by Adam). (b) hyper-parameter (graph construction). (c) parameter - the model *learns* how much to trust each view; this is a modelling choice that turned a would-be hyper-parameter into a parameter. (d) hyper-parameter (optimisation / implicit early stopping). (e) parameters. (f) hyper-parameter (training-task design). (g) hyper-parameter (MBiRW has no learned parameters at all). (h) neither: they are **optimiser state** - internal running statistics of Adam, updated automatically but not part of the model and not chosen by you. (Adam's $\beta_1, \beta_2, \epsilon$ are hyper-parameters.)
</details>

**Exercise 2 (C, ★★).** Explain in your own words why choosing the number of epochs by looking at test-fold AUPR, and then reporting that AUPR, is a form of over-fitting - even though no weights were trained on the test fold.

<details><summary>Solution</summary>

The test score at each epoch is a noisy estimate of the true performance at that epoch. Picking the epoch with the highest *observed* test score preferentially picks an epoch whose estimate happened to be lucky (winner's curse). The decision (which epoch) was fitted to the noise in the test data, so the reported number is biased upward and the test set no longer measures generalisation. Over-fitting is not only about weights: any choice fitted to a data set over-fits that data set's noise. The fix is to choose the epoch on validation data (or fix it in advance, as the project does with `epochs=600` chosen on the validation split).
</details>

**Exercise 3 (M, ★★).** Prove the bias-variance decomposition $\mathbb E[(y - \hat f_S(x))^2] = \sigma^2 + (f(x) - \bar f(x))^2 + \mathrm{Var}_S(\hat f_S(x))$, stating clearly which independence assumption kills which cross term.

<details><summary>Solution</summary>

See section 5.2. Decompose $y - \hat f_S = \varepsilon + (f - \bar f) + (\bar f - \hat f_S)$. Expanding the square gives three squares and three cross terms:
* $2(f - \bar f)\mathbb E[\varepsilon] = 0$ because the noise has mean 0;
* $2(f - \bar f)\mathbb E_S[\bar f - \hat f_S] = 0$ by the definition $\bar f = \mathbb E_S[\hat f_S]$ (and $f - \bar f$ is a constant);
* $2\mathbb E[\varepsilon(\bar f - \hat f_S)] = 2\mathbb E[\varepsilon]\mathbb E[\bar f - \hat f_S] = 0$ because the **test noise is independent of the training set** $S$ (and has mean 0).
The squares are $\mathbb E[\varepsilon^2] = \sigma^2$, $(f - \bar f)^2$ = bias², and $\mathbb E_S[(\hat f_S - \bar f)^2]$ = variance. ∎
</details>

**Exercise 4 (M, ★★).** For gradient descent on $J(w) = \frac L2 (w - w^\star)^2$: (a) derive $w_{t+1} - w^\star = (1 - \eta L)(w_t - w^\star)$; (b) find all $\eta$ for which it converges; (c) with $L = 4$, $w_0 - w^\star = 1$, $\eta = 0.2$, how many steps until $|w_t - w^\star| < 10^{-3}$?

<details><summary>Solution</summary>

(a) $\nabla J = L(w - w^\star)$, so $w_{t+1} = w_t - \eta L(w_t - w^\star)$; subtract $w^\star$: $w_{t+1} - w^\star = (1 - \eta L)(w_t - w^\star)$.
(b) Converges iff $|1 - \eta L| < 1 \iff 0 < \eta < 2/L$.
(c) Factor $1 - 0.2 \times 4 = 0.2$; error after $t$ steps $= 0.2^t$. Need $0.2^t < 10^{-3} \iff t > \ln(10^{-3})/\ln(0.2) = 6.908/1.609 = 4.29$, so $t = 5$ steps ($0.2^5 = 3.2\times10^{-4}$; $0.2^4 = 1.6\times10^{-3}$ is not yet enough).
</details>

**Exercise 5 (M, ★★).** Show that with $\beta_1 = 0.9$, Adam's uncorrected $m_t$ under a constant gradient $g$ equals $(1 - 0.9^t)g$, and compute the relative size of the uncorrected first step if bias correction were omitted (for both moments, $\beta_2 = 0.999$).

<details><summary>Solution</summary>

$m_t = 0.9m_{t-1} + 0.1g$ with $m_0 = 0$ gives $m_t = 0.1g\sum_{s=0}^{t-1}0.9^s = 0.1g\frac{1 - 0.9^t}{1 - 0.9} = (1 - 0.9^t)g$. Similarly $v_t = (1 - 0.999^t)g^2$.
At $t = 1$ without correction: $m_1 = 0.1g$, $v_1 = 0.001g^2$, step $= \eta\frac{0.1g}{\sqrt{0.001}|g|} = \eta \times 0.1/0.0316 = 3.16\eta$ in the sign direction. With correction it is $\eta$. So omitting correction makes the first step about 3.16 times too *large* (because $v$ is far more under-estimated than $m$). Later the ratio tends to 1 as $0.9^t, 0.999^t \to 0$; the $v$ bias lasts longest (~1000 steps).
</details>

**Exercise 6 (M, ★★).** Derive the weight-decay form of the gradient step for $J(\theta) = L(\theta) + \frac\lambda2\|\theta\|^2$. For the project's $\eta$ and $\lambda$, what fraction of a weight would plain SGD decay away over 600 steps if the data gradient were zero? Why is the answer different for `torch.optim.Adam`?

<details><summary>Solution</summary>

$\nabla J = \nabla L + \lambda\theta$; step: $\theta \leftarrow \theta - \eta\nabla L - \eta\lambda\theta = (1 - \eta\lambda)\theta - \eta\nabla L$.
$\eta\lambda = 0.002 \times 5\times10^{-4} = 10^{-6}$; after 600 steps the factor is $(1 - 10^{-6})^{600} \approx e^{-0.0006} = 0.9994$: only 0.06% decayed.
`torch.optim.Adam` adds $\lambda\theta$ to the gradient *before* normalising by $\sqrt{\hat v}$. If the data gradient were zero, the normalised step would be $\approx \eta\,\mathrm{sign}(\theta)$ - a fixed 0.002 per step towards zero regardless of $\lambda$ (Lab 4). With nonzero data gradients, the L2 term's effect is scaled by $1/\sqrt{\hat v}$ per coordinate, so it is not a uniform multiplicative decay. AdamW restores the SGD-like behaviour.
</details>

**Exercise 7 (M, ★).** In inverted dropout with rate $p$, show that $\mathbb E[\tilde h] = h$ and $\mathrm{Var}(\tilde h_j) = \frac{p}{1-p}h_j^2$. What is the variance factor for the project's $p = 0.2$?

<details><summary>Solution</summary>

$\tilde h_j = m_j h_j/(1-p)$ with $m_j \sim \mathrm{Bernoulli}(1-p)$. $\mathbb E[\tilde h_j] = (1-p)h_j/(1-p) = h_j$. $\mathbb E[\tilde h_j^2] = (1-p)h_j^2/(1-p)^2 = h_j^2/(1-p)$, so $\mathrm{Var} = h_j^2/(1-p) - h_j^2 = \frac{p}{1-p}h_j^2$.
For $p = 0.2$: $0.2/0.8 = 0.25$, i.e. the dropout noise has SD $0.5|h_j|$. For $p = 0.5$ the factor is 1 (SD equal to the activation).
</details>

**Exercise 8 (C, ★★).** A friend tunes MV-HGAT's 5 sensitivity hyper-parameters with a full grid (4×3×4×4×4 = 768 configurations) using 5-fold CV on Fdataset and reports the best mean AUPR. Name two problems and propose a protocol that fixes them within a reasonable compute budget.

<details><summary>Solution</summary>

Problems: (1) **selection bias** - with 768 configurations, the best CV score is strongly inflated by the winner's curse; the reported number is not an estimate of anything. (2) The **test folds were used for tuning**; there is no untouched data left. Also (3) compute: 768 × 5 trainings.
Protocol: hold out the CV test folds entirely from tuning. Within the training part (or a validation split as in `05_sensitivity.py`), run **random search** over, say, 40 configurations with 1-2 seeds, keep the top 5, re-evaluate them with 3 seeds, and pick the simplest within one SE of the best. Then run the outer 5-fold CV once with that configuration. For a fully unbiased estimate, put this whole search inside each outer fold (nested CV) - 5× the tuning cost - or at least use a final held-out test set not used for any decision.
</details>

**Exercise 9 (P, ★★).** Modify Lab 1 to implement momentum ($\beta = 0.9$) and Adam ($\eta = 0.1$) on $J(w) = (w-3)^2$ and print the first 3 iterates of each. Check against the tables in sections 8.4 and 8.5.

<details><summary>Solution</summary>

```python
import numpy as np

def momentum(lr=0.1, beta=0.9, steps=3, w=0.0):
    v, out = 0.0, []
    for _ in range(steps):
        g = 2 * (w - 3)
        v = beta * v + g
        w = w - lr * v
        out.append(round(w, 3))
    return out

def adam(lr=0.1, b1=0.9, b2=0.999, eps=1e-8, steps=3, w=0.0):
    m = v = 0.0
    out = []
    for t in range(1, steps + 1):
        g = 2 * (w - 3)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        out.append(round(float(w), 3))
    return out

print("momentum:", momentum())
print("Adam    :", adam())
```

```text
momentum: [0.6, 1.62, 2.814]
Adam    : [0.1, 0.2, 0.3]
```

Both match the hand tables. Momentum accelerates (steps 0.6, 1.02, 1.19); Adam moves ~0.1 per step because the normalised gradient is ≈ 1 in magnitude.
</details>

**Exercise 10 (P, ★★).** Implement **group k-fold by disease** for a drugs × diseases matrix: each disease (column) belongs to exactly one fold, and a fold's test set is all cells of its diseases. Verify on a toy 20 × 10 matrix that no disease appears in two test folds and that every cell is tested once. How does this relate to `run_lodo`?

<details><summary>Solution</summary>

```python
import numpy as np

def disease_group_folds(A, k, seed):
    rng = np.random.default_rng(seed)
    n_r, n_d = A.shape
    diseases = rng.permutation(n_d)
    for f, ds in enumerate(np.array_split(diseases, k)):
        cells = (np.arange(n_r)[:, None] * n_d + ds[None, :]).ravel()   # flat indices of those columns
        yield f, np.sort(ds), cells

rng0 = np.random.default_rng(0)
A = (rng0.random((20, 10)) < 0.1).astype(float)
seen_cells, seen_dis = [], []
for f, ds, cells in disease_group_folds(A, 5, seed=0):
    seen_cells += list(cells); seen_dis += list(ds)
    print(f"fold {f}: diseases {ds.tolist()}, {len(cells)} test cells, {int(A.ravel()[cells].sum())} positives")
print("each disease in exactly one fold:", sorted(seen_dis) == list(range(10)))
print("every cell tested once:", sorted(seen_cells) == list(range(A.size)))
```

```text
fold 0: diseases [4, 6], 40 test cells, 2 positives
fold 1: diseases [2, 7], 40 test cells, 5 positives
fold 2: diseases [3, 5], 40 test cells, 5 positives
fold 3: diseases [0, 9], 40 test cells, 5 positives
fold 4: diseases [1, 8], 40 test cells, 4 positives
each disease in exactly one fold: True
every cell tested once: True
```

Training for fold $f$ would zero all columns in `ds` (and remove them from the negative pool). With $k$ = number of diseases this is exactly leave-one-disease-out (`run_lodo`), i.e. leave-one-group-out CV. Note the positives per fold now vary (2 to 5): stratification by label is no longer guaranteed when whole groups move together.
</details>

**Exercise 11 (C/M, ★★★).** Using section 3.3, estimate the expected optimism when picking the best of the 4 `hidden` values in `05_sensitivity.py`, if each validation AUPR estimate has noise SD 0.012 (a rough value from the seed SDs) and all four were truly equal. Is the observed gap between `hidden=64` (0.521) and the next best (0.479) explained by selection alone?

<details><summary>Solution</summary>

$\mathbb E[\max$ of 4 standard normals$] \approx 1.03$. Optimism $\approx 0.012 \times 1.03 \approx 0.012$. The observed gap to the next best is 0.042, about 3.5 times this, and the per-configuration seed SDs are 0.007-0.018. So selection noise alone is unlikely to explain it; `hidden=64` is probably genuinely better on this split. Caveats: the seed SD does not include split noise (one fixed split), so the true noise is larger than 0.012; and the selected value's 0.521 is still somewhat optimistic as an absolute number. (The expected maximum of 4 i.i.d. standard normals is $\approx 1.029$; check by simulation as in Lab 10.)
</details>

**Exercise 12 (P, ★★★).** Turn Lab 6 into a proper early-stopping experiment: split the 300 training examples into 240 for training and 60 for an *inner* validation set; train with patience 30 on inner-validation AUC; then report the AUC on the big 2,000-example set at the restored best epoch. Compare with the "best val AUC" column of Lab 6 and explain the difference.

<details><summary>Solution</summary>

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
def make(n):
    X = rng.normal(size=(n, 40)).astype(np.float32)
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 2] - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)
Xtr, ytr = make(300)
Xte, yte = make(2000)                 # plays the role of the untouched test set
Xin, yin, Xiv, yiv = Xtr[:240], ytr[:240], Xtr[240:], ytr[240:]

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.002)
best, best_state, best_ep, patience, wait = -1.0, None, 0, 30, 0
for ep in range(1, 301):
    net.train()
    loss = F.binary_cross_entropy_with_logits(net(Xin).squeeze(1), yin)
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        auc = roc_auc_score(yiv.numpy(), net(Xiv).squeeze(1).numpy())
    if auc > best:
        best, best_state, best_ep, wait = auc, copy.deepcopy(net.state_dict()), ep, 0
    else:
        wait += 1
        if wait >= patience:
            break
net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    test_auc = roc_auc_score(yte.numpy(), net(Xte).squeeze(1).numpy())
print(f"stopped at epoch {ep}, best inner-val AUC {best:.4f} at epoch {best_ep}")
print(f"test AUC of the restored model: {test_auc:.4f}")
```

```text
stopped at epoch 123, best inner-val AUC 0.8323 at epoch 93
test AUC of the restored model: 0.7518
```

The inner-validation AUC of the chosen epoch (0.83) is optimistic - it is the maximum over 123 epochs of a noisy estimate from only 60 examples - while the honest test AUC of the restored model is 0.75. Lab 6's "best val AUC" column (0.761 without regularisation) chose the epoch on the same 2,000 examples it reports on, so it is also optimistic, though less so, because 2,000 examples give a far less noisy estimate than 60. The early-stopped model (0.752) comes close to that figure even though it was trained on 20% fewer examples and its epoch was chosen without ever looking at the 2,000 test examples. Lessons: early stopping works, but its own validation score must not be reported; small validation sets make the stopping point noisy. Remedies: a larger validation set, smoothing the validation curve, or adding weight decay so the curve is flatter.
</details>

**Exercise 13 (M, ★★).** For the ridge solution $w_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$, show using the SVD $X = U\Sigma V^\top$ that the component of $w_\lambda$ along the $j$-th right singular vector equals $\frac{s_j^2}{s_j^2 + \lambda}$ times the least-squares component (for $s_j > 0$).

<details><summary>Solution</summary>

$X^\top X = V\Sigma^2V^\top$, so $X^\top X + \lambda I = V(\Sigma^2 + \lambda I)V^\top$ and its inverse is $V(\Sigma^2 + \lambda I)^{-1}V^\top$. Also $X^\top y = V\Sigma U^\top y$. Hence $w_\lambda = V(\Sigma^2 + \lambda I)^{-1}\Sigma U^\top y$, whose $j$-th component in the $V$ basis is $\frac{s_j}{s_j^2 + \lambda}u_j^\top y$. Least squares ($\lambda = 0$) gives $\frac{1}{s_j}u_j^\top y$. Ratio: $\frac{s_j/(s_j^2 + \lambda)}{1/s_j} = \frac{s_j^2}{s_j^2 + \lambda}$. ∎ Directions with large singular values (well determined by the data) are kept; directions with small singular values (where least squares would amplify noise) are shrunk most - this is how L2 reduces variance. ($d_j = s_j^2$ are the eigenvalues mentioned in section 6.1.)
</details>

---

## 16. Answers to the PREREQUISITES.md self-check questions (Unit B1)

### Self-check 1: "Why must hyper-parameters be tuned on a validation split and not on the test folds?"

**Short answer.** Because the test folds exist to give an *unbiased* estimate of how the final, fully specified method performs on data it has never influenced. Choosing hyper-parameters is a form of fitting; if the test folds are used to choose, the method has been fitted to them, and their score becomes optimistically biased. It no longer measures generalisation.

**The mechanism in detail.**

1. **Every score is noisy.** A test-fold AUPR is computed on ~387 positives among ~37,000 cells; across folds and seeds the project sees SDs of 0.02-0.03 in AUPR. Each measured score = true performance + noise.
2. **Selection amplifies noise.** When you compare many configurations and keep the best measured one, you preferentially keep the configuration whose noise was positive. The winner's measured score exceeds its true score by about (noise SD) × (expected max of $M$ standard normals): ≈ 1.2 SD for 5 candidates, ≈ 2 SD for 30 (Lab 10: +0.041 with SD 0.02). For this project, that is comparable to the differences between methods (MV-HGAT vs SCMFDD AUPR differ by 0.007; vs DRRS by 0.10).
3. **The bias does not show up anywhere.** Nothing in the code crashes; the numbers simply look better than they will be on new data. The only defence is procedural: keep decision-making data and reporting data separate.
4. **It also corrupts comparisons.** If your model is tuned on the test folds and the baselines are not (or less), the comparison is unfair in your favour.

**What the project does.** `scripts/05_sensitivity.py` hides 20% of known links and 20% of unknown pairs as a **validation split**, trains on the rest, and scores the candidates on the hidden 20% only. The CV test folds are never consulted when choosing `hidden`, `layers`, `k`, `neg_ratio`, `drop_edge` (or `lr`, `weight_decay`, `dropout`, `epochs`). The chosen defaults are frozen in `MVHGATConfig` before `03_evaluate.py` runs the CV. The baselines' key hyper-parameters were tuned with the same split, keeping the comparison fair.

**The honest footnote.** The validation cells are drawn from the same full matrix that CV later partitions, so roughly 20% of each test fold's cells were validation cells during tuning. They were never used to *score* tuning on test folds, and only ~19 one-at-a-time configurations were compared, so the bias is small; but the fully rigorous alternatives are a final held-out test set or **nested CV** (tune inside each outer training fold; Lab 11 shows non-nested optimism of +0.06 in a small, noisy, 42-configuration example).

**What validation is allowed to do.** Anything: be looked at repeatedly, drive grid/random searches, decide early stopping, pick features. The price is that its scores become optimistic - which is fine, because you do not report them as your final result.

### Self-check 2: "What do dropout and weight decay each do?"

Both are **regularisers**: they reduce over-fitting (variance) at the cost of a little extra training error (bias). They act very differently.

**Weight decay (L2 regularisation)** - *a constraint on the size of the weights.*
* **What it does mathematically:** adds $\frac\lambda2\|\theta\|^2$ to the loss; the gradient gains a term $\lambda\theta$; for SGD each step becomes $\theta \leftarrow (1 - \eta\lambda)\theta - \eta\nabla L$: weights shrink geometrically unless the data gradient keeps pushing them out.
* **Effect on the learned function:** large weights are needed to make sharp, wiggly decision functions that fit noise; penalising them biases the model towards smoother, simpler functions. In linear models it shrinks poorly determined directions most (factor $s_j^2/(s_j^2 + \lambda)$).
* **Probabilistic meaning:** MAP estimation with a Gaussian prior centred at zero ("I believe weights are small unless the data insist").
* **Always on**, at train and test alike (it changes the weights you end up with; there is nothing to switch off at test time).
* **In this project:** `weight_decay = 5e-4` in `torch.optim.Adam`, i.e. **L2 added to the gradient** before Adam's normalisation - not AdamW's decoupled decay (Lab 4). SCMFDD writes its L2 penalty explicitly into its loss (`mu * (U.pow(2).sum() + V.pow(2).sum())`); NIMCGCN and LAGCN use `weight_decay=1e-4`.

**Dropout** - *noise injected into the network's activations during training.*
* **What it does mechanically:** on every forward pass in training mode, each unit is zeroed with probability $p$ and the survivors are scaled by $1/(1-p)$ so the expected activation is unchanged. At evaluation (`model.eval()`), it is the identity.
* **Effect on the learned function:** units cannot co-adapt (rely on specific partners), so they learn individually useful, redundant features; the network effectively trains an ensemble of exponentially many thinned sub-networks with shared weights, and the test-time network approximates their average - an averaging that reduces variance.
* **Switched on only during training** - which is why `model.train()` and `model.eval()` matter.
* **In this project:** `dropout = 0.2` (each activation dropped with probability 0.2, survivors × 1.25), applied to the input projections (`MVHGAT.encode`), each layer's output embeddings (`HeteroLayer.forward`), and the GAT attention coefficients (`DenseGAT.forward`) - the last means each node randomly ignores some of its neighbours each epoch. NIMCGCN uses 0.3; LAGCN 0.0.

**Side by side:**

| | Weight decay | Dropout |
|---|---|---|
| Acts on | parameters (weights) | activations (and attention) |
| Kind | deterministic penalty | random noise / implicit ensemble |
| Train vs test | same | on in training, off in evaluation |
| Bayesian reading | Gaussian prior on weights | approximate model averaging |
| Project setting | 5e-4 (L2 in Adam) | 0.2 |

Lab 6 shows both on an over-fitting MLP: without regularisation training BCE → 0 while validation AUC is 0.754; weight decay raises it to 0.78-0.79, dropout to 0.771, and combining them gives 0.793.

---

## 17. Summary and cheat sheet

**Setup**
* Risk $R(\theta) = \mathbb E[\ell(f_\theta(x), y)]$; empirical risk $\hat R_S = \frac1n\sum\ell$; ERM + regulariser.
* Training error is optimistic; a fixed model's test error is unbiased; selection among $M$ models inflates the winner by ≈ SD × E[max of $M$ normals] (1.16 for 5, 1.87 for 20, 2.04 for 30).

**Splits**
* Train → fit parameters. Validation → all decisions. Test → one final estimate, no decisions.
* Link prediction leakage: test links in inputs, test negatives in the negative pool, label-derived similarities, tuning on test folds.

**Bias-variance (squared loss)**
* $\mathbb E[(y - \hat f)^2] = \sigma^2 + \text{bias}^2 + \text{variance}$.
* Under-fit: high bias, train ≈ val, both bad. Over-fit: high variance, train ≪ val.

**Regularisation**
* L2: $J + \frac\lambda2\|\theta\|^2$; SGD step $\theta \leftarrow (1 - \eta\lambda)\theta - \eta\nabla L$; ridge $w = (X^\top X + \lambda I)^{-1}X^\top y$; shrink factor $s^2/(s^2 + \lambda)$; = Gaussian-prior MAP.
* `torch.optim.Adam(weight_decay=λ)` = L2 in the gradient (coupled); `AdamW` = decoupled decay.
* Dropout: $\tilde h = m\odot h/(1-p)$, $\mathbb E[\tilde h] = h$, off in `eval()`.
* Early stopping: epochs as capacity; monitor the reported metric; choose on validation.

**Losses**
* MSE ↔ Gaussian NLL; BCE ↔ Bernoulli NLL; weighted BCE ↔ weighted likelihood; hinge (SVM); BPR (ranking).
* Optimise a smooth surrogate (BCE) for a non-differentiable target (AUPR).

**Optimisers**
* GD: $\theta \leftarrow \theta - \eta\nabla J$; quadratic: converge iff $0 < \eta < 2/L$; rate $(\kappa - 1)/(\kappa + 1)$ at best.
* SGD: unbiased noisy gradient from a mini-batch.
* Momentum: $v \leftarrow \beta v + g$, $\theta \leftarrow \theta - \eta v$; tuned rate $(\sqrt\kappa - 1)/(\sqrt\kappa + 1)$.
* Adam: $m, v$ EMAs ($\beta_1 = 0.9$, $\beta_2 = 0.999$), bias-corrected, step $\eta\hat m/(\sqrt{\hat v} + \epsilon)$ ≈ $\eta$ per coordinate; project $\eta = 0.002$, 600 full-graph steps.
* Loop: `train()` → forward → loss → `zero_grad()` → `backward()` → `step()`; then `eval()` + `no_grad()`.

**Parameters vs hyper-parameters**
* Parameters: learned (MV-HGAT: 436,108). Hyper-parameters: chosen (23 fields of `MVHGATConfig`).

**Cross-validation**
* k-fold (5 or 10), stratified (positives and negatives split separately), repeated (different seeds), grouped (LODO = leave-one-disease-out).
* CV estimates the procedure's average performance; fold scores are correlated.

**Tuning**
* Grid < random search for the same budget; OAT for sensitivity; log scale for $\eta$, $\lambda$.
* One-SE rule; several seeds.
* Nested CV for an unbiased estimate of "tune + train"; `cross_val_score(GridSearchCV(...))`.

---

## 18. Curated further resources

All links were checked on 2026-10-01 (publisher pages for paywalled papers sometimes block automated checks but are standard DOI/publisher links).

**Courses and lecture notes**
* [Stanford CS229 Machine Learning - main lecture notes (PDF)](https://cs229.stanford.edu/main_notes.pdf) - **Free.** Rigorous notes on supervised learning, logistic regression, generalisation, regularisation and model selection; the best single written reference for this unit.
* [Cornell CS4780 Machine Learning for Intelligent Systems (Kilian Weinberger)](https://www.cs.cornell.edu/courses/cs4780/2018fa/) with [lecture notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/) - **Free.** Exceptionally clear lectures (videos on YouTube) on bias-variance, ERM, regularisation and model selection.
* [MIT 6.036 Introduction to Machine Learning (OCW, Fall 2020)](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/) - **Free.** Gentler MIT course; good on gradient descent, regularisation and evaluation.
* [Stanford CS231n notes: Optimization](https://cs231n.github.io/optimization-1/), [Neural Networks Part 2 (data, regularisation, dropout)](https://cs231n.github.io/neural-networks-2/) and [Part 3 (learning, SGD, momentum, Adam, hyper-parameter search)](https://cs231n.github.io/neural-networks-3/) - **Free.** The practical "how to train a network" reference.
* [Andrew Ng, Machine Learning Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction) - **Free to audit, paid certificate.** Courses 1-2 cover exactly this unit at a beginner pace (recommended in PREREQUISITES.md).
* [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) - **Free.** Short interactive modules on generalisation, train/validation/test, regularisation and logistic regression.

**Interactive explainers**
* [MLU-Explain: The Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/), [Train, Test, and Validation Sets](https://mlu-explain.github.io/train-test-validation/), and [Cross-Validation](https://mlu-explain.github.io/cross-validation/) (Amazon) - **Free.** Beautiful scrollable visual explanations.
* [Distill: "Why Momentum Really Works" (Goh, 2017)](https://distill.pub/2017/momentum/) - **Free.** Interactive, mathematically precise treatment of momentum and condition numbers.
* [Sebastian Ruder, "An overview of gradient descent optimization algorithms"](https://www.ruder.io/optimizing-gradient-descent/) ([arXiv version](https://arxiv.org/abs/1609.04747)) - **Free.** The standard survey of SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam.

**Videos (StatQuest - free)**
* [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) - quick intuition.
* [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) - k-fold CV visually.
* [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) - the algorithm by hand.
* [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) - what the L2 penalty does.

**Textbooks**
* [Goodfellow, Bengio & Courville, *Deep Learning* (2016)](https://www.deeplearningbook.org/) - **Free online.** Ch. 5 (ML basics, capacity, bias-variance, MLE/MAP), ch. 7 (regularisation: L2, early stopping, dropout), ch. 8 (optimisation: SGD, momentum, Adam).
* [James, Witten, Hastie, Tibshirani (& Taylor), *An Introduction to Statistical Learning*](https://www.statlearning.com/) - **Free PDF (R and Python editions).** Ch. 2 (bias-variance), 5 (cross-validation, bootstrap), 6 (ridge/lasso): the gentlest rigorous treatment.
* [Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) - **Free PDF.** Ch. 7 (model assessment and selection, including the "wrong way to do cross-validation") is essential reading.
* [Zhang et al., *Dive into Deep Learning*](https://d2l.ai/) - **Free.** Runnable PyTorch notebooks for weight decay, dropout, and every optimiser in this unit.
* [Murphy, *Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html) - **Free PDF.** Ch. 4 (MLE, MAP, regularisation) and ch. 8 (optimisation) from a probabilistic angle.

**Key papers**
* [Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - **Free.** The Adam algorithm and bias-correction derivation.
* [Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) - **Free.** Why L2 ≠ weight decay in Adam; introduces AdamW.
* [Srivastava et al. (2014), "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", *JMLR*](https://jmlr.org/papers/v15/srivastava14a.html) - **Free.** The dropout paper.
* [Rong et al. (2020), "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification"](https://arxiv.org/abs/1907.10903) - **Free.** The graph analogue of dropout used (as `drop_edge`) in this project.
* [Bergstra & Bengio (2012), "Random Search for Hyper-Parameter Optimization", *JMLR*](https://jmlr.org/papers/v13/bergstra12a.html) - **Free.** Why random search beats grid search.
* [Cawley & Talbot (2010), "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", *JMLR*](https://jmlr.org/papers/v11/cawley10a.html) - **Free.** The definitive argument for nested CV.
* [Varma & Simon (2006), "Bias in error estimation when using cross-validation for model selection", *BMC Bioinformatics*](https://doi.org/10.1186/1471-2105-7-91) - **Free.** Demonstrates the bias on biological data.
* [Bates, Hastie & Tibshirani (2021), "Cross-validation: what does it estimate and how well does it do it?"](https://arxiv.org/abs/2104.00673) - **Free.** What CV actually estimates; why naive CV intervals are too narrow.
* [Raschka (2018), "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning"](https://arxiv.org/abs/1811.12808) - **Free.** Long, practical survey bridging this unit with A3's statistical tests.
* [Belkin et al. (2019), "Reconciling modern machine learning practice and the bias-variance trade-off"](https://arxiv.org/abs/1812.11118) - **Free.** Double descent: how the classical U-curve extends to over-parameterised models.

**Documentation**
* [PyTorch `torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) - **Free.** The exact algorithm PyTorch runs, including how `weight_decay` enters the gradient.
* [scikit-learn User Guide: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [Nested versus non-nested cross-validation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) - **Free.** All CV iterators (KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut) and the nested-CV pattern of Lab 11.

---

## 19. Glossary

* **Adam:** adaptive optimiser combining momentum (first-moment EMA) with per-coordinate scaling by the root of the second-moment EMA, with bias correction.
* **AdamW:** Adam with weight decay applied directly to the weights, decoupled from the gradient normalisation.
* **Bias (statistical):** systematic error of the average predictor relative to the truth.
* **Bias-variance decomposition:** expected squared error = noise + bias² + variance.
* **Capacity:** richness of the set of functions a model can represent.
* **Condition number $\kappa$:** ratio of largest to smallest curvature; governs how slowly gradient descent converges.
* **Cross-validation (k-fold):** rotate which fold is the test set; average the scores.
* **Data leakage:** any path by which test-label information influences training or model selection.
* **DropEdge:** randomly removing graph edges during training as a regulariser.
* **Dropout:** randomly zeroing activations during training (scaled by $1/(1-p)$); off at evaluation.
* **Early stopping:** stopping training at the epoch with the best validation metric.
* **Empirical risk minimisation (ERM):** choosing parameters that minimise the average training loss.
* **Epoch:** one pass over the training data; in this project, one full-graph optimiser step.
* **Generalisation / generalisation gap:** performance on unseen data / the difference between test and training error.
* **Gradient descent:** iterative update $\theta \leftarrow \theta - \eta\nabla J$.
* **Group k-fold / leave-one-group-out:** CV where whole groups (e.g. diseases) are assigned to folds.
* **Hyper-parameter:** a setting chosen before training (not learned by the optimiser).
* **Hypothesis class:** the set of functions a model family can represent.
* **i.i.d.:** independent and identically distributed.
* **L1 / L2 regularisation:** penalties $\lambda\|\theta\|_1$ / $\frac\lambda2\|\theta\|_2^2$.
* **Learning curve:** training and validation performance plotted against epochs or data size.
* **Learning rate $\eta$:** step size of the optimiser.
* **Loss function:** per-example measure of prediction error minimised in training.
* **Mini-batch SGD:** gradient descent using the gradient of a random subset of examples.
* **Momentum:** accumulating an exponentially decaying sum of past gradients to accelerate along consistent directions.
* **Nested cross-validation:** an outer CV for evaluation with an inner CV for tuning inside each outer training set.
* **One-at-a-time (OAT) sensitivity analysis:** varying one hyper-parameter while keeping others fixed.
* **One-standard-error rule:** prefer the simplest configuration within one SE of the best.
* **Optimiser state:** internal running statistics of an optimiser (e.g. Adam's $m$, $v$).
* **Over-fitting / under-fitting:** fitting noise (high variance) / failing to fit the signal (high bias).
* **Parameter:** a quantity learned from data by the optimiser (weights, biases).
* **Patience:** number of epochs without validation improvement before early stopping triggers.
* **Random search:** sampling hyper-parameter configurations at random.
* **Regularisation:** any technique that reduces test error by restricting effective capacity.
* **Ridge regression:** linear least squares with an L2 penalty.
* **Risk (true / empirical):** expected loss under the data distribution / average loss on a sample.
* **Stratification:** keeping class proportions equal across folds.
* **Surrogate loss:** a smooth loss optimised in place of a non-differentiable target metric.
* **Test set:** data used once, for the final unbiased estimate.
* **Validation set:** data used for model selection and tuning.
* **Variance (of a learner):** sensitivity of the learned predictor to the particular training sample.
* **Weight decay:** shrinking weights by a constant factor each step; equals L2 regularisation for SGD but not for Adam.
* **Winner's curse (selection bias):** the best of several noisy estimates overstates its true value.
