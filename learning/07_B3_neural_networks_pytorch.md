# Unit B3 — Neural Networks and PyTorch

*From a single artificial neuron to the full training loop of MV-HGAT.*

---

## 0. Before you start

**Prerequisites (earlier units in this course)**

| Unit | What you need from it here |
|---|---|
| A1 Python / NumPy / pandas | arrays, shapes, broadcasting, boolean masks, fancy indexing |
| A2 Linear algebra | matrix–vector and matrix–matrix products, transpose, the bilinear form $x^\top W y$ |
| A3 Probability & statistics | the Bernoulli distribution, likelihood, binary cross-entropy (BCE) |
| B1 Supervised learning | train/validation/test, gradient descent, Adam, over-fitting, weight decay, dropout as an idea |

If you can multiply a $3\times 2$ matrix by a $2$-vector in your head and you know why we minimise
$-\log p$ for a correct label, you are ready.

**Estimated study time:** 20–28 hours (about 8 h reading and re-deriving, 8 h running and modifying
the code, 6–10 h on the exercises, 2 h walking through `model.py` and `methods.py` with this chapter
open next to them).

**Learning objectives.** After this unit you will be able to:

1. Write the equations of a multi-layer perceptron (MLP) in both "one example" and "batched" form, and
   prove that without non-linear activations depth adds no expressive power.
2. State the universal approximation theorem informally, build a "bump" out of ReLUs, and explain what
   the theorem does **not** promise.
3. Give the formula, derivative, range and main failure mode of sigmoid, tanh, ReLU, LeakyReLU, ELU and
   softplus, and say which of them the project uses and where.
4. Derive the softmax Jacobian, show that softmax + cross-entropy has gradient $p-y$, and explain
   exactly what `masked_fill(~mask, -inf)` followed by `softmax` computes, including the empty-row case.
5. Run backpropagation by hand through a two-layer network (every intermediate value and gradient), and
   check your numbers against PyTorch autograd and against finite differences.
6. Derive the matrix-form gradients of a linear layer ($\partial L/\partial W = G^\top X$ etc.).
7. Use PyTorch tensors confidently: dtype, device, broadcasting, views vs copies, indexing, `einsum`.
8. Explain autograd (leaf tensors, `grad_fn`, gradient accumulation, `detach`, `no_grad`) and write a
   correct training loop from memory (forward → loss → `zero_grad` → `backward` → `step`).
9. Build models with `nn.Module`, `nn.Linear`, `nn.Parameter`, `nn.ModuleList`, `nn.ModuleDict`, and
   count their parameters by hand.
10. Distinguish `model.train()`/`model.eval()` from `torch.no_grad()`, and LayerNorm from BatchNorm.
11. Justify Xavier/Glorot initialisation from a variance argument, and explain residual connections as
    a "gradient highway".
12. Avoid the classic numerical traps (overflow in `exp`, `log(0)`, NaN from empty softmax rows) using
    log-sum-exp and `binary_cross_entropy_with_logits`.
13. Make runs reproducible with seeds and know the limits of determinism on a GPU.
14. Debug shape errors systematically.

---

## 1. Motivation: why this unit matters for *this* project

Open `src/drepo/methods.py` and scroll to `MVHGATMethod.fit_predict`. Strip away the
drug-repositioning details and what remains is the following skeleton, which runs 600 times:

```py
model.train()
logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
opt.zero_grad()
loss.backward()
opt.step()
```

Every line of that loop is a concept from this unit:

* `model` is an `nn.Module` (the `MVHGAT` class in `model.py`) whose parameters are a few hundred
  thousand numbers stored in `nn.Linear` layers and `nn.Parameter`s.
* `model(...)` is the **forward pass**: a long chain of matrix products, LeakyReLU, softmax, tanh,
  LayerNorm, ELU and dropout that turns similarity rows into a $593\times 313$ matrix of **logits**
  (one real number per drug–disease pair for Fdataset).
* `binary_cross_entropy_with_logits` is the numerically safe form of BCE (Section 8).
* `loss.backward()` is **backpropagation**, done automatically by **autograd** (Sections 6 and 10).
* `opt.step()` is one **Adam** update; `opt.zero_grad()` clears gradients that PyTorch would
  otherwise *add up* across epochs.

After training, the code switches modes:

```py
model.eval()
P, deg = propagation(A_full), degrees(A_full)
with torch.no_grad():
    logits, _ = model(X, graphs, P=P, deg=deg)
self.last = (model, X, graphs, P, deg)
return torch.sigmoid(logits).cpu().numpy()
```

`model.eval()` turns dropout off; `torch.no_grad()` stops PyTorch from recording the computation for
backpropagation; `torch.sigmoid` turns logits into probabilities; `.cpu().numpy()` copies the result off
the RTX 3050 GPU so that the evaluation code (written in NumPy and scikit-learn) can compute AUC/AUPR.

**A concrete example.** Suppose that in some epoch the known link *(drug 17, disease 42)* is hidden and
used as a supervised positive, and the model currently gives it logit $z=-1.0$, i.e. probability
$\sigma(-1)=0.27$. The loss for that pair is $-\log 0.27 = 1.31$ and its gradient with respect to the
logit is $\sigma(z)-1 = -0.73$ (Section 8 derives this). Backpropagation pushes that $-0.73$ back
through the bilinear decoder $h_{17}^\top W h_{42}$, through two attention layers, through LayerNorm and
ELU, all the way to the first `nn.Linear` that read drug 17's similarity rows, and every one of the
model's weights receives a small nudge that would make this logit larger next time. If you understand
that one sentence completely — which this chapter aims for — you understand how the project learns.

The two self-check questions for this unit in `docs/PREREQUISITES.md` are:

1. *Why is `torch.no_grad()` used at prediction time?*
2. *What does `masked_fill(~mask, -inf)` followed by softmax achieve?*

Both are answered briefly where the concepts appear and in depth in Section 19.

---

## 2. From a neuron to a network

### 2.1 The artificial neuron

**Intuition.** A neuron takes several numbers, decides how much each one matters, adds them up, and
then makes a decision about how strongly to "fire". "How much each matters" is a set of **weights**,
the baseline firing tendency is a **bias**, and the decision rule is an **activation function**.

**Definition.** Given an input vector $x\in\mathbb{R}^n$, a weight vector $w\in\mathbb{R}^n$, a bias
$b\in\mathbb{R}$ and an activation $\phi:\mathbb{R}\to\mathbb{R}$, a neuron computes

$$
z = w^\top x + b = \sum_{k=1}^{n} w_k x_k + b, \qquad a = \phi(z).
$$

$z$ is called the **pre-activation** (or *logit* when it is the final output of a classifier) and $a$
the **activation**.

Three classical choices of $\phi$ give three classical models:

| $\phi(z)$ | Model | Output |
|---|---|---|
| $z$ (identity) | linear regression | any real number |
| $\mathbb{1}[z>0]$ (step) | Rosenblatt's perceptron (1958) | class 0/1 |
| $\sigma(z)=1/(1+e^{-z})$ | logistic regression | probability in $(0,1)$ |

**Geometry.** The set $\{x : w^\top x + b = 0\}$ is a hyperplane (a line in 2-D). A single neuron with a
step or sigmoid activation can therefore only separate classes with a straight cut. $w$ is the normal
vector of the cut and $-b/\lVert w\rVert$ its signed distance from the origin.

### 2.2 The perceptron learning rule and its limit

Rosenblatt's perceptron learns with a beautifully simple rule. For a training example $(x, y)$ with
$y\in\{0,1\}$ and prediction $\hat y = \mathbb{1}[w^\top x+b>0]$:

$$
w \leftarrow w + \eta\,(y-\hat y)\,x, \qquad b \leftarrow b + \eta\,(y-\hat y).
$$

If the prediction is right, nothing changes; if a positive was missed, $x$ is added to $w$ (tilting the
hyperplane towards it); if a negative was wrongly accepted, $x$ is subtracted. The **perceptron
convergence theorem** says that if the data are linearly separable, this rule finds a separating
hyperplane in a finite number of updates.

The catch, made famous by Minsky and Papert (1969), is the word *if*. The **XOR** function

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

is not linearly separable. **Proof.** Suppose $w_1x_1+w_2x_2+b>0$ exactly on the XOR-positive points.
From $(0,0)$: $b\le 0$. From $(0,1)$ and $(1,0)$: $w_2+b>0$ and $w_1+b>0$. Adding these two gives
$w_1+w_2+2b>0$, so $w_1+w_2+b > -b \ge 0$, i.e. $(1,1)$ would be classified positive. Contradiction. $\square$

Drug repositioning is full of XOR-like structure ("drugs of chemical class X treat disease family Y
*unless* they also hit target Z"), so we need something more powerful than one neuron.

### 2.3 Layers and the multi-layer perceptron (MLP)

**Intuition.** Put several neurons side by side (a **layer**), feed the outputs of one layer into the
next, and you get a **multi-layer perceptron**. Each layer re-describes the data in new coordinates;
the last layer only needs to draw a straight line in the final coordinates.

**Definition (single example).** An $L$-layer MLP with widths $n_0, n_1, \dots, n_L$ has weight
matrices $W^{(\ell)}\in\mathbb{R}^{n_\ell\times n_{\ell-1}}$ and bias vectors $b^{(\ell)}\in\mathbb{R}^{n_\ell}$:

$$
h^{(0)} = x,\qquad
z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)},\qquad
h^{(\ell)} = \phi\!\left(z^{(\ell)}\right)\quad(\ell=1,\dots,L-1),\qquad
\hat y = z^{(L)}\ \text{(or } \phi_{\text{out}}(z^{(L)})\text{)}.
$$

Layers $1,\dots,L-1$ are **hidden layers**; their outputs are **hidden representations** or
**embeddings**. In this project the word *embedding* means exactly that: the 64-dimensional hidden
vector $h_i$ that the network computes for drug $i$.

**Definition (batched, PyTorch convention).** We almost never process one example at a time. Stack $B$
examples as the **rows** of a matrix $X\in\mathbb{R}^{B\times n_0}$. Then

$$
Z^{(\ell)} = H^{(\ell-1)}\, W^{(\ell)\top} + \mathbf{1}\, b^{(\ell)\top}, \qquad H^{(\ell)} = \phi(Z^{(\ell)}),
$$

where $\phi$ acts element-wise and adding the bias row to every row is **broadcasting** (Section 9.3).
This is literally what `nn.Linear` computes: `y = x @ W.T + b`, with `W` of shape `(out_features,
in_features)`. Memorise the shape convention: **weights are stored (out, in)**, inputs are
**(batch, in)**, outputs are **(batch, out)**.

In MV-HGAT, the "batch" dimension is the set of all nodes of one type: `X["drug"]` has shape
$(593, F)$ for Fdataset, one row per drug, and `self.inp["drug"]` is an `nn.Linear(F, 64)`.

### 2.4 Why the non-linearity is essential (a proof)

**Claim.** If every $\phi$ is the identity, an MLP of any depth computes an affine function
$x\mapsto Mx+c$, so depth adds no expressive power.

**Proof** by induction on depth. One layer: $W^{(1)}x+b^{(1)}$ is affine. If $h^{(\ell-1)} = M x + c$, then
$h^{(\ell)} = W^{(\ell)}(Mx+c)+b^{(\ell)} = (W^{(\ell)}M)\,x + (W^{(\ell)}c+b^{(\ell)})$, again affine. $\square$

So a 10-layer linear network is just a (possibly rank-restricted) single linear layer. The activation
function is what lets the network bend space.

### 2.5 Worked example: XOR with two ReLU neurons

Let $\text{ReLU}(z)=\max(0,z)$. Take the hidden layer

$$
h_1 = \text{ReLU}(x_1 + x_2),\qquad h_2 = \text{ReLU}(x_1 + x_2 - 1),
$$

and the output $\hat y = h_1 - 2h_2$. Check all four inputs:

| $(x_1,x_2)$ | $x_1+x_2$ | $h_1$ | $h_2$ | $\hat y = h_1 - 2h_2$ |
|---|---|---|---|---|
| (0,0) | 0 | 0 | 0 | 0 |
| (0,1) | 1 | 1 | 0 | 1 |
| (1,0) | 1 | 1 | 0 | 1 |
| (1,1) | 2 | 2 | 1 | 0 |

Exactly XOR. In matrix form: $W^{(1)}=\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $b^{(1)}=\begin{pmatrix}0\\-1\end{pmatrix}$,
$W^{(2)}=\begin{pmatrix}1&-2\end{pmatrix}$, $b^{(2)}=0$. The hidden layer mapped the four corners onto the
points $(0,0),(1,0),(1,0),(2,1)$ in $(h_1,h_2)$-space, where a straight line *does* separate them.

```python
import torch

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
W1 = torch.tensor([[1., 1.], [1., 1.]]); b1 = torch.tensor([0., -1.])
W2 = torch.tensor([[1., -2.]]);          b2 = torch.tensor([0.])

H = torch.relu(X @ W1.T + b1)     # (4, 2): batched form, rows = examples
y_hat = H @ W2.T + b2             # (4, 1)
print("hidden:\n", H)
print("output:", y_hat.squeeze(1).tolist())
```

**Output:**
```text
hidden:
 tensor([[0., 0.],
        [1., 0.],
        [1., 0.],
        [2., 1.]])
output: [0.0, 1.0, 1.0, 0.0]
```

---

## 3. What can an MLP represent? The universal approximation theorem

### 3.1 Intuition: building functions out of ReLU "hinges"

$\text{ReLU}(x-t)$ is a hinge: flat (zero) left of $t$, a ramp of slope 1 to the right. Differences of
hinges make useful building blocks:

* $\text{ReLU}(x-t_1)-\text{ReLU}(x-t_2)$ for $t_1<t_2$ is a **ramp that levels off**: 0 before $t_1$,
  rising, then constant $t_2-t_1$ after $t_2$.
* Combining three hinges makes a **triangle "bump"**:
  $\text{bump}(x)=\text{ReLU}(x-a)-2\,\text{ReLU}(x-m)+\text{ReLU}(x-c)$ with $m=(a+c)/2$ is zero outside
  $[a,c]$ and peaks at height $m-a$ at $x=m$.

A weighted sum of many narrow bumps can trace out any continuous curve you like, like a histogram with
tiny bins. Each hinge is one hidden neuron, and the weighted sum is the output layer. That is the
whole idea of the theorem.

```
 ReLU(x-a) - 2 ReLU(x-m) + ReLU(x-c)

        /\
       /  \
 _____/    \______
      a  m  c
```

### 3.2 The theorem (informal statement)

> **Universal approximation theorem.** Let $\phi$ be any continuous activation that is not a
> polynomial (sigmoid, tanh, ReLU, ELU... all qualify). For every continuous function $f$ on a closed,
> bounded set $K\subset\mathbb{R}^n$ and every tolerance $\varepsilon>0$ there is a one-hidden-layer
> network $g(x)=\sum_{k=1}^{N} c_k\,\phi(w_k^\top x+b_k)$ with $\sup_{x\in K}|f(x)-g(x)|<\varepsilon$.

Historical versions: Cybenko (1989) for sigmoids, Hornik, Stinchcombe & White (1989) and Hornik (1991)
for general "squashing" functions, and Leshno, Lin, Pinkus & Schocken (1993) for the clean statement
"any non-polynomial continuous activation works".

**What it does not say** (this is the part people get wrong):

1. **How many neurons.** $N$ can be astronomically large — for some functions it grows exponentially
   with the input dimension.
2. **How to find the weights.** It is an existence theorem. Gradient descent may not find them.
3. **Generalisation.** Fitting the training points perfectly says nothing about unseen drug–disease
   pairs. In this project, generalisation (AUPR on held-out links) is the only thing that matters.
4. **That depth is useless.** Deep networks can represent some functions with exponentially fewer
   neurons than shallow ones (Telgarsky 2016 gives explicit examples), which is one reason we stack
   layers in practice.

### 3.3 Demonstration: approximating $\sin$ with ReLU hinges

Below, the hidden layer is fixed (hinges at evenly spaced knots) and only the output weights are fitted
by least squares, so you can see approximation quality improve as $N$ grows, without any training
noise.

```python
import numpy as np

x = np.linspace(0, 2 * np.pi, 2001)
f = np.sin(x)
for N in (2, 4, 8, 16, 32, 64):
    knots = np.linspace(0, 2 * np.pi, N, endpoint=False)
    H = np.maximum(0.0, x[:, None] - knots[None, :])      # (2001, N) hidden ReLU layer
    H = np.hstack([H, np.ones((len(x), 1))])               # output bias
    c, *_ = np.linalg.lstsq(H, f, rcond=None)              # output weights
    err = np.abs(H @ c - f).max()
    print(f"N = {N:3d} hidden ReLUs -> max |error| = {err:.5f}")
```

**Output:**
```text
N =   2 hidden ReLUs -> max |error| = 0.95350
N =   4 hidden ReLUs -> max |error| = 0.19134
N =   8 hidden ReLUs -> max |error| = 0.05310
N =  16 hidden ReLUs -> max |error| = 0.01292
N =  32 hidden ReLUs -> max |error| = 0.00321
N =  64 hidden ReLUs -> max |error| = 0.00080
```

The error falls roughly by a factor of 4 every time $N$ doubles, which is the expected rate for
piecewise-linear interpolation of a smooth function (error $\propto$ spacing$^2$).

---

## 4. Activation functions and their gradients

Backpropagation (Section 6) multiplies together the derivatives of every activation along a path, so
the *derivative* of an activation matters as much as its shape. Two failure modes recur:

* **Vanishing gradients:** if $|\phi'(z)|\ll 1$ for typical $z$, products of many such factors shrink
  to zero and early layers stop learning.
* **Dead units:** if $\phi'(z)=0$ for all inputs a neuron sees, it receives no gradient ever again.

### 4.1 The catalogue

| Name | $\phi(z)$ | $\phi'(z)$ | Range | Used in this project |
|---|---|---|---|---|
| Sigmoid $\sigma$ | $\dfrac{1}{1+e^{-z}}$ | $\sigma(z)\,(1-\sigma(z))$ | $(0,1)$ | final probabilities; degree gate `gnn_gate` |
| tanh | $\dfrac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$ | $1-\tanh^2(z)$ | $(-1,1)$ | `ViewAttention` score $q^\top\tanh(Pm+b)$ |
| ReLU | $\max(0,z)$ | $1$ if $z>0$, $0$ if $z<0$ | $[0,\infty)$ | baseline `_GCN` (NIMCGCN, LAGCN) |
| LeakyReLU$_\alpha$ | $z$ if $z>0$, $\alpha z$ otherwise | $1$ or $\alpha$ | $\mathbb{R}$ | GAT scores, $\alpha=0.2$ (`DenseGAT`) |
| ELU$_\alpha$ | $z$ if $z>0$, $\alpha(e^{z}-1)$ otherwise | $1$ or $\alpha e^{z}$ | $(-\alpha,\infty)$ | after every layer in `MVHGAT.encode`, `HeteroLayer` ($\alpha=1$) |
| Softplus | $\log(1+e^{z})$ | $\sigma(z)$ | $(0,\infty)$ | non-negative view weights `view_weights` |

### 4.2 Derivations

**Sigmoid.** Write $\sigma(z)=(1+e^{-z})^{-1}$. Then
$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)\,\bigl(1-\sigma(z)\bigr).
$$
The maximum is at $z=0$ where $\sigma=\tfrac12$, so $\sigma'(z)\le \tfrac14$. Through 10 sigmoid layers the
gradient is multiplied by at most $(1/4)^{10}\approx 10^{-6}$ — the classic vanishing-gradient problem
that kept deep networks untrainable for years. Sigmoid is fine as the *last* step that turns a logit
into a probability, which is how the project uses it.

**tanh.** $\tanh(z) = 2\sigma(2z)-1$ (check: multiply numerator and denominator of $\tanh$ by $e^{-z}$).
Differentiating, $\tanh'(z) = 4\sigma'(2z) = 1-\tanh^2(z)$, with maximum 1 at $z=0$. tanh is
zero-centred, which makes it nicer than sigmoid as a hidden activation, but it still saturates for
$|z|\gtrsim 3$.

**ReLU.** For $z\ne 0$ the derivative is 0 or 1; at $z=0$ it is undefined and frameworks simply use 0.
Because the derivative is exactly 1 on the active side, gradients pass through active ReLUs
undiminished — the main reason ReLU made deep learning practical (Glorot, Bordes & Bengio 2011;
Krizhevsky et al. 2012). The downside is the **dying ReLU**: a unit whose pre-activation is negative for
every input outputs 0 and has gradient 0, so it can never recover.

**LeakyReLU.** Gives negative inputs a small slope $\alpha$ (0.01 by default in PyTorch; GAT uses 0.2),
so the gradient is never exactly zero. In `DenseGAT`, LeakyReLU is applied to the raw attention score
$e_{ij}$ — exactly as in the original GAT paper — so that negative scores still differ from each other
before the softmax.

**ELU** (Clevert, Unterthiner & Hochreiter 2015). For $z\le 0$, $\phi(z)=\alpha(e^z-1)$ and
$\phi'(z)=\alpha e^z = \phi(z)+\alpha$. Properties: smooth at 0 when $\alpha=1$ (both one-sided
derivatives equal 1), negative outputs push the mean activation towards zero (which helps the next layer),
and it saturates gently at $-\alpha$ for very negative inputs (a "soft floor" that is robust to noise).
MV-HGAT uses ELU after every layer, again following the GAT paper.

**Softplus.** $\log(1+e^z)$ is a smooth ReLU: $\approx 0$ for $z\ll 0$, $\approx z$ for $z\gg 0$, and its
derivative is precisely $\sigma(z)$. The project uses it as a *parameterisation trick*: `prop_w` can be
any real number, but `F.softplus(self.prop_w) * self.prop_scale` is always positive, so each view's
weight in the propagation head is guaranteed non-negative.

### 4.3 Checking the derivatives with autograd

```python
import torch
import torch.nn.functional as F

z = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
acts = {
    "sigmoid":   (torch.sigmoid,                   lambda z: torch.sigmoid(z) * (1 - torch.sigmoid(z))),
    "tanh":      (torch.tanh,                      lambda z: 1 - torch.tanh(z) ** 2),
    "relu":      (F.relu,                          lambda z: (z > 0).float()),
    "leaky0.2":  (lambda z: F.leaky_relu(z, 0.2),  lambda z: torch.where(z > 0, 1.0, 0.2)),
    "elu":       (F.elu,                           lambda z: torch.where(z > 0, 1.0, torch.exp(z))),
    "softplus":  (F.softplus,                      torch.sigmoid),
}
for name, (f, df) in acts.items():
    (g,) = torch.autograd.grad(f(z).sum(), z)       # d/dz_k of sum_k f(z_k) = f'(z_k)
    ok = torch.allclose(g, df(z.detach()))
    print(f"{name:9s} f(z) = {[round(v, 4) for v in f(z).tolist()]}")
    print(f"{'':9s} f'(z)= {[round(v, 4) for v in g.tolist()]}  matches formula: {ok}")
```

**Output:**
```text
sigmoid   f(z) = [0.0474, 0.3775, 0.6225, 0.9526]
          f'(z)= [0.0452, 0.235, 0.235, 0.0452]  matches formula: True
tanh      f(z) = [-0.9951, -0.4621, 0.4621, 0.9951]
          f'(z)= [0.0099, 0.7864, 0.7864, 0.0099]  matches formula: True
relu      f(z) = [0.0, 0.0, 0.5, 3.0]
          f'(z)= [0.0, 0.0, 1.0, 1.0]  matches formula: True
leaky0.2  f(z) = [-0.6, -0.1, 0.5, 3.0]
          f'(z)= [0.2, 0.2, 1.0, 1.0]  matches formula: True
elu       f(z) = [-0.9502, -0.3935, 0.5, 3.0]
          f'(z)= [0.0498, 0.6065, 1.0, 1.0]  matches formula: True
softplus  f(z) = [0.0486, 0.4741, 0.9741, 3.0486]
          f'(z)= [0.0474, 0.3775, 0.6225, 0.9526]  matches formula: True
```

Notice: sigmoid's derivative at $\pm 3$ is already only 0.045; ReLU's is exactly 0 for negative inputs;
LeakyReLU keeps 0.2; ELU keeps $e^{-3}\approx 0.05$ and $e^{-0.5}\approx 0.61$.

### 4.4 Choosing an activation — practical rules

* Hidden layers of an MLP/GNN: ReLU is the default; ELU or LeakyReLU when you see dead units or want
  zero-centred outputs; GELU/SiLU in transformers.
* Output layer: none (identity) for logits fed to a `...WithLogits` loss; sigmoid only when you need a
  probability for reporting; softmax for a distribution over classes or neighbours.
* Positivity constraints on parameters: softplus or `exp`.
* Gates in $(0,1)$: sigmoid (as in `MVHGAT.gnn_gate`).

---

## 5. Softmax, masked softmax and the log-sum-exp trick

### 5.1 Definition and properties

Given a vector of real **scores** $z\in\mathbb{R}^K$, the **softmax** returns a probability vector:

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}.
$$

Properties (each one-line provable):

1. **Positive and sums to 1.** Every $e^{z_i}>0$ and the denominator normalises.
2. **Order-preserving.** $z_i>z_j \Rightarrow s_i>s_j$.
3. **Shift-invariant.** $\operatorname{softmax}(z+c\mathbf{1})=\operatorname{softmax}(z)$ because
   $e^{z_i+c}/\sum_j e^{z_j+c} = e^c e^{z_i}/(e^c\sum_j e^{z_j})$. Only *differences* of scores matter.
4. **Temperature.** $\operatorname{softmax}(z/T)$ approaches one-hot on the arg-max as $T\to 0$ and the
   uniform distribution as $T\to\infty$. "Soft" max: a differentiable version of choosing the largest.
5. **Two classes reduce to sigmoid.** $\operatorname{softmax}(z_1,z_2)_1 = 1/(1+e^{-(z_1-z_2)}) = \sigma(z_1-z_2)$.

**Worked example.** $z=(2,\,1,\,0.1)$: $e^{z}=(7.389,\ 2.718,\ 1.105)$, sum $=11.212$, so
$\operatorname{softmax}(z)=(0.659,\ 0.242,\ 0.099)$.

**Where it appears in the project.** Attention is "score → softmax → weighted sum":
in `DenseGAT` the softmax runs over the *neighbours* of each node (`dim=1`, the source axis), producing
weights $\alpha_{ij}$ that sum to 1 over $j$; in `ViewAttention` it runs over *relations* (`dim=0`),
producing the view weights $\beta_i^r$ that sum to 1 over $r$.

### 5.2 The softmax Jacobian

Let $s=\operatorname{softmax}(z)$. For $i=j$:
$$
\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i}\sum_k e^{z_k} - e^{z_i}e^{z_i}}{(\sum_k e^{z_k})^2} = s_i - s_i^2 .
$$
For $i\neq j$:
$$
\frac{\partial s_i}{\partial z_j} = \frac{0 - e^{z_i}e^{z_j}}{(\sum_k e^{z_k})^2} = -s_i s_j .
$$
Together: $\dfrac{\partial s_i}{\partial z_j} = s_i(\delta_{ij}-s_j)$, i.e. $J = \operatorname{diag}(s) - s s^\top$.

During backprop, if the upstream gradient is $g=\partial L/\partial s$, then
$$
\frac{\partial L}{\partial z} = J^\top g = s\odot\bigl(g - (s^\top g)\,\mathbf 1\bigr),
$$
which is how autograd implements it (no $K\times K$ matrix is ever formed).

### 5.3 Softmax + cross-entropy: the gradient is $p-y$

With a one-hot target $y$ and $p=\operatorname{softmax}(z)$, cross-entropy is $L=-\sum_i y_i\log p_i$.
Using $\partial \log p_i/\partial z_j = \delta_{ij}-p_j$:
$$
\frac{\partial L}{\partial z_j} = -\sum_i y_i(\delta_{ij}-p_j) = -y_j + p_j\sum_i y_i = p_j - y_j.
$$
The same result holds for sigmoid + binary cross-entropy: $\partial L/\partial z = \sigma(z)-y$
(Section 8.2). "Prediction minus target" is the error signal that starts every backward pass in this
project.

### 5.4 Masked softmax

Often only *some* entries should compete. In a graph, node $i$ must attend only over its actual
neighbours; in `ViewAttention`, a node must ignore relations in which it has no neighbours. The trick is
to set the scores of forbidden entries to $-\infty$ **before** the softmax:

$$
\tilde z_j = \begin{cases} z_j & \text{if } \text{mask}_j\\ -\infty & \text{otherwise}\end{cases},
\qquad
\operatorname{softmax}(\tilde z)_i = \frac{\text{mask}_i\, e^{z_i}}{\sum_{j:\ \text{mask}_j} e^{z_j}},
$$

because $e^{-\infty}=0$. The forbidden entries get weight exactly 0 and the allowed ones are
renormalised among themselves. **Worked example:** $z=(2.0,\ 1.0,\ 0.1,\ 3.0)$, mask $=(T,F,T,F)$:
$\tilde z=(2.0,-\infty,0.1,-\infty)$, weights $=(e^{2}, 0, e^{0.1}, 0)/(e^2+e^{0.1}) = (0.870,\ 0,\ 0.130,\ 0)$.
Note that the entry with the *largest* raw score (3.0) gets zero weight because it is masked.

**Why not just multiply the softmax output by the mask?** Because then the surviving weights would not
sum to 1 (they would sum to $(e^2+e^{0.1})/(e^2+e^1+e^{0.1}+e^3)$), and the masked scores would still have
influenced the result through the denominator. Masking *before* the softmax is the correct operation.

**The empty-row case.** If *every* entry is masked, the softmax computes $0/0$ = NaN for the whole
row. In this project that really happens. The similarity-view graphs always contain self-loops
(`knn_mask` sets the diagonal to `True`), so their rows are never empty — even a biologic drug with no
SMILES has itself as a neighbour in `chem_ecfp`. But the association relations are different: in an
epoch where a disease is chosen for "cold-start practice", all its links are hidden, so its row of the
`assoc>disease` mask is entirely `False`; and a disease whose gene profile is empty has no
`gene_bridge` neighbours. Without care these rows would produce NaNs that spread through every later
computation (NaN times anything is NaN). That is why `DenseGAT.forward` follows the softmax with

```py
att = torch.nan_to_num(att, nan=0.0)                      # nodes w/o neighbours
```

and `ViewAttention.forward` does the same for nodes with no valid relation. The resulting message for
such a node is the zero vector, and `DenseGAT` also returns `mask.any(1)` ("does this node have at least
one neighbour?") so that `ViewAttention` can mask out that relation entirely for that node.

Does the NaN poison the *gradient*? No — every score in an empty row was replaced by a constant by
`masked_fill`, and the backward pass of `masked_fill` sends zero gradient to replaced positions. The
code below checks this.

```python
import torch
import torch.nn.functional as F

z = torch.tensor([[2.0, 1.0, 0.1, 3.0],      # row 0: a node with two neighbours
                  [0.5, 0.2, 0.9, 0.4]],     # row 1: a node with NO neighbours
                 requires_grad=True)
mask = torch.tensor([[True, False, True, False],
                     [False, False, False, False]])

att = torch.softmax(z.masked_fill(~mask, float("-inf")), dim=1)
print("raw softmax:\n", att.detach())
att = torch.nan_to_num(att, nan=0.0)
print("after nan_to_num:\n", att.detach())

values = torch.tensor([10.0, 20.0, 30.0, 40.0])   # one scalar "message" per neighbour
out = att @ values                                 # weighted sum per node
out.sum().backward()
print("messages:", out.detach().tolist())
print("grad wrt scores:\n", z.grad)
```

**Output:**
```text
raw softmax:
 tensor([[0.8699, 0.0000, 0.1301, 0.0000],
        [   nan,    nan,    nan,    nan]])
after nan_to_num:
 tensor([[0.8699, 0.0000, 0.1301, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000]])
messages: [12.60216999053955, 0.0]
grad wrt scores:
 tensor([[-2.2636,  0.0000,  2.2636,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])
```

Row 0's message is $0.870\cdot 10+0.130\cdot 30=12.6$; row 1 gets 0 with a perfectly finite (zero)
gradient. Masked entries in row 0 receive zero gradient too: they had no influence, so changing them
cannot change the loss.

### 5.5 Overflow, underflow and the log-sum-exp trick

In 32-bit floating point (`float32`, the default in PyTorch and in this project), the largest finite
number is about $3.4\times10^{38}$, so $e^{z}$ overflows to `inf` once $z>88.7$; and $e^{z}$ underflows
to 0 for $z<-103$ or so. A naive softmax of $(1000,1001,1002)$ computes `inf/inf = nan`.

Shift-invariance rescues us: subtract the maximum first. With $m=\max_j z_j$,
$$
\operatorname{softmax}(z)_i = \frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
$$
where now every exponent is $\le 0$ and at least one term in the denominator equals 1, so neither
overflow nor division by zero can happen. The same idea gives a stable **log-sum-exp**:
$$
\operatorname{LSE}(z) = \log\sum_j e^{z_j} = m + \log\sum_j e^{z_j-m}.
$$
Example: $\operatorname{LSE}(1000,1001,1002) = 1002 + \log(e^{-2}+e^{-1}+1) = 1002 + \log 1.5032 = 1002.4076$.
`torch.softmax`, `torch.logsumexp` and `F.log_softmax` all do this internally. Never write
`torch.exp(z) / torch.exp(z).sum()` yourself.

```python
import torch

z = torch.tensor([1000.0, 1001.0, 1002.0])
naive = torch.exp(z) / torch.exp(z).sum()
stable = torch.exp(z - z.max()) / torch.exp(z - z.max()).sum()
print("naive  :", naive.tolist())
print("stable :", [round(v, 4) for v in stable.tolist()])
print("torch  :", [round(v, 4) for v in torch.softmax(z, 0).tolist()])
print("naive LSE :", torch.log(torch.exp(z).sum()).item())
print("stable LSE:", round(torch.logsumexp(z, 0).item(), 4))
```

**Output:**
```text
naive  : [nan, nan, nan]
stable : [0.09, 0.2447, 0.6652]
torch  : [0.09, 0.2447, 0.6652]
naive LSE : inf
stable LSE: 1002.4076
```

---

## 6. Computational graphs and backpropagation

Training means adjusting every parameter $\theta$ a little in the direction $-\partial L/\partial\theta$.
MV-HGAT has about 436,000 parameters on Fdataset (Section 11.4 counts them). We need all those
partial derivatives, every epoch, cheaply. **Backpropagation** is the algorithm that delivers them, and it
is nothing more than the chain rule applied in a clever order.

### 6.1 The chain rule, three ways

**Scalar.** If $y=f(u)$ and $u=g(x)$ then $\dfrac{dy}{dx} = \dfrac{dy}{du}\,\dfrac{du}{dx}$.

**Multivariate (sum over paths).** If $L$ depends on $x$ through several intermediate variables
$u_1,\dots,u_m$, then
$$
\frac{\partial L}{\partial x} = \sum_{k=1}^{m} \frac{\partial L}{\partial u_k}\,\frac{\partial u_k}{\partial x}.
$$
Each term is one *path* from $x$ to $L$. When a value is used in two places (**fan-out**), the gradients
coming back from both uses are **added**.

**Vector (Jacobians).** If $u=g(x)$ with $x\in\mathbb{R}^n$, $u\in\mathbb{R}^m$ and $L$ is a scalar,
then $\nabla_x L = J_g(x)^\top\,\nabla_u L$, where $J_g$ is the $m\times n$ Jacobian $\partial u_k/\partial x_j$.
Backprop never builds $J_g$ explicitly; each operation knows how to compute the product
$J^\top v$ ("vector–Jacobian product", VJP) directly.

### 6.2 Computational graphs

A **computational graph** is a directed acyclic graph (DAG) whose nodes are elementary operations
(add, multiply, matmul, exp, ReLU, ...) and whose edges carry the intermediate values. Every program that
computes a loss from parameters defines such a graph.

Example: $f(x,y,z) = (x+y)\,z$ at $x=3,\ y=-1,\ z=4$.

```
 forward  (values)                      backward (gradients df/d.)

  x = 3 ─┐                               df/dx = 4  ◄─┐
         (+)── q = 2 ─┐                               (+)◄── df/dq = z = 4 ─┐
  y =-1 ─┘            (×)── f = 8         df/dy = 4  ◄─┘                    (×)◄── df/df = 1
  z = 4 ──────────────┘                   df/dz = q = 2 ◄───────────────────┘
```

* Forward: $q=x+y=2$, $f=qz=8$.
* Backward starts with $\partial f/\partial f=1$.
* Multiply node: $\partial f/\partial q = z = 4$ and $\partial f/\partial z = q = 2$ — a multiply gate
  **swaps** its inputs.
* Add node: $\partial q/\partial x = \partial q/\partial y = 1$, so it **distributes** the incoming
  gradient unchanged: $\partial f/\partial x = \partial f/\partial y = 4$.

Useful local rules to remember:

| Operation | Forward | Backward (given upstream $g$) |
|---|---|---|
| add $u=a+b$ | | $g_a=g,\ g_b=g$ (distribute) |
| multiply $u=ab$ | | $g_a=g\,b,\ g_b=g\,a$ (swap) |
| max $u=\max(a,b)$ | | all of $g$ to the larger input, 0 to the other (route) |
| ReLU | $\max(0,a)$ | $g\cdot\mathbb 1[a>0]$ (gate) |
| copy / fan-out | $a$ used twice | sum of both incoming gradients |
| matmul $U=AB$ | | $g_A = G B^\top,\ g_B = A^\top G$ |

### 6.3 The backpropagation algorithm

1. **Forward pass.** Evaluate the graph from inputs to the loss, in topological order, and *store*
   every intermediate value that some local derivative will need (e.g. ReLU needs to remember which
   inputs were positive; sigmoid needs its output).
2. **Backward pass.** Set $\bar L = \partial L/\partial L = 1$. Visit nodes in *reverse* topological
   order. For each node $u = f(a_1,\dots,a_k)$ with accumulated upstream gradient $\bar u$, add
   $\bar u\cdot\partial u/\partial a_i$ (or the VJP for tensors) into $\bar a_i$.
3. When the backward pass finishes, $\bar\theta = \partial L/\partial\theta$ for every parameter $\theta$.

**Why "reverse" mode?** The cost of one backward pass is a small constant (typically 2–3) times the
cost of the forward pass, *regardless of the number of parameters*. The alternative, forward-mode
differentiation, propagates $\partial(\cdot)/\partial\theta_k$ forward for one parameter at a time and would
need one pass per parameter — about 436,000 passes per step for MV-HGAT. When there are many inputs and one scalar
output (a loss), reverse mode wins by a factor of the number of parameters. The price is memory: all
forward intermediates must be kept until the backward pass uses them. That is precisely the memory that
`torch.no_grad()` saves at prediction time.

### 6.4 Worked example: backpropagation through a two-layer network by hand

**Network.** Input $x\in\mathbb{R}^2$, hidden layer of 2 ReLU units, one output logit, sigmoid + BCE.

$$
z^{(1)} = W_1 x + b_1,\quad h=\text{ReLU}(z^{(1)}),\quad z^{(2)} = W_2 h + b_2,\quad p=\sigma(z^{(2)}),\quad L = -\bigl[y\log p + (1-y)\log(1-p)\bigr].
$$

**Numbers.**
$$
x=\begin{pmatrix}1\\2\end{pmatrix},\
W_1=\begin{pmatrix}0.5&0.25\\-1.0&0.25\end{pmatrix},\
b_1=\begin{pmatrix}0\\0.25\end{pmatrix},\
W_2=\begin{pmatrix}1.5&-2.0\end{pmatrix},\
b_2=-0.5,\ y=1.
$$
Think of $x$ as two input features of a drug–disease pair and $y=1$ as "known indication".

**Forward pass.**

| Quantity | Computation | Value |
|---|---|---|
| $z^{(1)}_1$ | $0.5\cdot1+0.25\cdot2+0$ | $1.0$ |
| $z^{(1)}_2$ | $-1.0\cdot1+0.25\cdot2+0.25$ | $-0.25$ |
| $h$ | $(\max(0,1.0),\ \max(0,-0.25))$ | $(1.0,\ 0)$ |
| $z^{(2)}$ | $1.5\cdot1.0 + (-2.0)\cdot 0 - 0.5$ | $1.0$ |
| $p$ | $\sigma(1.0)=1/(1+e^{-1})$ | $0.731059$ |
| $L$ | $-\log 0.731059$ | $0.313262$ |

**Backward pass.** (Each line uses only the line above it and a stored forward value.)

| Gradient | Rule | Value |
|---|---|---|
| $\partial L/\partial z^{(2)}$ | $p-y$ (sigmoid + BCE, Section 8.2) | $0.731059-1=-0.268941$ |
| $\partial L/\partial W_2$ | $\dfrac{\partial L}{\partial z^{(2)}}\,h^\top$ | $(-0.268941,\ 0)$ |
| $\partial L/\partial b_2$ | $\dfrac{\partial L}{\partial z^{(2)}}$ | $-0.268941$ |
| $\partial L/\partial h$ | $W_2^\top\,\dfrac{\partial L}{\partial z^{(2)}}$ | $(1.5\cdot -0.268941,\ -2.0\cdot -0.268941) = (-0.403412,\ 0.537883)$ |
| $\partial L/\partial z^{(1)}$ | $\dfrac{\partial L}{\partial h}\odot\mathbb 1[z^{(1)}>0]$ | $(-0.403412,\ 0)$ |
| $\partial L/\partial W_1$ | $\dfrac{\partial L}{\partial z^{(1)}}\,x^\top$ | $\begin{pmatrix}-0.403412&-0.806824\\0&0\end{pmatrix}$ |
| $\partial L/\partial b_1$ | $\dfrac{\partial L}{\partial z^{(1)}}$ | $(-0.403412,\ 0)$ |
| $\partial L/\partial x$ | $W_1^\top\,\dfrac{\partial L}{\partial z^{(1)}}$ | $(0.5\cdot-0.403412,\ 0.25\cdot -0.403412)=(-0.201706,\ -0.100853)$ |

Observations worth internalising:

* **The dead hidden unit blocks gradient.** $h_2$ received gradient $0.5379$, but because
  $z^{(1)}_2<0$ the ReLU gate multiplied it by 0, so row 2 of $W_1$ and $b_{1,2}$ get **no update** from
  this example. Also $\partial L/\partial W_{2,2}=0$ because $h_2=0$: a weight whose input is zero can't
  have affected the output.
* **Negative gradients mean "increase this parameter".** Gradient descent subtracts the gradient, so
  every non-zero parameter above will *grow*, pushing $z^{(2)}$ up and $p$ towards $y=1$.

**One gradient-descent step** with learning rate $\eta=0.1$:
$W_1\leftarrow\begin{pmatrix}0.540341&0.330682\\-1.0&0.25\end{pmatrix}$,
$b_1\leftarrow(0.040341,\ 0.25)$, $W_2\leftarrow(1.526894,\ -2.0)$, $b_2\leftarrow-0.473106$.
New forward pass: $z^{(1)}_1 = 0.540341+0.661365+0.040341=1.242047$, $z^{(1)}_2=-0.25$ (unchanged),
$z^{(2)} = 1.526894\cdot1.242047-0.473106 = 1.423369$, $p=\sigma(1.423369)=0.805866$, $L=0.215838$. The
loss fell from 0.3133 to 0.2158.

Now let PyTorch confirm every number, and confirm PyTorch with **finite differences**:
$\dfrac{\partial L}{\partial\theta}\approx\dfrac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$
(the central difference has error $O(\epsilon^2)$; use `float64` so rounding does not dominate).

```python
import torch

def forward(W1, b1, W2, b2, x, y):
    z1 = W1 @ x + b1
    h = torch.relu(z1)
    z2 = W2 @ h + b2
    p = torch.sigmoid(z2)
    L = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    return z1, h, z2, p, L.sum()

dt = torch.float64
x = torch.tensor([1.0, 2.0], dtype=dt); y = torch.tensor(1.0, dtype=dt)
W1 = torch.tensor([[0.5, 0.25], [-1.0, 0.25]], dtype=dt, requires_grad=True)
b1 = torch.tensor([0.0, 0.25], dtype=dt, requires_grad=True)
W2 = torch.tensor([[1.5, -2.0]], dtype=dt, requires_grad=True)
b2 = torch.tensor([-0.5], dtype=dt, requires_grad=True)
params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

z1, h, z2, p, L = forward(W1, b1, W2, b2, x, y)
print(f"z1={z1.tolist()} h={h.tolist()} z2={z2.item():.6f} p={p.item():.6f} L={L.item():.6f}")
L.backward()
for name, P in params.items():
    print(f"dL/d{name} = {[round(v, 6) for v in P.grad.flatten().tolist()]}")

# finite-difference check of every parameter entry
eps, worst = 1e-6, 0.0
with torch.no_grad():
    for name, P in params.items():
        flat = P.view(-1)
        for k in range(flat.numel()):
            old = flat[k].item()
            flat[k] = old + eps; Lp = forward(W1, b1, W2, b2, x, y)[-1].item()
            flat[k] = old - eps; Lm = forward(W1, b1, W2, b2, x, y)[-1].item()
            flat[k] = old
            worst = max(worst, abs((Lp - Lm) / (2 * eps) - P.grad.view(-1)[k].item()))
print(f"largest |autograd - finite difference| = {worst:.2e}")

# one step of gradient descent
with torch.no_grad():
    for P in params.values():
        P -= 0.1 * P.grad
_, _, z2, p, L = forward(W1, b1, W2, b2, x, y)
print(f"after one step: z2={z2.item():.6f} p={p.item():.6f} L={L.item():.6f}")
```

**Output:**
```text
z1=[1.0, -0.25] h=[1.0, 0.0] z2=1.000000 p=0.731059 L=0.313262
dL/dW1 = [-0.403412, -0.806824, 0.0, 0.0]
dL/db1 = [-0.403412, 0.0]
dL/dW2 = [-0.268941, -0.0]
dL/db2 = [-0.268941]
largest |autograd - finite difference| = 9.80e-11
after one step: z2=1.423369 p=0.805866 L=0.215838
```

### 6.5 Matrix-form gradients of a linear layer

Real layers are batched, so we need the gradient formulas in matrix form. Let
$Y = XW^\top + \mathbf 1 b^\top$ with $X\in\mathbb{R}^{B\times n}$, $W\in\mathbb{R}^{m\times n}$,
$b\in\mathbb{R}^m$, $Y\in\mathbb{R}^{B\times m}$, and let $G=\partial L/\partial Y\in\mathbb{R}^{B\times m}$ be the
upstream gradient. In index form $Y_{po} = \sum_k X_{pk}W_{ok} + b_o$, so

$$
\frac{\partial L}{\partial W_{ok}} = \sum_{p}\frac{\partial L}{\partial Y_{po}}\frac{\partial Y_{po}}{\partial W_{ok}} = \sum_p G_{po}X_{pk}
\ \Rightarrow\ \boxed{\frac{\partial L}{\partial W} = G^\top X}\quad (m\times n),
$$
$$
\frac{\partial L}{\partial X_{pk}} = \sum_o G_{po} W_{ok}\ \Rightarrow\ \boxed{\frac{\partial L}{\partial X} = G\,W}\quad (B\times n),
\qquad
\frac{\partial L}{\partial b_o} = \sum_p G_{po}\ \Rightarrow\ \boxed{\frac{\partial L}{\partial b} = G^\top\mathbf 1}.
$$

For an element-wise activation $H=\phi(Z)$: $\partial L/\partial Z = (\partial L/\partial H)\odot\phi'(Z)$.

**The shape rule.** A gradient always has the shape of the thing it is the gradient *of*. If you
forget a formula, write down the shapes and there is usually only one way to multiply the available
matrices to get the right shape: $\partial L/\partial W$ must be $m\times n$, the only candidates are
$G$ ($B\times m$) and $X$ ($B\times n$), so it must be $G^\top X$.

**The bilinear decoder.** MV-HGAT scores all pairs at once with $S = H_r W H_d^\top$
(`logits = Hr @ self.W @ Hd.T`), where $H_r$ is $n_r\times d$, $W$ is $d\times d$, $H_d$ is $n_d\times d$. Applying
the matmul rule twice with $G=\partial L/\partial S$ ($n_r\times n_d$):
$$
\frac{\partial L}{\partial W} = H_r^\top G H_d,\qquad
\frac{\partial L}{\partial H_r} = G H_d W^\top,\qquad
\frac{\partial L}{\partial H_d} = G^\top H_r W .
$$
In the project, $G$ is non-zero only at the sampled positive and negative cells (the loss only reads
those logits), so each epoch only the embeddings of drugs and diseases that appear in sampled pairs
receive a direct decoder gradient — the GNN layers then spread it to their neighbours.

```python
import torch
torch.manual_seed(0)
nr, nd, d = 5, 4, 3
Hr = torch.randn(nr, d, requires_grad=True)
Hd = torch.randn(nd, d, requires_grad=True)
W = torch.randn(d, d, requires_grad=True)
S = Hr @ W @ Hd.T                      # (5, 4) all pair scores
G = torch.randn(nr, nd)                # pretend upstream gradient dL/dS
S.backward(G)                          # vector-Jacobian product with G

print(torch.allclose(W.grad,  Hr.T @ G @ Hd))
print(torch.allclose(Hr.grad, G @ Hd @ W.T))
print(torch.allclose(Hd.grad, G.T @ Hr @ W))
```

**Output:**
```text
True
True
True
```

### 6.6 Backprop from scratch: a NumPy MLP that learns XOR

Writing backprop once by hand is the best way to never be confused by it again. The network is
$2\to 8\to 1$ with tanh hidden units and sigmoid + BCE output, trained by full-batch gradient descent.

```python
import numpy as np

rng = np.random.default_rng(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

n_in, n_h = 2, 8
W1 = rng.normal(0, np.sqrt(1 / n_in), (n_h, n_in)); b1 = np.zeros(n_h)
W2 = rng.normal(0, np.sqrt(1 / n_h), (1, n_h));     b2 = np.zeros(1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for step in range(2001):
    # forward (batched: rows are examples)
    Z1 = X @ W1.T + b1          # (4, 8)
    H = np.tanh(Z1)             # (4, 8)
    Z2 = H @ W2.T + b2          # (4, 1)
    P = sigmoid(Z2)
    loss = -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))
    # backward
    G2 = (P - y) / len(X)       # dL/dZ2, the mean over the batch gives the 1/B
    dW2 = G2.T @ H;  db2 = G2.sum(0)
    GH = G2 @ W2                # dL/dH
    G1 = GH * (1 - H ** 2)      # through tanh
    dW1 = G1.T @ X;  db1 = G1.sum(0)
    # gradient descent
    for P_, dP in ((W1, dW1), (b1, db1), (W2, dW2), (b2, db2)):
        P_ -= 1.0 * dP
    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

print("predictions:", P.ravel().round(3))
```

**Output:**
```text
step    0  loss 0.6784
step  500  loss 0.0027
step 1000  loss 0.0011
step 1500  loss 0.0007
step 2000  loss 0.0005
predictions: [0.    1.    0.999 0.001]
```

Every line of the backward section is one row of the table in Section 6.4, written for a batch.

### 6.7 Gradient checking in practice

When you write a custom operation (or doubt a library), compare analytic and numerical gradients using
the **relative error** $\dfrac{|g_a-g_n|}{\max(|g_a|,|g_n|,10^{-12})}$. In `float64` with $\epsilon\approx10^{-6}$,
expect $<10^{-7}$; values $>10^{-3}$ almost always mean a bug. Caveats: kinks (ReLU exactly at 0) give
spurious errors, and dropout must be disabled (or its random mask fixed) during the check. PyTorch
ships `torch.autograd.gradcheck`, which does exactly this for a function of `float64` inputs.

---

## 7. Making deep networks trainable: initialisation, dropout, normalisation, residuals

### 7.1 Weight initialisation

**Why not zeros?** If every weight of a layer starts equal, every hidden unit computes the same value
and receives the same gradient, so they stay identical forever — the layer behaves like a single unit.
Random initialisation **breaks the symmetry**.

**Why not "any random numbers"?** Because the *scale* matters. Consider $z=\sum_{k=1}^{n_{in}} w_k x_k$
with independent, zero-mean $w_k$ and $x_k$. Then
$$
\operatorname{Var}(z) = \sum_{k=1}^{n_{in}}\operatorname{Var}(w_k x_k) = n_{in}\,\operatorname{Var}(w)\,\operatorname{Var}(x).
$$
If $n_{in}\operatorname{Var}(w)>1$ the signal's variance is multiplied by that factor at every layer and
explodes exponentially with depth; if $<1$ it shrinks to nothing (and so do the gradients, by the same
argument applied to the backward pass, which involves $W^\top$ and therefore $n_{out}$).

**Xavier / Glorot initialisation** (Glorot & Bengio 2010). To keep variance constant forward we want
$n_{in}\operatorname{Var}(w)=1$; backward we want $n_{out}\operatorname{Var}(w)=1$. The compromise is
$$
\operatorname{Var}(w)=\frac{2}{n_{in}+n_{out}}.
$$
A uniform distribution $U(-a,a)$ has variance $a^2/3$, so the uniform version uses
$$
a = \sqrt{\frac{6}{n_{in}+n_{out}}}\qquad\text{(PyTorch: } \texttt{nn.init.xavier\_uniform\_}\text{)}.
$$
The derivation assumes activations that are roughly linear near 0 (tanh, sigmoid's middle). For ReLU,
half the units are zero, which halves the variance, so **He/Kaiming initialisation** (He et al. 2015)
uses $\operatorname{Var}(w)=2/n_{in}$.

**PyTorch defaults.** `nn.Linear(in, out)` initialises its weight from $U(-1/\sqrt{in},\ 1/\sqrt{in})$
(Kaiming-uniform with a particular slope parameter) and its bias similarly. You only need to initialise
by hand when you create raw `nn.Parameter`s, because `torch.empty` contains arbitrary garbage.

**In the project.** `DenseGAT` creates `a_src` and `a_dst` with `torch.empty(heads, self.dh)` and then
calls `nn.init.xavier_uniform_` on them; `MVHGAT` does the same for the decoder matrix `W`. For a 2-D
tensor PyTorch takes $n_{in}=$ `size(1)` and $n_{out}=$ `size(0)`, so

* `a_src`, shape $(4,16)$: $a=\sqrt{6/(16+4)}=0.5477$;
* decoder `W`, shape $(192,192)$ (because jumping knowledge concatenates 3 layers of 64): $a=\sqrt{6/384}=0.125$.

The experiment below pushes a batch through 20 tanh layers of width 256 and prints the standard
deviation of the activations at a few depths under three schemes.

```python
import torch
torch.manual_seed(0)
x0 = torch.randn(512, 256)

def run(init):
    h, stds = x0, []
    for layer in range(20):
        W = torch.empty(256, 256)
        init(W)
        h = torch.tanh(h @ W.T)
        stds.append(h.std().item())
    return stds

schemes = {
    "N(0, 1)      ": lambda W: torch.nn.init.normal_(W, 0, 1.0),
    "N(0, 0.01^2) ": lambda W: torch.nn.init.normal_(W, 0, 0.01),
    "Xavier unif. ": torch.nn.init.xavier_uniform_,
}
for name, init in schemes.items():
    s = run(init)
    print(name, "std at layers 1, 5, 10, 20:", [f"{s[i]:.4f}" for i in (0, 4, 9, 19)])
```

**Output:**
```text
N(0, 1)       std at layers 1, 5, 10, 20: ['0.9750', '0.9746', '0.9737', '0.9746']
N(0, 0.01^2)  std at layers 1, 5, 10, 20: ['0.1562', '0.0001', '0.0000', '0.0000']
Xavier unif.  std at layers 1, 5, 10, 20: ['0.6274', '0.3190', '0.2321', '0.1648']
```

With $\mathcal N(0,1)$ weights every tanh saturates (std near 1, i.e. outputs pinned at $\pm1$, derivatives
near 0). With tiny weights the signal dies geometrically. Xavier keeps a usable, only slowly decaying
scale; the remaining decay comes from tanh shrinking its inputs, which is why PyTorch offers a *gain*
factor (`nn.init.calculate_gain('tanh')` $=5/3$) to compensate.

### 7.2 Dropout

**Definition.** During training, each element of a tensor is independently set to zero with
probability $p$ and the survivors are multiplied by $1/(1-p)$ (**inverted dropout**):
$$
\tilde h_k = \frac{m_k}{1-p}\,h_k,\qquad m_k\sim\text{Bernoulli}(1-p).
$$
The scaling keeps the expected value unchanged, $\mathbb{E}[\tilde h_k]=h_k$, so at evaluation time dropout
can simply be switched off (identity) with no rescaling.

**Why it regularises** (Srivastava et al. 2014). A unit cannot rely on any particular other unit being
present, so the network learns redundant, more robust features; one can also view training with dropout
as training an exponentially large ensemble of thinned networks that share weights, with evaluation
approximating their average.

**In the project** (`dropout = 0.2`):
* `MVHGAT.encode`: dropout on the input embedding `self.drop(F.elu(self.inp[t](X[t])))`;
* `HeteroLayer.forward`: dropout on every layer's output;
* `DenseGAT.forward`: dropout on the **attention coefficients** `self.drop(att)` — this randomly removes
  edges from each node's neighbourhood each epoch (as in the GAT paper), a graph-specific regulariser.

Dropout's behaviour depends on the module's mode: `model.train()` → random masks; `model.eval()` →
identity. Forgetting `model.eval()` before predicting gives noisy, randomly degraded scores.

```python
import torch
torch.manual_seed(0)
drop = torch.nn.Dropout(p=0.5)
x = torch.ones(10)
drop.train()
print("train mode:", drop(x).tolist())
print("mean over 100k draws:", round(drop(torch.ones(100_000)).mean().item(), 3))
drop.eval()
print("eval mode :", drop(x).tolist())
```

**Output:**
```text
train mode: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0]
mean over 100k draws: 0.999
eval mode : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### 7.3 Normalisation layers: BatchNorm vs LayerNorm

Both layers standardise activations to zero mean and unit variance and then apply a learned scale
$\gamma$ and shift $\beta$; they differ in **which axis** the statistics are computed over.

For an activation matrix $H\in\mathbb{R}^{B\times d}$ (rows = examples/nodes, columns = features):

**BatchNorm** (Ioffe & Szegedy 2015) normalises each **feature column** over the batch:
$$
\mu_k = \frac1B\sum_{p}H_{pk},\quad \sigma_k^2=\frac1B\sum_p (H_{pk}-\mu_k)^2,\quad
\text{BN}(H)_{pk} = \gamma_k\frac{H_{pk}-\mu_k}{\sqrt{\sigma_k^2+\epsilon}}+\beta_k .
$$
At evaluation time it cannot use batch statistics (a test "batch" might be one example), so it uses
**running averages** accumulated during training. Consequence: BatchNorm behaves *differently* in
`train()` and `eval()` modes, and each example's output depends on which other examples share its batch.

**LayerNorm** (Ba, Kiros & Hinton 2016) normalises each **row** over its own features:
$$
\mu_p = \frac1d\sum_k H_{pk},\quad \sigma_p^2=\frac1d\sum_k(H_{pk}-\mu_p)^2,\quad
\text{LN}(H)_{pk} = \gamma_k\frac{H_{pk}-\mu_p}{\sqrt{\sigma_p^2+\epsilon}}+\beta_k .
$$
No batch statistics, no running averages, identical behaviour in training and evaluation, and each
node's output depends only on that node.

```
           features →                       features →
        ┌───┬───┬───┬───┐                ┌───┬───┬───┬───┐
 nodes  │ ▓ │   │   │   │         nodes  │ ▓ │ ▓ │ ▓ │ ▓ │  ← LayerNorm: one row at a time
   ↓    │ ▓ │   │   │   │           ↓    │   │   │   │   │
        │ ▓ │   │   │   │                │   │   │   │   │
        └───┴───┴───┴───┘                └───┴───┴───┴───┘
     BatchNorm: one column at a time
```

**Why the project uses LayerNorm.** MV-HGAT processes the whole graph as one "batch" in which rows are
not independent samples: a node's representation already mixes in its neighbours, and node populations
are heterogeneous (a hub drug with 22 known indications next to drugs with one). LayerNorm keeps each
node's scale under control independently of all others, behaves identically during training and the
`no_grad` prediction pass, and has no running statistics to go stale when 20% of links are hidden each
epoch. This is also the standard choice in transformers and many GNNs. In `HeteroLayer` there is one
`nn.LayerNorm(64)` per node type (`self.norm["drug"]`, `self.norm["disease"]`).

```python
import torch
torch.manual_seed(0)
H = torch.randn(4, 6) * 3 + 2                 # 4 nodes, 6 features, mean ~2, std ~3

ln = torch.nn.LayerNorm(6)                    # gamma=1, beta=0 at initialisation
mu = H.mean(1, keepdim=True)
var = H.var(1, unbiased=False, keepdim=True)  # LayerNorm uses the biased variance
manual = (H - mu) / torch.sqrt(var + ln.eps)
print("LayerNorm matches manual formula:", torch.allclose(ln(H), manual, atol=1e-6))
print("row means after LN:", [round(v, 4) + 0.0 for v in ln(H).mean(1).tolist()])

bn = torch.nn.BatchNorm1d(6)
bn.train(); out_train = bn(H)                 # uses this batch's column statistics
bn.eval();  out_eval = bn(H)                  # uses running averages (after ONE update)
print("column means after BN (train):", [round(v, 4) + 0.0 for v in out_train.mean(0).tolist()])
print("BN train vs eval outputs equal?", torch.allclose(out_train, out_eval))
ln.eval()
print("LN train vs eval outputs equal?", torch.allclose(ln(H), manual, atol=1e-6))
```

**Output:**
```text
LayerNorm matches manual formula: True
row means after LN: [0.0, 0.0, 0.0, 0.0]
column means after BN (train): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
BN train vs eval outputs equal? False
LN train vs eval outputs equal? True
```

### 7.4 Residual (skip) connections

**Definition.** Instead of $h' = F(h)$, a residual block computes $h' = h + F(h)$ (He et al. 2016). When
the dimensions differ, the identity is replaced by a learned linear projection, $h'=Ph+F(h)$.

**Why it helps — the gradient highway.** By the chain rule,
$$
\frac{\partial h'}{\partial h} = I + \frac{\partial F}{\partial h}.
$$
Even if $\partial F/\partial h$ is tiny (saturated activations, small weights), the identity term passes the
gradient straight through. Through $L$ blocks the gradient contains a term that is simply the product of
identities, so it cannot vanish purely because of depth. A second view: each block only has to learn a
*correction* $F$ to the identity, which is easier than learning the whole map.

**In the project.** `HeteroLayer.forward` computes
```py
new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
```
— the attention output $z$ plus a **projection skip** `self.skip[t] = nn.Linear(in_dim, out_dim)` of the
layer's input, then LayerNorm, ELU and dropout. In GNNs the skip has an extra benefit: it keeps a node's
*own* information alive while neighbours' information is mixed in, which counteracts **over-smoothing**
(all nodes converging to similar embeddings as layers are stacked; Unit C3). The
**jumping-knowledge** concatenation in `MVHGAT.encode` (`torch.cat(outs["drug"], 1)` of
$h^{(0)},h^{(1)},h^{(2)}$) is a related idea: the decoder sees every layer's representation directly.

```python
import torch
torch.manual_seed(0)
depth, d = 50, 64
Ws = [torch.randn(d, d) * 0.05 for _ in range(depth)]   # deliberately small weights

def first_layer_grad(residual):
    x = torch.randn(32, d, requires_grad=True)
    h = x
    for W in Ws:
        f = torch.tanh(h @ W.T)
        h = h + f if residual else f
    h.pow(2).sum().backward()
    return x.grad.norm().item()

print(f"plain stack    : |dL/dx| = {first_layer_grad(False):.3e}")
print(f"residual stack : |dL/dx| = {first_layer_grad(True):.3e}")
```

**Output:**
```text
plain stack    : |dL/dx| = 0.000e+00
residual stack : |dL/dx| = 2.775e+03
```

Fifty layers with small weights: the plain stack's input gradient has underflowed to exactly zero, while
the residual stack still delivers a large, non-zero gradient to its input (the exact size depends on the
scale of this toy loss; what matters is that it did not vanish).

---

## 8. Loss functions and numerical stability

### 8.1 Binary cross-entropy (recap from A3)

The model outputs a logit $z$ for a pair; $p=\sigma(z)$ is the modelled probability that the pair is a
true indication. For a label $y\in\{0,1\}$ the Bernoulli likelihood is $p^y(1-p)^{1-y}$; its negative log is
$$
\ell(p,y) = -\bigl[y\log p + (1-y)\log(1-p)\bigr],
$$
and the training loss is the mean over the sampled pairs. Minimising BCE = maximising the likelihood of
the observed labels.

### 8.2 BCE *with logits*: one function, two reasons

Substitute $p=\sigma(z)$, use $\log\sigma(z) = -\log(1+e^{-z})$ and $\log(1-\sigma(z)) = -z-\log(1+e^{-z})$:
$$
\ell(z,y) = \log(1+e^{-z}) + (1-y)\,z = \log(1+e^{z}) - y z = \operatorname{softplus}(z) - yz .
$$
**Gradient:** $\dfrac{\partial\ell}{\partial z} = \sigma(z) - y$. Bounded between $-1$ and $1$, never
vanishing while the prediction is wrong. (Compare: through a separate sigmoid, the gradient of BCE
w.r.t. $p$ is $-1/p$ for $y=1$, which explodes as $p\to0$, then gets multiplied by
$\sigma'(z)$, which vanishes. Fusing the two cancels the explosion against the vanishing analytically.)

**Stable evaluation.** $\log(1+e^{z})$ overflows for large $z$. The standard stable form is
$$
\ell(z,y) = \max(z,0) - zy + \log\bigl(1+e^{-|z|}\bigr),
$$
in which the exponent is never positive. `F.binary_cross_entropy_with_logits` (and the module
`nn.BCEWithLogitsLoss`) implements this.

**Why `sigmoid` then `binary_cross_entropy` is dangerous.** In `float32`, $\sigma(-200)$ is exactly 0 (since
$e^{200}$ overflows). The true loss for $y=1$ is $200$; $\log 0=-\infty$. PyTorch's `F.binary_cross_entropy`
clamps $\log$ at $-100$ to avoid infinities, so it reports a loss of 100 and — worse — a **zero gradient**,
silently stopping learning on exactly the examples the model gets most wrong.

```python
import torch
import torch.nn.functional as F

y = torch.tensor(1.0)
for z0 in (-5.0, -50.0, -200.0):
    z = torch.tensor(z0, requires_grad=True)
    l1 = F.binary_cross_entropy(torch.sigmoid(z), y)
    (g1,) = torch.autograd.grad(l1, z)
    z = torch.tensor(z0, requires_grad=True)
    l2 = F.binary_cross_entropy_with_logits(z, y)
    (g2,) = torch.autograd.grad(l2, z)
    print(f"z={z0:7.1f} | sigmoid+BCE: loss={l1.item():9.4f} grad={g1.item():+.4f} "
          f"| with_logits: loss={l2.item():9.4f} grad={g2.item():+.4f}")
```

**Output:**
```text
z=   -5.0 | sigmoid+BCE: loss=   5.0067 grad=-0.9933 | with_logits: loss=   5.0067 grad=-0.9933
z=  -50.0 | sigmoid+BCE: loss=  50.0000 grad=-0.0000 | with_logits: loss=  50.0000 grad=-1.0000
z= -200.0 | sigmoid+BCE: loss= 100.0000 grad=-0.0000 | with_logits: loss= 200.0000 grad=-1.0000
```

At $z=-5$ the two agree. At $-50$ the separate version still gets the loss right but its gradient has
collapsed to about $-2\times10^{-10}$ (printed as $-0.0000$): the backward of `binary_cross_entropy` guards
against division by zero by clamping $p(1-p)$ at $10^{-12}$, while the sigmoid's own derivative here is
$\approx 2\times10^{-22}$, so their product is negligible. At $-200$ even the loss is wrong. The fused
version is exact everywhere with gradient $\sigma(z)-1\approx-1$.

That is why the project's training loss is
```py
loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
```
and the model's `forward` returns **logits**, applying `torch.sigmoid` only at the very end for reporting.

### 8.3 Other numerical hazards and how the project handles them

| Hazard | Where it could bite | Project's guard |
|---|---|---|
| Division by zero | row-normalising a kNN kernel when a node has no neighbours | `knn_kernel`: `s[s == 0] = 1.0` |
| Division by zero in $D^{-1/2}$ | normalising a graph with an isolated node | `sym_norm`: `d[d == 0] = 1`; LAGCN: `.clamp(min=1e-12).rsqrt()` |
| $0/0$ in softmax | node with no neighbours / no valid relation | `torch.nan_to_num(att, nan=0.0)` in `DenseGAT` and `ViewAttention` |
| Missing similarities (NaN) | biologics without SMILES, diseases without genes | `fill_missing` replaces NaN with 0 and puts 1 on the diagonal |
| $\log 0$ | degree 0 in the degree gate | `torch.log1p(deg)` $=\log(1+\text{deg})$, which is 0 at degree 0 |
| Overflow in `exp` | softmax / BCE | library functions with built-in max-subtraction / stable forms |

General facts about floating point you should know:

* `float32` has about 7 significant decimal digits (machine epsilon $\approx 1.19\times10^{-7}$);
  `float64` about 16. GPUs are much faster in `float32`, so models train in `float32`; gradient checks and
  delicate linear algebra use `float64`.
* Floating-point addition is **not associative**: $(0.1+0.2)+0.3 \ne 0.1+(0.2+0.3)$ in binary floating point.
  The order of a sum changes the last bits — the root of GPU non-determinism (Section 14).
* NaN is contagious: any arithmetic with NaN gives NaN, and NaN $\ne$ NaN. Detect it early with
  `torch.isnan(x).any()` or `torch.autograd.set_detect_anomaly(True)` (slow; debugging only).

### 8.4 How the labels are made in the project

For each epoch, `fit_predict` builds the supervised set from flat cell indices (cell $k$ of the
$593\times313$ matrix is drug $k // 313$, disease $k \% 313$):

* positives `sup` — the known training links hidden this epoch;
* negatives `neg` — `sup.numel() * neg_ratio` cells drawn at random (with replacement) from `neg_pool`,
  the unknown cells that are allowed as negatives in this fold;
* labels `y = [1,…,1, 0,…,0]` built with `torch.cat([torch.ones(...), torch.zeros(...)])`.

`logits.reshape(-1)` flattens the logit matrix in the same row-major order, so `logits[sup]` picks
exactly the logits of those cells. Treating sampled unknowns as negatives is the "unknown ≠ negative"
compromise discussed in Unit B4.

---

## 9. PyTorch tensors

A **tensor** is PyTorch's n-dimensional array: like a NumPy `ndarray`, plus two superpowers — it can live
on a GPU, and it can record the operations applied to it so that gradients can be computed (Section 10).

### 9.1 Creating tensors and reading their metadata

```python
import numpy as np
import torch

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # from Python data
b = torch.zeros(2, 3)
c = torch.arange(6).reshape(2, 3)
d = torch.randn(2, 3, generator=torch.Generator().manual_seed(0))
print(a.shape, a.ndim, a.numel(), a.dtype, a.device)
print(c.dtype, (c > 2).dtype)

S = np.random.default_rng(0).random((3, 3))          # NumPy defaults to float64
print(torch.as_tensor(S).dtype, torch.as_tensor(S, dtype=torch.float32).dtype)
```

**Output:**
```text
torch.Size([2, 3]) 2 6 torch.float32 cpu
torch.int64 torch.bool
torch.float64 torch.float32
```

### 9.2 dtypes you will meet, and why they matter

| dtype | Alias | Used for | In the project |
|---|---|---|---|
| `torch.float32` | `float` | parameters, activations, losses | every feature/similarity tensor via `t(x)` |
| `torch.float64` | `double` | gradient checks, careful linear algebra | NumPy arrays default to it — convert! |
| `torch.int64` | `long` | indices (`x[idx]`, `torch.randint`, embedding lookups) | `pos`, `neg_pool` via `t(..., torch.long)` |
| `torch.bool` | | masks | adjacency masks via `t(knn_mask(...), torch.bool)` |
| `torch.float16`/`bfloat16` | half | mixed-precision training | not used |

The helper in `methods.py`,

```py
def t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)
```

exists mainly to enforce two things at once: everything coming from NumPy (which is `float64` by default)
becomes `float32`, and everything lands on the same device. Mixing `float64` data with `float32`
parameters raises errors like *"expected scalar type Float but found Double"*; mixing devices raises
*"Expected all tensors to be on the same device"*.

Conversions: `x.float()`, `x.long()`, `x.bool()`, `x.to(torch.float64)`. A boolean mask turned into
numbers with `.float()` gives 0/1 — `degrees` in `fit_predict` uses exactly this: `Am.float().sum(1)` counts
the visible links of every drug.

### 9.3 Broadcasting

**Rule.** To combine two tensors element-wise, PyTorch aligns their shapes **from the right**. Two
dimensions are compatible if they are equal or one of them is 1 (or missing). Size-1 dimensions are
virtually stretched (no memory is copied). The result has, in each position, the larger size.

```
  (593, 1, 4)        zd scores, one per (drug, head), with a new source axis
+ (1, 593, 4)        zs scores, one per (source, head), with a new destination axis
= (593, 593, 4)      e[i, j, h] = score_dst[i, h] + score_src[j, h]
```

Indexing with `None` (or `unsqueeze`) inserts a size-1 axis exactly where you want one. Four project
lines rely on this:

| Line | Shapes | Meaning |
|---|---|---|
| `(zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]` (`DenseGAT`) | $(D,1,H)+(1,S,H)\to(D,S,H)$ | all pairwise GAT scores without a loop |
| `.masked_fill(~mask[..., None], -inf)` (`DenseGAT`) | mask $(D,S)\to(D,S,1)$ against $(D,S,H)$ | same neighbour mask for every head |
| `(beta[..., None] * msgs).sum(0)` (`ViewAttention`) | $(R,N,1)\cdot(R,N,64)$ then sum over $R$ | weighted sum of relation messages |
| `sigmoid(...)[:, None] * sigmoid(...)[None, :]` (`gnn_gate`) | $(n_r,1)\cdot(1,n_d)\to(n_r,n_d)$ | an outer product: one gate per pair |

```python
import torch

s_dst = torch.tensor([[1.0], [2.0], [3.0]])        # (3, 1)
s_src = torch.tensor([[10.0, 20.0]])               # (1, 2)
print((s_dst + s_src).shape)
print(s_dst + s_src)

# The "silent bug" broadcast: row-normalising a SQUARE matrix
A = torch.tensor([[1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0],
                  [0.0, 1.0, 1.0]])
deg = A.sum(1)                                     # shape (3,), NOT (3, 1)
wrong = A / deg                                    # (3,3)/(3,) -> divides COLUMN j by deg[j]
right = A / deg[:, None]                           # (3,3)/(3,1) -> divides ROW i by deg[i]
print("row sums wrong:", wrong.sum(1).tolist())
print("row sums right:", right.sum(1).tolist())
```

**Output:**
```text
torch.Size([3, 2])
tensor([[11., 21.],
        [12., 22.],
        [13., 23.]])
row sums wrong: [0.8333333730697632, 1.3333333730697632, 0.8333333730697632]
row sums right: [1.0, 1.0, 1.0]
```

The second example is the single most common silent bug in graph code: on a square matrix, a shape
`(n,)` vector broadcasts against the *last* axis, i.e. columns. No error is raised. NumPy's `knn_kernel`
avoids it with `keepdims=True`: `s = W.sum(1, keepdims=True)` has shape $(n,1)$.

### 9.4 Reshaping, views and copies

* `x.view(shape)` reinterprets the same memory with a new shape; it requires the memory layout to be
  compatible ("contiguous"). `x.reshape(shape)` does the same when possible and copies otherwise.
* `x.T` / `x.transpose(0, 1)` / `x.permute(...)` return **views** with swapped strides — no data moved,
  but the result is usually non-contiguous, so `.view` on it fails; use `.reshape` or `.contiguous().view`.
* `x.squeeze(-1)` removes a size-1 axis; `x.unsqueeze(1)` or `x[:, None]` adds one.
* `-1` in a shape means "infer this size": `self.W_src(h_src).view(-1, self.h, self.dh)` turns
  $(S, 64)$ into $(S, 4, 16)$ — 4 heads of 16 dimensions each.
* Operations ending in `_` are **in-place** (`x.add_(1)`, `nn.init.xavier_uniform_(W)`): they modify the
  tensor itself.
* `torch.from_numpy(arr)` and `torch.as_tensor(arr)` (same dtype, CPU) **share memory** with the NumPy array;
  `torch.tensor(arr)` always copies. `x.numpy()` on a CPU tensor also shares memory. Mutating one mutates
  the other.

```python
import numpy as np
import torch

x = torch.arange(12.0)
v = x.view(3, 4)
v[0, 0] = 100.0                       # writes through to x: same memory
print(x[:3].tolist(), v.is_contiguous(), v.T.is_contiguous())

heads = torch.arange(2 * 8.0).view(2, 8).view(-1, 4, 2)   # (2 nodes, 4 heads, 2 dims)
print(heads.shape, heads.reshape(2, -1).shape)            # back to (2, 8): concatenated heads

arr = np.zeros(3)
shared = torch.from_numpy(arr); copied = torch.tensor(arr)
arr[0] = 7.0
print(shared.tolist(), copied.tolist())
```

**Output:**
```text
[100.0, 1.0, 2.0] True False
torch.Size([2, 4, 2]) torch.Size([2, 8])
[7.0, 0.0, 0.0] [0.0, 0.0, 0.0]
```

### 9.5 Indexing: boolean masks and flat indices

The project stores drug–disease cells by **flat index** $k = i\cdot n_d + j$ (row-major order, as
`np.ravel` and `torch.reshape(-1)` use). Recovering coordinates: $i = k\ //\ n_d$, $j = k \bmod n_d$.
In `fit_predict`, `cold[pos % n_d]` looks up, for each positive cell, whether *its disease* was picked
for cold-start practice, and `Am[pos[~hide]] = True` writes the still-visible links into a flat boolean
vector that is then `view`ed back to $(n_r, n_d)$.

```python
import torch

n_r, n_d = 3, 4
A = torch.tensor([[1, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0]])
pos = torch.nonzero(A.reshape(-1)).squeeze(1)      # flat indices of the 1s
print("flat positives:", pos.tolist())
print("(drug, disease):", [(int(k) // n_d, int(k) % n_d) for k in pos])

hide = torch.tensor([False, True, False, False])    # hide the 2nd known link
Am = torch.zeros(n_r * n_d, dtype=torch.bool)
Am[pos[~hide]] = True                               # boolean-mask then integer-index assignment
print(Am.view(n_r, n_d).int())

logits = torch.arange(12.0).view(n_r, n_d)          # pretend scores
print("logits of hidden links:", logits.reshape(-1)[pos[hide]].tolist())
```

**Output:**
```text
flat positives: [0, 3, 6, 9]
(drug, disease): [(0, 0), (0, 3), (1, 2), (2, 1)]
tensor([[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0]], dtype=torch.int32)
logits of hidden links: [3.0]
```

### 9.6 Reductions and the `dim` argument

`x.sum(dim=1)` sums **over** axis 1, so that axis disappears (or becomes size 1 with `keepdim=True`).
`torch.softmax(e, dim=1)` normalises **over** axis 1, so that the entries along axis 1 sum to 1. In
`DenseGAT`, `e` has shape $(D,S,H)$ and `dim=1` is the source/neighbour axis: for every destination node
and head, the attention over its neighbours sums to 1. In `ViewAttention`, `s` has shape $(R,N)$ and
`dim=0` normalises over relations for each node. `mask.any(1)` asks, for each destination, "is any source
present?".

### 9.7 `einsum`: index notation as code

`torch.einsum` lets you write a tensor contraction exactly as in index notation. In `DenseGAT`:
$$
\text{out}_{d,h,k} = \sum_{s}\ \text{att}_{d,s,h}\ \text{zs}_{s,h,k}
\qquad\Longleftrightarrow\qquad
\texttt{torch.einsum("dsh,shk->dhk", att, zs)} .
$$
Indices that appear in the inputs but not in the output (`s`) are summed; the others are kept. Per head
$h$ this is the matrix product $\text{att}[:,:,h]\ @\ \text{zs}[:,h,:]$: each destination's new vector is the
attention-weighted average of its neighbours' projected vectors.

```python
import torch
torch.manual_seed(0)
D, S, H, K = 3, 5, 2, 4
att = torch.softmax(torch.randn(D, S, H), dim=1)
zs = torch.randn(S, H, K)
out = torch.einsum("dsh,shk->dhk", att, zs)
loop = torch.stack([att[:, :, h] @ zs[:, h, :] for h in range(H)], dim=1)
print(out.shape, torch.allclose(out, loop))
print("attention sums over neighbours:", att.sum(1)[0].tolist())
```

**Output:**
```text
torch.Size([3, 2, 4]) True
attention sums over neighbours: [1.0, 0.9999999403953552]
```

($0.99999994$ instead of $1$ is ordinary `float32` rounding — one unit in the last place. Compare
floating-point results with `torch.allclose`, never with `==`.)

---

## 10. Autograd: automatic differentiation in PyTorch

### 10.1 The recorded graph

When a tensor has `requires_grad=True`, every operation that uses it creates a node in a computational
graph (Section 6) and the result remembers its creator in `.grad_fn`. Tensors created directly by you
(parameters, inputs) are **leaves**; their `.grad` field receives gradients. PyTorch builds this graph
**dynamically**, anew on every forward pass ("define-by-run"), which is why ordinary Python control flow
(`if c.feat_prop:`, loops over relations) just works inside `forward`.

`loss.backward()` walks the recorded graph backwards, accumulates $\partial L/\partial\text{leaf}$ into each
leaf's `.grad`, and then frees the graph (the saved intermediate tensors) to reclaim memory.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)       # a leaf "parameter"
x = torch.tensor(3.0)                           # data: no gradient needed
y = w * x + w ** 2
print(w.is_leaf, y.is_leaf, y.grad_fn.__class__.__name__)

y.backward()                                    # dy/dw = x + 2w = 7
print("after 1st backward:", w.grad.item())
y = w * x + w ** 2
y.backward()                                    # gradients ACCUMULATE
print("after 2nd backward:", w.grad.item())
w.grad = None                                   # what optimizer.zero_grad() does
y = w * x + w ** 2
y.backward()
print("after zeroing     :", w.grad.item())
```

**Output:**
```text
True False AddBackward0
after 1st backward: 7.0
after 2nd backward: 14.0
after zeroing     : 7.0
```

### 10.2 Why gradients accumulate, and why we call `zero_grad()`

Accumulation is a feature: it is what makes fan-out correct (Section 6.1: gradients from two uses of a
value are *added*), and it allows "gradient accumulation" over several small batches. But it means that,
in a training loop, gradients from the previous step must be cleared before the next `backward()`. If you
forget `opt.zero_grad()`, step $t$ uses the sum of all gradients so far — effectively an exploding
learning rate. Modern PyTorch's `zero_grad()` sets `.grad` to `None` (cheaper than filling zeros).

### 10.3 `backward` on non-scalars

`backward()` without arguments only works on a scalar. For a tensor output `Y` you must supply the
upstream gradient `G` of the same shape: `Y.backward(G)` computes the vector–Jacobian product
(Section 6.5 used this). `loss.backward()` is the special case `G = 1`.

### 10.4 Switching recording off: `detach`, `no_grad`, `inference_mode`

* `x.detach()` returns a tensor sharing data with `x` but cut from the graph: gradients will not flow
  through it. Use it to treat something as a constant (e.g. a target).
* `with torch.no_grad():` — inside this block, no operation is recorded. Results have
  `requires_grad=False`, no intermediates are saved for a backward pass, so memory and time are saved.
  Used for evaluation/prediction and for manual parameter updates (Section 6.4's `P -= 0.1 * P.grad`).
* `@torch.no_grad()` as a **decorator** does the same for a whole function: the project's
  `view_attention`, `view_weights` and `occlusion` methods are decorated this way.
* `torch.inference_mode()` is a stricter, slightly faster version of `no_grad` for pure inference.
* `p.requires_grad_(False)` freezes a parameter permanently (fine-tuning).

```python
import torch

lin = torch.nn.Linear(1000, 1000)
x = torch.randn(256, 1000)
y = lin(x)
print("recorded:", y.requires_grad, y.grad_fn is not None)
with torch.no_grad():
    y2 = lin(x)
print("inside no_grad:", y2.requires_grad, y2.grad_fn)
print("same numbers:", torch.equal(y, y2))
try:
    y2.sum().backward()
except RuntimeError as err:
    print("backward fails:", str(err).split("\n")[0])
```

**Output:**
```text
recorded: True True
inside no_grad: False None
same numbers: True
backward fails: element 0 of tensors does not require grad and does not have a grad_fn
```

### 10.5 Two errors you will eventually see

* *"Trying to backward through the graph a second time"*: the graph was freed by the first `backward()`.
  Either recompute the forward pass or pass `retain_graph=True` (rarely what you actually want).
* *"one of the variables needed for gradient computation has been modified by an inplace operation"*:
  you changed, in place, a tensor that the backward pass had saved. Replace `x += ...` / `x[mask] = ...`
  by an out-of-place version (`x = x + ...`, `x = x.masked_fill(mask, ...)`). Notice the project uses the
  out-of-place `masked_fill` (no underscore) on tensors in the graph, while in-place writes like
  `Am[pos[~hide]] = True` happen only on tensors that do not require gradients.

---

## 11. Building models: `nn.Module`, `nn.Linear`, `nn.Parameter`

### 11.1 Anatomy of a module

```py
class MyLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()          # mandatory: sets up parameter/submodule registries
        self.lin = nn.Linear(...)   # assigning a Module registers it as a submodule
        self.w = nn.Parameter(...)  # assigning a Parameter registers it as a parameter
    def forward(self, x):           # the computation; never call it directly
        return ...
```

Calling `layer(x)` runs `nn.Module.__call__`, which runs any hooks and then `forward`. Registration is
what makes `model.parameters()`, `model.to(device)`, `model.train()/eval()`, `model.state_dict()` and the
optimiser "see" everything inside, recursively.

### 11.2 `nn.Parameter` vs a plain tensor vs a buffer

* `nn.Parameter(t)` — a tensor flagged as a learnable parameter (`requires_grad=True`), registered when
  assigned as an attribute. The project uses raw parameters for things that are not a standard layer:
  GAT attention vectors `a_src`/`a_dst`, decoder matrix `W`, propagation weights `prop_w`, `prop_scale`,
  `bias`, and the degree-gate coefficients `gate`.
* A plain tensor attribute — **not** registered: not in `parameters()` (never trained), not moved by
  `.to(device)`.
* `self.register_buffer("name", t)` — registered state that is moved and saved but not trained
  (e.g. BatchNorm's running mean).

### 11.3 `nn.Linear` precisely

`nn.Linear(in_features, out_features, bias=True)` holds `weight` of shape `(out, in)` and `bias` of shape
`(out,)` and computes $y = xW^\top + b$ over the last axis of $x$ (any number of leading axes are treated as
batch axes). `bias=False` drops $b$ — `DenseGAT` uses this for `W_src`/`W_dst` because a bias in the GAT
projection would add the same vector to every neighbour's message and is not part of the GAT equations.
Parameters: $\text{in}\cdot\text{out} + \text{out}$ with bias.

### 11.4 Containers and counting parameters

* `nn.Sequential(a, b, c)` — chains modules.
* `nn.ModuleList([...])` — a list whose elements are registered (MV-HGAT's stack of `HeteroLayer`s).
* `nn.ModuleDict({...})` — a dict whose values are registered, keyed by string (one `DenseGAT` per
  relation name, one `LayerNorm` per node type).

**A plain Python `list` or `dict` of modules is invisible to PyTorch** — the optimiser would never update
those weights, and `.to("cuda")` would leave them on the CPU. The code below demonstrates the bug and
then counts the parameters of the project's own modules.

```python
import sys
import torch
import torch.nn as nn
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import DenseGAT, HeteroLayer, MVHGAT

class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = {"drug": nn.Linear(4, 4)}           # plain dict: NOT registered
class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({"drug": nn.Linear(4, 4)})
count = lambda m: sum(p.numel() for p in m.parameters())
print("plain dict params:", count(Bad()), "| ModuleDict params:", count(Good()))

g = DenseGAT(64, 64, heads=4, dropout=0.2)
print("DenseGAT(64,64,4):", {n: tuple(p.shape) for n, p in g.named_parameters()}, "total", count(g))

# the default Fdataset graph: 3 drug views, 3 disease views, assoc in both directions
rels = {f"view:{v}": ("drug", "drug") for v in ("chem_cdk", "chem_ecfp", "gene_r")}
rels.update({f"view:{v}": ("disease", "disease") for v in ("pheno_mim", "sem_mondo", "gene_d")})
rels.update({"assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")})
print("HeteroLayer with", len(rels), "relations:", count(HeteroLayer(rels, 64, 64, 4, 0.2)))

in_drug = 3 * 593 + 313      # 3 drug similarity rows + visible association row
in_dis = 3 * 313 + 593       # 3 disease similarity rows + visible association column
m = MVHGAT(rels, in_drug, in_dis, hidden=64, layers=2, heads=4, dropout=0.2, n_prop=6)
by_part = {}
for n, p in m.named_parameters():
    key = n.split(".")[0]
    by_part[key] = by_part.get(key, 0) + p.numel()
print("MVHGAT (Fdataset sizes):", by_part, "total", count(m))
```

**Output:**
```text
plain dict params: 0 | ModuleDict params: 20
DenseGAT(64,64,4): {'a_src': (4, 16), 'a_dst': (4, 16), 'W_src.weight': (64, 64), 'W_dst.weight': (64, 64)} total 8320
HeteroLayer with 8 relations: 83584
MVHGAT (Fdataset sizes): {'prop_w': 6, 'prop_scale': 1, 'bias': 1, 'gate': 4, 'W': 36864, 'inp': 232064, 'layers': 167168} total 436108
```

Check `DenseGAT` by hand: `W_src` and `W_dst` are $64\times64=4096$ each (no bias) and `a_src`, `a_dst`
are $4\times16=64$ each: $2\cdot4096+2\cdot64=8320$. A `HeteroLayer` has 8 GATs ($66{,}560$), two
`ViewAttention`s ($2\times(64\cdot64+64+64)=8448$), two skip `Linear`s ($2\times4160=8320$) and two
LayerNorms ($2\times128=256$): $83{,}584$. Most of MV-HGAT's parameters sit in the first `nn.Linear` that
reads the long similarity rows (`inp`: $2092\cdot64+64 + 1532\cdot64+64 = 232{,}064$).

### 11.5 Saving, loading, moving

`model.state_dict()` is an ordered dict of all parameters and buffers; `torch.save(model.state_dict(), path)`
and `model.load_state_dict(torch.load(path))` are the standard checkpointing idiom. `model.to(DEVICE)`
moves every registered parameter and buffer; it is **in-place for modules** (returns `self`), but
`x.to(DEVICE)` for a **tensor returns a new tensor** — you must write `x = x.to(DEVICE)`.

---

## 12. Optimisers and the training loop

### 12.1 Optimisers in one paragraph each (details in B1)

**SGD:** $\theta\leftarrow\theta-\eta\,g$. **Momentum:** keep a velocity $v\leftarrow\mu v+g$ and step
$\theta\leftarrow\theta-\eta v$, smoothing noisy gradients. **Adam** (Kingma & Ba 2015) keeps running
averages of the gradient and of its square,
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\quad
\hat m_t = \frac{m_t}{1-\beta_1^t},\ \hat v_t=\frac{v_t}{1-\beta_2^t},\quad
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
$$
($\beta_1=0.9$, $\beta_2=0.999$ by default), so each parameter gets its own effective step size — robust to
badly scaled gradients, which is why it is the default for GNNs. The project uses
`torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)`.

**Weight decay in Adam vs AdamW.** In `torch.optim.Adam`, `weight_decay=λ` adds $\lambda\theta$ to the
gradient *before* the adaptive scaling (classic L2 regularisation). `torch.optim.AdamW` (Loshchilov &
Hutter 2019) instead shrinks the weights directly, $\theta\leftarrow\theta-\eta\lambda\theta$, decoupled from the
adaptive scaling; with Adam the two are not equivalent. Either is a legitimate regulariser; just know
which one you used when you report it.

### 12.2 The canonical loop

```py
model = Model(...).to(device)
opt = torch.optim.Adam(model.parameters(), lr=..., weight_decay=...)
for epoch in range(n_epochs):
    model.train()                 # dropout on
    logits = model(inputs)        # 1. forward
    loss = loss_fn(logits, y)     # 2. loss
    opt.zero_grad()               # 3. clear old gradients
    loss.backward()               # 4. backward: fill p.grad for every parameter
    opt.step()                    # 5. update parameters using p.grad
model.eval()                      # dropout off
with torch.no_grad():             # no graph recording
    scores = torch.sigmoid(model(inputs))
```

`zero_grad()` may go anywhere *before* `backward()` (the project puts it right before). The loop above is
**full-batch**: one forward pass over the whole graph per epoch, which is what MV-HGAT does (the graph has
under 1,100 nodes). Large datasets use **mini-batches** from a `DataLoader`, with the same five steps per
batch. An **epoch** is one pass over the training data; with full-batch training, epoch = step.

### 12.3 A complete, runnable example

A two-hidden-layer MLP on the "two moons" toy problem (non-linearly separable), with dropout, Adam,
BCE-with-logits, a validation split, and correct use of `train`/`eval`/`no_grad`.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

torch.manual_seed(0)
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X = torch.as_tensor(X, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)
perm = torch.randperm(len(X))
tr, va = perm[:450], perm[450:]

class MLP(nn.Module):
    def __init__(self, d_in, d_h, p_drop):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_h)
        self.l2 = nn.Linear(d_h, d_h)
        self.out = nn.Linear(d_h, 1)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        h = self.drop(F.elu(self.l1(x)))
        h = self.drop(F.elu(self.l2(h)))
        return self.out(h).squeeze(-1)          # logits, shape (B,)

model = MLP(2, 32, 0.2)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

def accuracy(idx):
    model.eval()
    with torch.no_grad():
        return ((model(X[idx]) > 0).float() == y[idx]).float().mean().item()

for epoch in range(301):
    model.train()
    logits = model(X[tr])
    loss = F.binary_cross_entropy_with_logits(logits, y[tr])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 75 == 0:
        print(f"epoch {epoch:3d}  train loss {loss.item():.4f}  "
              f"train acc {accuracy(tr):.3f}  val acc {accuracy(va):.3f}")
```

**Output:**
```text
epoch   0  train loss 0.7201  train acc 0.827  val acc 0.853
epoch  75  train loss 0.3119  train acc 0.882  val acc 0.887
epoch 150  train loss 0.2734  train acc 0.893  val acc 0.873
epoch 225  train loss 0.1922  train acc 0.940  val acc 0.927
epoch 300  train loss 0.1624  train acc 0.938  val acc 0.920
```

The initial loss is close to $\ln 2 = 0.693$, which is what BCE gives when all logits are near 0
(predicting 0.5 for everything). Checking that number at the start of training is a free sanity test.
(Exact digits can differ slightly between PyTorch versions and CPUs; the shape of the curve will not.)

### 12.4 `model.train()` / `model.eval()` vs `torch.no_grad()` — two independent switches

| | gradients recorded (`no_grad` off) | no gradients (`no_grad` on) |
|---|---|---|
| **`train()` mode** | normal training step | rarely useful (e.g. computing training-mode statistics) |
| **`eval()` mode** | e.g. computing gradients w.r.t. inputs for saliency | **prediction / evaluation** |

* `train()`/`eval()` change the **behaviour of certain layers** — Dropout (random vs identity) and
  BatchNorm (batch vs running statistics). They do *not* affect gradient recording.
* `no_grad()` changes **whether autograd records** — it does *not* change dropout.

So prediction needs **both**: `model.eval()` for deterministic, full-strength outputs, and
`torch.no_grad()` to save memory and time. `fit_predict` does exactly this. Note that `eval()` mode
persists until you call `train()` again, which is why the project calls `model.train()` at the top of every
epoch.

```python
import torch
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
x = torch.ones(1, 4)

net.train()
with torch.no_grad():
    a, b = net(x), net(x)
print("train + no_grad -> outputs differ (dropout active):", not torch.equal(a, b))
net.eval()
c = net(x)
print("eval without no_grad -> still records a graph:", c.requires_grad)
with torch.no_grad():
    d = net(x)
print("eval + no_grad -> deterministic and no graph:", torch.equal(c, d), d.requires_grad)
```

**Output:**
```text
train + no_grad -> outputs differ (dropout active): True
eval without no_grad -> still records a graph: True
eval + no_grad -> deterministic and no graph: True False
```

---

## 13. Devices: CPU and GPU

**Why a GPU.** A GPU (here an NVIDIA RTX 3050 with 6 GB of memory) runs thousands of simple arithmetic
units in parallel. MV-HGAT's work is dominated by dense matrix products and element-wise operations on
$(593\times593\times4)$-sized tensors, exactly the workload GPUs excel at.

**The pattern used in `methods.py`** (abridged, comments added):

```py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...
model = MVHGAT(...).to(DEVICE)                     # move all parameters
hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge   # create directly on the device
return torch.sigmoid(logits).cpu().numpy()          # back to host memory for NumPy
```

Rules:

1. **All operands of an operation must be on the same device.** Otherwise: *"Expected all tensors to be
   on the same device, but found at least two devices, cuda:0 and cpu!"* The helper `t()` puts every input
   on `DEVICE`; random tensors are created with `device=DEVICE`.
2. **NumPy only understands CPU memory.** `x.numpy()` on a CUDA tensor fails; use `x.cpu().numpy()`
   (and `x.detach()` first if `x` requires grad, unless you are inside `no_grad`).
3. **GPU calls are asynchronous.** Python queues kernels and moves on; anything that needs the actual
   value (`.item()`, `.cpu()`, `print(x)`) forces a synchronisation. Many `.item()` calls inside a loop slow
   training down; this is why `fit_predict` does not log the loss every epoch.
4. **Memory.** A dense attention tensor for one drug–drug relation is $593\cdot593\cdot4$ float32 values
   $=5.6$ MB, and autograd stores several such intermediates per relation per layer for the backward pass —
   tens to a few hundred MB in total, comfortable on 6 GB. Under `no_grad` most of it is not kept.
5. **Choosing devices from outside.** Setting the environment variable `CUDA_VISIBLE_DEVICES=-1` hides
   all GPUs from a process, so `torch.cuda.is_available()` returns `False` and the code falls back to the
   CPU. (Use `-1`, not an empty string: on Windows an empty environment variable may simply be dropped.)
   Every code example in this chapter was verified on the CPU this way, so the outputs below say `cpu`;
   on a machine with a visible GPU the first line would say `cuda`.

```python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used here:", DEVICE)
x = torch.randn(3, 3, device=DEVICE)
lin = torch.nn.Linear(3, 2).to(DEVICE)
print(lin.weight.device == x.device, lin(x).shape)
print(type(lin(x).detach().cpu().numpy()).__name__)
```

**Output:**
```text
device used here: cpu
True torch.Size([3, 2])
ndarray
```

---

## 14. Seeds, reproducibility and determinism

Neural-network training uses randomness everywhere: weight initialisation, dropout masks, negative
sampling, which links are hidden each epoch. A **pseudo-random number generator (PRNG)** produces a
deterministic sequence from a **seed**, so fixing the seed fixes the sequence. Python, NumPy, PyTorch-CPU
and PyTorch-CUDA each have their *own* generator, which is why `methods.py` seeds all four:

```py
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

and `fit_predict` calls `set_seed(seed)` at the start of every fit, so each (fold, repeat) is
reproducible and different repeats use different seeds (giving the mean ± std reported in the results).

What seeding does and does not guarantee:

* **Same code, same machine, same library versions, CPU:** bit-for-bit identical results.
* **Order matters.** The sequence is consumed by every random call. Adding one extra `torch.rand` before
  model construction changes the initial weights. Seeds give reproducibility, not "the same weights
  regardless of code changes".
* **GPU non-determinism.** Some CUDA kernels (e.g. `index_add_`, scatter-based aggregations and some
  backward passes) sum numbers in an order that depends on thread scheduling; since floating-point
  addition is not associative, results can differ in the last bits between runs, and over hundreds of
  epochs these differences can grow into visibly different metrics. Remedies:
  `torch.use_deterministic_algorithms(True)` (raises an error if a non-deterministic op is used; some
  cuBLAS operations also require the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`),
  and `torch.backends.cudnn.benchmark = False`. These cost speed.
* **Different hardware or versions** can change results even with all of the above. That is why
  research reports mean ± std over several seeds rather than a single run.

```python
import torch

def init_weights(seed):
    torch.manual_seed(seed)
    return torch.nn.Linear(3, 2).weight.detach().clone()

print("same seed, same weights:", torch.equal(init_weights(0), init_weights(0)))
print("different seed         :", torch.equal(init_weights(0), init_weights(1)))

torch.manual_seed(0)
_ = torch.rand(1)                                  # one extra random call...
shifted = torch.nn.Linear(3, 2).weight.detach()
print("extra random call first:", torch.equal(init_weights(0), shifted))

print("float addition associative?", (0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3))
big, one = torch.tensor(1e8), torch.tensor(1.0)      # float32 has ~7 significant digits
print("(1e8 + 1) - 1e8 =", ((big + one) - big).item(), "| (1e8 - 1e8) + 1 =", ((big - big) + one).item())
```

**Output:**
```text
same seed, same weights: True
different seed         : False
extra random call first: False
float addition associative? False
(1e8 + 1) - 1e8 = 0.0 | (1e8 - 1e8) + 1 = 1.0
```

---

## 15. Debugging: shapes first, then numbers

### 15.1 A systematic method for shape errors

1. **Annotate shapes in comments**, as `model.py` does (`# S,H,dh`, `# D,H,dh`, `# R,N`). Writing the
   shape forces you to think about it.
2. **Read the error message literally.** *"mat1 and mat2 shapes cannot be multiplied (593x64 and 128x64)"*
   means the inner dimensions (64 vs 128) disagree: the layer expected 128 input features.
   *"The size of tensor a (593) must match the size of tensor b (313) at non-singleton dimension 1"* is a
   broadcasting failure: you probably mixed up drugs and diseases (or forgot a transpose like `A.T`).
3. **Print shapes at module boundaries**, temporarily, or register a forward hook that prints them.
4. **Assert invariants** that must hold: `assert att.shape == (D, S, H)`,
   `assert torch.allclose(att.sum(1)[mask.any(1)], torch.ones(...))`.
5. **Test with tiny, *distinct* sizes.** Use 5 drugs, 4 diseases, 3 features — never square sizes, which
   hide transposition bugs (Section 9.3).

### 15.2 When the shapes are right but learning is not

| Symptom | Likely cause | Check |
|---|---|---|
| Loss is NaN from the start | `log(0)`, $0/0$ in an empty softmax row, NaN in input | `torch.isnan(X).any()`; anomaly mode |
| Loss is NaN after a while | learning rate too high, exploding logits | lower `lr`, gradient clipping (`torch.nn.utils.clip_grad_norm_`) |
| Loss does not move | forgot `opt.step()`/`zero_grad()`, parameters not registered, `lr` tiny, everything detached | print `p.grad.norm()` per parameter; some may be `None` |
| Train loss falls, validation does not | over-fitting or train/test mismatch | the project's hidden-link supervision fixed exactly this (HOW_IT_WORKS §10.1) |
| Results differ run to run | missing seed, GPU non-determinism | Section 14 |
| Predictions noisy at test time | forgot `model.eval()` | Section 12.4 |

**The "overfit one tiny batch" test** (Karpathy's recipe): take 10 examples and train until the loss is
near zero. If a model cannot memorise 10 examples, there is a bug, not a capacity problem.

---

## 16. In this project: reading `model.py` and `fit_predict` line by line

### 16.1 Concept → code map

| Concept (section) | Where in the project |
|---|---|
| `nn.Module`, `nn.ModuleDict`, `nn.ModuleList` (11) | every class in `model.py` |
| `nn.Linear` with / without bias (11.3) | `DenseGAT.W_src/W_dst` (no bias), `ViewAttention.proj`, `HeteroLayer.skip`, `MVHGAT.inp` |
| `nn.Parameter` + Xavier init (7.1, 11.2) | `DenseGAT.a_src/a_dst`, `MVHGAT.W`; plain `nn.Parameter` for `prop_w`, `prop_scale`, `bias`, `gate` |
| LeakyReLU, masked softmax, `nan_to_num` (4, 5.4) | `DenseGAT.forward` |
| tanh, softmax over relations (4, 5) | `ViewAttention.forward` |
| residual + LayerNorm + ELU + dropout (7) | `HeteroLayer.forward` |
| softplus, sigmoid, `log1p` (4, 8.3) | `MVHGAT.view_weights`, `MVHGAT.gnn_gate` |
| broadcasting, `einsum`, `view` (9) | `DenseGAT.forward`, `gnn_gate`, `forward` |
| training loop, Adam, BCE-with-logits (8, 12) | `MVHGATMethod.fit_predict` |
| `eval()` + `no_grad()` (12.4) | end of `fit_predict`; `@torch.no_grad()` on `view_attention`, `view_weights`, `occlusion` |
| devices (13), seeds (14) | `DEVICE`, `t()`, `set_seed` in `methods.py` |

### 16.2 `DenseGAT` — one relation, multi-head graph attention

```py
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
```

* `super().__init__()` — registers this object as a module (Section 11.1).
* `assert out_dim % heads == 0` — 64 output dims are split into 4 heads of `dh = 16`.
* `W_src`, `W_dst` — two projections without bias: one for nodes *sending* messages (sources), one for
  nodes *receiving* them (destinations). For a drug–drug relation both sides are drugs; for `assoc>drug`
  the sources are diseases and destinations drugs, so having two matrices lets each side be projected
  differently.
* `a_src`, `a_dst` — the attention vectors, one 16-dim vector per head per side. `torch.empty` allocates
  uninitialised memory, so Xavier init is mandatory (Section 7.1).
* `self.drop` — dropout applied to attention coefficients.

```py
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

Line by line, with Fdataset drug–drug sizes ($D=S=593$, $H=4$, $d_h=16$):

1. `zs`: project all source nodes, $(593,64)\to(593,64)$, and split into heads: $(593,4,16)$.
2. `zd`: the same for destinations.
3. `e`: `(zd * self.a_dst).sum(-1)` is, per node and head, the dot product $a_{dst}^{(h)\top} z_i^{(h)}$:
   shape $(D,H)$. Likewise for sources $(S,H)$. Broadcasting $(D,1,H)+(1,S,H)$ gives
   $e_{ijh} = a_{dst}^{(h)\top}z_i^{(h)} + a_{src}^{(h)\top}z_j^{(h)}$ for *all* pairs at once, shape $(D,S,H)$.
   This is GAT's $a^\top[Wh_i\,\|\,Wh_j]$ with $a$ split into its two halves (Unit C4).
4. LeakyReLU with slope 0.2 (Section 4), then **masked fill**: wherever $j$ is not a neighbour of $i$,
   the score becomes $-\infty$. `mask[..., None]` broadcasts the $(D,S)$ adjacency over heads.
5. `softmax(e, dim=1)` — normalise over sources: for each destination and head, attention over its
   neighbours sums to 1; non-neighbours get exactly 0 (Section 5.4).
6. `nan_to_num` — rows with no neighbours become all-zero instead of NaN.
7. `einsum("dsh,shk->dhk", ...)` — the attention-weighted sum of neighbour vectors, per head
   (Section 9.7), after dropping some attention weights; `.reshape(D, -1)` concatenates the 4 heads back
   into 64 dimensions.
8. Return the message and `mask.any(1)`: a boolean per destination, "this relation gave me at least one
   neighbour". `ViewAttention` uses it to ignore empty relations.

### 16.3 `ViewAttention` — softmax over relations

```py
    def forward(self, msgs, valid, uniform=False):
        # msgs: R,N,D   valid: R,N (relation has >= 1 neighbour for this node)
        if uniform:
            s = torch.zeros(valid.shape, device=msgs.device)
        else:
            s = self.q(torch.tanh(self.proj(msgs))).squeeze(-1)
        s = s.masked_fill(~valid, float("-inf"))
        beta = torch.nan_to_num(torch.softmax(s, dim=0), nan=0.0)  # R,N
        return (beta[..., None] * msgs).sum(0), beta
```

* `msgs` stacks the $R$ relation messages of all $N$ nodes of one type: $(R,N,64)$.
* Score: $s_i^r = q^\top\tanh(P m_i^r + b)$ — `proj` is `nn.Linear(64, 64)`, `q` is `nn.Linear(64, 1, bias=False)`,
  `squeeze(-1)` drops the trailing size-1 axis: $(R,N)$.
* `uniform=True` (the ablation "replace view attention by a mean") sets all scores to 0, so the softmax
  gives equal weights to all valid relations. Note `device=msgs.device`: new tensors must be created on
  the same device as the data (Section 13).
* Masking and softmax over `dim=0` — over relations — give $\beta_i^r$, the node-specific view weights that
  the project reports for interpretability.
* `beta[..., None] * msgs` broadcasts $(R,N,1)$ against $(R,N,64)$; `.sum(0)` gives $z_i=\sum_r\beta_i^r m_i^r$.

### 16.4 `HeteroLayer` — one full layer for both node types

```py
        self.gat = nn.ModuleDict({r: DenseGAT(in_dim, out_dim, heads, dropout) for r in relations})
        self.view_att = nn.ModuleDict({t: ViewAttention(out_dim) for t in ("drug", "disease")})
        self.skip = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in ("drug", "disease")})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(out_dim) for t in ("drug", "disease")})
```

`ModuleDict`s (not plain dicts — Section 11.4) keyed by relation name or node type: separate weights
for every relation and for every node type. In `forward`, each relation's GAT is run, its message is
appended to the list of its *destination* type, and then for each node type:

```py
            new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
```

which is residual (projection skip) → LayerNorm → ELU → dropout (Section 7). `drop_rel` replaces the
validity flags of selected relations by zeros, which removes them from the view softmax — the mechanism
behind the occlusion explanations.

### 16.5 `MVHGAT` — encoder, decoder, and the learned scalars

```py
        if n_prop:
            self.prop_w = nn.Parameter(torch.zeros(n_prop))
            self.prop_scale = nn.Parameter(torch.tensor(5.0))
            self.bias = nn.Parameter(torch.tensor(-3.0))
            self.gate = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 1.0]))
```

Raw `nn.Parameter`s with hand-chosen starting values (no random init needed — symmetry is not an issue
for scalars with distinct roles):

* `view_weights()` returns `F.softplus(self.prop_w) * self.prop_scale`; at initialisation every view weight
  is $\operatorname{softplus}(0)\cdot5=5\ln2\approx3.47$, and softplus keeps it positive forever (Section 4.2).
* `bias = -3` makes the initial base probability $\sigma(-3)\approx0.047$ — a sensible prior when only ~1% of
  pairs are positive (the sampled training set is 1/3 positive, but the bias still starts low).
* The degree gate, `gnn_gate`, computes
  $g_{ij}=\sigma(g_0+g_1\log(1+\deg_i))\cdot\sigma(g_2+g_3\log(1+\deg_j))$. With the initial values $(0,1,0,1)$
  and the identity $\sigma(\ln x) = x/(1+x)$, each factor is $(1+\deg)/(2+\deg)$: 0.5 for a node with no
  visible links, $0.8$ for 3 links, $0.875$ for 6. `[:, None] * [None, :]` broadcasts this into an
  $(n_r,n_d)$ matrix (an outer product).

```py
    def encode(self, X, graphs, drop_rel=()):
        h = {t: self.drop(F.elu(self.inp[t](X[t]))) for t in ("drug", "disease")}
        outs = {t: [h[t]] for t in h}
        ...
        return torch.cat(outs["drug"], 1), torch.cat(outs["disease"], 1), all_betas
```

The input `nn.Linear` compresses each node's long feature row to 64 dims, then ELU and dropout; each
layer's output is collected and concatenated along the feature axis (`dim=1`) — jumping knowledge, giving
$64\times3=192$ dims, hence the $192\times192$ decoder matrix `W`.

```py
        logits = Hr @ self.W @ Hd.T
        if self.n_prop and P is not None:
            if deg is not None:
                logits = self.gnn_gate(*deg) * logits
            logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
        return logits, betas                         # logits, drugs x diseases
```

The bilinear decoder scores every drug–disease pair in one matrix product: $(593,192)(192,192)(192,313)$
$\to(593,313)$ (Section 6.5 derived its gradients). The propagation head adds
$\sum_v w_v P_v[i,j]+b$: `view_weights()[:, None, None]` has shape $(6,1,1)$ and broadcasts against `P` of
shape $(6,593,313)$. The method returns **logits**, not probabilities (Section 8.2).

### 16.6 `fit_predict` — the training loop

```py
        model = MVHGAT(relations, X["drug"].shape[1], X["disease"].shape[1], c.hidden, c.layers,
                       c.heads, c.dropout, c.uniform_attention,
                       n_prop=len(self.prop_names) if c.prop_head else 0).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
```

Input sizes are read from the data (so the same code serves Fdataset and Cdataset), the model is moved to
the GPU, and Adam receives every registered parameter.

```py
        pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)
        neg_pool = t(np.flatnonzero((neg_mask & (A_train == 0)).ravel()), torch.long)
```

Flat `long` indices of training positives and of allowed negatives (Section 9.5).

Inside `for _ in range(c.epochs):` (600 times):

* `model.train()` — dropout on.
* `hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge` — each known link is hidden with
  probability 0.2; `cold = torch.rand(n_d, device=DEVICE) < c.cold_frac` picks ~10% of diseases, and
  `hide |= cold[pos % n_d]` hides all their links too. `Am` is the visible-link matrix built from them.
* `graphs["assoc>drug"], graphs["assoc>disease"] = Am, Am.T` — the message-passing graph only contains
  visible links.
* `logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))` — forward pass.
* `logits = logits.reshape(-1)` — flatten so flat indices can pick cells.
* `neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]` — sample negatives with
  replacement, `neg_ratio = 2` per supervised positive.
* `loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)` — stable BCE.
* `opt.zero_grad(); loss.backward(); opt.step()` — clear, backpropagate, update.

After the loop: restore the full association graph, `model.eval()`, compute `P` and degrees from all
training links, and run the final forward pass inside `torch.no_grad()`; `torch.sigmoid(...).cpu().numpy()`
returns probabilities as a NumPy array. `self.last` keeps the model and inputs for the explanation
methods, which are decorated with `@torch.no_grad()`.

### 16.7 A runnable miniature of MV-HGAT

The code below imports the project's real `MVHGAT` class (without modifying anything), builds a toy
heterogeneous graph with 7 drugs and 5 diseases, and runs the same loop structure as `fit_predict` on the
CPU. It prints the tensor shapes flowing through the model, checks that every node's view weights $\beta$
sum to 1, and shows the loss falling.

```python
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import MVHGAT
from drepo.data import knn_mask, knn_kernel

torch.manual_seed(0)
rng = np.random.default_rng(0)
n_r, n_d = 7, 5
A = (rng.random((n_r, n_d)) < 0.3).astype(np.float32)          # toy known links
def rand_sim(n):
    M = rng.random((n, n)); S = (M + M.T) / 2; np.fill_diagonal(S, 1.0); return S
Sr, Sd = rand_sim(n_r), rand_sim(n_d)

relations = {"view:chem": ("drug", "drug"), "view:pheno": ("disease", "disease"),
             "assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")}
Am = torch.as_tensor(A > 0)
graphs = {"view:chem": torch.as_tensor(knn_mask(Sr, 2)), "view:pheno": torch.as_tensor(knn_mask(Sd, 2)),
          "assoc>drug": Am, "assoc>disease": Am.T}
X = {"drug": torch.as_tensor(np.hstack([Sr, A]), dtype=torch.float32),
     "disease": torch.as_tensor(np.hstack([Sd, A.T]), dtype=torch.float32)}
Kr = torch.as_tensor(knn_kernel(Sr, 2), dtype=torch.float32)
Kd = torch.as_tensor(knn_kernel(Sd, 2), dtype=torch.float32)
At = torch.as_tensor(A)
P = torch.stack([Kr @ At, (Kd @ At.T).T])                      # (2 views, n_r, n_d)
deg = (At.sum(1), At.sum(0))

model = MVHGAT(relations, X["drug"].shape[1], X["disease"].shape[1],
               hidden=16, layers=2, heads=4, dropout=0.2, n_prop=2)
Hr, Hd, betas = model.encode(X, graphs)
print("X shapes:", tuple(X["drug"].shape), tuple(X["disease"].shape))
print("JK embeddings:", tuple(Hr.shape), tuple(Hd.shape), "| decoder W:", tuple(model.W.shape))
names, beta = betas[0]["drug"]
print("drug relations:", names, "| beta shape:", tuple(beta.shape),
      "| columns sum to 1:", torch.allclose(beta.sum(0), torch.ones(n_r)))
print("initial gate for (0 links, 0 links):",
      round(model.gnn_gate(torch.tensor([0.0]), torch.tensor([0.0])).item(), 4))

opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
pos = torch.nonzero(At.reshape(-1) > 0).squeeze(1)
neg_pool = torch.nonzero(At.reshape(-1) == 0).squeeze(1)
for epoch in range(61):
    model.train()
    logits, _ = model(X, graphs, P=P, deg=deg)
    logits = logits.reshape(-1)
    neg = neg_pool[torch.randint(neg_pool.numel(), (2 * pos.numel(),))]
    y = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())])
    loss = F.binary_cross_entropy_with_logits(torch.cat([logits[pos], logits[neg]]), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:2d} loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    scores = torch.sigmoid(model(X, graphs, P=P, deg=deg)[0])
print("scores:", tuple(scores.shape), "requires_grad:", scores.requires_grad)
print("mean score on known links %.3f vs unknown %.3f" % (scores[At > 0].mean(), scores[At == 0].mean()))
```

**Output:**
```text
X shapes: (7, 12) (5, 12)
JK embeddings: (7, 48) (5, 48) | decoder W: (48, 48)
drug relations: ['view:chem', 'assoc>drug'] | beta shape: (2, 7) | columns sum to 1: True
initial gate for (0 links, 0 links): 0.25
epoch  0 loss 0.7291
epoch 20 loss 0.2820
epoch 40 loss 0.1314
epoch 60 loss 0.1984
scores: (7, 5) requires_grad: False
mean score on known links 0.898 vs unknown 0.068
```

The loss is noisy from epoch to epoch (negatives are re-sampled and dropout is active), but the trend is
clearly downward. (This toy trains on the very links it can see — exactly the shortcut that the real `fit_predict` avoids
with hidden-link supervision. It is here only to show the mechanics.)

---

## 17. Common mistakes and misconceptions

1. **Applying `sigmoid` and then `BCEWithLogits`/`binary_cross_entropy_with_logits`.** That applies the
   sigmoid twice; the model can never output confident predictions. Pass raw logits.
2. **Applying `sigmoid` and then `binary_cross_entropy` on extreme logits.** Silent zero gradients
   (Section 8.2). Prefer the fused version.
3. **Forgetting `opt.zero_grad()`.** Gradients accumulate across steps (Section 10.2).
4. **Forgetting `model.eval()` before prediction.** Dropout stays active; scores are noisy and
   systematically different. Conversely, forgetting `model.train()` after an evaluation inside the loop
   silently trains without dropout.
5. **Believing `torch.no_grad()` turns off dropout** (it does not) **or that `model.eval()` turns off
   gradients** (it does not). They are independent switches (Section 12.4).
6. **Storing sub-modules in a plain `list`/`dict`.** They are never trained and never moved to the GPU.
   Use `nn.ModuleList`/`nn.ModuleDict` (Section 11.4).
7. **`x.to(device)` without assignment.** For tensors it returns a new tensor; the original stays put.
8. **Broadcasting a `(n,)` vector against an `(n, n)` matrix** when you meant rows (Section 9.3). Use
   `keepdim=True` or `[:, None]`.
9. **Testing with square or equal sizes.** Bugs from a missing transpose stay invisible when
   $n_r = n_d$. Use distinct small sizes.
10. **Feeding `float64` NumPy arrays to a `float32` model.** Convert with `torch.as_tensor(x, dtype=torch.float32)`.
11. **Masking after the softmax instead of before.** The weights then do not sum to 1, and masked
    entries still influenced the denominator (Section 5.4).
12. **Using a large finite negative (e.g. `-1e9`) instead of `-inf` and thinking empty rows are handled.**
    An all-masked row then becomes a *uniform* distribution over non-neighbours — silently wrong
    (Section 19, Q2).
13. **Initialising `nn.Parameter(torch.empty(...))` and forgetting to initialise it.** `empty` is garbage
    memory, sometimes huge numbers or NaN.
14. **Thinking the universal approximation theorem guarantees good test performance.** It is about
    representability on a compact set, not learnability or generalisation.
15. **Reading the loss every epoch with `.item()` on the GPU** and wondering why training is slow — each
    call synchronises.
16. **Expecting identical results across machines from a seed alone** (Section 14).
17. **In-place modification of a tensor needed for backward** (Section 10.5).
18. **Calling `.numpy()` on a tensor that requires grad or lives on the GPU.** Use
    `.detach().cpu().numpy()` (or compute it inside `no_grad`).

---

## 18. Exercises

Exercises are graded: **[C]** conceptual, **[M]** mathematical, **[P]** programming; one to three stars
for difficulty. Try each before opening the solution.

**Exercise 1 [C ★].** A colleague proposes a 3-layer network with no activation functions "because it is
deeper, so it can learn XOR". Explain why it cannot.

<details><summary>Solution</summary>

A composition of affine maps is affine (Section 2.4): $W_3(W_2(W_1x+b_1)+b_2)+b_3 = Mx+c$ with
$M=W_3W_2W_1$, $c=W_3W_2b_1+W_3b_2+b_3$. A thresholded affine function separates the plane with a single line,
and Section 2.2 proved no line separates XOR. Depth without non-linearity only changes the
parameterisation (and can even *restrict* $M$ to low rank if a middle layer is narrow), never the
function class.
</details>

**Exercise 2 [M ★].** (a) Derive the derivative of ELU$_\alpha$ for $z<0$ and show the derivative is
continuous at 0 iff $\alpha=1$. (b) What is the derivative of LeakyReLU$_{0.2}$ at $z=-3$ and at $z=3$?

<details><summary>Solution</summary>

(a) For $z<0$, $\frac{d}{dz}\alpha(e^z-1)=\alpha e^z$. As $z\to0^-$ this tends to $\alpha$; for $z>0$ the
derivative is 1. The two one-sided limits agree iff $\alpha=1$. (The function itself is continuous at 0
for every $\alpha$, since $\alpha(e^0-1)=0$.) PyTorch's `F.elu` uses $\alpha=1$ by default, as the project does.

(b) $0.2$ at $z=-3$ and $1$ at $z=3$. The slope on the negative side is constant, independent of $|z|$.
</details>

**Exercise 3 [M ★].** Compute by hand the masked softmax of $z=(1,\ 3,\ 2,\ 0)$ with mask
$(T,T,F,T)$. Then compute it again after adding 100 to every score. Which property did you use?

<details><summary>Solution</summary>

Allowed entries: $1, 3, 0$. $e^1=2.71828$, $e^3=20.0855$, $e^0=1$; sum $=23.8038$. Weights:
$(2.71828/23.8038,\ 20.0855/23.8038,\ 0,\ 1/23.8038) = (0.1142,\ 0.8438,\ 0,\ 0.0420)$.
Adding 100 to every score gives the same result by **shift invariance** (Section 5.1): the factor $e^{100}$
cancels between numerator and denominator (masked entries remain $-\infty$). This is also why a library
can safely subtract the row maximum.

```python
import torch
z = torch.tensor([1.0, 3.0, 2.0, 0.0]); mask = torch.tensor([True, True, False, True])
for shift in (0.0, 100.0):
    w = torch.softmax((z + shift).masked_fill(~mask, float("-inf")), 0)
    print([round(v, 4) for v in w.tolist()])
```

**Output:**
```text
[0.1142, 0.8438, 0.0, 0.042]
[0.1142, 0.8438, 0.0, 0.042]
```
</details>

**Exercise 4 [M ★★].** Logistic regression on one example: $x=(2,-1)$, $w=(0.5,\ 1.0)$, $b=0.5$, label
$y=0$. Compute $z$, $p$, the BCE loss, and $\partial L/\partial w$, $\partial L/\partial b$ by hand. Then check with
autograd.

<details><summary>Solution</summary>

$z = 0.5\cdot2 + 1.0\cdot(-1) + 0.5 = 0.5$; $p=\sigma(0.5)=0.622459$; $L=-\log(1-p)=-\log(0.377541)=0.974077$.
$\partial L/\partial z = p-y = 0.622459$; $\partial L/\partial w = (p-y)\,x = (1.244919,\ -0.622459)$;
$\partial L/\partial b = 0.622459$. Interpretation: the model wrongly leans positive, so gradient descent will
decrease $w_1$ (whose input is positive), increase $w_2$ (whose input is negative — note the minus sign) and
decrease $b$.

```python
import torch
x = torch.tensor([2.0, -1.0]); y = torch.tensor(0.0)
w = torch.tensor([0.5, 1.0], requires_grad=True); b = torch.tensor(0.5, requires_grad=True)
z = w @ x + b
L = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
L.backward()
print(round(z.item(), 6), round(torch.sigmoid(z).item(), 6), round(L.item(), 6))
print([round(v, 6) for v in w.grad.tolist()], round(b.grad.item(), 6))
```

**Output:**
```text
0.5 0.622459 0.974077
[1.244919, -0.622459] 0.622459
```
</details>

**Exercise 5 [M ★★].** Let $Y = XW^\top$ with $X\in\mathbb{R}^{B\times n}$ and $L=\sum_{p,o}Y_{po}$ (sum of all
outputs). Use the formula of Section 6.5 to show that every row of $\partial L/\partial W$ equals the vector of
column sums of $X$. Verify on a random example.

<details><summary>Solution</summary>

$G=\partial L/\partial Y$ is the all-ones $B\times m$ matrix. $\partial L/\partial W = G^\top X$; row $o$ of $G^\top$ is
the all-ones vector $\mathbf 1^\top$ of length $B$, and $\mathbf 1^\top X$ is the row vector of column sums of $X$.
So every row is identical: each output neuron's weights receive the same gradient — which also shows why
a symmetric loss plus a symmetric initialisation would keep neurons identical (Section 7.1).

```python
import torch
torch.manual_seed(0)
X = torch.randn(5, 3)
W = torch.randn(4, 3, requires_grad=True)
(X @ W.T).sum().backward()
print(torch.allclose(W.grad, X.sum(0).expand(4, 3)))
```

**Output:**
```text
True
```
</details>

**Exercise 6 [M ★★].** (a) Inputs $x_k$ have variance 1 and there are $n=100$ of them. What weight
variance keeps $\operatorname{Var}(w^\top x)=1$? (b) What is the Xavier-uniform bound $a$ for a weight of shape
$(64, 2092)$ (the drug input layer), and what is PyTorch's default bound for `nn.Linear(2092, 64)`?
(c) Which is larger, and why might the default be preferred for a first layer followed by ELU?

<details><summary>Solution</summary>

(a) $\operatorname{Var}(z)=n\operatorname{Var}(w)\operatorname{Var}(x)$, so $\operatorname{Var}(w)=1/100=0.01$ (std 0.1).
(b) Xavier: $a=\sqrt{6/(2092+64)}=\sqrt{6/2156}=0.05275$. Default: $1/\sqrt{2092}=0.02186$.
(c) Xavier's is larger because it averages fan-in and fan-out, and fan-out (64) is small. The default
depends only on fan-in, which is what controls the forward variance; with 2092 inputs, the more
conservative default keeps pre-activations small so ELU starts in its well-behaved region. Both are
reasonable; the important thing is that neither is "std = 1", which would give pre-activations with
std $\approx\sqrt{2092}\approx46$.

```python
import math
print(round(math.sqrt(6 / (2092 + 64)), 5), round(1 / math.sqrt(2092), 5))
```

**Output:**
```text
0.05275 0.02186
```
</details>

**Exercise 7 [P ★★].** Implement `bce_with_logits(z, y)` in NumPy using the stable formula of Section 8.2,
and compare with PyTorch for $z\in\{-1000,-5,0,5,1000\}$, $y\in\{0,1\}$. Also implement the naive version
`-(y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z)))` and show where it breaks.

<details><summary>Solution</summary>

```python
import numpy as np
import torch
import torch.nn.functional as F

def bce_with_logits(z, y):
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

def naive(z, y):
    p = 1 / (1 + np.exp(-z))
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
with np.errstate(all="ignore"):
    for y in (0.0, 1.0):
        ref = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.full((5,), y),
                                                 reduction="none").numpy()
        print(f"y={y:.0f} stable:", np.round(bce_with_logits(z, y), 4).tolist())
        print(f"    torch :", np.round(ref, 4).tolist())
        print(f"    naive :", np.round(naive(z, y), 4).tolist())
```

**Output:**
```text
y=0 stable: [0.0, 0.0067, 0.6931, 5.0067, 1000.0]
    torch : [0.0, 0.0066999997943639755, 0.6930999755859375, 5.006700038909912, 1000.0]
    naive : [nan, 0.0067, 0.6931, 5.0067, inf]
y=1 stable: [1000.0, 5.0067, 0.6931, 0.0067, 0.0]
    torch : [1000.0, 5.006700038909912, 0.6930999755859375, 0.0066999997943639755, 0.0]
    naive : [inf, 5.0067, 0.6931, 0.0067, nan]
```

The naive version returns `inf` (it computes $\log 0$) whenever the model is confidently wrong, and even
`nan` when it is confidently *right*: for $y=0,\ z=-1000$ it evaluates $0\cdot\log 0 = 0\cdot(-\infty)$,
which is undefined in floating point. `np.log1p(u)` computes $\log(1+u)$ accurately even for tiny $u$.
</details>

**Exercise 8 [P ★★].** Write `masked_softmax(scores, mask, dim)` that returns exact zeros for empty rows
**without ever creating a NaN**, and check that its output and gradient agree with the project's
`masked_fill(-inf) → softmax → nan_to_num` recipe.

<details><summary>Solution</summary>

Replace masked scores by a finite value, subtract the per-row maximum, exponentiate, multiply by the
mask, and divide by the row sum clamped away from zero:

```python
import torch

def masked_softmax(scores, mask, dim):
    s = scores.masked_fill(~mask, 0.0)
    s = s - s.max(dim=dim, keepdim=True).values.detach()   # stability; shift-invariance
    e = torch.exp(s) * mask
    return e / e.sum(dim=dim, keepdim=True).clamp(min=1e-30)

torch.manual_seed(0)
z1 = torch.randn(3, 4, requires_grad=True)
z2 = z1.detach().clone().requires_grad_(True)
mask = torch.tensor([[True, False, True, True], [False] * 4, [True, True, False, False]])
a1 = masked_softmax(z1, mask, 1)
a2 = torch.nan_to_num(torch.softmax(z2.masked_fill(~mask, float("-inf")), 1), nan=0.0)
w = torch.randn(3, 4)
(a1 * w).sum().backward(); (a2 * w).sum().backward()
print("outputs equal:", torch.allclose(a1, a2), "| grads equal:", torch.allclose(z1.grad, z2.grad))
print("any NaN created:", torch.isnan(a1).any().item())
```

**Output:**
```text
outputs equal: True | grads equal: True
any NaN created: False
```

Detaching the maximum is safe because softmax is shift-invariant: the shift's gradient contribution is
exactly zero mathematically.
</details>

**Exercise 9 [P ★].** Without running code first, compute the number of parameters of `ViewAttention(64)`
and of `MVHGAT` with Cdataset sizes (663 drugs, 409 diseases, 3+3 views, `feat_assoc=True`, default
hyper-parameters, 6 propagation views). Then verify.

<details><summary>Solution</summary>

`ViewAttention(64)`: `proj` $64\cdot64+64=4160$, `q` $64$ (no bias): **4224**.
Cdataset input sizes: drugs $3\cdot663+409=2398$, diseases $3\cdot409+663=1890$.
`inp`: $2398\cdot64+64+1890\cdot64+64=153{,}472+64+120{,}960+64=274{,}560$.
Two `HeteroLayer`s with 8 relations: $2\times83{,}584=167{,}168$. Decoder `W`: $192^2=36{,}864$.
Scalars: $6+1+1+4=12$. Total **478,604**.

```python
import sys
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import MVHGAT, ViewAttention
count = lambda m: sum(p.numel() for p in m.parameters())
rels = {f"view:{v}": ("drug", "drug") for v in ("chem_cdk", "chem_ecfp", "gene_r")}
rels.update({f"view:{v}": ("disease", "disease") for v in ("pheno_mim", "sem_mondo", "gene_d")})
rels.update({"assoc>drug": ("drug", "disease"), "assoc>disease": ("disease", "drug")})
print(count(ViewAttention(64)), count(MVHGAT(rels, 3 * 663 + 409, 3 * 409 + 663, n_prop=6)))
```

**Output:**
```text
4224 478604
```
</details>

**Exercise 10 [P ★★].** The following loop has **four** bugs. Find them, explain each, and fix them.

```py
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 16), nn.Linear(16, 1)]
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        return torch.sigmoid(self.layers[1](self.drop(F.relu(self.layers[0](x))))).squeeze(-1)

net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for epoch in range(200):
    loss = F.binary_cross_entropy_with_logits(net(X), y)
    loss.backward()
    opt.step()
scores = net(X_test)
```

<details><summary>Solution</summary>

1. `self.layers` is a plain list → no parameters registered; `net.parameters()` is empty and Adam raises
   *"optimizer got an empty parameter list"*. Use `nn.ModuleList`.
2. `forward` applies `sigmoid`, then the loss applies it again (`..._with_logits`). Return logits.
3. No `opt.zero_grad()` → gradients accumulate.
4. Prediction without `net.eval()` and `torch.no_grad()` → dropout active, graph recorded. (Also call
   `net.train()` in the loop for safety.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

torch.manual_seed(0)
Xn, yn = make_moons(400, noise=0.2, random_state=1)
X = torch.as_tensor(Xn[:300], dtype=torch.float32); y = torch.as_tensor(yn[:300], dtype=torch.float32)
X_test = torch.as_tensor(Xn[300:], dtype=torch.float32); y_test = torch.as_tensor(yn[300:])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 16), nn.Linear(16, 1)])   # fix 1
        self.drop = nn.Dropout(0.2)
    def forward(self, x):                                                    # fix 2: logits
        return self.layers[1](self.drop(F.relu(self.layers[0](x)))).squeeze(-1)

net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for epoch in range(200):
    net.train()
    loss = F.binary_cross_entropy_with_logits(net(X), y)
    opt.zero_grad()                                                          # fix 3
    loss.backward()
    opt.step()
net.eval()                                                                   # fix 4
with torch.no_grad():
    scores = torch.sigmoid(net(X_test))
print("final train loss %.3f, test accuracy %.3f" % (loss.item(), ((scores > 0.5).long() == y_test).float().mean()))
```

**Output:**
```text
final train loss 0.230, test accuracy 0.860
```
</details>

**Exercise 11 [C ★★].** Suppose you replaced every `nn.LayerNorm` in `HeteroLayer` by `nn.BatchNorm1d`.
Describe two concrete problems that would arise in this project.

<details><summary>Solution</summary>

(1) **Train/eval mismatch.** BatchNorm normalises with the statistics of the current "batch" — here, all
593 drugs under this epoch's random hiding of 20% of links and 10% of diseases — and at prediction time
uses running averages. The prediction pass sees the *full* graph (no hidden links), whose activation
statistics differ systematically, so normalisation would be miscalibrated exactly when it matters.
LayerNorm has no running statistics and behaves identically.
(2) **Coupling between nodes.** With BatchNorm, a drug's normalised features depend on every other drug's
features in the same epoch (through the column mean and variance). A few hub nodes with large activations
would shift everyone else's. LayerNorm normalises each node by its own statistics. (A third, minor point:
the disease "batch" is only 313 rows and the two node types would need separate running statistics, which
the `ModuleDict` would provide, but the first two problems remain.)
</details>

**Exercise 12 [M ★★].** (a) Prove $\sigma(\ln x)=x/(1+x)$ for $x>0$. (b) With the initial gate
parameters $(0,1,0,1)$, compute `gnn_gate` for $(\deg_r,\deg_d)\in\{(0,0),(3,6),(22,84)\}$. (c) Why is it sensible
that the gate starts at 0.25 for a cold-start pair rather than 0?

<details><summary>Solution</summary>

(a) $\sigma(\ln x)=1/(1+e^{-\ln x})=1/(1+1/x)=x/(x+1)$.
(b) Each factor is $\sigma(\ln(1+\deg))=(1+\deg)/(2+\deg)$: $(0,0)\to\tfrac12\cdot\tfrac12=0.25$;
$(3,6)\to\tfrac45\cdot\tfrac78=0.7$; $(22,84)\to\tfrac{23}{24}\cdot\tfrac{85}{86}=0.94719$.
(c) A gate of exactly 0 would make the GNN term's gradient zero for cold-start pairs (the product rule
multiplies by the gate), so the model could never learn whether the GNN helps them. Starting at 0.25 lets
training decide, through the learnable $g_0,\dots,g_3$, how much to trust the GNN at low degree.

```python
import sys, torch
sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo.model import MVHGAT
m = MVHGAT({"r": ("drug", "drug")}, 3, 3, n_prop=1)
deg_r = torch.tensor([0.0, 3.0, 22.0]); deg_d = torch.tensor([0.0, 6.0, 84.0])
print([round(v, 5) for v in m.gnn_gate(deg_r, deg_d).diagonal().tolist()])
```

**Output:**
```text
[0.25, 0.7, 0.94719]
```
</details>

**Exercise 13 [P ★★★].** Rebuild the hand-worked network of Section 6.4 from two `nn.Linear` modules by
copying the given weights into them, and confirm the gradients of `fc1.weight` and `fc2.weight` match the
table. (Hint: the weight of `nn.Linear(2, 2)` is stored `(out, in)`.)

<details><summary>Solution</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

fc1, fc2 = nn.Linear(2, 2), nn.Linear(2, 1)
with torch.no_grad():                         # writing into parameters must not be recorded
    fc1.weight.copy_(torch.tensor([[0.5, 0.25], [-1.0, 0.25]])); fc1.bias.copy_(torch.tensor([0.0, 0.25]))
    fc2.weight.copy_(torch.tensor([[1.5, -2.0]]));             fc2.bias.copy_(torch.tensor([-0.5]))
x = torch.tensor([[1.0, 2.0]])                # a batch of ONE example: shape (1, 2)
logit = fc2(F.relu(fc1(x))).squeeze()
loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(1.0))
loss.backward()
print(round(loss.item(), 6))
print(fc1.weight.grad)
print(fc2.weight.grad, fc2.bias.grad)
```

**Output:**
```text
0.313262
tensor([[-0.4034, -0.8068],
        [ 0.0000,  0.0000]])
tensor([[-0.2689,  0.0000]]) tensor([-0.2689])
```

The numbers match the table of Section 6.4 (printed with PyTorch's default 4 decimals).
</details>

**Exercise 14 [C ★★★].** In `fit_predict`, the loss is computed only on `sup` (links hidden this epoch)
and sampled negatives. Using what you know about the computational graph, explain which parameters get
gradient from a single hidden positive pair $(i,j)$, and why an *unsampled* pair contributes nothing.

<details><summary>Solution</summary>

The loss is a sum over the selected logits; an unselected logit does not appear in the graph's path to
the loss, so $\partial L/\partial S_{kl}=0$ for it (the matrix $G$ of Section 6.5 is zero there). For the pair
$(i,j)$: the decoder gives gradient to $W$ (via $H_r^\top G H_d$), to the embedding rows $h_i$ and $h_j$,
to the propagation weights $w_v$ (via $P_v[i,j]$), the bias, and the gate parameters (via $\deg_i$,
$\deg_j$). Then $h_i$ and $h_j$ pass gradient back through the two `HeteroLayer`s: to the GAT and view
attention parameters of every relation used, to the embeddings of $i$'s and $j$'s neighbours (1 hop) and
their neighbours (2 hops), and finally to the input `nn.Linear` weights through the features of every node
in that 2-hop receptive field. Parameters shared across nodes (all of them — weights are not per-node)
thus receive the sum of contributions from all sampled pairs.
</details>

---

## 19. Answers to the self-check questions (PREREQUISITES.md, Unit B3)

### Q1. Why is `torch.no_grad()` used at prediction time?

**Short answer.** Because at prediction time we never call `backward()`, so recording the computation
for autograd is pure waste — of memory, of time — and it also leaves the outputs attached to a graph,
which gets in the way of converting them to NumPy.

**In depth.**

1. **What autograd does by default.** Whenever an operation involves a tensor with
   `requires_grad=True` (every model parameter), PyTorch records a `grad_fn` node and **saves the inputs
   that the backward formula will need**: the attention matrices for the softmax backward, the
   pre-activations for ELU and LeakyReLU, the inputs of every matmul, the normalisation statistics of every
   LayerNorm, the dropout masks... (Sections 6.3 and 10.1).
2. **Memory.** In MV-HGAT one forward pass stores, per layer and per relation, several $(D\times S\times H)$
   float32 tensors (e.g. $593\cdot593\cdot4\cdot4$ bytes $\approx5.6$ MB each for a drug–drug relation), plus
   all the $(N\times64)$ activations. Under `no_grad()` each intermediate is freed as soon as the next
   operation has consumed it. On a 6 GB laptop GPU this is the difference between comfortably predicting
   and running out of memory on larger graphs.
3. **Time.** Building the graph and saving tensors costs time; skipping it makes prediction faster.
4. **Clean outputs.** Results computed under `no_grad` have `requires_grad=False` and no `grad_fn`, so
   `torch.sigmoid(logits).cpu().numpy()` works directly; outside `no_grad`, `.numpy()` would raise
   *"Can't call numpy() on Tensor that requires grad"* and you would need `.detach()`.
5. **Safety.** Nothing computed during prediction can accidentally add into the parameters' `.grad`
   buffers or be part of a later `backward()`.
6. **What it does *not* do.** It does not switch off dropout and does not change normalisation layers.
   That is `model.eval()`'s job. Prediction needs both, which is exactly what the end of `fit_predict`
   does (`model.eval()` then `with torch.no_grad():`). The explanation methods `view_attention`,
   `view_weights` and `occlusion` use the decorator form `@torch.no_grad()` for the same reasons — they
   run the model dozens of times (occlusion runs it once per evidence group).
7. **When you would *not* use it at prediction time:** if you want gradients with respect to the
   *inputs* (saliency/attribution methods), you keep recording on, typically in `eval()` mode.

```python
import torch

lin = torch.nn.Linear(4, 1)
x = torch.randn(3, 4)
try:
    torch.sigmoid(lin(x)).numpy()
except RuntimeError as err:
    print("without no_grad:", str(err).split(".")[0])
with torch.no_grad():
    print("with no_grad   :", torch.sigmoid(lin(x)).numpy().shape)
```

**Output:**
```text
without no_grad: Can't call numpy() on Tensor that requires grad
with no_grad   : (3, 1)
```

### Q2. What does `masked_fill(~mask, -inf)` followed by softmax achieve?

**Short answer.** It computes a softmax **restricted to the allowed entries** (the neighbours of a node,
or the relations in which a node has neighbours): disallowed entries get weight exactly 0, and the allowed
weights are renormalised to sum to 1 among themselves.

**In depth.**

1. **Mechanics.** `mask` is a boolean tensor, `True` where an entry is allowed (in `DenseGAT`, the
   adjacency `graphs[rel]`, $\text{mask}[i,j]=$ "$j$ is a neighbour of $i$"). `~mask` flips it;
   `masked_fill(~mask, -inf)` writes $-\infty$ into every disallowed score. Because $e^{-\infty}=0$,
   $$
   \alpha_{ij}=\frac{\mathbb 1[j\in\mathcal N(i)]\,e^{e_{ij}}}{\sum_{k\in\mathcal N(i)}e^{e_{ik}}} .
   $$
   This is exactly GAT's "softmax over the neighbourhood $\mathcal N(i)$" (Unit C4), computed for all
   nodes in parallel on a dense $(D,S,H)$ score tensor. In `ViewAttention` the same recipe over `dim=0`
   restricts the view softmax to relations in which the node actually has neighbours.
2. **Why before, not after.** Masking the softmax *output* would leave weights that do not sum to 1 and
   let non-neighbours influence the normaliser (Section 5.4).
3. **Why $-\infty$ rather than "a very negative number".** With $-\infty$ the masked weights are exactly
   0, not merely tiny, and gradients to masked scores are exactly 0. More subtly, the two choices differ on
   **empty rows**: with $-10^9$, a node with no neighbours would get a *uniform* distribution over all
   non-neighbours (all entries equal, shift invariance), silently aggregating messages from nodes it is not
   connected to; with $-\infty$ the row becomes NaN, which the project then deliberately converts to all
   zeros with `nan_to_num`, giving a zero message and `mask.any(1) = False` so the relation is ignored by
   the view attention.
4. **Gradients stay finite.** Although the forward pass of an empty row produces NaN before
   `nan_to_num`, every score in that row was overwritten by `masked_fill`, whose backward sends 0 to
   overwritten positions — so no NaN reaches the parameters (verified in Section 5.4).
5. **Per head.** `mask[..., None]` broadcasts the same adjacency to all 4 heads; each head then normalises
   its own scores independently.
6. **What it buys the project.** It turns a dense all-pairs computation into *graph* attention: the
   kNN sparsification of each similarity view (Unit C1) and the training-fold-only association edges
   are enforced exactly through this mask, which is also how hidden links are really hidden from message
   passing each epoch (`graphs["assoc>drug"] = Am`).

```python
import torch

scores = torch.tensor([[0.3, 2.0, -1.0],      # node with neighbours 0 and 2
                       [0.5, 0.1, 0.9]])      # node with NO neighbours
mask = torch.tensor([[True, False, True], [False, False, False]])
for fill in (-1e9, float("-inf")):
    att = torch.softmax(scores.masked_fill(~mask, fill), dim=1)
    print(f"fill={fill}:", [[round(v, 4) for v in row] for row in torch.nan_to_num(att, nan=0.0).tolist()])
```

**Output:**
```text
fill=-1000000000.0: [[0.7858, 0.0, 0.2142], [0.3333, 0.3333, 0.3333]]
fill=-inf: [[0.7858, 0.0, 0.2142], [0.0, 0.0, 0.0]]
```

With $-10^9$ the empty row silently becomes $(1/3,1/3,1/3)$ — attention to three non-neighbours. With
$-\infty$ (+ `nan_to_num`) it is correctly all zero. The first row is the same either way:
$0.7858 = e^{0.3}/(e^{0.3}+e^{-1})$ and $0.2142$ is its complement.

---

## 20. Summary and cheat sheet

**Big picture.** A neural network is a differentiable program with parameters. Training = run it forward
to get a loss, run backpropagation to get the gradient of the loss w.r.t. every parameter, update with an
optimiser, repeat. PyTorch provides tensors (on CPU/GPU), autograd (automatic backprop), modules (to
organise parameters) and optimisers.

**Formulas**

| Item | Formula |
|---|---|
| Linear layer (batched) | $Y = XW^\top + b$, $W$ is (out, in) |
| Linear layer gradients | $\partial L/\partial W = G^\top X$, $\partial L/\partial X = GW$, $\partial L/\partial b=G^\top\mathbf 1$ |
| Bilinear decoder gradients | $S=H_rWH_d^\top$: $\partial_W=H_r^\top GH_d$, $\partial_{H_r}=GH_dW^\top$, $\partial_{H_d}=G^\top H_rW$ |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$, $\sigma'=\sigma(1-\sigma)\le\tfrac14$ |
| tanh | $\tanh'=1-\tanh^2$, $\tanh(z)=2\sigma(2z)-1$ |
| ReLU / LeakyReLU | slope 1 / ($0$ or $\alpha$) |
| ELU | $\alpha(e^z-1)$ for $z\le0$, derivative $\alpha e^z$ |
| Softplus | $\log(1+e^z)$, derivative $\sigma(z)$ |
| Softmax Jacobian | $\partial s_i/\partial z_j = s_i(\delta_{ij}-s_j)$ |
| Softmax/sigmoid + CE gradient | $p-y$ |
| BCE with logits (stable) | $\max(z,0)-zy+\log(1+e^{-\lvert z\rvert})$ |
| Log-sum-exp | $m+\log\sum_j e^{z_j-m}$, $m=\max_j z_j$ |
| Xavier uniform | $a=\sqrt{6/(n_{in}+n_{out})}$; He: $\operatorname{Var}=2/n_{in}$ |
| Inverted dropout | $\tilde h=m\odot h/(1-p)$, identity at eval |
| LayerNorm | normalise each row over its features, then $\gamma,\beta$ |
| Residual | $h'=h+F(h)$, Jacobian $I+\partial F/\partial h$ |
| Adam | $\theta\leftarrow\theta-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ |

**Code idioms**

```py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.as_tensor(np_array, dtype=torch.float32, device=DEVICE)
model = Model(...).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=5e-4)
for epoch in range(E):
    model.train()
    loss = F.binary_cross_entropy_with_logits(model(x)[idx], y)
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(x)).cpu().numpy()
att = torch.nan_to_num(torch.softmax(e.masked_fill(~mask, float("-inf")), dim=1), nan=0.0)
e = s_dst[:, None, :] + s_src[None, :, :]          # all pairs by broadcasting
out = torch.einsum("dsh,shk->dhk", att, zs)         # attention-weighted sum per head
```

**Ten things to remember**
1. Non-linearity is what makes depth useful.
2. Backprop = chain rule in reverse order; gradients add at fan-out.
3. A gradient has the shape of what it differentiates.
4. Return logits; use `..._with_logits` losses.
5. Mask scores with $-\infty$ *before* softmax; handle empty rows.
6. `zero_grad` → `backward` → `step`, every iteration.
7. `eval()` ≠ `no_grad()`; prediction needs both.
8. Register submodules with `ModuleList`/`ModuleDict`; initialise raw `Parameter`s.
9. Same device, same dtype, for every operand.
10. Seed everything; report mean ± std across seeds.

---

## 21. Further resources (curated; all links checked)

**Courses and video series**

* **Andrej Karpathy — *Neural Networks: Zero to Hero*** — <https://karpathy.ai/zero-to-hero.html> —
  *Free.* The best way to *feel* backprop: you build an autograd engine (micrograd) and then a language
  model from scratch. Start with "The spelled-out intro to neural networks and backpropagation: building
  micrograd" (<https://www.youtube.com/watch?v=VMj-3S1tku0>); "Building makemore Part 3: Activations &
  Gradients, BatchNorm" (<https://www.youtube.com/watch?v=P6sfmUTpUmc>) is the best treatment of
  initialisation and normalisation in practice.
* **Stanford CS231n** — <https://cs231n.stanford.edu/> — *Free.* The course notes on backpropagation
  (<https://cs231n.github.io/optimization-2/>) and on network architecture and setup
  (<https://cs231n.github.io/neural-networks-1/>, <https://cs231n.github.io/neural-networks-2/>) are the
  clearest written treatment of computational graphs, activations, initialisation and data preprocessing.
* **MIT 6.S191 Introduction to Deep Learning** — <http://introtodeeplearning.com/> — *Free.* Compact,
  well-produced lecture series with hands-on software labs; good for a fast first pass.
* **3Blue1Brown — Neural networks** — <https://www.3blue1brown.com/topics/neural-networks> — *Free.* Visual
  intuition; chapters "But what is a neural network?" (<https://www.youtube.com/watch?v=aircAruvnKk>),
  "Backpropagation, intuitively" (<https://www.youtube.com/watch?v=Ilg3gGewQ5U>) and "Backpropagation
  calculus" (<https://www.youtube.com/watch?v=tIeHLnjs5U8>).

**Books**

* **Zhang, Lipton, Li & Smola — *Dive into Deep Learning*** — <https://d2l.ai/> — *Free.* Runnable PyTorch
  notebooks for everything in this unit (chapters on preliminaries/autograd, linear networks, MLPs,
  builders' guide, numerical stability and initialisation).
* **Goodfellow, Bengio & Courville — *Deep Learning*** — <https://www.deeplearningbook.org/> — *Free
  online (print version paid).* The rigorous reference: chapter 6 (feedforward networks, back-propagation,
  universal approximation), chapter 7 (regularisation, dropout), chapter 8 (optimisation, initialisation,
  batch normalisation).
* **Simon Prince — *Understanding Deep Learning*** — <https://udlbook.github.io/udlbook/> — *Free PDF
  (print paid).* Modern, beautifully illustrated; chapters 3–7 (shallow/deep networks, loss functions,
  fitting, gradients and initialisation) and 11 (residual networks) match this unit closely.
* **Michael Nielsen — *Neural Networks and Deep Learning*** — <http://neuralnetworksanddeeplearning.com/> —
  *Free.* Chapter 2 (<http://neuralnetworksanddeeplearning.com/chap2.html>) is a gentle full derivation of
  backprop; chapter 4 (<http://neuralnetworksanddeeplearning.com/chap4.html>) is a visual proof of
  universal approximation.
* **François Fleuret — *The Little Book of Deep Learning*** — <https://fleuret.org/francois/lbdl.html> —
  *Free.* A 170-page phone-sized summary; ideal for revision.

**Official PyTorch documentation**

* *Learn the Basics* — <https://pytorch.org/tutorials/beginner/basics/intro.html> — *Free.* Tensors,
  autograd, building models, optimisation loop, save/load.
* *Automatic differentiation with torch.autograd* —
  <https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html>; and *Autograd mechanics* —
  <https://docs.pytorch.org/docs/stable/notes/autograd.html> (how the graph, `no_grad` and in-place checks
  work). *Free.*
* *What is torch.nn really?* — <https://pytorch.org/tutorials/beginner/nn_tutorial.html> — *Free.* Refactors
  a hand-written loop into `nn.Module`/`optim` step by step; excellent for understanding what the
  abstractions do.
* *Broadcasting semantics* — <https://docs.pytorch.org/docs/stable/notes/broadcasting.html>;
  *Reproducibility* — <https://docs.pytorch.org/docs/stable/notes/randomness.html>;
  `BCEWithLogitsLoss` — <https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>;
  `torch.nn.init` — <https://docs.pytorch.org/docs/stable/nn.init.html>;
  `LayerNorm` — <https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>. *Free.*

**Short readings on backprop and matrix calculus**

* Christopher Olah, "Calculus on Computational Graphs: Backpropagation" —
  <https://colah.github.io/posts/2015-08-Backprop/> — *Free.* Why reverse mode is efficient, in 10 minutes.
* Parr & Howard, "The Matrix Calculus You Need For Deep Learning" — <https://explained.ai/matrix-calculus/>
  — *Free.* Jacobians and the vector chain rule, from scratch.
* Roger Grosse, CSC321 lecture notes on backpropagation —
  <https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L06%20Backpropagation.pdf> — *Free.*
  A rigorous university-level derivation with worked examples.
* Baydin et al., "Automatic differentiation in machine learning: a survey" —
  <https://arxiv.org/abs/1502.05767> — *Free.* Forward vs reverse mode in depth.
* Andrej Karpathy, "A Recipe for Training Neural Networks" — <https://karpathy.github.io/2019/04/25/recipe/>
  — *Free.* Practical debugging discipline (overfit a batch, check the initial loss, etc.).

**Original papers (all free)**

* Glorot & Bengio (2010), Xavier initialisation — <https://proceedings.mlr.press/v9/glorot10a.html>
* He et al. (2015), He/Kaiming initialisation and PReLU — <https://arxiv.org/abs/1502.01852>
* He et al. (2016), Residual networks — <https://arxiv.org/abs/1512.03385>
* Srivastava et al. (2014), Dropout — <https://jmlr.org/papers/v15/srivastava14a.html>
* Ioffe & Szegedy (2015), Batch normalisation — <https://arxiv.org/abs/1502.03167>
* Ba, Kiros & Hinton (2016), Layer normalisation — <https://arxiv.org/abs/1607.06450>
* Clevert et al. (2015), ELU — <https://arxiv.org/abs/1511.07289>
* Kingma & Ba (2015), Adam — <https://arxiv.org/abs/1412.6980>
* Loshchilov & Hutter (2019), AdamW / decoupled weight decay — <https://arxiv.org/abs/1711.05101>
* Veličković et al. (2018), Graph Attention Networks (the model `DenseGAT` implements) —
  <https://arxiv.org/abs/1710.10903>

---

## 22. Glossary

| Term | Meaning |
|---|---|
| **Activation (function)** | Element-wise non-linearity $\phi$ applied after a linear map; also the output $a=\phi(z)$. |
| **Adam** | Optimiser with per-parameter adaptive step sizes from running averages of gradients and squared gradients. |
| **Autograd** | PyTorch's automatic differentiation engine: records operations and runs backprop. |
| **Backpropagation** | Computing gradients of a scalar loss w.r.t. all parameters by applying the chain rule in reverse topological order. |
| **BatchNorm** | Normalises each feature over the batch; uses running statistics at evaluation. |
| **BCE / BCE with logits** | Binary cross-entropy; the "with logits" form takes raw scores and is numerically stable. |
| **Bias** | The additive constant $b$ of a neuron or layer. |
| **Broadcasting** | Automatic expansion of size-1 dimensions so tensors of different shapes can be combined element-wise. |
| **Computational graph** | DAG of operations that computes the loss from inputs and parameters. |
| **Contiguous** | A tensor whose elements are laid out in memory in row-major order; required by `.view`. |
| **Dead ReLU** | A ReLU unit that outputs 0 for every input and so never receives gradient. |
| **Define-by-run** | Building the computational graph dynamically as Python code executes. |
| **Device** | Where a tensor lives and is computed on: `cpu` or `cuda`. |
| **Dropout** | Randomly zeroing activations during training (with rescaling) as a regulariser. |
| **dtype** | Element type of a tensor (`float32`, `int64`, `bool`, ...). |
| **einsum** | Index-notation tensor contraction (`"dsh,shk->dhk"`). |
| **Embedding** | A learned vector representation of an entity (drug, disease) — a hidden layer's output. |
| **Epoch** | One pass over the training data (one step in full-batch training). |
| **Fan-in / fan-out** | Number of inputs / outputs of a layer; used in initialisation formulas. |
| **Finite differences** | Numerical approximation of derivatives used for gradient checking. |
| **Forward pass** | Evaluating the network from inputs to loss. |
| **Gradient** | Vector of partial derivatives of the loss w.r.t. parameters. |
| **Head (attention)** | One of several independent attention computations whose outputs are concatenated. |
| **Hidden layer** | Any layer between input and output. |
| **Inverted dropout** | Dropout that rescales survivors by $1/(1-p)$ at training time so evaluation needs no change. |
| **Jumping knowledge** | Concatenating the outputs of all layers to form the final representation. |
| **LayerNorm** | Normalises each example/node over its own features; same behaviour in train and eval. |
| **Leaf tensor** | A tensor created by the user (not by an operation); parameters are leaves and receive `.grad`. |
| **Log-sum-exp (LSE)** | $\log\sum e^{z_j}$, computed stably by subtracting the maximum. |
| **Logit** | A raw real-valued score before sigmoid/softmax. |
| **Masked softmax** | Softmax restricted to allowed entries by setting others to $-\infty$ first. |
| **MLP** | Multi-layer perceptron: alternating linear layers and activations. |
| **`nn.Module`** | Base class for PyTorch models; registers parameters and submodules. |
| **`nn.Parameter`** | A tensor registered as a learnable parameter of a module. |
| **`no_grad`** | Context/decorator that disables autograd recording. |
| **Over-smoothing** | GNN node embeddings becoming indistinguishable as layers are stacked (Unit C3). |
| **Perceptron** | A single neuron with a step activation and Rosenblatt's learning rule. |
| **Pre-activation** | $z=w^\top x+b$, before the activation is applied. |
| **PRNG / seed** | Pseudo-random number generator / the value that fixes its sequence. |
| **Residual (skip) connection** | Adding a block's input (or a projection of it) to its output. |
| **Saturation** | Region where an activation is flat, so its derivative is near zero. |
| **Softmax** | Maps a score vector to a probability vector: $e^{z_i}/\sum_j e^{z_j}$. |
| **Softplus** | $\log(1+e^z)$; smooth, positive; derivative is the sigmoid. |
| **`state_dict`** | Dictionary of a module's parameters and buffers, used for saving/loading. |
| **Tensor** | PyTorch's n-dimensional array with dtype, device and optional gradient tracking. |
| **`train()` / `eval()`** | Module modes that switch dropout/BatchNorm behaviour. |
| **Universal approximation theorem** | One hidden layer with a non-polynomial activation can approximate any continuous function on a compact set. |
| **Vanishing / exploding gradients** | Gradients shrinking / growing exponentially with depth. |
| **Vector–Jacobian product (VJP)** | $J^\top v$; the primitive each operation implements for reverse-mode autodiff. |
| **Weight decay** | Penalising large weights (L2 in Adam's gradient, or decoupled in AdamW). |
| **Xavier / Glorot initialisation** | Random weights with variance $2/(n_{in}+n_{out})$ to keep signal scale stable. |
