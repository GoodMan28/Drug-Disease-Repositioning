# Unit E2: Interpretability and its limits

*Track E (Research practice), unit 2 of 3. Part of the self-contained course that goes with the project
"Heterogeneous Graph-Based Computational Drug Repositioning for Disease-Drug Association Prediction"
(MV-HGAT).*

---

## 0. Before you start

**Prerequisites**

| You need | Where it is taught | Why you need it here |
|---|---|---|
| Partial derivatives, the chain rule, integrals of one variable | any calculus course; unit B1 | gradients, integrated gradients |
| Logistic regression, logits vs probabilities, the sigmoid | unit B1 | almost every explanation can be computed on the logit *or* on the probability, and the two disagree |
| PyTorch tensors and autograd (`.backward()`, `.grad`) | unit B3 | all gradient-based code in this chapter |
| Graph attention (GAT), softmax attention, heterogeneous graphs and HAN-style "semantic" attention | units C3 to C6 | the project's view attention β |
| The MV-HGAT model at the level of `docs/HOW_IT_WORKS.md` sections 3 and 4 | that document | the "In this project" section |
| Leakage, ablations, repeated runs | unit E1 | ablation is an interpretability tool too |

**Estimated study time:** 14 to 18 hours. Roughly 6 h reading sections 1 to 10 carefully, 3 h running and
modifying the code, 5 h on the exercises, 1 to 2 h reading one or two of the primary papers in section 16
(start with Jain & Wallace 2019 and Wiegreffe & Pinter 2019).

**Learning objectives.** When you finish this unit you should be able to:

1. **Classify** any explanation method along four axes (intrinsic vs post-hoc, global vs local,
   model-specific vs model-agnostic, faithful vs merely plausible) and place the project's three tools
   (global view weights $w_v$, per-node view attention $\beta$, occlusion) on those axes.
2. **Compute by hand** gradients, gradient × input, integrated gradients, exact Shapley values and occlusion
   scores for a function of three inputs, and **state and check** the completeness/efficiency property.
3. **Derive** the completeness axiom of integrated gradients from the fundamental theorem of calculus, and
   write down the Shapley value formula together with its four axioms.
4. **Implement from scratch** (NumPy/PyTorch) gradient × input, integrated gradients, exact Shapley values,
   KernelSHAP, LIME, occlusion and a GNNExplainer-style edge mask, and evaluate an explanation with
   fidelity+ / fidelity- / sparsity.
5. **Construct** a model in which attention weights and occlusion disagree, and **explain** at least five
   distinct reasons why a relation can receive high attention but have low occlusion impact (and the
   reverse), including the reasons specific to MV-HGAT.
6. **Summarise** the Jain & Wallace (2019) vs Wiegreffe & Pinter (2019) debate in a paragraph that a
   reviewer would accept as accurate.
7. **Prove** that in an additive scoring head, $w_v P_v$ is simultaneously the exact occlusion effect and the
   exact Shapley value of view $v$ on the logit, and list what this does *not* tell you.
8. **Write** an explanation of a single prediction for a biologist that is faithful to the model, uses no
   causal language the evidence cannot support, and names the next experimental step.
9. **Argue** precisely why a faithful model explanation is still not biological evidence.

---

## 1. Motivation: "why did the model say amantadine?"

Run `scripts/06_case_study.py --dataset C` and the first row it prints for Alzheimer disease is
(from `results/logs/C_case.log`):

```
 rank drugbank_id   drug        score  CTD_curated known_in_Fdataset clinical_trials  top_evidence
    1     DB00915   Amantadine  0.9016           -                 -              49  chem_ecfp (+0.179), chem_cdk (+0.147)
```

Amantadine is an old antiviral that is also used in Parkinson's disease. It is **not** a known
Alzheimer's indication in the Cdataset benchmark, yet the model gives it a score of 0.90, the highest of
all 663 drugs. Imagine you show this to a pharmacologist. The very first thing she will ask is *why*.
And the answer matters a great deal:

* If the answer is "because amantadine's closest chemical relative, **memantine**, is an approved
  Alzheimer's drug", she can judge that immediately. Memantine is literally a dimethyl derivative of
  amantadine, and both block NMDA receptors. The prediction becomes a sensible, checkable hypothesis.
  (We verify in section 11.4 that this is indeed what the model's evidence says.)
* If the answer were "because Alzheimer's disease has many links in the training data and amantadine is a
  popular drug", she would rightly dismiss the prediction as an artefact of popularity.
* If the answer were "the attention weights of the disease node are 0.42 on known links, 0.21 on
  phenotype similarity and 0.19 on ontology similarity", she would not know what to do with it. Worse, as
  we will see, those numbers do not even say which evidence *produced* this particular score.

So an explanation can do several different jobs, and the project relies on each of them:

| Job | Example in this project |
|---|---|
| **Building trust / triage** | A biologist decides which of the top-10 drugs is worth a literature search or a wet-lab assay. |
| **Debugging and leakage detection** | If occlusion showed that predictions were driven by an evidence source that is secretly derived from the answer (e.g. CTD "therapeutic" gene-disease links, deliberately excluded, `HOW_IT_WORKS.md` section 6, assumption 8), the explanation would expose the leak. |
| **Scientific insight** | "MONDO ontology similarity complements phenotype similarity" is a claim *about the data* that the explanations, combined with ablations, support. |
| **Communication and accountability** | A paper that claims "interpretability over similarity views" (Objective 3 of the proposal) must show *what* the model used and *how we know*. |
| **Hypothesis generation** | A neighbour-level explanation ("because memantine...") turns a number into a mechanistic hypothesis a biologist can test. |

In biomedicine these jobs are unusually important, because (a) predictions are used to spend expensive
lab and clinical resources, (b) the training labels are biased (well-studied drugs and diseases have more
links), so models readily learn shortcuts, and (c) the end users are domain experts who can catch nonsense
*if* they are shown the reasoning.

But this unit has a second, equally important message, which is in its title: **interpretability has
limits**. Different explanation methods can disagree about the same prediction; attention weights are
easily mistaken for explanations; and even a perfectly faithful explanation of the model tells you about
the model and the data it was trained on, not about biology. The proposal's interpretability objective is
only credible if you can state these limits precisely. That is what the rest of this chapter teaches.

---

## 2. A vocabulary for explanations

### 2.1 What exactly is being explained?

The word "explanation" hides three different targets. Keep them apart in your head and in your writing:

```
   THE WORLD (biology)          THE DATA                     THE MODEL
   amantadine blocks NMDA  -->  benchmark says memantine --> MV-HGAT scores amantadine
   receptors; does that         treats AD; memantine is      0.90 for AD because its
   help AD patients?            chemically ~ amantadine      chem views carry memantine
         ^                             ^                            ^
   needs experiments,          needs statistics,             needs explanation methods
   trials                      curation checks               (this unit)
```

An **explanation method** (attribution, attention, occlusion, ...) speaks only about the right-hand box.
It can, at best, tell you faithfully what the model computed. Whether the model's reasoning reflects real
regularities in the data (middle box) is a question of evaluation and of data quality; whether those
regularities reflect biology (left box) is a question for experiments. Section 10 returns to this.

### 2.2 Intrinsic versus post-hoc interpretability

* An **intrinsically interpretable** (or "interpretable by design", "glass-box") model is one whose
  structure itself is the explanation: a short decision list, a sparse linear model, a generalised additive
  model (GAM, a sum of one-variable functions), a scorecard. You do not need a separate method; you read
  the parameters.
* A **post-hoc** explanation is produced *after* training, by a separate procedure that probes a model
  which was not designed to be read: gradients, SHAP, LIME, occlusion, GNNExplainer.

Lipton (2016, "The Mythos of Model Interpretability") splits intrinsic interpretability further into
*simulatability* (a person can run the model in their head), *decomposability* (each part, input and
parameter has an intuitive meaning) and *algorithmic transparency* (we understand the learning algorithm).

Rudin (2019, "Stop explaining black box machine learning models for high stakes decisions and use
interpretable models instead") makes a forceful argument: post-hoc explanations of black boxes are
necessarily approximations, can be wrong in ways that are hard to detect, and in many high-stakes problems
with meaningful features an interpretable model is about as accurate as the black box. Her recommendation
is to prefer intrinsically interpretable models whenever possible.

MV-HGAT is a hybrid, and that is a deliberate design choice:

```
logit(i,j) =  gate(i,j) * h_i^T W h_j           <- black-box part (GNN): needs post-hoc tools
            + sum_v  w_v * P_v[i,j]  +  b       <- glass-box part (additive head): read it directly
```

The additive propagation head is intrinsically interpretable; the GNN term is not. Section 7 makes this
precise.

### 2.3 Global versus local

* A **global** explanation describes the model's behaviour over the whole input space or a population
  ("overall, the model weighs the phenotype view most"). Examples: the coefficients of a linear model,
  permutation feature importance, mean |SHAP| per feature, the project's $w_v$, the dataset-average view
  attention in `results/case_studies/Cdataset_disease_view_attention.csv`.
* A **local** explanation describes one prediction ("*this* score of 0.90 for amantadine and AD comes
  mostly from the chemical views"). Examples: the SHAP values of one input, a GNNExplainer subgraph for one
  edge, the project's occlusion drops for one drug-disease pair.

There is also an intermediate level: explanations for one *entity* (one node), like the project's
per-node β, which describe how a given disease aggregates its neighbours for all of its predictions.

A global explanation can be badly misleading about a particular prediction (the average hides the
variation), and a pile of local explanations does not automatically give a coherent global picture.
Report both and label which is which.

### 2.4 Model-specific versus model-agnostic

* **Model-agnostic** methods only need to *query* the model: LIME, KernelSHAP, occlusion/perturbation,
  counterfactual search. They work for any model but can be expensive and they depend on how you define a
  "removed" input.
* **Model-specific** methods use the model's internals: gradients (need differentiability), attention
  weights (need an attention layer), TreeSHAP (needs trees), GNNExplainer (needs a message-passing model
  whose edges can be masked), and the additive head decomposition (needs that head).

### 2.5 Faithfulness versus plausibility

This is the most important distinction in the unit. Following Jacovi & Goldberg (2020):

* **Faithfulness** (fidelity): how accurately the explanation reflects *the actual reasoning process of
  the model*. A faithful explanation of a silly model is silly.
* **Plausibility** (persuasiveness): how convincing the explanation looks *to a human*. A plausible
  explanation can be completely unfaithful: it tells a pleasing story the model never followed.

The danger in biomedicine is that we are expert at generating plausible stories. Show a pharmacologist any
list of "important features" and she will find a mechanism that fits. A plausible but unfaithful
explanation is therefore worse than none: it adds false confidence. Faithfulness must be *tested*, never
assumed (section 6.6), and plausibility must never be used as evidence of faithfulness.

Other desirable properties you will meet:

| Property | Meaning | How you would check it |
|---|---|---|
| Completeness / efficiency | attributions add up to the output (minus a reference output) | sum them (sections 3.4, 3.5) |
| Sparsity / compactness | few elements carry the explanation | count non-negligible attributions; `sparsity` (section 6.6) |
| Stability / robustness | similar inputs, or re-trained models with another seed, give similar explanations | rank correlation across seeds |
| Contrastivity | the explanation answers "why A *rather than* B?" | counterfactual methods (section 8) |
| Sensitivity to the model | if you randomise the model, the explanation changes | the model-randomisation sanity check of Adebayo et al. (2018) |

### 2.6 How to evaluate interpretability (Doshi-Velez & Kim 2017)

Doshi-Velez and Kim's "Towards a rigorous science of interpretable machine learning" proposes three
levels of evaluation:

1. **Application-grounded**: real experts doing the real task (e.g. pharmacologists using explanations to
   choose which predictions to test, measuring how many of their picks are later confirmed). Best evidence,
   most expensive.
2. **Human-grounded**: lay people on simplified tasks (e.g. "which of these two explanations lets you
   predict what the model will output?").
3. **Functionally-grounded**: no humans; a formal proxy such as fidelity, sparsity or agreement with a
   known ground truth in synthetic data.

A research project like ours can realistically do functionally-grounded evaluation (fidelity of occlusion,
agreement between methods, stability over seeds) and a light form of application-grounded evaluation
(the case-study tables, judged against CTD and ClinicalTrials.gov). Say so explicitly in the paper.

### 2.7 The units of an explanation

Explanations can be phrased in terms of different units. Choose the unit that the user can act on.

| Unit | Example | Methods |
|---|---|---|
| Input features | "column 37 of the drug feature vector" | gradients, IG, SHAP, LIME |
| Groups of features / evidence sources | "the ECFP chemical view" | grouped occlusion, grouped SHAP, the additive head |
| Edges / relations | "the edge memantine -> amantadine in the chemical kNN graph" | GNNExplainer, PGExplainer, edge occlusion |
| Nodes / subgraphs | "the 3-node subgraph {memantine, amantadine, AD}" | SubgraphX, GNNExplainer node masks |
| Training examples | "the known link memantine-AD" | influence functions (Koh & Liang 2017), the neighbour decomposition of $P_v$ |
| Counterfactuals | "if memantine-AD were unknown, amantadine would drop to rank 40" | counterfactual search |

For biologists the most actionable units are usually **evidence sources** and **training examples**
("which known treatments made you say this?"). The project's case-study tables use evidence sources; section
11.4 shows how to go one level further to training examples.

---

## 3. Feature attribution from first principles

### 3.1 The setting

Let $f:\mathbb{R}^d\to\mathbb{R}$ be a trained model, evaluated at an input $x\in\mathbb{R}^d$. A
**feature attribution** is a vector $a(x)\in\mathbb{R}^d$ whose $i$-th entry says how much feature $i$ is
"responsible" for $f(x)$. Two choices must be made before any number is meaningful:

1. **Which output?** For a classifier, $f$ can be the logit $z$ or the probability $p=\sigma(z)$. Because
   $\sigma$ is flat near 0 and 1, the same logit contribution produces very different probability changes
   depending on where you start (section 7.4 computes this). The project's occlusion uses probabilities;
   its additive head is naturally read on logits.
2. **Responsible relative to what?** Almost every attribution compares $x$ with a **reference** or
   **baseline** input $x'$ that represents "absence of the features" (all zeros, a mean input, a blurred
   image, an empty graph). The attribution then explains $f(x)-f(x')$, not $f(x)$. A different baseline
   is a different question.

We will use three running examples small enough to do by hand:

* $f_1(x)=2x_1+x_2x_3$ at $x=(1,1,1)$, baseline $x'=(0,0,0)$, so $f_1(x)-f_1(x')=3$. It has a pure
  additive part ($2x_1$) and an *interaction* ($x_2x_3$): neither $x_2$ nor $x_3$ does anything alone.
* $f_2(x)=1-\mathrm{ReLU}(1-x_1)$ at $x_1=2$, baseline 0. It rises from 0 to 1 as $x_1$ goes from 0 to 1,
  then is flat: a **saturating** function, like a sigmoid far from 0.
* $f_3(x)=\max(x_1,x_2)$ at $x=(1,2)$, baseline 0. A **redundancy** function: once one input is large, the
  other no longer matters.

### 3.2 Gradients (saliency)

**Intuition.** Ask: if I nudge feature $i$ a tiny bit, how much does the output move? That is the partial
derivative.

**Definition.** $a_i^{\text{grad}}(x)=\dfrac{\partial f}{\partial x_i}(x)$, often in absolute value
(Simonyan et al. 2013 called the image of $|\nabla f|$ a *saliency map*).

**Justification.** The first-order Taylor expansion $f(x+\delta)\approx f(x)+\nabla f(x)^\top\delta$ says
the gradient is the best *local linear* summary of $f$ around $x$.

**Problems.**

* **Saturation.** For $f_2$ at $x_1=2$ the gradient is 0, although $x_1$ is the *only* reason the output is
  1 rather than 0. Gradients measure sensitivity *here*, not contribution *so far*. Deep networks are full
  of saturating units (ReLUs that switched off, sigmoids near 0 or 1), so this is a practical problem.
* **Not in output units.** A gradient has units of "output per unit of input"; it does not add up to
  anything. For $f_1$ the gradient is $(2,1,1)$, summing to 4, while the output moved by 3.
* **Noise.** In deep networks gradients fluctuate sharply between nearby inputs (the motivation for
  SmoothGrad, which averages gradients over noisy copies of $x$).

### 3.3 Gradient × input

**Definition.** $a_i^{\text{GxI}}(x)=x_i\,\dfrac{\partial f}{\partial x_i}(x)$.

**Why it is sensible for linear models.** If $f(x)=w^\top x+b$ then $x_i\,\partial f/\partial x_i=w_ix_i$,
exactly the term that feature $i$ adds to the output relative to $x_i=0$. Gradient × input is the
"contribution" of each feature for a linear model with a zero baseline.

**Where it breaks.** For non-linear models it inherits the gradient's problems. For $f_1$ it gives
$(2,1,1)$, which sums to 4, not 3: the interaction $x_2x_3$ is counted twice (once through each factor).
For $f_2$ it is 0. Methods such as DeepLIFT (Shrikumar et al. 2017) and Layer-wise Relevance Propagation
replace the gradient by "discrete gradients" that compare activations with a reference activation, which
fixes saturation and restores a summation property.

### 3.4 Integrated gradients (Sundararajan, Taly & Yan 2017)

**Intuition.** The gradient fails for $f_2$ because it looks only at the end point, where the function has
already gone flat. So look *along the whole way*: walk in a straight line from the baseline $x'$ to the
input $x$, and add up, for each feature, the gradient you experience times the distance that feature moved.
Contributions made early in the walk (before saturation) are then counted.

**Definition.** With the straight path $\gamma(\alpha)=x'+\alpha(x-x')$, $\alpha\in[0,1]$:

$$
\mathrm{IG}_i(x)=(x_i-x'_i)\int_0^1 \frac{\partial f}{\partial x_i}\bigl(x'+\alpha(x-x')\bigr)\,d\alpha .
$$

**Derivation of completeness.** Define $g(\alpha)=f(\gamma(\alpha))$. By the chain rule,

$$
g'(\alpha)=\sum_{i=1}^d \frac{\partial f}{\partial x_i}(\gamma(\alpha))\,\frac{d\gamma_i}{d\alpha}
=\sum_{i=1}^d \frac{\partial f}{\partial x_i}(\gamma(\alpha))\,(x_i-x'_i).
$$

By the fundamental theorem of calculus, $g(1)-g(0)=\int_0^1 g'(\alpha)\,d\alpha$. But $g(1)=f(x)$ and
$g(0)=f(x')$, and swapping the finite sum with the integral gives

$$
\boxed{\;\sum_{i=1}^d \mathrm{IG}_i(x)=f(x)-f(x')\;}
$$

This is the **completeness** axiom: integrated gradients always account exactly for the difference between
the output and the baseline output. (It needs $f$ to be differentiable almost everywhere along the path,
which holds for networks built from ReLU, sigmoid, tanh, softmax and so on.)

**Worked example: $f_1(x)=2x_1+x_2x_3$, $x=(1,1,1)$, $x'=0$.** The path is $\gamma(\alpha)=(\alpha,\alpha,\alpha)$.

* $\partial f_1/\partial x_1=2$ everywhere, so $\mathrm{IG}_1=(1-0)\int_0^1 2\,d\alpha=2$.
* $\partial f_1/\partial x_2=x_3$, which on the path equals $\alpha$, so
  $\mathrm{IG}_2=(1-0)\int_0^1\alpha\,d\alpha=\tfrac12$. By symmetry $\mathrm{IG}_3=\tfrac12$.
* Sum $=2+\tfrac12+\tfrac12=3=f_1(x)-f_1(x')$. Completeness holds, and the interaction $x_2x_3=1$ has been
  split equally between its two factors.

For $f_2$ at $x_1=2$: on the path $\gamma(\alpha)=2\alpha$ the derivative is 1 while $2\alpha<1$ (that is,
$\alpha<\tfrac12$) and 0 afterwards, so $\mathrm{IG}_1=2\int_0^{1/2}1\,d\alpha=1$. Integrated gradients
correctly give the whole output to $x_1$, where the plain gradient gave 0.

**The axioms.** Sundararajan et al. motivate IG with axioms that a good attribution method should satisfy:

| Axiom | Statement | Gradients | GxI | IG |
|---|---|---|---|---|
| Sensitivity (a) | if $x$ and $x'$ differ in one feature only and $f(x)\ne f(x')$, that feature gets non-zero attribution | fails ($f_2$) | fails ($f_2$) | holds |
| Sensitivity (b) / dummy | a feature the function does not depend on gets zero | holds | holds | holds |
| Implementation invariance | two networks that compute the same function get the same attributions | holds | holds | holds (DeepLIFT and LRP fail it) |
| Completeness | attributions sum to $f(x)-f(x')$ | fails | fails | holds |
| Linearity | attributions of $\alpha f+\beta g$ are $\alpha a(f)+\beta a(g)$ | holds | holds | holds |
| Symmetry-preserving | two symmetric variables with equal values get equal attributions (for the straight path) | | | holds |

Citing a result from cost-sharing theory (Friedman 2004), the paper notes that *path methods* (integrals of
the gradient along some path from $x'$ to $x$) are the only methods that always satisfy implementation
invariance, sensitivity (b), linearity and completeness, and it shows that the straight-line path is the
one that is *symmetry-preserving*.
In cooperative game theory IG on the straight path corresponds to the **Aumann-Shapley** cost-sharing
method, a continuous cousin of the Shapley value, which brings us to the next section.

**Practicalities.**

* The integral is approximated by a Riemann sum with $m$ steps:
  $\mathrm{IG}_i\approx(x_i-x'_i)\frac1m\sum_{k=1}^m\partial_i f(x'+\tfrac{k-1/2}{m}(x-x'))$ (midpoints).
  Choose $m$ by *checking completeness*: if $\sum_i\mathrm{IG}_i$ differs from $f(x)-f(x')$ by more than a
  few percent, increase $m$ (20 to 300 steps is typical).
* **The baseline is a modelling choice**, not a technicality. For images a black image is common; for our
  similarity inputs, "zero similarity" or "the average drug" are both defensible and give different
  answers. The Distill article by Sturmfels, Lundberg & Lee (2020) shows this vividly.

### 3.5 Shapley values

**Intuition.** Treat the features as players in a team who together produce a payout $f(x)-f(x')$. How do
you split the payout fairly when players interact? Lloyd Shapley's 1953 answer: imagine the players
arriving one by one in a random order; each player is paid the *extra* value they add on arrival; average
that over all possible orders.

**The game.** Let $N=\{1,\dots,d\}$ be the features. For a coalition $S\subseteq N$, the **value function**
$v(S)$ is the model output when the features in $S$ are "present" (set to their values in $x$) and the
rest are "absent". The simplest definition of absence is the baseline:

$$
v(S)=f(x_S, x'_{N\setminus S}),\qquad v(\varnothing)=f(x'),\quad v(N)=f(x).
$$

**Definition.**

$$
\phi_i(v)=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(d-|S|-1)!}{d!}\Bigl[v(S\cup\{i\})-v(S)\Bigr].
$$

The weight $\frac{|S|!(d-|S|-1)!}{d!}$ is exactly the probability that, in a uniformly random ordering of
the $d$ players, the players who arrive before $i$ are precisely the set $S$ ($|S|!$ orderings of those
who came before, $(d-|S|-1)!$ of those after, out of $d!$). So equivalently

$$
\phi_i(v)=\frac{1}{d!}\sum_{\pi\in\text{orderings}}\Bigl[v(\mathrm{Pre}_\pi(i)\cup\{i\})-v(\mathrm{Pre}_\pi(i))\Bigr].
$$

**The four axioms**, and the theorem that makes Shapley values special:

1. **Efficiency:** $\sum_{i\in N}\phi_i=v(N)-v(\varnothing)$. The payout is distributed completely. (For
   attributions: they sum to $f(x)-f(x')$, the same property as IG's completeness.)
2. **Symmetry:** if $v(S\cup\{i\})=v(S\cup\{j\})$ for every $S$ not containing $i,j$, then $\phi_i=\phi_j$.
3. **Dummy (null player):** if $v(S\cup\{i\})=v(S)$ for every $S$, then $\phi_i=0$.
4. **Additivity:** for two games, $\phi_i(v+w)=\phi_i(v)+\phi_i(w)$.

**Theorem (Shapley 1953).** The Shapley value is the *only* allocation rule satisfying all four axioms.

This uniqueness is why Lundberg & Lee (2017) built SHAP on it: they showed that within the class of
*additive feature attribution methods* (explanations of the form $g(z')=\phi_0+\sum_i\phi_iz'_i$ on binary
"present/absent" indicators $z'$), only Shapley values satisfy their properties of *local accuracy*
(= efficiency), *missingness* and *consistency*. They also showed that LIME, DeepLIFT and LRP are
special cases of the same additive form with different weightings.

**Worked example 1, fully by hand: $f_1(x)=2x_1+x_2x_3$, $x=(1,1,1)$, baseline 0.**

Step 1, the value of every coalition (present features take the value 1, absent ones 0):

| $S$ | $\varnothing$ | {1} | {2} | {3} | {1,2} | {1,3} | {2,3} | {1,2,3} |
|---|---|---|---|---|---|---|---|---|
| $v(S)$ | 0 | 2 | 0 | 0 | 2 | 2 | 1 | 3 |

Step 2, the weights for $d=3$: $|S|=0\Rightarrow\frac{0!\,2!}{3!}=\frac13$; $|S|=1\Rightarrow\frac{1!\,1!}{3!}=\frac16$;
$|S|=2\Rightarrow\frac{2!\,0!}{3!}=\frac13$.

Step 3, feature 1: the marginal contribution $v(S\cup\{1\})-v(S)$ is 2 for every $S$
($2-0$, $2-0$, $2-0$, $3-1$), so $\phi_1=2\,(\frac13+\frac16+\frac16+\frac13)=2$.

Feature 2: marginal contributions are $v(\{2\})-v(\varnothing)=0$, $v(\{1,2\})-v(\{1\})=0$,
$v(\{2,3\})-v(\{3\})=1$, $v(\{1,2,3\})-v(\{1,3\})=1$. So
$\phi_2=\frac13\cdot0+\frac16\cdot0+\frac16\cdot1+\frac13\cdot1=\frac12$. By symmetry $\phi_3=\frac12$.

Step 4, check efficiency: $2+\frac12+\frac12=3=v(N)-v(\varnothing)$. ✓

Same answer via orderings: of the 6 orderings of {1,2,3}, feature 2 adds value 1 exactly when it arrives
after feature 3 (orderings 3-2-1, 1-3-2, 3-1-2) and 0 otherwise: $\phi_2=3/6=\frac12$.

**Worked example 2: $f_3=\max(x_1,x_2)$ at $x=(1,2)$.** $v(\varnothing)=0$, $v(\{1\})=1$, $v(\{2\})=2$,
$v(\{1,2\})=2$. Two orderings: in (1,2), player 1 adds 1 and player 2 adds 1; in (2,1), player 2 adds 2 and
player 1 adds 0. Hence $\phi_1=\frac{1+0}{2}=0.5$ and $\phi_2=\frac{1+2}{2}=1.5$, summing to 2. ✓
Note that integrated gradients give $(0,2)$ here: on the straight path $x_2=2\alpha$ is always the larger
input, so the gradient with respect to $x_1$ is always 0. **IG and Shapley are different methods** that
coincide on some functions (like $f_1$) and not on others.

**What does "absent" mean?** This is the deep question behind every Shapley-based explanation, and there
are two families of answers:

* **Interventional (baseline / marginal) Shapley:** set absent features to a baseline, or average over
  values drawn independently from the data distribution. It explains *the model's function*. It may feed
  the model unrealistic combinations (a drug that has ECFP neighbours but no CDK neighbours).
* **Conditional (observational) Shapley:** average absent features over their distribution *given* the
  present ones. It stays on the data manifold but credit can flow to features the model never uses, if they
  are correlated with ones it does use.

Janzing, Minorics & Blöbaum (2020) argue that the interventional version is the right one for explaining a
model's computation (it is a causal question about the model), and Kumar et al. (2020) catalogue further
problems with Shapley-based feature importance (it answers a question about a game, which may not be the
question the user had). The lesson for us: **always state the baseline / absence definition**.

**Cost and approximations.** Exact Shapley values need $v(S)$ for all $2^d$ coalitions. That is fine for
the project's handful of evidence sources (6 views plus known links: $2^7=128$ forward passes), and
hopeless for thousands of input features. Approximations:

* **Permutation sampling:** sample random orderings and average marginal contributions (unbiased).
* **KernelSHAP:** fit a weighted linear regression $g(z')=\phi_0+\sum_i\phi_iz'_i$ to $v$ on sampled
  coalitions, with the *Shapley kernel* weight
  $$\pi(z')=\frac{d-1}{\binom{d}{|z'|}\,|z'|\,(d-|z'|)},$$
  and (in practice) a very large weight on the empty and full coalitions to enforce efficiency. With all
  $2^d$ coalitions, the regression solution equals the exact Shapley values (our code in 3.8 confirms it).
* **Model-specific:** TreeSHAP (exact and fast for tree ensembles), DeepSHAP (DeepLIFT-based) and
  GradientSHAP (an expectation of IG over random baselines).

**Shapley values of a linear model.** If $f(x)=b+\sum_iw_ix_i$ and absent features take baseline values
$x'_i$, then every marginal contribution of feature $i$ is $w_i(x_i-x'_i)$, regardless of the coalition.
Hence $\phi_i=w_i(x_i-x'_i)$; with the mean as baseline, $\phi_i=w_i(x_i-\mathbb{E}[x_i])$. Remember
this: it is the reason the project's additive head is "faithful by construction" (section 7).

### 3.6 LIME (Ribeiro, Singh & Guestrin 2016)

**Intuition.** A complicated model may be approximately linear in a small neighbourhood of one input. So
sample perturbed versions of the input near $x$, ask the model for its outputs, and fit a simple,
interpretable model (usually a sparse linear model) to those outputs, giving more weight to perturbations
that are closer to $x$. The simple model's coefficients are the explanation.

**Formal objective.**

$$
\xi(x)=\arg\min_{g\in G}\;\mathcal{L}(f,g,\pi_x)+\Omega(g),
$$

where $G$ is a family of interpretable models (e.g. linear models on binary "feature present"
indicators), $\pi_x(z)$ is a proximity kernel (commonly $\exp(-D(x,z)^2/\sigma^2)$), $\mathcal{L}$ is the
$\pi_x$-weighted squared error between $f$ and $g$ on the samples, and $\Omega(g)$ penalises complexity
(e.g. the number of non-zero coefficients).

**Strengths:** model-agnostic; produces sparse, readable explanations; works for text and images via
"interpretable representations" (words present, super-pixels on).

**Weaknesses:**

* **The kernel width $\sigma$ changes the answer.** Small $\sigma$ approximates the local slope (close to
  gradient × input); large $\sigma$ approaches a global average. There is no principled choice; the
  reference implementation uses a heuristic default (for tabular data $0.75\sqrt{d}$). Our code in 3.8
  shows LIME moving from $(2,1,1)$ to roughly $(2,0.5,0.5)$ for $f_1$ as $\sigma$ grows.
* **Sampling noise:** two runs with different random perturbations can give different explanations.
* **Off-manifold perturbations:** randomly switching features off creates inputs unlike any real one, and
  the model's behaviour there may be arbitrary. Slack et al. (2020) exploited exactly this to build models
  that behave in a biased way on real data but look innocent to LIME and SHAP.

KernelSHAP is LIME with a specific kernel ($\pi$ above), no complexity penalty and the efficiency
constraint; that particular choice is what turns the LIME regression into Shapley values.

### 3.7 A map of attribution methods

| Method | Needs | Satisfies completeness? | Handles saturation? | Handles interactions? | Cost |
|---|---|---|---|---|---|
| Gradient | gradients | no | no | no (local slope only) | 1 backward pass |
| Gradient × input | gradients | only for linear $f$ | no | double-counts products | 1 backward pass |
| Integrated gradients | gradients, baseline | **yes** | yes | splits along path | $m$ backward passes |
| Shapley (exact) | queries, baseline/absence rule | **yes** (efficiency) | yes | splits fairly, averaged over coalitions | $2^d$ queries |
| KernelSHAP | queries | approximately (exactly with all coalitions) | yes | as Shapley | many queries |
| LIME | queries, kernel | no | partly | no (linear surrogate) | many queries |
| Occlusion (leave-one-out) | queries | no | yes | badly (section 4.2) | $d$ queries |

### 3.8 Code: all of the above, from scratch

The first script implements gradient, gradient × input, integrated gradients (midpoint Riemann sum),
exact Shapley values (enumerating coalitions) and occlusion, and runs them on $f_1$, $f_2$, $f_3$. It is
CPU-only and takes well under a second once PyTorch is imported.

```python
# Block 1: gradient, gradient x input, integrated gradients, exact Shapley, occlusion
# on three tiny functions. CPU only, runs in well under a second.
import itertools
import math

import torch

torch.set_default_dtype(torch.float64)   # exact-looking numbers for teaching


def f_inter(x):      # 2*x1 + x2*x3  (an interaction between features 2 and 3)
    return 2 * x[..., 0] + x[..., 1] * x[..., 2]


def f_sat(x):        # 1 - ReLU(1 - x1): rises with x1, then flat (saturates) at x1 >= 1
    return 1 - torch.relu(1 - x[..., 0])


def f_max(x):        # max(x1, x2)
    return torch.maximum(x[..., 0], x[..., 1])


def gradient(f, x):
    x = x.clone().requires_grad_(True)
    f(x).backward()
    return x.grad.detach()


def grad_x_input(f, x):
    return gradient(f, x) * x


def integrated_gradients(f, x, baseline, steps=1000):
    # Riemann (midpoint) approximation of  (x - x') * integral_0^1 df/dx(x' + a(x - x')) da
    alphas = (torch.arange(steps) + 0.5) / steps                     # midpoints in (0, 1)
    path = baseline + alphas[:, None] * (x - baseline)              # steps x d
    path.requires_grad_(True)
    f(path).sum().backward()                                        # grads at every point
    return (x - baseline) * path.grad.mean(0)


def shapley_exact(f, x, baseline):
    """Exact Shapley values; 'absent' features are set to their baseline value."""
    d = len(x)

    def v(S):                                                       # value of coalition S
        z = baseline.clone()
        z[list(S)] = x[list(S)]
        return f(z).item()

    phi = torch.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for size in range(d):
            for S in itertools.combinations(others, size):
                w = math.factorial(size) * math.factorial(d - size - 1) / math.factorial(d)
                phi[i] += w * (v(S + (i,)) - v(S))
    return phi


def occlusion(f, x, baseline):
    """Drop in output when ONE feature is replaced by its baseline value."""
    out = []
    for i in range(len(x)):
        z = x.clone()
        z[i] = baseline[i]
        out.append((f(x) - f(z)).item())
    return torch.tensor(out)


cases = [("2*x1 + x2*x3", f_inter, torch.tensor([1., 1., 1.]), torch.zeros(3)),
         ("1 - ReLU(1 - x1)", f_sat, torch.tensor([2.]), torch.zeros(1)),
         ("max(x1, x2)", f_max, torch.tensor([1., 2.]), torch.zeros(2))]

for name, f, x, b in cases:
    print(f"f = {name},  x = {x.tolist()},  baseline = {b.tolist()},  "
          f"f(x) - f(baseline) = {(f(x) - f(b)).item():.3f}")
    for label, a in [("gradient", gradient(f, x)),
                     ("grad x input", grad_x_input(f, x)),
                     ("integrated grads", integrated_gradients(f, x, b)),
                     ("Shapley (exact)", shapley_exact(f, x, b)),
                     ("occlusion", occlusion(f, x, b))]:
        print(f"   {label:17s} {[round(v, 3) + 0.0 for v in a.tolist()]}   sum = {a.sum().item():.3f}")
    print()
```

Expected output:

```
f = 2*x1 + x2*x3,  x = [1.0, 1.0, 1.0],  baseline = [0.0, 0.0, 0.0],  f(x) - f(baseline) = 3.000
   gradient          [2.0, 1.0, 1.0]   sum = 4.000
   grad x input      [2.0, 1.0, 1.0]   sum = 4.000
   integrated grads  [2.0, 0.5, 0.5]   sum = 3.000
   Shapley (exact)   [2.0, 0.5, 0.5]   sum = 3.000
   occlusion         [2.0, 1.0, 1.0]   sum = 4.000

f = 1 - ReLU(1 - x1),  x = [2.0],  baseline = [0.0],  f(x) - f(baseline) = 1.000
   gradient          [0.0]   sum = 0.000
   grad x input      [0.0]   sum = 0.000
   integrated grads  [1.0]   sum = 1.000
   Shapley (exact)   [1.0]   sum = 1.000
   occlusion         [1.0]   sum = 1.000

f = max(x1, x2),  x = [1.0, 2.0],  baseline = [0.0, 0.0],  f(x) - f(baseline) = 2.000
   gradient          [0.0, 1.0]   sum = 1.000
   grad x input      [0.0, 2.0]   sum = 2.000
   integrated grads  [0.0, 2.0]   sum = 2.000
   Shapley (exact)   [0.5, 1.5]   sum = 2.000
   occlusion         [0.0, 1.0]   sum = 1.000
```

How to read it:

* Only integrated gradients and Shapley values always sum to $f(x)-f(x')$ (completeness / efficiency).
* For the interaction, gradient-based methods and occlusion give 1 to *both* $x_2$ and $x_3$, double
  counting the single unit of value they create together.
* For the saturating function, gradient and gradient × input say "irrelevant", which is false.
* For the max, IG and Shapley disagree. Neither is "wrong": they answer different questions (a path
  question vs an average-over-coalitions question). Occlusion gives $x_1$ zero credit because $x_2$ alone
  already produces the output: **occlusion is blind to redundancy**. Keep this in mind for section 5.

The second script implements KernelSHAP (with all $2^3$ coalitions) and LIME (500 random perturbations,
three kernel widths) for $f_1$:

```python
# Block 2: LIME and KernelSHAP from scratch on f(x) = 2*x1 + x2*x3, x = (1, 1, 1), baseline 0.
import itertools
from math import comb

import numpy as np

f = lambda X: 2 * X[:, 0] + X[:, 1] * X[:, 2]
x, base = np.array([1., 1., 1.]), np.zeros(3)
M = 3


def h(Z):
    """Map binary 'is the feature present?' masks to model inputs."""
    return np.where(Z == 1, x, base)


def weighted_lstsq(Z, y, w):
    A = np.column_stack([np.ones(len(Z)), Z])           # intercept + one coef per feature
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(A * W, y * W[:, 0], rcond=None)
    return coef                                         # [phi_0, phi_1, ..., phi_M]


# --- KernelSHAP with ALL 2^M coalitions: recovers the exact Shapley values -----
Z = np.array(list(itertools.product([0, 1], repeat=M)), dtype=float)
size = Z.sum(1).astype(int)
w = np.array([1e6 if s in (0, M) else (M - 1) / (comb(M, s) * s * (M - s)) for s in size])
print("KernelSHAP (exact enumeration):", np.round(weighted_lstsq(Z, f(h(Z)), w), 3) + 0.0)

# --- LIME: random perturbations, exponential kernel on distance, linear surrogate
rng = np.random.default_rng(0)
Zs = rng.integers(0, 2, size=(500, M)).astype(float)
dist = 1 - Zs.mean(1)                                   # share of features switched off
for width in (0.25, 0.75, 5.0):
    wl = np.exp(-dist ** 2 / width ** 2)
    print(f"LIME  kernel width {width:4.2f}:          ", np.round(weighted_lstsq(Zs, f(h(Zs)), wl), 3))
```

Expected output (the first number in each row is the intercept $\phi_0$):

```
KernelSHAP (exact enumeration): [0.  2.  0.5 0.5]
LIME  kernel width 0.25:           [-0.99   1.999  0.995  0.995]
LIME  kernel width 0.75:           [-0.465  2.021  0.653  0.668]
LIME  kernel width 5.00:           [-0.283  2.026  0.515  0.526]
```

KernelSHAP recovers $\phi=(2,0.5,0.5)$ with $\phi_0=f(x')=0$. LIME's answer depends on the kernel width:
with a narrow kernel it effectively fits the three perturbations closest to $x$ (one feature off) and
reports the local slopes $(2,1,1)$; with a wide kernel it averages over all coalitions and approaches the
Shapley values. Same model, same input, three different "explanations". This is the strongest argument
for always reporting *which* method and *which* settings produced an explanation.

---

## 4. Perturbation: occlusion and ablation

### 4.1 Occlusion (leave-one-out)

**Intuition.** The most direct way to find out whether the model used something is to take it away and
watch what happens. Zeiler & Fergus (2014) slid a grey square over an image and recorded how much the
class score fell for each position; the same idea, applied to any group of inputs, is called
**occlusion**, **leave-one-out (LOO)** or **perturbation** attribution.

**Definition.** For a group of inputs $G$ (one feature, one evidence source, one edge),

$$
\mathrm{occ}_G(x)=f(x)-f(x_{\setminus G}),
$$

where $x_{\setminus G}$ is $x$ with the group "removed" (set to a baseline, deleted from the graph, or
masked). A large positive value means the prediction depended on $G$; a negative value means $G$ was
pulling the score *down*.

**Strengths.** Model-agnostic; cheap ($|{\rm groups}|+1$ forward passes); measured in output units; it
directly answers a counterfactual question about *this trained model*: "what would it have said without
this evidence?". That is why the project uses it as its main local explanation.

### 4.2 Occlusion versus Shapley: redundancy and synergy

Compare the definitions. Occlusion is a *single* marginal contribution, the one where every other player
is present:

$$
\mathrm{occ}_i=v(N)-v(N\setminus\{i\}),
$$

while the Shapley value averages the marginal contribution over *all* coalitions. The two agree when
features do not interact (additive models). They disagree, in opposite directions, for the two basic kinds
of interaction:

**Redundancy (an OR).** $f=\max(x_1,x_2)$ with $x=(1,1)$, baseline 0. Remove $x_1$: output stays 1. Remove
$x_2$: output stays 1. Occlusion: $(0,0)$, summing to 0, although the output is 1. Shapley: $(0.5,0.5)$.
*Occlusion says neither mattered, because each backs the other up.*

**Synergy (an AND).** $f=\min(x_1,x_2)$ with $x=(1,1)$. Remove either: output drops to 0. Occlusion:
$(1,1)$, summing to 2, although the output is only 1. Shapley: $(0.5,0.5)$. *Occlusion says both were
fully responsible.*

Redundancy is everywhere in the project. The two chemical views `chem_cdk` and `chem_ecfp` both encode
structure; the known-association relation is duplicated in the input features; three disease views overlap.
So we should *expect* occlusion of a single source to under-state its importance whenever another source
carries the same information. Section 11 shows this in the real outputs. The remedy, when you need it, is
grouped occlusion (remove both chemical views together) or Shapley values over the 7 evidence sources
(128 forward passes per pair; affordable).

### 4.3 What does "removed" mean?

Every perturbation method must decide how to remove something, and every choice has side effects:

| Removal rule | Side effect |
|---|---|
| set features to 0 | 0 may itself be meaningful ("zero similarity" is a statement) |
| set to the dataset mean | the mean drug does not exist |
| sample from the data distribution | expensive; correlated features become inconsistent |
| delete an edge / relation from the graph | changes node degrees and normalisations, so *other* messages are re-weighted |
| mask a relation inside a softmax | the remaining relations are re-normalised and absorb its weight |

The last row is exactly what MV-HGAT does (section 11.2): occluding a relation sets its `valid` flag to
`False`, so the view-level softmax spreads the weight over the remaining relations. The occlusion score
therefore measures "what the model says if it had to make do with the other sources", which includes the
other sources' ability to compensate. That is a perfectly good question, but a different one from "how
much did this source contribute in the original forward pass".

Perturbed inputs can also be **out of distribution**: the model was never trained on a drug with no chemical
neighbours, so its behaviour there may be arbitrary. Hooker et al. (2019) proposed ROAR (*RemOve And
Retrain*) partly for this reason: to evaluate an attribution method, remove the features it ranks highest
*and retrain*, so the model is evaluated in-distribution. That brings us to ablation.

### 4.4 Ablation is also an explanation (of a different kind)

An **ablation** removes a component or input source *before training* and re-trains. It answers a
different question from occlusion:

| | Occlusion | Ablation |
|---|---|---|
| When | at inference, on the trained model | before training; a new model is trained |
| Question | did *this* model use source $G$ for *this* prediction? | does the method *need* source $G$ to reach its accuracy? |
| Level | local (one prediction), or averaged | global (whole test set) |
| Compensation | other sources compensate only via re-normalisation | the re-trained model can learn to use other sources instead |
| Cost | one forward pass per group | a full cross-validation per variant |

The two can disagree, and the disagreement is informative. From the project's ablation
(`results/Fdataset/ablation_cv5/`, 5 folds, one repeat; AUPR mean ± std over folds):

| Variant | AUPR |
|---|---|
| Full MV-HGAT | 0.5004 ± 0.0285 |
| w/o MONDO semantic view | 0.5097 ± 0.0129 |
| w/o ECFP chemical view | 0.4628 ± 0.0223 |
| w/o view attention (plain mean) | 0.4776 ± 0.0428 |

Yet the trained full model gives `sem_mondo` an average view attention of about 0.17 to 0.21 per disease
(`Cdataset_disease_view_attention.csv` and its Fdataset counterpart), and occlusion sometimes ranks it among the top two drivers of a prediction.
So: the trained model *uses* the semantic view (attention and occlusion), but the method does not *need*
it on Fdataset in warm-start CV (ablation: removing it changes AUPR by less than one standard deviation, in
either direction depending on the fold). Both statements are true; they answer different questions.
Unit E3 discusses how to report this honestly. (The semantic view matters much more in cold start, where
`HOW_IT_WORKS.md` section 10 shows it carries real signal on its own.)

---

## 5. Attention as explanation

### 5.1 What attention computes

An attention layer computes a weighted average of "value" vectors, with weights produced by a softmax over
scores:

$$
\beta_r=\frac{\exp(s_r)}{\sum_{r'}\exp(s_{r'})},\qquad z=\sum_r\beta_r\,m_r .
$$

In MV-HGAT there are two attention levels (`docs/HOW_IT_WORKS.md` section 4.2): node-level GAT attention
$\alpha_{ij}$ over neighbours inside each relation, and **view-level** attention $\beta_i^r$ over the
relation messages $m_i^r$ of node $i$, with score $s_i^r=q^\top\tanh(Pm_i^r+b)$. The $\beta$'s are
non-negative and sum to one, so it is extremely tempting to read them as "node $i$ relies 45% on known links
and 20% on phenotype similarity". The question is whether that reading is justified.

### 5.2 Why attention is not automatically an explanation

Write the output of a downstream linear read-out $u$ as

$$
y=u^\top z=\sum_r\beta_r\,(u^\top m_r).
$$

The contribution of relation $r$ is $\beta_r\,(u^\top m_r)$: the attention weight **times** how large the
message is in the direction the rest of the network cares about. A big $\beta_r$ on a message that is
small, or orthogonal to $u$, contributes little; a tiny $\beta_r$ on a huge message can dominate. And in a
real network the read-out is not linear: after the attention come residual (skip) connections, layer
normalisation, a second layer, a decoder and, in MV-HGAT, an additive head and a gate that do not pass
through the attention at all.

More subtly, attention scores are computed *from the messages themselves*. If two relations carry the same
information, the softmax can split the weight between them in any proportion without changing the output,
so the split is not identifiable: many attention distributions are equally consistent with the
prediction.

### 5.3 The debate: Jain & Wallace (2019) versus Wiegreffe & Pinter (2019)

**Jain & Wallace, "Attention is not Explanation" (NAACL 2019).** Working with recurrent (BiLSTM) text
models with attention on classification, question answering and natural language inference tasks, they
made two empirical arguments:

1. Learned attention weights were frequently **uncorrelated** (low Kendall $\tau$) with gradient-based and
   leave-one-out measures of token importance.
2. They could construct **counterfactual attention distributions**: by permuting attention weights, and by
   adversarially searching for distributions as different as possible (in Jensen-Shannon divergence) from
   the original that left the prediction (almost) unchanged. Such alternative distributions were easy to
   find. If very different "explanations" give the same output, the original attention cannot be *the*
   explanation.

Their conclusion: standard attention modules should not be treated as providing faithful explanations.

**Wiegreffe & Pinter, "Attention is not not Explanation" (EMNLP 2019).** They did not claim the opposite;
they argued that whether attention explains depends on the *definition* of explanation and on the
*task*, and that Jain & Wallace's tests were not sufficient to show it does not. Their points and tests:

1. **Uniform-weights baseline.** First ask whether attention matters at all for the task: freeze the
   attention to uniform weights and retrain. On several of the datasets used, performance barely changed,
   so attention there could not be explaining much anyway.
2. **Variance calibration across seeds.** Retrain with different random seeds to see how much attention
   naturally varies; an "adversarial" distribution is only surprising if it is further away than that.
3. **A diagnostic test.** Take the attention weights learned by a model and use them to guide a simple,
   non-contextual model (an MLP over the tokens). If the learned weights help it more than uniform or
   random weights, they carry real information about token importance.
4. **Model-consistent adversarial training.** Jain & Wallace manipulated attention *per instance*, with no
   model that would actually produce those distributions. Wiegreffe & Pinter trained a whole adversarial
   *model* to produce different attention while keeping predictions. It was possible, but the adversarial
   attention performed poorly in the diagnostic test.

Their conclusion: attention can provide *an* explanation, a plausible and sometimes useful one, but not
necessarily *the* (unique, faithful) explanation; claims must be backed by tests like the above.

Serrano & Smith ("Is Attention Interpretable?", ACL 2019) added a direct test: zero out the highest-attention
components and see whether the decision flips. Often it did not, and gradient-based rankings identified
decision-flipping components more efficiently than attention rankings.

**The consensus you can safely write in a paper:** attention weights describe how a model *mixes* its
inputs; they are not, without further checks, a faithful measure of how much each input *contributed* to
the output. Faithfulness must be tested with perturbation (occlusion, removal and retraining) or with
attribution methods that satisfy completeness. This is exactly what the project's `PREREQUISITES.md`
says: "we report both because attention alone is not proof".

### 5.4 A toy model where attention and occlusion disagree

The following model mimics MV-HGAT's decoder for a single drug-disease pair. Three relations (`assoc`,
`chem`, `pheno`) send messages into a view-attention softmax; the attended embedding is read out by $u$;
and, as in MV-HGAT, `chem` and `pheno` also feed an additive propagation head. Occlusion follows the
project's rule: remove the relation from the softmax (re-normalising the others) *and* zero its head term.

```python
# Block 3: a toy "MV-HGAT-like" scorer where attention and occlusion disagree.
# One drug-disease pair; three relations feed a view-attention layer, and two of
# them also feed an additive propagation head (as in model.py).
import numpy as np

rel = ["assoc", "chem", "pheno"]
m = {"assoc": np.array([2.0, 0.0]),     # message from known-link neighbours
     "chem":  np.array([1.9, 0.1]),     # almost the SAME information as assoc (redundant)
     "pheno": np.array([0.0, 0.3])}     # small, different message
q = np.array([2.0, -4.0])               # attention query: likes dimension 0, dislikes dim 1
u = np.array([0.5, 0.5])                # read-out of the GNN embedding
head = {"chem": 0.2, "pheno": 2.0}      # w_v * P_v[i, j] of the propagation head
b = -1.0
sig = lambda t: 1 / (1 + np.exp(-t))


def score(active):
    """Logit and probability using only the relations in `active` (occlusion =
    removing a relation from the softmax, like valid=False in ViewAttention,
    AND zeroing its propagation-head term, like MVHGATMethod.occlusion)."""
    s = np.array([q @ np.tanh(m[r]) for r in active])
    beta = np.exp(s - s.max()); beta /= beta.sum()
    z = sum(bt * m[r] for bt, r in zip(beta, active))
    logit = u @ z + sum(head.get(r, 0.0) for r in active) + b
    return logit, sig(logit), dict(zip(active, beta))


L, p, beta = score(rel)
print(f"full model: logit {L:.3f}, probability {p:.3f}\n")
print(f"{'relation':8s} {'attention beta':>15s} {'occlusion dLogit':>17s} {'occlusion dProb':>16s}")
for r in rel:
    L2, p2, _ = score([x for x in rel if x != r])
    print(f"{r:8s} {beta[r]:15.3f} {L - L2:17.3f} {p - p2:16.3f}")
```

Expected output:

```
full model: logit 2.177, probability 0.898

relation  attention beta  occlusion dLogit  occlusion dProb
assoc              0.586             0.032            0.003
chem               0.387             0.214            0.021
pheno              0.027             1.977            0.348
```

Read the table row by row:

* **`assoc` has the highest attention (0.586) but almost no occlusion impact (0.003).** Its message is
  nearly identical to `chem`'s, so when it is removed the softmax hands its weight to `chem` and the
  attended embedding barely moves. *High attention, low impact, because of redundancy plus
  re-normalisation.*
* **`pheno` has almost no attention (0.027) but by far the largest impact (0.348).** Its influence does not
  flow through the attention at all: it flows through the additive head ($w_vP_v=2.0$ on the logit).
  *Low attention, high impact, because of a pathway that bypasses the attention.*
* **`chem`** is in between: removing it drops the head term (0.2) and slightly changes the embedding.

Nothing in this toy is exotic: redundant evidence sources, a softmax that re-normalises, and a parallel
additive pathway are all present in MV-HGAT. That is why the case-study tables in this project report
**occlusion** as the per-prediction explanation, and attention only as a description of the encoder.

### 5.5 Using attention responsibly

Attention is not useless. It is a cheap, built-in *descriptive statistic of the computation*, and it can
reveal things (e.g. "drug nodes put 77% of their view attention on the known-link relation" tells you the
encoder is largely collaborative-filtering). Use it with these safeguards:

1. **Run the uniform-attention ablation.** The project already does: "w/o view attention (plain mean)"
   gives AUPR 0.4776 ± 0.0428 against 0.5004 ± 0.0285 for the full model on Fdataset (lower in 4 of 5
   folds, by 0.023 on average). The attention helps a little, but the model does not stand or fall with it.
   That bounds how much explanatory weight $\beta$ can carry.
2. **Check stability across seeds.** Average β over seeds (the case-study script averages 5) and report the
   spread; if the ranking of relations changes between seeds, do not interpret it.
3. **Compare with a perturbation measure** (occlusion) and say where they agree and disagree.
4. **Do not use causal or quantitative language** ("node $i$ relies 45% on...") unless backed by a
   faithful method. Prefer "the encoder assigns on average 45% of its view-level attention to...".

---

## 6. Explaining graph neural networks

### 6.1 What is special about graphs

In a GNN, a node's prediction depends on its **computation graph**: its $L$-hop neighbourhood, unrolled
over the $L$ message-passing layers. For link prediction, the score of the pair $(i,j)$ depends on the
union of the neighbourhoods of $i$ and $j$. An explanation can therefore point at:

* **edges** (which messages mattered),
* **nodes** or **node features** (which neighbours or attributes mattered),
* **subgraphs** (a small connected motif, often the most human-readable form),
* **relation types / meta-paths** in a heterogeneous graph (which *kinds* of edges mattered; this is the
  project's level),
* or, at the **model level**, which graph patterns maximise a class score (e.g. XGNN generates such graphs).

Two features make graphs harder than images or tables: the input is **discrete** (an edge is there or not,
so gradients with respect to edges need a continuous relaxation), and **edges interact** through the
message-passing normalisation (removing one edge changes the weights of the others).

### 6.2 A taxonomy (Yuan, Yu, Gui & Ji, "Explainability in graph neural networks: a taxonomic survey")

| Family | Idea | Examples |
|---|---|---|
| Gradient / feature based | gradients or activations w.r.t. node features or edge weights | saliency, Grad-CAM for graphs, IG on edge weights |
| Perturbation based | learn or search a mask whose application preserves the prediction | GNNExplainer, PGExplainer, GraphMask, SubgraphX |
| Decomposition | propagate the output score back to inputs layer by layer | LRP for GNNs, GNN-LRP, Excitation BP |
| Surrogate | fit an interpretable model on perturbed neighbourhoods | GraphLime, PGM-Explainer |
| Model-level / generation | find graph patterns that maximise a prediction | XGNN |

### 6.3 GNNExplainer (Ying, Bourgeois, You, Zitnik & Leskovec 2019)

**Goal.** For one prediction $\hat y$ of a trained GNN $\Phi$ on graph $G$ with features $X$, find a small
subgraph $G_S\subseteq G$ (and optionally a subset of feature dimensions $X_S$) that is most informative
about the prediction.

**Objective.** Maximise mutual information:

$$
\max_{G_S}\;\mathrm{MI}\bigl(Y,(G_S,X_S)\bigr)=H(Y)-H\bigl(Y\mid G=G_S,X=X_S\bigr).
$$

$H(Y)$ is fixed by the trained model, so this means *minimising the uncertainty of the model's prediction
when it only sees $G_S$*.

**Continuous relaxation.** Searching over subgraphs is combinatorial, so GNNExplainer learns a real-valued
**edge mask** $M\in\mathbb{R}^{|E|}$, turns it into weights $\sigma(M)\in(0,1)^{|E|}$, and runs the
frozen GNN on the soft-masked adjacency $A\odot\sigma(M)$. For a prediction of class $c$ it minimises

$$
\mathcal{L}(M)=-\log P_\Phi\bigl(Y=c\mid A\odot\sigma(M),X\bigr)
+\lambda_{\text{size}}\sum_e\sigma(M_e)
+\lambda_{\text{ent}}\,\frac{1}{|E|}\sum_e H\bigl(\sigma(M_e)\bigr),
$$

where $H(p)=-p\log p-(1-p)\log(1-p)$. The first term keeps the prediction; the **size** term prefers few
edges; the **entropy** term pushes each mask value towards 0 or 1 so the result is a crisp subgraph. A
feature mask can be learned in the same way. After optimisation, edges with mask above a threshold (or the
top-$k$) form the explanation. The optimisation is run separately **for each prediction** (hundreds of
gradient steps each).

### 6.4 PGExplainer and others

**PGExplainer** (Luo et al. 2020, "Parameterized explainer for graph neural network") replaces the
per-instance mask by a small neural network that *predicts* each edge's mask value from the trained GNN's
embeddings of the edge's two end nodes. It is trained once over many instances (using a
reparameterisation trick for discrete edge selection) and then explains new instances with a single forward
pass. Advantages: much faster, and *collective* (explanations are consistent across instances because they
come from one explainer); it can also explain unseen instances (*inductive*).

Other methods worth knowing by name: **GraphMask** (Schlichtkrull et al. 2021: learns which edges can be
dropped at each layer without changing predictions), **SubgraphX** (Yuan et al. 2021: Monte Carlo tree
search over connected subgraphs scored by Shapley values), **PGM-Explainer** (a probabilistic graphical
model surrogate), and **CF-GNNExplainer** (Lucic et al. 2022: counterfactual, section 8). PyTorch Geometric
ships many of these in `torch_geometric.explain`, and Captum provides IG, SHAP-style and occlusion methods
for any PyTorch model.

### 6.5 Code: a GNNExplainer-style edge mask from scratch

A 7-node star graph: node 0 is the target; neighbours 1 and 2 carry strong positive evidence (feature 2.0),
3 to 5 carry weak evidence (0.1) and neighbour 6 carries *counter*-evidence (−1.0). The frozen "GNN"
computes the edge-weighted sum of neighbour features and passes it through a logistic read-out.

```python
# Block 4: a from-scratch GNNExplainer-style edge mask on a 7-node star graph,
# then fidelity and sparsity of the resulting explanation.
import torch

torch.manual_seed(0)
x = torch.tensor([0.0, 2.0, 2.0, 0.1, 0.1, 0.1, -1.0])   # node features; node 0 is the target
edges = [1, 2, 3, 4, 5, 6]                                 # edge j -> 0 for each neighbour j


def model(edge_weight):
    """A frozen one-layer 'GNN': weighted sum of neighbour features, then a logistic read-out."""
    h0 = (edge_weight * x[1:]).sum()
    return torch.sigmoid(1.5 * h0 - 2.0)


full = torch.ones(6)
p_full = model(full).item()
print(f"prediction with all edges: p = {p_full:.3f}")

# GNNExplainer objective (Ying et al. 2019), for a positive prediction:
#   minimise  -log p(masked graph)  +  l_size * sum(mask)  +  l_ent * mean entropy(mask)
logits = torch.zeros(6, requires_grad=True)               # mask = sigmoid(logits), starts at 0.5
opt = torch.optim.Adam([logits], lr=0.1)
for step in range(300):
    mask = torch.sigmoid(logits)
    ent = -(mask * torch.log(mask + 1e-9) + (1 - mask) * torch.log(1 - mask + 1e-9)).mean()
    loss = -torch.log(model(mask)) + 0.05 * mask.sum() + 0.1 * ent
    opt.zero_grad()
    loss.backward()
    opt.step()

mask = torch.sigmoid(logits).detach()
print("learned edge mask:", {f"{j}->0": round(v, 2) for j, v in zip(edges, mask.tolist())})
S = (mask > 0.5).float()                                   # hard explanation: kept edges
print("explanation subgraph:", [f"{j}->0" for j, s in zip(edges, S) if s])

p_without = model(full - S).item()                         # remove the explanation
p_only = model(S).item()                                   # keep only the explanation
print(f"fidelity+ (necessity)   = p(G) - p(G without S) = {p_full - p_without:+.3f}")
print(f"fidelity- (sufficiency) = p(G) - p(S only)      = {p_full - p_only:+.3f}")
print(f"sparsity                = 1 - |S|/|E|           = {1 - S.sum().item() / len(edges):.3f}")
```

Expected output:

```
prediction with all edges: p = 0.950
learned edge mask: {'1->0': 0.98, '2->0': 0.98, '3->0': 0.0, '4->0': 0.0, '5->0': 0.0, '6->0': 0.01}
explanation subgraph: ['1->0', '2->0']
fidelity+ (necessity)   = p(G) - p(G without S) = +0.905
fidelity- (sufficiency) = p(G) - p(S only)      = -0.032
sparsity                = 1 - |S|/|E|           = 0.667
```

Check the numbers by hand: the full graph gives $h_0=2+2+0.3-1=3.3$, logit $1.5\cdot3.3-2=2.95$,
$p=\sigma(2.95)=0.950$. Keeping only edges 1 and 2 gives $h_0=4$, logit 4, $p=0.982$, so fidelity−
$=0.950-0.982=-0.032$ (negative: the explanation alone is *more* convincing than the full graph). Removing
them gives $h_0=-0.7$, logit $-3.05$, $p=0.045$, so fidelity+ $=0.905$.

Notice what the explainer did with edge 6 (the counter-evidence): it dropped it, because dropping it
*raises* the prediction. **A "why yes" explanation silently hides the evidence for "maybe not".** When you
present an explanation to a biologist, also show the sources that pushed the score down (negative
occlusion values), as the project's occlusion does when a drop is negative.

### 6.6 Evaluating explanations

Because faithfulness cannot be read off an explanation, it has to be *measured*. Standard quantities (the
names follow the GNN survey of Yuan et al.; GraphFramEx by Amara et al. 2022 and the GraphXAI benchmark of
Agarwal et al. 2023 discuss them in depth). For $N$ explained instances with explanation masks $m_i$ and
predicted class $y_i$:

$$
\text{Fidelity}^+=\frac1N\sum_{i=1}^N\Bigl[f(G_i)_{y_i}-f\bigl(G_i^{\,1-m_i}\bigr)_{y_i}\Bigr]
\qquad\text{(remove the explanation; higher = explanation was necessary)}
$$

$$
\text{Fidelity}^-=\frac1N\sum_{i=1}^N\Bigl[f(G_i)_{y_i}-f\bigl(G_i^{\,m_i}\bigr)_{y_i}\Bigr]
\qquad\text{(keep only the explanation; lower = explanation was sufficient)}
$$

$$
\text{Sparsity}=\frac1N\sum_{i=1}^N\Bigl(1-\frac{|m_i|}{|M_i|}\Bigr)
\qquad\text{(share of the graph NOT in the explanation)}
$$

(Variants use the change in *predicted class* instead of probability.) Fidelity and sparsity trade off: a
bigger explanation is usually more sufficient but less sparse, so compare methods at equal sparsity.

Other evaluation tools:

* **Ground-truth benchmarks:** synthetic graphs with planted motifs (BA-Shapes, tree-cycles, the ShapeGGen
  generator) where the "true" explanation is known. Useful but artificial.
* **Deletion / insertion curves:** remove (or add) inputs in the order of their attribution and plot the
  output; a faithful ranking makes the curve fall (rise) fast. ROAR retrains after removal.
* **Sanity checks** (Adebayo et al. 2018): randomise the model's weights (or train on shuffled labels) and
  check that the explanation changes. Some popular saliency methods did *not* change, meaning they were
  explaining the input, not the model.
* **Stability:** Ghorbani et al. (2019) showed that imperceptible input perturbations can drastically change
  saliency maps without changing the prediction. Measure stability across seeds and small perturbations.
* **Agreement between methods:** if attention, occlusion and Shapley values agree, that is reassuring; if
  they disagree, find out why (section 5.4 is the template).

---

## 7. Faithful by construction: additive models and the propagation head

### 7.1 Additive models

A **generalised additive model** (GAM) predicts with a sum of terms, each depending on one input (or one
small group of inputs):

$$
g\bigl(\mathbb{E}[y\mid x]\bigr)=\beta_0+\sum_{v}f_v(x_v).
$$

Linear and logistic regression are the special case $f_v(x_v)=w_vx_v$. The defining advantage: the
contribution of each term is *visible in the formula*. No approximation, sampling or baseline search is
needed to say how much term $v$ added; it added $f_v(x_v)$. Rudin's argument (section 2.2) is that,
whenever accuracy allows, you should build such a model rather than explain a black box after the fact.

### 7.2 The project's multi-view propagation head

`src/drepo/model.py`, `MVHGAT.forward`, computes for drug $i$ and disease $j$:

$$
\text{logit}(i,j)=\underbrace{\mathrm{gate}(i,j)\;h_i^\top W h_j}_{\text{GNN term}}
\;+\;\underbrace{\sum_{v}w_v\,P_v[i,j]}_{\text{propagation head}}\;+\;b,
\qquad \text{score}=\sigma(\text{logit}).
$$

* For a **drug view** $v$ (`chem_cdk`, `chem_ecfp`, `gene_r`): $P_v=K_vA$, where $K_v$ is the
  row-normalised $k$-nearest-neighbour similarity matrix of that view (`data.knn_kernel`, $k=10$, self
  excluded) and $A$ is the matrix of *visible* known links. So
  $P_v[i,j]=\sum_kK_v[i,k]\,A[k,j]$ is "the similarity-weighted share of drug $i$'s ten nearest neighbours
  in view $v$ that are known to treat disease $j$".
* For a **disease view** $u$ (`pheno_mim`, `sem_mondo`, `gene_d`): $P_u=(K_uA^\top)^\top$, so
  $P_u[i,j]=\sum_lK_u[j,l]\,A[i,l]$ is "the share of disease $j$'s nearest diseases that drug $i$ treats".
* $w_v$ = `view_weights()` = `softplus(prop_w) * prop_scale`: one global weight per view, learned.

Define the **contribution of view $v$ to pair $(i,j)$** as

$$
c_v(i,j)=w_v\,P_v[i,j].
$$

### 7.3 Proposition: in the head, $c_v$ is the occlusion effect *and* the Shapley value

Consider the game whose players are the views, with value function "the logit when the views in $S$ keep
their propagation term and the others have $P_v$ set to 0" (all else held fixed):

$$
\text{val}(S)=T+\sum_{v\in S}c_v,\qquad T=\text{GNN term}+b .
$$

*Occlusion on the logit:* $\text{val}(N)-\text{val}(N\setminus\{v\})=c_v$.

*Shapley value:* for every coalition $S\not\ni v$, the marginal contribution is
$\text{val}(S\cup\{v\})-\text{val}(S)=c_v$, the same number for every $S$. A weighted average of a
constant is that constant, and the Shapley weights sum to one, so $\phi_v=c_v$.

*Efficiency:* $\sum_v\phi_v=\sum_vc_v=\text{val}(N)-\text{val}(\varnothing)$. ∎

So for the head, three notions that disagreed everywhere else in this chapter (occlusion, Shapley value,
"the term in the formula") **coincide exactly**. That is what `HOW_IT_WORKS.md` section 4.5 means by
"$w_v\cdot P_v[i,j]$ is *literally* view $v$'s share of the logit... a faithful explanation, not an
approximation".

### 7.4 What the proposition does *not* say

Be precise about the scope; reviewers will be.

1. **It covers the head, not the whole logit.** The GNN term $\mathrm{gate}\cdot h_i^\top Wh_j$ also
   depends on the views (they define the GNN's relations and its input features), but not additively.
   The head decomposition is exact for $\sum_vw_vP_v$ and silent about the rest. A complete statement is:
   "the logit = (GNN term) + (sum of per-view head contributions) + bias, where the head contributions are
   exact".
2. **It is on the logit scale.** On the probability scale, contributions are not additive. A logit
   contribution of 1.0 lowers the probability by 0.23 if the score starts at 0.5 but by only 0.011 if it
   starts at 0.993 (the code below computes this). This is why the breast-cancer predictions in the case
   study, all scored around 0.93 to 0.99, show small occlusion drops (Estrone: score 0.9937, largest drop
   +0.041) while the Alzheimer's ones, scored 0.3 to 0.9, show larger drops. Do not compare probability-scale
   occlusion drops across pairs with very different scores; compare logit-scale ones.
3. **"Share" needs a reference.** $c_v$ is the contribution *relative to $P_v=0$* ("no neighbour evidence
   in this view"). Relative to the average evidence, the Shapley value would be
   $w_v(P_v[i,j]-\overline{P_v})$. Both are legitimate; say which you report.
4. **Faithful to the model is not true about biology.** $c_v$ says what the model did with the evidence,
   not whether the evidence is right (section 10).
5. **A sign caveat specific to this implementation.** The docstring says $w_v\ge0$. `softplus(prop_w)` is
   positive, but it is multiplied by `prop_scale`, a free learned scalar initialised at 5.0, so the
   non-negativity holds only while `prop_scale` stays positive. It is very likely to (it starts large and the
   propagation evidence is positively predictive), but check it once by printing `m.view_weights()` after
   training, and state the observed values in the paper.

### 7.5 Code: Shapley = $w_vP_v$, and why probability-scale drops shrink

```python
# Block 5: in an additive head, w_v * P_v IS the Shapley value of view v (on the logit),
# and why probability-scale occlusion shrinks for very confident predictions.
import itertools
import math

import numpy as np

views = ["chem_cdk", "chem_ecfp", "gene_r", "pheno_mim", "sem_mondo", "gene_d"]
w = np.array([0.8, 1.6, 0.3, 2.4, 1.2, 0.5])          # learned global view weights w_v >= 0
P = np.array([0.30, 0.45, 0.10, 0.60, 0.20, 0.00])    # P_v[i, j] for one drug-disease pair
gnn, b = 0.9, -3.0                                    # gated GNN term and bias for this pair
sig = lambda t: 1 / (1 + np.exp(-t))


def logit(present):                                    # views not in `present` have P_v = 0
    return gnn + sum(w[v] * P[v] for v in present) + b


def shapley(value, d):
    phi = np.zeros(d)
    for i in range(d):
        rest = [j for j in range(d) if j != i]
        for k in range(d):
            for S in itertools.combinations(rest, k):
                wt = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
                phi[i] += wt * (value(S + (i,)) - value(S))
    return phi


d = len(views)
all_v = tuple(range(d))
phi_logit = shapley(logit, d)
phi_prob = shapley(lambda S: sig(logit(S)), d)
occ_prob = [sig(logit(all_v)) - sig(logit(tuple(j for j in all_v if j != i))) for i in all_v]

print(f"logit = {logit(all_v):.3f}   probability = {sig(logit(all_v)):.3f}\n")
print(f"{'view':10s} {'w_v*P_v':>8s} {'Shapley(logit)':>15s} {'Shapley(prob)':>14s} {'occlusion(prob)':>16s}")
for i, v in enumerate(views):
    print(f"{v:10s} {w[i] * P[i]:8.3f} {phi_logit[i]:15.3f} {phi_prob[i]:14.3f} {occ_prob[i]:16.3f}")
print(f"{'sum':10s} {np.sum(w * P):8.3f} {phi_logit.sum():15.3f} {phi_prob.sum():14.3f} {np.sum(occ_prob):16.3f}")
print(f"\nShapley(prob) sums to p(all) - p(no views) = {sig(logit(all_v)) - sig(logit(())):.3f}")

print("\nSame logit contribution (1.0), different starting points:")
for L in (0.0, 2.0, 5.0):
    print(f"  logit {L:3.1f}: p = {sig(L):.4f} -> {sig(L - 1):.4f}, drop = {sig(L) - sig(L - 1):.4f}")
```

Expected output:

```
logit = 0.570   probability = 0.639

view        w_v*P_v  Shapley(logit)  Shapley(prob)  occlusion(prob)
chem_cdk      0.240           0.240          0.045            0.057
chem_ecfp     0.720           0.720          0.140            0.176
gene_r        0.030           0.030          0.006            0.007
pheno_mim     1.440           1.440          0.294            0.344
sem_mondo     0.240           0.240          0.045            0.057
gene_d        0.000           0.000          0.000            0.000
sum           2.670           2.670          0.530            0.641

Shapley(prob) sums to p(all) - p(no views) = 0.530

Same logit contribution (1.0), different starting points:
  logit 0.0: p = 0.5000 -> 0.2689, drop = 0.2311
  logit 2.0: p = 0.8808 -> 0.7311, drop = 0.1497
  logit 5.0: p = 0.9933 -> 0.9820, drop = 0.0113
```

On the logit, the three columns `w_v*P_v`, `Shapley(logit)` and (not printed, but trivially) logit-scale
occlusion are identical. On the probability scale the Shapley values still sum correctly (efficiency, 0.530)
but the occlusion drops do not (0.641 ≠ 0.530), because the sigmoid makes the views interact. The ranking of
views is preserved here; their *sizes* are not comparable across pairs with different scores.

---

## 8. Counterfactual explanations

**Intuition.** People rarely ask "what is the contribution of each feature?". They ask contrastive
questions: "why amantadine *rather than* donepezil?", "what would have to change for this drug *not* to be
predicted?". A **counterfactual explanation** answers with the smallest change to the input that changes
the output in a specified way.

**Definition (Wachter, Mittelstadt & Russell 2017).** Given input $x$, model $f$ and desired output $y'$,

$$
x^\star=\arg\min_{x'}\;\max_{\lambda}\;\lambda\bigl(f(x')-y'\bigr)^2+d(x,x'),
$$

where $d$ is a distance (often an L1 norm, which favours changing few features). In words: stay as close as
possible to the original while reaching the target output.

**In graphs.** CF-GNNExplainer (Lucic et al. 2022) searches for the *minimal set of edge deletions* that
flips a GNN's prediction. For link prediction in this project, the natural counterfactual questions are
about **evidence**, not about the drug's chemistry (we cannot change a molecule's structure, but we can ask
how much the conclusion rests on a particular piece of knowledge):

* "Which single known link, if it were unknown, would push amantadine out of Alzheimer's top 10?"
* "If memantine had never been approved for Alzheimer's, would amantadine still be predicted?"

**A worked counterfactual with the additive head.** Suppose a pair has logit $L=0.57$ (score 0.639, as in
the Block 5 example) and you want to know how much phenotype evidence is needed to keep it above the
decision threshold logit 0 (score 0.5). With everything else fixed, the logit falls by $w_{\rm pheno}\,\Delta P$
when $P_{\rm pheno}$ falls by $\Delta P$, so the score crosses 0.5 when $w_{\rm pheno}\Delta P=0.57$, i.e.
$\Delta P=0.57/2.4\approx0.24$. Since $P_{\rm pheno}=0.60$, the prediction survives losing up to 40% of its
phenotype-neighbour evidence: a *robustness* statement a biologist can understand. For the GNN term no such
closed form exists, and you would search (delete links, re-run the model).

**Properties to watch.** Counterfactuals should be *sparse* (few changes), *proximal*, *plausible*
(realistic inputs) and *actionable*. There are usually many valid counterfactuals (the "Rashomon" problem);
showing one is a choice. And a counterfactual about the *model* ("without memantine's link, the model
would not predict amantadine") is not a counterfactual about *biology* ("without memantine's mechanism,
amantadine would not work").

---

## 9. Communicating explanations to biologists

An explanation is only useful if the person receiving it can act on it correctly. Some rules that work well
with pharmacologists, clinicians and wet-lab biologists:

1. **Speak in evidence, not in architecture.** "Relation `view:chem_ecfp`" means nothing to them; "drugs
   with similar chemical structure (ECFP fingerprints)" does. Keep a translation table in the paper.
2. **Name the training examples.** The most persuasive and checkable explanation in this project is
   neighbour-level: "amantadine is scored highly for Alzheimer's because its nearest chemical neighbour,
   memantine, is a known Alzheimer's drug". Biologists can evaluate *that* in seconds.
3. **Separate three kinds of statement** and label them:
   *model evidence* ("the model's score rests mostly on chemical similarity to memantine"),
   *external support* ("CTD lists no curated link; ClinicalTrials.gov returns 49 records for amantadine and
   Alzheimer disease"), and
   *biological hypothesis* ("amantadine, like memantine, is an NMDA-receptor antagonist; it may share some
   of memantine's symptomatic effect").
4. **Show uncertainty and counter-evidence.** Report the spread across seeds, and the sources that pushed
   the score down.
5. **Avoid causal and clinical language.** Not "amantadine treats Alzheimer's" or "the model discovered
   that"; instead "the model ranks amantadine first among drugs not yet linked to Alzheimer's in this
   benchmark".
6. **Say what the model cannot know:** dose, blood-brain barrier penetration, safety in elderly patients,
   the direction of an effect (CTD "marker/mechanism" links can mean a drug *causes* a phenotype), and
   whether a drug that is structurally similar has the same pharmacology.
7. **Suggest the next step.** Literature check, then a cell or animal model, then (only with much more
   evidence) clinical investigation. Section 10.3 gives the ladder.

**An "explanation card" for one prediction** (a format you can use in the paper's supplement):

```
Prediction: amantadine (DB00915) for Alzheimer disease type 1 (OMIM 104300)
Model score 0.90 (mean of 5 training seeds), rank 1 of 655 drugs not linked to this disease in Cdataset

Why the model says so (faithful to the model)
  - Removing chemical-structure evidence lowers the score the most:
      ECFP chemical similarity  -0.18,  CDK chemical similarity  -0.15   (occlusion, probability scale)
  - In both chemical views the key neighbour is memantine (approved for Alzheimer's in this benchmark);
    it accounts for all of the ECFP neighbour evidence (P = 0.31) and most of the CDK evidence (0.22 of 0.29).
  - Phenotype / ontology neighbours of Alzheimer's do not use amantadine (P = 0).

External information (not used by the model)
  - CTD curated chemical-disease link: none.   Known in Fdataset: no.
  - ClinicalTrials.gov: 49 records match "amantadine" + "Alzheimer disease" (raw search count; not reviewed).

Caveats
  - Structural similarity to memantine is not evidence of the same clinical effect.
  - The trial count includes trials of any design and outcome, and possibly records that only mention
    the drug; it must be read manually before it is cited as support.

Suggested next step: literature review of amantadine in dementia; compare NMDA-receptor pharmacology
of amantadine and memantine.
```

(The numbers come from `results/logs/C_case.log` and the neighbour analysis in section 11.4.)

---

## 10. The limits: model explanations are not biological truth

### 10.1 Five reasons a faithful explanation can still mislead

1. **The model learned the data, including its biases.** Fdataset and Cdataset are curated from
   2011-2016 knowledge; well-studied drugs and diseases have many more links ("literature bias"); CTD
   gene sets reflect what has been studied, not what exists. A faithful explanation faithfully reports
   these biases.
2. **Correlation inside the training data is not mechanism.** The Alzheimer's list in the case study is
   dominated by *Parkinson's* drugs (ropinirole, pramipexole, rasagiline, levodopa, entacapone,
   ethopropazine), most of them driven by the `assoc` relation and by phenotype and ontology similarity.
   The model has learned that drugs for one neurodegenerative disease tend to be linked to related ones.
   That is a real regularity of the *benchmark*, but it does not mean dopaminergic drugs help Alzheimer's
   disease. A plausible-sounding biological story could be invented for each one; that is precisely the
   plausibility trap of section 2.5.
3. **Unknown is treated as negative.** Every explanation is relative to a label matrix in which absent links
   are assumed false. A "counter-evidence" source may be penalising a pair only because a true indication
   was never recorded.
4. **The Rashomon effect.** Many different models fit the same data almost equally well (different seeds,
   architectures, hyper-parameters) and can explain the same prediction differently. An explanation of one
   trained model is one sample from that set. Averaging over seeds, as `06_case_study.py` does, reduces but
   does not remove this.
5. **Explanation methods can be fooled or be fragile.** Slack et al. (2020) built classifiers whose LIME
   and SHAP explanations hid a discriminatory feature; Ghorbani et al. (2019) changed saliency maps with
   imperceptible perturbations; Adebayo et al. (2018) found saliency methods insensitive to the model. Any
   single explanation method is a fallible instrument.

### 10.2 External "support" has limits too

The case-study tables check predictions against CTD, the other benchmark and ClinicalTrials.gov. Read those
columns carefully:

* A **CTD "marker/mechanism"** link means the chemical is associated with the disease or plays a role in
  its mechanism; it can mean the drug *causes or worsens* a phenotype. Only "therapeutic" links indicate a
  (putative) treatment relationship.
* A **ClinicalTrials.gov count** is the number of registry records matched by a search for the condition
  and the intervention. It counts trials that failed, trials that were withdrawn, trials in which the drug is
  a comparator or a background therapy, and possibly records matched only through synonyms. "49 trials"
  is evidence that *someone thought it worth testing*, not evidence of efficacy. Alzheimer's disease in
  particular has a long history of negative trials.
* **"6/10 have independent support"** is a recall-style sanity check, not a validation: the candidate drugs
  were not chosen at random, and drugs that are popular in general are more likely both to be predicted and
  to appear in trials. A fairer comparison would be the support rate of random drugs, or of a simple
  baseline's top 10, for the same disease.

### 10.3 The evidence ladder

```
  model score + faithful explanation                         <- this project
        |  "the model ranks X high because of evidence E"
        v
  independent database / literature support                  <- case-study columns
        |  "others have reported an X-disease association"
        v
  in vitro (cells, organoids), target engagement
        v
  in vivo (animal models)
        v
  observational clinical data (e.g. electronic health records)
        v
  randomised controlled trials                               <- only this establishes efficacy
```

Interpretability helps you choose which hypotheses to send up the ladder and lets experts reject nonsense
early. It does not let you skip a rung. A sentence that belongs in every drug-repositioning paper's
Discussion: *"Predictions are hypotheses for experimental follow-up and must not be used to guide
treatment."*

---

## 11. In this project

This section walks through the project's three interpretability tools in the code, then through the real
outputs. All quotes are from the project files; read them side by side with this text.

### 11.1 Where the three tools sit on the map

| Tool | Code | Intrinsic / post-hoc | Global / local | Faithful? | Scale |
|---|---|---|---|---|---|
| (1) view weights $w_v$ and contributions $w_vP_v[i,j]$ | `MVHGAT.view_weights`, `MVHGATMethod.view_weights` | intrinsic (additive head) | $w_v$ global; $w_vP_v$ local | exactly, for the head term | logit |
| (2) view attention $\beta$ | `ViewAttention`, `MVHGATMethod.view_attention` | intrinsic *mechanism*, but read post hoc | per node (averaged: global) | **not established**; descriptive | weights in [0,1] |
| (3) occlusion of an evidence source | `HeteroLayer(drop_rel=...)`, `MVHGATMethod.occlusion` | post-hoc, model-specific removal rule | local (per pair) | faithful to "what this model says without the source" | probability |

`HOW_IT_WORKS.md` section 4.5 draws the right conclusion: lead with (1) and (3), use (2) as supporting
detail.

### 11.2 The code, line by line

(In the quotes below, docstrings are omitted and one long line is wrapped; the code is otherwise verbatim.)

**View attention** (`src/drepo/model.py`, class `ViewAttention`, method `forward`):

```python
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

* `s` is the score $q^\top\tanh(Pm+b)$, one per relation and node. It is computed **from the message
  itself**, so β is a function of the content (section 5.2).
* Relations with no neighbours for a node (`valid` false) get score $-\infty$, hence β = 0, and the
  softmax re-normalises over the rest. The `uniform=True` branch is the ablation "w/o view attention
  (plain mean)".

**Occlusion inside the layer** (`HeteroLayer.forward`):

```python
        for rel, (dst, src) in self.relations.items():
            out, has = self.gat[rel](h[dst], h[src], graphs[rel])
            if rel in drop_rel:                      # occlusion for explanations
                has = torch.zeros_like(has)
```

Occluding a relation marks it as "no neighbours" for every node, in **every layer**, so it simply
disappears from the view-level softmax: its weight is redistributed (the re-normalisation effect of section
4.3).

**Global view weights and the decoder** (`MVHGAT.view_weights`, `MVHGAT.gnn_gate`, `MVHGAT.forward`):

```python
    def view_weights(self):
        return F.softplus(self.prop_w) * self.prop_scale

    def gnn_gate(self, deg_r, deg_d):
        g = self.gate
        return torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))[:, None] * \
               torch.sigmoid(g[2] + g[3] * torch.log1p(deg_d))[None, :]

    def forward(self, X, graphs, drop_rel=(), P=None, deg=None):
        Hr, Hd, betas = self.encode(X, graphs, drop_rel)
        logits = Hr @ self.W @ Hd.T
        if self.n_prop and P is not None:
            if deg is not None:
                logits = self.gnn_gate(*deg) * logits
            logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
        return logits, betas
```

The last `logits = ...` line is the additive head: `(self.view_weights()[:, None, None] * P).sum(0)` is
$\sum_vw_vP_v$ for every pair at once. Each slice `self.view_weights()[v] * P[v]` is the exact, faithful
contribution $c_v$ of section 7. Note that the GNN term is multiplied by the degree gate, which grows with
the number of known links of the drug and the disease; for a disease with few links the head dominates, so
the faithful part of the explanation covers more of the logit exactly where predictions are hardest.

**The three explanation methods of the wrapper** (`src/drepo/methods.py`, class `MVHGATMethod`):

```python
    @torch.no_grad()
    def view_attention(self):
        model, X, graphs, _, _ = self.last
        _, _, betas = model.encode(X, graphs)
        out = {}
        for tp in ("drug", "disease"):
            names = betas[0][tp][0]
            B = torch.stack([b[tp][1] for b in betas]).mean(0).T.cpu().numpy()
            out[tp] = (names, B)
        return out
```

β is computed for every node, **averaged over the two layers**, and returned as a nodes × relations
array. (Averaging over layers is a summary choice: in layer 2 each message already mixes information
gathered in layer 1 from *all* relations, so "relation r" in layer 2 no longer means "pure view r
evidence".)

```python
    @torch.no_grad()
    def occlusion(self, pairs):
        model, X, graphs, P, deg = self.last
        base = torch.sigmoid(model(X, graphs, P=P, deg=deg)[0])
        groups = {}
        for r in graphs:
            groups.setdefault(r.split(">")[0], []).append(r)
        res = {}
        for g, rels in groups.items():
            Pg = P
            if P is not None and g in self.prop_names:
                Pg = P.clone()
                Pg[self.prop_names.index(g)] = 0
            s = torch.sigmoid(model(X, graphs, drop_rel=set(rels), P=Pg, deg=deg)[0])
            res[g] = np.array([(base[i, j] - s[i, j]).item() for i, j in pairs])
        return res
```

Things to notice, each of which matters when you interpret or report the numbers:

1. **Grouping.** `r.split(">")[0]` puts `assoc>drug` and `assoc>disease` into one group `assoc`;
   each similarity view (`view:chem_ecfp`, ...) is its own group. So the project computes 7 occlusion
   scores per pair (6 views + known links).
2. **What is removed.** For a view: its graph relation (every layer) **and** its propagation-head slice
   `P[v]`. For `assoc`: only the message-passing relation (it has no head slice).
3. **What is *not* removed.** The input features `X` still contain each view's similarity rows
   (`feat_views="all"` concatenates them) and the visible association rows (`feat_assoc=True`); the degree
   gate still sees the true link counts (`deg` is unchanged). So occlusion is a *partial* removal of a
   source, and it **under-estimates** the total dependence on that source, most of all for `assoc`, whose
   information survives in the features. This is one concrete, project-specific reason for "high
   attention, low occlusion impact" (section 14).
4. **Scale.** Drops are on the **probability** scale (`base - s` after `torch.sigmoid`), with the
   saturation consequences of section 7.4.
5. **Mode.** `fit_predict` calls `model.eval()` before storing `self.last`, so dropout is off and the
   occlusion is deterministic for a trained model.
6. **Cost.** One forward pass per group plus one baseline pass for *all* pairs at once: 8 passes. Exact
   Shapley values over the 7 groups would need $2^7=128$ passes; still cheap (exercise 10).

**How the case study uses them** (`scripts/06_case_study.py`):

```python
for s in range(args.seeds):
    m = MVHGATMethod()
    scores.append(m.fit_predict(data, data.A, data.A == 0, seed=s))
    betas.append(m.view_attention())
    vweights.append(m.view_weights())
    pairs = [(i, j) for j in cols for i in range(data.n_drugs)]
    occl.append(m.occlusion(pairs))
S = np.mean(scores, 0)
occ = {g: np.mean([o[g] for o in occl], 0).reshape(len(cols), data.n_drugs) for g in occl[0]}
```

The model is trained on **all** known links (no held-out test set: the goal is discovery, not evaluation)
with 5 seeds; scores, β and occlusion are **averaged over seeds** (a good stability practice). Then for
each predicted drug:

```python
        drivers = sorted(occ, key=lambda g: -occ[g][c, i])
        ...
            "top_evidence": ", ".join(f"{g.replace('view:', '')} ({occ[g][c, i]:+.3f})"
                                      for g in drivers[:2]),
```

the two groups with the largest mean probability drop become the `top_evidence` column. Note that
`vweights` are collected but not written out; printing them (e.g. `pd.DataFrame(vweights)`) would give
you the global $w_v$ per seed for the paper, and the sign check of section 7.4.

### 11.3 Attention versus occlusion on the real outputs

```python
# Block 6: attention vs occlusion on the project's REAL case-study outputs (read-only).
from collections import Counter
from pathlib import Path

import pandas as pd

CS = Path(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\results\case_studies")

# 1) global attention: mean beta per relation, averaged over all nodes and 5 seeds
for side in ("drug", "disease"):
    att = pd.read_csv(CS / f"Cdataset_{side}_view_attention.csv")
    print(f"{side:7s} mean beta:", ", ".join(f"{r.replace('view:', '')} {b:.2f}"
                                            for r, b in zip(att.relation, att.mean_beta)))

# 2) local occlusion: which source is the TOP driver of each of the 30 case-study predictions?
top1 = Counter()
for omim in ("104300", "114480", "176807"):
    df = pd.read_csv(CS / f"Cdataset_OMIM{omim}.csv")
    top1.update(s.split(" (")[0] for s in df.top_evidence)
print("\ntop-1 occlusion driver over 30 predictions:", dict(top1.most_common()))
```

Expected output (from the results produced by the project's run; it will change if you re-run the case
study):

```
drug    mean beta: chem_cdk 0.08, chem_ecfp 0.08, gene_r 0.08, assoc>drug 0.77
disease mean beta: pheno_mim 0.20, sem_mondo 0.17, gene_d 0.17, assoc>disease 0.45

top-1 occlusion driver over 30 predictions: {'assoc': 7, 'chem_ecfp': 5, 'gene_d': 5, 'chem_cdk': 5, 'pheno_mim': 4, 'gene_r': 4}
```

Interpretation:

* **Attention says the encoder is mostly collaborative:** drug nodes put 77% and disease nodes 45% of
  their view attention on the known-link relation.
* **Occlusion says individual predictions are driven by a spread of sources**: known links are the top
  driver of only 7 of 30 predictions, and *all 7 are in the Alzheimer's list* (dopaminergic Parkinson's
  drugs and others whose own known links put them near Alzheimer's). For breast and prostate cancer the top
  drivers are chemical, gene and phenotype views.
* The two tools therefore tell different stories, and section 14 explains why that is expected rather than
  a bug. The paper should present them as answering different questions: *what the encoder mixes* (β) vs
  *what each prediction depends on* (occlusion), with the additive head contributions as the exact
  account of the propagation part.

### 11.4 Opening up one prediction: amantadine for Alzheimer's

The occlusion table says amantadine's score rests on the two chemical views. Because those views also
enter the additive head, we can go one level deeper and ask *which known treatments* produce the evidence,
using $P_v[i,j]=\sum_kK_v[i,k]A[k,j]$:

```python
# Block 7: open up one propagation-head term, P_v[i, j] = sum_k K_v[i, k] * A[k, j],
# to see WHICH neighbour drugs/diseases make the evidence (read-only, CPU, ~2 s).
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\src")
from drepo import data as D                                   # noqa: E402

data = D.load("C")
i = list(data.drug_names).index("Amantadine")
j = list(data.disease_ids).index("104300")                    # Alzheimer disease type 1
print(f"{data.drug_names[i]} -> {data.disease_names[j]}; known link? {bool(data.A[i, j])}\n")

for v in ("chem_ecfp", "chem_cdk"):                           # drug-side views
    K = D.knn_kernel(data.drug_view(v), 10)                   # row-normalised kNN weights
    print(f"P_{v}[amantadine, AD] = {K[i] @ data.A[:, j]:.3f}; neighbours that treat AD:")
    for k in np.argsort(-K[i])[:10]:
        if data.A[k, j]:
            print(f"    {data.drug_names[k]:14s} weight {K[i, k]:.3f}  "
                  f"(similarity {data.drug_view(v)[i, k]:.2f})")

for u in ("pheno_mim", "sem_mondo"):                          # disease-side views
    K = D.knn_kernel(data.disease_view(u), 10)
    print(f"P_{u}[amantadine, AD] = {K[j] @ data.A[i, :]:.3f}  "
          f"(share of AD's neighbour diseases treated by amantadine)")
```

Expected output:

```
Amantadine -> Alzheimer disease type 1; known link? False

P_chem_ecfp[amantadine, AD] = 0.310; neighbours that treat AD:
    Memantine      weight 0.310  (similarity 0.45)
P_chem_cdk[amantadine, AD] = 0.290; neighbours that treat AD:
    Memantine      weight 0.220  (similarity 1.00)
    Valproic Acid  weight 0.070  (similarity 0.32)
P_pheno_mim[amantadine, AD] = 0.000  (share of AD's neighbour diseases treated by amantadine)
P_sem_mondo[amantadine, AD] = 0.000  (share of AD's neighbour diseases treated by amantadine)
```

This is the explanation a pharmacologist wants: **all** of amantadine's ECFP evidence and three quarters of
its CDK evidence come from a single training example, memantine, an approved Alzheimer's drug and a close
structural analogue (memantine is 3,5-dimethyl-amantadine). Two further observations are worth a sentence
each in the paper:

* The benchmark's CDK fingerprint gives amantadine and memantine a similarity of **1.00**: that
  fingerprint cannot tell them apart. The modern ECFP view distinguishes them (0.45). This is a concrete
  illustration of why the project added the ECFP view.
* The whole prediction rests on **one** neighbour. A counterfactual follows immediately: without the
  memantine-Alzheimer link, both chemical $P_v$ terms would fall to (nearly) zero. The explanation therefore
  also tells you how *fragile* the prediction is.

The disease-side terms are 0: none of Alzheimer's ten nearest diseases (by phenotype or ontology) is
treated by amantadine in the benchmark, so the head's support is purely chemical, consistent with the
occlusion ranking.

### 11.5 What to report in the paper, and what to add

**Report:**

1. A table of global view weights $w_v$ (mean ± std over the 5 case-study seeds) and the dataset-average
   β per relation (`08_make_figures.py` already draws the β bar chart `view_attention_<dataset>.png`),
   with a sentence saying β is descriptive.
2. The case-study tables with the top-2 occlusion drivers (already produced), with a footnote: "probability
   drop when the evidence source is removed from the graph and from the propagation head; averaged over 5
   seeds; input features unchanged".
3. One or two neighbour-level explanations like 11.4.

**Easy, valuable additions** (they do not change the model):

* Report occlusion on the **logit** scale as well (replace `torch.sigmoid(...)` by the raw logits) so
  drops are comparable across pairs with different scores.
* Report the **head contributions** $w_vP_v[i,j]$ for each case-study pair: exact and free.
* Compute **exact Shapley values over the 7 evidence groups** for the 30 case-study pairs ($2^7=128$
  forward passes per seed, all pairs at once), to handle the redundancy between the two chemical views.
* Measure **stability**: Spearman correlation of the per-pair occlusion vectors between seeds.
* Run Adebayo's **model-randomisation sanity check**: re-initialise the trained model's weights and confirm
  the occlusion rankings change.

---

## 12. Common mistakes and misconceptions

1. **"Attention weights are importances."** They are mixing coefficients. Contribution also depends on
   message size, alignment with downstream computation, redundancy and parallel pathways (section 5).
2. **"The explanation is plausible, so it is correct."** Plausibility is not faithfulness; experts can
   rationalise almost any feature list (section 2.5).
3. **Forgetting the baseline.** Every attribution explains $f(x)-f(x')$. An IG or SHAP value reported
   without its baseline is uninterpretable.
4. **Mixing logit and probability scales.** Comparing probability-scale occlusion drops across pairs with
   very different scores, or calling probability-scale numbers "additive".
5. **Treating occlusion as Shapley.** Occlusion is one marginal contribution; it misses redundancy
   (two chemical views back each other up) and double-counts synergy.
6. **Concluding "the view is useless" from a small occlusion** (it may be redundant), or "the view is
   essential" from a large one (the model may have learned to use it, but a re-trained model might not
   need it: check the ablation).
7. **Confusing occlusion with ablation.** One probes a trained model, the other retrains (section 4.4).
8. **Over-reading a single seed.** Explanations vary between equally good models; average and report the
   spread.
9. **Hiding counter-evidence.** Mask-based explainers such as GNNExplainer return only the evidence *for*
   the prediction.
10. **Claiming "faithful" for the whole model** when only the additive head is exactly decomposed.
11. **Using causal or clinical language** ("the model discovered that amantadine treats Alzheimer's").
12. **Treating a ClinicalTrials.gov hit count as validation.** It counts registrations, not successes.
13. **Running LIME once with default settings** and reporting it as *the* explanation (kernel width and
    sampling change the answer).
14. **Skipping sanity checks.** If the explanation does not change when the model is randomised, it was
    never explaining the model.

---

## 13. Exercises

Difficulty: ★ routine, ★★ needs thought, ★★★ a small project.

**Exercise 1 ★ (gradient × input on a linear model).** For $f(x)=3x_1-2x_2+0.5x_3+1$ at $x=(2,1,4)$ with
baseline 0, compute the gradient, gradient × input, integrated gradients and Shapley values. Verify
completeness.

<details><summary>Solution</summary>

Gradient $=(3,-2,0.5)$. Gradient × input $=(6,-2,2)$. For a linear $f$ the gradient is constant along the
path, so IG $=(x-x')\odot\nabla f=(6,-2,2)$. Every marginal contribution of feature $i$ is $w_ix_i$
regardless of the coalition, so the Shapley values are also $(6,-2,2)$. Check:
$f(x)-f(0)=(6-2+2+1)-1=6=6-2+2$. ✓ All four methods agree for linear models with a zero baseline; they
diverge only with non-linearity.
</details>

**Exercise 2 ★ (Shapley by hand).** For $f(x)=x_1x_2x_3$ at $x=(1,1,1)$, baseline 0, compute the Shapley
values. Then do the same for $f(x)=x_1x_2+x_3$ at $x=(1,1,1)$.

<details><summary>Solution</summary>

$x_1x_2x_3$: $v(S)=1$ only for $S=N$, else 0. Feature $i$ has a non-zero marginal contribution only when it
arrives last ($S=N\setminus\{i\}$, weight $\frac{2!0!}{3!}=\frac13$), contributing 1. So
$\phi=(\frac13,\frac13,\frac13)$, summing to 1. ✓

$x_1x_2+x_3$: by additivity, $\phi$ = Shapley of $x_1x_2$ + Shapley of $x_3$. For the two-player game
$x_1x_2$ each gets $\frac12$ (symmetry and efficiency), and $x_3$ gets 1 from its own term and 0 from the
other. Total $\phi=(\frac12,\frac12,1)$, sum $2=f(x)-f(0)$. ✓
</details>

**Exercise 3 ★ (integrated gradients by hand).** Compute IG for $f(x)=x_1^2x_2$ at $x=(2,3)$, baseline 0.
Check completeness.

<details><summary>Solution</summary>

Path $\gamma(\alpha)=(2\alpha,3\alpha)$. $\partial f/\partial x_1=2x_1x_2=2(2\alpha)(3\alpha)=12\alpha^2$, so
$\mathrm{IG}_1=2\int_0^112\alpha^2d\alpha=2\cdot4=8$. $\partial f/\partial x_2=x_1^2=4\alpha^2$, so
$\mathrm{IG}_2=3\int_0^14\alpha^2d\alpha=3\cdot\frac43=4$. Sum $=12=f(2,3)=4\cdot3$. ✓ Note the 2:1 split:
along a straight path, a variable entering with power 2 gets twice the credit of one entering with power 1.
(The Shapley values with baseline 0 would be $(6,6)$, since $v$ is non-zero only for the full coalition:
another case where IG and Shapley differ.)
</details>

**Exercise 4 ★ (occlusion vs Shapley).** A model predicts with $f=\max(x_{\rm cdk},x_{\rm ecfp})+0.5\,x_{\rm pheno}$,
all inputs 1, baseline 0. Compute occlusion and Shapley values. Which method would make a reader conclude
that "chemical similarity does not matter"?

<details><summary>Solution</summary>

$f(x)=1+0.5=1.5$. Occlusion: removing CDK leaves $\max(0,1)=1$, drop 0; same for ECFP, drop 0; removing
pheno drops 0.5. Occlusion $=(0,0,0.5)$, summing to 0.5. Shapley: the max term gives each chemical view
0.5 (worked example in section 4.2), pheno gets 0.5: $(0.5,0.5,0.5)$, summing to 1.5. ✓ Occlusion would
suggest chemistry is irrelevant, purely because the two chemical views are redundant. Grouped occlusion
(remove both chemical views) would give a drop of 1.0 and fix the misreading.
</details>

**Exercise 5 ★★ (high attention, low impact).** In Block 3, change `m["chem"]` to `[0.0, 0.1]` (no longer
redundant with `assoc`) and re-run. Predict the change in `assoc`'s occlusion before running. Explain.

<details><summary>Solution</summary>

Prediction: `assoc`'s occlusion impact will rise sharply, because nothing can stand in for it any more.
Running the modified script gives:

```
full model: logit 2.085, probability 0.889

relation  attention beta  occlusion dLogit  occlusion dProb
assoc              0.875             0.803            0.107
chem               0.085             0.122            0.013
pheno              0.040             1.970            0.361
```

`assoc`'s logit-scale occlusion went from 0.032 to 0.803 (25 times larger). Its attention also rose (0.586
to 0.875), but that is not why the impact rose: the impact rose because `chem` no longer duplicates it, so
when `assoc` is removed its weight flows to messages that say something different. Lesson: occlusion
measures irreplaceability given the other sources; attention measures mixing. Note also that `pheno` is
unaffected (its impact comes from the additive head).
</details>

**Exercise 6 ★★ (Shapley efficiency on probabilities).** In Block 5, explain why `Shapley(prob)` sums to
0.530 while `occlusion(prob)` sums to 0.641. Which would you report to a biologist and why?

<details><summary>Solution</summary>

Shapley values satisfy efficiency, so they sum to $p(\text{all views})-p(\text{no views})=\sigma(0.57)-\sigma(-2.1)=0.639-0.109=0.530$.
Occlusion drops are each measured from the full model, whose logit (0.57) is close to 0, where the sigmoid is
steepest; Shapley values average the same logit steps over coalitions whose logits lie lower (down to −2.1),
where the sigmoid is flatter. So each occlusion drop is larger than the corresponding Shapley value, and the
drops do not add up to anything in particular (here they over-count). For a biologist, report logit-scale head
contributions or Shapley values (they add up and are comparable), and translate the total into a
probability change once. Probability-scale occlusion is fine as a ranking within one pair.
</details>

**Exercise 7 ★★ (GNNExplainer settings).** In Block 4 set the size penalty to 0 (`0.0 * mask.sum()`).
What happens to edges 3 to 5? Then try 0.5 and 2.0. What happens, and what does that teach you?

<details><summary>Solution</summary>

Running the variants gives:

```
size penalty 0.0 : {'1->0': 0.99, '2->0': 0.99, '3->0': 1.0, '4->0': 1.0, '5->0': 1.0, '6->0': 0.01}  -> S = {1,2,3,4,5}
size penalty 0.05: {'1->0': 0.98, '2->0': 0.98, '3->0': 0.0, '4->0': 0.0, '5->0': 0.0, '6->0': 0.01}  -> S = {1,2}
size penalty 0.5 : {'1->0': 0.6,  '2->0': 0.6,  '3->0': 0.0, '4->0': 0.0, '5->0': 0.0, '6->0': 0.0}   -> S = {1,2}
size penalty 2.0 : {'1->0': 0.21, '2->0': 0.21, '3->0': 0.0, '4->0': 0.0, '5->0': 0.0, '6->0': 0.0}   -> S = {} (empty)
```

With no size penalty, edges 3 to 5 carry small positive evidence (0.1 each), so keeping them slightly
raises the prediction and the optimiser keeps them: the explanation is less sparse. Edge 6 (counter-evidence)
is always dropped. With a penalty of 0.5 the important edges survive only just (mask 0.6), and with 2.0 the
penalty outweighs the evidence and the "explanation" is empty. The explanation is a product of the
objective's trade-off, not a fact about the model alone: always report the explainer's hyper-parameters
and compare explainers at a fixed sparsity.
</details>

**Exercise 8 ★★ (fidelity).** For Block 4's star graph, compute fidelity+ and fidelity− for the
explanation $S=\{1\}$ (edge 1 only), by hand. Compare with $S=\{1,2\}$.

<details><summary>Solution</summary>

$S=\{1\}$. Without $S$: $h_0=2+0.3-1=1.3$, logit $1.5\cdot1.3-2=-0.05$, $p=0.488$; fidelity+ $=0.950-0.488=0.462$.
Only $S$: $h_0=2$, logit 1, $p=0.731$; fidelity− $=0.950-0.731=0.219$. Sparsity $=5/6=0.833$.
For $S=\{1,2\}$: fidelity+ 0.905, fidelity− −0.032, sparsity 0.667. The larger explanation is both more
necessary and sufficient, at the cost of sparsity. To compare explainers fairly, fix sparsity.
</details>

**Exercise 9 ★★ (counterfactual with the head).** A pair has logit 1.2, and its only non-zero head term is
$w_{\rm ecfp}P_{\rm ecfp}=1.5$ with $w_{\rm ecfp}=3$. By how much must $P_{\rm ecfp}$ fall for the score to
drop below 0.5? Express it as a fraction of the current $P_{\rm ecfp}$.

<details><summary>Solution</summary>

Score $<0.5\iff$ logit $<0$, so the logit must fall by more than 1.2, i.e. $3\,\Delta P>1.2$,
$\Delta P>0.4$. Current $P_{\rm ecfp}=1.5/3=0.5$, so it must lose more than $0.4/0.5=80\%$ of its
ECFP-neighbour evidence. Removing the view entirely (ΔP = 0.5) would take the logit to −0.3 and the score
to $\sigma(-0.3)=0.43$. This assumes the GNN term is unchanged, which is exact for the head and only an
approximation for the whole model.
</details>

**Exercise 10 ★★★ (exact Shapley over evidence groups in the project).** Write a function
`shapley_groups(method, pairs)` for a trained `MVHGATMethod` that computes exact Shapley values of the 7
evidence groups on the **logit** for a list of pairs, re-using the occlusion mechanism (absent group =
dropped relation + zeroed head slice). How many forward passes does it need? Do not run it while a GPU job
is running.

<details><summary>Solution</summary>

```python
import itertools, math
import torch

@torch.no_grad()
def shapley_groups(method, pairs):
    model, X, graphs, P, deg = method.last
    groups = {}
    for r in graphs:
        groups.setdefault(r.split(">")[0], []).append(r)
    names = list(groups)
    d = len(names)
    ii = torch.tensor([i for i, _ in pairs]); jj = torch.tensor([j for _, j in pairs])
    cache = {}
    def value(present):                                 # logits of all pairs with only `present` groups
        key = frozenset(present)
        if key not in cache:
            absent = [g for g in names if g not in key]
            Pg = P.clone() if P is not None else None
            for g in absent:
                if Pg is not None and g in method.prop_names:
                    Pg[method.prop_names.index(g)] = 0
            drop = {r for g in absent for r in groups[g]}
            cache[key] = model(X, graphs, drop_rel=drop, P=Pg, deg=deg)[0][ii, jj]
        return cache[key]
    phi = {g: torch.zeros(len(pairs)) for g in names}
    for a, g in enumerate(names):
        rest = names[:a] + names[a + 1:]
        for k in range(d):
            w = math.factorial(k) * math.factorial(d - k - 1) / math.factorial(d)
            for S in itertools.combinations(rest, k):
                phi[g] += w * (value(S + (g,)) - value(S)).cpu()
    return phi
```

With caching, each of the $2^7=128$ coalitions is evaluated once (one forward pass gives all pairs), so 128
passes per trained model. Efficiency check: $\sum_g\phi_g$ should equal the logit with all groups minus
the logit with none. Caveat: "no groups" still has input features, so the "empty" baseline is not an empty
model; state this baseline in the paper.
</details>

**Exercise 11 ★★ (reading the case-study table critically).** In the Alzheimer's table, ropinirole's
top drivers are `assoc (+0.311), gene_r (+0.171)`. Explain in plain words what each driver means for this
drug, and write one sentence for the paper that is faithful and avoids over-claiming.

<details><summary>Solution</summary>

`assoc`: removing the known-link relation from message passing lowers the score by 0.31; ropinirole's own
known indications (e.g. Parkinson's-related diseases) and the drugs sharing diseases with it shape its
embedding so that it sits near Alzheimer's drugs. `gene_r`: removing the drug gene-overlap view (CTD
chemical-gene sets) lowers it by 0.17; drugs with overlapping gene sets are linked to Alzheimer's or related
diseases. Sentence: "Ropinirole's score is driven mainly by its known-link neighbourhood and by gene-set
similarity to other drugs, consistent with the model's tendency to transfer indications among
neurodegenerative diseases; no independent support for an Alzheimer's indication was found (CTD: none;
ClinicalTrials.gov: 0 records)."
</details>

**Exercise 12 ★★ (attention debate).** A reviewer writes: "The authors' Figure X shows view attention; by
Jain & Wallace (2019) attention is not explanation, so the interpretability claim is invalid." Draft a
4-6 sentence response.

<details><summary>Solution</summary>

"We agree that attention weights are not, on their own, faithful explanations (Jain & Wallace 2019), and
we do not use them as such. Following Wiegreffe & Pinter (2019), we treat view attention as a description
of how the encoder mixes evidence, and we report a uniform-attention ablation (AUPR 0.478 vs 0.500 on
Fdataset) bounding its role. Our per-prediction explanations are instead based on (i) the additive
propagation head, whose per-view terms are exact contributions to the logit (Section X; they equal both the
occlusion effect and the Shapley value on the logit), and (ii) occlusion of each evidence source. We have
revised the text and the caption of Figure X to make this distinction explicit."
</details>

**Exercise 13 ★★★ (sanity check).** Design (in words and pseudo-code) Adebayo's model-randomisation test
for the project's occlusion and say what result would worry you.

<details><summary>Solution</summary>

Train MV-HGAT, compute occlusion rankings for the 30 case-study pairs. Then, layer by layer from the top
(decoder `W`, view weights, then the GNN layers), re-initialise the parameters randomly and recompute the
rankings; compute the Spearman correlation with the original rankings after each step. Expected: correlation
falls towards 0 as more of the model is randomised. Worrying result: rankings stay highly correlated with a
randomised model, which would mean they are determined by the input data (e.g. which views have
neighbours) rather than by what the model learned. Pseudo-code: `for block in [W, prop_w, layers[1],
layers[0]]: reinit(block); r = spearman(occ_orig, occlusion(model))`.
</details>

---

## 14. Answers to the PREREQUISITES.md self-check

> **Self-check (E2):** *Why might a relation get high attention but low occlusion impact?*

Short answer: attention measures how much weight a node *places* on a relation's message when it mixes its
messages; occlusion measures how much the *final output changes* when the relation is taken away. Many
things sit between the two, so they can disagree in either direction.

In depth, there are general reasons and reasons specific to MV-HGAT.

**General reasons (true of any attention model):**

1. **Redundancy plus re-normalisation.** If another relation carries similar information, removing the
   high-attention relation hands its weight to the substitute through the softmax, and the output barely
   moves (Block 3: `assoc` β 0.586, occlusion 0.003, because `chem` duplicates it). Occlusion measures
   irreplaceability, not use.
2. **Small or misaligned messages.** The contribution of relation $r$ is roughly
   $\beta_r\times$(message size in the direction the downstream layers use). A high β on a small, or
   orthogonal, message contributes little.
3. **Downstream dilution.** After the attention come skip connections, LayerNorm, another layer and the
   decoder; the attended vector is only part of the next representation, and normalisation can shrink
   differences.
4. **Non-identifiability of attention.** When messages are similar, many attention distributions give the
   same output (Jain & Wallace's counterfactual attention); the particular split learned is partly
   arbitrary, so a high β need not mean high necessity.
5. **Output saturation.** If the score is already close to 1 (or 0), even a substantial change in the
   logit produces a tiny change in probability. Occlusion on the probability scale then looks small
   whatever the attention (breast-cancer predictions at 0.93 to 0.99).
6. **Mixing across layers.** In layer 2 a "chemical" message already contains information gathered in layer
   1 from all relations, so occluding one relation does not remove "its" information from the other
   relations' messages, and β of layer 2 is not purely about that relation's source data.

**Reasons specific to MV-HGAT:**

7. **The known-link information survives occlusion.** `MVHGATMethod.occlusion` removes the `assoc`
   relation from message passing, but the visible association rows remain in the input features
   (`feat_assoc=True`) and the degree gate still sees the link counts. So the relation with the highest β
   (`assoc`: 0.77 for drugs, 0.45 for diseases in Cdataset) can show modest occlusion impact: its
   information enters the model by other doors. In the real outputs it is the top occlusion driver for only
   7 of 30 case-study predictions.
8. **A parallel pathway that bypasses attention.** The propagation head adds $w_vP_v$ directly to the
   logit, independent of β. A view with *low* attention can therefore have *high* occlusion impact (Block
   3's `pheno`: β 0.027, occlusion 0.348), and a view's high attention says nothing about its head term.
9. **β is averaged** over layers, over nodes (for the global table) and over seeds; a relation can have high
   average attention while being unimportant for the particular pair being explained, whose occlusion is
   local.

**The reverse case** (low attention, high occlusion) arises from reasons 2 and 8: a large message with a
small weight, or a pathway such as the propagation head that does not go through the attention at all.

**What to do about it:** treat β as descriptive, explain individual predictions with occlusion and the
exact head contributions, use grouped occlusion or Shapley values over groups when sources are redundant,
check stability across seeds, and run the uniform-attention ablation to bound how much the attention
mechanism matters at all.

---

## 15. Summary and cheat sheet

**Big ideas**

* Explanations describe the **model**, not the data or the world. Faithful ≠ plausible ≠ true.
* Always state: **method, output scale (logit/probability), baseline / removal rule, global or local**.
* **Completeness/efficiency** (IG, Shapley) is what makes attributions add up; gradients, GxI, LIME and
  occlusion do not have it.
* **Occlusion** = one marginal contribution: blind to redundancy, double-counts synergy. **Ablation**
  retrains and answers a different question.
* **Attention ≠ explanation** without further tests (Jain & Wallace); it can be *an* explanation if it
  passes them (Wiegreffe & Pinter).
* **Additive heads are faithful by construction:** $w_vP_v$ = occlusion = Shapley on the logit, for that
  term only.
* **Model explanation → hypothesis**, never → clinical conclusion.

**Formulas**

| Quantity | Formula |
|---|---|
| Gradient × input | $x_i\,\partial f/\partial x_i$ |
| Integrated gradients | $(x_i-x'_i)\int_0^1\partial_if(x'+\alpha(x-x'))\,d\alpha$; $\sum_i\mathrm{IG}_i=f(x)-f(x')$ |
| Shapley value | $\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{\lvert S\rvert!\,(d-\lvert S\rvert-1)!}{d!}[v(S\cup\{i\})-v(S)]$ |
| Shapley axioms | efficiency, symmetry, dummy, additivity (unique) |
| Shapley kernel (KernelSHAP) | $\pi(z')=\frac{d-1}{\binom{d}{\lvert z'\rvert}\lvert z'\rvert(d-\lvert z'\rvert)}$ |
| LIME | $\arg\min_g\mathcal{L}(f,g,\pi_x)+\Omega(g)$ |
| Occlusion | $f(x)-f(x_{\setminus G})$ |
| GNNExplainer | $\min_M -\log P(y\mid A\odot\sigma(M))+\lambda_1\lVert\sigma(M)\rVert_1+\lambda_2H(\sigma(M))$ |
| Fidelity± / sparsity | $f(G)-f(G\setminus S)$; $f(G)-f(S)$; $1-\lvert S\rvert/\lvert E\rvert$ |
| Counterfactual | $\arg\min_{x'}\lambda(f(x')-y')^2+d(x,x')$ |
| MV-HGAT head contribution | $c_v(i,j)=w_vP_v[i,j]$, $P_v=K_vA$ (drug view) |

**Project quick reference**

| Need | Call |
|---|---|
| global view weights | `m.view_weights()` |
| per-node β (layers averaged) | `m.view_attention()` |
| occlusion per pair (probability drop) | `m.occlusion(pairs)` |
| exact head contribution | `m.last[0].view_weights()[v] * m.last[3][v]` (logit) |
| neighbour-level evidence | `data.knn_kernel(view, 10)[i] * A[:, j]` |

---

## 16. Further resources

All links were checked on 2026-10-01. "Free" = openly readable; "paid" = publisher paywall (a free preprint
is noted when one exists).

**Books and surveys**

* Molnar, *Interpretable Machine Learning* (online book; the best single overview of SHAP, LIME, PDP,
  counterfactuals) - free - <https://christophm.github.io/interpretable-ml-book/>
* Yuan, Yu, Gui & Ji, "Explainability in Graph Neural Networks: A Taxonomic Survey" (TPAMI 2022; taxonomy
  and fidelity/sparsity metrics) - free preprint - <https://arxiv.org/abs/2012.15445>
* Jiménez-Luna, Grisoni & Schneider, "Drug discovery with explainable artificial intelligence" (Nat. Mach.
  Intell. 2020; XAI in the drug-discovery context) - paid (preprint on arXiv) -
  <https://www.nature.com/articles/s42256-020-00236-4>

**Foundations and position papers**

* Doshi-Velez & Kim (2017), "Towards a rigorous science of interpretable machine learning" - free -
  <https://arxiv.org/abs/1702.08608>
* Lipton (2016), "The Mythos of Model Interpretability" - free - <https://arxiv.org/abs/1606.03490>
* Rudin (2019), "Stop explaining black box machine learning models for high stakes decisions and use
  interpretable models instead" (Nat. Mach. Intell.) - free preprint <https://arxiv.org/abs/1811.10154>;
  journal version (paid) <https://www.nature.com/articles/s42256-019-0048-x>
* Jacovi & Goldberg (2020), "Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate
  Faithfulness?" - free - <https://aclanthology.org/2020.acl-main.386/>

**Attribution methods**

* Simonyan, Vedaldi & Zisserman (2013), saliency maps - free - <https://arxiv.org/abs/1312.6034>
* Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep Networks" (integrated gradients) -
  free - <https://arxiv.org/abs/1703.01365>
* Shrikumar, Greenside & Kundaje (2017), DeepLIFT - free - <https://arxiv.org/abs/1704.02685>
* Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions" (SHAP) - free -
  <https://arxiv.org/abs/1705.07874>
* Ribeiro, Singh & Guestrin (2016), "Why Should I Trust You?" (LIME) - free - <https://arxiv.org/abs/1602.04938>
* Zeiler & Fergus (2014), "Visualizing and Understanding Convolutional Networks" (occlusion) - free -
  <https://arxiv.org/abs/1311.2901>
* Sturmfels, Lundberg & Lee (2020), "Visualizing the Impact of Feature Attribution Baselines" (Distill) -
  free - <https://distill.pub/2020/attribution-baselines/>
* Janzing, Minorics & Blöbaum (2020), "Feature relevance quantification in explainable AI: A causal
  problem" - free - <https://arxiv.org/abs/1910.13413>
* Kumar et al. (2020), "Problems with Shapley-value-based explanations as feature importance measures" -
  free - <https://arxiv.org/abs/2002.11097>
* Koh & Liang (2017), "Understanding Black-box Predictions via Influence Functions" - free -
  <https://arxiv.org/abs/1703.04730>
* Wachter, Mittelstadt & Russell (2017), counterfactual explanations - free - <https://arxiv.org/abs/1711.00399>

**Attention as explanation**

* Jain & Wallace (2019), "Attention is not Explanation" - free - <https://aclanthology.org/N19-1357/>
* Wiegreffe & Pinter (2019), "Attention is not not Explanation" - free - <https://aclanthology.org/D19-1002/>
* Serrano & Smith (2019), "Is Attention Interpretable?" - free - <https://aclanthology.org/P19-1282/>

**Graph neural network explanations**

* Ying et al. (2019), "GNNExplainer: Generating Explanations for Graph Neural Networks" - free -
  <https://arxiv.org/abs/1903.03894>
* Luo et al. (2020), "Parameterized Explainer for Graph Neural Network" (PGExplainer) - free -
  <https://arxiv.org/abs/2011.04573>
* Yuan et al. (2021), "On Explainability of Graph Neural Networks via Subgraph Explorations" (SubgraphX) -
  free - <https://arxiv.org/abs/2102.05152>
* Schlichtkrull, De Cao & Titov (2021), GraphMask - free - <https://arxiv.org/abs/2010.00577>
* Lucic et al. (2022), "CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks" - free -
  <https://arxiv.org/abs/2102.03322>
* Sanchez-Lengeling et al. (2020), "Evaluating Attribution for Graph Neural Networks" (NeurIPS) - free -
  <https://proceedings.neurips.cc/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html>
* Amara et al. (2022), "GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph
  Neural Networks" - free - <https://arxiv.org/abs/2206.09677>
* Agarwal et al. (2023), "Evaluating explainability for graph neural networks" (Sci. Data; GraphXAI,
  ShapeGGen) - free - <https://www.nature.com/articles/s41597-023-01974-x>

**Evaluation, sanity checks and failure modes**

* Adebayo et al. (2018), "Sanity Checks for Saliency Maps" - free - <https://arxiv.org/abs/1810.03292>
* Hooker et al. (2019), "A Benchmark for Interpretability Methods in Deep Neural Networks" (ROAR) - free -
  <https://arxiv.org/abs/1806.10758>
* Ghorbani, Abid & Zou (2019), "Interpretation of Neural Networks is Fragile" - free -
  <https://arxiv.org/abs/1710.10547>
* Kindermans et al. (2017/2019), "The (Un)reliability of saliency methods" - free -
  <https://arxiv.org/abs/1711.00867>
* Slack et al. (2020), "Fooling LIME and SHAP" - free - <https://arxiv.org/abs/1911.02508>

**Software**

* Captum (PyTorch attribution library: IG, SHAP variants, occlusion) - free - <https://captum.ai/>
* PyTorch Geometric explainability module (GNNExplainer, PGExplainer, Captum bridge) - free -
  <https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html>
* SHAP library documentation - free - <https://shap.readthedocs.io/en/latest/>

---

## 17. Glossary

* **Ablation** - removing a component or input source before training and re-training, to measure whether
  the method needs it.
* **Additive model / GAM** - a model whose output is a sum of terms, each depending on one input or group;
  each term's contribution is read directly.
* **Attention weight** - a softmax-normalised mixing coefficient; in MV-HGAT, α (over neighbours) and β
  (over relations).
* **Aumann-Shapley value** - the continuous analogue of the Shapley value; equals integrated gradients on
  the straight path.
* **Baseline (reference)** - the input representing "absence", relative to which an attribution explains
  $f(x)-f(x')$.
* **Completeness / efficiency** - attributions sum to the output minus the baseline output.
* **Counterfactual explanation** - the smallest change to the input that changes the output as specified.
* **Faithfulness (fidelity)** - how accurately an explanation reflects the model's actual computation.
* **Fidelity+ / Fidelity−** - output drop when the explanation is removed / when only the explanation is
  kept.
* **Global explanation** - describes the model's behaviour overall; **local** - one prediction.
* **Gradient × input** - attribution $x_i\,\partial f/\partial x_i$; exact for linear models with zero
  baseline.
* **Integrated gradients (IG)** - path integral of gradients from a baseline to the input, times the input
  difference; satisfies completeness.
* **Interventional vs conditional Shapley** - "absent" features set to baseline/independent values vs
  sampled conditionally on the present ones.
* **Intrinsic (glass-box) interpretability** - the model's structure is the explanation; **post-hoc** - a
  separate method explains a trained model.
* **KernelSHAP** - weighted linear regression on coalitions with the Shapley kernel; recovers Shapley
  values.
* **LIME** - local interpretable model-agnostic explanation by fitting a weighted simple surrogate.
* **Model-agnostic / model-specific** - needs only queries / needs internals.
* **Occlusion (leave-one-out, perturbation)** - output change when one input group is removed.
* **PGExplainer** - a trained network that predicts edge masks for any instance.
* **GNNExplainer** - per-instance optimisation of a soft edge (and feature) mask that preserves the
  prediction with few edges.
* **Plausibility** - how convincing an explanation is to a human, regardless of faithfulness.
* **Propagation head** - MV-HGAT's additive decoder term $\sum_vw_vP_v[i,j]$.
* **Rashomon effect** - many near-equally good models that explain the data differently.
* **Redundancy / synergy** - two inputs that can each replace the other (OR) / that only matter together
  (AND).
* **Saliency map** - visualisation of gradient magnitudes.
* **Sanity check (model randomisation)** - an explanation should change when the model's weights are
  randomised.
* **Saturation** - a flat region of a function where gradients vanish though the input mattered.
* **Shapley value** - the unique fair allocation of a coalition game's payout satisfying efficiency,
  symmetry, dummy and additivity.
* **Sparsity (of an explanation)** - share of the input not included in the explanation.
* **View (evidence source)** - in this project, one similarity matrix (`chem_cdk`, `chem_ecfp`, `gene_r`,
  `pheno_mim`, `sem_mondo`, `gene_d`) or the known-link relation (`assoc`).
