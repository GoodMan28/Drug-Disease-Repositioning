# Unit E3: Reproducibility and writing up results

*Track E (Research practice), unit 3 of 3, and the last unit of the course. Part of the self-contained
course that goes with the project "Heterogeneous Graph-Based Computational Drug Repositioning for
Disease-Drug Association Prediction" (MV-HGAT).*

---

## 0. Before you start

**Prerequisites**

| You need | Where it is taught | Why you need it here |
|---|---|---|
| Mean, standard deviation, the t distribution, p-values | any first statistics course; unit B1 | reporting mean ± std, confidence intervals, paired tests |
| AUC and AUPR, k-fold CV, leave-one-disease-out | units B1, B2 | these are the numbers you will report |
| Leakage, fair baselines, ablation design, repeated runs | unit E1 | honest comparisons build on it |
| Interpretability and its limits | unit E2 | the case-study and interpretability parts of the paper |
| Running the project's scripts; reading JSON results | `docs/HOW_IT_WORKS.md` section 7 | every example uses the real outputs |
| Basic git (commit, log) | any git tutorial | version control is part of reproducibility |

**Estimated study time:** 16 to 20 hours: 7 h reading, 3 h running and adapting the code, 5 h exercises,
and 2 to 4 h producing a first full outline of your own paper using section 14 (strongly recommended: it is
the real deliverable of this unit).

**Learning objectives.** When you finish this unit you should be able to:

1. **Define** reproducibility, replicability, robustness and generalisability, and say which of them a given
   check establishes.
2. **List** the sources of non-determinism in a PyTorch/CUDA pipeline and **apply** the controls (seeds,
   generators, deterministic algorithms, environment variables), knowing what each one does and does not
   guarantee.
3. **Produce** a run manifest that records configuration, code version, software environment, hardware,
   and the version, checksum, licence and access date of every data source.
4. **Compute and correctly label** mean ± standard deviation, standard error and confidence intervals over
   cross-validation runs, explain why fold-level intervals are too narrow, and **run** a paired comparison
   between two methods evaluated on the same folds (including the Nadeau-Bengio correction).
5. **Design** tables and figures that follow good practice (one axis, colour-blind-safe colours, direct
   labels, honest axes, data shown) and **critique** a figure that does not.
6. **Distinguish** re-implemented from published baseline numbers and ensure tuning and information parity.
7. **Write** each section of a bioinformatics methods paper to the standard reviewers expect, including a
   Methods section that a stranger could re-run, Results 5.1-5.7 with honest claims, and a Limitations
   paragraph.
8. **Apply** reproducibility checklists (NeurIPS paper checklist, the ML Reproducibility Checklist, DOME,
   MIABi, journal policies) and **write** data and code availability statements.
9. **Respond** to reviewer comments in a way that maximises the chance of acceptance.

---

## 1. Motivation: three true stories from this project

**Story 1: a result that reproduced to the last digit.** The main 5-fold run
(`results/Fdataset/cv5/mvhgat.json`) and the "Full MV-HGAT" row of the ablation
(`results/Fdataset/ablation_cv5/full.json`) were produced by two different scripts, in two different
processes, at different times. Their first-repeat fold metrics are identical:

```
fold   main run AUC / AUPR      ablation "full" AUC / AUPR
  0    0.94018 / 0.50918        0.94018 / 0.50918
  1    0.93951 / 0.46586        0.93951 / 0.46586
  2    0.93493 / 0.51194        0.93493 / 0.51194
  3    0.95568 / 0.54324        0.95568 / 0.54324
  4    0.94666 / 0.47155        0.94666 / 0.47155
```

That did not happen by luck. The folds are drawn with a seeded generator (`kfold_splits` uses
`np.random.default_rng(seed)`), every model fit receives a seed derived from the repeat and fold
(`seed * 1000 + r * 100 + f` in `run_kfold`), and `set_seed` seeds every random number generator before
training. The same code, data, hardware and software gave the same answer: **reproducibility** in its
strictest sense.

**Story 2: the same seed, a different answer.** While writing unit E2 we trained MV-HGAT for 5 epochs with
seed 0 once on the GPU and once on the CPU and computed the same quantity (a logit difference for two
drug-disease pairs). The GPU gave −5.98 and −5.34; the CPU gave −5.88 and −4.69. Nothing was wrong: random
numbers drawn on the GPU (`torch.rand(..., device="cuda")`) come from a different generator than those drawn
on the CPU, and GPU kernels add numbers in a different order. PyTorch's documentation says so explicitly:
results are not guaranteed to be identical across CPU and GPU, platforms or releases, even with identical
seeds. So "we fixed the seed" is not the whole story; you must also record *where* and *with what* you ran.

**Story 3: the headline claim that the data do not support.** On Fdataset, 5-fold CV, MV-HGAT has AUC
0.939 ± 0.007 against SCMFDD's 0.893 ± 0.010, but AUPR 0.488 ± 0.027 against SCMFDD's 0.495 ± 0.020. A
paper that says "MV-HGAT outperforms all baselines" is false for AUPR, the metric `HOW_IT_WORKS.md` itself
calls "the honest one" for this imbalanced problem. A reviewer looking at the precision-recall curve will
see it in seconds. The honest sentence is different, more interesting, and still publishable (section 7.6).

These three stories are what this unit is about: making results that others (including future you) can
re-create; reporting their uncertainty properly; comparing fairly; and writing it all up so that a
reviewer trusts it. In computational biology these are not formalities: Sandve et al. (2013) open their
"Ten simple rules for reproducible computational research" by noting that the reproducibility of
computational results is a basic requirement of science that is often not met, and studies of the machine
learning literature (Kapoor & Narayanan 2023, unit E1) have traced many published claims to leakage and
undocumented choices.

---

## 2. Reproducibility, replicability, robustness, generalisability

Different communities use these words differently, so define them in your paper. The most widely used
modern scheme is the 2 × 2 matrix of *The Turing Way*:

```
                         SAME data                DIFFERENT data
                    +------------------------+------------------------+
  SAME analysis     |  REPRODUCIBLE          |  REPLICABLE            |
                    |  re-run our code on    |  run our method on a   |
                    |  Fdataset -> same AUC  |  new benchmark -> a    |
                    |                        |  similar conclusion    |
                    +------------------------+------------------------+
  DIFFERENT         |  ROBUST                |  GENERALISABLE         |
  analysis          |  re-implement in DGL,  |  different code AND    |
                    |  or other splits ->    |  different data ->     |
                    |  same conclusion       |  same conclusion       |
                    +------------------------+------------------------+
```

The U.S. National Academies' report *Reproducibility and Replicability in Science* (2019) uses the same
split between the first two: **reproducibility** means obtaining consistent results using the same input
data, computational steps, methods, code and conditions of analysis; **replicability** means obtaining
consistent results across studies that address the same question, each with its own data. Goodman,
Fanelli & Ioannidis (2016) instead distinguish *methods* reproducibility (enough detail to repeat the
procedure), *results* reproducibility (a new study gets the same result) and *inferential* reproducibility
(the same conclusions are drawn from the results).

It also helps to distinguish three **strengths** of reproduction:

| Strength | Meaning | Achievable when |
|---|---|---|
| Bitwise | identical bits in every output | same code, data, seeds, software versions, hardware and deterministic kernels |
| Numerical | equal within a small tolerance (say 1e-4 on AUC) | same code and data; small software/hardware differences |
| Statistical / conclusion-level | different runs, same conclusion within the reported uncertainty | always the goal; the one reviewers and readers actually need |

The project currently achieves bitwise reproducibility on one machine (story 1). Across machines it can
realistically promise numerical-to-statistical reproducibility, and that is why reporting **variation over
seeds and folds** (section 5) matters more than chasing identical bits.

Where the project's own checks sit in the matrix:

* re-running `run_all.ps1` → reproducibility;
* cross-dataset validation (train on Fdataset, test on links only Cdataset has, `07_cross_dataset.py`) →
  a small step towards replicability (new data, same analysis);
* the ablation "benchmark similarities only" and 10-fold vs 5-fold CV → robustness of the conclusion to
  analysis choices.

---

## 3. Non-determinism and how to control it

### 3.1 Pseudo-random number generators

Computers generate "random" numbers with deterministic algorithms (pseudo-random number generators,
PRNGs). A PRNG has an internal **state**; the **seed** sets the initial state; every draw advances it. Same
seed + same sequence of draws = same numbers. A typical PyTorch project touches *several independent*
PRNGs:

| Generator | Used by | Seeded by |
|---|---|---|
| Python `random` | `random.shuffle`, some libraries | `random.seed(s)` |
| NumPy legacy global | `np.random.rand`, `np.random.shuffle`, scikit-learn when `random_state=None` | `np.random.seed(s)` |
| NumPy `Generator` objects | `np.random.default_rng(s)` (the project's fold splits) | the seed passed when created |
| PyTorch CPU generator | `torch.rand`, `torch.randn`, weight initialisation, dropout on CPU | `torch.manual_seed(s)` |
| PyTorch CUDA generators (one per GPU) | the same operations on GPU tensors | `torch.manual_seed(s)` (seeds all devices) or `torch.cuda.manual_seed_all(s)` |
| DataLoader worker processes | random augmentation inside workers | `worker_init_fn` + a `generator` passed to the DataLoader |
| Python hash randomisation | iteration order of `set`s of strings | environment variable `PYTHONHASHSEED`, set *before* Python starts |

The project's `set_seed` (`src/drepo/methods.py`) covers the first, second, fourth and fifth rows:

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

(`torch.manual_seed` already seeds CUDA devices, so the last line is redundant but harmless.) The fold
splits use a *local* `Generator` (row three), which is good practice: local generators isolate randomness,
so an extra random draw somewhere else cannot silently shift the folds. The project has no DataLoader and
iterates over sets only for membership tests, so the last two rows do not bite.

A subtle but important point is **what the seed is attached to**. In `run_kfold` the fit seed is
`seed * 1000 + r * 100 + f`, a deterministic function of the repeat and fold. That is why every method sees
the same folds *and* each (repeat, fold) has its own, reproducible model initialisation. Re-running one
fold alone gives the same result as running it inside the loop, which makes debugging much easier.

### 3.2 Floating-point arithmetic is not associative

Even with identical random numbers, results can differ, because computer arithmetic rounds. In 32-bit
floating point, $(10^8+(-10^8))+1=1$ but $10^8+((-10^8)+1)=0$: the 1 is lost when it is first added to the
huge number. Summing a million numbers in two orders gives two slightly different totals. GPUs compute
sums and matrix products in parallel, splitting the work across thousands of threads; the order in which
partial sums are combined may depend on scheduling, and some operations (`index_add_`, `scatter_add_`,
some backward passes) use **atomic additions** whose order varies from run to run. Tiny differences are
then amplified by training: a different rounding in epoch 1 changes a gradient, which changes the weights,
which changes everything after.

### 3.3 Sources of variation, from the code down to the hardware

```
 code version (git commit)          <- record it
 configuration / hyper-parameters   <- save the config object with the results
 data version (download date, release, checksum)   <- section 4
 random seeds (all generators)      <- seed and record
 library versions (PyTorch, NumPy, scikit-learn, RDKit, BLAS)   <- lock file
 CUDA / cuDNN / driver versions     <- record
 algorithm selection (cuDNN benchmark, TF32 maths on recent GPUs) <- set flags
 parallel reduction order, atomics  <- deterministic algorithms
 hardware (CPU vs GPU, GPU model)   <- record; expect numerical, not bitwise, agreement
 external services (APIs that change daily)        <- snapshot responses, record query dates
```

The last line is easy to forget. `06_case_study.py` queries ClinicalTrials.gov live; the count of trials
for "amantadine" and "Alzheimer disease" (49 in our log) will change as new trials are registered. A
reader re-running the script next year will get a different table. Record the query date and save the raw
responses.

### 3.4 Controls in PyTorch

PyTorch's reproducibility notes describe the knobs. In summary:

* `torch.manual_seed(s)`: seeds the RNG of all devices.
* `torch.backends.cudnn.benchmark = False`: cuDNN otherwise times several algorithms on the first call and
  picks the fastest, and the choice can differ between runs.
* `torch.use_deterministic_algorithms(True)`: use deterministic implementations where they exist, and
  **raise an error** for operations known to be non-deterministic, so you find out rather than silently
  getting different numbers. (The older, narrower `torch.backends.cudnn.deterministic = True` covers only
  cuDNN convolutions.)
* Environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`): required for deterministic cuBLAS
  matrix products on CUDA 10.2 and later when deterministic algorithms are on; it must be set before the
  first CUDA call (best: in the shell, or at the very top of the script).
* DataLoader with workers: pass `worker_init_fn` that seeds NumPy/`random` from `torch.initial_seed()` and
  a seeded `torch.Generator` as `generator=`.
* Determinism may cost speed, and a few operations have no deterministic CUDA implementation at all.

**What to aim for in practice.** Make every run *re-runnable* with the same seeds; make it bitwise
reproducible on your own machine if it is cheap to do so; and, most importantly, run **several seeds and
splits and report the spread**. A reader on a different GPU will not get your bits; they should get your
conclusions.

### 3.5 Code: where non-determinism comes from, and seeding that works

```python
# Block 1: where non-determinism comes from, and how seeding controls it (CPU only).
import os
import random

import numpy as np
import torch

# (a) Floating-point addition is not associative: the ORDER of a sum changes the result.
#     Parallel GPU kernels (atomicAdd, split reductions) may add in a different order each run.
a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
print("(a + b) + c =", (a + b) + c, "   a + (b + c) =", a + (b + c))
v = np.random.default_rng(0).standard_normal(1_000_000).astype(np.float32)
print("same numbers, two summation orders differ by",
      float(v.sum() - v[np.random.default_rng(1).permutation(len(v))].sum()))


def seed_everything(seed: int, deterministic: bool = True):
    """Seed every RNG the project touches and ask PyTorch for deterministic kernels."""
    random.seed(seed)                      # Python's own RNG (e.g. random.shuffle)
    np.random.seed(seed)                   # legacy NumPy global RNG
    torch.manual_seed(seed)                # CPU and all CUDA devices
    if deterministic:
        # needed by cuBLAS for deterministic matrix multiplies on CUDA >= 10.2;
        # must be set before the first CUDA call to take effect
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False      # no auto-tuner picking kernels by timing
        torch.use_deterministic_algorithms(True)    # error out if an op has no deterministic version


def train_tiny(seed):
    seed_everything(seed)
    X = torch.randn(256, 8)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                              torch.nn.Dropout(0.2), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(50):
        idx = torch.randint(0, 256, (64,))                      # random mini-batch
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[idx]).squeeze(1), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), torch.cat([p.detach().flatten() for p in net.parameters()])


l1, w1 = train_tiny(0)
l2, w2 = train_tiny(0)
l3, w3 = train_tiny(1)
print(f"\nseed 0, run 1: final loss {l1:.10f}")
print(f"seed 0, run 2: final loss {l2:.10f}   bit-identical weights: {torch.equal(w1, w2)}")
print(f"seed 1       : final loss {l3:.10f}   bit-identical weights: {torch.equal(w1, w3)}")

# (c) A local generator isolates randomness: drawing extra numbers elsewhere
#     does not shift the stream used for, e.g., fold assignment.
g = torch.Generator().manual_seed(42)
print("\nlocal generator draw:", torch.randint(0, 100, (5,), generator=g).tolist())
```

Expected output (run on the project's environment: Python 3.13, PyTorch 2.14.1, NumPy 2.5.3, on CPU; the
second line may differ slightly with another NumPy version or CPU, the rest should not):

```
(a + b) + c = 1.0    a + (b + c) = 0.0
same numbers, two summation orders differ by -0.00018310546875

seed 0, run 1: final loss 0.1453370452
seed 0, run 2: final loss 0.1453370452   bit-identical weights: True
seed 1       : final loss 0.1905208230   bit-identical weights: False

local generator draw: [42, 67, 76, 14, 26]
```

Note the honest scope of the demonstration: two runs in the same process on the same CPU are
bit-identical; a different seed gives a different model (and a different final loss: 0.145 vs 0.191, a
reminder that one seed is one sample). Nothing here promises the same bits on a GPU.

---

## 4. Capturing the environment and the data

### 4.1 Software environment

Your result depends on every library in the import chain. The project's `requirements.txt` lists
packages **without versions** (`torch>=2.4`, `numpy`, `scipy`, ...). That is convenient for installation and
bad for reproducibility: in a year, `pip install -r requirements.txt` will fetch different versions. Keep
two files:

* `requirements.txt`: what the code needs, loosely pinned (for people who want to *use* it);
* a **lock file** with exact versions of everything installed (`pip freeze > requirements-lock.txt`, or
  `conda env export > environment.yml`, or a tool such as `uv`/`pip-tools`) for people who want to
  *reproduce* it.

Also record what pip does not see: Python version, operating system, CUDA toolkit version that PyTorch was
built with, cuDNN version, GPU model and driver, CPU model. The strongest form of environment capture is a
**container** image (Docker/Apptainer) or a capsule on a service such as Code Ocean, which freezes the whole
software stack; Heil et al. (2021) make this the "silver" level of their reproducibility standard.

### 4.2 Data provenance

For every external data source, record **what** you used, **where** it came from, **when**, **which
version**, **under which licence**, and **how it was processed**. This project uses many sources:

| Source | How obtained | Version information available | Licence / terms (summary) |
|---|---|---|---|
| Fdataset, Cdataset benchmarks (`.mat`) | GitHub mirror (DRHGCN repo), `01_download_data.py` | none in the file; record the commit of the mirror and the download date; cite Gottlieb et al. 2011 and Luo et al. 2016 | IDs are DrugBank/OMIM identifiers; follow the original papers' terms |
| CTD (chemicals, diseases, chem-gene, gene-disease, chem-disease) | `ctdbase.org/reports/`, `01_download_data.py` | each file header has a `Report created:` line (here: *Tue Sep 29 13:07:12 EDT 2026*) | free for research and education; must cite CTD; commercial use needs permission |
| MONDO (`mondo.obo`) | **not in the download script** (fetched by hand) | `data-version: releases/2026-09-01` in the header | CC BY 4.0 |
| Disease Ontology (`doid.obo`) | `01_download_data.py` | `data-version: releases/2026-09-30/doid.obo` | CC0 1.0 |
| MedGen ID mappings | **not in the download script** | file name only; record date | NCBI public resource; check NCBI's policies |
| PubChem (structures, names) | REST API, cached by `02_build_features.py` | live service; record access dates | free to use; individual depositors may have terms |
| DrugBank identifiers | inside the benchmarks; DrugBank XML not used | n/a | DrugBank full data CC BY-NC 4.0; "DrugBank Open Data" (vocabulary, structures) CC0 |
| ClinicalTrials.gov | API v2, live, in `06_case_study.py` | live; changes daily | public registry; record query date |

Two concrete gaps the table exposes, both easy to fix and both worth fixing before submission:

1. `mondo.obo` and `MedGenIDMappings.txt.gz` are read by `02_build_features.py`, but no script downloads
   them (`01_download_data.py`'s `FILES` dictionary does not contain them, although `HOW_IT_WORKS.md` says
   "everything is fetched by `scripts/01_download_data.py`"). Add their URLs, or document the manual step.
2. Nothing records *when* each file was downloaded. File modification times are not a record (copying a
   folder can change them). Write a small provenance file at download time (Block 2 below shows the idea).

The **FAIR principles** (Wilkinson et al. 2016: Findable, Accessible, Interoperable, Reusable) are the
general framework: give data persistent identifiers (a Zenodo DOI for your processed `.npz` files),
rich metadata, and a clear licence.

### 4.3 Code and results organisation

The project already follows most of the good practice recommended by Noble (2009, "A quick guide to
organizing computational biology projects") and Wilson et al. (2017, "Good enough practices in scientific
computing"):

| Practice | In this project |
|---|---|
| Separate raw, intermediate and processed data | `data/raw`, `data/interim`, `data/processed` |
| Library code separate from scripts | `src/drepo/` vs `scripts/` |
| Numbered pipeline scripts, one step each, re-runnable | `01_download_data.py` ... `08_make_figures.py` |
| One command reproduces everything | `scripts/run_all.ps1` (and `run_rest.ps1` with a progress file) |
| Configuration in one typed object, not scattered constants | `MVHGATConfig` dataclass; ablations are overrides of it (`04_ablation.py` `VARIANTS`) |
| Machine-readable results, per fold | `results/<dataset>/<protocol>/<method>.json` with all folds; `_preds.npz` with raw predictions |
| Tables and figures generated from results, never edited by hand | `08_make_figures.py` writes `RESULTS.md` and all PNGs |
| Logs of every run | `results/logs/*.log`, `*.err` |
| Raw data behind every plot kept | `_preds.npz`, `sensitivity.json`, case-study CSVs |

What is missing, in order of importance:

1. **Version control with commits.** The project folder sits inside a git repository rooted at the home
   directory that has *no commits*. Run `git init` in the project folder, add a `.gitignore` for `data/`
   and `.venv/`, and commit. Then record the commit hash in every results file.
2. **Save config, seed list and environment *with* each result.** `save()` in `runner.py` writes the summary,
   folds and label. Adding `dataclasses.asdict(cfg)`, the git hash and the environment summary (Block 2)
   would make every JSON self-describing (Sandve's rule 1: for every result, keep track of how it was
   produced).
3. **A lock file** (section 4.1).
4. **Consistent log encoding.** `results/logs/F_cv5.log` stores "±" as the single byte 0xB1 (Windows code
   page 1252, the console default), while `F_sens.log` (written via `Start-Process` in `run_rest.ps1`) is
   UTF-8. Opened as UTF-8, the first shows "�". Set
   `$env:PYTHONUTF8 = "1"` (or `PYTHONIOENCODING=utf-8`) at the top of the PowerShell scripts so every log
   is UTF-8.
5. **Snapshots of live queries** (ClinicalTrials.gov) with dates.

### 4.4 Code: a run manifest

The following script builds a JSON "manifest" with the configuration, environment, code version, data
checksums and source versions. It reads the project's files but writes only to `%TEMP%`.

```python
# Block 2: capture config + environment + data provenance in one JSON "run manifest".
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

PROJECT = Path(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning")


@dataclass
class RunConfig:                         # a cut-down stand-in for MVHGATConfig
    dataset: str = "Fdataset"
    protocol: str = "cv5"
    repeats: int = 5
    seed: int = 0
    hidden: int = 64
    lr: float = 2e-3
    epochs: int = 600


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def source_version(path):
    """Read the release/creation stamp a source file carries in its own header."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8", errors="replace") as fh:
        for _, line in zip(range(40), fh):
            if line.startswith(("data-version:", "# Report created:")):
                return line.strip().lstrip("# ")
    return None


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              cwd=PROJECT).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "config": asdict(RunConfig()),
    "command": " ".join(sys.argv),
    "environment": {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "packages": {p: pkg(p) for p in ["torch", "numpy", "scipy", "pandas",
                                          "scikit-learn", "rdkit", "matplotlib"]},
        "torch_built_with_cuda": torch.version.cuda,
        "cuda_visible": torch.cuda.device_count() > 0,
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    },
    "code": {"git_commit": run(["git", "rev-parse", "--verify", "--quiet", "HEAD"]),
             "git_dirty": bool(run(["git", "status", "--porcelain", "--", "."]))},
    "data": {p.name: {"sha256": sha256(p)[:16] + "...", "bytes": p.stat().st_size}
             for p in sorted((PROJECT / "data" / "processed").glob("*.npz"))},
    "sources": {p.name: source_version(p) for p in
                [PROJECT / "data/raw/ontology/mondo.obo", PROJECT / "data/raw/ontology/doid.obo",
                 PROJECT / "data/raw/ctd/CTD_diseases.tsv.gz"]},
}
out = Path(tempfile.gettempdir()) / "drepo_run_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print(json.dumps({k: manifest[k] for k in ("config", "environment", "code", "data", "sources")}, indent=1))
print("written to", out)
```

Expected output on the project machine. We ran it with the GPU hidden
(`CUDA_VISIBLE_DEVICES=-1`, so as not to disturb a running experiment), which is why `cuda_visible` is
`false`; on your normal shell it will be `true`. `nvidia-smi` still reports the GPU. The `data` checksums
are shown truncated to 16 characters; store the full 64 in practice.

```
{
 "config": {
  "dataset": "Fdataset",
  "protocol": "cv5",
  "repeats": 5,
  "seed": 0,
  "hidden": 64,
  "lr": 0.002,
  "epochs": 600
 },
 "environment": {
  "python": "3.13.5",
  "platform": "Windows-11-10.0.26200-SP0",
  "cpu": "Intel64 Family 6 Model 151 Stepping 2, GenuineIntel",
  "packages": {
   "torch": "2.14.1+cu126",
   "numpy": "2.5.3",
   "scipy": "1.18.1",
   "pandas": "3.0.6",
   "scikit-learn": "1.9.1",
   "rdkit": "2026.3.6",
   "matplotlib": "3.11.2"
  },
  "torch_built_with_cuda": "12.6",
  "cuda_visible": false,
  "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU, 581.86"
 },
 "code": {
  "git_commit": null,
  "git_dirty": true
 },
 "data": {
  "Cdataset.npz": {
   "sha256": "eaefb69c0c61b098...",
   "bytes": 4843583
  },
  "Fdataset.npz": {
   "sha256": "a0577bee23149ffa...",
   "bytes": 3836801
  }
 },
 "sources": {
  "mondo.obo": "data-version: releases/2026-09-01",
  "doid.obo": "data-version: releases/2026-09-30/doid.obo",
  "CTD_diseases.tsv.gz": "Report created: Tue Sep 29 13:07:12 EDT 2026"
 }
}
written to C:\Users\ABHINE~1\AppData\Local\Temp\drepo_run_manifest.json
```

Read the output critically, as a reviewer would:

* `"git_commit": null`: there is no commit to point to. This is the single most important gap.
* The processed data checksums let a reader verify they rebuilt *exactly* your `Fdataset.npz` from the raw
  files; if `02_build_features.py` is re-run after PubChem or CTD changes, the hash will change and you will
  know.
* The source stamps are the "version of data" the proposal and `PREREQUISITES.md` ask you to record ("the
  date you downloaded CTD"). They come from the files themselves, which is more trustworthy than memory.

---

## 5. The statistics of reporting

### 5.1 What varies, and what your ± should mean

A cross-validated AUPR of "0.488" is one draw from a distribution. It would have been different with:

* a different **split** of links into folds (the test set changes),
* a different **initialisation seed** (the model changes),
* different **negative samples** during training (MV-HGAT re-samples negatives every epoch),
* a different **hyper-parameter choice** (if tuning is noisy),
* a different **dataset** altogether (replicability).

The project's CV numbers, e.g. `0.4878 ± 0.0273`, are the mean and standard deviation **over all 25
fold-level values** (5 repeats × 5 folds; `summarise()` in `evaluation.py`). That ± therefore mixes
split-to-split variation with seed-to-seed variation, and it describes **how much a single fold's score
varies**, not how precisely we know the mean. Always say in the caption exactly what the ± is: "mean ±
standard deviation over 25 test folds (5 repetitions of 5-fold CV)".

### 5.2 Standard deviation, standard error and confidence intervals

For $n$ values $x_1,\dots,x_n$ with mean $\bar x$:

* **Standard deviation (SD)**: $s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}$ (the "sample" SD, divisor
  $n-1$, `np.std(x, ddof=1)`). NumPy's default `np.std(x)` divides by $n$ (`ddof=0`), which is what
  `summarise()` uses. For 25 values the difference is small (0.0273 vs 0.0278 for MV-HGAT's AUPR on
  Fdataset), but say which you used. SD describes the **spread of individual runs**.
* **Standard error of the mean (SEM)**: $s/\sqrt n$. It describes **how precisely the mean is estimated**
  *if the $n$ values are independent* (0.0278/5 = 0.0056 here).
* **95% confidence interval for the mean** (t-based): $\bar x\pm t_{0.975,\,n-1}\,s/\sqrt n$. With
  $n=25$, $t_{0.975,24}=2.064$, giving $0.488\pm0.011$, i.e. [0.476, 0.499].

Reporting SD makes your numbers look less precise than SEM does, and readers know it; never switch between
them to make a result look better, and never write "±" without saying which.

### 5.3 Why fold-level confidence intervals are too narrow

The t-interval assumes the 25 values are independent. They are not:

* The 5 folds within one repeat share 75% of their training data with each other (each training set is 4/5
  of the data), so their models are strongly correlated.
* Across repeats, the same links are re-used in different arrangements.

Bengio & Grandvalet (2004) proved that there is **no unbiased estimator of the variance of k-fold
cross-validation** based on the fold values alone, because of these correlations. The naive SEM
underestimates the true uncertainty, so naive confidence intervals and naive t-tests are **over-confident**.

Practical options, from simplest to most careful:

1. **Report SD over folds** (as the project does) and avoid claiming more precision than that.
2. **Use repeat-level means as the unit.** Average the 5 folds of each repeat; the 5 repeat means are much
   closer to independent (they are still computed on the same data). For MV-HGAT's AUPR on Fdataset the
   repeat means are 0.500, 0.486, 0.488, 0.490, 0.475: SD 0.0092.
3. **Use the corrected resampled t-test** (Nadeau & Bengio 2003) for comparisons: it inflates the variance
   to account for overlapping training sets (section 5.4).
4. **Bootstrap** the test predictions, resampling *diseases* (clusters) rather than individual pairs,
   because pairs that share a disease are correlated. This gives a CI for a single model's AUPR.

### 5.4 Comparing two methods: paired tests

Because every method in the project is evaluated on **exactly the same folds** (the splits depend only on
the seed), you can and should compare them **pairwise, fold by fold**. Pairing removes the variation that
comes from "this fold happened to be easy", which is shared by both methods. For fold $j$ let
$d_j=\text{score}_A(j)-\text{score}_B(j)$, $j=1,\dots,J$.

* **Win count / sign test:** how many of the $J$ folds does A win? Under "no difference", wins follow a
  Binomial($J$, 0.5).
* **Wilcoxon signed-rank test:** uses the ranks of $|d_j|$ and their signs; robust to outliers. With 25
  folds that are *all* positive, the smallest possible two-sided p-value is $2/2^{25}\approx6\times10^{-8}$.
* **Paired t-test:** $t=\bar d/(s_d/\sqrt J)$ with $J-1$ degrees of freedom; assumes independence, so it is
  over-confident for CV folds.
* **Corrected resampled t-test** (Nadeau & Bengio 2003; applied to repeated k-fold CV by Bouckaert & Frank
  2004):
  $$
  t=\frac{\bar d}{\sqrt{\left(\frac1J+\frac{n_{\text{test}}}{n_{\text{train}}}\right)s_d^2}},\qquad
  \frac{n_{\text{test}}}{n_{\text{train}}}=\frac{1}{k-1}\ \text{for }k\text{-fold CV},
  $$
  again with $J-1$ degrees of freedom. The extra term $n_{\text{test}}/n_{\text{train}}$ accounts for the
  correlation induced by overlapping training sets. For 5 × 5-fold CV it multiplies the variance by
  $(1/25+1/4)/(1/25)=7.25$, so $t$ shrinks by a factor of about 2.7.

**Multiple comparisons.** Comparing MV-HGAT against 5 baselines, on 2 metrics, 2 datasets and 2 protocols
gives 40 tests; at α = 0.05, two "significant" results would be expected by chance even if nothing
differed. Control the family-wise error with the **Holm-Bonferroni** procedure: sort the $m$ p-values
increasingly, $p_{(1)}\le\dots\le p_{(m)}$, and reject $H_{(i)}$ as long as $p_{(i)}\le\alpha/(m-i+1)$; stop
at the first failure. (For many methods over many datasets, Demšar (2006) recommends the Friedman test with
post-hoc tests, but with two datasets that machinery does not apply.)

**Effect size and practical significance.** A p-value says whether a difference is distinguishable from
noise, not whether it matters. Always give the difference itself, with its uncertainty: "+0.046 AUC (all
25 folds)" says more than "p < 0.001".

### 5.5 Worked example by hand: is MV-HGAT's AUPR different from SCMFDD's?

The repeat-level AUPR differences (MV-HGAT − SCMFDD, Fdataset, 5 × 5-fold CV; each is the mean over one
repeat's 5 folds) are

$$
d=(+0.0124,\;+0.0020,\;-0.0096,\;-0.0098,\;-0.0291).
$$

1. Mean: $\bar d=(0.0124+0.0020-0.0096-0.0098-0.0291)/5=-0.0341/5=-0.0068$.
2. Deviations from the mean: $(0.0192, 0.0088, -0.0028, -0.0030, -0.0223)$; squares sum to
   $0.000369+0.000077+0.000008+0.000009+0.000497\approx0.000960$; divide by $n-1=4$: $0.000240$; so
   $s_d\approx0.0155$.
3. Standard error: $0.0155/\sqrt5=0.00693$.
4. $t=-0.0068/0.00693=-0.99$ with 4 degrees of freedom: two-sided $p\approx0.38$.
5. 95% CI: $-0.0068\pm2.776\times0.00693=[-0.026,\ +0.012]$.

Conclusion: the AUPR difference is small and its CI comfortably includes 0. **We cannot claim either method
has the better AUPR on Fdataset.** The fold-level tests in Block 3 agree (Wilcoxon p = 0.19, corrected
t-test p = 0.70; MV-HGAT wins only 8 of 25 folds). For AUC the same procedure gives a difference of
+0.046 in every one of the 25 folds; that one is not in doubt.

### 5.6 Rounding

Report as many digits as the uncertainty supports, usually two significant digits of the SD and the mean to
the same decimal place: "0.939 ± 0.007", not "0.93924 ± 0.00721". Four decimals (as in `RESULTS.md`) are
fine in a supplementary table where people may recompute things.

### 5.7 Code: a results table with mean ± SD and CIs, and paired comparisons

```python
# Block 3: a results table (mean ± std, 95% CI) and honest paired comparisons,
# computed from the project's per-fold results (read-only).
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")                  # so "±" survives Windows consoles and log files
R = Path(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\results\Fdataset\cv5")
K = 5                                                     # folds per repeat
methods = ["mvhgat", "mbirw", "drrs", "scmfdd", "nimcgcn", "lagcn"]
folds = {m: pd.DataFrame(json.loads((R / f"{m}.json").read_text())["folds"]) for m in methods}
label = {m: json.loads((R / f"{m}.json").read_text())["label"] for m in methods}

rows = []
for m in methods:
    row = {"Method": label[m]}
    for metric in ("AUC", "AUPR"):
        x = folds[m][metric].to_numpy()
        n = len(x)
        sd = x.std(ddof=1)                                # sample std (n - 1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)   # naive 95% CI half-width
        row[metric] = f"{x.mean():.3f} ± {sd:.3f}"
        row[f"{metric} 95% CI (naive)"] = f"[{x.mean() - half:.3f}, {x.mean() + half:.3f}]"
    rows.append(row)
print(f"Fdataset, 5-fold CV x {len(folds['mvhgat']) // K} repeats "
      f"(n = {len(folds['mvhgat'])} folds); ± = sample std over folds\n")
print(pd.DataFrame(rows).to_markdown(index=False))


def paired(a, b, metric):
    """MV-HGAT (a) vs baseline (b) on the SAME folds."""
    d = folds[a][metric].to_numpy() - folds[b][metric].to_numpy()
    J = len(d)
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(J))
    # Nadeau & Bengio (2003) corrected resampled t: training sets overlap, so inflate variance
    t_corr = d.mean() / np.sqrt((1 / J + 1 / (K - 1)) * d.var(ddof=1))
    p = lambda t: 2 * stats.t.sf(abs(t), J - 1)
    # treat each repeat (average of its 5 folds) as one unit: 5 nearly independent numbers
    rep = pd.Series(d).groupby(folds[a]["repeat"].to_numpy()).mean().to_numpy()
    return {"comparison": f"{label[a]} - {label[b]}", "metric": metric,
            "mean diff": f"{d.mean():+.4f}", "wins": f"{(d > 0).sum()}/{J}",
            "p naive t": f"{p(t_naive):.2g}", "p corrected t": f"{p(t_corr):.2g}",
            "p Wilcoxon": f"{stats.wilcoxon(d).pvalue:.2g}",
            "repeat-level diffs": " ".join(f"{v:+.3f}" for v in rep)}


print()
print(pd.DataFrame([paired("mvhgat", b, m) for b in ("scmfdd", "mbirw")
                    for m in ("AUC", "AUPR")]).to_markdown(index=False, disable_numparse=True))
```

Expected output:

```
Fdataset, 5-fold CV x 5 repeats (n = 25 folds); ± = sample std over folds

| Method         | AUC           | AUC 95% CI (naive)   | AUPR          | AUPR 95% CI (naive)   |
|:---------------|:--------------|:---------------------|:--------------|:----------------------|
| MV-HGAT (ours) | 0.939 ± 0.007 | [0.936, 0.942]       | 0.488 ± 0.028 | [0.476, 0.499]        |
| MBiRW          | 0.883 ± 0.010 | [0.879, 0.887]       | 0.311 ± 0.019 | [0.303, 0.318]        |
| DRRS           | 0.879 ± 0.013 | [0.874, 0.884]       | 0.389 ± 0.018 | [0.382, 0.397]        |
| SCMFDD         | 0.893 ± 0.010 | [0.889, 0.898]       | 0.495 ± 0.020 | [0.486, 0.503]        |
| NIMCGCN        | 0.838 ± 0.009 | [0.835, 0.842]       | 0.096 ± 0.014 | [0.090, 0.101]        |
| LAGCN          | 0.833 ± 0.015 | [0.827, 0.840]       | 0.133 ± 0.018 | [0.125, 0.140]        |

| comparison              | metric   | mean diff   | wins   | p naive t   | p corrected t   | p Wilcoxon   | repeat-level diffs                 |
|:------------------------|:---------|:------------|:-------|:------------|:----------------|:-------------|:-----------------------------------|
| MV-HGAT (ours) - SCMFDD | AUC      | +0.0458     | 25/25  | 2e-19       | 5.1e-10         | 6e-08        | +0.049 +0.048 +0.047 +0.044 +0.041 |
| MV-HGAT (ours) - SCMFDD | AUPR     | -0.0068     | 8/25   | 0.31        | 0.7             | 0.19         | +0.012 +0.002 -0.010 -0.010 -0.029 |
| MV-HGAT (ours) - MBiRW  | AUC      | +0.0561     | 25/25  | 9.3e-22     | 4.9e-12         | 6e-08        | +0.057 +0.057 +0.055 +0.059 +0.053 |
| MV-HGAT (ours) - MBiRW  | AUPR     | +0.1773     | 25/25  | 3.8e-20     | 1.2e-10         | 6e-08        | +0.187 +0.173 +0.182 +0.179 +0.166 |
```

Three lessons are visible in this one table:

1. The naive and corrected p-values differ by many orders of magnitude for large effects and change the
   conclusion's strength for small ones (0.31 vs 0.70). Use the corrected one (or report both and say
   which you rely on).
2. AUC and AUPR can tell different stories: MV-HGAT is clearly better than SCMFDD in AUC and
   indistinguishable in AUPR.
3. The "naive 95% CI" column is narrower than the true uncertainty (section 5.3). It is printed here to
   show you what *not* to rely on for between-method claims; in a paper, either drop it or label it
   precisely.

Running the same comparison for all four warm-start settings (we did, with the same code) gives:

| Setting | ΔAUC vs SCMFDD (wins) | ΔAUPR vs SCMFDD (wins, corrected p) |
|---|---|---|
| Fdataset, 5 × 5-fold | +0.046 (25/25) | −0.007 (8/25, p = 0.70) |
| Fdataset, 10-fold | +0.043 (10/10) | −0.022 (2/10, p = 0.21) |
| Cdataset, 5 × 5-fold | +0.048 (25/25) | −0.002 (11/25, p = 0.87) |
| Cdataset, 10-fold | +0.045 (10/10) | −0.017 (3/10, p = 0.22) |

That table *is* the honest headline of the project's warm-start comparison (section 7.6).

---

## 6. Tables and figures that follow good practice

### 6.1 Tables

A good results table is **self-contained**: a reader who sees only the table and its caption knows what was
measured, on what, how often, and what the ± means. Checklist:

* Caption states dataset, protocol (5-fold CV, repeats), metric definitions, what ± is (SD over 25 folds),
  and where the numbers come from (re-implemented baselines on identical splits).
* Same number of decimals in a column; uncertainty on every number that has one.
* Order rows meaningfully (your method first or last, baselines by year or by performance) and keep the
  order the same in every table.
* Mark the best value **only if** it is distinguishable from the runner-up; otherwise mark all values
  within the uncertainty (e.g. bold both MV-HGAT and SCMFDD for AUPR) or use a footnote.
* Never mix numbers from different protocols or sources in one table without saying so (section 7.1).

### 6.2 Figures: rules that prevent most mistakes

Rougier, Droettboom & Bourne (2014), "Ten simple rules for better figures", give the core list: know your
audience; identify your message; adapt the figure to the medium (paper vs slide); captions are not
optional; do not trust the defaults; use colour effectively; do not mislead the reader; avoid chartjunk;
message trumps beauty; get the right tool. Concretely, for this project:

1. **One message per figure, one scale per axis.** Never put two different quantities on two y-axes of the
   same panel (a "dual-axis" chart): the visual relation between the two curves is an artefact of how you
   chose the two scales. Use two panels (small multiples) or index both to a common base.
2. **Show the data, not only a summary.** Bar charts of means hide the distribution; Weissgerber et al.
   (2015, "Beyond bar and line graphs") showed how often very different data sets produce the same bar
   chart. With 25 folds per method, plot all 25 points plus the mean.
3. **Honest axes.** Bars encode values by *length*, so a bar chart must start at zero. For dot plots and
   lines, a non-zero axis is acceptable *if you say so*. `08_make_figures.py` does exactly this for the
   ablation figure: its title reads "Ablation - Fdataset (x-axis does not start at 0)". Good practice.
4. **Colour-blind-safe, fixed colours.** About 1 in 12 men has a colour-vision deficiency. Use a validated
   palette (e.g. Okabe & Ito's, or the one in `08_make_figures.py`, which we checked with a CVD validator:
   all adjacent pairs remain distinguishable under simulated colour blindness), assign each method **one
   colour in every figure**, never encode meaning by red vs green alone, and avoid rainbow colour maps for
   ordered data (Crameri, Shephard & Heron 2020).
5. **Direct labels** instead of a legend when possible: the reader should not have to match colours. When
   colours have low contrast against the background (light yellow and pink on white), the text label is what
   carries identity.
6. **Self-contained captions:** what is plotted, on what data, what error bars mean, how many runs.
7. **Readable at final size:** fonts ≥ 7-8 pt in the printed figure; vector formats (PDF/SVG) for line art;
   at least 300 dpi for raster images.
8. **Store the raw data behind every plot** (Sandve rule 7). The project keeps `_preds.npz` and the JSONs.

### 6.3 Critique of the project's ROC/PR figure

Open `results/figures/curves_Fdataset_cv5.png`. What it does well: a fixed, validated colour per method;
the proposed method drawn thicker and on top; the metric value in each legend entry; a dotted chance
diagonal on the ROC panel; a bold title naming dataset and protocol.

What a reviewer might ask you to change:

* **The PR panel has no chance line.** For PR curves the "random" level is the positive rate (about 1% here),
  a horizontal line near the bottom. Without it, readers may not realise how far above random 0.1 is.
* **Curves and legend numbers come from different data.** In `08_make_figures.py` the curves are computed
  from pooled predictions of the **first repeat** only (`run_kfold` keeps `r == 0` for curves), while the
  legend shows the **mean over all 25 folds** (`summary["AUC"]`). Both are fine, but the caption must say
  so, or a careful reader will find that the area under the drawn curve does not exactly match the number
  next to it.
* **The informative region of the ROC is squeezed.** With 99% negatives, everything interesting happens at
  false-positive rates below 0.1. A log-scaled x-axis or an inset makes the differences visible.
* **Legend instead of direct labels.** With six crossing curves, direct labelling is hard, so a legend is a
  reasonable compromise; keep the legend order identical to the table order (it is).

### 6.4 Code: a minimal good-practice figure

This dot plot shows every fold's AUPR for every method, the mean as a large dot, a zero-based axis, the
random baseline, direct labels and the project's validated colours. It writes a PNG to `%TEMP%`.

```python
# Block 4: a minimal good-practice figure (one axis, zero-based, all data shown,
# direct labels, colour-blind-validated colours), saved to %TEMP%.
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(r"C:\Users\Abhineet Anand\Desktop\DrugRepositioning\results\Fdataset\cv5")
order = ["mvhgat", "scmfdd", "drrs", "mbirw", "lagcn", "nimcgcn"]          # ours, then by mean AUPR
colors = {"mvhgat": "#2a78d6", "mbirw": "#eb6834", "drrs": "#1baf7a",       # same colour per method
          "scmfdd": "#eda100", "nimcgcn": "#e87ba4", "lagcn": "#008300"}    # as 08_make_figures.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

fig, ax = plt.subplots(figsize=(6.4, 3.4))
rng = np.random.default_rng(0)                     # jitter only; fixed so the figure is reproducible
for y, m in enumerate(order[::-1]):
    j = json.loads((R / f"{m}.json").read_text())
    x = np.array([f["AUPR"] for f in j["folds"]])
    ax.scatter(x, y + rng.uniform(-0.15, 0.15, len(x)), s=10, color=colors[m], alpha=0.45, lw=0)
    ax.plot([x.mean()], [y], marker="o", ms=8, color=colors[m], mec="white", mew=1.5, zorder=3)
    ax.text(x.max() + 0.015, y, f"{x.mean():.3f}", va="center", fontsize=9, color=INK2)
    ax.text(-0.01, y, j["label"], va="center", ha="right", fontsize=9, color=INK)   # direct label
ax.axvline(0.0104, color=INK2, lw=1, ls=":")
ax.text(0.0104 + 0.008, len(order) - 0.45, "random (positive rate 1.04%)", fontsize=8, color=INK2)
ax.set_xlim(0, 0.65)                                # zero-based axis: no exaggerated gaps
ax.set_ylim(-0.6, len(order) - 0.2)
ax.set_yticks([])
ax.set_xlabel("AUPR (each dot = one test fold; large dot = mean of 25 folds)", color=INK2)
ax.set_title("Fdataset, 5-fold CV x 5 repeats", loc="left", color=INK, fontsize=11)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
out = Path(tempfile.gettempdir()) / "aupr_dotplot_Fdataset_cv5.png"
fig.savefig(out, dpi=200)
print("saved", out)
```

Expected output (your user name may appear in full instead of the short form `ABHINE~1`):

```
saved C:\Users\ABHINE~1\AppData\Local\Temp\aupr_dotplot_Fdataset_cv5.png
```

What you will see: six rows, one per method, labelled directly on the left; 25 faint dots per row and a
large dot for the mean, with its value printed at the right; a dotted vertical line at 0.0104 labelled as the
random level. The MV-HGAT and SCMFDD clouds overlap almost completely, which is the visual version of
"no detectable AUPR difference" and is far more convincing than any p-value. NIMCGCN and LAGCN sit well
below the others but far above random. If you need a vector version for the journal, change the suffix to
`.pdf`.

---

## 7. Honest comparisons

### 7.1 Re-implemented versus published baselines

There are two ways to get a baseline's number, and they answer different questions:

| | Re-implemented baseline (this project) | Published number (copied from a paper) |
|---|---|---|
| Splits | identical folds for every method | the original authors' splits, usually unknown |
| Negatives, preprocessing | identical | may differ (e.g. different handling of unknown pairs) |
| Information | controlled (all baselines use the 2 benchmark similarities) | whatever the paper used |
| Tuning | your effort; must be documented | the authors' effort (often extensive) |
| Risk | your re-implementation may be weaker than the original (bugs, missing tricks) | not comparable with your protocol |

**Rules.** (1) Never put published numbers and your own numbers in one table as if they were comparable.
(2) If you quote published numbers, put them in a separate table or column, labelled "as reported in
[ref]; different splits". (3) Say clearly that baselines are re-implementations, what was simplified (the
project's `MBiRW` docstring says it omits the clustering-based similarity adjustment of the original), and
that your numbers can differ from the published ones. `HOW_IT_WORKS.md` section 5.4 already says exactly
this; carry it into the paper. (4) If a public implementation by the original authors exists, the
strongest option is to run *their* code on *your* splits as a check of your re-implementation.

### 7.2 Tuning parity

A new method is usually tuned for weeks; a baseline is often run with defaults. That makes any comparison
unfair in your favour, and reviewers know it. Melis, Dyer & Blunsom (2018) showed that carefully tuned
"old" LSTM language models outperformed several more recent, more complex architectures. Dodge et al.
(2019, "Show your work") recommend reporting **expected validation performance as a function of the tuning
budget**, so readers see whether your advantage survives equal effort. Minimum standard for the paper: for
every method, the hyper-parameters searched, the search range, the number of configurations tried, the
selection criterion and the data used for selection (the project's validation split, never the test folds).

### 7.3 Information parity

MV-HGAT uses six similarity views; every baseline uses two. Any advantage could come from **more data**
rather than **better modelling**. The project's ablation "benchmark similarities only" answers this
(Fdataset, 5 folds, one repeat): AUC 0.9431 ± 0.0068 vs 0.9434 ± 0.0072 for the full model, AUPR 0.481 ±
0.018 vs 0.500 ± 0.029. So the AUC advantage over the baselines (≈ 0.89 for the best one) is almost
entirely architectural, while the extra views add a little AUPR. That is precisely the kind of sentence that
makes a reviewer trust a comparison.

### 7.4 Report everything you ran, including the inconvenient

* **Where you lose:** AUPR vs SCMFDD (section 5.7); in cross-dataset validation (train on all of Fdataset,
  recover the 57 links that only Cdataset contains among 174,330 candidate pairs; random AUPR 0.0003),
  MV-HGAT has the best AUC (0.944) but MBiRW has the best AUPR (0.0136 vs 0.0086).
* **Where a component did not help:** in the ablation, "w/o hidden-link supervision" scored *higher* AUPR
  than the full model in all 5 folds (0.546 vs 0.500), although `HOW_IT_WORKS.md` section 10 reports that
  hidden-link supervision was crucial during development (validation AUC 0.71 to 0.92, with an earlier
  version of the model). Both facts can be true: the propagation head, added later, may have taken over the
  role hidden-link supervision played. Report the ablation as it is, and discuss it. Quietly removing the
  row would be a serious breach of honesty.
* **What did not work at all:** the gene bridge (`+ drug-gene-disease bridge`: AUPR 0.476, no gain; its
  standalone AUPR is near random). Negative results save other people time.
* **Experiments that are not finished:** the leave-one-disease-out (cold-start) runs. Do not describe them
  as done or extrapolate from partial logs.

Selective reporting has names: *cherry-picking* (choosing favourable datasets, metrics or seeds) and
*HARKing* (Hypothesising After the Results are Known: presenting a post-hoc explanation as if it had been
the plan). Lipton & Steinhardt (2018, "Troubling trends in machine learning scholarship") list these
together with "mathiness" and failing to identify the sources of empirical gains (which ablations exist to
do).

### 7.5 Same splits, same negatives, same evaluation code

The project's design gets the essentials right, and the paper should say so explicitly because it is a
strength: every method implements `fit_predict(data, A_train, neg_mask, seed)` and is evaluated by the same
`run_kfold` / `run_lodo` code on identical splits, with negatives drawn only from training-fold unknowns.
Unit E1 explains why each of these matters for leakage.

### 7.6 Worked example: rewriting a dishonest results paragraph

**Weak (and wrong):**

> MV-HGAT significantly outperforms all state-of-the-art methods, achieving an AUC of 0.9392 and an AUPR
> of 0.4878 on Fdataset, demonstrating the superiority of heterogeneous graph attention for drug
> repositioning.

Problems: "significantly" with no test; "all" is false for AUPR; four decimals without uncertainty;
"state-of-the-art" for re-implemented baselines on one benchmark; "superiority of attention" is not what
the ablation shows (the propagation head matters more than the attention).

**Strong:**

> On Fdataset (5 repetitions of 5-fold CV, identical folds for all methods), MV-HGAT achieved an AUC of
> 0.939 ± 0.007 (mean ± SD over 25 folds), higher than all five re-implemented baselines in every fold
> (best baseline SCMFDD: 0.893 ± 0.010; corrected resampled t-test p < 10⁻⁹). Its AUPR (0.488 ± 0.027)
> was higher than that of MBiRW, DRRS, NIMCGCN and LAGCN but statistically indistinguishable from
> SCMFDD's (0.495 ± 0.020; p = 0.70), a pattern repeated on Cdataset and with 10-fold CV (Table 2). Thus
> MV-HGAT ranks known indications above unknown pairs more consistently over the whole ranking, while
> similarity-constrained matrix factorisation is more precise among its highest-ranked predictions (recall
> below about 40%; Figure 2b). The advantage in AUC persisted when MV-HGAT was restricted to the two benchmark
> similarities used by the baselines (AUC 0.943; Section 5.4), indicating that it stems from the model
> rather than from the additional similarity views.

It is longer, more precise, and much harder for a reviewer to attack.

---

## 8. The anatomy of a bioinformatics methods paper

### 8.1 The shape: IMRaD and the hourglass

Almost every methods paper in bioinformatics follows **IMRaD**: Introduction, Methods, Results and
Discussion, framed by a title and abstract at the front and availability statements, references and
supplementary material at the back. Its logic is an hourglass:

```
  Introduction     broad: why drug repositioning matters
                   narrower: computational approaches, their gap
                   narrowest: our question and contribution
  Methods          specific: exactly what we did
  Results          specific: exactly what we found
  Discussion       narrower -> broad: what it means, limits, what next
```

Mensh & Kording (2017, "Ten simple rules for structuring papers") give the principle behind each part:
focus the paper on **one central contribution**, communicated in the title; write for intelligent readers
who do not know your work; use **context-content-conclusion** at every scale (the paper, each section, each
paragraph: first say what the paragraph is about, then the content, then what it means); avoid zig-zag
(keep each topic in one place) and use parallel structure; tell a complete story in the abstract; say in
the introduction why the paper matters; deliver results as a logical sequence of statements, each supported
by a figure; discuss how the gap was filled, the limitations and the relevance; and spend your time where
it matters most (title, abstract, figures, outline). Whitesides (2004) adds a practical method: write the
paper as an **outline of figures and tables first**, then the prose around them, and treat the outline as a
tool for planning the research itself, not only for reporting it.

### 8.2 Section by section, with what reviewers look for

| Section | Its job | Contents | Reviewers check |
|---|---|---|---|
| **Title** | state the central contribution | method + task (+ key property) | specific? not overclaiming? |
| **Abstract** | the whole story in miniature | Bioinformatics uses a structured abstract (Motivation; Results; Availability and implementation; plus contact and supplementary-information lines). Results must contain numbers | do the numbers match the paper? is the claim proportionate? is the code link there? |
| **Introduction** | why it matters, what is missing, what you add | the problem (repositioning cost/time), prior approaches (network propagation, matrix factorisation, GNNs), the specific gap (single similarity views, opaque attention, weak cold-start evaluation), a bulleted list of contributions | is the gap real? is related work fair and recent? are contributions verifiable in the Results? |
| **Materials and Methods** | let someone re-do it | data (sources, versions, sizes, mapping), similarity views, graph construction, model, training, evaluation protocols, baselines, statistics, implementation | leakage? fair baselines? hyper-parameter selection? enough detail to reproduce? |
| **Results** | the evidence, in a logical order | 5.1-5.7 (section 10), each subsection: claim sentence, figure/table, the numbers, a one-sentence interpretation | are claims supported? uncertainty? statistical tests? ablations isolate the contribution? |
| **Discussion** | meaning, limits, next steps | summary of findings, why the method works (with ablation evidence), comparison with literature, limitations, future work | are limitations acknowledged honestly? no new results here? no overreach? |
| **Conclusion** | 1 paragraph | the take-home message | consistent with results? |
| **Availability** | let others use it | code URL + archive DOI + licence; data statement | do links work? is code actually runnable? |
| **Supplementary** | everything else | full tables (4 decimals), all case studies, hyper-parameter grids, extra figures | complete? consistent with the main text? |

Length: *Bioinformatics* Original Papers are up to 7 pages (about 5,000 words excluding figures);
Application Notes up to 4 pages (about 2,600 words, or 2,000 words plus one figure). *Briefings in
Bioinformatics* publishes longer problem-solving papers and reviews. Check the current guidelines before
submitting; they change.

### 8.3 Writing well: sentence-level habits

* One idea per paragraph; the first sentence states it.
* Prefer the active voice and concrete subjects: "We hid 20% of the training links each epoch" rather than
  "A proportion of links was hidden".
* Define every abbreviation once; use the same word for the same thing throughout (do not alternate
  "view", "modality", "similarity source" and "relation" for one concept; pick one and define it).
* Every number has units or a definition and, if it is an estimate, an uncertainty.
* Every claim in the text points to a figure, a table or a reference (Sandve rule 9: connect textual
  statements to underlying results).
* Reserve "significant" for statistical significance.

---

## 9. Writing a Methods section someone else can re-run

### 9.1 The test

Give your Methods (plus the code) to a competent student who has never seen the project. Can they
reproduce Table 2 without asking you a single question? Every question they would ask is a missing
sentence. The ML Reproducibility Checklist and DOME (section 12) are, in effect, lists of those questions.

### 9.2 What must be there, for this project

**Data.** Benchmark names, origin and citations; sizes (593 drugs × 313 diseases, 1,933 links for
Fdataset; 663 × 409, 2,532 links for Cdataset; and the correction of the proposal's numbers); orientation
issue in the files (the CSV matrix is diseases × drugs); how IDs were obtained (`.mat` files from the
DRHGCN repository, verified identical to the CSVs); every external source with version or access date
(section 4.2); identifier mapping chains and their coverage (`HOW_IT_WORKS.md` section 2.3).

**Similarity views.** For each of the six: formula (Tanimoto on 2048-bit Morgan radius-2 fingerprints;
Jaccard on CTD gene sets; Wang semantic similarity on MONDO with the decay factor), how missing values
are handled (no neighbours; never imputed), and the kNN sparsification ($k=10$, symmetrised).

**Model.** Architecture with equations (GAT within relations, view-level attention, 2 layers, jumping
knowledge, bilinear decoder, propagation head, degree gate), dimensions (hidden 64, 4 heads), and the
parameterisation of $w_v$.

**Training.** Loss (binary cross-entropy), negative sampling (2 per positive per epoch, re-sampled, only
from training-fold unknowns), hidden-link supervision (20% hidden per epoch), cold-start practice (10% of
diseases hidden per epoch), optimiser (Adam, learning rate 0.002, weight decay 5e-4), dropout 0.2, 600
epochs, no early stopping, seeds.

**Hyper-parameter selection.** The validation split (20% of links and of unknown pairs, fixed seed 12345,
`05_sensitivity.py`), the grid, the number of seeds, the selection criterion, and the honest caveat that
the validation data overlap the later CV data (not nested CV).

**Evaluation.** Protocols (5-fold CV with 5 repetitions, 10-fold CV, LODO on a seeded subset of 100
diseases or all diseases), how negatives enter the test set (all unknown pairs of the fold), metrics (AUC;
AUPR as average precision), what is averaged and what ± means, statistical tests.

**Baselines.** For each: the original reference, what was re-implemented, what was simplified, its
hyper-parameters and how they were chosen, which inputs it receives.

**Implementation and compute.** Language and library versions (with a lock file), hardware (RTX 3050 6 GB
laptop GPU), run time (about 4 hours for `run_all.ps1`; the NeurIPS checklist asks for compute per run and
in total), code repository and archive DOI.

### 9.3 Worked example 1: the evaluation paragraph

**Weak:**

> We evaluated our model using 5-fold cross-validation and compared it with several baselines. AUC and
> AUPR were used as metrics.

**Strong:**

> *Warm-start evaluation.* We used k-fold cross-validation with k = 5 (five repetitions with different
> random partitions) and k = 10 (one repetition). In each repetition, the known drug-disease associations
> and the unknown pairs were each shuffled and split into k folds with a seeded random generator, so that
> every method was evaluated on identical folds. For fold f, the associations in f were removed from the
> training matrix; the model was trained on the remaining associations, drawing negative examples only from
> unknown pairs outside fold f; and it scored the test set consisting of fold f's associations (positives)
> and fold f's unknown pairs (negatives). We report the area under the ROC curve (AUC) and the average
> precision (AUPR; random level equal to the positive rate, 1.04% for Fdataset and 0.93% for Cdataset) as
> mean ± standard deviation over all folds of all repetitions. Methods were compared fold by fold with the
> corrected resampled t-test (Nadeau & Bengio, 2003) and the Wilcoxon signed-rank test, with Holm's
> correction for multiple comparisons.

Every sentence answers a question a re-implementer would otherwise have to ask.

### 9.4 Worked example 2: the data paragraph

**Weak:**

> Drug and disease data were downloaded from DrugBank, CTD and MeSH.

(This is also *untrue* for this project: DrugBank XML was not used, and MONDO replaced MeSH for semantic
similarity. A Methods section must describe what was done, not what was planned in the proposal.)

**Strong:**

> *Data sources.* We used two benchmark datasets, Fdataset (Gottlieb et al., 2011; 593 drugs, 313
> diseases, 1,933 associations) and Cdataset (Luo et al., 2016; 663 drugs, 409 diseases, 2,532
> associations), obtained as MATLAB files that include DrugBank and OMIM identifiers (downloaded from the
> DRHGCN repository on 1 October 2026; matrices verified identical to the original CSV files). Drug
> structures were retrieved from PubChem through the DrugBank-deposited substance records (97.5% and 98.9%
> of drugs mapped). Drug-gene interactions and curated disease-gene associations were taken from the
> Comparative Toxicogenomics Database (CTD; files generated 29 September 2026). Disease semantic
> similarity used the MONDO ontology (release 2026-09-01). CTD curated chemical-disease associations were
> used only to check case-study predictions, never as model input. Identifier mappings, coverage and all
> download scripts are provided in the repository (`scripts/01_download_data.py`,
> `scripts/02_build_features.py`; mapping tables in `data/interim/`).

(Use the real dates from your provenance record; the ones above are those of the files in this project.)

### 9.5 Worked example 3: the training paragraph

**Weak:**

> The model was trained with Adam for 600 epochs with standard hyper-parameters.

**Strong:**

> *Training.* Model parameters were optimised with Adam (learning rate 2 × 10⁻³, weight decay 5 × 10⁻⁴)
> for 600 full-batch epochs, without early stopping, minimising binary cross-entropy. In each epoch we
> (i) hid a random 20% of the training associations from the graph, the input features and the
> propagation terms, and additionally all associations of a random 10% of diseases, and (ii) computed the
> loss on the hidden associations as positives and on twice as many unknown training pairs, sampled
> uniformly, as negatives. Dropout (0.2) was applied to node embeddings and attention coefficients.
> Hyper-parameters were chosen on a validation split (20% of associations and unknown pairs, held out from
> training) by varying one hyper-parameter at a time around the defaults (Supplementary Table S3). Each
> model fit used a seed determined by the repetition and fold; all random number generators (Python,
> NumPy, PyTorch CPU and CUDA) were seeded.

---

## 10. Writing the Results, Discussion and Limitations

### 10.1 Results: a sequence of supported statements

Each Results subsection should have the same skeleton:

1. **Topic sentence that states the finding** (not "Table 2 shows the results" but "MV-HGAT achieved the
   highest AUC in all warm-start settings").
2. **Pointer** to the figure or table.
3. **The numbers**, with uncertainty and comparison.
4. **One sentence of interpretation**, kept modest; longer interpretation belongs in the Discussion.

Keep the order of subsections logical: main comparison → why (ablation) → robustness (sensitivity,
cross-dataset) → usefulness (case studies, interpretability). The project's proposal planned exactly this
order (5.1-5.7), and section 14 gives the full mapping with suggested sentences.

### 10.2 Discussion

A good Discussion does five things, in this order:

1. **Restate the main findings** in two or three sentences, in plain language.
2. **Explain them**, using your own evidence: e.g. the ablation shows the multi-view propagation head
   contributes most (removing it costs 0.067 AUPR on average, lower in 4 of 5 folds), consistent with
   neighbourhood propagation being strong on these benchmarks (MBiRW is itself a propagation method).
3. **Relate them to the literature**: where your results agree or disagree with published methods, and
   why (different splits, re-implementation).
4. **Limitations** (section 10.3), stated plainly.
5. **Implications and future work**: what should be done next, by you or others.

Do not introduce new results in the Discussion, and do not repeat the Results section number by number.

### 10.3 Limitations: write them before the reviewers do

Reviewers trust authors who state their limitations clearly; they distrust authors who hide them. For this
project, `HOW_IT_WORKS.md` section 6 is essentially a ready-made list. Group them:

* **Data:** unknown pairs treated as negatives (all scores pessimistic, the field's convention); benchmarks
  frozen at 2011-2016 knowledge; literature bias in CTD; half of the diseases lack curated genes; identifier
  mapping errors possible.
* **Evaluation:** hyper-parameters tuned on a validation split that overlaps the later CV data (not nested
  CV); LODO on a subset of 100 diseases (if that is what you report); baselines re-implemented and possibly
  weaker than originals; fold-level variances under-estimate uncertainty.
* **Model:** transductive (new drugs require re-training); kNN with k = 10 discards weaker relations; the
  degree gate and propagation head favour well-connected neighbourhoods.
* **Interpretation:** attention weights are not faithful explanations; occlusion and head contributions
  are faithful to the model only; nothing in the explanations is evidence of mechanism (unit E2).
* **Case studies:** support is checked against databases and a trial registry, not experiments; trial
  counts are raw registry hits.

A useful formula for each limitation: *what it is* → *what effect it could have on the conclusions* →
*what you did or could do about it*. Example: "Because unknown pairs are treated as negatives, some test
negatives are undiscovered indications; this biases all methods' AUPR downwards and may penalise methods
that rank such pairs highly. Cross-dataset validation, in which links absent from Fdataset but present in
Cdataset were recovered with AUC 0.94, suggests that some 'false positives' are indeed true associations."

---

## 11. Reporting case studies, and the wet-lab and clinical caveats

Case studies are the most-read and most-criticised part of a drug-repositioning paper. Reviewers have seen
many lists of "novel predictions confirmed by the literature" and know the tricks. Report them like this:

1. **Pre-specify the diseases and the rule.** Say why these diseases were chosen (here Alzheimer's
   disease, breast cancer and prostate cancer, the script's defaults; a defensible justification is that they
   are common, have several known drugs in the benchmark, and are standard case studies in prior
   repositioning papers, which makes comparison possible; give your actual reasons) and that the top 10
   *new* candidates are shown, all of them, in rank order. Showing only the confirmed hits is
   cherry-picking.
2. **Define "support" before looking.** The project's rule (`06_case_study.py`): a CTD curated
   chemical-disease link of any evidence type, a link in the other benchmark, or at least one
   ClinicalTrials.gov record. State it, and state its weaknesses: CTD "marker/mechanism" can mean the drug
   *causes* the phenotype; trial records include failed, withdrawn and comparator-arm trials; the
   registry is searched by name, so synonyms and generic queries add noise.
3. **Report the denominator and a reference rate.** "10/10 for prostate cancer, 6/10 for breast cancer, 5/10
   for Alzheimer's disease". Then ask: what fraction of *random* drugs, or of a simple baseline's top 10,
   would meet the same rule? Without a reference rate, "6/10" has no scale. (Cancers have many trials of many
   drugs, so a high rate is easier to achieve there; that is part of why prostate cancer scores 10/10.)
4. **Show the evidence that drove each prediction** (the `top_evidence` column) and, for the most
   interesting ones, a neighbour-level explanation (unit E2, section 11.4: amantadine's score comes from its
   structural analogue memantine).
5. **Discuss failures and implausible predictions too.** The Alzheimer's list is dominated by
   Parkinson's-disease drugs; say so and explain the likely reason (shared neighbourhood of
   neurodegenerative diseases), rather than inventing a mechanism for each.
6. **Use careful verbs.** "is supported by", "has been investigated in", "is a candidate for"; never
   "treats", "is effective", "was validated".
7. **State the next steps and the gap to the clinic.** In vitro assays, animal models, observational
   health-record analyses, and only then trials; and practical barriers the model ignores (dose,
   blood-brain-barrier penetration, safety in the target population, patent and commercial status).
8. **Include the mandatory sentence**: predictions are hypotheses for further research, not treatment
   recommendations.
9. **Make it reproducible:** record the date of the ClinicalTrials.gov queries and save the raw responses;
   record the CTD release; average over seeds (the project uses 5) and say so.

---

## 12. Reproducibility checklists, journal policies and availability statements

### 12.1 The checklists you should know

**Sandve, Nekrutenko, Taylor & Hovig (2013), "Ten simple rules for reproducible computational research":**
(1) for every result, keep track of how it was produced; (2) avoid manual data manipulation steps;
(3) archive the exact versions of all external programs used; (4) version control all custom scripts;
(5) record all intermediate results, when possible in standardised formats; (6) for analyses that include
randomness, note underlying random seeds; (7) always store raw data behind plots; (8) generate hierarchical
analysis output, allowing layers of increasing detail to be inspected; (9) connect textual statements to
underlying results; (10) provide public access to scripts, runs and results.

**The Machine Learning Reproducibility Checklist** (Pineau et al.; used at NeurIPS 2019 and described in
"Improving Reproducibility in Machine Learning Research", JMLR 2021). For every model and algorithm: a clear
description, complexity analysis, and code link. For theoretical claims: assumptions and proofs. For
datasets: statistics, details of train/validation/test splits, excluded data and preprocessing, a download
link. For code: dependency specification, training and evaluation code, pre-trained models, and a README
with a results table and commands. For experimental results: the range of hyper-parameters considered and
the method used to select them; the exact number of training and evaluation runs; a clear definition of the
measures and statistics reported; a description of results with central tendency **and variation** (error
bars); average run time; and the computing infrastructure.

**The NeurIPS Paper Checklist** (current form): 16 items, each answered Yes / No / NA with a justification:
claims; limitations; theory, assumptions and proofs; experimental result reproducibility; open access to
data and code; experimental setting/details; experiment statistical significance (error bars or tests, and
*what variability they capture* and how they were computed); experiments compute resources (worker type,
memory, time per run, total compute including preliminary experiments); code of ethics; broader impacts;
safeguards; licences; new assets; crowdsourcing and human subjects; IRB approvals; declaration of LLM usage.

**DOME** (Walsh et al. 2021, *Nature Methods*): recommendations for supervised ML in biology, organised as
**D**ata (provenance, splits, redundancy between training and test, availability), **O**ptimisation
(algorithm, data encoding, parameters, features, fitting strategy, regularisation, availability of
configuration), **M**odel (interpretability, output, execution time, availability) and **E**valuation
(protocol, performance measures, comparison with baselines, confidence in the reported numbers, availability
of raw evaluation files).

**Heil et al. (2021), "Reproducibility standards for machine learning in the life sciences"**
(*Nature Methods*): three levels. *Bronze*: data, models and code are published. *Silver*: additionally,
dependencies are specified and the environment can be recreated (e.g. a container), and the analysis is
deterministic or seeds are fixed. *Gold*: additionally, the entire analysis can be reproduced with a single
command.

**MIABi** (Tan et al. 2010, *BMC Genomics*), "Minimum Information About a Bioinformatics investigation",
proposed by the Asia-Pacific Bioinformatics Network: persistence of data and software (they must remain
available), re-instantiability of databases, reproducibility of results, and unambiguous identification of
authors and contributors. It is older and less prescriptive than DOME, but its emphasis on *persistence*
(archived, not just "available on request") is now in most journal policies.

**Journal policies.** *Bioinformatics* (author guidelines, checked 2026-10-01): a data availability
statement is required; software and source code must be freely available to non-commercial users at a
stable URL without request, with test data; and the submitted version should be archived in a repository
such as Zenodo, Figshare or Software Heritage. *Briefings in Bioinformatics* also requires a data
availability statement and encourages sharing all data and code on which the conclusions rely.

### 12.2 This project against the checklists

| Item | Status | Action before submission |
|---|---|---|
| Code for every step, one-command pipeline | done (`run_all.ps1`) | add a Linux/macOS shell equivalent |
| Seeds fixed and derived per fold | done | list seeds in the paper |
| Per-fold results and raw predictions stored | done (`*.json`, `*_preds.npz`) | archive them |
| Tables/figures generated by script | done (`08_make_figures.py`) | |
| Version control | **missing** (no commits) | `git init`, commit, tag the submitted version |
| Exact software versions | partial (README names Python/PyTorch/CUDA) | add a lock file |
| Data versions and dates | partial (in file headers) | write a provenance file; add MONDO/MedGen URLs to the download script |
| Live API results | not snapshotted | save ClinicalTrials.gov responses with dates |
| Hyper-parameter search documented | partial (`05_sensitivity.py`, defaults in `MVHGATConfig`) | add the baselines' tuning grids |
| Variation reported | done (mean ± SD) | add paired tests and say what ± means |
| Compute resources | partial ("about 4 h on an RTX 3050") | add time per method and per protocol |
| Licence for code | missing | add `LICENSE` (e.g. MIT or Apache-2.0) |
| Archive with DOI | missing | Zenodo release of code + processed data |

### 12.3 Data and code availability statements

They should be specific, with persistent links and licences. Templates for this project (replace the
placeholders):

> **Code availability.** The source code of MV-HGAT, the five baseline re-implementations and all
> experiment scripts is available at https://github.com/&lt;user&gt;/&lt;repo&gt; under the MIT licence. The
> exact version used in this paper is archived at Zenodo (https://doi.org/10.5281/zenodo.&lt;id&gt;). A
> single script (`scripts/run_all.ps1`) reproduces all results, tables and figures; the software
> environment is specified in `requirements-lock.txt`.

> **Data availability.** The Fdataset and Cdataset benchmarks are publicly available (Gottlieb et al.,
> 2011; Luo et al., 2016). All other inputs are public: CTD (https://ctdbase.org; files generated 29
> September 2026; used under CTD's terms, which require citation and prohibit commercial use without
> permission), MONDO (release 2026-09-01; CC BY 4.0), the Human Disease Ontology (release 2026-09-30;
> CC0), NCBI MedGen and PubChem. Scripts to download and process them are provided. The processed
> feature files (`Fdataset.npz`, `Cdataset.npz`), per-fold results and raw predictions are archived with
> the code (Zenodo DOI above). Clinical trial counts were retrieved from the ClinicalTrials.gov API v2 on
> &lt;date&gt;; the raw responses are included.

Note what the templates avoid: "available upon reasonable request" (journals increasingly reject it; it is
also how data disappear), unversioned links, and missing licences.

---

## 13. Getting through peer review

### 13.1 Common reasons for rejection of computational methods papers

| Reason | What it looks like | How this project avoids it (or must) |
|---|---|---|
| Incremental novelty | "another GNN on Fdataset" | frame the contribution: multi-view evidence with an exactly decomposable head, honest cold-start and cross-dataset evaluation |
| Weak or unfair baselines | defaults vs tuned model; published numbers mixed with own | same splits, information parity ablation, documented tuning |
| Leakage | test links visible during training; similarities derived from labels | unit E1; say how each leak is prevented |
| No uncertainty or tests | single numbers, "significantly better" without a test | mean ± SD, paired corrected tests |
| Overclaiming | "state of the art", "validated", "superior" | section 7.6 |
| Missing ablations | cannot tell which component helps | `04_ablation.py` (report all rows) |
| Anecdotal case studies | only confirmed hits, no denominator | section 11 |
| Irreproducible | no code, broken links, missing data | section 12 |
| Unclear writing | reviewers cannot follow the model | equations, a model figure, consistent terminology |
| Out of scope | wrong journal | read the journal's aims and recent papers |

### 13.2 How to respond to reviewers

When the decision is "major revision", the paper has a real chance; the response letter decides the
outcome. Noble (2017, "Ten simple rules for writing a response to reviewers") gives a reliable recipe:
(1) provide an overview, then quote the full set of reviews; (2) be polite and respectful of all reviewers;
(3) accept the blame (if a reviewer misunderstood, the text was unclear); (4) make the response
self-contained; (5) respond to every point raised; (6) use typography (e.g. reviewer text in italics,
changes in a different colour) to help the reviewer navigate; (7) whenever possible, begin each response
with a direct answer to the point; (8) when possible, do what the reviewer asks; (9) be clear about what
changed relative to the previous version (quote the new text with its location); (10) if necessary, write
the response twice (once to vent, once to send).

**Worked example.**

> *Reviewer 2, comment 3: "The authors claim their method outperforms the baselines, but Figure 3 suggests
> SCMFDD has a higher precision at low recall. Please clarify."*
>
> **Response.** Thank you; the reviewer is right, and our original wording overstated the result.
> MV-HGAT has a higher AUC than all baselines in every fold, but its AUPR is statistically
> indistinguishable from SCMFDD's on both datasets (Fdataset: 0.488 ± 0.027 vs 0.495 ± 0.020, corrected
> resampled t-test p = 0.70), and SCMFDD is more precise at recall below about 40%, i.e. among the top-ranked predictions. We have
> (i) revised the abstract and Section 5.2 accordingly; (ii) added paired fold-level comparisons for all
> methods and metrics (new Supplementary Table S4); and (iii) added a sentence to the Discussion on the
> complementary strengths of the two approaches. The revised text in Section 5.2 now reads: "..."
> (page 7, lines 12-20, changes in blue).

Notice: a direct answer first, agreement where it is due, specific changes with locations, new evidence,
and no defensiveness. When you disagree, do so politely with evidence ("We respectfully disagree, because
... ; to address the concern nonetheless we added ...").

---

## 14. In this project: the paper, section by section

This section is a complete outline of the paper this project would produce, with suggested wording for the
key sentences (using the real numbers available on 2026-10-01) and, for every result, the script, command,
output file and figure that produce it. Treat the wording as a draft to adapt, not to paste: re-check every
number against `results/RESULTS.md` after your final runs.

### 14.1 Mapping the proposal's Results (5.1-5.7) to the code

| Proposal item | Paper section | Command (from the project root) | Output | Figure / table |
|---|---|---|---|---|
| 5.1 AUC/AUPR mean ± std | 3.1 | `python scripts/03_evaluate.py --dataset F --protocol cv5 --repeats 5` (and `--dataset C`; `--protocol cv10 --repeats 1`) | `results/<Dataset>/cv5/<method>.json` (and `cv10/`) | Table 2 (via `08_make_figures.py` → `RESULTS.md`) |
| 5.2 Baseline comparison | 3.1-3.2 | same runs (all 6 methods by default) + paired tests (Block 3) | same JSONs (`folds`) | Table 2; Supplementary Table S4 (paired tests) |
| Cold start (Objective 4) | 3.2 | `python scripts/03_evaluate.py --dataset F --protocol lodo --lodo-subset 100` (full: drop `--lodo-subset`) | `results/<Dataset>/lodo/*.json` | Table 3 |
| 5.3 ROC/PR curves | 3.3 | `python scripts/08_make_figures.py` | `results/figures/curves_<Dataset>_<protocol>.png` | Figure 2 (+ AUPR dot plot, Block 4) |
| 5.4 Ablation | 3.4 | `python scripts/04_ablation.py --dataset F --protocol cv5 --repeats 1` (use ≥ 3 repeats for the final paper) | `results/Fdataset/ablation_cv5/*.json` | Figure 3 (`ablation_Fdataset_ablation_cv5.png`), Table S5 |
| 5.5 Parameter sensitivity | 3.5 | `python scripts/05_sensitivity.py --dataset F --seeds 2` | `results/Fdataset/sensitivity.json` | Figure 4 (`sensitivity_Fdataset.png`) |
| 5.6 Case studies (top-10) | 3.6 | `python scripts/06_case_study.py --dataset C` | `results/case_studies/Cdataset_OMIM*.csv`, `*_case_studies.md`, `*_view_attention.csv`; log `results/logs/C_case.log` | Table 4; Figure 5 (`view_attention_Cdataset.png`) |
| 5.7 Cross-dataset validation | 3.7 | `python scripts/07_cross_dataset.py --source F --target C` (and C → F) | `results/cross_Fdataset_to_Cdataset.json` | Table 5 |
| Everything | | `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1` | all of the above | |

### 14.2 Title and abstract

**Title options** (one central contribution, specific, no hype):

* "Multi-view heterogeneous graph attention with an interpretable propagation head for drug repositioning"
* "MV-HGAT: integrating chemical, ontological and genomic similarity views for interpretable drug-disease
  association prediction"

**Structured abstract (draft, about 200 words; trim to the journal's limit):**

> **Motivation:** Computational drug repositioning ranks existing drugs as candidates for new indications.
> Most graph-based methods use a single drug and a single disease similarity, and their attention weights
> are often presented as explanations without evidence that they are faithful.
>
> **Results:** We present MV-HGAT, a heterogeneous graph attention network over three drug views
> (two chemical fingerprints, gene interactions) and three disease views (phenotype, MONDO ontology,
> curated genes), whose decoder adds an additive multi-view propagation term in which each view's
> contribution to a prediction is exact. On two benchmarks with identical cross-validation folds for all
> methods, MV-HGAT achieved the highest AUC (Fdataset 0.939 ± 0.007; Cdataset 0.960 ± 0.005; five
> re-implemented baselines ≤ 0.913), with AUPR comparable to the best baseline (0.488 vs 0.495; 0.586
> vs 0.588). [Cold-start result.] Trained on Fdataset, it recovered associations present only in Cdataset
> with AUC 0.94. For prostate cancer, all ten top-ranked new candidates had independent support in CTD or
> ClinicalTrials.gov.
>
> **Availability and implementation:** Code, data processing scripts and results: [URL], archived at
> [DOI].

Notice what the abstract does: numbers with uncertainty, the honest AUPR comparison, a placeholder for the
cold-start result (do not invent it), and a modest case-study statement.

### 14.3 Introduction (four paragraphs + contributions)

1. *Context.* "Developing a new drug takes over a decade and costs billions; repositioning approved drugs
   shortens this path because their safety profiles are known." (cite a review with the figures you quote)
2. *Prior work.* Network propagation (MBiRW), matrix completion and factorisation (DRRS, SCMFDD), graph
   neural networks (NIMCGCN, LAGCN, DRAGNN, HEDDI-Net). "These methods typically integrate one drug and
   one disease similarity, although many complementary similarities are available."
3. *Gap.* "First, it is unclear how much each type of evidence contributes; second, attention weights are
   often presented as explanations although attention is not necessarily faithful (Jain & Wallace, 2019);
   third, evaluation is mostly warm-start, while newly characterised diseases require cold-start prediction."
4. *This work.* "We propose MV-HGAT ... and evaluate it with identical folds for all methods, a
   cold-start protocol, cross-dataset validation and case studies."

**Contributions** (each must be verifiable in the Results):

* a multi-view heterogeneous graph with six similarity views, including three computed for this work
  (ECFP fingerprints, MONDO semantic similarity, CTD gene overlap);
* a decoder with an additive multi-view propagation head and a degree gate, whose per-view terms are exact
  contributions to the prediction;
* an evaluation with re-implemented baselines on identical folds, paired statistical tests, ablations,
  cold-start and cross-dataset validation;
* case studies with evidence attribution by occlusion and neighbour-level explanations.

### 14.4 Materials and methods (subsections)

2.1 Datasets (benchmarks, sizes, ID recovery; correction of the sizes stated in the proposal).
2.2 External resources and identifier mapping (PubChem, CTD, MONDO, MedGen; versions; coverage table).
2.3 Similarity views (six formulas; missing values; kNN graphs).
2.4 Heterogeneous graph (relations; known links from the training fold only; gene meta-paths).
2.5 MV-HGAT (node-level and view-level attention; jumping knowledge; decoder; propagation head; gate).
2.6 Training (loss, negative sampling, hidden-link supervision, cold-start practice, optimiser).
2.7 Interpretability (global $w_v$, exact head contributions, view attention as a descriptive statistic,
occlusion; unit E2).
2.8 Evaluation protocols and metrics (CV, LODO, cross-dataset; AUC, AUPR; statistics).
2.9 Baselines (five; re-implementation details; inputs; tuning).
2.10 Implementation (software versions, hardware, run time, seeds).

Use sections 9.3-9.5 as templates for 2.1, 2.6 and 2.8.

### 14.5 Results, with suggested wording

**3.1 Overall performance (proposal 5.1, 5.2).** Table 2: both datasets, 5-fold (5 repeats) and 10-fold,
six methods, AUC and AUPR as mean ± SD.

> "MV-HGAT achieved the highest AUC on both benchmarks and under both protocols (Table 2): 0.939 ± 0.007
> on Fdataset and 0.960 ± 0.005 on Cdataset with 5-fold CV (10-fold: 0.954 ± 0.005 and 0.970 ± 0.003),
> exceeding the strongest baseline, SCMFDD, in every fold (paired, corrected resampled t-test p < 0.001
> in all four settings). Its AUPR was higher than that of MBiRW, DRRS, NIMCGCN and LAGCN, and statistically
> indistinguishable from SCMFDD's (Fdataset 0.488 ± 0.027 vs 0.495 ± 0.020; Cdataset 0.586 ± 0.026 vs
> 0.588 ± 0.021; Supplementary Table S4)."

(The 10-fold rows come from a single repetition; say so in the caption, and give the 10-fold AUPR values
too: Fdataset 0.524 ± 0.034 vs SCMFDD 0.546 ± 0.027; Cdataset 0.619 ± 0.032 vs 0.636 ± 0.034.)

**3.2 Cold start.** Table 3 from the LODO runs. Write it only when the runs are complete; compare against
MBiRW explicitly, since `HOW_IT_WORKS.md` section 10 reports it is a strong cold-start baseline.

> "In leave-one-disease-out evaluation on [all / a random subset of 100] diseases, MV-HGAT achieved
> [pooled AUC, AUPR, mean per-disease AUC] compared with [...] for MBiRW ..."

**3.3 ROC and precision-recall curves (5.3).** Figure 2.

> "MV-HGAT's ROC curve lies above those of all baselines across the whole range of false-positive rates
> (Figure 2a; e.g. true-positive rate 0.86 vs 0.79 for SCMFDD at a false-positive rate of 0.1). The
> precision-recall curves (Figure 2b) show where the AUPR tie comes from: SCMFDD is more precise at recall
> between about 5% and 40% (e.g. 0.91 vs 0.82 at 20% recall), the two are equal near 50%, and MV-HGAT is
> more precise beyond (0.38 vs 0.29 at 60% recall)."

(These values are read from the pooled predictions of the first CV repetition on Fdataset, with
interpolated precision; recompute them for every dataset you describe. Caption: "Curves from pooled
predictions of the first CV repetition; values in the legend are means over all 25 folds; dotted line:
random classifier.")

**3.4 Ablation (5.4).** Figure 3 / Table S5, every variant reported.

> "Removing the multi-view propagation head caused the largest drop (AUPR 0.433 ± 0.034 vs 0.500 ± 0.029,
> lower in 4 of 5 folds; AUC 0.934 vs 0.943), followed by removing the ECFP view (AUPR 0.463) and the gene
> views (0.468). Restricting the model to the two benchmark similarities reduced AUPR modestly (0.481)
> but left AUC unchanged (0.943), indicating that the AUC advantage over baselines stems from the model
> rather than from the additional views. Contrary to our expectation from development experiments,
> disabling hidden-link supervision increased AUPR (0.546 ± 0.020, higher in all 5 folds), and removing
> cold-start practice or the MONDO view did not reduce warm-start performance; these components are
> intended for cold-start prediction, which we evaluate in Section 3.2."

(This is single-repeat data: 5 folds. Before submission, re-run the ablation with at least 3 repeats and
check that the conclusions survive; if hidden-link supervision still hurts warm-start AUPR, report it and
show whether it helps cold start, which is its purpose.)

**3.5 Sensitivity (5.5).** Figure 4 (validation split, 2 seeds; one parameter varied at a time).

> "Performance was stable across hyper-parameters: validation AUC stayed between 0.932 and 0.944 for all
> settings tested. AUPR was more sensitive: it peaked at the default embedding size (64; 0.521 vs 0.465-0.479
> for 16, 32 and 128) and decreased for large neighbourhoods (k = 40: 0.475) and high edge-hiding rates
> (0.6: 0.483)."

**3.6 Case studies and interpretability (5.6).** Table 4 (all 10 candidates per disease, with the support
columns and top-2 evidence sources), Figure 5 (view attention), and one neighbour-level explanation.

> "We trained MV-HGAT on all known associations of Cdataset (five seeds, averaged) and examined the ten
> highest-ranked drugs not already associated with Alzheimer's disease, breast cancer and prostate cancer
> (Table 4). Applying a pre-specified support criterion (a curated CTD chemical-disease association, an
> association in Fdataset, or at least one ClinicalTrials.gov record), 10/10 candidates for prostate cancer,
> 6/10 for breast cancer and 5/10 for Alzheimer's disease had independent support. The top candidate for
> Alzheimer's disease, amantadine (49 trial records), was driven by chemical-similarity evidence: its
> nearest structural neighbour, memantine, an approved Alzheimer's drug, accounted for all of its ECFP
> propagation evidence. Several other Alzheimer's candidates were dopaminergic drugs approved for
> Parkinson's disease, reflecting the proximity of neurodegenerative diseases in the association data
> rather than an established mechanism."

Add the reference rate (exercise 8) if you can compute it, and a sentence on the attention figure that
labels it as descriptive (unit E2, section 5.5).

**3.7 Cross-dataset validation (5.7).** Table 5.

> "Fdataset and Cdataset share 574 drugs and 307 diseases. Trained on all of Fdataset, MV-HGAT ranked the
> 57 associations that are present only in Cdataset among 174,330 candidate pairs with AUC 0.944 (best of
> all methods; DRRS 0.930, MBiRW 0.898). AUPR was low for every method (MV-HGAT 0.0086, MBiRW 0.0136;
> random 0.0003), as expected for 57 positives among 174,330 pairs. The reverse direction could not be
> evaluated: every Fdataset association among the shared drugs and diseases is also in Cdataset."

### 14.6 Discussion, limitations, conclusion

* *Why it works:* ablation evidence for the propagation head; neighbourhood propagation is strong on these
  benchmarks; the GNN adds collaborative signal for well-connected entities (degree gate).
* *Complementarity with matrix factorisation:* AUC vs AUPR pattern.
* *Interpretability:* the head gives exact per-view contributions; attention is descriptive; occlusion
  shows predictions draw on diverse evidence; the memantine example illustrates neighbour-level
  explanations, and their fragility.
* *Limitations:* section 10.3.
* *Future work:* DrugBank targets as a view; LINCS L1000; nested CV; full LODO; prospective validation.
* *Conclusion:* one paragraph, no new claims.

### 14.7 Back matter and supplementary material

Code and data availability (section 12.3), funding, competing interests, author contributions,
acknowledgements (data providers: CTD asks to be cited). Supplementary: full four-decimal tables (`RESULTS.md`),
per-fold results, all case-study tables for both datasets, hyper-parameter grids for all methods, identifier
mapping coverage, the run manifest.

### 14.8 Templates from the literature

`PREREQUISITES.md` recommends two recent papers as models for the Results structure: **DRAGNN** (Meng et
al. 2024, *Briefings in Bioinformatics*; a weighted local-information-augmented GNN evaluated with 10
times 10-fold CV on three benchmarks) and **HEDDI-Net** (Su et al. 2025, *Journal of Translational
Medicine*; heterogeneous network embedding with an Alzheimer's disease case study and recovery analyses).
Read their Results sections with section 10 of this chapter in hand: note how they order subsections, how
they caption tables, how they phrase case-study support, and where you would ask them for more (e.g.
paired tests, reference rates).

---

## 15. Common mistakes and misconceptions

1. **"I fixed the seed, so my results are reproducible."** Only on the same software, hardware and code.
   Record those too, and report variation across seeds.
2. **Writing "±" without saying what it is** (SD? SEM? CI? over folds? over repeats?).
3. **Treating CV folds as independent** in confidence intervals and t-tests.
4. **Unpaired comparisons of paired data**, which throws away the power of identical folds.
5. **"Significantly better"** without a test, or with a naive test on correlated folds, or with 40 tests and
   no correction.
6. **Mixing published and re-implemented baseline numbers** in one table.
7. **Tuning your method and not the baselines** (or tuning on test folds).
8. **Bolding the best number** when the difference is within noise.
9. **Truncated bar-chart axes**, dual y-axes, rainbow colour maps, red/green-only encodings.
10. **Unpinned dependencies** and "the latest version" of a database without a date.
11. **Manual steps** that are not scripted (the MONDO and MedGen downloads here).
12. **Describing the proposal instead of what was done** in the Methods (DrugBank/MeSH vs PubChem/MONDO).
13. **Hiding inconvenient ablation rows** or unfinished experiments.
14. **Case studies without a denominator or reference rate**, and "validated" for database hits.
15. **"Data available upon request"**.
16. **Arguing with reviewers instead of answering them.**

---

## 16. Exercises

Difficulty: ★ routine, ★★ needs thought, ★★★ a small project.

**Exercise 1 ★ (classify).** Which of reproducible / replicable / robust / generalisable does each establish?
(a) A colleague runs `run_all.ps1` on her laptop and gets AUCs within 0.002 of yours. (b) You run MV-HGAT on
a third benchmark (e.g. DNdataset) and it again beats the baselines in AUC. (c) You re-implement the model
in DGL and get the same conclusions on Fdataset. (d) A different group, with its own code and its own
dataset, finds that multi-view propagation helps.

<details><summary>Solution</summary>

(a) Reproducible (same data, same analysis), at the numerical level. (b) Replicable (same analysis,
different data). (c) Robust (same data, different implementation of the analysis). (d) Generalisable
(different analysis and different data).
</details>

**Exercise 2 ★ (SD, SEM, CI).** The five repeat-level AUPR means of MV-HGAT on Fdataset are 0.5004,
0.4864, 0.4880, 0.4896, 0.4747. Compute the mean, SD, SEM and a 95% t-interval. Compare the interval with
the "naive" fold-level interval [0.476, 0.499] from Block 3 and comment.

<details><summary>Solution</summary>

Mean 0.4878. Deviations: 0.0126, −0.0014, 0.0002, 0.0018, −0.0131; squares sum ≈ 0.000336; divided by 4:
0.0000840; SD ≈ 0.0092. SEM = 0.0092/√5 = 0.0041. $t_{0.975,4}=2.776$, half-width 0.0114: interval
[0.476, 0.499]. (Exact values from the unrounded data: SD 0.00916, SEM 0.00410, interval [0.4764, 0.4992].)
It happens to coincide with the naive fold-level interval here, because two effects cancel: the
repeat-level analysis has far fewer units (5 vs 25, and $t_{0.975,4}$ is larger) but much less spread
(averaging 5 folds removes fold-to-fold noise). Do not count on this; the repeat-level interval is the
more defensible one because its units are closer to independent, but it still ignores the fact that all
repeats use the same data, so it describes the variability of the procedure on *this* dataset only.
</details>

**Exercise 3 ★ (seeding).** A script calls `torch.manual_seed(0)` at the top, then creates the model on
the GPU, uses `index_add_` in the forward pass, and shuffles a Python list with `random.shuffle`. Two runs
give different results. Name every likely cause and the fix for each.

<details><summary>Solution</summary>

(1) Python's `random` is not seeded: add `random.seed(0)`. (2) `index_add_` on CUDA uses atomic additions
whose order varies: call `torch.use_deterministic_algorithms(True)` (PyTorch then uses a deterministic
implementation where one exists, or raises an error telling you which operation is the problem).
(3) cuBLAS matrix products may be non-deterministic: set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the first
CUDA call. (4) cuDNN algorithm selection: `torch.backends.cudnn.benchmark = False`. (5) If NumPy is used
anywhere: `np.random.seed(0)` or a seeded `default_rng`. After the fixes, results should be identical on
the same machine; across machines, expect small numerical differences.
</details>

**Exercise 4 ★★ (corrected t-test by hand).** For Fdataset 10-fold CV (one repetition, 10 folds), the
AUPR differences MV-HGAT − SCMFDD have mean −0.0219 and SD 0.0351. Compute the naive paired t statistic and
the Nadeau-Bengio corrected one, and their two-sided p-values (df = 9; $t_{0.975,9}=2.262$). What do you
conclude?

<details><summary>Solution</summary>

Naive: SE = 0.0351/√10 = 0.0111; t = −0.0219/0.0111 = −1.97; p ≈ 0.08. Corrected: variance multiplier
$1/J+1/(k-1)=1/10+1/9=0.211$; SE = √(0.211 × 0.0351²) = 0.0351 × 0.459 = 0.0161; t = −1.36; p ≈ 0.21.
(Code gives t = −1.36, p = 0.208.) Neither reaches 0.05; with the correction the evidence is weak. Conclusion:
no detectable AUPR difference, although the point estimate favours SCMFDD in this setting (MV-HGAT wins 2 of
10 folds). Report it exactly like that.
</details>

**Exercise 5 ★★ (multiple comparisons).** The corrected p-values for MV-HGAT vs SCMFDD across the four
warm-start settings are: AUC 5.1e-10, 7.9e-4, 9.1e-13, 4.6e-5; AUPR 0.70, 0.21, 0.87, 0.22. Apply Holm's
procedure at α = 0.05.

<details><summary>Solution</summary>

m = 8. Sorted: 9.1e-13 (≤ 0.05/8 = 0.00625, reject), 5.1e-10 (≤ 0.05/7, reject), 4.6e-5 (≤ 0.05/6, reject),
7.9e-4 (≤ 0.05/5 = 0.01, reject), 0.21 (> 0.05/4 = 0.0125, stop). All four AUC differences are significant
after correction; none of the AUPR differences is.
</details>

**Exercise 6 ★★ (figure critique).** A draft figure shows a bar chart of AUPR for six methods with the
y-axis starting at 0.40, bars coloured with the `jet` colour map, error bars not explained, and a second
y-axis on the right showing training time. List every problem and the fix.

<details><summary>Solution</summary>

(1) Truncated bar axis exaggerates differences (bars encode length): start at 0, or use a dot plot and say
the axis is truncated. (2) Rainbow (`jet`) colours are not perceptually ordered and not colour-blind safe,
and colour encodes nothing here: use one neutral colour plus a highlight for the proposed method, or the
fixed method palette. (3) Error bars unexplained: state SD over 25 folds (or whatever they are) in the
caption. (4) Dual y-axis: move training time to a separate panel or a table. (5) Bars hide the
distribution: overlay fold-level points. (6) Add the random-level reference line for AUPR.
</details>

**Exercise 7 ★★ (Methods gaps).** List at least eight facts missing from: "We built a heterogeneous graph
from drug and disease similarities and trained a GAT with attention. Hyper-parameters were tuned. The model
was evaluated by cross-validation."

<details><summary>Solution</summary>

Which similarities and how computed (formulas, fingerprints, ontologies, versions); how the graph is
sparsified (kNN, k); which links are edges (training fold only?); architecture details (layers,
dimensions, heads, attention levels, decoder); loss and negative sampling; optimiser, learning rate, epochs,
regularisation, dropout; which hyper-parameters were tuned, over what ranges, on which data, by what
criterion; CV details (k, repetitions, how negatives are split, identical folds for baselines); metrics and
what is averaged; seeds; software and hardware; baselines and their inputs.
</details>

**Exercise 8 ★★★ (a reference rate for case-study support).** Design an analysis that tells you whether
"6/10 supported" for breast cancer is better than chance, and sketch the code.

<details><summary>Solution</summary>

For each case-study disease, apply the same support rule to (a) 10 drugs drawn at random from the drugs not
linked to the disease, repeated 1,000 times, giving a null distribution of the support count, and (b) the
top 10 of a simple baseline (e.g. MBiRW, or ranking drugs by their number of known indications, a
"popularity" baseline). Report MV-HGAT's count with the null mean and the empirical p-value
($\frac{1+\#\{\text{null}\ge\text{observed}\}}{1+1000}$). Sketch:
```python
import numpy as np
cand = [i for i in range(data.n_drugs) if data.A[i, j] == 0]
support = np.array([supported(i, j) for i in cand])        # CTD / other dataset / trials, cached
rng = np.random.default_rng(0)
null = [support[rng.choice(len(cand), 10, replace=False)].sum() for _ in range(1000)]
p = (1 + sum(n >= observed for n in null)) / 1001
```
Cache the trial lookups (one query per candidate drug) and record their date. Because popular drugs are
both more often predicted and more often trialled, the popularity baseline (b) is the more demanding
comparison.
</details>

**Exercise 9 ★★ (data availability with restrictions).** Suppose you later add a DrugBank target view
built from the full DrugBank XML (CC BY-NC 4.0, academic account required). Write the corresponding
sentences of the data availability statement.

<details><summary>Solution</summary>

"Drug-target annotations were obtained from DrugBank (version X.Y, released <date>) under its academic
licence (CC BY-NC 4.0). Because redistribution requires a DrugBank licence, we do not include the raw
DrugBank file; the script `02_build_features.py` regenerates the target view from a user-supplied copy.
The derived drug-drug target similarity matrix is included [only if the licence permits redistributing
derived data for non-commercial use; check the terms], together with checksums that allow users to verify
their regenerated files." The key elements: version and date, licence, what can and cannot be shared, and
how a reader can regenerate what you cannot share.
</details>

**Exercise 10 ★★ (why do two "full model" numbers differ?).** `RESULTS.md` reports MV-HGAT's Fdataset
5-fold AUPR as 0.4878 ± 0.0273, while the ablation's "Full MV-HGAT" row reports 0.5004 ± 0.0285. A reviewer
asks whether the model changed between experiments. Answer.

<details><summary>Solution</summary>

The model and code are the same. The main experiment uses 5 repetitions × 5 folds (25 values); the
ablation used 1 repetition (`--repeats 1`), i.e. the first 5 folds only. Those 5 folds are identical in both
experiments (same split seed) and give identical fold-level metrics (Section 1, story 1); their mean is
0.5004, while the mean over all 25 folds is 0.4878 (the first repetition happened to be the easiest). We
will state the number of repetitions in every caption and, for the final version, re-run the ablation with
5 repetitions so that its "full" row matches Table 2.
</details>

**Exercise 11 ★★★ (store provenance with every result, without changing the project).** Write a function
`save_with_provenance(res, dataset, protocol, key, cfg)` that you could use *instead of* `runner.save` in a
new script: it should call the original `save` and then add `config`, `git_commit`, `environment` and
`created_utc` to the JSON.

<details><summary>Solution</summary>

```python
import json, platform, subprocess, sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from drepo.paths import RESULTS
from drepo.runner import save

def save_with_provenance(res, dataset, protocol, key, cfg, extra=None):
    save(res, dataset, protocol, key, extra)                    # original behaviour
    f = RESULTS / dataset / protocol / f"{key}.json"
    j = json.loads(f.read_text())
    commit = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    j["provenance"] = {
        "config": asdict(cfg) if cfg is not None else None,
        "git_commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": {p: version(p) for p in ("torch", "numpy", "scikit-learn")},
        "command": " ".join(sys.argv),
    }
    f.write_text(json.dumps(j, indent=2))
```
(For baselines, which have no config dataclass, pass `vars(method)` instead.) Note `asdict` turns the
tuples in `MVHGATConfig` into lists; that is fine for JSON.
</details>

**Exercise 12 ★★★ (separating sources of variance).** Design an experiment that separates the variance due
to the data split from the variance due to the model seed, for MV-HGAT on Fdataset.

<details><summary>Solution</summary>

Two-factor design: S split seeds × M model seeds (e.g. 5 × 5), every combination trained and evaluated
(5-fold CV each, so 125 fits). For a metric $y_{sm}$ (mean over the folds of split $s$ with model seed
$m$), estimate variance components with a two-way random-effects ANOVA:
$\mathrm{Var}(y)=\sigma^2_{\text{split}}+\sigma^2_{\text{seed}}+\sigma^2_{\text{residual}}$. If seed variance is
comparable to the difference between two methods, the comparison needs more seeds; if split variance
dominates, more CV repetitions. Bouthillier et al. (2021, "Accounting for variance in machine learning
benchmarks") show that ignoring such sources often leads to false conclusions and recommend randomising as
many of them as possible. Budget: on the RTX 3050 a 5-fold run takes about 2 minutes, so 125 fits ≈ 4 h.
</details>

**Exercise 13 ★ (response letter).** A reviewer asks you to compare against the AUC values *published* in
the LAGCN paper. Write a short response.

<details><summary>Solution</summary>

"We thank the reviewer. We have added the published values in a separate Supplementary Table S6, labelled
'as reported in Yu et al. (2021)'. We do not merge them into Table 2 because they were obtained with
different splits and preprocessing, and are therefore not directly comparable with our results, in which
all methods were evaluated on identical folds. Our re-implementation of LAGCN reaches AUC 0.833 ± 0.015 on
Fdataset; the difference from the published value may reflect [different negative sets / tuning / omitted
components; state what you know]. To address this concern further, we [ran the authors' public code on our
splits / tuned our re-implementation over the grid in Table S3], obtaining [...]." Never state a published
number you have not checked in the original paper.
</details>

---

## 17. Self-check: questions and answers

`PREREQUISITES.md` lists no explicit self-check questions for E3, only what to learn: fixed seeds; recording
versions of data (the date you downloaded CTD); reporting mean ± std; honest comparison with baselines
(re-implementations vs published numbers); writing a methods section that someone else can re-run. The
questions below test exactly those items.

**Q1. Why do we fix seeds, and what does a fixed seed *not* guarantee?**
Seeds make every random choice (fold assignment, initialisation, negative sampling, dropout, edge hiding)
repeatable, so a result can be regenerated exactly, debugged fold by fold, and compared across methods on
identical splits (the project derives each fit's seed from the repetition and fold). A seed does not
guarantee identical numbers on different hardware (CPU vs GPU, different GPUs), with different library
versions, or with non-deterministic GPU kernels; and one seed is one sample, so a fixed seed does not make a
result *reliable*, only *repeatable*. Reliability comes from reporting the spread over several seeds and
splits.

**Q2. Why record the date you downloaded CTD (and the versions of every other source)?**
CTD, PubChem, MONDO and ClinicalTrials.gov change continually: identifiers are merged, curated links are
added, ontologies are restructured, trials are registered. The same script run a year later produces
different features and different case-study support, so without the version or date nobody (including you)
can know which data produced the paper's numbers. Record the release stamp the file carries (CTD's "Report
created" line, MONDO's `data-version`), the download date, a checksum, the URL and the licence, at download
time, in a machine-readable file; snapshot live API responses.

**Q3. What should "mean ± std" be computed over, and how should it be reported?**
Over the independent-as-possible units of your protocol: in the project, all 25 test folds (5 repetitions ×
5 folds), which mixes split and seed variation; repeat-level means are a more conservative unit. Say which
SD (divisor $n$ or $n-1$), over what, and how many values. Remember it describes the spread of single runs,
not the precision of the mean; fold-level confidence intervals and t-tests are over-confident because folds
share training data, so use paired comparisons with the corrected resampled t-test, and round to the
precision the SD supports.

**Q4. Why must re-implemented and published baseline numbers be kept apart?**
Because they come from different experiments: different splits, negatives, preprocessing, similarity
inputs and tuning effort. Placing them in one table invites a comparison that measures those differences,
not the methods. Re-implemented baselines on identical folds (the project's approach) give a fair
comparison but carry the risk of a weaker re-implementation, which you must disclose (e.g. MBiRW without
its clustering step) and, ideally, check against the authors' code. Published numbers, if quoted, go in a
separate, clearly labelled table. Parity also covers tuning budget and information (the "benchmark
similarities only" ablation).

**Q5. What makes a Methods section re-runnable by someone else?**
Every decision that affects the numbers is stated: data sources with versions, sizes and preprocessing;
every formula (similarities, model, loss); every hyper-parameter and how it was chosen; the exact evaluation
protocol (splits, negatives, metrics, what is averaged, statistics); baselines and their settings; seeds,
software versions and hardware; and a link to archived code that runs the whole pipeline with one command.
The test: a competent stranger reproduces the main table without asking you anything.

---

## 18. Summary and cheat sheet

**Reproducibility**

* Reproducible (same data, same analysis) ≠ replicable (new data) ≠ robust (new analysis) ≠ generalisable.
* Seed all generators; derive per-run seeds from the run's identity; prefer local generators.
* `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark = False`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bitwise repeatability on one machine; across machines aim for
  statistical agreement.
* Record config, git commit, lock file, hardware, data versions/checksums/licences/dates; snapshot APIs.
* One command regenerates everything; no manual steps; outputs never hand-edited.

**Statistics**

| Quantity | Formula / rule |
|---|---|
| SD | $\sqrt{\sum(x_i-\bar x)^2/(n-1)}$ (say if you use $n$) |
| SEM | $s/\sqrt n$ |
| 95% CI | $\bar x\pm t_{0.975,n-1}\,s/\sqrt n$ (over-confident for CV folds) |
| Paired difference | $d_j=A_j-B_j$ on identical folds |
| Corrected resampled t | $t=\bar d/\sqrt{(1/J+1/(k-1))\,s_d^2}$, df $J-1$ |
| Holm | reject $p_{(i)}\le\alpha/(m-i+1)$ in order, stop at first failure |

**Figures and tables:** one message; one scale per axis; show the data; zero-based bars (say so if an axis
is truncated); validated colour-blind-safe fixed colours; direct labels; self-contained captions; raw data
stored.

**Honesty:** separate published from re-implemented numbers; tuning and information parity; report losses,
surprising ablations and unfinished experiments; "significant" only with a (corrected) test.

**Paper:** IMRaD; context-content-conclusion; Methods a stranger can re-run; Results as supported statements
(5.1-5.7 → section 14.1 map); Limitations before reviewers find them; case studies with pre-specified rule,
denominator and reference rate; availability statements with DOI and licence.

**Reviewers:** answer every point, directly, politely, with changes and locations.

---

## 19. Further resources

All links were checked on 2026-10-01. "Free" = openly readable; "paid" = publisher paywall.

**Reproducibility: principles and practice**

* Sandve, Nekrutenko, Taylor & Hovig (2013), "Ten Simple Rules for Reproducible Computational Research"
  (PLOS Comput. Biol.) - free - <https://doi.org/10.1371/journal.pcbi.1003285>
* Wilson et al. (2017), "Good enough practices in scientific computing" - free -
  <https://doi.org/10.1371/journal.pcbi.1005510>
* Noble (2009), "A Quick Guide to Organizing Computational Biology Projects" - free -
  <https://doi.org/10.1371/journal.pcbi.1000424>
* The Turing Way, definitions of reproducible / replicable / robust / generalisable - free -
  <https://book.the-turing-way.org/reproducible-research/overview/overview-definitions>
* National Academies (2019), *Reproducibility and Replicability in Science* - free -
  <https://doi.org/10.17226/25303>
* PyTorch documentation, "Reproducibility" notes - free -
  <https://docs.pytorch.org/docs/stable/notes/randomness.html>
* Wilkinson et al. (2016), "The FAIR Guiding Principles for scientific data management and stewardship" -
  free - <https://doi.org/10.1038/sdata.2016.18>
* Papers with Code, "Tips for releasing research code in Machine Learning" - free -
  <https://github.com/paperswithcode/releasing-research-code>

**Checklists and standards**

* Pineau et al. (2021), "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS
  2019 Reproducibility Program)" (JMLR) - free - <https://jmlr.org/papers/v22/20-303.html>
* The Machine Learning Reproducibility Checklist (v2.0, PDF) - free -
  <https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf>
* NeurIPS Paper Checklist guidelines - free - <https://neurips.cc/public/guides/PaperChecklist>
* Walsh et al. (2021), "DOME: recommendations for supervised machine learning validation in biology"
  (Nat. Methods) - free - <https://doi.org/10.1038/s41592-021-01205-4>; DOME registry - free -
  <https://dome-ml.org/>
* Heil et al. (2021), "Reproducibility standards for machine learning in the life sciences" (Nat. Methods)
  - paid (free author versions exist) - <https://doi.org/10.1038/s41592-021-01256-7>
* Tan et al. (2010), "Advancing standards for bioinformatics activities: persistence, reproducibility,
  disambiguation and Minimum Information About a Bioinformatics investigation (MIABi)" (BMC Genomics) -
  free - <https://pmc.ncbi.nlm.nih.gov/articles/PMC3005918/>
* Kapoor & Narayanan (2023), "Leakage and the reproducibility crisis in machine-learning-based science"
  (Patterns) - free - <https://doi.org/10.1016/j.patter.2023.100804>

**Journal policies**

* *Bioinformatics* (OUP) author guidelines - free - <https://academic.oup.com/bioinformatics/pages/author-guidelines>
* *Briefings in Bioinformatics* general instructions - free - <https://academic.oup.com/bib/pages/General_Instructions>
* CTD legal notices / terms of use (needed for your data statement) - free - <https://ctdbase.org/about/legal.jsp>

**Statistics of evaluation**

* Bengio & Grandvalet (2004), "No Unbiased Estimator of the Variance of K-Fold Cross-Validation" (JMLR) -
  free - <https://www.jmlr.org/papers/v5/grandvalet04a.html>
* Nadeau & Bengio (2003), "Inference for the Generalization Error" (Machine Learning) - paid -
  <https://doi.org/10.1023/A:1024068626366>
* Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised Classification Learning
  Algorithms" (Neural Computation) - paid - <https://doi.org/10.1162/089976698300017197>
* Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR) - free -
  <https://www.jmlr.org/papers/v7/demsar06a.html>
* Bouthillier et al. (2021), "Accounting for Variance in Machine Learning Benchmarks" - free -
  <https://arxiv.org/abs/2103.03098>
* Dodge et al. (2019), "Show Your Work: Improved Reporting of Experimental Results" - free -
  <https://aclanthology.org/D19-1224/>
* Melis, Dyer & Blunsom (2018), "On the State of the Art of Evaluation in Neural Language Models" - free -
  <https://arxiv.org/abs/1707.05589>
* Lipton & Steinhardt (2018), "Troubling Trends in Machine Learning Scholarship" - free -
  <https://arxiv.org/abs/1807.03341>

**Figures**

* Rougier, Droettboom & Bourne (2014), "Ten Simple Rules for Better Figures" - free -
  <https://doi.org/10.1371/journal.pcbi.1003833>
* Weissgerber et al. (2015), "Beyond Bar and Line Graphs: Time for a New Data Presentation Paradigm" (PLOS
  Biol.) - free - <https://doi.org/10.1371/journal.pbio.1002128>
* Crameri, Shephard & Heron (2020), "The misuse of colour in science communication" (Nat. Commun.) - free -
  <https://doi.org/10.1038/s41467-020-19160-7>
* Okabe & Ito, "Color Universal Design (CUD): how to make figures and presentations that are friendly to
  colorblind people" - free - <https://jfly.uni-koeln.de/color/>

**Writing and reviewing**

* Mensh & Kording (2017), "Ten simple rules for structuring papers" - free -
  <https://doi.org/10.1371/journal.pcbi.1005619>
* Zhang (2014), "Ten Simple Rules for Writing Research Papers" - free -
  <https://doi.org/10.1371/journal.pcbi.1003453>
* Whitesides (2004), "Whitesides' Group: Writing a Paper" (Adv. Mater.) - journal version paid
  <https://doi.org/10.1002/adma.200400767>; a freely hosted PDF copy -
  <https://mason.gmu.edu/~hjing2/advice%20for%20phd%20students/Writing%20a%20paper-George%20Whiteside.pdf>
* Noble (2017), "Ten simple rules for writing a response to reviewers" - free -
  <https://doi.org/10.1371/journal.pcbi.1005730>

**Template papers recommended in `PREREQUISITES.md`**

* Meng et al. (2024), "Drug repositioning based on weighted local information augmented graph neural
  network" (DRAGNN; Brief. Bioinform.) - free (PMC) - <https://pmc.ncbi.nlm.nih.gov/articles/PMC10686358/>
* Su et al. (2025), "HEDDI-Net: heterogeneous network embedding for drug-disease association prediction
  and drug repurposing, with application to Alzheimer's disease" (J. Transl. Med.) - free -
  <https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-024-05938-6>

**Archiving**

* Zenodo (free DOIs for code and data releases) - free - <https://zenodo.org/>
* Choose a licence (plain-language comparison of open-source licences) - free - <https://choosealicense.com/>

---

## 20. Glossary

* **Ablation** - an experiment that removes one component and re-trains, to measure its contribution.
* **Atomic operation** - a GPU memory update that cannot be interrupted; concurrent atomic additions
  complete in an unpredictable order, a source of non-determinism.
* **Availability statement** - the paper section stating where code and data are, under what licence.
* **Bitwise / numerical / statistical reproducibility** - identical bits / equal within tolerance / same
  conclusion within uncertainty.
* **Checksum (hash)** - a short fingerprint of a file (e.g. SHA-256); identical files have identical
  checksums.
* **Cherry-picking** - reporting only favourable datasets, metrics, seeds or examples.
* **Confidence interval** - a range that, under the stated assumptions, covers the true value in 95% of
  repeated experiments.
* **Corrected resampled t-test** - a paired t-test whose variance is inflated by $1/J+n_{\text{test}}/n_{\text{train}}$
  to account for overlapping training sets.
* **Data provenance** - the record of where data came from, which version, when, and how it was processed.
* **Deterministic algorithm** - an implementation that gives identical results on identical inputs.
* **DOME** - Data, Optimisation, Model, Evaluation: recommendations for reporting supervised ML in biology.
* **FAIR** - Findable, Accessible, Interoperable, Reusable data principles.
* **HARKing** - Hypothesising After the Results are Known.
* **Holm-Bonferroni** - a step-down correction controlling the family-wise error rate over several tests.
* **IMRaD** - Introduction, Methods, Results and Discussion.
* **Information parity** - baselines and the proposed method receive the same input information.
* **Lock file** - a file listing the exact version of every installed package.
* **MIABi** - Minimum Information About a Bioinformatics investigation.
* **Paired comparison** - comparing two methods on the same test units (folds), using their differences.
* **PRNG / seed** - pseudo-random number generator / the value that initialises its state.
* **Replicability** - consistent results with the same analysis on new data.
* **Reproducibility** - consistent results with the same data and the same analysis.
* **Run manifest** - a machine-readable record of the config, code version, environment and data of a run.
* **SD / SEM** - standard deviation (spread of values) / standard error of the mean (precision of the mean).
* **Structured abstract** - an abstract with fixed headings (e.g. Motivation, Results, Availability and
  implementation).
* **Tuning parity** - comparable hyper-parameter search effort for all compared methods.
