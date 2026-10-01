# Unit A1 — Python for Data Work: NumPy, pandas, Scripts and Virtual Environments

> Course: *Computational Drug Repositioning with Graph Neural Networks* — Track A (Foundations), Unit 1 of 20.
> Companion files: `docs/PREREQUISITES.md` (study plan), `docs/HOW_IT_WORKS.md` (project walkthrough).

---

## 0. Front matter

**Prerequisites.** None from this course; this is the first unit. You should be able to write a
small Python program (variables, `if`, `for`, functions). If you have never programmed at all,
work through the first six chapters of the official Python tutorial (linked in §11) before
starting, then come back.

**Estimated study time.** 18–24 hours in total:

| Part | Hours |
|---|---|
| §2–3 environments, scripts, Python essentials | 3–4 |
| §4 NumPy (read + run every example) | 7–9 |
| §5 pandas (read + run every example) | 4–5 |
| §6 project walkthrough | 2 |
| §8 exercises | 3–4 |

**Learning objectives.** After this unit you will be able to:

1. Create, activate and populate a virtual environment on Windows, explain what `pyvenv.cfg`
   and `site-packages` are, and diagnose "wrong interpreter" errors with `sys.executable`.
2. Explain the difference between a *script*, a *module* and a *package*, why the project's scripts
   call `sys.path.insert(0, ".../src")`, and what `if __name__ == "__main__":` does.
3. State the shape, dtype, memory size and strides of any NumPy array you meet in the project, and
   predict the result shape of any broadcasting expression **before** running it.
4. Convert between a flat index `k` and a (drug, disease) pair `(i, j)` in both directions, by hand
   and with `np.unravel_index` / `np.ravel_multi_index`.
5. Use basic slicing, integer ("fancy") indexing and boolean masks, and say for each whether the
   result is a *view* or a *copy*.
6. Select the top-k entries of every row of a matrix with `np.argpartition`, explain why it is
   asymptotically cheaper than `np.argsort`, and know when you still need a sort afterwards.
7. Produce reproducible random splits with `np.random.default_rng(seed)` and `np.array_split`.
8. Load, filter, group, merge and reshape tabular data in pandas; read a multi-gigabyte compressed
   file in bounded memory with `chunksize`.
9. Read every line of `evaluation.py::kfold_splits`, `data.py::knn_mask`, `data.py::knn_kernel`,
   `similarity.py::jaccard` and the CTD loaders in `scripts/02_build_features.py` and explain
   what each line does to the shapes and values involved.

---

## 1. Motivation: why this unit matters for *this* project

Everything in the project is, at bottom, a NumPy array or a pandas table:

```
 data/raw/ctd/*.tsv.gz  ──pandas.read_csv(chunksize=…)──►  gene sets (Python dicts of sets)
 data/raw/benchmarks/*.mat ──scipy.io.loadmat──►  A (593 x 313), similarity matrices
 data/raw/ontology/mondo.obo ──plain Python parsing──►  parent dictionary of the disease DAG
                                     │
                         scripts/02_build_features.py
                                     ▼
                   data/processed/Fdataset.npz  (NumPy arrays only)
                                     │
               src/drepo/data.py::load  →  DDData(A, drug_views, disease_views, …)
                                     │
        evaluation.py (folds = index arrays)  →  methods.py (torch tensors on the GPU)
```

Here is a concrete example of why you need *precise* control over arrays. The 5-fold
cross-validation in `src/drepo/evaluation.py` starts like this:

```py
# src/drepo/evaluation.py :: kfold_splits
pos = np.flatnonzero(A.ravel() > 0)
neg = np.flatnonzero(A.ravel() == 0)
```

`A` is the 593 × 313 drug–disease matrix of Fdataset. The code does not keep a list of
`(drug, disease)` pairs; it flattens the matrix into one long vector of 593·313 = 185,609 cells
and stores *positions* in that vector. Position `k` means drug `k // 313` and disease `k % 313`.
If you do not understand that one convention, you cannot check that test links are really
hidden from training, you cannot read the per-disease bookkeeping in
`methods.py::MVHGATMethod.fit_predict` (`hide |= cold[pos % n_d]`), and you cannot debug a
leak — the single most dangerous bug in link prediction (Unit E1).

A second example: `scripts/02_build_features.py` has to read
`CTD_chemicals_diseases.tsv.gz`. On disk it is 162 MB, but decompressed it is about
**913 MB of text with 9.9 million rows** (we measured this for the copy in `data/raw/ctd/`).
Loaded naively into pandas with every column as a Python string it would need several gigabytes
of RAM — more than many laptops have free while a GPU experiment is running. Reading it in
chunks of one million rows, keeping only the rows about *our* drugs, is what makes the pipeline
run at all.

The rest of the unit builds up exactly the tools those two examples need, and then walks
through the project code line by line (§6).

---

## 2. Your working environment: interpreters, packages and virtual environments

### 2.1 What "Python" actually is on your machine

When you type `python` in a terminal, the operating system looks through the folders listed in
the `PATH` environment variable and runs the first `python.exe` it finds. That executable is the
**interpreter**: the program that reads your `.py` files and executes them. An interpreter comes
with

* the **standard library** (modules such as `json`, `gzip`, `pathlib`, `argparse`, `random`), and
* a folder called **`site-packages`**, where third-party **packages** (NumPy, pandas, PyTorch,
  RDKit …) are installed by `pip`.

A **package** is an installable bundle of Python code (sometimes with compiled C/C++/CUDA
code inside, as for NumPy and PyTorch). **`pip`** is the package installer; it downloads
packages (usually as pre-compiled "wheels") from the Python Package Index (PyPI) and unpacks
them into `site-packages`.

Your laptop can have several interpreters at once. On this machine there is an Anaconda Python
in `C:\Users\Abhineet Anand\anaconda3` *and* the project's own environment in
`DrugRepositioning\.venv`. Which one runs depends on `PATH` and on which environment is
**activated**. "It works in Jupyter but not in the terminal" is almost always two different
interpreters with two different `site-packages`.

### 2.2 Virtual environments

A **virtual environment** ("venv") is a folder that behaves like a private copy of an
interpreter with its *own* `site-packages`. Packages installed inside it do not affect any other
project, and vice versa. Why this matters for research:

* **Isolation.** The project needs PyTorch built for CUDA 12.6; another project may need an
  older PyTorch. Each gets its own venv.
* **Reproducibility.** A reviewer (or you, in six months) can rebuild the exact environment from
  `requirements.txt` and reproduce your numbers.
* **Safety.** You can delete a broken venv and recreate it in minutes without touching the
  system Python.

The project's venv was created with the standard-library `venv` module. Its configuration file
`.venv\pyvenv.cfg` reads:

```text
home = C:\Users\Abhineet Anand\anaconda3
include-system-site-packages = false
version = 3.13.5
executable = C:\Users\Abhineet Anand\anaconda3\python.exe
command = C:\Users\Abhineet Anand\anaconda3\python.exe -m venv C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv
```

Line by line: `home` is the **base interpreter** the venv was built from (the Anaconda Python);
`include-system-site-packages = false` means the base interpreter's packages are *invisible*
inside the venv (full isolation); `version` is the Python version (3.13.5); `command` is the exact
command that created it. Inside `.venv` you will find `Scripts\python.exe` (a small launcher that
uses the base interpreter but points it at the venv's library folder), `Scripts\pip.exe`,
`Scripts\Activate.ps1`, and `Lib\site-packages\` (where NumPy, torch, … live).

**The life cycle of a venv (Windows PowerShell):**

```powershell
# 1. create (once)
python -m venv .venv

# 2. activate (every new terminal). Afterwards the prompt shows (.venv)
.venv\Scripts\Activate.ps1
#    if PowerShell refuses ("running scripts is disabled"), run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. install packages INTO the venv. 'python -m pip' guarantees the pip
#    of the interpreter you are actually using.
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt

# 4. record exact versions for the paper / reproducibility
python -m pip freeze > requirements-lock.txt

# 5. leave the venv
deactivate
```

Two details deserve emphasis.

* **Activation is a convenience, not magic.** It only puts `.venv\Scripts` at the front of
  `PATH` for that terminal. You can skip activation completely by calling the venv's interpreter
  by its full path, which is exactly what `scripts/run_all.ps1` does:
  `$py = ".venv\Scripts\python.exe"; & $py scripts/03_evaluate.py ...`.
* **Why `--index-url` for PyTorch?** PyPI's default `torch` wheel for Windows is the CPU build.
  The CUDA builds are hosted on PyTorch's own index, so the project's `requirements.txt` tells
  you to install torch from `https://download.pytorch.org/whl/cu126` *first*, and only then the
  rest.

**`requirements.txt` versus a lock file.** The project's `requirements.txt` lists *names*
(`numpy`, `pandas`, `torch>=2.4` …) without pinning exact versions; that keeps installation
flexible. For a paper you should additionally save `pip freeze` output, which pins every package
to an exact version (`numpy==2.5.3`, …), so that the environment can be rebuilt bit-for-bit.

**conda vs venv.** Anaconda's `conda` can also create environments and can install non-Python
dependencies (C libraries, CUDA toolkits). `venv` + `pip` is lighter and is what this project
uses. Do not mix them inside one environment unless you know why: `conda install` and
`pip install` keep separate records and can overwrite each other's files.

You can always ask Python which interpreter is running and whether it is inside a venv:

```python
import sys
print("interpreter :", sys.executable)
print("version     :", sys.version.split()[0])
print("in a venv?  :", sys.prefix != sys.base_prefix)   # True inside a virtual environment
print("venv folder :", sys.prefix)
print("base Python :", sys.base_prefix)
import numpy, pandas, scipy, sklearn, torch
for m in (numpy, pandas, scipy, sklearn, torch):
    print(f"{m.__name__:8s} {m.__version__}")
```

Output:

```text
interpreter : C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv\Scripts\python.exe
version     : 3.13.5
in a venv?  : True
venv folder : C:\Users\Abhineet Anand\Desktop\DrugRepositioning\.venv
base Python : C:\Users\Abhineet Anand\anaconda3
numpy    2.5.3
pandas   3.0.6
scipy    1.18.1
sklearn  1.9.1
torch    2.14.1+cu126
```

`sys.prefix` is the environment's root folder and `sys.base_prefix` is the base installation;
they differ exactly when you are inside a venv. Put the first two lines at the top of any script
that misbehaves mysteriously — nine times out of ten the culprit is the wrong interpreter.

### 2.3 Scripts, modules and packages (and how `import` finds them)

* A **script** is a `.py` file you run directly: `python scripts/03_evaluate.py --dataset F`.
* A **module** is any `.py` file that other code imports: `src/drepo/data.py` is the module
  `drepo.data`.
* A **package** (in the import sense) is a folder of modules with an `__init__.py` file:
  `src/drepo/` is the package `drepo`.

When Python executes `import drepo`, it searches the folders in the list **`sys.path`** in
order. That list starts with the folder of the script being run, then the standard library, then
`site-packages`. The project keeps its library in `src/drepo/`, which is *not* on that list
when you run `scripts/03_evaluate.py`. Every script therefore starts with:

```py
# scripts/03_evaluate.py (top)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                                     # noqa: E402
```

Read it from the inside out: `__file__` is the path of the running script;
`.resolve()` makes it absolute; `.parents[0]` is `scripts/`, `.parents[1]` is the project root;
`/ "src"` appends a folder name (the `/` operator is overloaded by `pathlib.Path` to join paths
portably on Windows and Linux); `sys.path.insert(0, ...)` puts it *first* on the search list.
The `# noqa: E402` comment silences a linter warning about "import not at top of file". (The
alternative, installing the project as a package with `pip install -e .`, needs a
`pyproject.toml`; the `sys.path` trick keeps the repository simple.)

**`if __name__ == "__main__":`.** Every module has a variable `__name__`. When a file is run as
a script, Python sets it to the string `"__main__"`; when the file is imported, `__name__` is the
module's name (e.g. `"drepo.data"`). So

```py
# scripts/01_download_data.py (bottom)
if __name__ == "__main__":
    for rel, url in FILES.items():
        download(url, RAW / rel)
```

runs the downloads only when you execute the script, not when another file merely imports
`download` from it.

**Command-line arguments with `argparse`.** The evaluation script exposes options such as
`--dataset F --protocol cv5 --repeats 5`:

```py
# scripts/03_evaluate.py
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="F")
ap.add_argument("--protocol", default="cv5", help="cv5 | cv10 | lodo")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--lodo-subset", type=int, default=None, help="...")
args = ap.parse_args()
```

`parse_args()` reads `sys.argv` (the words typed after the script name), converts each value
with `type=` (strings by default), fills in defaults, and returns an object whose attributes are
the option names with dashes turned into underscores (`args.lodo_subset`). `--help` is generated
for free. You can test a parser without a terminal by passing a list:

```python
import argparse
ap = argparse.ArgumentParser(prog="03_evaluate.py")
ap.add_argument("--dataset", default="F")
ap.add_argument("--protocol", default="cv5", help="cv5 | cv10 | lodo")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--lodo-subset", type=int, default=None)
args = ap.parse_args(["--dataset", "C", "--repeats", "5", "--lodo-subset", "60"])
print(args)
print(type(args.repeats), args.lodo_subset + 1)
print(int(args.protocol[2:]))   # how runner.py turns 'cv5' into k = 5
```

Output:

```text
Namespace(dataset='C', protocol='cv5', repeats=5, lodo_subset=60)
<class 'int'> 61
5
```

**`pathlib` for paths.** `src/drepo/paths.py` defines all folders relative to the file's own
location — `ROOT = Path(__file__).resolve().parents[2]`, `RAW = ROOT / "data" / "raw"` — so
the code works no matter which directory you launch it from. Never hard-code
`"C:\\Users\\..."` in library code.

---

## 3. The Python you will meet in the project code

This section is a compact tour of the core-language features that appear in `src/` and
`scripts/`. Each item names where it is used.

### 3.1 Built-in containers

| Type | Literal | Mutable? | Ordered? | Typical project use |
|---|---|---|---|---|
| `list` | `[1, 2, 3]` | yes | yes | lists of SMILES strings, of fold results |
| `tuple` | `(593, 313)` | no | yes | array shapes, `(dst_type, src_type)` relation pairs |
| `dict` | `{"F": "Fdataset"}` | yes | insertion order | `DATASETS`, `FILES`, `METHODS` registries |
| `set` | `{"TP53", "EGFR"}` | yes | no | a drug's gene set (fast membership, union `\|`, intersection `&`) |

Sets are what make the gene-profile code fast: `g in wanted` is a hash lookup (constant time on
average) instead of a scan through a list.

**`collections.defaultdict`** creates missing keys on first access:

```python
from collections import defaultdict
genes = defaultdict(set)                     # missing key -> empty set
for chem, gene in [("D001", "TP53"), ("D002", "EGFR"), ("D001", "BRCA1"), ("D001", "TP53")]:
    genes[chem].add(gene)                    # no KeyError on first sight of a chemical
print(dict(genes))
print(sorted(genes["D001"] & genes["D002"]), sorted(genes["D001"] | genes["D002"]))
```

Output:

```text
{'D001': {'TP53', 'BRCA1'}, 'D002': {'EGFR'}}
[] ['BRCA1', 'EGFR', 'TP53']
```

(Set printing order is arbitrary, so robust code sorts before printing, as the second line does.)
This is exactly the pattern of `ctd_drug_genes` and `ctd_disease_genes` in
`scripts/02_build_features.py`.

### 3.2 Comprehensions, `zip`, `enumerate`

A **list comprehension** `[f(x) for x in xs if cond(x)]` builds a list in one expression; there
are also dict (`{k: v for ...}`), set and generator versions. `zip(a, b)` walks two sequences in
lock-step; `enumerate(xs, 1)` yields `(1, x0), (2, x1), …`. In `02_build_features.py`:

```py
by_cid = {c: i for c, i in zip(ch.PubChemCID, ch.ChemicalID) if c}   # dict comprehension
for k, db in enumerate(todo, 1):          # k counts from 1 for progress messages
    ...
```

### 3.3 Functions, default arguments, `lambda`, type hints

`def knn_mask(S: np.ndarray, k: int, symmetric: bool = True) -> np.ndarray:` declares a
function with a default argument and **type hints** (the `: np.ndarray` annotations). Hints are
documentation for humans and tools; Python does not enforce them at run time. The line
`from __future__ import annotations` at the top of most project files makes annotations lazy
strings, which allows modern syntax like `str | None` and avoids import-order problems.

A `lambda` is a one-line anonymous function. `MBiRW` uses
`logistic = lambda S: 1 / (1 + np.exp(-15 * S + np.log(9999)))`.

### 3.4 Generators and `yield`

A function containing `yield` is a **generator**: calling it returns an iterator that produces
values lazily, one per `for` iteration, resuming where it left off. `kfold_splits` is a
generator; `run_kfold` consumes it with `for f, test_pos, test_neg in kfold_splits(A, k, seed + r):`.
Generators keep memory flat: fold 3 is not computed until fold 2 has been used.

```python
def countdown(n):
    while n > 0:
        yield n          # pause here, hand n to the caller
        n -= 1           # resume here on the next iteration
g = countdown(3)
print(next(g), next(g))  # 3 2
print(list(g))           # the remaining values
```

Output:

```text
3 2
[1]
```

### 3.5 Classes and `@dataclass`

`DDData` in `src/drepo/data.py` is a **dataclass**: the decorator `@dataclass` writes the
`__init__`, `__repr__` and `__eq__` methods for you from the annotated fields. `@property`
turns a method into a computed attribute, so `data.n_drugs` (no parentheses) returns
`self.A.shape[0]`. `MVHGATConfig` is also a dataclass; every field is a hyper-parameter with a
default value, and `MVHGATMethod(**overrides)` changes some of them with `setattr`.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Tiny:
    name: str
    A: np.ndarray
    @property
    def n_drugs(self):
        return self.A.shape[0]

d = Tiny("toy", np.zeros((4, 3)))
print(d.n_drugs, d.A.shape, d.name)
```

Output:

```text
4 (4, 3) toy
```

### 3.6 f-strings and format specifications

`f"{m['AUC']:.4f}"` formats a float with 4 decimals; `f"{0.0104:.2%}"` prints `1.04%`;
`f"{name:10s}"` pads to width 10; `f"{n:,}"` inserts thousands separators. The evaluation and
build scripts use these everywhere for logging.

### 3.7 Exceptions and context managers

`try: ... except requests.RequestException: ...` in `_get` retries failed downloads.
`with requests.get(...) as r:` and `with open(tmp, "wb") as fh:` are **context managers**:
the resource is closed automatically when the block ends, even after an error.
`with np.errstate(invalid="ignore", divide="ignore"):` (in `similarity.py::jaccard`) is a
NumPy context manager that silences the warnings for `0/0` inside the block only.

---

## 4. NumPy from first principles

### 4.1 Why NumPy exists

A Python `list` of numbers is a list of *pointers* to separate Python objects scattered in
memory; every `+` in a Python loop has to look up types, allocate a new object and follow
pointers. A NumPy **ndarray** ("n-dimensional array") is instead a single contiguous block of
raw machine numbers of one type, plus a small header describing how to interpret it. Operations
on whole arrays (`A + B`, `A @ B`, `A.sum()`) run as compiled loops in C (and call optimised
BLAS libraries for matrix products), typically 10–1000× faster than the equivalent Python loop.
Writing code as whole-array operations instead of explicit loops is called **vectorisation**.

Vectorisation is not only about speed. It is also *shorter* and, once you can read it, *clearer*:
`similarity.py::jaccard` computes all 593² pairwise Jaccard similarities in four lines.

```python
import time
import numpy as np

rng = np.random.default_rng(0)
G = (rng.random((300, 2000)) < 0.02).astype(np.float64)   # 300 entities x 2000 genes, ~2% ones

# (a) pure-Python version with sets
sets = [set(np.flatnonzero(row)) for row in G]
t0 = time.perf_counter()
S_loop = np.zeros((300, 300))
for i in range(300):
    for j in range(300):
        u = len(sets[i] | sets[j])
        S_loop[i, j] = len(sets[i] & sets[j]) / u if u else 0.0
t_loop = time.perf_counter() - t0

# (b) vectorised version (the idea used in similarity.py::jaccard)
def jaccard_vec(G):
    inter = G @ G.T                                  # shared genes for every pair
    size = G.sum(1)
    union = size[:, None] + size[None, :] - inter    # |a| + |b| - |a & b|
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

times = []
for _ in range(5):                                   # best of 5: timings are noisy
    t0 = time.perf_counter()
    S_vec = jaccard_vec(G)
    times.append(time.perf_counter() - t0)
t_vec = min(times)

print("same result:", np.allclose(S_loop, S_vec))
print(f"loop {t_loop:.3f}s  vectorised {t_vec:.4f}s  speed-up ~{t_loop / t_vec:.0f}x")
```

Output:

```text
same result: True
loop 0.345s  vectorised 0.0030s  speed-up ~115x
```

(Your timings will differ from run to run and machine to machine; the *ratio* is what matters.)

### 4.2 Anatomy of an ndarray

Every array carries:

| Attribute | Meaning | For Fdataset's `A` |
|---|---|---|
| `shape` | tuple of axis lengths | `(593, 313)` |
| `ndim` | number of axes = `len(shape)` | `2` |
| `size` | number of elements = product of shape | `185609` |
| `dtype` | element type | `float32` |
| `itemsize` | bytes per element | `4` |
| `nbytes` | `size * itemsize` | `742436` (≈ 0.74 MB) |
| `strides` | bytes to step along each axis | `(1252, 4)` |

**Worked example (by hand).** `A` is 593 × 313 `float32`. `size = 593 × 313 = 185,609`;
`nbytes = 185,609 × 4 = 742,436` bytes. In the default **row-major** ("C") layout, the 313
entries of row 0 come first, then the 313 entries of row 1, and so on. Moving one step along
axis 1 (next disease, same drug) moves 4 bytes; moving one step along axis 0 (next drug, same
disease) skips a whole row: 313 × 4 = 1,252 bytes. Hence `strides = (1252, 4)`. The address of
element `(i, j)` is `base + 1252·i + 4·j`.

Axes are numbered from 0. For a 2-D array, **axis 0 runs down the rows** (it indexes drugs in
`A`) and **axis 1 runs across the columns** (diseases). The project's similarity stack
`drug_views` has shape `(3, 593, 593)`: axis 0 = which view, axis 1 = drug *i*, axis 2 = drug *j*.

```python
import sys
sys.path.insert(0, "src")                 # run from the project root
from drepo import data as D

d = D.load("F")
A = d.A
print(A.shape, A.ndim, A.size, A.dtype, A.itemsize, A.nbytes, A.strides)
print(d.drug_views.shape, d.drug_views.dtype, f"{d.drug_views.nbytes / 1e6:.1f} MB")
print(d.drug_ids[:3], d.disease_ids[:3])
print("known links:", int(A.sum()), f"density {A.mean():.2%}")
```

Output:

```text
(593, 313) 2 185609 float32 4 742436 (1252, 4)
(3, 593, 593) float64 8.4 MB
['DB00007' 'DB00010' 'DB00014'] ['102100' '102300' '102400']
known links: 1933 density 1.04%
```

### 4.3 Creating arrays

```python
import numpy as np
print(np.array([[1, 0, 1], [0, 1, 0]]))          # from nested lists
print(np.zeros((2, 3), dtype=bool))               # all False
print(np.full((2, 2), np.nan))                    # what morgan_tanimoto starts from
print(np.eye(3))                                  # identity matrix
print(np.arange(0, 10, 3), np.linspace(0, 1, 5))  # ranges
print(np.repeat(np.arange(3), 2), np.tile([7, 8], 3))
```

Output:

```text
[[1 0 1]
 [0 1 0]]
[[False False False]
 [False False False]]
[[nan nan]
 [nan nan]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[0 3 6 9] [0.   0.25 0.5  0.75 1.  ]
[0 0 1 1 2 2] [7 8 7 8 7 8]
```

`np.repeat(np.arange(n), k)` — "each row number, k times" — is the trick `knn_mask` uses to pair
every row with its k neighbour columns (§6.2).

### 4.4 dtypes and casting

The **dtype** fixes how the bytes are interpreted. The ones you meet in the project:

* `float64` (8 bytes, ~16 significant digits) — NumPy's default; similarity matrices.
* `float32` (4 bytes, ~7 digits) — the association matrix and *all* PyTorch tensors, because GPUs
  are fastest in 32-bit and it halves memory. `load()` does `z["A"].astype(np.float32)`.
* `bool` (1 byte) — masks such as `knn_mask`'s output and `neg_mask`.
* `int64` — index arrays returned by `np.flatnonzero`, `np.argsort`, `np.argpartition`.
* `<U7` — fixed-width Unicode strings of at most 7 characters, e.g. DrugBank IDs `'DB00007'`.

`astype` makes a converted **copy**. Comparisons produce `bool` arrays, and `bool` arrays behave
as 0/1 in arithmetic (`mask.sum()` counts `True`s, `mask.mean()` gives the fraction). `NaN`
("not a number") exists only for floating types, so an integer array cannot hold a missing value.

```python
import numpy as np
x = np.array([0.0, 0.3, 0.8])
m = x > 0.25
print(m, m.dtype, m.sum(), m.mean())
print(m.astype(np.float32), (x * m))                 # bools act as 0/1
print(np.array([1, 2, 3]) / 2, np.array([1, 2, 3]) // 2)   # true vs floor division
print(np.float32(1) / 3, np.float64(1) / 3)           # 7 vs 16 significant digits
```

Output:

```text
[False  True  True] bool 2 0.6666666666666666
[0. 1. 1.] [0.  0.3 0.8]
[0.5 1.  1.5] [0 1 1]
0.33333334 0.3333333333333333
```

### 4.5 Indexing I — basic indexing and slicing (views)

`A[i, j]` is a single element; `A[i]` or `A[i, :]` is row `i`; `A[:, j]` is column `j`;
`A[2:5, ::2]` takes rows 2,3,4 and every second column. Negative indices count from the end
(`A[-1]` is the last row). Slices use the half-open convention `start:stop:step`, stop excluded.

**Basic indexing returns a view**: a new header pointing into the *same* memory. Writing through
a view changes the original. This is efficient (no copying), but a classic source of bugs:

```python
import numpy as np
A = np.zeros((3, 4))
col = A[:, 1]          # a view of column 1
col[:] = 7             # writes into A!
print(A)
print("shares memory:", np.shares_memory(A, col))
B = A[:, 1].copy()     # an independent copy
B[:] = -1
print(A[:, 1])         # unchanged by B
```

Output:

```text
[[0. 7. 0. 0.]
 [0. 7. 0. 0.]
 [0. 7. 0. 0.]]
shares memory: True
[7. 7. 7.]
```

This is why `run_lodo` begins every disease with `A_tr = A.copy()` before `A_tr[:, j] = 0`:
without `.copy()` it would zero the *real* data matrix and every later disease would be evaluated
on corrupted labels.

### 4.6 Indexing II — integer ("fancy") indexing (copies)

Indexing with an **array of integers** selects arbitrary elements and always returns a **copy**.
With one index array you pick rows; with two index arrays of the same shape you pick *pairs*:
`A[rows, cols]` returns `A[rows[0], cols[0]], A[rows[1], cols[1]], …`. This is *not* the
sub-matrix; for the sub-matrix use `np.ix_`.

```python
import numpy as np
A = np.arange(12).reshape(3, 4)
print(A)
print(A[[0, 2]])                      # rows 0 and 2
print(A[[0, 2], [1, 3]])              # the PAIRS (0,1) and (2,3)  -> 2 elements
print(A[np.ix_([0, 2], [1, 3])])      # the 2x2 SUB-MATRIX rows {0,2} x cols {1,3}
M = np.zeros((3, 4), dtype=bool)
M[[0, 1, 1], [3, 0, 2]] = True        # fancy ASSIGNMENT writes into M (used by knn_mask)
print(M.astype(int))
```

Output:

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
[[ 0  1  2  3]
 [ 8  9 10 11]]
[ 1 11]
[[ 1  3]
 [ 9 11]]
[[0 0 0 1]
 [1 0 1 0]
 [0 0 0 0]]
```

Note the asymmetry: fancy indexing on the *right* of `=` copies, but fancy indexing on the *left*
of `=` writes into the original array. `knn_mask` uses the latter (`M[rows[keep], idx.ravel()[keep]] = True`).

### 4.7 Indexing III — boolean masks

A **boolean mask** is a `bool` array of the same shape as the data. `x[mask]` returns the
selected elements as a 1-D copy; `x[mask] = v` assigns to them. Masks are combined with the
element-wise operators `&` (and), `|` (or), `~` (not), `^` (xor). Because these operators bind
more tightly than comparisons, **always parenthesise the comparisons**: `(A == 0) & neg_mask`,
not `A == 0 & neg_mask`.

`np.flatnonzero(mask)` returns the flat positions of the `True`s; `np.nonzero(mask)` returns one
index array per axis (row indices, column indices); `np.where(cond, a, b)` picks element-wise.

```python
import numpy as np
A = np.array([[1, 0, 0],
              [0, 0, 1]], dtype=np.float32)
print(np.flatnonzero(A.ravel() > 0))      # flat positions of the 1s
print(np.nonzero(A))                      # (row indices, column indices)
neg_mask = (A == 0)
neg_mask[1, 0] = False                    # pretend cell (1,0) is a TEST cell
print(neg_mask)
print(np.flatnonzero(neg_mask & (A == 0)))
print(np.where(A > 0, "link", "-"))
```

Output:

```text
[0 5]
(array([0, 1]), array([0, 2]))
[[False  True  True]
 [False  True False]]
[1 2 4]
[['link' '-' '-']
 ['-' '-' 'link']]
```

In the project, masks are *data*: `neg_mask` says which unknown cells the model may use as
training negatives; `knn_mask` returns a boolean adjacency matrix; `graphs[...]` in the model are
boolean tensors that `DenseGAT` uses to block attention to non-neighbours.

### 4.8 Shapes: `reshape`, `ravel`, and the flat index formula

`reshape(new_shape)` reinterprets the same elements with a different shape (the product of the
dimensions must stay the same; one dimension may be `-1`, meaning "infer it"). `ravel()` flattens
to 1-D and returns a **view when it can** (always, for a contiguous array); `flatten()` always
copies. Both read the elements in **row-major (C) order** by default: the last index changes
fastest.

**Theorem (flat index of a cell).** For an array of shape $(n, m)$ in C order, element $(i, j)$
sits at flat position
$$k = i\cdot m + j, \qquad 0 \le i < n,\ 0 \le j < m,$$
and conversely $i = \lfloor k / m \rfloor$, $j = k \bmod m$.

*Proof.* In C order rows are stored one after the other. Rows $0,\dots,i-1$ occupy $i\cdot m$
positions, so row $i$ starts at $i\cdot m$, and element $j$ of that row is $j$ further on. For the
converse, $k = i m + j$ with $0 \le j < m$ is exactly the division of $k$ by $m$ with quotient
$i$ and remainder $j$; by uniqueness of Euclidean division these are $\lfloor k/m\rfloor$ and
$k \bmod m$. $\square$

For a 3-D array of shape $(a, b, c)$ the same reasoning gives $k = (i\,b + j)\,c + l$.

**Worked example (by hand).** For Fdataset, $m = 313$ diseases.

* Drug 2, disease 5 → $k = 2\cdot 313 + 5 = 631$.
* Flat position $k = 1000$ → $i = \lfloor 1000/313\rfloor = 3$ (since $3\cdot313 = 939$),
  $j = 1000 - 939 = 61$. So `A.ravel()[1000]` is `A[3, 61]`: drug 3, disease 61.
* The last cell, $(592, 312)$, is $k = 592\cdot313 + 312 = 185{,}608 = 185{,}609 - 1$. ✓

NumPy has these conversions built in: `np.ravel_multi_index((i, j), shape)` and
`np.unravel_index(k, shape)`; plain integer arithmetic `divmod(k, m)` works too, and so does
`k // m`, `k % m` on whole arrays (this is what `fit_predict` does with `pos % n_d`).

```python
import numpy as np
shape = (593, 313)
print(np.ravel_multi_index((2, 5), shape))          # 631
print(np.unravel_index(1000, shape))                # (3, 61); NumPy 2 prints np.int64(...)
print(divmod(1000, 313))                            # (3, 61)
k = np.array([0, 312, 313, 1000, 185608])
print(k // 313, k % 313)                            # drug and disease of each flat index

A = np.arange(12).reshape(3, 4)
print(A.ravel())                    # row-major: 0..11
print(A.T.ravel())                  # the TRANSPOSE flattens in a different order!
print(A.ravel(order="F"))           # column-major (Fortran) order = same as A.T.ravel()
r = A.ravel(); r[0] = 99            # ravel of a contiguous array is a view ...
print(A[0, 0])                      # ... so A changed
```

Output:

```text
631
(np.int64(3), np.int64(61))
(3, 61)
[  0   0   1   3 592] [  0 312   0  61 312]
[ 0  1  2  3  4  5  6  7  8  9 10 11]
[ 0  4  8  1  5  9  2  6 10  3  7 11]
[ 0  4  8  1  5  9  2  6 10  3  7 11]
99
```

**The flat-index round trip used in cross-validation.** `run_kfold` hides the test positives
like this:

```py
A_tr = A.copy().ravel()      # flat copy: 185,609 cells
A_tr[test_pos] = 0           # test_pos are flat indices
A_tr = A_tr.reshape(shape)   # back to 593 x 313
```

Because `ravel` and `reshape` use the *same* (C) order, position `k` goes back to cell
`(k // 313, k % 313)` — exactly the cell it came from. Mixing orders (e.g. flattening `A.T`) would
silently hide the wrong cells. Note also that `A.T` is a *view* with swapped strides `(4, 1252)`,
so `A.T.ravel()` must copy.

### 4.9 Broadcasting

**Broadcasting** is NumPy's rule for combining arrays of different shapes element-wise without
copying data. It is what lets `jaccard` write `size[:, None] + size[None, :]` to get an $n\times n$
matrix from a length-$n$ vector.

**The rules.** To combine arrays $x$ and $y$:

1. If they have different numbers of dimensions, prepend 1s to the shape of the shorter one.
2. Compare the shapes axis by axis. Two lengths are compatible if they are **equal**, or **one
   of them is 1**.
3. Along an axis where one length is 1, that array is (virtually) repeated to match the other.
   The result's length on each axis is the maximum of the two.
4. If some axis has two different lengths, neither 1, raise an error.

`x[:, None]` (equivalently `x[:, np.newaxis]` or `x.reshape(-1, 1)`) turns a length-$n$ vector
into an $n\times 1$ **column**; `x[None, :]` makes a $1 \times n$ **row**.

**Worked examples (by hand).**

| Expression | Shapes | Aligned | Result |
|---|---|---|---|
| `S * d[:, None]` | (593,593) and (593,1) | (593,593) vs (593,1) | (593,593): row $i$ scaled by $d_i$ |
| `S * d[None, :]` | (593,593) and (1,593) | — | (593,593): column $j$ scaled by $d_j$ |
| `S * d` | (593,593) and (593,) | (593,593) vs (1,593) | same as `d[None, :]` — **columns** |
| `size[:, None] + size[None, :]` | (n,1) and (1,n) | — | (n,n) with entry $s_i + s_j$ |
| `W / W.sum(1, keepdims=True)` | (n,n) and (n,1) | — | each row divided by its own sum |
| `W / W.sum(1)` | (n,n) and (n,) | (n,n) vs (1,n) | divides **column** $j$ by row-sum $j$ — a bug! |
| `P (6,593,313) * w[:, None, None]` | (6,593,313), (6,1,1) | — | slice $v$ scaled by $w_v$ |
| `(3,4) + (3,)` | — | (3,4) vs (1,3) | **error**: 4 vs 3 |

The rows marked "a bug!" and "error" are the most common broadcasting mistakes: a 1-D vector is
treated as a *row*, so `S * d` scales columns, not rows. `keepdims=True` keeps the reduced axis
with length 1 so that the result broadcasts the way you mean.

```python
import numpy as np
S = np.array([[1.0, 0.5, 0.0],
              [0.5, 1.0, 0.2],
              [0.0, 0.2, 1.0]])
d = np.array([1.0, 2.0, 3.0])
print(S * d[:, None])         # rows scaled by 1, 2, 3
print(S * d)                  # columns scaled by 1, 2, 3
size = np.array([2, 3, 1])
print(size[:, None] + size[None, :])
W = np.array([[0.0, 2.0, 2.0],
              [1.0, 0.0, 3.0],
              [5.0, 5.0, 0.0]])
print(W / W.sum(1, keepdims=True))   # rows sum to 1
print((W / W.sum(1, keepdims=True)).sum(1))
try:
    np.ones((3, 4)) + np.ones(3)
except ValueError as e:
    print("ValueError:", e)
```

Output:

```text
[[1.  0.5 0. ]
 [1.  2.  0.4]
 [0.  0.6 3. ]]
[[1.  1.  0. ]
 [0.5 2.  0.6]
 [0.  0.4 3. ]]
[[4 5 3]
 [5 6 4]
 [3 4 2]]
[[0.   0.5  0.5 ]
 [0.25 0.   0.75]
 [0.5  0.5  0.  ]]
[1. 1. 1.]
ValueError: operands could not be broadcast together with shapes (3,4) (3,) 
```

**Memory note.** Broadcasting never materialises the repeated copies of the *inputs*, but the
*result* is a full array: `size[:, None] + size[None, :]` for 17,626 genes would allocate a
17,626² float64 matrix = 2.5 GB. Always estimate the output size of a broadcast.

### 4.10 Reductions and axes

A **reduction** collapses an axis: `sum`, `mean`, `max`, `min`, `argmax`, `any`, `all`, `std`.
Mnemonic: **the axis you name is the axis that disappears.** For `A` (drugs × diseases):

* `A.sum(axis=1)` → shape `(593,)`: for each drug, the number of known diseases (its **degree**).
* `A.sum(axis=0)` → shape `(313,)`: for each disease, the number of known drugs.
* `A.sum()` → a scalar: total links (1,933).

`keepdims=True` keeps the reduced axis as length 1. `np.nansum`, `np.nanmean` ignore NaNs.
`mask.any(1)` asks "does each row have at least one True?" — `DenseGAT.forward` returns
`mask.any(1)` to flag nodes that have at least one neighbour in a relation.

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
A = D.load("F").A
deg_drug, deg_dis = A.sum(1), A.sum(0)
print(deg_drug.shape, deg_dis.shape, A.sum())
print("drug degree  min/mean/max:", deg_drug.min(), round(deg_drug.mean(), 2), deg_drug.max())
print("disease degree min/mean/max:", deg_dis.min(), round(deg_dis.mean(), 2), deg_dis.max())
print("diseases with exactly one known drug:", int((deg_dis == 1).sum()))
print("keepdims shape:", A.sum(1, keepdims=True).shape)
print("most-treated disease index:", int(deg_dis.argmax()))
```

Output:

```text
(593,) (313,) 1933.0
drug degree  min/mean/max: 1.0 3.26 22.0
disease degree min/mean/max: 1.0 6.18 84.0
diseases with exactly one known drug: 97
keepdims shape: (593, 1)
most-treated disease index: 108
```

### 4.11 Sorting and selection: `sort`, `argsort`, `argpartition`

* `np.sort(x)` returns the sorted values (ascending).
* `np.argsort(x)` returns the **indices** that would sort `x`: `x[np.argsort(x)]` is sorted. For
  descending order use `np.argsort(-x)` (or `[::-1]`, which reverses ties too). `kind="stable"`
  keeps equal elements in their original order.
* `np.argpartition(x, kth)` returns indices arranged so that **the element at position `kth` is
  the one that would be there in a full sort, every element before it is ≤ it, and every element
  after it is ≥ it** — but the elements on each side are in *no particular order*.

So `np.argpartition(x, kth)[:kth]` gives the indices of the `kth` smallest values (unordered),
and with `-x` the `kth` largest. The project uses both conventions:
`knn_mask` calls `np.argpartition(-work, k, axis=1)[:, :k]` and `topk_bipartite` calls
`np.argpartition(-B, k - 1, axis=1)[:, :k]`. Both are correct: with `kth=k-1` the k smallest
of `-B` occupy positions `0..k-1`; with `kth=k` the k+1 smallest occupy `0..k` and the first k
of those are still the k smallest.

**Why partition instead of sort?** A full sort of $n$ items costs $O(n \log n)$ comparisons.
Selecting the $k$ smallest needs only $O(n)$ on average, using a **selection algorithm**
(NumPy uses *introselect*: quickselect with a fallback that guarantees $O(n)$ worst case). The
idea of quickselect: pick a pivot, split the array into "< pivot" and "> pivot" (one linear pass),
then recurse into **only the side that contains position k** — the expected work is
$n + n/2 + n/4 + \dots \le 2n$. A sort must recurse into both sides. For one row of 593
similarities the difference is small; for every row of a 593 × 593 matrix (and inside a loop over
folds and hyper-parameter settings) it adds up, and it matters a lot for big $n$. It also states
intent: we need the *set* of the k nearest neighbours, not their order.

**Worked example (by hand).** Row of similarities `s = [0.20, 0.90, 0.10, 0.70, 0.50]`, k = 2.
The two largest are 0.90 (index 1) and 0.70 (index 3). `np.argpartition(-s, 2)` must put the
third-smallest value of `-s` (that is −0.50, index 4) at position 2, the two smaller ones
(−0.90, −0.70 → indices 1, 3, in some order) before it, and indices 0, 2 after it. So
`np.argpartition(-s, 2)[:2]` is `{1, 3}` in an unspecified order.

```python
import time
import numpy as np
s = np.array([0.20, 0.90, 0.10, 0.70, 0.50])
p = np.argpartition(-s, 2)
print(p, "-> top-2 set:", sorted(p[:2].tolist()))
print("argsort descending:", np.argsort(-s))

# top-k of EVERY row at once, then order them if you need a ranking
rng = np.random.default_rng(1)
S = rng.random((4, 8)).round(2)
k = 3
idx = np.argpartition(-S, k - 1, axis=1)[:, :k]               # unordered top-k per row
order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)    # sort only those k
topk_sorted = np.take_along_axis(idx, order, 1)
print(S[0], idx[0], topk_sorted[0])

# timing on a bigger problem (3000 x 3000) -- your numbers will differ
X = rng.random((3000, 3000))
t0 = time.perf_counter(); np.argsort(-X, axis=1)[:, :10]; t_sort = time.perf_counter() - t0
t0 = time.perf_counter(); np.argpartition(-X, 10, axis=1)[:, :10]; t_part = time.perf_counter() - t0
print(f"argsort {t_sort:.2f}s   argpartition {t_part:.2f}s")
```

Output:

```text
[1 3 4 0 2] -> top-2 set: [1, 3]
argsort descending: [1 3 4 0 2]
[0.51 0.95 0.14 0.95 0.31 0.42 0.83 0.41] [1 3 6] [1 3 6]
argsort 0.22s   argpartition 0.06s
```

**Ties.** If several neighbours have exactly the same similarity at the k-th position, which
ones are kept is arbitrary (it depends on the algorithm's internal pivoting). This is harmless
for similarity graphs, but if you need reproducible tie-breaking, sort with `kind="stable"` or add
a tiny deterministic jitter.

### 4.12 Random numbers and reproducible splits

Modern NumPy uses **Generator** objects: `rng = np.random.default_rng(seed)`. The same seed
always produces the same sequence, which is what makes the cross-validation folds reproducible
(`kfold_splits(A, k, seed)` uses `seed + r` for repeat `r`). Useful methods:

* `rng.shuffle(x)` — shuffles **in place** (returns `None`!).
* `rng.permutation(n)` — a shuffled copy of `0..n-1`.
* `rng.choice(n, size, replace=False)` — sampling without replacement (`runner.py` uses this to
  pick a LODO subset of diseases).
* `rng.random(shape)`, `rng.integers(low, high, size)`, `rng.normal(size=...)`.

The older global API (`np.random.seed(0)`, `np.random.rand(...)`) still exists;
`methods.py::set_seed` calls it together with `random.seed` and `torch.manual_seed`, because
PyTorch and Python's `random` module keep their *own* generators.

`np.array_split(x, k)` cuts `x` into k nearly equal consecutive pieces: if $n = qk + r$ with
$0 \le r < k$, the first $r$ pieces get $q + 1$ elements and the rest get $q$.
(`np.split` instead *requires* equal sizes and raises an error otherwise.)

**Worked example (by hand).** Fdataset has 1,933 positives. $1933 = 5 \cdot 386 + 3$, so the 5
positive folds have sizes 387, 387, 387, 386, 386. There are $185{,}609 - 1{,}933 = 183{,}676$
unknown cells; $183{,}676 = 5\cdot 36{,}735 + 1$, so the negative folds have
36,736, 36,735, 36,735, 36,735, 36,735 cells.

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import kfold_splits

A = D.load("F").A
sizes = [(len(p), len(n)) for _, p, n in kfold_splits(A, 5, seed=0)]
print(sizes)
f0 = next(kfold_splits(A, 5, seed=0))[1][:5]
f0_again = next(kfold_splits(A, 5, seed=0))[1][:5]
print(f0, f0_again, "(same seed -> same folds)")
allpos = np.concatenate([p for _, p, _ in kfold_splits(A, 5, seed=0)])
print("folds partition the positives:", len(allpos) == len(np.unique(allpos)) == int(A.sum()))
```

Output:

```text
[(387, 36736), (387, 36735), (387, 36735), (386, 36735), (386, 36735)]
[172704  30760 144287 104589  73228] [172704  30760 144287 104589  73228] (same seed -> same folds)
folds partition the positives: True
```

### 4.13 Missing values: NaN

`np.nan` is the IEEE "not a number" float. It **propagates** (`1 + nan = nan`) and it is **not
equal to itself** (`nan == nan` is `False`), so you must test with `np.isnan`. The similarity
functions use NaN deliberately to mean "this entity is not covered by this view" (e.g. a biologic
drug with no SMILES has a NaN row in `chem_ecfp`). `data.py::fill_missing` then decides how to
treat it: `np.nan_to_num(S, nan=0.0)` replaces NaN by 0 and `np.fill_diagonal(S, 1.0)` restores
self-similarity, so an uncovered entity becomes "similar only to itself" — no neighbours.

Two gotchas: `np.fill_diagonal` works **in place and returns `None`** (so `S = np.fill_diagonal(S, 1)`
destroys `S`); and `0/0` produces NaN with a `RuntimeWarning`, which `jaccard` silences locally
with `np.errstate`.

```python
import numpy as np
S = np.array([[1.0, np.nan, 0.4],
              [np.nan, np.nan, np.nan],
              [0.4, np.nan, 1.0]])
print(np.nan == np.nan, np.isnan(S).sum(1))
print(S.sum(1), np.nansum(S, 1))          # NaN propagates; nansum ignores it
F = np.nan_to_num(S.copy(), nan=0.0)
print(np.fill_diagonal(F, 1.0))           # returns None ...
print(F)                                  # ... but F was modified in place
with np.errstate(invalid="ignore"):
    print(np.array([0.0, 1.0]) / np.array([0.0, 2.0]))
```

Output:

```text
False [1 3 1]
[nan nan nan] [1.4 0.  1.4]
None
[[1.  0.  0.4]
 [0.  1.  0. ]
 [0.4 0.  1. ]]
[nan 0.5]
```

### 4.14 Saving and loading arrays

`np.save("x.npy", arr)` / `np.load("x.npy")` store one array in NumPy's binary format (exact,
fast, includes dtype and shape). `np.savez_compressed("f.npz", A=A, names=names, ...)` stores
several named arrays in one zip file; `np.load("f.npz")` returns a lazy dictionary-like object
(`z["A"]`, `z.files`). The project's `data/processed/Fdataset.npz` holds 11 arrays (`A`, ids,
names, the stacked views, the gene bridge, the gene vocabulary).

`load()` passes `allow_pickle=False`. Pickle is Python's general object serialiser, and loading a
pickle can execute arbitrary code — so NumPy refuses object arrays unless you opt in. Because the
project stores names as fixed-width Unicode (`<U71`) rather than Python objects, it never needs
pickling.

```python
import os, tempfile
import numpy as np
p = os.path.join(tempfile.gettempdir(), "demo_a1.npz")
A = (np.random.default_rng(0).random((4, 3)) < 0.3).astype(np.float32)
np.savez_compressed(p, A=A, drug_ids=np.array(["DB00007", "DB00010", "DB00014", "DB00035"]))
z = np.load(p, allow_pickle=False)
print(z.files, z["A"].dtype, z["drug_ids"].dtype)
print(np.array_equal(z["A"], A))
z.close(); os.remove(p)
```

Output:

```text
['A', 'drug_ids'] float32 <U7
True
```

### 4.15 Thinking in arrays: three patterns worth memorising

1. **Matrix products count co-occurrences.** If `G` is a 0/1 matrix (entities × genes), then
   `(G @ G.T)[i, j]` $= \sum_g G_{ig} G_{jg}$ = number of genes shared by entities $i$ and $j$.
   `(Gr @ Gd.T)[i, j]` counts genes shared by drug $i$ and disease $j$ — the number of
   drug→gene→disease paths (the "gene bridge" of `cosine_cross`). Unit A2 explains *why* this works.
2. **Row-normalise with `keepdims`.** `W / W.sum(1, keepdims=True)` turns weights into
   per-row averages; guard against zero rows first (`knn_kernel` sets `s[s == 0] = 1`).
3. **Pair rows with columns via `repeat`.** `rows = np.repeat(np.arange(n), k)` together with
   `idx.ravel()` from an `(n, k)` index matrix lists all `n*k` (row, column) pairs, ready for fancy
   assignment.

---

## 5. pandas: tables with labels

NumPy arrays are anonymous grids of one dtype. Real data files are **tables**: columns with
names and *different* types (IDs as strings, scores as floats), rows with identifiers, missing
entries. **pandas** provides two labelled structures on top of NumPy:

* a **Series**: a 1-D array with an **index** (row labels) and a name;
* a **DataFrame**: a dict-like collection of Series sharing one index — one Series per column.

In this project pandas is the *ingestion* layer (reading CTD, MedGen, PubChem results, writing
the readable `data/interim/*.csv` mapping tables), and NumPy is the *computation* layer. The
hand-off happens in `02_build_features.py`, where gene sets gathered with pandas become the 0/1
NumPy matrices of `gene_matrix`.

> **Version note.** The project uses pandas 3.0. Two changes from older tutorials matter:
> (i) text columns get the dedicated string dtype, displayed as `str`, instead of the generic
> `object`; (ii) **Copy-on-Write** is always on: any selection behaves like a copy, so
> "chained assignment" such as `df[df.x > 0]["y"] = 1` never modifies `df` — write
> `df.loc[df.x > 0, "y"] = 1` instead.

### 5.1 Series, DataFrame, Index

```python
import pandas as pd
drugs = pd.DataFrame({
    "drugbank_id": ["DB00007", "DB00010", "DB00014", "DB00035"],
    "name": ["Leuprolide", "Sermorelin", "Goserelin", "Desmopressin"],
    "n_genes": [12, 0, 7, 3],
})
print(drugs)
print(drugs.dtypes)
print(drugs.shape, list(drugs.columns))
s = drugs["n_genes"]                    # a Series
print(type(s).__name__, s.mean(), s.to_numpy())
d2 = drugs.set_index("drugbank_id")     # use IDs as row labels
print(d2.loc["DB00014", "name"])        # label-based lookup
print(d2.iloc[0, 1])                    # position-based lookup (row 0, column 1)
```

Output:

```text
  drugbank_id          name  n_genes
0     DB00007    Leuprolide       12
1     DB00010    Sermorelin        0
2     DB00014     Goserelin        7
3     DB00035  Desmopressin        3
drugbank_id      str
name             str
n_genes        int64
dtype: object
(4, 3) ['drugbank_id', 'name', 'n_genes']
Series 5.5 [12  0  7  3]
Goserelin
12
```

Two indexers, never to be confused:

* **`.loc[row_labels, column_labels]`** — by *label*; slices include the end label.
* **`.iloc[row_positions, column_positions]`** — by *integer position*; slices exclude the end,
  exactly like NumPy.

`df["col"]` returns a column (Series); `df[["a", "b"]]` returns a DataFrame with those columns;
`df[bool_series]` filters rows.

### 5.2 Reading files with `read_csv`

`pd.read_csv` is the workhorse. The arguments the project uses are worth knowing exactly:

| Argument | Meaning | Project example |
|---|---|---|
| `sep="\t"` | field separator (TSV = tab) | all CTD files |
| `comment="#"` | ignore text after `#` (CTD header lines start with `#`) | all CTD files |
| `names=[...]` | supply column names (the CTD header is a comment, so pandas cannot see it) | `CTD_DISEASE_COLS` |
| `usecols=[...]` | parse only these columns — saves memory and time | `ctd_drug_genes` keeps 3 of 11 |
| `dtype=str` | read every column as text; IDs like `"0102100"` keep leading zeros | everywhere |
| `chunksize=N` | return an iterator of DataFrames of N rows each | 500,000 and 1,000,000 |
| compression | inferred from the `.gz` extension — no need to decompress | all CTD files |
| `nrows=N` | read only the first N rows (great for exploring) | — |
| `.fillna("")` | replace missing values by empty strings after reading | everywhere |

Why `dtype=str`? Type inference guesses. An OMIM ID such as `"102100"` would become the integer
102100 (fine), but a column containing `"0012"` would lose its leading zeros, and a column that is
numeric in the first rows but has text further down raises a "mixed types" warning. IDs are
labels, not quantities: read them as strings.

The real CTD files start with about 29 comment lines; the column names appear only inside a
comment (`# Fields:` followed by `# ChemicalName	ChemicalID	...`). Here is a miniature file with
the same layout that we create in a temporary folder, so the example runs anywhere:

```python
import gzip, os, tempfile
import pandas as pd

path = os.path.join(tempfile.gettempdir(), "mini_CTD_chem_gene_ixns.tsv.gz")
lines = [
    "# Comparative Toxicogenomics Database (miniature)",
    "# Fields:",
    "# ChemicalName\tChemicalID\tCasRN\tGeneSymbol\tGeneID\tGeneForms\tOrganism\tOrganismID\tInteraction\tInteractionActions\tPubMedIDs",
    "#",
    "Aspirin\tD001241\t50-78-2\tPTGS1\t5742\tprotein\tHomo sapiens\t9606\tAspirin inhibits PTGS1\tdecreases^activity\t111",
    "Aspirin\tD001241\t50-78-2\tPTGS2\t5743\tprotein\tHomo sapiens\t9606\tAspirin inhibits PTGS2\tdecreases^activity\t112",
    "Aspirin\tD001241\t50-78-2\tPtgs2\t19225\tprotein\tMus musculus\t10090\tAspirin inhibits Ptgs2\tdecreases^activity\t113",
    "Metformin\tD008687\t657-24-9\tPRKAA1\t5562\tprotein\tHomo sapiens\t9606\tMetformin activates PRKAA1\tincreases^activity\t114",
    "Metformin\tD008687\t657-24-9\tPTGS2\t5743\tmRNA\tHomo sapiens\t9606\tMetformin affects PTGS2\taffects^expression\t115",
    "Caffeine\tD002110\t58-08-2\tADORA2A\t135\tprotein\tHomo sapiens\t9606\tCaffeine binds ADORA2A\taffects^binding\t116",
    "Sildenafil\tD000068677\t139755-83-2\tPDE5A\t8654\tprotein\tHomo sapiens\t9606\tSildenafil inhibits PDE5A\tdecreases^activity\t117",
]
with gzip.open(path, "wt", encoding="utf8") as fh:
    fh.write("\n".join(lines) + "\n")

cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
        "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
df = pd.read_csv(path, sep="\t", comment="#", names=cols, dtype=str,
                 usecols=["ChemicalID", "GeneSymbol", "OrganismID"])
print(df)
print({c: str(t) for c, t in df.dtypes.items()})   # pandas 3: text columns -> "str"
```

Output:

```text
   ChemicalID GeneSymbol OrganismID
0     D001241      PTGS1       9606
1     D001241      PTGS2       9606
2     D001241      Ptgs2      10090
3     D008687     PRKAA1       9606
4     D008687      PTGS2       9606
5     D002110    ADORA2A       9606
6  D000068677      PDE5A       9606
{'ChemicalID': 'str', 'GeneSymbol': 'str', 'OrganismID': 'str'}
```

Notice that `usecols` kept only three columns, in file order, and that `comment="#"` silently
skipped the four header lines.

### 5.3 Selecting and filtering rows

A comparison on a column gives a boolean Series; indexing with it keeps the `True` rows. Combine
conditions with `&`, `|`, `~` and **parentheses** (same precedence rule as NumPy). Useful
vectorised helpers:

* `s.isin(collection)` — membership test against a set/list (the CTD loaders use
  `chunk.ChemicalID.isin(wanted)`).
* `s.str.contains(pattern)`, `s.str.startswith`, `s.str.replace`, `s.str.split("|")`,
  `s.str.lower()` — string methods applied to every element.
* `s.notna()` / `s.isna()` — missing-value tests.
* `df.query("OrganismID == '9606'")` — the same filter written as a string expression.

```python
import gzip, os, tempfile
import pandas as pd
# the miniature file written by the example in section 5.2 (run that one first)
path = os.path.join(tempfile.gettempdir(), "mini_CTD_chem_gene_ixns.tsv.gz")
cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
        "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
df = pd.read_csv(path, sep="\t", comment="#", names=cols, dtype=str)

wanted = {"D001241", "D008687"}                     # aspirin, metformin
human = df[(df.OrganismID == "9606") & df.ChemicalID.isin(wanted)]
print(human[["ChemicalName", "GeneSymbol"]])
print(df[df.InteractionActions.str.contains("decreases")].GeneSymbol.tolist())
print(df.query("OrganismID != '9606'")[["ChemicalName", "Organism"]])
print(df.loc[df.GeneSymbol == "PTGS2", "ChemicalName"].tolist())
```

Output:

```text
  ChemicalName GeneSymbol
0      Aspirin      PTGS1
1      Aspirin      PTGS2
3    Metformin     PRKAA1
4    Metformin      PTGS2
['PTGS1', 'PTGS2', 'Ptgs2', 'PDE5A']
  ChemicalName      Organism
2      Aspirin  Mus musculus
['Aspirin', 'Metformin']
```

### 5.4 Adding and transforming columns

* New column from a vectorised expression: `df["n"] = df.a + df.b`.
* Element-wise mapping through a dict or function: `s.map(d)` (unmatched keys → NaN).
* `s.apply(f)` calls a Python function per element — flexible but slow (it is a Python loop);
  prefer vectorised `.str` methods or NumPy where possible.
* `df.assign(n=...)` returns a new frame with the column added (handy in method chains).

`02_build_features.py` does `cd["drugbank_id"] = cd.ChemicalID.map(lambda c: "|".join(ctd2db.get(c, [])))`:
for each CTD chemical it looks up the list of DrugBank IDs mapped to it and joins them with `|`.

### 5.5 `groupby`: split – apply – combine

`df.groupby(key)` **splits** the rows into groups sharing the same key, **applies** a function to
each group, and **combines** the results into a new Series/DataFrame indexed by the keys.
Common aggregations: `size()` (rows per group), `count()` (non-missing per column), `nunique()`,
`sum()`, `mean()`, `agg(["min", "max"])`, `apply(list)` / `agg(list)`, `apply(set)`.

**Worked example (by hand).** Curated gene–disease rows:

| GeneSymbol | DiseaseID |
|---|---|
| TP53 | D1 |
| BRCA1 | D1 |
| EGFR | D2 |
| TP53 | D2 |
| TP53 | D3 |

`groupby("DiseaseID").size()` → D1: 2, D2: 2, D3: 1.
`groupby("GeneSymbol").DiseaseID.nunique()` → BRCA1: 1, EGFR: 1, TP53: 3.
`groupby("DiseaseID").GeneSymbol.apply(set).to_dict()` →
`{"D1": {"TP53", "BRCA1"}, "D2": {"EGFR", "TP53"}, "D3": {"TP53"}}` — the "gene set per
disease" structure that `ctd_disease_genes` builds (it uses an explicit loop with a
`defaultdict(set)` instead, which is equivalent).

```python
import pandas as pd
gd = pd.DataFrame({"GeneSymbol": ["TP53", "BRCA1", "EGFR", "TP53", "TP53"],
                   "DiseaseID": ["D1", "D1", "D2", "D2", "D3"]})
print(gd.groupby("DiseaseID").size())
print(gd.groupby("GeneSymbol").DiseaseID.nunique())
sets = gd.groupby("DiseaseID").GeneSymbol.apply(lambda g: sorted(set(g))).to_dict()
print(sets)
# several aggregations at once, with readable names
print(gd.groupby("DiseaseID").agg(n_genes=("GeneSymbol", "nunique"),
                                  first_gene=("GeneSymbol", "first")))
```

Output:

```text
DiseaseID
D1    2
D2    2
D3    1
dtype: int64
GeneSymbol
BRCA1    1
EGFR     1
TP53     3
Name: DiseaseID, dtype: int64
{'D1': ['BRCA1', 'TP53'], 'D2': ['EGFR', 'TP53'], 'D3': ['TP53']}
           n_genes first_gene
DiseaseID                    
D1               2       TP53
D2               2       EGFR
D3               1       TP53
```

The same idiom appears verbatim in `disease_table`:

```py
# scripts/02_build_features.py :: disease_table
mg_mesh = mg[mg.src == "MeSH"].groupby("cui").src_id.apply(list).to_dict()
```

Read it left to right: keep the MedGen rows whose source is MeSH; group them by UMLS concept ID
(`cui`); for each concept, collect its MeSH IDs into a list; turn the resulting Series into a
plain dict `{cui: [mesh ids]}` for fast lookups later.

### 5.6 `merge`: joining tables on keys

`pd.merge(left, right, on=key, how=...)` (or `left.merge(right, ...)`) is SQL's JOIN:

| `how` | keeps |
|---|---|
| `"inner"` (default) | only keys present in **both** tables |
| `"left"` | every row of the left table; unmatched right columns become NaN |
| `"right"` | every row of the right table |
| `"outer"` | keys in either table |

If a key appears several times on both sides, the result contains **every combination** (a
many-to-many join multiplies rows). That is the most common way a merge silently inflates a
dataset. Protect yourself with `validate="one_to_one"` / `"many_to_one"` (raises an error if the
assumption is violated) and inspect `indicator=True`, which adds a `_merge` column saying where
each row came from.

**Worked example (by hand).** Left: drugs DB1, DB2, DB3. Right (a DrugBank → CTD chemical map):
DB1 → C1, DB3 → C3, DB3 → C3b (DB3 maps to two chemicals). Then

* inner join → 3 rows: (DB1, C1), (DB3, C3), (DB3, C3b) — DB2 disappears, DB3 doubles;
* left join → 4 rows: the 3 above plus (DB2, NaN).

```python
import pandas as pd
drugs = pd.DataFrame({"drugbank_id": ["DB1", "DB2", "DB3"], "name": ["a", "b", "c"]})
chem = pd.DataFrame({"drugbank_id": ["DB1", "DB3", "DB3"], "ctd_id": ["C1", "C3", "C3b"]})
print(drugs.merge(chem, on="drugbank_id", how="inner"))
print(drugs.merge(chem, on="drugbank_id", how="left", indicator=True))
try:
    drugs.merge(chem, on="drugbank_id", validate="one_to_one")
except Exception as e:
    print(type(e).__name__, "->", str(e)[:60])
```

Output:

```text
  drugbank_id name ctd_id
0         DB1    a     C1
1         DB3    c     C3
2         DB3    c    C3b
  drugbank_id name ctd_id     _merge
0         DB1    a     C1       both
1         DB2    b    NaN  left_only
2         DB3    c     C3       both
3         DB3    c    C3b       both
MergeError -> Merge keys are not unique in right dataset; not a one-to-one
```

The project mostly avoids `merge` and instead builds **lookup dictionaries** (`by_cid`, `by_ik`,
`by_name` in `ctd_chemical_map`) and resolves each drug with a fallback chain
`by_cid.get(...) or by_ik.get(...) or by_name.get(...)`. That is a deliberate choice: a merge
can only use one key at a time, whereas entity resolution here wants "PubChem CID if available,
otherwise InChIKey, otherwise the name". Knowing `merge` is still essential for checking such
mappings (e.g. "which drugs did *not* map?" is a left join with `indicator=True`).

`pd.concat([df1, df2], ignore_index=True)` stacks tables vertically; `pubchem_drugs` uses it to
append newly fetched rows to the cached ones, and `ctd_curated_chem_disease` uses it to glue the
filtered chunks together.

### 5.7 Other everyday operations

```python
import pandas as pd
df = pd.DataFrame({"drug": ["DB3", "DB1", "DB2", "DB1"],
                   "disease": ["D9", "D2", "D2", "D2"],
                   "evidence": ["therapeutic", "marker/mechanism", None, "marker/mechanism"]})
print(df.drop_duplicates())                         # removes the repeated (DB1, D2) row
print(df.sort_values(["disease", "drug"]).reset_index(drop=True))
print(df.evidence.value_counts(dropna=False))
print(df.fillna("").evidence.tolist())
print(df.evidence.str.contains("marker", na=False).tolist())
```

Output:

```text
  drug disease          evidence
0  DB3      D9       therapeutic
1  DB1      D2  marker/mechanism
2  DB2      D2               NaN
  drug disease          evidence
0  DB1      D2  marker/mechanism
1  DB1      D2  marker/mechanism
2  DB2      D2               NaN
3  DB3      D9       therapeutic
evidence
marker/mechanism    2
therapeutic         1
NaN                 1
Name: count, dtype: int64
['therapeutic', 'marker/mechanism', '', 'marker/mechanism']
[False, True, False, True]
```

### 5.8 Iterating over rows (and why to avoid it)

Sometimes a row loop is unavoidable — for example when each row needs a dictionary lookup that
depends on several columns. Then:

* `for r in df.itertuples():` — fast; `r.ColumnName` attribute access; the project uses it in
  `ctd_disease_genes` and `ctd_chemical_map`.
* `for c, g in zip(df.ChemicalID, df.GeneSymbol):` — fastest; just two columns.
* `for i, r in df.iterrows():` — slow (builds a Series per row) and does not preserve dtypes;
  avoid except for tiny tables (`pubchem_drugs` uses it on a few hundred rows, where it does not matter).

But first ask whether the loop can be replaced by a vectorised operation (`isin`, `map`,
`groupby`, `str` methods).

### 5.9 Reading huge files in chunks

**The problem.** `CTD_chemicals_diseases.tsv.gz` has 9,903,450 data rows (913 MB uncompressed).
A DataFrame of Python strings needs memory for every string object, typically 50–80 bytes each
plus an 8-byte pointer, so ten columns × ten million rows easily means 5+ GB — measured below.

**The solution.** With `chunksize=N`, `read_csv` returns an **iterator** (a `TextFileReader`)
that yields DataFrames of at most N rows. You process each chunk — usually *filter* it down to the
rows you need — and keep only the small result. Peak memory is roughly one chunk plus what you
keep, independent of file size. The pattern in the project:

```py
# scripts/02_build_features.py :: ctd_curated_chem_disease
keep = []
for chunk in pd.read_csv(RAW / "ctd" / "CTD_chemicals_diseases.tsv.gz", sep="\t",
                         comment="#", names=cols, dtype=str, chunksize=1_000_000,
                         usecols=[...7 of the 10 columns...]):
    chunk = chunk[chunk.DirectEvidence.notna() & chunk.ChemicalID.isin(wanted)]
    keep.append(chunk)
return pd.concat(keep, ignore_index=True).fillna("")
```

Three memory savers are stacked here: `usecols` drops three columns at parse time; the filter
keeps only *curated* rows (`DirectEvidence` not missing — inferred rows have it empty) about *our*
drugs; and `chunksize` bounds the working set to one million rows at a time.

The next example measures, on the real file, how much memory 200,000 rows take and extrapolates,
then runs the chunked pattern on the miniature file so you can see each chunk:

```python
import os, tempfile
import pandas as pd

cols = ["ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID", "DirectEvidence",
        "InferenceGeneSymbol", "InferenceScore", "OmimIDs", "PubMedIDs"]
sample = pd.read_csv("data/raw/ctd/CTD_chemicals_diseases.tsv.gz", sep="\t", comment="#",
                     names=cols, dtype=str, nrows=200_000)
mb = sample.memory_usage(deep=True).sum() / 1e6
print(f"200,000 rows: {mb:.0f} MB  ->  9,903,450 rows: ~{mb * 9_903_450 / 200_000 / 1e3:.1f} GB")
print("curated share in sample:", f"{sample.DirectEvidence.notna().mean():.2%}")

path = os.path.join(tempfile.gettempdir(), "mini_CTD_chem_gene_ixns.tsv.gz")   # from section 5.2
gcols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
         "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
genes = {}
for n, chunk in enumerate(pd.read_csv(path, sep="\t", comment="#", names=gcols, dtype=str,
                                      usecols=["ChemicalID", "GeneSymbol", "OrganismID"],
                                      chunksize=3)):
    print(f"chunk {n}: {len(chunk)} rows, index {chunk.index.min()}..{chunk.index.max()}")
    chunk = chunk[chunk.OrganismID == "9606"]
    for c, g in zip(chunk.ChemicalID, chunk.GeneSymbol):
        genes.setdefault(c, set()).add(g)
print({c: sorted(g) for c, g in genes.items()})
```

Output:

```text
200,000 rows: 111 MB  ->  9,903,450 rows: ~5.5 GB
curated share in sample: 0.51%
chunk 0: 3 rows, index 0..2
chunk 1: 3 rows, index 3..5
chunk 2: 1 rows, index 6..6
{'D001241': ['PTGS1', 'PTGS2'], 'D008687': ['PRKAA1', 'PTGS2'], 'D002110': ['ADORA2A'], 'D000068677': ['PDE5A']}
```

So the naive full load would need on the order of 5 GB, whereas the chunked loop never holds more
than one million rows of 7 columns (a few hundred MB) at once and keeps only a tiny curated subset. Note that chunk
indices continue across chunks (0–2, 3–5, …): each chunk is a slice of one long virtual table.

**Alternatives worth knowing.** Pass `dtype={"col": "category"}` for low-cardinality columns
(stores each distinct string once); use Parquet files for repeated reading; or use libraries
designed for out-of-core data (Polars, DuckDB, Dask). For this project, chunked filtering is
simple and sufficient.

### 5.10 From tables to matrices and back

Graph and matrix code needs integer positions, not string IDs. The standard translation:

1. Fix an **order** for each entity type (e.g. `drugs = sorted(...)`).
2. Build a dict `{id: position}`.
3. Fill a NumPy matrix by looking up positions.

That is exactly `gene_matrix` in `02_build_features.py`. The reverse direction — matrix back to a
readable table — uses `np.nonzero`:

```python
import numpy as np
import pandas as pd

edges = pd.DataFrame({"drug": ["DB1", "DB1", "DB2", "DB3"],
                      "disease": ["Asthma", "COPD", "Asthma", "Gout"]})
drugs = sorted(edges.drug.unique())
diseases = sorted(edges.disease.unique())
r_idx = {d: i for i, d in enumerate(drugs)}
c_idx = {d: j for j, d in enumerate(diseases)}

A = np.zeros((len(drugs), len(diseases)), dtype=np.float32)
A[edges.drug.map(r_idx).to_numpy(), edges.disease.map(c_idx).to_numpy()] = 1   # fancy assignment
print(drugs, diseases)
print(A)

# the same matrix with pandas in one line
print(pd.crosstab(edges.drug, edges.disease))

# matrix -> edge list
i, j = np.nonzero(A)
print(pd.DataFrame({"drug": np.array(drugs)[i], "disease": np.array(diseases)[j]}))
```

Output:

```text
['DB1', 'DB2', 'DB3'] ['Asthma', 'COPD', 'Gout']
[[1. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
disease  Asthma  COPD  Gout
drug                       
DB1           1     1     0
DB2           1     0     0
DB3           0     0     1
  drug disease
0  DB1  Asthma
1  DB1    COPD
2  DB2  Asthma
3  DB3    Gout
```

### 5.11 Writing tables

`df.to_csv(path, index=False)` writes a CSV without the row index (use `index=True`, the default,
when the index carries information, as for `drug_tab.to_csv(INTERIM / "drugs_all.csv")` where the
index is the DrugBank ID). The `data/interim/*.csv` files exist so that a human can open them in
Excel and audit every ID mapping — a good research habit.

---

## 6. In this project: the code, line by line

All snippets below are quoted from the repository (file and function named in the first
comment). Run the demonstrations from the project root with the project's interpreter.

### 6.1 `evaluation.py::kfold_splits` and the masks of `run_kfold`

```py
# src/drepo/evaluation.py :: kfold_splits
def kfold_splits(A, k, seed):
    rng = np.random.default_rng(seed)          # (1)
    pos = np.flatnonzero(A.ravel() > 0)        # (2)
    neg = np.flatnonzero(A.ravel() == 0)       # (3)
    rng.shuffle(pos)                           # (4)
    rng.shuffle(neg)
    pf, nf = np.array_split(pos, k), np.array_split(neg, k)   # (5)
    for f in range(k):
        yield f, pf[f], nf[f]                  # (6)
```

1. A private, seeded generator: the folds depend only on `seed`, not on anything else that drew
   random numbers before.
2. `A.ravel()` is a 185,609-long *view*; `> 0` gives a boolean vector; `flatnonzero` returns the
   flat positions of the 1,933 known links (int64, ascending).
3. Same for the 183,676 unknown cells.
4. Shuffle each list in place, so consecutive chunks are random subsets.
5. Cut each shuffled list into k nearly equal pieces (§4.12): **positives and negatives are
   stratified separately**, so each fold has ~1/k of each class and the class ratio is preserved.
6. A generator: fold `f`'s test positives and test negatives as flat indices.

Then, inside `run_kfold` for one fold:

```py
# src/drepo/evaluation.py :: run_kfold
A_tr = A.copy().ravel()          # flat COPY of the labels
A_tr[test_pos] = 0               # hide this fold's positives
A_tr = A_tr.reshape(shape)       # back to drugs x diseases (same C order)
neg_mask = (A == 0).ravel()      # all unknown cells are candidate negatives ...
neg_mask[test_neg] = False       # ... except this fold's test negatives
neg_mask = neg_mask.reshape(shape)
S = method.fit_predict(data, A_tr, neg_mask, seed=...)
idx = np.concatenate([test_pos, test_neg])
y = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
s = S.ravel()[idx]               # predicted scores at the test cells, same order as y
```

Note what `neg_mask` does **not** exclude: the hidden test *positives* are 0 in `A_tr` but are
still `False` in `neg_mask` because `neg_mask` starts from the *true* `A == 0`. So a test positive
can never be sampled as a training negative. (`fit_predict` additionally intersects with
`A_train == 0`.) Let us verify all of these invariants on the real data:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
from drepo.evaluation import kfold_splits

A = D.load("F").A
shape = A.shape
f, test_pos, test_neg = next(kfold_splits(A, 5, seed=0))
A_tr = A.copy().ravel(); A_tr[test_pos] = 0; A_tr = A_tr.reshape(shape)
neg_mask = (A == 0).ravel(); neg_mask[test_neg] = False; neg_mask = neg_mask.reshape(shape)

print("training links:", int(A_tr.sum()), "=", int(A.sum()), "-", len(test_pos))
print("all test positives hidden:", A_tr.ravel()[test_pos].max() == 0)
print("test positives never negatives:", not neg_mask.ravel()[test_pos].any())
print("test negatives never negatives:", not neg_mask.ravel()[test_neg].any())
print("negative pool size:", int(neg_mask.sum()), "=", (A == 0).sum() - len(test_neg))
i, j = np.unravel_index(test_pos[0], shape)
print(f"first test positive: flat {test_pos[0]} -> drug {i}, disease {j}; A[i,j] = {A[i, j]}")
```

Output:

```text
training links: 1546 = 1933 - 387
all test positives hidden: True
test positives never negatives: True
test negatives never negatives: True
negative pool size: 146940 = 146940
first test positive: flat 172704 -> drug 551, disease 241; A[i,j] = 1.0
```

`run_lodo` uses 2-D indexing instead of flat indices because the hidden set is a whole column:
`A_tr = A.copy(); A_tr[:, j] = 0; neg_mask = A == 0; neg_mask[:, j] = False`.

### 6.2 `data.py::knn_mask`

```py
# src/drepo/data.py :: knn_mask
def knn_mask(S, k, symmetric=True):
    S = fill_missing(S)                                   # (1) NaN -> 0, diagonal -> 1
    n = S.shape[0]
    work = S.copy()                                       # (2)
    np.fill_diagonal(work, -np.inf)                       # (3) never pick yourself
    k = min(k, n - 1)                                     # (4)
    idx = np.argpartition(-work, k, axis=1)[:, :k]        # (5) (n, k) column indices
    M = np.zeros_like(S, dtype=bool)                      # (6)
    rows = np.repeat(np.arange(n), k)                     # (7) 0,0,..,0,1,1,..,1,...
    vals = work[rows, idx.ravel()]                        # (8) the n*k similarity values
    keep = vals > 0                                       # (9) never link on zero similarity
    M[rows[keep], idx.ravel()[keep]] = True               # (10) fancy assignment
    if symmetric:
        M |= M.T                                          # (11) i~j if either picked the other
    np.fill_diagonal(M, True)                             # (12) self-loops
    return M
```

1. Missing view entries become 0 (no evidence), self-similarity 1.
2. A working copy, so the diagonal edit does not touch `S`.
3. Setting the diagonal to −∞ guarantees a node is never its own "nearest neighbour" (in `-work`
   it becomes +∞, the largest value, so it sorts last).
4. A node cannot have more than n−1 other neighbours.
5. **The top-k selection.** Partition each row of `-work` (largest similarity = smallest value)
   so that the first k positions hold the k most similar *other* nodes, unordered. Shape `(n, k)`.
6. An all-False n × n boolean matrix.
7. Row numbers repeated k times — length `n*k` — to pair with the flattened column indices.
8. Fancy indexing with two equal-length arrays → the `n*k` selected similarity values.
9. If a node has fewer than k positive similarities (e.g. a drug with no SMILES in `chem_ecfp`),
   the partition is forced to pick zeros; this line drops them. (Among tied zeros, *which* ones
   argpartition picked is arbitrary — it does not matter because they are all dropped.)
10. Write `True` at the kept (row, column) pairs.
11. `M.T` is the transpose (a view); `|=` is in-place OR. The graph becomes **undirected**: an edge
    exists if *either* endpoint chose the other. Consequently many nodes end up with *more*
    than k neighbours (popular "hub" nodes are chosen by many others).
12. Every node is its own neighbour, so attention always has at least one target.

A toy run that you can check by hand. Row 4 has only one positive similarity, so it keeps only one
neighbour; row 1 gains neighbour 3 through symmetrisation because node 3 chose node 1:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D

S = np.array([[1.0, 0.9, 0.2, 0.0, 0.4],
              [0.9, 1.0, 0.3, 0.1, 0.0],
              [0.2, 0.3, 1.0, 0.8, 0.0],
              [0.0, 0.1, 0.8, 1.0, 0.0],
              [0.4, 0.0, 0.0, 0.0, 1.0]])
print(D.knn_mask(S, 2, symmetric=False).astype(int))
print(D.knn_mask(S, 2).astype(int))

d = D.load("F")
M = D.knn_mask(d.drug_view("chem_cdk"), 10)
deg = M.sum(1) - 1                      # neighbours excluding self
print("chem_cdk kNN graph: neighbours per drug min/mean/max =",
      deg.min(), round(deg.mean(), 2), deg.max(), "| symmetric:", (M == M.T).all())
```

Output:

```text
[[1 1 0 0 1]
 [1 1 1 0 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [1 0 0 0 1]]
[[1 1 0 0 1]
 [1 1 1 1 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [1 0 0 0 1]]
chem_cdk kNN graph: neighbours per drug min/mean/max = 10 14.79 94 | symmetric: True
```

By hand for row 0: similarities to others are (0.9, 0.2, 0.0, 0.4) for nodes (1, 2, 3, 4); the two
largest are nodes 1 and 4 → row 0 = `1 1 0 0 1` (including the diagonal). Row 3: (0.0, 0.1, 0.8, 0.0)
for nodes (0, 1, 2, 4) → nodes 2 and 1. Since 3 chose 1, symmetrisation adds the edge 1–3.

### 6.3 `data.py::topk_bipartite` and `data.py::knn_kernel`

`topk_bipartite` is the rectangular version for drug × disease scores (the optional gene bridge):
same argpartition idiom with `kth = k - 1`, no symmetrisation (a rectangular matrix has no
diagonal and its transpose has a different shape; the caller builds the reverse direction from
`B.T`).

```py
# src/drepo/data.py :: knn_kernel
W = fill_missing(S) * knn_mask(S, k, symmetric=False)   # keep only each row's k best weights
np.fill_diagonal(W, 0.0)                                # exclude self
s = W.sum(1, keepdims=True)                             # (n, 1) row sums
s[s == 0] = 1.0                                         # rows with no neighbours: avoid 0/0
return W / s                                            # each row sums to 1 (or is all 0)
```

Multiplying a boolean mask by a float matrix casts `True/False` to `1/0`, so `W` keeps the
similarity values on kNN edges and zeros elsewhere. Dividing by `keepdims` row sums gives a
**row-stochastic** matrix: row $i$ holds weights that sum to 1 over node $i$'s neighbours, so
`K @ A` averages the association rows of $i$'s neighbours (Unit A2 explains the algebra). For the toy
`S` above, row 0 of the kernel is $0.9/1.3 = 0.692$ on node 1 and $0.4/1.3 = 0.308$ on node 4.

### 6.4 `similarity.py::jaccard`

```py
# src/drepo/similarity.py :: jaccard
G = (G > 0).astype(np.float64)                  # binary entity x gene matrix
inter = G @ G.T                                  # (n, n) shared-gene counts
size = G.sum(1)                                  # (n,) genes per entity
union = size[:, None] + size[None, :] - inter    # (n,1) + (1,n) - (n,n) -> (n,n)
with np.errstate(invalid="ignore", divide="ignore"):
    S = inter / union                            # 0/0 -> NaN for two empty entities
empty = size == 0
S[empty, :] = np.nan                             # boolean mask on axis 0: whole rows
S[:, empty] = np.nan                             # and whole columns
np.fill_diagonal(S, 1.0)
```

The union line is the inclusion–exclusion identity $|a \cup b| = |a| + |b| - |a\cap b|$,
vectorised by broadcasting a column and a row (§4.9). The two NaN lines use a boolean mask on one
axis combined with a full slice on the other.

### 6.5 Flat indices in PyTorch: `methods.py::MVHGATMethod.fit_predict`

PyTorch tensors follow the same C-order conventions as NumPy, so the same arithmetic appears:

```py
# src/drepo/methods.py :: MVHGATMethod.fit_predict
n_r, n_d = A_train.shape
pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)     # flat indices of training links
...
hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge   # hide ~20% of links
cold = torch.rand(n_d, device=DEVICE) < c.cold_frac           # ~10% of diseases go "cold"
hide |= cold[pos % n_d]                                       # disease of each link = pos % n_d
Am = torch.zeros(n_r * n_d, dtype=torch.bool, device=DEVICE)
Am[pos[~hide]] = True                                         # visible links, flat
Am = Am.view(n_r, n_d)                                        # torch's reshape-as-view
```

`pos % n_d` converts every flat link index into its disease column, then `cold[...]` (fancy
indexing with a long tensor) looks up whether that disease is cold this epoch. Every link of a cold
disease is hidden. Here is the same logic in miniature (CPU only):

```python
import numpy as np
import torch
torch.manual_seed(0)
A = torch.tensor([[1, 0, 1],
                  [0, 1, 1]], dtype=torch.float32)          # 2 drugs x 3 diseases
n_r, n_d = A.shape
pos = torch.as_tensor(np.flatnonzero(A.numpy().ravel() > 0))
print("flat links:", pos.tolist(), "-> diseases:", (pos % n_d).tolist(), "drugs:", (pos // n_d).tolist())
cold = torch.tensor([False, False, True])                   # make disease 2 cold
hide = torch.zeros(pos.numel(), dtype=torch.bool) | cold[pos % n_d]
Am = torch.zeros(n_r * n_d, dtype=torch.bool)
Am[pos[~hide]] = True
print(Am.view(n_r, n_d).int())                              # column 2 emptied
print("supervised (hidden) links:", pos[hide].tolist())
```

Output:

```text
flat links: [0, 2, 4, 5] -> diseases: [0, 2, 1, 2] drugs: [0, 0, 1, 1]
tensor([[1, 0, 0],
        [0, 1, 0]], dtype=torch.int32)
supervised (hidden) links: [2, 5]
```

### 6.6 `scripts/02_build_features.py`: chunked CTD reading and gene matrices

```py
# scripts/02_build_features.py :: ctd_drug_genes
wanted = set(chem_ids)                         # set: O(1) membership tests
genes = defaultdict(set)
for chunk in pd.read_csv(..., names=cols, usecols=["ChemicalID", "GeneSymbol", "OrganismID"],
                         dtype=str, chunksize=500_000):
    chunk = chunk[(chunk.OrganismID == "9606") & chunk.ChemicalID.isin(wanted)]   # human + our drugs
    for c, g in zip(chunk.ChemicalID, chunk.GeneSymbol):
        genes[c].add(g)                        # chemical -> set of human genes
return genes
```

`9606` is the NCBI Taxonomy ID of *Homo sapiens*: mouse and rat interactions (also in CTD) are
dropped. Only three of eleven columns are parsed, each 500,000-row chunk is filtered immediately,
and only a dictionary of small sets survives. Then:

```py
# scripts/02_build_features.py :: gene_matrix
idx = {g: i for i, g in enumerate(vocab)}                 # gene symbol -> column
G = np.zeros((len(entities), len(vocab)), dtype=np.float32)
for r, e in enumerate(entities):
    for g in entity_genes.get(e, ()):
        if g in idx:
            G[r, idx[g]] = 1
```

…which is the "tables → matrix" translation of §5.10. In `main`, the gene vocabulary is restricted
to genes that touch at least two entities of the dataset (`cnt[g] >= 2`): a gene seen by one entity
only can never contribute to a similarity or a bridge, so dropping it shrinks the matrices without
changing any result. Finally, the views are stacked into 3-D arrays with
`np.stack(list(views_r.values()))` (shape `(3, n_drugs, n_drugs)`; `np.stack` adds a **new** leading
axis, whereas `np.concatenate` joins along an *existing* axis) and saved with
`np.savez_compressed`.

### 6.7 `data.py::load` and `runner.py`

`load` opens the `.npz`, casts `A` to `float32`, converts the view-name arrays to Python lists
(so that `.index(name)` works in `drug_view`), and wraps everything in the `DDData` dataclass.
`runner.py::run_protocol` parses the protocol string (`int(protocol[2:])` turns `"cv10"` into 10)
and, for a LODO subset, draws diseases with `rng.choice(data.n_diseases, lodo_subset, replace=False)`.
`runner.py::save` writes the summary as JSON and the pooled `(y, s)` arrays as
`np.savez_compressed`, and `load_table` gathers all JSON files of a protocol into one
`pd.DataFrame` for printing.

---

## 7. Common mistakes and misconceptions

| Mistake | Symptom | Fix |
|---|---|---|
| Installing with one interpreter, running with another | `ModuleNotFoundError` although "pip says it is installed" | `python -m pip install ...`; print `sys.executable` |
| Forgetting `.copy()` before modifying a slice | original data silently changed; later folds wrong | `A_tr = A.copy()`; check with `np.shares_memory` |
| Believing fancy indexing gives a view | edits to `B = A[[0, 2]]` do not reach `A` | assign through the original: `A[[0, 2]] = ...` |
| `A[rows, cols]` expecting a sub-matrix | 1-D result of pairs | `A[np.ix_(rows, cols)]` |
| `x == 0 & m` without parentheses | wrong mask or a dtype error | `(x == 0) & m` |
| Scaling rows with `S * d` | columns scaled instead | `S * d[:, None]` |
| `W / W.sum(1)` | each column divided by a row sum | `keepdims=True` |
| Flattening with one order, reshaping with another (`A.T.ravel()`, `order="F"`) | wrong cells hidden/evaluated | always C order; use `np.unravel_index` to check |
| `S = np.fill_diagonal(S, 1)` / `x = rng.shuffle(x)` | `S`/`x` becomes `None` | these work in place: call them without assignment |
| Testing `x == np.nan` | always `False` | `np.isnan(x)` |
| Expecting `argpartition(...)[:k]` to be sorted | wrong ranking order | sort the k selected afterwards |
| `np.split` with a length not divisible by k | `ValueError` | `np.array_split` |
| Re-seeding only NumPy | PyTorch results still vary | seed `random`, NumPy and torch (`set_seed`) |
| Reading IDs with type inference | leading zeros lost, mixed-type warnings | `dtype=str` |
| Loading a multi-GB table at once | MemoryError or heavy swapping | `usecols`, `chunksize`, filter per chunk |
| Many-to-many `merge` | row count explodes | `validate=...`, check `len()` before/after |
| Chained assignment `df[mask]["c"] = v` | no effect in pandas 3 | `df.loc[mask, "c"] = v` |
| Confusing `.loc` (labels, end inclusive) and `.iloc` (positions, end exclusive) | off-by-one rows | pick one deliberately |
| `iterrows` on millions of rows | very slow | vectorise, or `itertuples` / `zip` |
| float64 arrays sent to PyTorch | dtype errors or 2× GPU memory | cast to `float32` (`methods.py::t` does it) |

---

## 8. Exercises

Exercises are graded: **[C]** conceptual, **[M]** mathematical (by hand), **[P]** programming.
Run programming solutions from the project root. Try each before opening the solution.

**Exercise 1 [C].** For each statement, say whether `A` (a NumPy array) is modified:
(a) `B = A[2:5]; B[0] = 9`; (b) `B = A[[2, 3, 4]]; B[0] = 9`; (c) `A[[2, 3]] = 9`;
(d) `B = A.ravel(); B[0] = 9` (A contiguous); (e) `B = A.T.ravel(); B[0] = 9`;
(f) `B = A.astype(np.float32); B[0] = 9` (A is float64); (g) `B = A; B[0] = 9`.

<details><summary>Solution</summary>

(a) **Yes** — basic slicing gives a view. (b) **No** — fancy indexing returns a copy. (c) **Yes** —
fancy indexing on the left of `=` assigns into `A`. (d) **Yes** — `ravel` of a C-contiguous array is
a view. (e) **No** — `A.T` is not C-contiguous, so `ravel` must copy; `B` is independent (its element
`B[0]` has the value of `A[0, 0]` but lives in new memory). (f) **No** — `astype` to a different dtype always copies. (g) **Yes** — plain assignment
just binds a second name to the same object.

```python
import numpy as np
def fresh(): return np.arange(12, dtype=np.float64).reshape(6, 2)
cases = {
    "a": lambda A: A[2:5].__setitem__(0, 9),
    "b": lambda A: A[[2, 3, 4]].__setitem__(0, 9),
    "c": lambda A: A.__setitem__([2, 3], 9),
    "d": lambda A: A.ravel().__setitem__(0, 9),
    "e": lambda A: A.T.ravel().__setitem__(0, 9),
    "f": lambda A: A.astype(np.float32).__setitem__(0, 9),
    "g": lambda A: A.__setitem__(0, 9),
}
for name, op in cases.items():
    A = fresh(); op(A)
    print(name, "modified" if not np.array_equal(A, fresh()) else "unchanged")
```

Output:

```text
a modified
b unchanged
c modified
d modified
e unchanged
f unchanged
g modified
```

</details>

**Exercise 2 [M].** Cdataset has 663 drugs × 409 diseases. (a) Which flat index does
(drug 10, disease 20) have? (b) Which (drug, disease) is flat index 100,000? (c) If somebody
flattened `A.T` (diseases × drugs) instead, which flat index would (drug 10, disease 20) get?
(d) What is the largest valid flat index?

<details><summary>Solution</summary>

(a) $k = 10\cdot 409 + 20 = 4{,}110$. (b) $100{,}000 / 409 = 244.5\ldots$, so $i = 244$;
$244 \cdot 409 = 99{,}796$; $j = 100{,}000 - 99{,}796 = 204$ → (244, 204). (c) In `A.T`, rows are
diseases and there are 663 columns: $k' = 20\cdot 663 + 10 = 13{,}270$ — a completely different
cell, which is why the flatten/reshape order must match. (d) $663 \cdot 409 - 1 = 271{,}166$.

```python
import numpy as np
print(np.ravel_multi_index((10, 20), (663, 409)), np.unravel_index(100_000, (663, 409)),
      np.ravel_multi_index((20, 10), (409, 663)), 663 * 409 - 1)
```

Output:

```text
4110 (np.int64(244), np.int64(204)) 13270 271166
```

</details>

**Exercise 3 [M].** For 10-fold CV on Cdataset (2,532 positives), what are the positive and
negative fold sizes produced by `np.array_split`?

<details><summary>Solution</summary>

Positives: $2532 = 10\cdot 253 + 2$ → two folds of 254 and eight of 253.
Negatives: $663\cdot409 = 271{,}167$ cells, minus 2,532 = 268,635 $= 10 \cdot 26{,}863 + 5$ → five
folds of 26,864 and five of 26,863.

```python
import sys; sys.path.insert(0, "src")
from drepo import data as D
from drepo.evaluation import kfold_splits
A = D.load("C").A
print([len(p) for _, p, _ in kfold_splits(A, 10, 0)])
print([len(n) for _, _, n in kfold_splits(A, 10, 0)])
```

Output:

```text
[254, 254, 253, 253, 253, 253, 253, 253, 253, 253]
[26864, 26864, 26864, 26864, 26864, 26863, 26863, 26863, 26863, 26863]
```

</details>

**Exercise 4 [M].** Predict the result shape (or "error") without running:
(a) `np.ones((593, 1)) * np.ones(313)`; (b) `np.ones((3, 593, 593)).sum(0)`;
(c) `np.ones((3, 593, 593)).mean(axis=(1, 2))`; (d) `np.ones((593, 313)) + np.ones(593)`;
(e) `np.ones((6, 593, 313)) * np.ones(6)[:, None, None]`; (f) `np.ones((593, 313)).sum(1, keepdims=True) / np.ones((593, 313))`.

<details><summary>Solution</summary>

(a) (593, 313) — a column times a row is an outer product. (b) (593, 593) — axis 0 disappears.
(c) (3,) — one mean per view. (d) **error**: (593, 313) vs (1, 593), 313 ≠ 593 (to add a per-drug
value write `np.ones(593)[:, None]`). (e) (6, 593, 313) — the propagation-head pattern
`view_weights()[:, None, None] * P` in `model.py`. (f) (593, 313).

```python
import numpy as np
exprs = {
    "a": lambda: np.ones((593, 1)) * np.ones(313),
    "b": lambda: np.ones((3, 593, 593)).sum(0),
    "c": lambda: np.ones((3, 593, 593)).mean(axis=(1, 2)),
    "d": lambda: np.ones((593, 313)) + np.ones(593),
    "e": lambda: np.ones((6, 593, 313)) * np.ones(6)[:, None, None],
    "f": lambda: np.ones((593, 313)).sum(1, keepdims=True) / np.ones((593, 313)),
}
for k, f in exprs.items():
    try:
        print(k, f().shape)
    except ValueError:
        print(k, "error")
```

Output:

```text
a (593, 313)
b (593, 593)
c (3,)
d error
e (6, 593, 313)
f (593, 313)
```

</details>

**Exercise 5 [P].** Using Fdataset, list the 5 diseases with the most known drugs (name and
count), first with NumPy (`argsort`), then with pandas (`nlargest`), and check that both agree.

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from drepo import data as D
d = D.load("F")
deg = d.A.sum(0)
top = np.argsort(-deg, kind="stable")[:5]
for j in top:
    print(f"{int(deg[j]):3d}  {d.disease_names[j][:50]}")
s = pd.Series(deg, index=d.disease_names).nlargest(5)
print(list(s.index) == list(d.disease_names[top]), s.astype(int).tolist())
```

Output:

```text
 84  Progressive hereditary glomerulonephritis without 
 71  Insensitivity to pain with hyperplastic Myelinopat
 68  HYPERTENSION, DIASTOLIC, RESISTANCE TO
 41  Mismatch repair cancer syndrome 1
 36  Asthma, nasal polyps, and aspirin intolerance
True [84, 71, 68, 41, 36]
```

`kind="stable"` and `nlargest` both keep the first occurrence among ties, so the two orders agree.

</details>

**Exercise 6 [P].** Write `topk_rows(S, k)` that returns, for every row, the column indices of the
k largest entries **in descending order**, using `argpartition` + a small sort. Verify on a random
200 × 300 matrix that it agrees with a full `argsort`.

<details><summary>Solution</summary>

```python
import numpy as np
def topk_rows(S, k):
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]            # unordered top-k, O(n) per row
    vals = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-vals, axis=1)                           # sort only k items per row
    return np.take_along_axis(idx, order, axis=1)

rng = np.random.default_rng(0)
S = rng.random((200, 300))                                      # continuous -> no ties
print(np.array_equal(topk_rows(S, 7), np.argsort(-S, axis=1)[:, :7]))
print(topk_rows(np.array([[0.2, 0.9, 0.1, 0.7, 0.5]]), 3))
```

Output:

```text
True
[[1 3 4]]
```

</details>

**Exercise 7 [P].** For the `chem_ecfp` view of Fdataset with k = 10: (a) how many directed kNN
edges are there before symmetrisation (excluding self-loops)? (b) how many undirected edges after
`M |= M.T`? (c) what fraction of directed choices were *mutual*? (d) how many drugs have no
neighbour at all (only the self-loop), and why?

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
d = D.load("F")
S = d.drug_view("chem_ecfp")
Md = D.knn_mask(S, 10, symmetric=False); np.fill_diagonal(Md, False)
Ms = D.knn_mask(S, 10);                  np.fill_diagonal(Ms, False)
print("(a) directed edges:", int(Md.sum()))
print("(b) undirected edges:", int(Ms.sum()) // 2)
print("(c) mutual share:", round((Md & Md.T).sum() / Md.sum(), 3))
isolated = Ms.sum(1) == 0
uncovered = np.array([np.isnan(np.delete(S[i], i)).all() for i in range(len(S))])
print("(d) isolated drugs:", int(isolated.sum()), "| uncovered (no SMILES):", int(uncovered.sum()),
      "| same set:", bool((isolated == uncovered).all()))
```

Output:

```text
(a) directed edges: 5761
(b) undirected edges: 4067
(c) mutual share: 0.588
(d) isolated drugs: 15 | uncovered (no SMILES): 15 | same set: True
```

(a) Each covered drug chooses up to 10 others. (b) An undirected edge counts once for both
endpoints: it is the number of pairs where at least one chose the other, so it lies between
(a)/2 (all choices mutual) and (a) (none mutual). (c) The mutual share tells you how far the
kNN relation is from being symmetric. (d) Drugs without a SMILES (biologics) have an all-NaN row,
which `fill_missing` turns into "similar to nothing", and nobody chooses them either, so after
`fill_diagonal` they have only their self-loop. The model copes because their other views
(`chem_cdk`, `gene_r`) and their known links still provide messages.

</details>

**Exercise 8 [P].** Write a function `lodo_masks(A, j)` that returns `(A_tr, neg_mask)` for
leave-one-disease-out on disease `j`, exactly as `run_lodo` does, and assert three invariants:
column `j` of `A_tr` is all zero; no cell of column `j` is in `neg_mask`; `A` itself is unchanged.

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D

def lodo_masks(A, j):
    A_tr = A.copy()
    A_tr[:, j] = 0
    neg_mask = A == 0          # a NEW boolean array (comparison never returns a view)
    neg_mask[:, j] = False
    return A_tr, neg_mask

A = D.load("F").A
before = A.copy()
j = int(A.sum(0).argmax())
A_tr, neg = lodo_masks(A, j)
assert A_tr[:, j].sum() == 0
assert not neg[:, j].any()
assert np.array_equal(A, before)
print("disease", j, "hidden links:", int(A[:, j].sum()), "| remaining links:", int(A_tr.sum()),
      "| negative pool:", int(neg.sum()))
```

Output:

```text
disease 108 hidden links: 84 | remaining links: 1849 | negative pool: 183167
```

</details>

**Exercise 9 [P].** Given a DrugBank → CTD mapping table with a missing drug and a drug mapped
twice, use `merge(..., indicator=True)` to list (a) drugs that did not map, (b) drugs that mapped to
more than one chemical.

<details><summary>Solution</summary>

```python
import pandas as pd
drugs = pd.DataFrame({"drugbank_id": ["DB00007", "DB00010", "DB00014", "DB00035"]})
mapping = pd.DataFrame({"drugbank_id": ["DB00007", "DB00014", "DB00014", "DB00035"],
                        "ctd_id": ["D016729", "D017273", "C000001", "D003894"]})
m = drugs.merge(mapping, on="drugbank_id", how="left", indicator=True)
print("(a) unmapped:", m.loc[m._merge == "left_only", "drugbank_id"].tolist())
counts = mapping.groupby("drugbank_id").ctd_id.nunique()
print("(b) ambiguous:", counts[counts > 1].index.tolist())
```

Output:

```text
(a) unmapped: ['DB00010']
(b) ambiguous: ['DB00014']
```

</details>

**Exercise 10 [P].** Count interactions per organism in the first 600,000 data rows of the real
`CTD_chem_gene_ixns.tsv.gz`, reading it in chunks of 200,000 rows and accumulating counts in a
`pandas.Series`. Report the three most frequent organisms and confirm that the chunked total
equals a direct count.

<details><summary>Solution</summary>

```python
import pandas as pd
cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
        "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
path = "data/raw/ctd/CTD_chem_gene_ixns.tsv.gz"
total = pd.Series(dtype="int64")
for chunk in pd.read_csv(path, sep="\t", comment="#", names=cols, usecols=["Organism"],
                         dtype=str, chunksize=200_000, nrows=600_000):
    total = total.add(chunk.Organism.value_counts(), fill_value=0)
total = total.astype(int).sort_values(ascending=False)
print(total.head(3))
direct = pd.read_csv(path, sep="\t", comment="#", names=cols, usecols=["Organism"],
                     dtype=str, nrows=600_000).Organism.value_counts()
print("chunked == direct:", total.sort_index().equals(direct.sort_index().astype(int)))
```

Output:

```text
Organism
Homo sapiens         290808
Mus musculus         155454
Rattus norvegicus     96299
dtype: int64
chunked == direct: True
```

`Series.add(..., fill_value=0)` aligns on the index (organism names), treating organisms missing
from one side as 0 — the general recipe for merging per-chunk counts.

</details>

**Exercise 11 [P].** Check `similarity.jaccard` against a brute-force Python implementation on a
small random binary matrix that includes one empty row. What does the function return for the
empty row and why?

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo.similarity import jaccard
rng = np.random.default_rng(3)
G = (rng.random((5, 12)) < 0.3).astype(float)
G[4] = 0                                             # an entity with no genes
S = jaccard(G)
sets = [set(np.flatnonzero(r)) for r in G]
ok = all(np.isclose(S[i, j], len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
         for i in range(4) for j in range(4))
print("matches brute force on covered rows:", ok)
print(S[4].round(2))
```

Output:

```text
matches brute force on covered rows: True
[nan nan nan nan  1.]
```

The empty row is NaN except for the diagonal 1: with no genes, Jaccard is $0/0$ (undefined), and the
project's convention is "unknown", not "dissimilar". `fill_missing` later converts it to "no
neighbours".

</details>

**Exercise 12 [C].** A colleague's script prints `ModuleNotFoundError: No module named 'rdkit'`,
although `pip install rdkit` reported success. List the diagnostic steps and the fix.

<details><summary>Solution</summary>

1. Print `sys.executable` inside the failing script and run `where python` / `where pip` in the
   terminal. Usually `pip` belongs to one interpreter (e.g. Anaconda base) and the script runs under
   another (the project `.venv`, or vice versa).
2. Install with the *same* interpreter: `& ".venv\Scripts\python.exe" -m pip install rdkit`, or
   activate the venv first and use `python -m pip`.
3. In VS Code / Jupyter, select the `.venv` interpreter/kernel explicitly.
4. Confirm with `python -c "import rdkit, sys; print(rdkit.__version__, sys.executable)"`.

</details>

**Exercise 13 [M].** Estimate the memory of (a) the propagation tensor `P` of shape (6, 593, 313)
in float32; (b) a dense float64 gene–gene matrix over the 17,626-gene vocabulary;
(c) `G @ G.T` for Fdataset drugs (593 × 17,626 float64 input).

<details><summary>Solution</summary>

(a) $6\cdot593\cdot313\cdot4 = 4{,}454{,}616$ bytes ≈ 4.5 MB — tiny.
(b) $17{,}626^2 \cdot 8 \approx 2.49\times10^9$ bytes ≈ 2.5 GB — avoid; this is why the project never
forms gene × gene matrices. (c) The input is $593\cdot17{,}626\cdot8 \approx 83.6$ MB and the output
$593^2 \cdot 8 \approx 2.8$ MB: computing entity × entity products is cheap.

```python
print(6 * 593 * 313 * 4, 17_626**2 * 8 / 1e9, 593 * 17_626 * 8 / 1e6, 593**2 * 8 / 1e6)
```

Output:

```text
4454616 2.485407008 83.617744 2.813192
```

</details>

**Exercise 14 [P, challenge].** Fdataset has 313 diseases but `A` has only 246 *distinct* columns.
Find the largest group of diseases with identical drug sets, and explain why identical columns
matter for low-rank methods (preview of Unit A2).

<details><summary>Solution</summary>

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
d = D.load("F")
cols, inverse, counts = np.unique(d.A.T, axis=0, return_inverse=True, return_counts=True)
print("distinct columns:", len(cols))
g = counts.argmax()
members = np.flatnonzero(inverse.ravel() == g)
print("largest group:", counts[g], "diseases sharing", int(cols[g].sum()), "drug(s)")
print([d.disease_names[j][:35] for j in members[:4]])
```

Output:

```text
distinct columns: 246
largest group: 8 diseases sharing 1 drug(s)
['Seizures, benign familial neonatal,', 'Gilbert syndrome', 'Arthrogryposis, renal dysfunction, ', 'Crigler-Najjar syndrome type 1']
```

`np.unique(..., axis=0)` on `A.T` treats each disease column as a row and finds duplicates. Many
diseases are known to be treated by exactly the same small set of drugs (often a single drug).
Identical columns are linearly dependent, so they cannot add to the rank: this is one reason the rank
of Fdataset's `A` (238) is below 313. Unit A2 develops rank properly.

</details>

---

## 9. Answers to the PREREQUISITES.md self-check questions (Unit A1)

### Q1. `A` is 593 × 313. What does `A.ravel()[k]` refer to in terms of (drug, disease)?

**Short answer.** Drug $i = \lfloor k/313 \rfloor$ (`k // 313`) and disease $j = k \bmod 313$
(`k % 313`); equivalently `A.ravel()[k] == A[k // 313, k % 313]`, and conversely the cell
$(i, j)$ is at $k = 313\,i + j$.

**Why.** NumPy stores `A` in row-major (C) order: all 313 diseases of drug 0, then all 313 of drug 1,
and so on (§4.8, with proof). `ravel()` walks memory in that order, so positions $313i, \dots,
313i + 312$ belong to drug $i$, and the offset inside that block is the disease index. In the
project the convention appears three times: `kfold_splits` stores folds as flat positions;
`run_kfold` hides them with `A_tr.ravel()[test_pos] = 0` and reads predictions with
`S.ravel()[idx]`; and `fit_predict` recovers the disease of each flat link with `pos % n_d`.

**Why it matters.** It only works because every `ravel`/`reshape`/`view` in the pipeline uses the same
C order *and* the same shape (drugs × diseases). Flattening `A.T` or using `order="F"` anywhere would
map index $k$ to a different cell — the folds would hide the wrong links and the evaluation would
silently leak. Example: $k = 1000 \Rightarrow (3, 61)$; in Fdataset that is drug 3
(*Calcitonin salmon*) and disease 61 (*Glaucoma 1, open angle, A*), an unknown pair:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from drepo import data as D
d = D.load("F")
k = 1000
i, j = divmod(k, d.n_diseases)
print(i, j, d.drug_names[i], "|", d.disease_names[j])
print(d.A.ravel()[k] == d.A[i, j], np.unravel_index(k, d.A.shape))
```

Output:

```text
3 61 Calcitonin salmon | Glaucoma 1, open angle, A
True (np.int64(3), np.int64(61))
```

### Q2. Why does `knn_mask` use `argpartition` instead of `argsort`?

**Short answer.** Because it needs only the **set** of the k most similar neighbours of each node,
not their order, and selection is cheaper than sorting: `argpartition` does $O(n)$ work per row
on average (introselect) versus $O(n\log n)$ for `argsort`.

**In depth.**

* *What `argpartition(-work, k, axis=1)` guarantees:* in every row, the entry at position `k` is
  the one a full sort would put there, everything left of it is no larger, everything right of it is
  no smaller. Taking `[:, :k]` therefore yields exactly the k largest similarities (because of the
  minus sign), in arbitrary order (§4.11).
* *Why order does not matter:* the result is written into a boolean adjacency matrix `M`. Setting
  `M[i, j] = True` is the same whichever neighbour is written first. The attention layer
  (`DenseGAT`) later assigns its own learned weights to the neighbours; the rank order of raw
  similarities is never used.
* *Cost:* a full sort recurses into both halves of every partition; quickselect recurses into only the
  half that contains position k, so the expected number of element visits is
  $n + n/2 + n/4 + \dots < 2n$. For one 593-long row this is a minor gain, but `knn_mask` is called
  for every view, every fold, every repeat and every hyper-parameter setting, and the same code must
  scale if someone runs it on a 10,000-drug dataset. In the timing example of §4.11 partitioning was
  already two to three times faster than sorting on a 3,000 × 3,000 matrix.
* *Subtleties handled by the surrounding code:* the diagonal is set to −∞ so a node never selects
  itself; `vals > 0` removes zero-similarity picks forced when a row has fewer than k positive
  entries; ties at the boundary are broken arbitrarily, which is harmless for a similarity graph.
  If you ever need the neighbours *ranked* (e.g. to print "top-10 most similar drugs"), partition
  first and then sort only the k selected values (Exercise 6).

### Q3. Why read `CTD_chemicals_diseases.tsv.gz` in chunks?

**Short answer.** Because the file is far larger than the part we need, and loading it at once
would need several gigabytes of RAM; reading it in chunks and filtering each chunk keeps memory
bounded to one chunk (plus the small filtered result) regardless of file size.

**In depth.**

* *Size:* 162 MB compressed, about 913 MB of text, 9,903,450 data rows (measured, §1). With every
  value held as a Python string, a sample of 200,000 rows takes ~111 MB in pandas, so the full table
  would take ~5.5 GB (§5.9) — on top of the decompression buffers and the copies pandas makes while
  filtering. On a laptop that is also running a GPU experiment, that means swapping or a
  `MemoryError`.
* *Need:* we keep only rows with curated evidence (`DirectEvidence` not empty — about 0.5% of rows in
  our sample, since the vast majority of CTD chemical–disease rows are *inferred* via genes) **and**
  whose chemical is one of our ~680 drugs. The final result is a tiny fraction of the file.
* *Mechanism:* `chunksize=1_000_000` turns `read_csv` into an iterator; gzip decompression streams
  along with it. Each chunk is filtered with vectorised `notna()` and `isin(wanted)` (a set), and only
  the survivors are appended to a list and concatenated once at the end (concatenating once is
  cheaper than growing a DataFrame inside the loop). `usecols` additionally skips three unused
  columns at parse time.
* *Bonus:* the same pattern (with `chunksize=500_000`) is used for the 3.2-million-row
  chemical–gene file in `ctd_drug_genes`, where only human (`OrganismID == "9606"`) interactions of
  our drugs are kept. Chunking also makes progress reporting and early stopping easy, and the code
  keeps working when CTD's next release is larger.
* *Why not just decompress and use another tool?* You could (DuckDB, Polars), but chunked pandas
  needs no extra dependency, is fast enough for a one-off preprocessing step, and is easy to audit.

---

## 10. Summary and cheat sheet

**Environment**

```text
python -m venv .venv            create            .venv\Scripts\Activate.ps1   activate (PowerShell)
python -m pip install -r requirements.txt         python -m pip freeze > lock.txt
sys.executable / sys.prefix != sys.base_prefix    which interpreter / inside a venv?
sys.path.insert(0, ROOT/"src")  make drepo importable from scripts/
if __name__ == "__main__":      run only when executed as a script
```

**NumPy essentials**

| Task | Code |
|---|---|
| shape / dtype / memory | `A.shape`, `A.dtype`, `A.nbytes` |
| flat ↔ 2-D index (C order) | $k = i m + j$; `np.ravel_multi_index`, `np.unravel_index`, `k // m`, `k % m` |
| view vs copy | slices, `ravel` (contiguous), `.T`, `reshape` → view; fancy/boolean indexing, `astype`, `flatten`, `.copy()` → copy |
| pairs vs sub-matrix | `A[rows, cols]` vs `A[np.ix_(rows, cols)]` |
| masks | `(A == 0) & m`, `~m`, `m.sum()`, `np.flatnonzero(m)`, `np.nonzero(m)`, `np.where(c, a, b)` |
| column / row vector | `x[:, None]` (n,1), `x[None, :]` (1,n) |
| broadcasting rule | align right; each axis equal or 1 |
| row-normalise | `W / W.sum(1, keepdims=True)` |
| reductions | the named axis disappears; `keepdims=True` keeps it as 1 |
| top-k per row | `np.argpartition(-S, k-1, axis=1)[:, :k]` (unordered; then sort the k) |
| ranking | `np.argsort(-x, kind="stable")` |
| reproducible randomness | `rng = np.random.default_rng(seed)`; `rng.shuffle` (in place), `rng.choice(n, m, replace=False)` |
| k folds | `np.array_split(x, k)`; sizes $q+1$ (first $r$) and $q$, where $n = qk + r$ |
| missing values | `np.isnan`, `np.nan_to_num`, `np.nansum`, `np.errstate` |
| in-place, returns None | `np.fill_diagonal`, `rng.shuffle`, `list.sort` |
| save / load | `np.savez_compressed(p, A=A, ...)`, `np.load(p, allow_pickle=False)` |

**pandas essentials**

| Task | Code |
|---|---|
| read a CTD file | `pd.read_csv(p, sep="\t", comment="#", names=cols, usecols=[...], dtype=str)` |
| huge file | `for chunk in pd.read_csv(..., chunksize=N): keep.append(chunk[filter])` then `pd.concat(keep)` |
| filter | `df[(df.a == x) & df.b.isin(S)]`, `df.loc[mask, "col"]`, `df.query("...")` |
| strings | `s.str.contains`, `s.str.split("\|")`, `s.str.replace`, `s.str.lower()` |
| group | `df.groupby(k).size()`, `.agg(name=("col", "nunique"))`, `.col.apply(list).to_dict()` |
| join | `l.merge(r, on=k, how="left", validate="many_to_one", indicator=True)` |
| stack | `pd.concat([df1, df2], ignore_index=True)` |
| table → matrix | id→position dicts + fancy assignment, or `pd.crosstab` |
| matrix → table | `i, j = np.nonzero(A)` |
| iterate (if you must) | `itertuples()` or `zip(df.a, df.b)`, not `iterrows()` |

---

## 11. Further resources (curated)

All links were checked to resolve when this unit was written. **Free** unless marked **Paid**.

**Core references**

* [*Python for Data Analysis*, 3rd ed. — Wes McKinney (open-access web edition)](https://wesmckinney.com/book/) — **Free.**
  Written by pandas' creator; chapters 4 (NumPy), 5 (pandas basics), 6 (file I/O), 7 (cleaning),
  8 (joins/reshaping) and 10 (groupby) map one-to-one onto §4–§5 of this unit.
* [*Python Data Science Handbook* — Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Free.**
  Chapter 2 is the clearest explanation of NumPy broadcasting, fancy indexing and `argpartition`
  anywhere; chapter 3 covers pandas.
* [NumPy: the absolute basics for beginners (numpy.org)](https://numpy.org/doc/stable/user/absolute_beginners.html) — **Free.** Official tutorial; the right first stop.
* [NumPy user guide: Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
  [Indexing on ndarrays](https://numpy.org/doc/stable/user/basics.indexing.html) and
  [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html) — **Free.** The authoritative rules behind §4.5–§4.9.
* [`numpy.argpartition` reference](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) and
  [Random Generator reference](https://numpy.org/doc/stable/reference/random/generator.html) — **Free.** Exact semantics of `kth` and of `default_rng`.
* [pandas: Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) and
  [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — **Free.** Official, short, visual.
* pandas user guide: [IO tools (read_csv, chunking)](https://pandas.pydata.org/docs/user_guide/io.html),
  [Group by](https://pandas.pydata.org/docs/user_guide/groupby.html),
  [Merge, join, concatenate](https://pandas.pydata.org/docs/user_guide/merging.html),
  [Scaling to large datasets](https://pandas.pydata.org/docs/user_guide/scale.html) — **Free.** Reference-grade detail for §5.

**Courses and lecture notes**

* [Stanford CS231n — Python/NumPy tutorial](https://cs231n.github.io/python-numpy-tutorial/) — **Free.** A compact crash course used by Stanford's deep-learning class.
* [Scientific Python Lectures (formerly SciPy Lecture Notes)](https://lectures.scientific-python.org/) — **Free.** Community course; the "NumPy: creating and manipulating numerical data" and "Advanced NumPy" chapters explain strides and memory layout.
* [MIT — The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) — **Free.** Shell, version control, environments: the tooling around your Python code.
* [Harvard CS50's Introduction to Programming with Python (CS50P)](https://cs50.harvard.edu/python/) — **Free** (certificate optional). If your core Python (functions, exceptions, classes) feels shaky.

**Practice**

* [100 NumPy exercises — Nicolas Rougier](https://github.com/rougier/numpy-100) — **Free.** Graded drills with solutions; ideal after §4.
* [From Python to NumPy — Nicolas Rougier](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) — **Free.** Teaches you to *think* in vectorised code; excellent on views vs copies.
* [Modern Pandas — Tom Augspurger](https://tomaugspurger.net/posts/modern-1-intro/) — **Free.** Idiomatic method chaining, indexes and performance from a pandas core developer.

**Environments and packaging**

* [Python docs: `venv` — creation of virtual environments](https://docs.python.org/3/library/venv.html) and
  [Tutorial §12: Virtual environments and packages](https://docs.python.org/3/tutorial/venv.html) — **Free.** The official reference for §2.
* [Python Packaging User Guide: Installing packages](https://packaging.python.org/en/latest/tutorials/installing-packages/) and
  [pip user guide](https://pip.pypa.io/en/stable/user_guide/) — **Free.** requirements files, `pip freeze`, index URLs.
* [Real Python — Python virtual environments: a primer](https://realpython.com/python-virtual-environments-a-primer/) — **Free** article (site has paid extras). Very clear on what activation actually does.
* [PyTorch — Get started locally](https://pytorch.org/get-started/locally/) — **Free.** Generates the exact `pip install torch --index-url …` command for your CUDA version.

**Paper**

* [Harris et al. (2020), "Array programming with NumPy", *Nature* 585](https://www.nature.com/articles/s41586-020-2649-2) — **Free** (open access). The design of the ndarray (strides, views, broadcasting) explained by its maintainers.

---

## 12. Glossary

* **argpartition** — NumPy function returning indices that place the k-th element in its sorted position with smaller elements before and larger after (unordered); used for top-k selection.
* **argsort** — indices that would sort an array.
* **axis** — one dimension of an array; axis 0 = rows, axis 1 = columns for 2-D arrays.
* **base interpreter** — the Python installation a virtual environment was created from (`home` in `pyvenv.cfg`).
* **boolean mask** — a `bool` array used to select or mark elements.
* **broadcasting** — NumPy's rule for combining arrays of different shapes by virtually repeating length-1 axes.
* **C order / row-major** — memory layout where the last index varies fastest; NumPy's and PyTorch's default.
* **chunked reading** — processing a large file as a sequence of smaller DataFrames (`chunksize`).
* **Copy-on-Write** — pandas 3 behaviour in which derived objects behave as independent copies.
* **copy** — an array with its own memory; changes do not affect the source.
* **DataFrame** — pandas 2-D labelled table; a collection of Series sharing an index.
* **dataclass** — a class whose `__init__`/`__repr__` are generated from annotated fields.
* **dtype** — the element type of an array (`float32`, `bool`, `int64`, `<U7` …).
* **fancy indexing** — indexing with integer arrays; returns a copy (assignment writes in place).
* **flat index** — the position of an element in the raveled (1-D) array; $k = i m + j$ in C order.
* **generator** — a function using `yield`, producing values lazily.
* **groupby** — split–apply–combine aggregation in pandas.
* **Index (pandas)** — the row (or column) labels of a Series/DataFrame.
* **introselect / quickselect** — selection algorithms that find the k-th smallest element in linear expected time.
* **keepdims** — reduction option that keeps the reduced axis with length 1 so the result broadcasts.
* **merge / join** — combining two tables on key columns (inner, left, right, outer).
* **module / package** — an importable `.py` file / a folder of modules with `__init__.py`.
* **NaN** — "not a number"; IEEE float value used for missing data; not equal to itself.
* **ndarray** — NumPy's n-dimensional, homogeneous, contiguous array type.
* **pip** — Python's package installer; `python -m pip` uses the current interpreter's pip.
* **reduction** — an operation that collapses an axis (sum, mean, max, any …).
* **requirements.txt / lock file** — list of package names (optionally pinned versions) needed to recreate an environment.
* **row-stochastic matrix** — non-negative matrix whose rows each sum to 1.
* **Series** — pandas 1-D labelled array.
* **site-packages** — the folder where an interpreter's third-party packages are installed.
* **strides** — bytes to move in memory to step one element along each axis.
* **sys.path** — the list of folders Python searches when importing.
* **vectorisation** — expressing computations as whole-array operations executed in compiled code.
* **view** — an array sharing memory with another array (different header, same data).
* **virtual environment (venv)** — an isolated Python environment with its own `site-packages`.
