# Heterogeneous Graph-Based Drug Repositioning (MV-HGAT)

Implementation of the project *"Heterogeneous Graph-Based Computational Drug
Repositioning for Disease-Drug Association Prediction"*.

* **Start here:** [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the full
  explanation, every assumption, and how to run each step.
* **Study plan:** [docs/PREREQUISITES.md](docs/PREREQUISITES.md): the topics
  to learn, in order, with self-check questions and resources.
* **Results:** [results/RESULTS.md](results/RESULTS.md) (generated) and `results/figures/`.

## Layout

```
data/raw/          downloaded sources (benchmarks .mat, CTD, MONDO, DO, MedGen)
data/interim/      readable mapping tables (drugs_all.csv, diseases_all.csv, ...)
data/processed/    model-ready Fdataset.npz / Cdataset.npz
src/drepo/         library: similarity, data, model, methods (ours + 5 baselines), evaluation
scripts/           01_download → 02_build_features → 03_evaluate → 04_ablation →
                   05_sensitivity → 06_case_study → 07_cross_dataset → 08_make_figures
results/           metrics (.json), predictions (.npz), case studies, figures
docs/              HOW_IT_WORKS.md, PREREQUISITES.md
```

## Quick start

```powershell
.venv\Scripts\activate          # or: python -m venv .venv; pip install -r requirements.txt
python scripts/01_download_data.py
python scripts/02_build_features.py
python scripts/03_evaluate.py --dataset F --protocol cv5 --repeats 5
python scripts/08_make_figures.py
```

Environment used: Python 3.13, PyTorch 2.14 (CUDA 12.6), RDKit, RTX 3050 6 GB.
