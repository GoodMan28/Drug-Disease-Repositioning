"""
Cross-dataset validation (Results 5.7).

Fdataset and Cdataset share many drugs (DrugBank IDs) and diseases (OMIM IDs)
but were curated separately, so each contains links the other lacks.

Train on ALL links of the source dataset, then ask: among drug-disease pairs
whose drug AND disease exist in both datasets and which are NOT links in the
source, do the target dataset's extra links get ranked above the rest?
This is a true external test: those links were never visible during training.

    python scripts/07_cross_dataset.py --source F --target C
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                               # noqa: E402
from drepo.evaluation import metrics                      # noqa: E402
from drepo.methods import METHODS, make                   # noqa: E402
from drepo.paths import RESULTS                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="F")
ap.add_argument("--target", default="C")
ap.add_argument("--methods", default=",".join(METHODS))
ap.add_argument("--seeds", type=int, default=3)
args = ap.parse_args()

src, tgt = D.load(args.source), D.load(args.target)
r_idx = {d: i for i, d in enumerate(tgt.drug_ids)}
d_idx = {d: i for i, d in enumerate(tgt.disease_ids)}
sr = [i for i, d in enumerate(src.drug_ids) if d in r_idx]
sd = [j for j, d in enumerate(src.disease_ids) if d in d_idx]
tr = [r_idx[src.drug_ids[i]] for i in sr]
td = [d_idx[src.disease_ids[j]] for j in sd]

A_src = src.A[np.ix_(sr, sd)]
A_tgt = tgt.A[np.ix_(tr, td)]
cand = A_src == 0                                  # pairs unseen in the source
y = A_tgt[cand]
print(f"shared drugs {len(sr)}/{src.n_drugs}, shared diseases {len(sd)}/{src.n_diseases}")
print(f"source links among shared entities {int(A_src.sum())}, target links "
      f"{int(A_tgt.sum())}, NEW target links to recover {int(y.sum())} "
      f"out of {cand.sum()} candidate pairs")
if y.sum() == 0:
    print(f"\nNothing to test: every {tgt.name} link among the shared drugs/diseases is "
          f"already a {src.name} link (the target is a subset of the source here).")
    stale = RESULTS / f"cross_{src.name}_to_{tgt.name}.json"
    stale.unlink(missing_ok=True)
    sys.exit(0)

res = {}
for key in args.methods.split(","):
    ss = []
    for s in range(args.seeds if key not in ("mbirw",) else 1):
        m = make(key)
        S = m.fit_predict(src, src.A, src.A == 0, seed=s)
        ss.append(S[np.ix_(sr, sd)][cand])
    r = metrics(y, np.mean(ss, 0))
    res[m.name] = r
    print(f"{m.name:16s} AUC {r['AUC']:.4f}  AUPR {r['AUPR']:.4f}  "
          f"(random AUPR = {y.mean():.4f})", flush=True)

out = RESULTS / f"cross_{src.name}_to_{tgt.name}.json"
out.write_text(json.dumps({"n_new_links": int(y.sum()), "n_candidates": int(cand.sum()),
                           "results": res}, indent=2))
