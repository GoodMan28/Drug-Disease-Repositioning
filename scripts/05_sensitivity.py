"""
Hyper-parameter tuning / sensitivity analysis (Methodology Step 5, Results 5.5).

To avoid tuning on the test data, we carve a VALIDATION split out of the data:
20% of the known links and 20% of the unknown pairs are hidden, the model
trains on the rest, and we score on the hidden 20%. One hyper-parameter is
varied at a time around the defaults in MVHGATConfig.

    python scripts/05_sensitivity.py --dataset F --seeds 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                               # noqa: E402
from drepo.evaluation import metrics                      # noqa: E402
from drepo.methods import MVHGATMethod, MVHGATConfig      # noqa: E402
from drepo.paths import RESULTS                           # noqa: E402

GRID = {
    "hidden": [16, 32, 64, 128],
    "layers": [1, 2, 3],
    "neg_ratio": [1, 2, 5, 10],
    "k": [5, 10, 20, 40],
    "drop_edge": [0.1, 0.2, 0.4, 0.6],
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="F")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--params", default=",".join(GRID))
args = ap.parse_args()

data = D.load(args.dataset)
A = data.A
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
idx = np.concatenate([val_pos, val_neg])
y = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])

out = {}
default = MVHGATConfig()
for p in args.params.split(","):
    out[p] = []
    for v in GRID[p]:
        runs = []
        for s in range(args.seeds):
            S = MVHGATMethod(**{p: v}).fit_predict(data, A_tr, neg_mask, seed=s)
            runs.append(metrics(y, S.ravel()[idx]))
        r = {"value": v,
             "AUC": float(np.mean([x["AUC"] for x in runs])),
             "AUC_std": float(np.std([x["AUC"] for x in runs])),
             "AUPR": float(np.mean([x["AUPR"] for x in runs])),
             "AUPR_std": float(np.std([x["AUPR"] for x in runs]))}
        out[p].append(r)
        star = "  <- default" if getattr(default, p) == v else ""
        print(f"{p:10s} = {str(v):5s}  AUC {r['AUC']:.4f}±{r['AUC_std']:.4f}  "
              f"AUPR {r['AUPR']:.4f}±{r['AUPR_std']:.4f}{star}", flush=True)

d = RESULTS / data.name
d.mkdir(parents=True, exist_ok=True)
prev = json.loads((d / "sensitivity.json").read_text()) if (d / "sensitivity.json").exists() else {}
prev.update(out)
(d / "sensitivity.json").write_text(json.dumps(prev, indent=2))
