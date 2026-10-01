"""Shared helpers for the experiment scripts: run a protocol, save, tabulate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .evaluation import run_kfold, run_lodo
from .paths import RESULTS


def run_protocol(method, data, protocol, repeats=1, seed=0, lodo_subset=None, verbose=True):
    if protocol.startswith("cv"):
        return run_kfold(method, data, k=int(protocol[2:]), repeats=repeats, seed=seed,
                         verbose=verbose)
    if protocol == "lodo":
        diseases = None
        if lodo_subset:
            rng = np.random.default_rng(seed)
            diseases = sorted(int(j) for j in rng.choice(data.n_diseases, lodo_subset, replace=False))
        return run_lodo(method, data, diseases, seed=seed, verbose=verbose)
    raise ValueError(protocol)


def save(res, dataset, protocol, key, extra=None):
    d = RESULTS / dataset / protocol
    d.mkdir(parents=True, exist_ok=True)
    out = {"summary": res["summary"], "folds": res["folds"], **(extra or {})}
    (d / f"{key}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(d / f"{key}_preds.npz", y=res["y"], s=res["s"])


def load_table(dataset, protocol):
    rows = []
    for f in sorted((RESULTS / dataset / protocol).glob("*.json")):
        j = json.loads(f.read_text())
        rows.append({"method": j.get("label", f.stem), **j["summary"]})
    return pd.DataFrame(rows)


def fmt(summary):
    if "AUC_std" in summary:
        return (f"AUC {summary['AUC']:.4f} ± {summary['AUC_std']:.4f} | "
                f"AUPR {summary['AUPR']:.4f} ± {summary['AUPR_std']:.4f}")
    return f"AUC {summary['AUC']:.4f} | AUPR {summary['AUPR']:.4f} | " \
           f"mean per-disease AUC {summary['mean_per_disease_AUC']:.4f}"
