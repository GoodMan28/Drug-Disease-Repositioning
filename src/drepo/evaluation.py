"""
Evaluation protocols.

k-fold CV (warm start)
    Known links (1s) AND unknown pairs (0s) are each shuffled into k folds.
    Fold f's test set = its 1s + its 0s. The model trains with fold f's 1s
    hidden and may draw negatives only from the other folds' 0s.

Leave-one-disease-out (cold start)
    For each disease j, ALL of its known drugs are hidden. The model must rank
    every drug for a disease it has never seen linked to anything, using only
    similarity and gene information. Predictions for all diseases are pooled
    before computing AUC/AUPR (plus we report the mean per-disease AUC).

Metrics
    AUC  = area under ROC curve (probability a random positive outranks a random
           negative; 0.5 = random).
    AUPR = average precision (area under precision-recall). With ~1% positives
           this is the harder, more honest number; random = positive rate.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(y, s):
    return {"AUC": float(roc_auc_score(y, s)), "AUPR": float(average_precision_score(y, s))}


def kfold_splits(A, k, seed):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(A.ravel() > 0)
    neg = np.flatnonzero(A.ravel() == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pf, nf = np.array_split(pos, k), np.array_split(neg, k)
    for f in range(k):
        yield f, pf[f], nf[f]


def run_kfold(method, data, k=5, repeats=1, seed=0, verbose=True):
    A = data.A
    shape = A.shape
    folds, pooled_y, pooled_s = [], [], []
    t0 = time.time()
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
            m.update(repeat=r, fold=f)
            folds.append(m)
            if r == 0:                  # keep one repeat for ROC/PR curves
                pooled_y.append(y)
                pooled_s.append(s)
            if verbose:
                print(f"    rep {r} fold {f}: AUC {m['AUC']:.4f}  AUPR {m['AUPR']:.4f}"
                      f"  ({time.time() - t0:.0f}s)")
    return summarise(folds, np.concatenate(pooled_y), np.concatenate(pooled_s))


def run_lodo(method, data, diseases=None, seed=0, verbose=True):
    A = data.A
    diseases = range(A.shape[1]) if diseases is None else diseases
    ys, ss, per = [], [], []
    t0 = time.time()
    for n, j in enumerate(diseases):
        A_tr = A.copy()
        A_tr[:, j] = 0
        neg_mask = A == 0
        neg_mask[:, j] = False
        S = method.fit_predict(data, A_tr, neg_mask, seed=int(seed + j))
        y, s = A[:, j], S[:, j]
        ys.append(y)
        ss.append(s)
        if 0 < y.sum() < len(y):
            per.append(roc_auc_score(y, s))
        if verbose and (n + 1) % 25 == 0:
            print(f"    {n + 1}/{len(diseases)} diseases  running AUC "
                  f"{metrics(np.concatenate(ys), np.concatenate(ss))['AUC']:.4f}"
                  f"  ({time.time() - t0:.0f}s)")
    y, s = np.concatenate(ys), np.concatenate(ss)
    res = metrics(y, s)
    res["mean_per_disease_AUC"] = float(np.mean(per))
    res["n_diseases"] = len(per)
    return {"summary": res, "folds": [], "y": y, "s": s}


def summarise(folds, y, s):
    auc = np.array([f["AUC"] for f in folds])
    aupr = np.array([f["AUPR"] for f in folds])
    return {"summary": {"AUC": float(auc.mean()), "AUC_std": float(auc.std()),
                        "AUPR": float(aupr.mean()), "AUPR_std": float(aupr.std()),
                        "n_runs": len(folds)},
            "folds": folds, "y": y, "s": s}
