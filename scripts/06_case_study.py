"""
Step 7 - Case studies (Results 5.6) with interpretability.

The model is trained on ALL known links of a dataset (averaged over several
seeds for stability). For each chosen disease we list the top-N drugs that are
NOT already known indications and check each one against independent evidence:

  * CTD curated chemical-disease link ('therapeutic' or 'marker/mechanism')
  * a known link in the OTHER benchmark dataset (F <-> C)
  * number of ClinicalTrials.gov studies testing that drug in that disease
  * which evidence source drove the prediction (view attention + occlusion)

    python scripts/06_case_study.py --dataset C --omim 104300,114480,176807
    python scripts/06_case_study.py --dataset C --search "alzheimer"   # find OMIM IDs
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                                # noqa: E402
from drepo.methods import MVHGATMethod                     # noqa: E402
from drepo.paths import INTERIM, RESULTS                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="C")
ap.add_argument("--omim", default="104300,114480,176807",
                help="comma-separated OMIM IDs (default: Alzheimer's, breast cancer, prostate cancer)")
ap.add_argument("--search", default=None, help="print diseases whose name matches and exit")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--no-trials", action="store_true", help="skip ClinicalTrials.gov lookups")
args = ap.parse_args()

data = D.load(args.dataset)
if args.search:
    for o, n in zip(data.disease_ids, data.disease_names):
        if args.search.lower() in n.lower():
            print(o, n, int(data.A[:, list(data.disease_ids).index(o)].sum()), "known drugs")
    sys.exit()

other = D.load("F" if data.name == "Cdataset" else "C")
other_links = {(r, d) for r, d in zip(*[other.drug_ids[np.nonzero(other.A)[0]],
                                        other.disease_ids[np.nonzero(other.A)[1]]])}
ctd = pd.read_csv(INTERIM / "ctd_curated_chem_disease.csv", dtype=str).fillna("")
dis_tab = pd.read_csv(INTERIM / "diseases_all.csv", dtype=str).fillna("").set_index("omim")


def ctd_evidence(db, omim):
    ids = set(filter(None, dis_tab.loc[omim, "ctd_ids"].split("|"))) if omim in dis_tab.index else set()
    rows = ctd[ctd.drugbank_id.str.contains(db) &
               (ctd.DiseaseID.isin(ids) | ctd.OmimIDs.str.split("|").apply(lambda x: omim in x))]
    return "|".join(sorted(set(rows.DirectEvidence))) or "-"


def trial_condition(name):
    """'Alzheimer disease type 1' -> 'Alzheimer disease' (trials use plain names)."""
    name = name.split(",")[0]
    name = re.sub(r"\s+(type\s+\d+\w*|susceptibility to)\b.*$", "", name, flags=re.I)
    return re.sub(r"\s+\d+[A-Z]?$", "", name.strip()).strip()


def trials(drug, disease):
    if args.no_trials or not drug or not disease or drug.startswith("DB"):
        return None
    disease = trial_condition(disease)
    try:
        r = requests.get("https://clinicaltrials.gov/api/v2/studies",
                         params={"query.cond": disease, "query.intr": drug,
                                 "countTotal": "true", "pageSize": 1}, timeout=30)
        time.sleep(0.3)
        return r.json().get("totalCount")
    except Exception:
        return None


# ---- train on everything, average several seeds ---------------------------
print(f"Training MV-HGAT on all {int(data.A.sum())} links of {data.name} x {args.seeds} seeds")
scores, betas, occl, vweights = [], [], [], []
omims = args.omim.split(",")
cols = [list(data.disease_ids).index(o) for o in omims if o in set(data.disease_ids)]
missing = [o for o in omims if o not in set(data.disease_ids)]
if missing:
    print("not in this dataset:", missing, "(use --search to find IDs)")
for s in range(args.seeds):
    m = MVHGATMethod()
    scores.append(m.fit_predict(data, data.A, data.A == 0, seed=s))
    betas.append(m.view_attention())
    vweights.append(m.view_weights())
    pairs = [(i, j) for j in cols for i in range(data.n_drugs)]
    occl.append(m.occlusion(pairs))
S = np.mean(scores, 0)
occ = {g: np.mean([o[g] for o in occl], 0).reshape(len(cols), data.n_drugs) for g in occl[0]}
beta_r = np.mean([b["drug"][1] for b in betas], 0)
beta_d = np.mean([b["disease"][1] for b in betas], 0)
rel_r, rel_d = betas[0]["drug"][0], betas[0]["disease"][0]

out_dir = RESULTS / "case_studies"
out_dir.mkdir(parents=True, exist_ok=True)

# global attention summary (which views does the model lean on overall?)
pd.DataFrame({"relation": rel_r, "mean_beta": beta_r.mean(0)}).to_csv(
    out_dir / f"{data.name}_drug_view_attention.csv", index=False)
pd.DataFrame({"relation": rel_d, "mean_beta": beta_d.mean(0)}).to_csv(
    out_dir / f"{data.name}_disease_view_attention.csv", index=False)

md = [f"# Case studies - {data.name}\n"]
for c, j in enumerate(cols):
    omim, dname = data.disease_ids[j], data.disease_names[j]
    known = np.flatnonzero(data.A[:, j])
    order = [i for i in np.argsort(-S[:, j]) if data.A[i, j] == 0][: args.top]
    rows = []
    for rank, i in enumerate(order, 1):
        db, name = data.drug_ids[i], data.drug_names[i]
        drivers = sorted(occ, key=lambda g: -occ[g][c, i])
        rows.append({
            "rank": rank, "drugbank_id": db, "drug": name, "score": round(float(S[i, j]), 4),
            "CTD_curated": ctd_evidence(db, omim),
            f"known_in_{other.name}": "yes" if (db, omim) in other_links else "-",
            "clinical_trials": trials(name, dname),
            "top_evidence": ", ".join(f"{g.replace('view:', '')} ({occ[g][c, i]:+.3f})"
                                      for g in drivers[:2]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{data.name}_OMIM{omim}.csv", index=False)
    top_views = ", ".join(f"{r.replace('view:', '')} {b:.2f}"
                          for r, b in sorted(zip(rel_d, beta_d[j]), key=lambda x: -x[1])[:3])
    print(f"\n=== {dname} (OMIM {omim}) - {len(known)} known drugs: "
          f"{', '.join(data.drug_names[known][:8])}{' ...' if len(known) > 8 else ''}")
    print(f"    disease view attention: {top_views}")
    print(df.to_string(index=False))
    confirmed = (df.CTD_curated != "-") | (df[f"known_in_{other.name}"] == "yes") | \
                (df.clinical_trials.fillna(0).astype(float) > 0)
    print(f"    {confirmed.sum()}/{len(df)} of the top {args.top} have independent support")
    md += [f"\n## {dname} (OMIM {omim})\n",
           f"Known drugs ({len(known)}): {', '.join(data.drug_names[known])}\n",
           f"Disease view attention: {top_views}\n",
           df.to_markdown(index=False), "\n"]
(out_dir / f"{data.name}_case_studies.md").write_text("\n".join(md), encoding="utf8")
print("\nSaved to", out_dir)
