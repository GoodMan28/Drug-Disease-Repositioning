"""
Figures and tables for the Results section (5.1 - 5.5, 5.7).

    python scripts/08_make_figures.py

Writes results/figures/*.png and results/RESULTS.md (all tables in one place).
Colours follow a fixed, colour-blind-validated order; each method keeps the
same colour in every figure.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo.paths import RESULTS                                    # noqa: E402

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHOD_ORDER = ["mvhgat", "mbirw", "drrs", "scmfdd", "nimcgcn", "lagcn"]
COLORS = dict(zip(METHOD_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                 "#008300"]))
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})
md = ["# Results\n", "All numbers are produced by the scripts in `scripts/`. "
      "CV rows: mean ± std over all folds of all repeats. LODO: pooled predictions.\n"]


def label_of(p):
    return json.loads(p.read_text()).get("label", p.stem)


# ---- 5.1 / 5.2  tables ------------------------------------------------------
for ds in ("Fdataset", "Cdataset"):
    for proto in ("cv5", "cv10", "lodo"):
        d = RESULTS / ds / proto
        if not d.exists():
            continue
        rows = []
        for key in METHOD_ORDER:
            f = d / f"{key}.json"
            if f.exists():
                s = json.loads(f.read_text())["summary"]
                row = {"Method": label_of(f)}
                if "AUC_std" in s:
                    row["AUC"] = f"{s['AUC']:.4f} ± {s['AUC_std']:.4f}"
                    row["AUPR"] = f"{s['AUPR']:.4f} ± {s['AUPR_std']:.4f}"
                else:
                    row.update(AUC=f"{s['AUC']:.4f}", AUPR=f"{s['AUPR']:.4f}",
                               **{"mean per-disease AUC": f"{s['mean_per_disease_AUC']:.4f}",
                                  "diseases": s["n_diseases"]})
                rows.append(row)
        if rows:
            md += [f"\n## {ds} - {proto}\n", pd.DataFrame(rows).to_markdown(index=False), ""]

# ---- 5.3  ROC + PR curves ---------------------------------------------------
for ds in ("Fdataset", "Cdataset"):
    for proto in ("cv5", "cv10", "lodo"):
        d = RESULTS / ds / proto
        keys = [k for k in METHOD_ORDER if (d / f"{k}_preds.npz").exists()]
        if not keys:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
        for k in keys:
            z = np.load(d / f"{k}_preds.npz")
            y, s = z["y"], z["s"]
            fpr, tpr, _ = roc_curve(y, s)
            pr, rc, _ = precision_recall_curve(y, s)
            lw, zo = (2.6, 5) if k == "mvhgat" else (1.6, 2)
            name = label_of(d / f"{k}.json")
            summ = json.loads((d / f"{k}.json").read_text())["summary"]
            a1.plot(fpr, tpr, color=COLORS[k], lw=lw, zorder=zo, label=f"{name} ({summ['AUC']:.3f})")
            a2.plot(rc, pr, color=COLORS[k], lw=lw, zorder=zo, label=f"{name} ({summ['AUPR']:.3f})")
        a1.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
        a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        a1.legend(loc="lower right", fontsize=8, title="AUC")
        a2.legend(loc="upper right", fontsize=8, title="AUPR")
        fig.suptitle(f"{ds}, {proto.upper()}", color=INK, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG / f"curves_{ds}_{proto}.png", dpi=200)
        plt.close(fig)

# ---- 5.4  ablation ----------------------------------------------------------
for ds in ("Fdataset", "Cdataset"):
    for d in sorted((RESULTS / ds).glob("ablation_*")):
        rows = []
        for f in sorted(d.glob("*.json")):
            s = json.loads(f.read_text())["summary"]
            rows.append({"key": f.stem, "Variant": label_of(f), **s})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        full = df[df.key == "full"]
        df = pd.concat([full, df[df.key != "full"].sort_values("AUPR", ascending=False)])
        tab = df[["Variant"]].copy()
        tab["AUC"] = [f"{a:.4f} ± {b:.4f}" if "AUC_std" in df else f"{a:.4f}"
                      for a, b in zip(df.AUC, df.get("AUC_std", df.AUC))]
        tab["AUPR"] = [f"{a:.4f} ± {b:.4f}" if "AUPR_std" in df else f"{a:.4f}"
                       for a, b in zip(df.AUPR, df.get("AUPR_std", df.AUPR))]
        md += [f"\n## Ablation - {ds} ({d.name.replace('ablation_', '')})\n",
               tab.to_markdown(index=False), ""]
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(df) + 1.2), sharey=True)
        ypos = np.arange(len(df))[::-1]
        for ax, metric in zip(axes, ["AUC", "AUPR"]):
            colors = ["#2a78d6" if k == "full" else "#a8a7a2" for k in df.key]
            ax.barh(ypos, df[metric], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2,
                    xerr=df.get(f"{metric}_std"), error_kw={"ecolor": INK2, "lw": 1})
            lo = max(0, df[metric].min() - 0.05)
            ax.set_xlim(lo, min(1, df[metric].max() + 0.02))
            ax.axvline(float(full[metric].iloc[0]), color="#2a78d6", lw=1, ls="--")
            ax.set_title(metric)
            ax.grid(axis="y", visible=False)
            for yv, v in zip(ypos, df[metric]):
                ax.text(v, yv, f" {v:.3f}", va="center", fontsize=8, color=INK2)
        axes[0].set_yticks(ypos, df.Variant)
        fig.suptitle(f"Ablation - {ds} (x-axis does not start at 0)", color=INK, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG / f"ablation_{ds}_{d.name}.png", dpi=200)
        plt.close(fig)

# ---- 5.5  sensitivity (small multiples, AUC and AUPR on separate rows) ------
for ds in ("Fdataset", "Cdataset"):
    f = RESULTS / ds / "sensitivity.json"
    if not f.exists():
        continue
    sens = json.loads(f.read_text())
    params = list(sens)
    fig, axes = plt.subplots(2, len(params), figsize=(3 * len(params), 5.2), squeeze=False)
    for c, p in enumerate(params):
        xs = [str(r["value"]) for r in sens[p]]
        for r_, metric in enumerate(["AUC", "AUPR"]):
            ax = axes[r_, c]
            m = np.array([r[metric] for r in sens[p]])
            sd = np.array([r[f"{metric}_std"] for r in sens[p]])
            ax.fill_between(range(len(xs)), m - sd, m + sd, color="#2a78d6", alpha=0.15, lw=0)
            ax.plot(range(len(xs)), m, color="#2a78d6", marker="o", ms=5)
            ax.set_xticks(range(len(xs)), xs)
            if r_ == 1:
                ax.set_xlabel(p)
            if c == 0:
                ax.set_ylabel(f"validation {metric}")
        md += [f"\n### Sensitivity - {ds} - {p}\n", pd.DataFrame(sens[p]).to_markdown(index=False)]
    fig.suptitle(f"Parameter sensitivity - {ds} (validation split, mean ± std over seeds)",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"sensitivity_{ds}.png", dpi=200)
    plt.close(fig)

# ---- view attention ---------------------------------------------------------
cs = RESULTS / "case_studies"
for ds in ("Fdataset", "Cdataset"):
    fr, fd = cs / f"{ds}_drug_view_attention.csv", cs / f"{ds}_disease_view_attention.csv"
    if not (fr.exists() and fd.exists()):
        continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, f, title in zip(axes, [fr, fd], ["Drug nodes", "Disease nodes"]):
        df = pd.read_csv(f).sort_values("mean_beta")
        names = df.relation.str.replace("view:", "", regex=False)
        ax.barh(names, df.mean_beta, color="#2a78d6", height=0.6)
        for yv, v in enumerate(df.mean_beta):
            ax.text(v, yv, f" {v:.2f}", va="center", fontsize=8, color=INK2)
        ax.set(title=title, xlabel="mean view-attention weight (beta)")
        ax.grid(axis="y", visible=False)
        md += [f"\n### View attention - {ds} - {title}\n", df.to_markdown(index=False)]
    fig.suptitle(f"Which evidence does MV-HGAT rely on? ({ds}, trained on all links)",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"view_attention_{ds}.png", dpi=200)
    plt.close(fig)

# ---- 5.7  cross-dataset -----------------------------------------------------
for f in sorted(RESULTS.glob("cross_*.json")):
    j = json.loads(f.read_text())
    df = pd.DataFrame(j["results"]).T.reset_index().rename(columns={"index": "Method"})
    md += [f"\n## Cross-dataset: {f.stem.replace('cross_', '').replace('_to_', ' -> ')}\n",
           f"{j['n_new_links']} new links among {j['n_candidates']} candidate pairs "
           f"(random AUPR = {j['n_new_links'] / j['n_candidates']:.4f})\n",
           df.to_markdown(index=False, floatfmt=".4f"), ""]

for f in sorted(cs.glob("*_case_studies.md")) if cs.exists() else []:
    md += ["\n", f.read_text(encoding="utf8")]

(RESULTS / "RESULTS.md").write_text("\n".join(md), encoding="utf8")
print("wrote", RESULTS / "RESULTS.md", "and", len(list(FIG.glob("*.png"))), "figures")
