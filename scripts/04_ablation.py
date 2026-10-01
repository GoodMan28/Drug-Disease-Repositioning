"""
Step 6b - Ablation study (Results 5.4).

Each variant switches off one part of MV-HGAT and is evaluated with the same
k-fold splits as the full model, so differences come from the component only.

    python scripts/04_ablation.py --dataset F --protocol cv5 --repeats 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                                     # noqa: E402
from drepo.methods import MVHGATMethod                           # noqa: E402
from drepo.runner import run_protocol, save, fmt, load_table     # noqa: E402

VARIANTS = {
    "full": ("Full MV-HGAT", {}),
    "no_prop_head": ("w/o multi-view propagation head", {"prop_head": False}),
    "no_degree_gate": ("w/o degree gate", {"degree_gate": False}),
    "no_cold_practice": ("w/o cold-start practice", {"cold_frac": 0.0}),
    "no_hidden_sup": ("w/o hidden-link supervision", {"supervise_hidden": False}),
    "no_attention": ("w/o view attention (plain mean)", {"uniform_attention": True}),
    "no_gene_views": ("w/o gene views", {"drug_views": ("chem_cdk", "chem_ecfp"),
                                         "disease_views": ("pheno_mim", "sem_mondo")}),
    "with_bridge": ("+ drug-gene-disease bridge", {"use_bridge": True}),
    "no_ecfp": ("w/o ECFP chemical view", {"drug_views": ("chem_cdk", "gene_r")}),
    "no_semantic": ("w/o MONDO semantic view", {"disease_views": ("pheno_mim", "gene_d")}),
    "bench_only": ("benchmark similarities only",
                   {"drug_views": ("chem_cdk",), "disease_views": ("pheno_mim",)}),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="F")
ap.add_argument("--protocol", default="cv5")
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--lodo-subset", type=int, default=None)
ap.add_argument("--variants", default=",".join(VARIANTS))
args = ap.parse_args()

data = D.load(args.dataset)
proto = f"ablation_{args.protocol}"
for key in args.variants.split(","):
    label, kw = VARIANTS[key]
    print(f"\n### {label}")
    res = run_protocol(MVHGATMethod(**kw), data, args.protocol, args.repeats,
                       lodo_subset=args.lodo_subset, verbose=False)
    save(res, data.name, proto, key, {"label": label})
    print(f"==> {fmt(res['summary'])}")

print("\n", load_table(data.name, proto).to_string(index=False, float_format="%.4f"))
