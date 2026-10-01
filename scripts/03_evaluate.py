"""
Step 5 + 6 - Evaluate the proposed model and the baselines (Results 5.1, 5.2, 5.3).

Examples
    python scripts/03_evaluate.py --dataset F --protocol cv5  --repeats 5
    python scripts/03_evaluate.py --dataset C --protocol cv10 --repeats 2
    python scripts/03_evaluate.py --dataset F --protocol lodo
    python scripts/03_evaluate.py --dataset F --protocol lodo --lodo-subset 60 --methods mvhgat,mbirw

Results: results/<Dataset>/<protocol>/<method>.json   (+ _preds.npz for curves)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo import data as D                                     # noqa: E402
from drepo.methods import METHODS, make                          # noqa: E402
from drepo.runner import run_protocol, save, load_table, fmt     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="F")
ap.add_argument("--protocol", default="cv5", help="cv5 | cv10 | lodo")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--methods", default=",".join(METHODS))
ap.add_argument("--lodo-subset", type=int, default=None,
                help="evaluate LODO on a random subset of diseases (faster)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

data = D.load(args.dataset)
print(f"{data.name}: {data.n_drugs} drugs, {data.n_diseases} diseases, "
      f"{int(data.A.sum())} known links ({data.A.mean():.2%} density)")

for key in args.methods.split(","):
    m = make(key)
    print(f"\n### {m.name}  [{args.protocol}]")
    res = run_protocol(m, data, args.protocol, args.repeats, args.seed, args.lodo_subset,
                       verbose=True)
    save(res, data.name, args.protocol, key, {"label": m.name})
    print(f"==> {m.name}: {fmt(res['summary'])}")

print("\n", load_table(data.name, args.protocol).to_string(index=False, float_format="%.4f"))
