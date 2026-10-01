"""
Step 1a - Download every public resource the project needs.

Run:  python scripts/01_download_data.py

Everything lands in data/raw/. Files that already exist are skipped, so the
script is safe to re-run. Nothing here needs a login or licence key.

What we fetch and why
---------------------
benchmarks/Fdataset.mat, Cdataset.mat
    The same matrices as your DiDrA/DrugSim/DiseaseSim CSVs, *plus* the entity
    IDs (DrugBank IDs for drugs, OMIM IDs for diseases). Without the IDs we
    could not map to any other database or report drug names in case studies.
    Source: github.com/TheWall9/DRHGCN (mirrors the BNNR / Gottlieb / Luo data).

ctd/CTD_chemicals.tsv.gz          chemical vocabulary (MeSH IDs, names, synonyms, CAS)
ctd/CTD_diseases.tsv.gz           MEDIC disease vocabulary = MeSH tree + OMIM terms
ctd/CTD_chem_gene_ixns.tsv.gz     curated chemical -> gene interactions
ctd/CTD_curated_genes_diseases.tsv.gz  curated gene -> disease links
ctd/CTD_chemicals_diseases.tsv.gz curated + inferred chemical -> disease links
    (used only for case-study validation, never for training)

ontology/doid.obo                 Human Disease Ontology (fallback OMIM mapping)

DrugBank itself is NOT downloaded: the full XML needs an approved academic
account and was unavailable. We replace what we needed from it (structures)
with PubChem, which hosts DrugBank's own deposited records - see
scripts/02_build_features.py.
"""
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

DRHGCN = "https://raw.githubusercontent.com/TheWall9/DRHGCN/master/dataset/"
CTD = "https://ctdbase.org/reports/"

FILES = {
    "benchmarks/Fdataset.mat": DRHGCN + "Fdataset.mat",
    "benchmarks/Cdataset.mat": DRHGCN + "Cdataset.mat",
    "ctd/CTD_chemicals.tsv.gz": CTD + "CTD_chemicals.tsv.gz",
    "ctd/CTD_diseases.tsv.gz": CTD + "CTD_diseases.tsv.gz",
    "ctd/CTD_chem_gene_ixns.tsv.gz": CTD + "CTD_chem_gene_ixns.tsv.gz",
    "ctd/CTD_curated_genes_diseases.tsv.gz": CTD + "CTD_curated_genes_diseases.tsv.gz",
    "ctd/CTD_chemicals_diseases.tsv.gz": CTD + "CTD_chemicals_diseases.tsv.gz",
    "ontology/doid.obo": "https://raw.githubusercontent.com/DiseaseOntology/"
                         "HumanDiseaseOntology/main/src/ontology/doid.obo",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.relative_to(ROOT)} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "drug-repositioning-research/1.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r       {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                    sys.stdout.flush()
    print()
    tmp.replace(dest)


if __name__ == "__main__":
    for rel, url in FILES.items():
        download(url, RAW / rel)
    print("\nAll raw files are in", RAW)
