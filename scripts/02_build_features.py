"""
Step 1b + Step 2 - Entity resolution and similarity-network construction.

Run:  python scripts/02_build_features.py

Inputs : data/raw/**           (from 01_download_data.py)
Outputs: data/interim/*.csv    (readable mapping tables - open them in Excel)
         data/processed/Fdataset.npz, Cdataset.npz   (everything the models need)

For each benchmark we end up with SIX similarity "views" plus a gene bridge:

  drug views                          disease views
  ----------                          -------------
  chem_cdk   benchmark matrix         pheno_mim   benchmark matrix
             (CDK fingerprints,                    (MimMiner text-mined
              Tanimoto; from paper)                 phenotype similarity)
  chem_ecfp  RDKit Morgan/ECFP4 on    sem_mondo   Wang semantic similarity
             PubChem SMILES                        on the MONDO disease DAG
  gene_r     Jaccard of CTD           gene_d      Jaccard of CTD curated
             chemical-gene sets                    gene-disease sets

  gene_bridge  drug x disease cosine of gene profiles
               (= drug -> gene -> disease meta-path of the tripartite graph)
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drepo.paths import RAW, INTERIM, PROCESSED, DATASETS          # noqa: E402
from drepo import similarity as sim                                # noqa: E402

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# =========================================================================== #
# 1. Benchmarks                                                               #
# =========================================================================== #
def load_benchmark(name):
    m = sio.loadmat(RAW / "benchmarks" / f"{name}.mat")
    drugs = [str(x[0]) for x in m["Wrname"].ravel()]
    # 'D102100' -> OMIM 102100 (the 'D' just marks it as a disease)
    omims = [str(x[0])[1:] for x in m["Wdname"].ravel()]
    A = m["didr"].T.astype(np.float32)          # -> drugs x diseases
    return drugs, omims, A, m["drug"].astype(np.float64), m["disease"].astype(np.float64)


# =========================================================================== #
# 2. Drugs: DrugBank ID -> PubChem (structure + name)                          #
# =========================================================================== #
def _get(url, retries=4):
    for k in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 503:          # PubChem throttling
                time.sleep(2 + 2 * k)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            time.sleep(1 + k)
    return None


def pubchem_drugs(drugbank_ids):
    """DrugBank deposits its records in PubChem as 'substances', so we can look
    a DrugBank ID up directly without needing the DrugBank download."""
    cache = INTERIM / "pubchem_drugs.csv"
    done = pd.read_csv(cache, dtype=str).fillna("") if cache.exists() else pd.DataFrame(
        columns=["drugbank_id", "cid", "name", "smiles", "inchikey"])
    have = set(done.drugbank_id)
    todo = [d for d in drugbank_ids if d not in have]
    rows = []
    for k, db in enumerate(todo, 1):
        txt = _get(f"{PUG}/substance/sourceid/DrugBank/{db}/cids/TXT")
        cid = txt.split()[0] if txt else ""
        name = ""
        if not cid:   # biologics: no compound, but the substance still has a name
            syn = _get(f"{PUG}/substance/sourceid/DrugBank/{db}/synonyms/TXT")
            name = syn.splitlines()[0].strip() if syn else ""
        rows.append({"drugbank_id": db, "cid": cid, "name": name, "smiles": "", "inchikey": ""})
        time.sleep(0.21)                      # stay under PubChem's 5 req/s limit
        if k % 50 == 0:
            print(f"   PubChem lookup {k}/{len(todo)}")
    new = pd.DataFrame(rows)

    # batch-fetch structures for all CIDs
    if len(new):
        cids = [c for c in new.cid if c]
        props = {}
        for i in range(0, len(cids), 100):
            chunk = ",".join(cids[i:i + 100])
            txt = _get(f"{PUG}/compound/cid/{chunk}/property/SMILES,InChIKey,Title/CSV")
            if txt:
                p = pd.read_csv(pd.io.common.StringIO(txt), dtype=str).fillna("")
                for r in p.itertuples():
                    props[r.CID] = (r.SMILES, r.InChIKey, r.Title)
        for i, r in new.iterrows():
            if r.cid in props:
                s, ik, t = props[r.cid]
                new.loc[i, ["smiles", "inchikey"]] = [s, ik]
                if not r["name"]:
                    new.loc[i, "name"] = t
    out = pd.concat([done, new], ignore_index=True)
    # PubChem compound titles are sometimes odd ("med.21724, Compound X") or
    # missing; DrugBank's own deposited substance lists the DrugBank name first.
    odd = out.name.fillna("").str.contains(r"[,;]|^\w+\.\d|^$", regex=True)
    for i in out.index[odd]:
        syn = _get(f"{PUG}/substance/sourceid/DrugBank/{out.loc[i, 'drugbank_id']}/synonyms/TXT")
        if syn and syn.strip():
            n = syn.splitlines()[0].strip()
            out.loc[i, "name"] = n[:1].upper() + n[1:]
        time.sleep(0.21)
    out.to_csv(cache, index=False)
    return out.set_index("drugbank_id")


# =========================================================================== #
# 3. Diseases: OMIM -> name, MONDO terms, MeSH / CTD terms                    #
# =========================================================================== #
def parse_mondo():
    terms, cur = {}, None
    for line in open(RAW / "ontology" / "mondo.obo", encoding="utf8"):
        line = line.rstrip("\n")
        if line.startswith("["):
            cur = None
            if line == "[Term]":
                cur = {"id": None, "name": "", "is_a": [], "xrefs": [], "obsolete": False}
        elif cur is not None:
            if line.startswith("id: "):
                cur["id"] = line[4:]
                terms[cur["id"]] = cur
            elif line.startswith("name: "):
                cur["name"] = line[6:]
            elif line.startswith("is_a: "):
                cur["is_a"].append(line[6:].split(" ")[0])
            elif line.startswith("xref: "):
                cur["xrefs"].append(line[6:].split(" ")[0])
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
    terms = {k: v for k, v in terms.items() if k.startswith("MONDO:") and not v["obsolete"]}
    parents = {k: [p for p in v["is_a"] if p in terms] for k, v in terms.items()}
    omim2mondo, mondo2mesh = defaultdict(list), defaultdict(list)
    for k, v in terms.items():
        for x in v["xrefs"]:
            if x.startswith("OMIM:"):
                omim2mondo[x[5:]].append(k)
            elif x.startswith("MESH:"):
                mondo2mesh[k].append(x)
    return terms, parents, omim2mondo, mondo2mesh


CTD_DISEASE_COLS = ["DiseaseName", "DiseaseID", "AltDiseaseIDs", "Definition", "ParentIDs",
                    "TreeNumbers", "ParentTreeNumbers", "Synonyms", "SlimMappings"]


def disease_table(omims):
    terms, parents, omim2mondo, mondo2mesh = parse_mondo()

    mg = pd.read_csv(RAW / "medgen" / "MedGenIDMappings.txt.gz", sep="|", dtype=str,
                     usecols=[0, 1, 2, 3])
    mg.columns = ["cui", "name", "src_id", "src"]
    mg_omim = mg[mg.src == "OMIM"].drop_duplicates("src_id").set_index("src_id")
    mg_mesh = mg[mg.src == "MeSH"].groupby("cui").src_id.apply(list).to_dict()

    medic = pd.read_csv(RAW / "ctd" / "CTD_diseases.tsv.gz", sep="\t", comment="#",
                        names=CTD_DISEASE_COLS, dtype=str).fillna("")
    medic_ids = set(medic.DiseaseID)
    omim2ctd = defaultdict(list)
    for r in medic.itertuples():
        for i in [r.DiseaseID] + r.AltDiseaseIDs.split("|"):
            if i.startswith("OMIM:"):
                omim2ctd[i[5:]].append(r.DiseaseID)

    rows = []
    for o in omims:
        mondo = omim2mondo.get(o, [])
        mesh = set()
        for m in mondo:
            mesh.update(mondo2mesh.get(m, []))
        if o in mg_omim.index:
            mesh.update("MESH:" + x for x in mg_mesh.get(mg_omim.loc[o, "cui"], []))
        ctd = set(omim2ctd.get(o, [])) | {m for m in mesh if m in medic_ids}
        name = mg_omim.loc[o, "name"] if o in mg_omim.index else (
            terms[mondo[0]]["name"] if mondo else "")
        rows.append({"omim": o, "name": name, "mondo": "|".join(mondo),
                     "ctd_ids": "|".join(sorted(ctd))})
    return pd.DataFrame(rows), parents


# =========================================================================== #
# 4. CTD gene profiles                                                         #
# =========================================================================== #
def ctd_chemical_map(drugs_pc):
    """DrugBank -> CTD chemical (MeSH) via PubChem CID, then InChIKey, then name."""
    cols = ["ChemicalName", "ChemicalID", "CasRN", "PubChemCID", "PubChemSID", "DTXSID",
            "InChIKey", "Definition", "ParentIDs", "TreeNumbers", "ParentTreeNumbers",
            "MESHSynonyms", "CTDCuratedSynonyms"]
    ch = pd.read_csv(RAW / "ctd" / "CTD_chemicals.tsv.gz", sep="\t", comment="#", names=cols,
                     dtype=str, usecols=["ChemicalName", "ChemicalID", "PubChemCID", "InChIKey",
                                         "MESHSynonyms"]).fillna("")
    ch["ChemicalID"] = ch.ChemicalID.str.replace("MESH:", "", regex=False)
    by_cid = {c: i for c, i in zip(ch.PubChemCID, ch.ChemicalID) if c}
    by_ik = {k: i for k, i in zip(ch.InChIKey, ch.ChemicalID) if k}
    by_name = {}
    for r in ch.itertuples():
        for n in [r.ChemicalName] + r.MESHSynonyms.split("|"):
            if n:
                by_name.setdefault(n.lower(), r.ChemicalID)
    out = {}
    for db, r in drugs_pc.iterrows():
        cid = by_cid.get(r.cid) or by_ik.get(r.inchikey) or by_name.get(str(r["name"]).lower())
        if cid:
            out[db] = cid
    return out


def ctd_drug_genes(chem_ids):
    wanted = set(chem_ids)
    genes = defaultdict(set)
    cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
            "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
    for chunk in pd.read_csv(RAW / "ctd" / "CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#",
                             names=cols, usecols=["ChemicalID", "GeneSymbol", "OrganismID"],
                             dtype=str, chunksize=500_000):
        chunk = chunk[(chunk.OrganismID == "9606") & chunk.ChemicalID.isin(wanted)]
        for c, g in zip(chunk.ChemicalID, chunk.GeneSymbol):
            genes[c].add(g)
    return genes


def ctd_disease_genes(dis):
    """Curated gene-disease links, *marker/mechanism* evidence only.
    ('therapeutic' gene-disease links are left out on purpose: they partly
    encode which drugs treat the disease, i.e. the very thing we predict.)"""
    cols = ["GeneSymbol", "GeneID", "DiseaseName", "DiseaseID", "DirectEvidence", "OmimIDs",
            "PubMedIDs"]
    gd = pd.read_csv(RAW / "ctd" / "CTD_curated_genes_diseases.tsv.gz", sep="\t", comment="#",
                     names=cols, dtype=str).fillna("")
    gd = gd[gd.DirectEvidence.str.contains("marker/mechanism")]
    by_ctd, by_omim = defaultdict(set), defaultdict(set)
    for r in gd.itertuples():
        by_ctd[r.DiseaseID].add(r.GeneSymbol)
        for o in r.OmimIDs.split("|"):
            if o:
                by_omim[o].add(r.GeneSymbol)
    out = {}
    for r in dis.itertuples():
        g = set(by_omim.get(r.omim, set()))
        for c in filter(None, r.ctd_ids.split("|")):
            g |= by_ctd.get(c, set())
        out[r.omim] = g
    return out


def ctd_curated_chem_disease(chem_ids):
    """Curated chemical-disease links for our drugs (validation only)."""
    wanted = set(chem_ids)
    cols = ["ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID", "DirectEvidence",
            "InferenceGeneSymbol", "InferenceScore", "OmimIDs", "PubMedIDs"]
    keep = []
    for chunk in pd.read_csv(RAW / "ctd" / "CTD_chemicals_diseases.tsv.gz", sep="\t",
                             comment="#", names=cols, dtype=str, chunksize=1_000_000,
                             usecols=["ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID",
                                      "DirectEvidence", "OmimIDs", "PubMedIDs"]):
        chunk = chunk[chunk.DirectEvidence.notna() & chunk.ChemicalID.isin(wanted)]
        keep.append(chunk)
    return pd.concat(keep, ignore_index=True).fillna("")


def gene_matrix(entity_genes, entities, vocab):
    idx = {g: i for i, g in enumerate(vocab)}
    G = np.zeros((len(entities), len(vocab)), dtype=np.float32)
    for r, e in enumerate(entities):
        for g in entity_genes.get(e, ()):
            if g in idx:
                G[r, idx[g]] = 1
    return G


# =========================================================================== #
# main                                                                        #
# =========================================================================== #
def main():
    bench = {k: load_benchmark(n) for k, n in DATASETS.items()}
    all_drugs = sorted({d for b in bench.values() for d in b[0]})
    all_omims = sorted({o for b in bench.values() for o in b[1]})
    print(f"{len(all_drugs)} unique drugs, {len(all_omims)} unique diseases across F + C")

    print("-> PubChem (structures, names) ...")
    pc = pubchem_drugs(all_drugs)

    print("-> Diseases (MedGen names, MONDO terms, CTD/MeSH terms) ...")
    dis_all, mondo_parents = disease_table(all_omims)

    print("-> CTD chemical mapping + gene profiles ...")
    db2ctd = ctd_chemical_map(pc.loc[all_drugs])
    drug_genes_by_chem = ctd_drug_genes(db2ctd.values())
    drug_genes = {db: drug_genes_by_chem.get(c, set()) for db, c in db2ctd.items()}
    dis_genes = ctd_disease_genes(dis_all)

    print("-> CTD curated chemical-disease links (validation table) ...")
    cd = ctd_curated_chem_disease(db2ctd.values())
    ctd2db = defaultdict(list)
    for db, c in db2ctd.items():
        ctd2db[c].append(db)
    cd["drugbank_id"] = cd.ChemicalID.map(lambda c: "|".join(ctd2db.get(c, [])))
    cd.to_csv(INTERIM / "ctd_curated_chem_disease.csv", index=False)

    drug_tab = pc.loc[all_drugs].copy()
    drug_tab["ctd_chemical"] = [db2ctd.get(d, "") for d in all_drugs]
    drug_tab["n_genes"] = [len(drug_genes.get(d, ())) for d in all_drugs]
    drug_tab.to_csv(INTERIM / "drugs_all.csv")
    dis_all["n_genes"] = [len(dis_genes.get(o, ())) for o in dis_all.omim]
    dis_all.to_csv(INTERIM / "diseases_all.csv", index=False)

    for key, name in DATASETS.items():
        drugs, omims, A, S_cdk, S_mim = bench[key]
        dt = drug_tab.loc[drugs]
        ds = dis_all.set_index("omim").loc[omims]
        print(f"\n== {name}: {len(drugs)} drugs x {len(omims)} diseases, {int(A.sum())} links")

        S_ecfp = sim.morgan_tanimoto(dt.smiles.tolist())
        S_sem = sim.wang_similarity([m.split("|") if m else [] for m in ds.mondo], mondo_parents)

        # restrict the gene vocabulary to genes that touch >= 2 entities of this
        # dataset (a gene seen once cannot create similarity or a bridge)
        cnt = defaultdict(int)
        for d in drugs:
            for g in drug_genes.get(d, ()):
                cnt[g] += 1
        for o in omims:
            for g in dis_genes.get(o, ()):
                cnt[g] += 1
        vocab = sorted(g for g, c in cnt.items() if c >= 2)
        Gr = gene_matrix(drug_genes, drugs, vocab)
        Gd = gene_matrix(dis_genes, omims, vocab)
        S_gr, S_gd = sim.jaccard(Gr), sim.jaccard(Gd)
        bridge = sim.cosine_cross(Gr, Gd)

        views_r = {"chem_cdk": S_cdk, "chem_ecfp": S_ecfp, "gene_r": S_gr}
        views_d = {"pheno_mim": S_mim, "sem_mondo": S_sem, "gene_d": S_gd}
        for vname, S in {**views_r, **views_d}.items():
            cov = (~np.isnan(S).all(1) & ~(np.nansum(S, 1) <= 1.0 + 1e-9)).mean()
            print(f"   view {vname:10s} coverage {cov:6.1%}")
        print(f"   gene vocab {len(vocab)}, bridge non-zero pairs {(bridge > 0).mean():.1%}")

        np.savez_compressed(
            PROCESSED / f"{name}.npz",
            A=A, drug_ids=np.array(drugs), disease_ids=np.array(omims),
            drug_names=np.array([n if n else d for n, d in zip(dt["name"].fillna(""), drugs)],
                                dtype=str),
            disease_names=np.array([n if n else f"OMIM {o}"
                                    for n, o in zip(ds["name"].fillna(""), omims)], dtype=str),
            drug_view_names=np.array(list(views_r)), disease_view_names=np.array(list(views_d)),
            drug_views=np.stack(list(views_r.values())),
            disease_views=np.stack(list(views_d.values())),
            gene_bridge=bridge, gene_vocab=np.array(vocab),
        )
    print("\nSaved processed datasets to", PROCESSED)


if __name__ == "__main__":
    main()
