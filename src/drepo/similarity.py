"""
Similarity measures used to build the drug and disease "views".

Every function returns an (n x n) matrix with 1 on the diagonal and values in
[0, 1]. Rows for entities we could not describe (no structure, no genes, not in
the ontology) are filled with NaN so the caller can decide how to treat them.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Chemical structure: Morgan / ECFP4 fingerprints + Tanimoto                   #
# --------------------------------------------------------------------------- #
def morgan_tanimoto(smiles: list[str | None], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    largest = rdMolStandardize.LargestFragmentChooser()

    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if mol is not None:
            mol = largest.choose(mol)   # drop counter-ions such as HCl
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)

    n = len(fps)
    S = np.full((n, n), np.nan)
    ok = [i for i, f in enumerate(fps) if f is not None]
    ok_fps = [fps[i] for i in ok]
    for a, i in enumerate(ok):
        sims = DataStructs.BulkTanimotoSimilarity(ok_fps[a], ok_fps)
        S[i, ok] = sims
    np.fill_diagonal(S, 1.0)
    return S


# --------------------------------------------------------------------------- #
# Gene profiles: Jaccard and cosine                                            #
# --------------------------------------------------------------------------- #
def jaccard(G: np.ndarray) -> np.ndarray:
    """G is a binary (entities x genes) matrix. Rows with no genes -> NaN."""
    G = (G > 0).astype(np.float64)
    inter = G @ G.T
    size = G.sum(1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        S = inter / union
    empty = size == 0
    S[empty, :] = np.nan
    S[:, empty] = np.nan
    np.fill_diagonal(S, 1.0)
    return S


def cosine_cross(G1: np.ndarray, G2: np.ndarray) -> np.ndarray:
    """Cosine between rows of two binary matrices over the same gene columns.

    This is the degree-normalised count of drug -> gene -> disease paths, i.e.
    the 'gene bridge' meta-path of the tripartite graph.
    """
    G1 = (G1 > 0).astype(np.float64)
    G2 = (G2 > 0).astype(np.float64)
    n1 = np.sqrt(G1.sum(1, keepdims=True))
    n2 = np.sqrt(G2.sum(1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        M = (G1 @ G2.T) / (n1 * n2.T)
    return np.nan_to_num(M)


# --------------------------------------------------------------------------- #
# Ontology: Wang et al. (2007) semantic similarity on a DAG                    #
# --------------------------------------------------------------------------- #
def wang_s_values(term: str, parents: dict[str, list[str]], w: float = 0.5) -> dict[str, float]:
    """Semantic contribution of every ancestor of `term` (including itself).

    S(term) = 1; S(parent) = max over children c in the sub-DAG of w * S(c).
    Closer ancestors contribute more; the root contributes very little.
    """
    S = {term: 1.0}
    stack = [term]
    while stack:
        t = stack.pop()
        v = S[t] * w
        for p in parents.get(t, ()):
            if v > S.get(p, 0.0):
                S[p] = v
                stack.append(p)
    return S


def wang_similarity(term_sets: list[list[str]], parents: dict[str, list[str]], w: float = 0.5) -> np.ndarray:
    """Pairwise similarity of entities, each mapped to one or more ontology terms.

    If an entity maps to several terms we take the maximum pairwise similarity.
    """
    cache: dict[str, dict[str, float]] = {}

    def sv(t):
        if t not in cache:
            cache[t] = wang_s_values(t, parents, w)
        return cache[t]

    def term_sim(a, b):
        if a == b:
            return 1.0
        Sa, Sb = sv(a), sv(b)
        common = Sa.keys() & Sb.keys()
        if not common:
            return 0.0
        num = sum(Sa[t] + Sb[t] for t in common)
        return num / (sum(Sa.values()) + sum(Sb.values()))

    n = len(term_sets)
    S = np.full((n, n), np.nan)
    for i in range(n):
        if not term_sets[i]:
            continue
        for j in range(i, n):
            if not term_sets[j]:
                continue
            s = max(term_sim(a, b) for a in term_sets[i] for b in term_sets[j])
            S[i, j] = S[j, i] = s
    np.fill_diagonal(S, 1.0)
    return S
