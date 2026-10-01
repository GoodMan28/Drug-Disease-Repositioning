"""Loading the processed benchmarks and turning similarity matrices into graphs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import PROCESSED, dataset_name


@dataclass
class DDData:
    name: str
    A: np.ndarray                 # drugs x diseases, 1 = known indication
    drug_ids: np.ndarray
    disease_ids: np.ndarray
    drug_names: np.ndarray
    disease_names: np.ndarray
    drug_view_names: list
    disease_view_names: list
    drug_views: np.ndarray        # (Vr, n_drugs, n_drugs), NaN rows = not covered
    disease_views: np.ndarray     # (Vd, n_dis, n_dis)
    gene_bridge: np.ndarray       # drugs x diseases cosine of gene profiles

    @property
    def n_drugs(self):
        return self.A.shape[0]

    @property
    def n_diseases(self):
        return self.A.shape[1]

    def drug_view(self, name):
        return self.drug_views[self.drug_view_names.index(name)]

    def disease_view(self, name):
        return self.disease_views[self.disease_view_names.index(name)]


def load(key: str) -> DDData:
    name = dataset_name(key)
    z = np.load(PROCESSED / f"{name}.npz", allow_pickle=False)
    return DDData(
        name=name, A=z["A"].astype(np.float32),
        drug_ids=z["drug_ids"], disease_ids=z["disease_ids"],
        drug_names=z["drug_names"], disease_names=z["disease_names"],
        drug_view_names=list(z["drug_view_names"]), disease_view_names=list(z["disease_view_names"]),
        drug_views=z["drug_views"], disease_views=z["disease_views"],
        gene_bridge=z["gene_bridge"],
    )


def fill_missing(S: np.ndarray) -> np.ndarray:
    """Uncovered entities get no neighbours in that view (only themselves)."""
    S = np.nan_to_num(S.copy(), nan=0.0)
    np.fill_diagonal(S, 1.0)
    return S


def knn_mask(S: np.ndarray, k: int, symmetric: bool = True) -> np.ndarray:
    """Keep each node's k most similar neighbours (+ itself).

    Dense similarity matrices connect everything to everything, which makes
    message passing blur all nodes together. A k-nearest-neighbour graph keeps
    only the informative, strongest links.
    """
    S = fill_missing(S)
    n = S.shape[0]
    work = S.copy()
    np.fill_diagonal(work, -np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(-work, k, axis=1)[:, :k]
    M = np.zeros_like(S, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    vals = work[rows, idx.ravel()]
    keep = vals > 0                         # never link on zero similarity
    M[rows[keep], idx.ravel()[keep]] = True
    if symmetric:
        M |= M.T
    np.fill_diagonal(M, True)
    return M


def topk_bipartite(B: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns per row of a rectangular score matrix (zeros never kept)."""
    M = np.zeros_like(B, dtype=bool)
    if k <= 0:
        return M
    k = min(k, B.shape[1])
    idx = np.argpartition(-B, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(B.shape[0]), k)
    keep = B[rows, idx.ravel()] > 0
    M[rows[keep], idx.ravel()[keep]] = True
    return M


def knn_kernel(S: np.ndarray, k: int) -> np.ndarray:
    """Row-normalised weights of each node's k most similar OTHER nodes.

    K @ A gives, for every node, the similarity-weighted average association
    profile of its neighbours ("which drugs do my most similar diseases take?").
    """
    W = fill_missing(S) * knn_mask(S, k, symmetric=False)
    np.fill_diagonal(W, 0.0)
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return W / s
