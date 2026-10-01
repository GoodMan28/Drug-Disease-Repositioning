"""
Every method exposes the same interface so the evaluation code can treat them
identically:

    scores = method.fit_predict(data, A_train, neg_mask, seed)

A_train  : drugs x diseases, the known links we are ALLOWED to see in this fold
neg_mask : cells that may be used as training negatives (excludes test cells)
scores   : drugs x diseases numpy array, higher = more likely an indication
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DDData, fill_missing, knn_kernel, knn_mask, topk_bipartite
from .model import MVHGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


# =========================================================================== #
# Proposed model                                                              #
# =========================================================================== #
@dataclass
class MVHGATConfig:
    drug_views: tuple | None = None       # None = all views in the file
    disease_views: tuple | None = None
    use_bridge: bool = False              # drug-gene-disease meta-path (weak signal; ablation)
    use_assoc_edges: bool = True          # known links as message-passing edges
    uniform_attention: bool = False       # ablation: replace view attention by a mean
    k: int = 10                           # neighbours kept per similarity view
    bridge_k: int = 10
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 5e-4
    epochs: int = 600
    neg_ratio: int = 2                    # sampled negatives per positive per epoch
    drop_edge: float = 0.2                # fraction of assoc edges hidden each epoch
    supervise_hidden: bool = True         # loss only on the links hidden this epoch
    feat_assoc: bool = True               # append (visible) association rows to features
    feat_views: str = "all"               # similarity rows used as input features:
                                          # "all" (concatenate), "mean", "first"
    cold_frac: float = 0.1                # share of diseases fully hidden per epoch
    feat_prop: bool = False               # per-view neighbour association profiles as features
    prop_head: bool = True                # add the multi-view propagation head to the decoder
    degree_gate: bool = True              # scale the GNN term by a learned link-count gate


class MVHGATMethod:
    name = "MV-HGAT (ours)"

    def __init__(self, cfg: MVHGATConfig | None = None, **overrides):
        self.cfg = cfg or MVHGATConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.last = None   # (model, X, graphs) of the last fit, for explanations

    # -- graph construction --------------------------------------------------
    def build(self, data: DDData, A_train):
        c = self.cfg
        rv = list(c.drug_views or data.drug_view_names)
        dv = list(c.disease_views or data.disease_view_names)
        def feats(mats):
            mats = [fill_missing(m) for m in mats]
            if c.feat_views == "first":
                return t(mats[0])
            if c.feat_views == "mean":
                return t(np.mean(mats, 0))
            if c.feat_views == "none":
                return t(np.zeros((len(mats[0]), 0)))
            return t(np.concatenate(mats, 1))

        X = {"drug": feats([data.drug_view(v) for v in rv]),
             "disease": feats([data.disease_view(v) for v in dv])}
        self._prop = {"drug": [t(knn_kernel(data.drug_view(v), c.k)) for v in rv],
                      "disease": [t(knn_kernel(data.disease_view(v), c.k)) for v in dv]}
        relations, graphs = {}, {}
        for v in rv:
            relations[f"view:{v}"] = ("drug", "drug")
            graphs[f"view:{v}"] = t(knn_mask(data.drug_view(v), c.k), torch.bool)
        for v in dv:
            relations[f"view:{v}"] = ("disease", "disease")
            graphs[f"view:{v}"] = t(knn_mask(data.disease_view(v), c.k), torch.bool)
        if c.use_assoc_edges:
            relations["assoc>drug"] = ("drug", "disease")
            relations["assoc>disease"] = ("disease", "drug")
            A = t(A_train > 0, torch.bool)
            graphs["assoc>drug"], graphs["assoc>disease"] = A, A.T
        if c.use_bridge:
            B = data.gene_bridge
            relations["gene_bridge>drug"] = ("drug", "disease")
            relations["gene_bridge>disease"] = ("disease", "drug")
            graphs["gene_bridge>drug"] = t(topk_bipartite(B, c.bridge_k), torch.bool)
            graphs["gene_bridge>disease"] = t(topk_bipartite(B.T, c.bridge_k), torch.bool)
        return relations, X, graphs

    # -- training ------------------------------------------------------------
    def fit_predict(self, data, A_train, neg_mask, seed=0):
        c = self.cfg
        set_seed(seed)
        relations, X0, graphs = self.build(data, A_train)
        A_full = t(A_train > 0)

        def features(Am):
            # everything derived from links uses only the VISIBLE links Am
            Am = Am.float()
            parts = {"drug": [X0["drug"]], "disease": [X0["disease"]]}
            if c.feat_assoc:
                parts["drug"].append(Am)
                parts["disease"].append(Am.T)
            if c.feat_prop:
                parts["drug"] += [K @ Am for K in self._prop["drug"]]
                parts["disease"] += [K @ Am.T for K in self._prop["disease"]]
            return {k: torch.cat(v, 1) for k, v in parts.items()}

        def propagation(Am):
            """One (drugs x diseases) score slice per view, from VISIBLE links only."""
            if not c.prop_head:
                return None
            Am = Am.float()
            return torch.stack([K @ Am for K in self._prop["drug"]] +
                               [(K @ Am.T).T for K in self._prop["disease"]])

        def degrees(Am):
            Am = Am.float()
            return (Am.sum(1), Am.sum(0)) if c.degree_gate else None

        self.prop_names = [f"view:{v}" for v in (list(c.drug_views or data.drug_view_names) +
                                                 list(c.disease_views or data.disease_view_names))]
        X = features(A_full)
        model = MVHGAT(relations, X["drug"].shape[1], X["disease"].shape[1], c.hidden, c.layers,
                       c.heads, c.dropout, c.uniform_attention,
                       n_prop=len(self.prop_names) if c.prop_head else 0).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

        n_r, n_d = A_train.shape
        pos = t(np.flatnonzero(A_train.ravel() > 0), torch.long)
        neg_pool = t(np.flatnonzero((neg_mask & (A_train == 0)).ravel()), torch.long)
        full_assoc = graphs.get("assoc>drug")

        for _ in range(c.epochs):
            model.train()
            # Each epoch a random share of the known links is HIDDEN (from the
            # graph and from the features). Supervising on exactly those hidden
            # links mimics test time, where the link to predict is never visible.
            # Cold-start practice: a few diseases lose ALL their links this epoch,
            # just like the held-out disease in leave-one-disease-out testing.
            sup = pos
            if c.drop_edge > 0 or c.cold_frac > 0:
                hide = torch.rand(pos.numel(), device=DEVICE) < c.drop_edge
                if c.cold_frac > 0:
                    cold = torch.rand(n_d, device=DEVICE) < c.cold_frac
                    hide |= cold[pos % n_d]
                Am = torch.zeros(n_r * n_d, dtype=torch.bool, device=DEVICE)
                Am[pos[~hide]] = True
                Am = Am.view(n_r, n_d)
                if c.supervise_hidden:
                    sup = pos[hide]
            else:
                Am = A_full.bool()
            if full_assoc is not None:
                graphs["assoc>drug"], graphs["assoc>disease"] = Am, Am.T
            logits, _ = model(features(Am), graphs, P=propagation(Am), deg=degrees(Am))
            logits = logits.reshape(-1)
            n_neg = min(neg_pool.numel(), sup.numel() * c.neg_ratio) if c.neg_ratio > 0 \
                else neg_pool.numel()
            neg = neg_pool[torch.randint(neg_pool.numel(), (n_neg,), device=DEVICE)]
            y = torch.cat([torch.ones(sup.numel(), device=DEVICE),
                           torch.zeros(n_neg, device=DEVICE)])
            loss = F.binary_cross_entropy_with_logits(torch.cat([logits[sup], logits[neg]]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if full_assoc is not None:
            graphs["assoc>drug"], graphs["assoc>disease"] = full_assoc, full_assoc.T
        model.eval()
        P, deg = propagation(A_full), degrees(A_full)
        with torch.no_grad():
            logits, _ = model(X, graphs, P=P, deg=deg)
        self.last = (model, X, graphs, P, deg)
        return torch.sigmoid(logits).cpu().numpy()

    # -- interpretability ----------------------------------------------------
    @torch.no_grad()
    def view_attention(self):
        """Per-node view weights (beta) of the GNN encoder, averaged over layers.
        Returns {'drug': (relation_names, n_drugs x R array), 'disease': ...}"""
        model, X, graphs, _, _ = self.last
        _, _, betas = model.encode(X, graphs)
        out = {}
        for tp in ("drug", "disease"):
            names = betas[0][tp][0]
            B = torch.stack([b[tp][1] for b in betas]).mean(0).T.cpu().numpy()
            out[tp] = (names, B)
        return out

    @torch.no_grad()
    def view_weights(self):
        """Global weight w_v of each view in the propagation head."""
        model = self.last[0]
        if not model.n_prop:
            return {}
        return dict(zip(self.prop_names, model.view_weights().cpu().numpy()))

    @torch.no_grad()
    def occlusion(self, pairs):
        """Switch off one evidence source at a time (its graph relation AND its
        propagation-head term) and measure how much each pair's probability
        falls. A large drop = that source drove the prediction."""
        model, X, graphs, P, deg = self.last
        base = torch.sigmoid(model(X, graphs, P=P, deg=deg)[0])
        groups = {}
        for r in graphs:
            groups.setdefault(r.split(">")[0], []).append(r)
        res = {}
        for g, rels in groups.items():
            Pg = P
            if P is not None and g in self.prop_names:
                Pg = P.clone()
                Pg[self.prop_names.index(g)] = 0
            s = torch.sigmoid(model(X, graphs, drop_rel=set(rels), P=Pg, deg=deg)[0])
            res[g] = np.array([(base[i, j] - s[i, j]).item() for i, j in pairs])
        return res


# =========================================================================== #
# Baselines (re-implemented; they use the two benchmark similarities only,    #
# exactly as in their original papers)                                       #
# =========================================================================== #
def bench_sims(data):
    return fill_missing(data.drug_views[0]), fill_missing(data.disease_views[0])


def sym_norm(S):
    d = S.sum(1)
    d[d == 0] = 1
    d = 1 / np.sqrt(d)
    return S * d[:, None] * d[None, :]


class MBiRW:
    """Luo et al. 2016. Logistic similarity adjustment + bi-random walk.
    (The paper also sharpens similarities with a clustering step; we keep the
    logistic adjustment, which is the part that matters most.)"""
    name = "MBiRW"

    def __init__(self, alpha=0.3, l=2, r=2):
        self.alpha, self.l, self.r = alpha, l, r

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        Sr, Sd = bench_sims(data)
        logistic = lambda S: 1 / (1 + np.exp(-15 * S + np.log(9999)))
        Mr, Md = sym_norm(logistic(Sr)), sym_norm(logistic(Sd))
        A0 = A_train / max(A_train.sum(), 1)
        R = A0.copy()
        for step in range(1, max(self.l, self.r) + 1):
            parts = []
            if step <= self.l:
                parts.append(self.alpha * Mr @ R + (1 - self.alpha) * A0)
            if step <= self.r:
                parts.append(self.alpha * R @ Md + (1 - self.alpha) * A0)
            R = sum(parts) / len(parts)
        return R


class DRRS:
    """Luo et al. 2018. Complete the heterogeneous matrix [[Sr, A],[A', Sd]] with
    singular value thresholding (low-rank assumption), randomized SVD."""
    name = "DRRS"

    def __init__(self, tau_rel=0.005, iters=200, rank=200):
        self.tau_rel, self.iters, self.rank = tau_rel, iters, rank

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        n_r = Sr.shape[0]
        T = t(np.block([[Sr, A_train], [A_train.T, Sd]]))
        Om = (T != 0).float()
        N = T.shape[0]
        delta = 1.2 * N * N / Om.sum()
        tau = self.tau_rel * torch.linalg.matrix_norm(T, ord=2) * N / 10
        Y = torch.zeros_like(T)
        X = T
        for _ in range(self.iters):
            U, S, V = torch.svd_lowrank(Y if Y.abs().sum() > 0 else T, q=self.rank, niter=2)
            S = torch.clamp(S - tau, min=0)
            X = (U * S) @ V.T
            Y = Y + delta * Om * (T - X)
        return X[:n_r, n_r:].cpu().numpy()


class SCMFDD:
    """Zhang et al. 2018. Similarity-constrained matrix factorisation:
    ||A - U V'||^2 + mu(||U||^2+||V||^2) + lam(tr U'Lr U + tr V'Ld V)."""
    name = "SCMFDD"

    def __init__(self, k=128, mu=0.05, lam=2.0, epochs=400, lr=0.02):
        self.k, self.mu, self.lam, self.epochs, self.lr = k, mu, lam, epochs, lr

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        Lr = t(np.eye(len(Sr)) - sym_norm(Sr))
        Ld = t(np.eye(len(Sd)) - sym_norm(Sd))
        A = t(A_train)
        U = nn.Parameter(0.1 * torch.randn(A.shape[0], self.k, device=DEVICE))
        V = nn.Parameter(0.1 * torch.randn(A.shape[1], self.k, device=DEVICE))
        opt = torch.optim.Adam([U, V], lr=self.lr)
        for _ in range(self.epochs):
            loss = ((A - U @ V.T) ** 2).sum() + self.mu * (U.pow(2).sum() + V.pow(2).sum()) \
                + self.lam * (torch.trace(U.T @ Lr @ U) + torch.trace(V.T @ Ld @ V))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (U @ V.T).detach().cpu().numpy()


def weighted_bce(logits, A, neg_mask):
    """BCE over all allowed cells, positives up-weighted to balance classes."""
    m = (A > 0) | neg_mask
    pw = (m & (A == 0)).sum() / max((A > 0).sum(), 1)
    w = torch.where(A > 0, pw, torch.ones_like(A)) * m
    return (F.binary_cross_entropy_with_logits(logits, A, reduction="none") * w).sum() / w.sum()


class _GCN(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.drop = nn.Dropout(dropout)

    def forward(self, Ahat, x, return_all=False):
        outs = []
        for lin in self.lins:
            x = F.relu(Ahat @ lin(self.drop(x)))
            outs.append(x)
        return outs if return_all else x


class NIMCGCN:
    """Li et al. 2020. Separate GCNs on the drug and disease similarity graphs,
    followed by neural inductive matrix completion (MLP projections, inner product)."""
    name = "NIMCGCN"

    def __init__(self, hidden=128, out=64, epochs=1500, lr=2e-3, dropout=0.3, k=10):
        self.h, self.o, self.epochs, self.lr, self.dropout, self.k = hidden, out, epochs, lr, dropout, k

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        Ar = t(sym_norm(Sr * knn_mask(Sr, self.k)))
        Ad = t(sym_norm(Sd * knn_mask(Sd, self.k)))
        Xr, Xd = t(Sr), t(Sd)
        gr = _GCN([Xr.shape[1], self.h, self.o], self.dropout).to(DEVICE)
        gd = _GCN([Xd.shape[1], self.h, self.o], self.dropout).to(DEVICE)
        fr, fd = nn.Linear(self.o, self.o).to(DEVICE), nn.Linear(self.o, self.o).to(DEVICE)
        params = [*gr.parameters(), *gd.parameters(), *fr.parameters(), *fd.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)
        A, nm = t(A_train), t(neg_mask, torch.bool)
        mods = [gr, gd]
        for _ in range(self.epochs):
            [m.train() for m in mods]
            logits = fr(gr(Ar, Xr)) @ fd(gd(Ad, Xd)).T
            loss = weighted_bce(logits, A, nm)
            opt.zero_grad()
            loss.backward()
            opt.step()
        [m.eval() for m in mods]
        with torch.no_grad():
            return torch.sigmoid(fr(gr(Ar, Xr)) @ fd(gd(Ad, Xd)).T).cpu().numpy()


class LAGCN:
    """Yu et al. 2021. GCN on the heterogeneous graph [[Sr, A],[A', Sd]]; the
    embeddings of all layers are combined with learned layer attention."""
    name = "LAGCN"

    def __init__(self, hidden=64, layers=3, epochs=1000, lr=2e-3, dropout=0.0, drop_edge=0.0):
        self.h, self.L, self.epochs, self.lr, self.dropout, self.de = hidden, layers, epochs, lr, dropout, drop_edge

    def fit_predict(self, data, A_train, neg_mask, seed=0):
        set_seed(seed)
        Sr, Sd = bench_sims(data)
        n_r = Sr.shape[0]
        H = t(np.block([[Sr, A_train], [A_train.T, Sd]]))
        gcn = _GCN([H.shape[1]] + [self.h] * self.L, self.dropout).to(DEVICE)
        att = nn.Parameter(torch.ones(self.L, device=DEVICE) / self.L)
        opt = torch.optim.Adam([*gcn.parameters(), att], lr=self.lr, weight_decay=1e-4)
        A, nm = t(A_train), t(neg_mask, torch.bool)
        assoc = torch.zeros_like(H, dtype=torch.bool)
        assoc[:n_r, n_r:] = True
        assoc[n_r:, :n_r] = True

        def embed(Hm):
            d = Hm.sum(1).clamp(min=1e-12).rsqrt()
            outs = gcn(Hm * d[:, None] * d[None, :], H, return_all=True)
            w = torch.softmax(att, 0)
            return sum(wi * o for wi, o in zip(w, outs))

        for _ in range(self.epochs):
            gcn.train()
            drop = assoc & (torch.rand_like(H) < self.de)
            E = embed(H.masked_fill(drop, 0))
            loss = weighted_bce(E[:n_r] @ E[n_r:].T, A, nm)
            opt.zero_grad()
            loss.backward()
            opt.step()
        gcn.eval()
        with torch.no_grad():
            E = embed(H)
            return torch.sigmoid(E[:n_r] @ E[n_r:].T).cpu().numpy()


METHODS = {
    "mvhgat": MVHGATMethod,
    "mbirw": MBiRW,
    "drrs": DRRS,
    "scmfdd": SCMFDD,
    "nimcgcn": NIMCGCN,
    "lagcn": LAGCN,
}


def make(name, **kw):
    return METHODS[name](**kw)
