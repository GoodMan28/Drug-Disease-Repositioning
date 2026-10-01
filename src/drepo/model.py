"""
MV-HGAT: Multi-View Heterogeneous Graph Attention Network for drug repositioning.

Graph
-----
Two node types (drug, disease) and several *relations*:

    drug    <- drug      one relation per drug view   (chem_cdk, chem_ecfp, gene_r)
    disease <- disease   one relation per disease view (pheno_mim, sem_mondo, gene_d)
    drug   <-> disease   'assoc'  : known indications in the TRAINING fold only
    drug   <-> disease   'bridge' : shared-gene meta-path drug -> gene -> disease

Gene nodes are folded into meta-paths (drug-gene-drug = gene_r view,
disease-gene-disease = gene_d view, drug-gene-disease = bridge), the standard
way HAN-style models handle an intermediate node type.

One layer
---------
1. Node-level attention (GAT): for every relation, each node attends over its
   neighbours in that relation and builds one message per relation.
2. View-level attention: each node weighs its relation messages with a learned
   softmax (beta). beta is node-specific, so for every drug/disease we can read
   off *which similarity views it relied on*. This is the interpretability hook.

Decoder
-------
logit(drug i, disease j) = h_i^T W h_j                        (GNN, bilinear)
                         + sum_v  w_v * P_v[i, j]  + b         (propagation head)
P_v[i, j] is how strongly drug i's / disease j's nearest neighbours in view v
are linked to the other side. The GNN term is multiplied by a learned gate that
grows with the number of visible links of the drug and the disease, so a
disease with no known drugs (cold start) relies on the propagation views. w_v >= 0 are learned, so each view's share of a
prediction is an explicit additive term (faithful interpretability).
Trained with BCE + negative sampling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGAT(nn.Module):
    """Multi-head graph attention over a dense boolean adjacency (dst x src)."""

    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()
        assert out_dim % heads == 0
        self.h, self.dh = heads, out_dim // heads
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_dst, h_src, mask):
        zs = self.W_src(h_src).view(-1, self.h, self.dh)          # S,H,dh
        zd = self.W_dst(h_dst).view(-1, self.h, self.dh)          # D,H,dh
        e = (zd * self.a_dst).sum(-1)[:, None, :] + (zs * self.a_src).sum(-1)[None, :, :]
        e = F.leaky_relu(e, 0.2).masked_fill(~mask[..., None], float("-inf"))
        att = torch.softmax(e, dim=1)
        att = torch.nan_to_num(att, nan=0.0)                      # nodes w/o neighbours
        out = torch.einsum("dsh,shk->dhk", self.drop(att), zs).reshape(h_dst.shape[0], -1)
        return out, mask.any(1)


class ViewAttention(nn.Module):
    """Node-specific softmax over relation messages (semantic attention, HAN)."""

    def __init__(self, dim, att_dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, att_dim)
        self.q = nn.Linear(att_dim, 1, bias=False)

    def forward(self, msgs, valid, uniform=False):
        # msgs: R,N,D   valid: R,N (relation has >= 1 neighbour for this node)
        if uniform:
            s = torch.zeros(valid.shape, device=msgs.device)
        else:
            s = self.q(torch.tanh(self.proj(msgs))).squeeze(-1)
        s = s.masked_fill(~valid, float("-inf"))
        beta = torch.nan_to_num(torch.softmax(s, dim=0), nan=0.0)  # R,N
        return (beta[..., None] * msgs).sum(0), beta


class HeteroLayer(nn.Module):
    def __init__(self, relations, in_dim, out_dim, heads, dropout):
        super().__init__()
        self.relations = relations                  # name -> (dst_type, src_type)
        self.gat = nn.ModuleDict({r: DenseGAT(in_dim, out_dim, heads, dropout) for r in relations})
        self.view_att = nn.ModuleDict({t: ViewAttention(out_dim) for t in ("drug", "disease")})
        self.skip = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in ("drug", "disease")})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(out_dim) for t in ("drug", "disease")})
        self.drop = nn.Dropout(dropout)

    def forward(self, h, graphs, uniform=False, drop_rel=()):
        msgs = {"drug": [], "disease": []}
        valid = {"drug": [], "disease": []}
        names = {"drug": [], "disease": []}
        for rel, (dst, src) in self.relations.items():
            out, has = self.gat[rel](h[dst], h[src], graphs[rel])
            if rel in drop_rel:                      # occlusion for explanations
                has = torch.zeros_like(has)
            msgs[dst].append(out)
            valid[dst].append(has)
            names[dst].append(rel)
        new, betas = {}, {}
        for t in ("drug", "disease"):
            z, beta = self.view_att[t](torch.stack(msgs[t]), torch.stack(valid[t]), uniform)
            new[t] = self.drop(F.elu(self.norm[t](z + self.skip[t](h[t]))))
            betas[t] = (names[t], beta)
        return new, betas


class MVHGAT(nn.Module):
    def __init__(self, relations, in_drug, in_dis, hidden=64, layers=2, heads=4, dropout=0.3,
                 uniform_view_attention=False, n_prop=0):
        super().__init__()
        # multi-view propagation head: one non-negative weight per view
        self.n_prop = n_prop
        if n_prop:
            self.prop_w = nn.Parameter(torch.zeros(n_prop))
            self.prop_scale = nn.Parameter(torch.tensor(5.0))
            self.bias = nn.Parameter(torch.tensor(-3.0))
            # degree gate: how much to trust the GNN term given #visible links
            self.gate = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 1.0]))
        self.inp = nn.ModuleDict({"drug": nn.Linear(in_drug, hidden),
                                  "disease": nn.Linear(in_dis, hidden)})
        self.layers = nn.ModuleList(
            HeteroLayer(relations, hidden, hidden, heads, dropout) for _ in range(layers))
        out = hidden * (layers + 1)                 # jumping-knowledge concat
        self.W = nn.Parameter(torch.empty(out, out))
        nn.init.xavier_uniform_(self.W)
        self.drop = nn.Dropout(dropout)
        self.uniform = uniform_view_attention

    def encode(self, X, graphs, drop_rel=()):
        h = {t: self.drop(F.elu(self.inp[t](X[t]))) for t in ("drug", "disease")}
        outs = {t: [h[t]] for t in h}
        all_betas = []
        for layer in self.layers:
            h, betas = layer(h, graphs, self.uniform, drop_rel)
            all_betas.append(betas)
            for t in h:
                outs[t].append(h[t])
        return torch.cat(outs["drug"], 1), torch.cat(outs["disease"], 1), all_betas

    def view_weights(self):
        return F.softplus(self.prop_w) * self.prop_scale

    def gnn_gate(self, deg_r, deg_d):
        g = self.gate
        return torch.sigmoid(g[0] + g[1] * torch.log1p(deg_r))[:, None] *             torch.sigmoid(g[2] + g[3] * torch.log1p(deg_d))[None, :]

    def forward(self, X, graphs, drop_rel=(), P=None, deg=None):
        """P: (n_views, drugs, diseases) propagation scores, one slice per view.
        deg: (visible links per drug, per disease) for the degree gate."""
        Hr, Hd, betas = self.encode(X, graphs, drop_rel)
        logits = Hr @ self.W @ Hd.T
        if self.n_prop and P is not None:
            if deg is not None:
                logits = self.gnn_gate(*deg) * logits
            logits = logits + (self.view_weights()[:, None, None] * P).sum(0) + self.bias
        return logits, betas                         # logits, drugs x diseases
