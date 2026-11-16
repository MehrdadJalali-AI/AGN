"""Baselines and the AGN birth model.

Non-learned baselines rank each paper's candidates by one classical growth mechanism:
  PA (preferential attachment: citations received), Fitness (citations received in the last year),
  Recency (newest first), Similarity (text cosine), Copy-NN (references of the most similar paper born last
  year, i.e. the copying model), Copy-kNN (similarity-weighted references of the 30 most similar last-year papers).
Learned baseline: LR-Ranker, a listwise-trained linear combination of the five mechanism features.

AGN: each candidate gets one score per mechanism "expert" (popularity, fitness, recency, learned text
similarity, learned birth memory). Expert scores are z-scored within the paper's candidate list, and a gate
network conditioned on the new paper's text mixes them: score = sum_m g_m(x_p) * a_m * z_m.
The gate weights g are the interpretable readout (which mechanism drives each birth).
Ablations: AGN-nogate (one global mixture), AGN-nomem (no birth-memory expert).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

BASELINES = ["PA", "Fitness", "Recency", "Similarity", "Copy-NN", "Copy-kNN"]
LEARNED = ["LR-Ranker", "AGN", "AGN-nogate", "AGN-nomem"]
EXPERTS = ["popularity", "fitness", "recency", "similarity", "memory"]


def baseline_scores(name, it):
    F = it["F"]
    eps = 1e-3 * F[:, 3]                                   # text cosine as tie-breaker
    if name == "PA":
        return F[:, 0] + 1e-3 * eps
    if name == "Fitness":
        return F[:, 1] + 1e-3 * F[:, 0]
    if name == "Recency":
        return -F[:, 2] + 1e-3 * F[:, 0]
    if name == "Similarity":
        return F[:, 3]
    if name == "Copy-NN":
        s = np.zeros(len(F), np.float32)
        s[it["ci"][it["qi"] == 0]] = 1.0
        return s + eps
    if name == "Copy-kNN":
        return F[:, 4] + eps
    raise ValueError(name)


def _zscore(E, mask):
    m = mask.unsqueeze(-1).float()
    n = m.sum(1, keepdim=True).clamp(min=1)
    mu = (E * m).sum(1, keepdim=True) / n
    sd = (((E - mu) ** 2) * m).sum(1, keepdim=True).div(n).sqrt().clamp(min=1e-4)
    return (E - mu) / sd


class LRRanker(nn.Module):
    def __init__(self, d=128, hidden=None):
        super().__init__()
        self.w = nn.Linear(5, 1)

    def forward(self, b):
        z = _zscore(b["F"], b["mask"])
        return self.w(z).squeeze(-1), None


class AGN(nn.Module):
    def __init__(self, d=128, hidden=64, gate=True, memory=True):
        super().__init__()
        self.use_gate, self.use_mem = gate, memory
        self.E = 5 if memory else 4
        self.P = nn.Linear(d, hidden, bias=False)
        self.Wq = nn.Linear(d, hidden, bias=False)
        self.Wk = nn.Linear(d, hidden, bias=False)
        self.att_cos = nn.Parameter(torch.tensor(10.0))
        self.sim_mix = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.scale = nn.Parameter(torch.ones(self.E))
        self.gate = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, self.E))
        self.glob = nn.Parameter(torch.zeros(self.E))
        self.h = hidden

    def forward(self, b):
        F, mask = b["F"], b["mask"]
        xp, Xc = b["xp"], b["Xc"]
        lsim = Fn.cosine_similarity(self.P(xp).unsqueeze(1), self.P(Xc), dim=-1)
        experts = [F[..., 0], F[..., 1], -F[..., 2], self.sim_mix[0] * lsim + self.sim_mix[1] * F[..., 3]]
        if self.use_mem:
            q = self.Wq(xp)                                       # B x h
            k = self.Wk(b["Xm"])                                  # B x Q x h
            logit = (k @ q.unsqueeze(-1)).squeeze(-1) / self.h ** 0.5 + self.att_cos * b["mc"]
            att = torch.softmax(logit, -1)                        # B x Q
            mass = torch.einsum("bq,bqc->bc", att, b["M"])
            experts.append(torch.log(1e-4 + mass))
        E = _zscore(torch.stack(experts, -1), mask) * self.scale
        g = torch.softmax(self.gate(xp), -1) if self.use_gate else torch.softmax(self.glob, -1).expand(len(xp), -1)
        return (E * g.unsqueeze(1)).sum(-1), g


def make(name, d, hidden):
    if name == "LR-Ranker":
        return LRRanker(d)
    return AGN(d, hidden, gate=(name != "AGN-nogate"), memory=(name != "AGN-nomem"))


def collate(items, Xt, negatives=None, rng=None):
    """Pad a batch. With `negatives`, keep all positives plus that many random negatives per paper."""
    sel = []
    for it in items:
        if negatives is None:
            sel.append(np.arange(len(it["cand"])))
        else:
            pos = np.where(it["y"])[0]
            neg = np.where(~it["y"])[0]
            if len(neg) > negatives:
                neg = rng.choice(neg, negatives, replace=False)
            sel.append(np.sort(np.concatenate([pos, neg])))
    B, C, Q = len(items), max(len(s) for s in sel), len(items[0]["mq"])
    cand = np.zeros((B, C), np.int64)
    F = np.zeros((B, C, 5), np.float32)
    mask = np.zeros((B, C), bool)
    y = np.zeros((B, C), bool)
    M = np.zeros((B, Q, C), np.float32)
    for i, (it, s) in enumerate(zip(items, sel)):
        n = len(s)
        cand[i, :n], F[i, :n], mask[i, :n], y[i, :n] = it["cand"][s], it["F"][s], True, it["y"][s]
        remap = -np.ones(len(it["cand"]), np.int64)
        remap[s] = np.arange(n)
        j = remap[it["ci"]]
        ok = j >= 0
        M[i, it["qi"][ok], j[ok]] = 1.0
    t = lambda a: torch.from_numpy(a)
    mq = np.stack([it["mq"] for it in items])
    return {"xp": Xt[[it["p"] for it in items]], "Xc": Xt[t(cand)], "Xm": Xt[t(mq)],
            "mc": t(np.stack([it["mc"] for it in items])), "F": t(F), "mask": t(mask), "y": t(y), "M": t(M),
            "sel": sel}


def listwise_loss(score, b):
    s = score.masked_fill(~b["mask"], -1e9)
    logp = torch.log_softmax(s, -1)
    y = b["y"].float()
    npos = y.sum(-1)
    keep = npos > 0
    return -((logp * y).sum(-1)[keep] / npos[keep]).mean()
