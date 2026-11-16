"""Learned scorers for first-time triples.

PairAdditive: logit = sum over the 3 pairs u(pair features) + sum over the 3 tags v(tag frequency) + b.
MLP: unconstrained MLP on the full symmetric feature vector phi (same information AGN sees).
AGN-Triad: PairAdditive base + residual z built by R rounds of pair-preserving updates. For each group of
8 triples with signs v, alpha = g([sum_s v_s phi_s, sum_s v_s logit_s]); z_s += alpha * v_s.
Each update keeps the sum of residuals over the triples containing any given pair unchanged, so all
genuinely three-way signal is a redistribution among competing combinations.
"""
import torch
import torch.nn as nn


class PairAdditive(nn.Module):
    def __init__(self, hidden=32, **_):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.v = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, t):
        return self.u(t["P"]).squeeze(-1).sum(1) + self.v(t["tag"]).squeeze(-1).sum(1) + self.b


class MLP(nn.Module):
    def __init__(self, d, hidden=32, **_):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, t):
        return self.f(t["phi"]).squeeze(-1)


class AGNTriad(nn.Module):
    def __init__(self, d, hidden=32, rounds=2, **_):
        super().__init__()
        self.base = PairAdditive(hidden)
        self.g = nn.Sequential(nn.Linear(d + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)
        self.rounds = rounds

    def forward(self, t, return_z=False):
        base = self.base(t)
        z = torch.zeros_like(base)
        G, v = t["groups"], t["sign"]
        if G.numel():
            phiG = t["phi"][G]                                     # n_groups x 8 x d
            cphi = (phiG * v[None, :, None]).sum(1)
            for _ in range(self.rounds):
                logit = (base + z)[G]
                cl = (logit * v[None, :]).sum(1, keepdim=True)
                alpha = self.g(torch.cat([cphi, cl], 1))           # n_groups x 1
                z = z.index_add(0, G.reshape(-1), (alpha * v[None, :]).reshape(-1))
        return (base + z, z) if return_z else base + z


def make(name, d, hidden, rounds):
    return {"PairAdditive": lambda: PairAdditive(hidden), "MLP": lambda: MLP(d, hidden),
            "AGN-Triad": lambda: AGNTriad(d, hidden, rounds)}[name]()
