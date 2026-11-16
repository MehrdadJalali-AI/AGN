"""Unit checks: pair-sum preservation and AP. Run: python3 -m tests.test_triad"""
import itertools
import numpy as np
import torch
from agnt.models import AGNTriad


def test_pair_sums_preserved():
    rng = np.random.default_rng(0)
    V = 8
    trip = list(itertools.combinations(range(V), 3))
    idx = {frozenset(t): i for i, t in enumerate(trip)}
    x, y, z = (0, 1), (2, 3), (4, 5)
    G = [[idx[frozenset((x[i], y[j], z[k]))] for i, j, k in itertools.product(range(2), repeat=3)]]
    sign = torch.tensor([(-1.0) ** (i + j + k) for i, j, k in itertools.product(range(2), repeat=3)])
    n = len(trip)
    m = AGNTriad(d=5, hidden=8, rounds=2)
    torch.nn.init.normal_(m.g[-1].weight)
    t = {"phi": torch.randn(n, 5), "P": torch.randn(n, 3, 4), "tag": torch.randn(n, 3, 1),
         "groups": torch.tensor(G), "sign": sign}
    _, zres = m(t, return_z=True)
    assert zres.abs().sum() > 0
    for p in itertools.combinations(range(V), 2):
        s = sum(zres[i].item() for i, tr in enumerate(trip) if set(p) <= set(tr))
        assert abs(s) < 1e-5, (p, s)


def test_ap():
    from agnt.tasks import average_precision
    assert abs(average_precision(np.array([3, 2, 1.0]), np.array([True, False, True])) - (1 + 2 / 3) / 2) < 1e-9


if __name__ == "__main__":
    test_pair_sums_preserved(); test_ap(); print("TESTS OK")
