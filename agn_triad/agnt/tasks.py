"""Per-cutoff forecasting task: eligible open triples, features, labels, and pair-preserving update groups."""
import itertools

import numpy as np

from .data import history, future_triples

PAIR_FEATS = ["logW", "logR", "L", "logT"]


def build_task(C, frac, cfg, seed):
    V = C.V
    c, c1 = C.idx(frac), C.idx(min(frac + cfg["window"], 1.0))
    h = history(C, c, cfg["decay_events"])
    a, b, d = np.array(list(itertools.combinations(range(V), 3))).T
    W = h["W"]
    elig = (W[a, b] > 0) & (W[a, d] > 0) & (W[b, d] > 0) & ~h["closed"][a, b, d]
    a, b, d = a[elig], b[elig], d[elig]
    tid = (a * V + b) * V + d
    y = np.isin(tid, future_triples(C, c, c1))

    def pf(x, z):
        return np.stack([np.log1p(W[x, z]), np.log1p(h["R"][x, z]), h["L"][x, z], np.log1p(h["T"][x, z])], -1)

    P = np.stack([pf(a, b), pf(a, d), pf(b, d)], 1)                       # n x 3 pairs x 4
    tagf = np.log1p(np.stack([h["f"][a], h["f"][b], h["f"][d]], 1))[..., None]  # n x 3 x 1
    nb = W > 0
    K = (nb[a] & nb[b] & nb[d]).sum(1) - 0                                  # common neighbours (a,b,d excluded: W diag 0)
    cl = h["closed"]
    Q = (cl[a, b] & cl[a, d] & cl[b, d]).sum(1)                             # 4th tags closing all three faces
    phi = np.concatenate([np.sort(P, 1).reshape(len(a), -1), np.sort(tagf[..., 0], 1),
                          np.log1p(np.stack([K, Q], 1))], 1).astype(np.float32)
    # copy-with-substitution evidence: E[x,z,w] = recency weight of closed triple {x,z,w}
    S = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    S = S @ S.T                                                               # tag cosine (co-occurrence profiles)
    task = {"frac": frac, "cutoff": c, "end": c1, "a": a, "b": b, "c": d, "y": y, "P": P.astype(np.float32),
            "tag": tagf.astype(np.float32), "phi": phi, "tage": h["tage"], "S": S, "V": V,
            "n_eligible": int(len(a)), "n_pos": int(y.sum())}
    task.update(sample_groups(task, cfg, seed))
    return task


def copy_scores(task, tau):
    E = np.where(np.isfinite(task["tage"]), np.exp(-task["tage"] / tau), 0.0)
    S = task["S"]
    a, b, c = task["a"], task["b"], task["c"]
    return (np.einsum("ny,ny->n", E[a, b], S[c]) + np.einsum("ny,ny->n", E[a, c], S[b])
            + np.einsum("ny,ny->n", E[b, c], S[a]))


def sample_groups(task, cfg, seed):
    """Groups of 8 eligible triples {x_i, y_j, z_k}, i,j,k in {0,1}, six distinct tags, sign (-1)^(i+j+k).
    Every pair {x_i, y_j} (etc.) appears in exactly two triples of opposite sign, so an update
    z += alpha * sign leaves the summed residual of every pair unchanged."""
    V, rng = task["V"], np.random.default_rng(seed)
    lookup = -np.ones(V ** 3, np.int64)
    for p in itertools.permutations(range(3)):
        t = [task["a"], task["b"], task["c"]]
        lookup[(t[p[0]] * V + t[p[1]]) * V + t[p[2]]] = np.arange(task["n_eligible"])
    groups, seen = [], set()
    n_prop = 0
    # for every unordered pair, the third tags that make an eligible (open) triple with it
    third = {}
    for x, y_, z in zip(task["a"], task["b"], task["c"]):
        for p_, q, r in ((x, y_, z), (x, z, y_), (y_, z, x)):
            third.setdefault((int(p_), int(q)), []).append(int(r))
    third = {k: np.array(v) for k, v in third.items()}
    empty = np.array([], np.int64)
    pick = lambda u, w: (lambda arr: int(arr[rng.integers(len(arr))]) if len(arr) else -1)(third.get((min(u, w), max(u, w)), empty))
    for _ in range(cfg["group_proposal_batches"]):
        if len(groups) >= cfg["groups_per_cutoff"] or task["n_eligible"] == 0:
            break
        m = cfg["group_proposal_size"]
        n_prop += m
        base = rng.integers(0, task["n_eligible"], m)
        perm = rng.permutation(3)
        trip = np.stack([task["a"][base], task["b"][base], task["c"][base]], 1)
        for i0, b0 in enumerate(base):
            x0, y0, z0 = trip[i0][perm]
            x1, y1, z1 = pick(y0, z0), pick(x0, z0), pick(x0, y0)
            if min(x1, y1, z1) < 0 or len({x0, y0, z0, x1, y1, z1}) < 6:
                continue
            r = []
            for i, j, k in itertools.product(range(2), repeat=3):
                xs, ys, zs = (x0, x1)[i], (y0, y1)[j], (z0, z1)[k]
                r.append(lookup[(xs * V + ys) * V + zs])
            if min(r) < 0:
                continue
            key = tuple(sorted(r))
            if key not in seen:
                seen.add(key); groups.append(r)
                if len(groups) >= cfg["groups_per_cutoff"]:
                    break
    sign = np.array([(-1) ** (i + j + k) for i, j, k in itertools.product(range(2), repeat=3)], np.float32)
    G = np.array(groups, np.int64).reshape(-1, 8)
    covered = np.zeros(task["n_eligible"], bool)
    covered[G.ravel()] = True
    return {"groups": G, "sign": sign, "n_groups": int(len(G)), "n_proposals": int(n_prop),
            "frac_covered": float(covered.mean()) if task["n_eligible"] else 0.0}


def average_precision(score, y):
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="stable")
    yy = y[o]
    hits = np.cumsum(yy)
    return float((hits[yy] / (np.nonzero(yy)[0] + 1)).mean())
