"""Timestamped tag sets (ScHoLP format) and per-cutoff history statistics.

Events are sorted by time; cutoffs are fractions of the event sequence. Only the `n_tags` most frequent tags
in the first `vocab_until` fraction of events are kept. History at cutoff c uses events with index < c only.
"""
import gzip
import itertools
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/arbenson/ScHoLP-Data/master/tags-math-sx/tags-math-sx-{}.txt.gz"


def _read(path):
    with gzip.open(path, "rt") as f:
        return np.array([int(x) for x in f.read().split()], dtype=np.int64)


def load_scholp(raw_dir, name="tags-math-sx"):
    os.makedirs(raw_dir, exist_ok=True)
    arrs = {}
    for part in ["nverts", "simplices", "times"]:
        p = os.path.join(raw_dir, f"{name}-{part}.txt.gz")
        if not os.path.exists(p):
            print(f"[data] downloading {URL.format(part)}", flush=True)
            urllib.request.urlretrieve(URL.format(part), p)
        arrs[part] = _read(p)
    nv, sx, t = arrs["nverts"], arrs["simplices"], arrs["times"]
    starts = np.concatenate([[0], np.cumsum(nv)[:-1]])
    events = [sx[s:s + k] for s, k in zip(starts, nv)]
    order = np.argsort(t, kind="stable")
    return [events[i] for i in order]


def synthetic(n_events=60000, n_tags=120, seed=0):
    """Smoke-test data: tags with latent topics; later events combine familiar tags."""
    rng = np.random.default_rng(seed)
    topic = rng.integers(0, 8, n_tags)
    pop = rng.pareto(1.5, n_tags) + 1
    ev = []
    for i in range(n_events):
        k = rng.integers(1, 6)
        t = rng.integers(0, 8)
        w = pop * np.where(topic == t, 5.0, 1.0)
        ev.append(rng.choice(n_tags, k, replace=False, p=w / w.sum()) + 1)
    return ev


class Corpus:
    def __init__(self, events, n_tags, vocab_until):
        n = len(events)
        self.n = n
        head = np.concatenate(events[:int(vocab_until * n)])
        ids, cnt = np.unique(head, return_counts=True)
        vocab = ids[np.argsort(-cnt, kind="stable")[:n_tags]]
        self.vocab = vocab
        remap = {int(v): i for i, v in enumerate(vocab)}
        self.V = len(vocab)
        tag_ev, tag_id, pr_ev, pr, tr_ev, tr = [], [], [], [], [], []
        for i, e in enumerate(events):
            s = sorted({remap[int(x)] for x in e if int(x) in remap})
            for a in s:
                tag_ev.append(i); tag_id.append(a)
            if len(s) >= 2:
                for a, b in itertools.combinations(s, 2):
                    pr_ev.append(i); pr.append(a * self.V + b)
            if len(s) >= 3:
                for a, b, c in itertools.combinations(s, 3):
                    tr_ev.append(i); tr.append((a * self.V + b) * self.V + c)
        self.tag_ev, self.tag_id = np.array(tag_ev), np.array(tag_id)
        self.pr_ev, self.pr = np.array(pr_ev), np.array(pr)
        self.tr_ev, self.tr = np.array(tr_ev), np.array(tr)

    def idx(self, frac):
        return int(round(frac * self.n))


def history(C, c, decay):
    """Statistics from events with index < c. Returns dict of V x V (x V) arrays."""
    V, H = C.V, decay * C.n
    m = C.tag_ev < c
    f = np.bincount(C.tag_id[m], minlength=V).astype(np.float64)
    m = C.pr_ev < c
    W = np.bincount(C.pr[m], minlength=V * V).astype(np.float64).reshape(V, V)
    R = np.bincount(C.pr[m], weights=np.exp(-(c - C.pr_ev[m]) / H), minlength=V * V).reshape(V, V)
    last = np.full(V * V, -np.inf)
    np.maximum.at(last, C.pr[m], C.pr_ev[m].astype(np.float64))
    W, R = W + W.T, R + R.T
    last = last.reshape(V, V); last = np.maximum(last, last.T)
    L = np.where(np.isfinite(last), np.minimum((c - last) / H, 10.0), 10.0)
    m = C.tr_ev < c
    closed = np.zeros(V ** 3, bool); closed[C.tr[m]] = True
    tlast = np.full(V ** 3, -np.inf)
    np.maximum.at(tlast, C.tr[m], C.tr_ev[m].astype(np.float64))
    closed, tlast = closed.reshape(V, V, V), tlast.reshape(V, V, V)
    for p in itertools.permutations(range(3)):           # symmetrise
        closed = closed | closed.transpose(p)
        tlast = np.maximum(tlast, tlast.transpose(p))
    T = closed.sum(2).astype(np.float64)                  # distinct third tags closed with each pair
    return {"f": f, "W": W, "R": R, "L": L, "closed": closed, "tage": np.where(np.isfinite(tlast), (c - tlast) / H, np.inf),
            "T": T}


def future_triples(C, c0, c1):
    m = (C.tr_ev >= c0) & (C.tr_ev < c1)
    return np.unique(C.tr[m])
