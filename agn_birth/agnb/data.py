"""Data loading (ogbn-arxiv or a synthetic smoke-test graph) and temporal snapshots.

Edges are (citing, cited). A snapshot at cutoff T uses only papers with year <= T.
Arrivals are papers of year T+1; their references are their citations to papers of year <= T
(same-year citations are excluded because their order is unknown).
"""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ARXIV_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"


def _convert(extract_dir, npz):
    r = os.path.join(extract_dir, "arxiv", "raw")
    e = pd.read_csv(os.path.join(r, "edge.csv.gz"), header=None).values.astype(np.int64)
    y = pd.read_csv(os.path.join(r, "node_year.csv.gz"), header=None).values.ravel().astype(np.int64)
    lab = pd.read_csv(os.path.join(r, "node-label.csv.gz"), header=None).values.ravel().astype(np.int64)
    X = pd.read_csv(os.path.join(r, "node-feat.csv.gz"), header=None, dtype=np.float32).values
    np.savez(npz, edges=e, year=y, label=lab, X=X)


def load_arxiv(data_dir, arxiv_zip):
    npz = os.path.join(data_dir, "arxiv.npz")
    if not os.path.exists(npz):
        os.makedirs(os.path.dirname(arxiv_zip) or ".", exist_ok=True)
        if not os.path.exists(arxiv_zip):
            print(f"[data] downloading {ARXIV_URL} -> {arxiv_zip}", flush=True)
            urllib.request.urlretrieve(ARXIV_URL, arxiv_zip)
        ex = os.path.join(data_dir, "raw", "extracted")
        if not os.path.exists(os.path.join(ex, "arxiv", "raw", "edge.csv.gz")):
            print("[data] extracting", flush=True)
            zipfile.ZipFile(arxiv_zip).extractall(ex)
        print("[data] converting to npz", flush=True)
        _convert(ex, npz)
    z = np.load(npz)
    return z["edges"], z["year"], z["label"], z["X"]


def synthetic(n=6000, d=32, L=6, seed=0):
    """Small growing citation graph for smoke tests only (preferential + copying + topical citations)."""
    rng = np.random.default_rng(seed)
    years = np.sort(rng.choice(np.arange(2008, 2021), n, p=np.linspace(1, 4, 13) / np.linspace(1, 4, 13).sum()))
    lab = rng.integers(0, L, n)
    cent = rng.normal(size=(L, d))
    X = (cent[lab] + 0.7 * rng.normal(size=(n, d))).astype(np.float32)
    refs = {}
    edges = []
    indeg = np.zeros(n)
    for i in range(n):
        older = np.where(years < years[i])[0]
        if older.size < 5:
            refs[i] = np.array([], int)
            continue
        k = rng.integers(2, 8)
        chosen = set()
        same = older[lab[older] == lab[i]]
        if same.size and rng.random() < 0.6:
            proto = rng.choice(same)
            for v in refs.get(proto, [])[:k // 2]:
                chosen.add(int(v))
        w = (1 + indeg[older]) * np.exp(-0.3 * (years[i] - years[older])) * (1 + 3 * (lab[older] == lab[i]))
        chosen.update(rng.choice(older, min(k, older.size), replace=False, p=w / w.sum()).tolist())
        r = np.array(sorted(chosen))
        refs[i] = r
        indeg[r] += 1
        edges.extend((i, int(v)) for v in r)
    return np.array(edges, np.int64), years.astype(np.int64), lab.astype(np.int64), X


def load(cfg):
    if cfg.get("synthetic"):
        return synthetic()
    return load_arxiv(cfg["data_dir"], cfg["arxiv_zip"])


def _group(e):
    if len(e) == 0:
        return {}
    e = e[np.argsort(e[:, 0], kind="stable")]
    u, idx = np.unique(e[:, 0], return_index=True)
    return dict(zip(u.tolist(), np.split(e[:, 1], idx[1:])))


class Snapshot:
    """Everything known at the end of year T."""

    def __init__(self, edges, year, label, T, min_refs=2):
        n = len(year)
        self.T = T
        past = year <= T
        self.past_ids = np.where(past)[0]
        ep = edges[past[edges[:, 0]] & past[edges[:, 1]]]
        self.indeg = np.bincount(ep[:, 1], minlength=n).astype(np.float32)
        self.recent = np.bincount(ep[year[ep[:, 0]] == T][:, 1], minlength=n).astype(np.float32)
        self.age = ((T + 1) - year).astype(np.float32)
        # birth memory: papers born in year T and their references to papers of year <= T-1
        mem = edges[(year[edges[:, 0]] == T) & (year[edges[:, 1]] <= T - 1)]
        self.mem_refs = _group(mem)
        self.mem_ids = np.array(sorted(self.mem_refs), dtype=np.int64)
        # arrivals: papers of year T+1 and their references into the past
        arr = edges[(year[edges[:, 0]] == T + 1) & past[edges[:, 1]]]
        ar = _group(arr)
        self.arr_refs = {p: r for p, r in ar.items() if len(r) >= min_refs}
        self.arrivals = np.array(sorted(self.arr_refs), dtype=np.int64)
