"""Candidate generation and per-(paper, candidate) mechanism features. Uses only snapshot (<= T) information
plus the new paper's own text features. Identical candidate sets are used by every method."""
import numpy as np

FEATURES = ["pop", "fit", "age", "cos", "copy"]


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _topk(M, k):
    k = min(k, M.shape[1])
    idx = np.argpartition(-M, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(M, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


def build(S, X, label, pids, cc, chunk=256):
    Xu = unit(X).astype(np.float32)
    pop = S.past_ids[np.argsort(-S.indeg[S.past_ids], kind="stable")[:cc["top_pop"]]]
    fit = S.past_ids[np.argsort(-S.recent[S.past_ids], kind="stable")[:cc["top_fit"]]]
    Xp, Xm = Xu[S.past_ids], Xu[S.mem_ids]
    Q = cc["mem_k"]
    out = []
    for s in range(0, len(pids), chunk):
        ch = pids[s:s + chunk]
        sim_top = _topk(Xu[ch] @ Xp.T, cc["top_sim"])
        msim = Xu[ch] @ Xm.T
        mem_top = _topk(msim, Q)
        for i, p in enumerate(ch):
            mq = S.mem_ids[mem_top[i]]
            mc = msim[i, mem_top[i]].astype(np.float32)
            memrefs = [S.mem_refs[int(q)] for q in mq]
            cand = np.unique(np.concatenate([pop, fit, S.past_ids[sim_top[i]]] + memrefs))
            qi = np.concatenate([np.full(len(r), k, np.int16) for k, r in enumerate(memrefs)])
            ci = np.concatenate([np.searchsorted(cand, r) for r in memrefs]).astype(np.int32)
            w = np.exp(10 * (mc - mc.max())); w /= w.sum()
            copy = np.zeros(len(cand), np.float32)
            np.add.at(copy, ci, w[qi])
            refs = S.arr_refs[int(p)]
            F = np.stack([np.log1p(S.indeg[cand]), np.log1p(S.recent[cand]),
                          np.minimum(S.age[cand], 30) / 10.0, Xu[cand] @ Xu[p], copy], 1).astype(np.float32)
            out.append({"p": int(p), "label": int(label[p]), "cand": cand.astype(np.int64), "F": F,
                        "y": np.isin(cand, refs), "nref": int(len(refs)), "mq": mq, "mc": mc, "qi": qi, "ci": ci})
    return out


def ceiling(items, K):
    return float(np.mean([min(it["y"].sum(), K) / it["nref"] for it in items]))
