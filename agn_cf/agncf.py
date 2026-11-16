"""AGN-CF, frozen in PREREGISTRATION_AGNCF.md. Tune, then run the final test once."""
import csv
import json
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from headroom import DATA, DATASETS, GCN, degrees_of, neighbor_lists, train_gcn

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SELECTED = os.path.join(RESULTS, "agncf_selected.json")
LOCK = os.path.join(RESULTS, "AGNCF_FINAL_LOCK")
TEST_CSV = os.path.join(RESULTS, "agncf_test_per_target.csv")
DECISION = os.path.join(RESULTS, "AGNCF_DECISION.md")
EXAMPLES = os.path.join(RESULTS, "agncf_examples.json")
LOG_PATH = os.path.join(RESULTS, "agncf_run.log")
GRID = [{"lambda": lam, "steps": 16, "budget": budget}
        for lam in (0.2, 1.0, 5.0) for budget in (8, 32)]
ADAM_LR = 0.1
SNAP = 0.05
TAO_STEPS = 100
BASELINES = ["B1", "B2", "Tao-reimpl"]
METHODS = ["B1", "B2", "Tao-reimpl", "AGN-CF", "AGN-CF-noplaus", "random-anchor"]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gat(data):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = GAT(data.num_features, 8, int(data.y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)
    return model, logits, pred


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_data(name):
    return Planetoid(root=DATA, name=name, split="public")[0]


def plausibility_stats(x, deg):
    n = x.size(0)
    nn = torch.empty(n)
    for i in range(0, n, 256):
        dist = torch.cdist(x[i:i + 256], x)
        rows = torch.arange(dist.size(0))
        dist[rows, i + rows] = torch.inf
        nn[i:i + dist.size(0)] = dist.min(dim=1).values
    active = (x > 0).sum(dim=1).cpu().numpy()
    deg_np = deg.cpu().numpy()
    return {
        "l2_p95": float(np.percentile(nn.numpy(), 95, method="linear")),
        "active_p5": float(np.percentile(active, 5, method="linear")),
        "active_p95": float(np.percentile(active, 95, method="linear")),
        "deg_min": int(deg_np.min()),
        "deg_max": int(deg_np.max()),
        "deg_median": float(np.median(deg_np)),
        "extra_cap": max(0, int(np.floor(float(np.median(deg_np)))) - 1),
        "b2_cap": int(np.floor(float(np.median(deg_np)))),
    }


def headroom_ids(data, pred):
    test_ids = torch.where(data.test_mask)[0]
    correct = test_ids[pred[test_ids] == data.y[test_ids]].cpu().numpy()
    if len(correct) < 200:
        raise SystemExit(f"fewer than 200 correct test nodes for headroom ids")
    return np.sort(np.random.default_rng(0).choice(correct, 200, replace=False))


def draw_ids(correct, exclude, n, seed):
    pool = np.setdiff1d(np.asarray(correct), np.asarray(exclude), assume_unique=False)
    if len(pool) < n:
        raise SystemExit(f"pool has {len(pool)} nodes, need {n}")
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))


def correct_ids(data, pred):
    test_ids = torch.where(data.test_mask)[0]
    return test_ids[pred[test_ids] == data.y[test_ids]].cpu().numpy()


def second_class(logits):
    top = logits.argmax(dim=1)
    masked = logits.clone()
    masked[torch.arange(logits.size(0)), top] = -torch.inf
    return masked.argmax(dim=1)


def make_edge(edge_index, n, nbrs):
    extra = []
    for v in nbrs:
        extra.append((int(v), int(n)))
        extra.append((int(n), int(v)))
    extra_t = torch.tensor(extra, dtype=torch.long).t()
    return torch.cat([edge_index, extra_t], 1)


def forward_logits(model, x, edge_index, feat, nbrs):
    x2 = torch.cat([x, feat.view(1, -1)], 0)
    return model(x2, make_edge(edge_index, x.size(0), nbrs))


def margin_tensor(logits, target, desired):
    score = logits[target]
    mask = torch.ones(score.numel(), dtype=torch.bool)
    mask[desired] = False
    return score[desired] - score[mask].max()


def evaluate(model, x, edge_index, feat, nbrs, target, desired):
    with torch.no_grad():
        score = forward_logits(model, x, edge_index, feat, nbrs)[target]
        others = score.clone()
        others[desired] = -torch.inf
        margin = float(score[desired] - others.max())
        return int(score.argmax()) == int(desired), margin


def feature_state(feat, x, stats):
    dist = float(torch.linalg.vector_norm(x - feat, dim=1).min())
    active = int((feat > 0).sum())
    ok = dist <= stats["l2_p95"] and stats["active_p5"] <= active <= stats["active_p95"]
    return ok, dist, active


def discrete_from_soft(soft, anchor, binary):
    if binary:
        hard = (soft >= 0.5).to(anchor.dtype)
        edits = int((hard - anchor).abs().sum().item())
    else:
        delta = (soft - anchor).abs()
        hard = torch.where(delta >= SNAP, soft, anchor)
        edits = int((delta >= SNAP).sum().item())
    return hard, edits


def better(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a["flipped"] != b["flipped"]:
        return a if a["flipped"] else b
    ka = (a["edits"], a["edges"], -a["margin"], a["anchor"])
    kb = (b["edits"], b["edges"], -b["margin"], b["anchor"])
    return a if ka <= kb else b


def greedy(model, x, edge_index, feat, neighbors, target, desired, stats, enforce):
    nbrs = [int(target)]
    flipped, margin = evaluate(model, x, edge_index, feat, nbrs, target, desired)
    pool = [int(v) for v in neighbors if int(v) != int(target)]
    for _ in range(stats["extra_cap"]):
        if flipped:
            break
        best = None
        for v in pool:
            trial = nbrs + [v]
            if enforce and not (stats["deg_min"] <= len(trial) <= stats["deg_max"]):
                continue
            fl, m = evaluate(model, x, edge_index, feat, trial, target, desired)
            if best is None or m > best[0] or (m == best[0] and v < best[1]):
                best = (m, v, fl)
        if best is None or not best[0] > margin:
            break
        nbrs.append(best[1])
        pool.remove(best[1])
        margin = best[0]
        flipped = best[2]
    return nbrs, flipped, margin


def collect_vectors(model, x, edge_index, anchor, target, desired, lam, steps, budget, binary, fmax, stats, enforce,
                    include_base=True):
    found = []
    seen = set()

    def consider(hard):
        hard = hard.detach()
        edits = int((hard - anchor).abs().sum().item()) if binary else int(((hard - anchor).abs() >= SNAP).sum().item())
        if edits > budget:
            return
        key = np.ascontiguousarray(hard.cpu().numpy()).tobytes()
        if key in seen:
            return
        if enforce and not feature_state(hard, x, stats)[0]:
            return
        seen.add(key)
        found.append((edits, hard.clone()))

    if include_base:
        consider(anchor)
    if steps <= 0:
        return found
    theta = torch.zeros(anchor.numel(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=ADAM_LR)
    upper = 1.0 if binary else float(fmax)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        soft = (anchor + theta).clamp(0, upper)
        logits = forward_logits(model, x, edge_index, soft, [int(target)])
        loss = -margin_tensor(logits, int(target), int(desired)) + lam * (soft - anchor).abs().sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            hard, _ = discrete_from_soft((anchor + theta).clamp(0, upper), anchor, binary)
        consider(hard)
    return found


def one_edge_record(model, x, edge_index, anchor_id, anchor, target, desired, stats, enforce):
    if enforce and not feature_state(anchor, x, stats)[0]:
        return None
    flipped, margin = evaluate(model, x, edge_index, anchor, [int(target)], target, desired)
    if not flipped:
        return None
    return {"flipped": True, "edits": 0, "edges": 1, "margin": margin, "anchor": int(anchor_id),
            "feat": anchor, "nbrs": [int(target)]}


def search_anchor(model, x, edge_index, anchor_id, anchor, neighbors, target, desired, lam, steps, budget,
                  binary, fmax, stats, enforce, mode="full"):
    anchor = anchor.detach()
    if mode == "one":
        return one_edge_record(model, x, edge_index, anchor_id, anchor, target, desired, stats, enforce)
    best = None
    anchor_ok = (not enforce) or feature_state(anchor, x, stats)[0]
    if anchor_ok:
        nbrs, flipped, margin = greedy(model, x, edge_index, anchor, neighbors, target, desired, stats, enforce)
        best = {"flipped": flipped, "edits": 0, "edges": len(nbrs), "margin": margin,
                "anchor": int(anchor_id), "feat": anchor, "nbrs": nbrs}
        # A zero-edit flip cannot be beaten by a vector with more edits.
        if flipped or mode == "zero":
            return best
    elif mode == "zero":
        return None
    vectors = collect_vectors(model, x, edge_index, anchor, target, desired, lam, steps, budget,
                              binary, fmax, stats, enforce, include_base=not anchor_ok)
    flip_edits = None
    for edits, hard in sorted(vectors, key=lambda item: item[0]):
        if flip_edits is not None and edits > flip_edits:
            break
        nbrs, flipped, margin = greedy(model, x, edge_index, hard, neighbors, target, desired, stats, enforce)
        rec = {"flipped": flipped, "edits": edits, "edges": len(nbrs), "margin": margin,
               "anchor": int(anchor_id), "feat": hard, "nbrs": nbrs}
        best = better(best, rec)
        if flipped and flip_edits is None:
            flip_edits = edits
    return best


def anchors_for(train_ids, y, prob, desired, k=10):
    pool = train_ids[y[train_ids] == int(desired)]
    if len(pool) == 0:
        raise SystemExit(f"no training node of class {int(desired)}")
    conf = prob[pool, int(desired)].cpu().numpy()
    order = pool[np.lexsort((pool, -conf))]
    return [int(v) for v in order[:k]]


def finish_record(rec, x, stats):
    if rec is None:
        return {"returned": 0, "flipped": 0, "plausible": 0, "edits": "", "edges": "",
                "distance": "", "anchor": "", "nbrs": [], "edit_idx": []}
    ok, dist, _ = feature_state(rec["feat"], x, stats)
    degree_ok = stats["deg_min"] <= rec["edges"] <= stats["deg_max"]
    anchor = rec["anchor"]
    return {"returned": 1, "flipped": int(rec["flipped"]), "plausible": int(ok and degree_ok and rec["flipped"]),
            "edits": rec["edits"], "edges": rec["edges"], "distance": dist, "anchor": anchor,
            "nbrs": rec["nbrs"], "feat": rec["feat"]}


def explain(model, data, stats, neigh, train_ids, y, prob, target, desired, binary, fmax, cfg, kind, rng):
    x, edge_index = data.x, data.edge_index
    if kind == "Tao-reimpl":
        feat = torch.zeros(x.size(1), requires_grad=True)
        opt = torch.optim.Adam([feat], lr=ADAM_LR)
        flipped, margin = evaluate(model, x, edge_index, feat.detach(), [target], target, desired)
        for _ in range(TAO_STEPS):
            if flipped:
                break
            opt.zero_grad(set_to_none=True)
            logits = forward_logits(model, x, edge_index, feat, [target])
            loss = -margin_tensor(logits, target, desired)
            loss.backward()
            opt.step()
            flipped, margin = evaluate(model, x, edge_index, feat.detach(), [target], target, desired)
        feat = feat.detach()
        ok, dist, _ = feature_state(feat, x, stats)
        edits = int((feat.abs() > 1e-6).sum())
        return {"returned": 1, "flipped": int(flipped), "plausible": int(ok and stats["deg_min"] <= 1 <= stats["deg_max"] and flipped),
                "edits": edits, "edges": 1, "distance": dist, "anchor": "", "nbrs": [target], "feat": feat}
    if kind in ("B1", "B2"):
        chosen = anchors_for(train_ids, y, prob, desired, 1)
        anchor = chosen[0]
        feat = x[anchor].detach()
        if kind == "B1":
            nbrs = [target]
        else:
            extra = [v for v in sorted(neigh[anchor], key=lambda v: (-int(stats_deg(data, v)), v)) if v != target][:stats["b2_cap"]]
            nbrs = [target] + extra
        flipped, margin = evaluate(model, x, edge_index, feat, nbrs, target, desired)
        ok, dist, _ = feature_state(feat, x, stats)
        degree_ok = stats["deg_min"] <= len(nbrs) <= stats["deg_max"]
        return {"returned": 1, "flipped": int(flipped), "plausible": int(ok and degree_ok and flipped),
                "edits": 0, "edges": len(nbrs), "distance": dist, "anchor": anchor, "nbrs": nbrs, "feat": feat}
    if kind == "random-anchor":
        pool = train_ids[y[train_ids] == int(desired)]
        chosen = [int(rng.choice(pool))]
        enforce = True
    else:
        chosen = anchors_for(train_ids, y, prob, desired, 10)
        enforce = kind == "AGN-CF"
    best = None
    for anchor in chosen:
        mode = "full"
        if best is not None and best["flipped"] and best["edits"] == 0:
            mode = "one" if best["edges"] == 1 else "zero"
        rec = search_anchor(model, x, edge_index, anchor, x[anchor].detach().clone(), neigh[anchor], target, desired,
                            cfg["lambda"], cfg["steps"], cfg["budget"], binary, fmax, stats, enforce, mode)
        best = better(best, rec)
    out = finish_record(best, x, stats)
    if best is not None:
        delta = (best["feat"] - x[best["anchor"]]).abs()
        if binary:
            out["edit_idx"] = torch.where(delta > 0)[0].cpu().tolist()
        else:
            out["edit_idx"] = torch.where(delta >= SNAP)[0].cpu().tolist()
        out["nbrs"] = best["nbrs"]
    else:
        out["edit_idx"] = []
    return out


def stats_deg(data, v):
    return int(degrees_of(data.edge_index, data.num_nodes)[v])


def prepare(name, model_name, gcn_pack=None):
    data = load_data(name)
    t0 = time.perf_counter()
    if model_name == "GCN":
        model, logits, pred, n_correct, n_test = train_gcn(data)
        log(f"{name} GCN test accuracy {n_correct}/{n_test}")
    else:
        model, logits, pred = train_gat(data)
        n_correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
        n_test = int(data.test_mask.sum())
        log(f"{name} GAT test accuracy {n_correct}/{n_test}")
    freeze(model)
    deg = degrees_of(data.edge_index, data.num_nodes)
    stats = plausibility_stats(data.x, deg)
    stats["seconds_stats"] = time.perf_counter() - t0
    neigh = neighbor_lists(data.edge_index, data.num_nodes)
    deg_np = deg.cpu().numpy()
    binary = bool(torch.all((data.x == 0) | (data.x == 1)))
    fmax = float(data.x.max())
    prob = torch.softmax(logits, dim=1)
    second = second_class(logits)
    train_ids = torch.where(data.train_mask)[0].cpu().numpy()
    gcn_pred = pred if gcn_pack is None else gcn_pack["pred"]
    hid = headroom_ids(data, gcn_pred)
    log(f"{name} plausibility l2_p95={stats['l2_p95']} active=[{stats['active_p5']}, {stats['active_p95']}] "
        f"degree=[{stats['deg_min']}, {stats['deg_max']}] median={stats['deg_median']} extra_cap={stats['extra_cap']}")
    return {"data": data, "model": model, "pred": pred, "prob": prob, "second": second,
            "stats": stats, "neigh": neigh, "deg_np": deg_np, "binary": binary, "fmax": fmax,
            "train_ids": train_ids, "y": data.y.cpu().numpy(), "headroom": hid,
            "correct": correct_ids(data, pred)}


def stats_deg_from(deg_np, v):
    return int(deg_np[v])


def explain_dispatch(pack, target, cfg, kind, rng):
    # B2 needs degrees from pack["deg_np"]; patch the closure used above via stats_deg by passing deg in stats.
    return explain(pack["model"], pack["data"], pack["stats"], pack["neigh"], pack["train_ids"], pack["y"],
                   pack["prob"], int(target), int(pack["second"][int(target)]), pack["binary"], pack["fmax"],
                   cfg, kind, rng)


def run_method(pack, targets, cfg, kind):
    rng = np.random.default_rng(3)
    rows = []
    t0 = time.perf_counter()
    # local B2 degree order uses pack deg_np; rewrite B2 here to avoid the helper bug
    for target in targets:
        target = int(target)
        desired = int(pack["second"][target])
        if kind == "B2":
            anchor = anchors_for(pack["train_ids"], pack["y"], pack["prob"], desired, 1)[0]
            feat = pack["data"].x[anchor].detach()
            extra = [v for v in sorted(pack["neigh"][anchor], key=lambda v: (-int(pack["deg_np"][v]), v)) if v != target]
            nbrs = [target] + extra[:pack["stats"]["b2_cap"]]
            flipped, _ = evaluate(pack["model"], pack["data"].x, pack["data"].edge_index, feat, nbrs, target, desired)
            ok, dist, _ = feature_state(feat, pack["data"].x, pack["stats"])
            degree_ok = pack["stats"]["deg_min"] <= len(nbrs) <= pack["stats"]["deg_max"]
            rec = {"returned": 1, "flipped": int(flipped), "plausible": int(ok and degree_ok and flipped),
                   "edits": 0, "edges": len(nbrs), "distance": dist, "anchor": anchor, "nbrs": nbrs,
                   "feat": feat, "edit_idx": []}
        else:
            rec = explain(pack["model"], pack["data"], pack["stats"], pack["neigh"], pack["train_ids"], pack["y"],
                          pack["prob"], target, desired, pack["binary"], pack["fmax"], cfg, kind, rng)
        rec["target"] = target
        rec["desired"] = desired
        rows.append(rec)
    return rows, time.perf_counter() - t0


def summarise(rows):
    n = len(rows)
    flips = sum(r["flipped"] for r in rows)
    plausible = sum(r["plausible"] for r in rows)
    used = [r for r in rows if r["returned"]]
    def mean(key):
        if not used:
            return None
        return float(np.mean([r[key] for r in used]))
    return {"n": n, "flips": flips, "plausible_flips": plausible,
            "raw_flip_rate": flips / n, "plausible_flip_rate": plausible / n,
            "mean_edits": mean("edits"), "mean_edges": mean("edges"), "mean_distance": mean("distance"),
            "returned": len(used)}


def cmd_tune():
    os.makedirs(RESULTS, exist_ok=True)
    log("tune start")
    pack = prepare("Cora", "GCN")
    tune_ids = draw_ids(pack["correct"], pack["headroom"], 100, 1)
    scored = []
    for i, cfg in enumerate(GRID):
        t0 = time.perf_counter()
        rows, _ = run_method(pack, tune_ids, cfg, "AGN-CF")
        summary = summarise(rows)
        summary.update(cfg)
        summary["index"] = i
        summary["seconds"] = time.perf_counter() - t0
        scored.append(summary)
        log(f"tune {cfg}: plausible {summary['plausible_flips']}/100 raw {summary['flips']}/100 "
            f"edits={summary['mean_edits']} edges={summary['mean_edges']} ({summary['seconds']:.1f}s)")
    def key(s):
        edits = s["mean_edits"] if s["mean_edits"] is not None else 1e18
        edges = s["mean_edges"] if s["mean_edges"] is not None else 1e18
        return (-s["plausible_flip_rate"], -s["raw_flip_rate"], edits, edges, s["index"])
    chosen = sorted(scored, key=key)[0]
    payload = {"lambda": chosen["lambda"], "steps": chosen["steps"], "budget": chosen["budget"],
               "tao": "reimplementation", "tuning_ids": [int(v) for v in tune_ids],
               "grid": scored, "chosen_tuning": {k: chosen[k] for k in
               ("plausible_flips", "flips", "raw_flip_rate", "plausible_flip_rate", "mean_edits", "mean_edges", "mean_distance")}}
    json.dump(payload, open(SELECTED, "w"), indent=2)
    log(f"selected lambda={chosen['lambda']} steps={chosen['steps']} budget={chosen['budget']}")


def write_decision(summary, seconds):
    lines = ["# AGN-CF final test", "",
             "Tao et al. is a reimplementation. Plausible flip rate is the primary metric. "
             "Means are over targets for which a node was returned.", ""]
    for model in ("GCN", "GAT"):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("```")
        lines.append(f"{'dataset':<10} {'method':<16} {'plausible':>12} {'raw':>12} {'edits':>14} {'edges':>10} {'distance':>14}")
        for name in DATASETS:
            for method in METHODS:
                s = summary[model][name][method]
                edits = "NA" if s["mean_edits"] is None else repr(s["mean_edits"])
                edges = "NA" if s["mean_edges"] is None else repr(s["mean_edges"])
                dist = "NA" if s["mean_distance"] is None else repr(s["mean_distance"])
                lines.append(f"{name:<10} {method:<16} {s['plausible_flips']:>4}/{s['n']:<7} "
                             f"{s['flips']:>4}/{s['n']:<7} {edits:>14} {edges:>10} {dist:>14}")
        lines.append("```")
        lines.append("")
    gcn_ok = True
    lines.append("## GCN conditions")
    lines.append("")
    for name in DATASETS:
        rates = {m: summary["GCN"][name][m]["plausible_flip_rate"] for m in BASELINES}
        best_m = max(BASELINES, key=lambda m: (rates[m], -BASELINES.index(m)))
        best = rates[best_m]
        agn = summary["GCN"][name]["AGN-CF"]["plausible_flip_rate"]
        ok = agn >= best + 0.15 and agn >= 1.25 * best
        gcn_ok = gcn_ok and ok
        lines.append(f"- {name}: best baseline {best_m} plausible flip rate = {best!r}; "
                     f"AGN-CF = {agn!r}; +15 points and ≥ 1.25×: {'PASS' if ok else 'FAIL'}")
    gat_wins = 0
    lines.append("")
    lines.append("## GAT conditions")
    lines.append("")
    for name in DATASETS:
        rates = {m: summary["GAT"][name][m]["plausible_flip_rate"] for m in BASELINES}
        best_m = max(BASELINES, key=lambda m: (rates[m], -BASELINES.index(m)))
        best = rates[best_m]
        agn = summary["GAT"][name]["AGN-CF"]["plausible_flip_rate"]
        gain = agn - best
        ok = gain >= 0.05
        gat_wins += int(ok)
        lines.append(f"- {name}: best baseline {best_m} = {best!r}; AGN-CF = {agn!r}; "
                     f"gain = {gain!r}; ≥ +5 points: {'PASS' if ok else 'FAIL'}")
    go = gcn_ok and gat_wins >= 2
    lines += ["", f"GCN condition on all three datasets: {'PASS' if gcn_ok else 'FAIL'}",
              f"GAT gains of at least +5 points: {gat_wins} of 3",
              "", f"## Verdict: **{'GO' if go else 'NO-GO — stop AGN-CF (no paper)'}**",
              "", f"Final-test runtime seconds: {seconds!r}", ""]
    open(DECISION, "w").write("\n".join(lines))
    log("verdict " + ("GO" if go else "NO-GO"))


def cmd_final():
    if os.path.exists(LOCK):
        raise SystemExit(f"AGNCF_FINAL_LOCK exists. The final test must not be run again.")
    if not os.path.exists(SELECTED):
        raise SystemExit("missing results/agncf_selected.json; run tune first")
    os.makedirs(RESULTS, exist_ok=True)
    cfg = json.load(open(SELECTED))
    use = {"lambda": cfg["lambda"], "steps": cfg["steps"], "budget": cfg["budget"]}
    tune_ids = np.array(cfg["tuning_ids"], dtype=int)
    log(f"final preparing with {use}")
    prepared = {}
    targets = {}
    for name in DATASETS:
        gcn = prepare(name, "GCN")
        gat = prepare(name, "GAT", gcn)
        prepared[("GCN", name)] = gcn
        prepared[("GAT", name)] = gat
        gcn_exclude = gcn["headroom"] if name != "Cora" else np.union1d(gcn["headroom"], tune_ids)
        gat_exclude = gcn["headroom"] if name != "Cora" else np.union1d(gcn["headroom"], tune_ids)
        targets[("GCN", name)] = draw_ids(gcn["correct"], gcn_exclude, 200, 2)
        targets[("GAT", name)] = draw_ids(gat["correct"], gat_exclude, 200, 2)
        if name == "Cora" and len(np.intersect1d(targets[("GCN", name)], tune_ids)):
            raise SystemExit("Cora GCN test intersects the tuning set")
        if name == "Cora" and len(np.intersect1d(targets[("GAT", name)], tune_ids)):
            raise SystemExit("Cora GAT test intersects the tuning set")
    with open(LOCK, "w") as f:
        f.write(time.strftime("opened %Y-%m-%d %H:%M:%S\n"))
    log("final lock written")
    t0 = time.perf_counter()
    summary = {m: {} for m in ("GCN", "GAT")}
    examples = []
    fields = ["model", "dataset", "method", "target", "desired", "anchor", "flipped", "plausible",
              "edits", "edges", "distance", "returned"]
    with open(TEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for model_name in ("GCN", "GAT"):
            for name in DATASETS:
                summary[model_name][name] = {}
                pack = prepared[(model_name, name)]
                ids = targets[(model_name, name)]
                for method in METHODS:
                    rows, seconds = run_method(pack, ids, use, method)
                    sm = summarise(rows)
                    sm["seconds"] = seconds
                    summary[model_name][name][method] = sm
                    log(f"final {model_name} {name} {method}: plausible {sm['plausible_flips']}/{sm['n']} "
                        f"raw {sm['flips']}/{sm['n']} edits={sm['mean_edits']} edges={sm['mean_edges']} "
                        f"dist={sm['mean_distance']} ({seconds:.1f}s)")
                    for rec in rows:
                        row = {k: rec.get(k, "") for k in fields}
                        row["model"] = model_name
                        row["dataset"] = name
                        row["method"] = method
                        writer.writerow(row)
                        if (len(examples) < 5 and model_name == "GCN" and method == "AGN-CF"
                                and rec["flipped"] and rec["plausible"]):
                            examples.append({"dataset": name, "target": rec["target"], "desired": rec["desired"],
                                             "anchor": rec["anchor"], "edited_features": rec.get("edit_idx", []),
                                             "edges": rec["nbrs"], "edits": rec["edits"]})
                    f.flush()
    json.dump({"summary": summary, "examples": examples, "settings": use}, open(os.path.join(RESULTS, "agncf_test_summary.json"), "w"), indent=2)
    json.dump(examples, open(EXAMPLES, "w"), indent=2)
    write_decision(summary, time.perf_counter() - t0)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "agncf_environment.txt"), "w") as f:
        import torch_geometric
        f.write(f"python {sys.version}\nnumpy {np.__version__}\ntorch {torch.__version__}\n"
                f"torch_geometric {torch_geometric.__version__}\nplatform {platform.platform()}\ncpus {os.cpu_count()}\n")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    cmd = sys.argv[1]
    {"tune": cmd_tune, "final": cmd_final}[cmd]()
