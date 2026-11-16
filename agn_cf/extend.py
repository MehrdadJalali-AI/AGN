"""AGN extension run, frozen in PREREGISTRATION_EXTEND.md."""
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from torch_geometric.datasets import Amazon, Coauthor
from torch_geometric.nn import SAGEConv

import agncf
import confirm
import extend_fast
from headroom import GCN

agncf.LOG_PATH = os.path.join(agncf.RESULTS, "extend_run.log")
agncf.forward_logits = extend_fast.forward_logits

LOCK = os.path.join(agncf.RESULTS, "EXTEND_LOCK")
SELECTED = os.path.join(agncf.RESULTS, "extend_selected.json")
CSV_PATH = os.path.join(agncf.RESULTS, "extend_per_target.csv")
DECISION = os.path.join(agncf.RESULTS, "EXTEND_DECISION.md")
FINAL_LOCK = open(agncf.LOCK).read()
CONFIRM_LOCK = open(confirm.LOCK).read()
CFG = {"lambda": 0.2, "steps": 16, "budget": 32}
GRID = [0.01, 0.05, 0.1]
DATASETS = [
    ("Amazon-Computers", Amazon, "Computers"),
    ("Amazon-Photo", Amazon, "Photo"),
    ("Coauthor-CS", Coauthor, "CS"),
    ("Coauthor-Physics", Coauthor, "Physics"),
]
MODELS = ("GCN", "GAT", "SAGE")
METHODS = ["B1", "B2", "Copy-best-of-10", "Copy-best-of-10-noedges", "constrained-attack", "AGN"]
PLAUSIBILITY = {}


def log(msg):
    agncf.log(msg)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


def split_masks(n):
    perm = np.random.default_rng(0).permutation(n)
    n_part = n // 5
    n_train = n - 2 * n_part
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    train[perm[:n_train]] = True
    val[perm[n_train:n_train + n_part]] = True
    test[perm[n_train + n_part:]] = True
    return train, val, test


def load_dataset(name):
    kind, cls, key = next(item for item in DATASETS if item[0] == name)
    data = cls(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), key)[0]
    data.y = data.y.long()
    train, val, test = split_masks(data.num_nodes)
    data.train_mask, data.val_mask, data.test_mask = train, val, test
    return data


def train_model(kind, data):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    out_dim = int(data.y.max()) + 1
    if kind == "GCN":
        model, lr = GCN(data.num_features, 16, out_dim), 0.01
    elif kind == "GAT":
        model, lr = agncf.GAT(data.num_features, 8, out_dim), 0.005
    else:
        model, lr = SAGE(data.num_features, 64, out_dim), 0.01
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            log(f"{kind} epoch {epoch + 1} loss={float(loss.detach()):.4f}")
    model.eval()
    agncf.freeze(model)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return model, logits


def loo_nn(x):
    x = x.float()
    norms = (x * x).sum(1)
    out = torch.empty(x.size(0))
    step = 128
    for start in range(0, x.size(0), step):
        sl = x[start:start + step]
        dist = norms[start:start + step].unsqueeze(1) + norms.unsqueeze(0) - 2 * (sl @ x.T)
        dist.clamp_(min=0)
        rows = torch.arange(dist.size(0))
        dist[rows, start + rows] = torch.inf
        out[start:start + dist.size(0)] = dist.min(1).values.sqrt()
        if start % (step * 20) == 0:
            log(f"nearest-neighbour rows {start}/{x.size(0)}")
    return out


def plausibility(data):
    deg = agncf.degrees_of(data.edge_index, data.num_nodes).cpu()
    nn = loo_nn(data.x.cpu())
    active = (data.x > 0).sum(1).cpu().numpy()
    deg_np = deg.numpy()
    median = float(np.median(deg_np))
    return {
        "l2_p95": float(np.percentile(nn.numpy(), 95, method="linear")),
        "active_p5": float(np.percentile(active, 5, method="linear")),
        "active_p95": float(np.percentile(active, 95, method="linear")),
        "deg_min": int(deg_np.min()),
        "deg_max": int(deg_np.max()),
        "deg_median": median,
        "extra_cap": max(0, int(np.floor(median)) - 1),
        "b2_cap": int(np.floor(median)),
    }, deg


def check_fast(model, data):
    target = int(torch.where(data.train_mask)[0][0])
    feat = data.x[target].detach().clone()
    nbrs = [target]
    extra = torch.tensor([[target, data.num_nodes], [data.num_nodes, target]])
    with torch.no_grad():
        full = model(torch.cat([data.x, feat.view(1, -1)], 0), torch.cat([data.edge_index, extra], 1))[target]
        local = model._fast.logits(feat, nbrs)
    gap = float((full - local).abs().max())
    log(f"local-logit check {model.__class__.__name__} max abs {gap}")
    if gap > 1e-3:
        raise SystemExit(f"local logits differ from the full model by {gap}")


def prepare(name, kind):
    data = load_dataset(name)
    t0 = time.perf_counter()
    model, logits = train_model(kind, data)
    pred = logits.argmax(1)
    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    n_test = int(data.test_mask.sum())
    log(f"{name} {kind} test accuracy {correct}/{n_test} train {time.perf_counter() - t0:.1f}s")
    extend_fast.attach(model, data.x, data.edge_index)
    check_fast(model, data)
    if name not in PLAUSIBILITY:
        PLAUSIBILITY[name] = plausibility(data)
    stats, deg = PLAUSIBILITY[name]
    binary = bool(torch.all((data.x == 0) | (data.x == 1)))
    prob = torch.softmax(logits, 1)
    log(f"{name} plausibility l2_p95={stats['l2_p95']} active=[{stats['active_p5']}, {stats['active_p95']}] "
        f"degree=[{stats['deg_min']}, {stats['deg_max']}] median={stats['deg_median']} extra_cap={stats['extra_cap']} "
        f"binary={binary}")
    return {"data": data, "model": model, "pred": pred, "prob": prob, "second": agncf.second_class(logits),
            "stats": stats, "neigh": agncf.neighbor_lists(data.edge_index, data.num_nodes),
            "deg_np": deg.numpy(), "binary": binary, "fmax": float(data.x.max()),
            "train_ids": torch.where(data.train_mask)[0].cpu().numpy(), "y": data.y.cpu().numpy(),
            "correct": agncf.correct_ids(data, pred), "accuracy": (correct, n_test), "logits": logits}


def run_method(pack, targets, method):
    if method == "AGN":
        return agncf.run_method(pack, targets, CFG, "AGN-CF")
    if method in ("B1", "B2"):
        return agncf.run_method(pack, targets, CFG, method)
    rows = []
    t0 = time.perf_counter()
    for target in targets:
        target = int(target)
        desired = int(pack["second"][target])
        if method == "Copy-best-of-10":
            rec = confirm.copy_search(pack, target, desired, True)
        elif method == "Copy-best-of-10-noedges":
            rec = confirm.copy_search(pack, target, desired, False)
        else:
            rec = confirm.constrained_attack(pack, target, desired)
        rec["target"] = target
        rec["desired"] = desired
        rows.append(rec)
    return rows, time.perf_counter() - t0


def fit_detector(data, deg):
    rows = np.concatenate([data.x.cpu().numpy(), deg.numpy().reshape(-1, 1)], 1)
    detector = IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=0, n_jobs=1)
    detector.fit(rows)
    return detector


def outlier_stats(rows, detector):
    mats = []
    for rec in rows:
        if not rec["returned"]:
            continue
        feat = rec["feat"].detach().cpu().flatten().numpy()
        mats.append(np.concatenate([feat, [float(len(rec["nbrs"]))]]))
    if not mats:
        return None, 0, 0
    pred = detector.predict(np.stack(mats))
    flagged = int((pred == -1).sum())
    return flagged / len(mats), flagged, len(mats)


def tune():
    log("tune start")
    pack = prepare("Amazon-Photo", "GCN")
    ids = agncf.draw_ids(pack["correct"], np.array([], dtype=int), 100, 1)
    rows, seconds = run_method(pack, ids, "AGN")
    summary = agncf.summarise(rows)
    summary["seconds"] = seconds
    grid = []
    for i, lr in enumerate(GRID):
        row = dict(summary)
        row.update({"continuous_lr": lr, "index": i, "binary_editor": True,
                    "seconds": seconds if i == 0 else 0.0})
        grid.append(row)
        log(f"tune lr={lr}: plausible {summary['plausible_flips']}/100 raw {summary['flips']}/100 "
            f"edits={summary['mean_edits']} (binary editor, learning rate unused)")
    chosen = grid[0]
    payload = {"continuous_lr": chosen["continuous_lr"], "binary_tuning_set": True,
               "tuning_ids": [int(v) for v in ids], "grid": grid,
               "chosen_tuning": {k: chosen[k] for k in ("plausible_flips", "flips", "plausible_flip_rate",
                                                        "raw_flip_rate", "mean_edits", "mean_edges", "mean_distance")}}
    json.dump(payload, open(SELECTED, "w"), indent=2)
    log(f"selected continuous_lr={chosen['continuous_lr']}")


def write_decision(summary, seconds, meta):
    lines = ["# AGN extension", "",
             "Continuous learning rate selected on Amazon-Photo: "
             f"{meta['continuous_lr']}. The tuning set is binary, so every grid row used the existing binary editor "
             "and the earliest row was selected. λ = 0.2, steps = 16, budget = 32 were not retuned.",
             "GANI: not run. The official repository https://github.com/alexfanjn/GANI trains a deeprobust GCN on the "
             "nettack split, injects 10 nodes, and searches a global homophily objective.",
             "", "Feature types and test accuracy:", ""]
    for name, _, _ in DATASETS:
        kind = "binary" if meta["binary"][name] else "continuous"
        lines.append(f"- {name}: {kind}")
        for model in MODELS:
            correct, n_test = meta["accuracy"][(model, name)]
            lines.append(f"  - {model}: {correct}/{n_test}")
    lines += ["", "Plausible flip rate is the primary metric. Means are over returned nodes. "
              "Outlier rate is the fraction of returned nodes flagged by the isolation forest.", ""]
    for model in MODELS:
        lines += [f"## {model}", "", "```",
                  f"{'dataset':<18} {'method':<24} {'plausible':>12} {'raw':>12} {'edits':>12} {'edges':>10} {'distance':>14} {'outlier':>10}"]
        for name, _, _ in DATASETS:
            for method in METHODS:
                s = summary[model][name][method]
                edits = "NA" if s["mean_edits"] is None else repr(s["mean_edits"])
                edges = "NA" if s["mean_edges"] is None else repr(s["mean_edges"])
                dist = "NA" if s["mean_distance"] is None else repr(s["mean_distance"])
                out = "NA" if s["outlier_rate"] is None else repr(s["outlier_rate"])
                lines.append(f"{name:<18} {method:<24} {s['plausible_flips']:>4}/{s['n']:<7} "
                             f"{s['flips']:>4}/{s['n']:<7} {edits:>12} {edges:>10} {dist:>14} {out:>10}")
        lines += ["```", ""]
    wins = 0
    lines.append("## Conditions")
    lines.append("")
    for name, _, _ in DATASETS:
        agn = sum(summary[m][name]["AGN"]["plausible_flips"] for m in MODELS)
        copy = sum(summary[m][name]["Copy-best-of-10"]["plausible_flips"] for m in MODELS)
        attack = sum(summary[m][name]["constrained-attack"]["plausible_flips"] for m in MODELS)
        best = max(copy, attack)
        ok = agn >= best + 60
        wins += int(ok)
        lines.append(f"- {name}: AGN {agn}/600, Copy-best-of-10 {copy}/600, constrained-attack {attack}/600, "
                     f"best {best}/600; +60 flips: {'PASS' if ok else 'FAIL'}")
    gaps = []
    for name, _, _ in DATASETS:
        agn = np.mean([summary[m][name]["AGN"]["outlier_rate"] for m in MODELS])
        copy = np.mean([summary[m][name]["Copy-best-of-10"]["outlier_rate"] for m in MODELS])
        gaps.append(float(agn - copy))
        lines.append(f"- {name} outlier gap AGN − Copy-best-of-10 = {gaps[-1]!r}")
    gap = float(np.mean(gaps))
    outlier_ok = gap <= 0.10 + 1e-9
    lines.append(f"- mean outlier gap = {gap!r}; ≤ 0.10: {'PASS' if outlier_ok else 'FAIL'}")
    go = wins >= 3 and outlier_ok
    verdict = "GO" if go else "NO-GO: AGN is limited to binary features. No paper."
    lines += ["", f"Datasets with the +60 flip margin: {wins} of 4",
              "", f"## Verdict: **{verdict}**", "", f"Final-run runtime seconds: {seconds!r}", ""]
    open(DECISION, "w").write("\n".join(lines))
    log("verdict " + ("GO" if go else "NO-GO"))


def final():
    if os.path.exists(LOCK):
        raise SystemExit("EXTEND_LOCK exists. The final extension run must not be run again.")
    if open(agncf.LOCK).read() != FINAL_LOCK or open(confirm.LOCK).read() != CONFIRM_LOCK:
        raise SystemExit("a closed lock changed before the extension run")
    saved = json.load(open(SELECTED))
    tune_ids = np.array(saved["tuning_ids"], dtype=int)
    log(f"final preparing, continuous_lr={saved['continuous_lr']}")
    packs = {}
    meta = {"continuous_lr": saved["continuous_lr"], "binary": {}, "accuracy": {}}
    for name, _, _ in DATASETS:
        for kind in MODELS:
            pack = prepare(name, kind)
            packs[(kind, name)] = pack
            meta["binary"][name] = pack["binary"]
            meta["accuracy"][(kind, name)] = pack["accuracy"]
    targets = {}
    for kind in MODELS:
        for name, _, _ in DATASETS:
            exclude = tune_ids if name == "Amazon-Photo" else np.array([], dtype=int)
            chosen = agncf.draw_ids(packs[(kind, name)]["correct"], exclude, 200, 2)
            if name == "Amazon-Photo" and len(np.intersect1d(chosen, tune_ids)):
                raise SystemExit("Amazon-Photo final targets intersect the tuning set")
            targets[(kind, name)] = chosen
            log(f"targets {kind} {name}: {len(chosen)}")
    with open(LOCK, "w") as f:
        f.write(time.strftime("opened %Y-%m-%d %H:%M:%S\n"))
    log("extend lock written")
    t0 = time.perf_counter()
    summary = {m: {} for m in MODELS}
    fields = ["model", "dataset", "method", "target", "desired", "anchor", "flipped", "plausible",
              "edits", "edges", "distance", "returned"]
    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for name, _, _ in DATASETS:
            detector = fit_detector(packs[("GCN", name)]["data"], torch.tensor(packs[("GCN", name)]["deg_np"]))
            for kind in MODELS:
                summary[kind][name] = {}
                pack = packs[(kind, name)]
                ids = targets[(kind, name)]
                for method in METHODS:
                    rows, seconds = run_method(pack, ids, method)
                    sm = agncf.summarise(rows)
                    rate, flagged, returned = outlier_stats(rows, detector)
                    sm.update({"seconds": seconds, "outlier_rate": rate, "outlier_flagged": flagged,
                               "outlier_returned": returned})
                    summary[kind][name][method] = sm
                    log(f"extend {kind} {name} {method}: plausible {sm['plausible_flips']}/{sm['n']} "
                        f"raw {sm['flips']}/{sm['n']} edits={sm['mean_edits']} outlier={rate} ({seconds:.1f}s)")
                    for rec in rows:
                        row = {k: rec.get(k, "") for k in fields}
                        row["model"] = kind
                        row["dataset"] = name
                        row["method"] = method
                        writer.writerow(row)
                    handle.flush()
                    del rows
    json.dump({"summary": summary, "meta": {"continuous_lr": saved["continuous_lr"],
                                            "binary": meta["binary"],
                                            "accuracy": {f"{a}|{b}": list(c) for (a, b), c in meta["accuracy"].items()}}},
              open(os.path.join(agncf.RESULTS, "extend_summary.json"), "w"), indent=2)
    write_decision(summary, time.perf_counter() - t0, meta)
    if open(agncf.LOCK).read() != FINAL_LOCK or open(confirm.LOCK).read() != CONFIRM_LOCK:
        raise SystemExit("a closed lock changed during the extension run")


def main():
    if os.path.exists(LOCK):
        raise SystemExit("EXTEND_LOCK exists. The final extension run must not be run again.")
    os.makedirs(agncf.RESULTS, exist_ok=True)
    if not os.path.exists(SELECTED):
        tune()
    final()


if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    main()
