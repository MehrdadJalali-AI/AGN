"""Pre-registered AGN-CF headroom check. Implements PREREGISTRATION.md only."""
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
from torch_geometric.nn import GCNConv

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
DATASETS = ["Cora", "CiteSeer", "PubMed"]
BASELINES = ["B1", "B2", "B3", "B4"]


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(RESULTS, "run.log"), "a") as f:
        f.write(line + "\n")


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gcn(data):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = GCN(data.num_features, 16, data.y.max().item() + 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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
    correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    n_test = int(data.test_mask.sum())
    return model, logits, pred, correct, n_test


def degrees_of(edge_index, n):
    return torch.bincount(edge_index[0], minlength=n)


def neighbor_lists(edge_index, n):
    lists = [[] for _ in range(n)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        lists[u].append(v)
    return [sorted(set(vs)) for vs in lists]


def nearest_l2(feat, x):
    return float(torch.linalg.vector_norm(x - feat, dim=1).min())


def run_dataset(name):
    t0 = time.perf_counter()
    dataset = Planetoid(root=DATA, name=name, split="public")
    data = dataset[0]
    n = data.num_nodes
    model, logits, pred, n_correct, n_test = train_gcn(data)
    prob = torch.softmax(logits, dim=1)
    masked = logits.clone()
    masked[torch.arange(n), pred] = -torch.inf
    second = masked.argmax(dim=1)
    test_ids = torch.where(data.test_mask)[0]
    correct_ids = test_ids[pred[test_ids] == data.y[test_ids]].cpu().numpy()
    if len(correct_ids) < 200:
        raise SystemExit(f"{name}: only {len(correct_ids)} correct test nodes")
    targets = np.sort(np.random.default_rng(0).choice(correct_ids, 200, replace=False))
    deg = degrees_of(data.edge_index, n).cpu().numpy()
    deg_min, deg_max = int(deg.min()), int(deg.max())
    cap = int(np.floor(np.median(deg)))
    neigh = neighbor_lists(data.edge_index, n)
    train_ids = torch.where(data.train_mask)[0].cpu().numpy()
    y = data.y.cpu().numpy()
    x = data.x
    binary = bool(torch.all((x == 0) | (x == 1)))
    b4_rng = np.random.default_rng(0)
    rows = {b: {"flips": 0, "edges": [], "l2": [], "degree_ok": 0} for b in BASELINES}
    with torch.no_grad():
        for t in targets:
            t = int(t)
            desired = int(second[t])
            pool = train_ids[y[train_ids] == desired]
            if len(pool) == 0:
                raise SystemExit(f"{name}: desired class {desired} has no training node")
            conf = prob[pool, desired].cpu().numpy()
            proto = int(pool[np.lexsort((pool, -conf))[0]])
            mean = x[pool].mean(dim=0)
            if binary:
                mean = (mean >= 0.5).to(x.dtype)
            b4 = int(b4_rng.choice(train_ids))
            order = sorted(neigh[proto], key=lambda v: (-int(deg[v]), v))
            extra = [v for v in order if v != t][:cap]
            specs = {
                "B1": (x[proto], [t]),
                "B2": (x[proto], [t] + extra),
                "B3": (mean, [t]),
                "B4": (x[b4], [t]),
            }
            for b, (feat, nbrs) in specs.items():
                nbrs = list(dict.fromkeys(nbrs))
                x2 = torch.cat([x, feat.view(1, -1)], 0)
                new = n
                extra_e = []
                for v in nbrs:
                    extra_e.append([v, new])
                    extra_e.append([new, v])
                edge2 = torch.cat([data.edge_index, torch.tensor(extra_e, dtype=torch.long).t()], 1)
                out = model(x2, edge2)
                flipped = int(out[t].argmax()) == desired
                rows[b]["flips"] += int(flipped)
                rows[b]["edges"].append(len(nbrs))
                rows[b]["l2"].append(nearest_l2(feat, x))
                dnew = len(nbrs)
                rows[b]["degree_ok"] += int(deg_min <= dnew <= deg_max)
    summary = {}
    for b, r in rows.items():
        summary[b] = {
            "flips": r["flips"],
            "n": 200,
            "flip_rate": r["flips"] / 200,
            "edges_mean": float(np.mean(r["edges"])),
            "edges_min": int(min(r["edges"])),
            "edges_max": int(max(r["edges"])),
            "mean_nearest_l2": float(np.mean(r["l2"])),
            "degree_in_range": r["degree_ok"],
            "degree_in_range_rate": r["degree_ok"] / 200,
        }
    info = {
        "dataset": name,
        "test_correct": n_correct,
        "test_n": n_test,
        "test_accuracy": n_correct / n_test,
        "n_correct_test_pool": int(len(correct_ids)),
        "binary_features": binary,
        "degree_min": deg_min,
        "degree_max": deg_max,
        "degree_median": float(np.median(deg)),
        "b2_neighbor_cap": cap,
        "seconds": time.perf_counter() - t0,
        "baselines": summary,
    }
    log(f"{name}: test accuracy {n_correct}/{n_test} = {info['test_accuracy']}; "
        f"binary={binary}; degree range [{deg_min}, {deg_max}]; median={info['degree_median']}; B2 cap={cap}")
    for b in BASELINES:
        s = summary[b]
        log(f"{name} {b}: flips {s['flips']}/200 rate={s['flip_rate']} "
            f"edges_mean={s['edges_mean']} edges=[{s['edges_min']}, {s['edges_max']}] "
            f"mean_nearest_l2={s['mean_nearest_l2']} degree_in_range={s['degree_in_range']}/200")
    return info


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "environment.txt"), "w") as f:
        f.write(f"python {sys.version}\nnumpy {np.__version__}\ntorch {torch.__version__}\n"
                f"platform {platform.platform()}\ncpus {os.cpu_count()}\n")
    t0 = time.perf_counter()
    out = [run_dataset(name) for name in DATASETS]
    payload = {"datasets": out, "seconds_total": time.perf_counter() - t0}
    json.dump(payload, open(os.path.join(RESULTS, "headroom.json"), "w"), indent=2)
    log(f"finished in {payload['seconds_total']:.6f}s")


if __name__ == "__main__":
    main()
