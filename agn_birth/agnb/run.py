"""AGN kill test pipeline.

  python -m agnb.run prepare  CONFIG   # data + candidates for train/val (never touches the test year)
  python -m agnb.run train    CONFIG   # baselines on val, grid search + seeds for learned models
  python -m agnb.run final    CONFIG   # ONE-TIME evaluation on the held-out test year (writes FINAL_LOCK)
  python -m agnb.run report   CONFIG   # applies the pre-registered decision rule to saved results
"""
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .data import load, Snapshot
from .candidates import build, ceiling
from .models import BASELINES, LEARNED, EXPERTS, baseline_scores, make, collate, listwise_loss


def log(cfg, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(cfg["results_dir"], "run.log"), "a") as f:
        f.write(line + "\n")


def metrics(scores, it, Ks):
    order = np.argsort(-scores, kind="stable")
    y = it["y"][order]
    out = {}
    for K in Ks:
        hits = y[:K]
        out[f"R@{K}"] = hits.sum() / it["nref"]
        disc = 1.0 / np.log2(np.arange(2, K + 2))
        ideal = disc[:min(K, it["nref"])].sum()
        out[f"N@{K}"] = float((hits * disc[:len(hits)]).sum() / ideal)
    return out


def split_items(cfg, D, which, n, rng_seed):
    path = os.path.join(cfg["results_dir"], "cache", f"cand_{which}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    edges, year, label, X = D
    S = Snapshot(edges, year, label, cfg["splits"][which], cfg["min_refs"])
    pids = S.arrivals
    if n is not None and len(pids) > n:
        pids = np.sort(np.random.default_rng(rng_seed).choice(pids, n, replace=False))
    t = time.time()
    items = build(S, X, label, pids, cfg["candidates"])
    info = {"split": which, "cutoff": cfg["splits"][which], "n_eligible": int(len(S.arrivals)),
            "n_used": len(items), "n_past": int(len(S.past_ids)), "n_memory": int(len(S.mem_ids)),
            "mean_candidates": float(np.mean([len(i["cand"]) for i in items])),
            "ceiling_R@20": ceiling(items, 20), "ceiling_R@50": ceiling(items, 50), "build_s": time.time() - t}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump((items, info), open(path, "wb"))
    log(cfg, f"candidates {which}: {json.dumps(info)}")
    return items, info


def evaluate_baselines(items, Ks):
    rows = []
    for it in items:
        for b in BASELINES:
            rows.append({"p": it["p"], "label": it["label"], "method": b, "seed": -1,
                         **metrics(baseline_scores(b, it), it, Ks)})
    return rows


@torch.no_grad()
def evaluate_model(model, items, Xt, Ks, bs=32):
    model.eval()
    rows, gates = [], []
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        b = collate(chunk, Xt)
        score, g = model(b)
        score = score.numpy()
        for i, it in enumerate(chunk):
            rows.append({"p": it["p"], "label": it["label"], **metrics(score[i, :len(it["cand"])], it, Ks)})
            if g is not None:
                gates.append(g[i].numpy())
    return rows, (np.array(gates) if gates else None)


def train_one(cfg, name, hidden, lr, seed, tr, va, Xt):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make(name, Xt.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(len(tr))
        losses = []
        for s in range(0, len(tr), cfg["batch_size"]):
            b = collate([tr[i] for i in perm[s:s + cfg["batch_size"]]], Xt, cfg["train_negatives"], rng)
            score, _ = model(b)
            loss = listwise_loss(score, b)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        rows, _ = evaluate_model(model, va, Xt, [20])
        r = float(np.mean([x["R@20"] for x in rows]))
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_R@20": r})
        if r > best + 1e-5:
            best, best_state, bad = r, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best, hist


def cmd_prepare(cfg):
    D = load(cfg)
    split_items(cfg, D, "train", cfg["n_train"], 1)
    split_items(cfg, D, "val", cfg["n_val"], 2)


def cmd_train(cfg):
    D = load(cfg)
    Xt = torch.from_numpy(D[3]).float()
    tr, _ = split_items(cfg, D, "train", cfg["n_train"], 1)
    va, vinfo = split_items(cfg, D, "val", cfg["n_val"], 2)
    tr = [it for it in tr if it["y"].any()]
    Ks = cfg["K"]
    val_rows = evaluate_baselines(va, Ks)
    sel = {}
    mdir = os.path.join(cfg["results_dir"], "models")
    os.makedirs(mdir, exist_ok=True)
    grid_log = []
    for name in LEARNED:
        hs = [None] if name == "LR-Ranker" else cfg["grid"]["hidden"]
        best = None
        for h in hs:
            for lr in (cfg["grid"]["lr_linear"] if name == "LR-Ranker" else cfg["grid"]["lr"]):
                t = time.time()
                model, r, hist = train_one(cfg, name, h or 0, lr, cfg["seeds"][0], tr, va, Xt)
                grid_log.append({"model": name, "hidden": h, "lr": lr, "val_R@20": r, "epochs": len(hist),
                                 "time_s": time.time() - t, "history": hist})
                log(cfg, f"grid {name} hidden={h} lr={lr}: val R@20={r:.4f} ({time.time() - t:.0f}s)")
                if best is None or r > best[0]:
                    best = (r, h, lr, model)
        sel[name] = {"hidden": best[1], "lr": best[2]}
        for seed in cfg["seeds"]:
            if seed == cfg["seeds"][0]:
                model = best[3]
            else:
                model, r, _ = train_one(cfg, name, best[1] or 0, best[2], seed, tr, va, Xt)
                log(cfg, f"seed {name} seed={seed}: val R@20={r:.4f}")
            torch.save(model.state_dict(), os.path.join(mdir, f"{name}_seed{seed}.pt"))
            rows, _ = evaluate_model(model, va, Xt, Ks)
            val_rows += [{**r_, "method": name, "seed": seed} for r_ in rows]
    json.dump(sel, open(os.path.join(cfg["results_dir"], "selected_hparams.json"), "w"), indent=1)
    json.dump(grid_log, open(os.path.join(cfg["results_dir"], "grid_log.json"), "w"), indent=1)
    pd.DataFrame(val_rows).to_csv(os.path.join(cfg["results_dir"], "val_per_paper.csv"), index=False)
    v = pd.DataFrame(val_rows).groupby("method")[[f"R@{k}" for k in Ks] + [f"N@{k}" for k in Ks]].mean()
    log(cfg, "validation summary (model selection data; not the decision):\n" + v.round(4).to_string())


def cmd_final(cfg):
    lock = os.path.join(cfg["results_dir"], "FINAL_LOCK")
    done = os.path.exists(os.path.join(cfg["results_dir"], "test_per_paper.csv"))
    if os.path.exists(lock) and done:
        sys.exit(f"FINAL_LOCK exists ({open(lock).read().strip()}) and test results are saved. The test year was "
                 f"already evaluated; run `report` instead. Do not delete the lock or re-run.")
    if os.path.exists(lock):
        log(cfg, "FINAL_LOCK exists but no test results were saved (previous run crashed); resuming.")
    for name in LEARNED:
        for seed in cfg["seeds"]:
            if not os.path.exists(os.path.join(cfg["results_dir"], "models", f"{name}_seed{seed}.pt")):
                sys.exit("Missing trained models; run `train` first.")
    with open(lock, "a") as f:
        f.write(time.strftime("opened %Y-%m-%d %H:%M:%S\n"))
    D = load(cfg)
    Xt = torch.from_numpy(D[3]).float()
    te, tinfo = split_items(cfg, D, "test", cfg["n_test"], 3)
    Ks = cfg["K"]
    rows = evaluate_baselines(te, Ks)
    sel = json.load(open(os.path.join(cfg["results_dir"], "selected_hparams.json")))
    gate_rows = []
    for name in LEARNED:
        for seed in cfg["seeds"]:
            model = make(name, Xt.shape[1], sel[name]["hidden"] or 0)
            model.load_state_dict(torch.load(os.path.join(cfg["results_dir"], "models", f"{name}_seed{seed}.pt")))
            r, g = evaluate_model(model, te, Xt, Ks)
            rows += [{**x, "method": name, "seed": seed} for x in r]
            if name == "AGN" and g is not None:
                for it, gi in zip(te, g):
                    gate_rows.append({"p": it["p"], "label": it["label"], "seed": seed,
                                      **{e: float(v) for e, v in zip(EXPERTS, gi)}})
    pd.DataFrame(rows).to_csv(os.path.join(cfg["results_dir"], "test_per_paper.csv"), index=False)
    pd.DataFrame(gate_rows).to_csv(os.path.join(cfg["results_dir"], "test_gates.csv"), index=False)
    json.dump(tinfo, open(os.path.join(cfg["results_dir"], "test_info.json"), "w"), indent=1)
    log(cfg, "final test evaluated once; now run `report`.")


def cmd_report(cfg):
    from .report import decide
    print(decide(cfg))


def environment(cfg):
    with open(os.path.join(cfg["results_dir"], "environment.txt"), "w") as f:
        f.write(f"python {sys.version}\nnumpy {np.__version__}\npandas {pd.__version__}\ntorch {torch.__version__}\n"
                f"platform {platform.platform()}\nmachine {platform.machine()}\ncpus {os.cpu_count()}\n")


if __name__ == "__main__":
    cmd, cfg_path = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(cfg["results_dir"], exist_ok=True)
    environment(cfg)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    {"prepare": cmd_prepare, "train": cmd_train, "final": cmd_final, "report": cmd_report}[cmd](cfg)
    log(cfg, f"{cmd} finished in {time.time() - t0:.0f}s")
