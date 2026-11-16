"""AGN-Triad kill test.

  python -m agnt.run prepare CONFIG   # corpus + train/val tasks (never builds the test period)
  python -m agnt.run train   CONFIG   # model selection on validation; seeds
  python -m agnt.run final   CONFIG   # ONE-TIME evaluation on the two test deciles (writes FINAL_LOCK)
  python -m agnt.run report  CONFIG   # pre-registered decision rule
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
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import load_scholp, synthetic, Corpus
from .tasks import build_task, copy_scores, average_precision
from .models import make

TORCH_MODELS = ["PairAdditive", "MLP", "AGN-Triad"]


def log(cfg, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(cfg["results_dir"], "run.log"), "a") as f:
        f.write(line + "\n")


def corpus(cfg):
    p = os.path.join(cfg["results_dir"], "cache", "corpus.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    ev = synthetic() if cfg["synthetic"] else load_scholp(cfg["raw_dir"], cfg["dataset"])
    C = Corpus(ev, cfg["n_tags"], cfg["vocab_until"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pickle.dump(C, open(p, "wb"))
    log(cfg, f"corpus: {C.n} events, {C.V} tags kept, {len(C.tr)} triple occurrences")
    return C


def tasks(cfg, split):
    p = os.path.join(cfg["results_dir"], "cache", f"tasks_{split}.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    C = corpus(cfg)
    out = []
    for i, fr in enumerate(cfg["cutoffs"][split]):
        t0 = time.time()
        t = build_task(C, fr, cfg, seed=1000 + int(round(fr * 100)))
        out.append(t)
        log(cfg, f"task {split} cutoff={fr}: eligible={t['n_eligible']} positives={t['n_pos']} "
                 f"base_rate={t['n_pos'] / max(t['n_eligible'], 1):.4f} groups={t['n_groups']} "
                 f"(from {t['n_proposals']} proposals) covered={t['frac_covered']:.3f} ({time.time() - t0:.0f}s)")
    pickle.dump(out, open(p, "wb"))
    return out


def standardiser(train):
    st = {}
    for k in ["phi", "P", "tag"]:
        X = np.concatenate([t[k].reshape(len(t[k]), -1) for t in train])
        st[k] = (X.mean(0), X.std(0) + 1e-6)
    return st


def tens(t, st):
    out = {}
    for k in ["phi", "P", "tag"]:
        mu, sd = st[k]
        X = (t[k].reshape(len(t[k]), -1) - mu) / sd
        out[k] = torch.tensor(X.reshape(t[k].shape), dtype=torch.float32)
    out["groups"] = torch.tensor(t["groups"], dtype=torch.long)
    out["sign"] = torch.tensor(t["sign"], dtype=torch.float32)
    out["y"] = torch.tensor(t["y"], dtype=torch.float32)
    return out


@torch.no_grad()
def ap_of(model, T, raw):
    model.eval()
    return [average_precision(model(t).numpy(), r["y"]) for t, r in zip(T, raw)]


def train_torch(cfg, name, hidden, lr, seed, tr, va, tr_raw, va_raw, d):
    torch.manual_seed(seed)
    model = make(name, d, hidden, cfg["update_rounds"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        loss = 0
        for t in tr:
            pw = torch.tensor(min(100.0, float((1 - t["y"]).sum() / t["y"].sum().clamp(min=1))))
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(model(t), t["y"], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(np.nanmean(ap_of(model, va, va_raw)))
        hist.append(v)
        if v > best + 1e-6:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(state)
    return model, best, len(hist)


def cmd_prepare(cfg):
    tasks(cfg, "train"); tasks(cfg, "val")


def cmd_train(cfg):
    tr_raw, va_raw = tasks(cfg, "train"), tasks(cfg, "val")
    st = standardiser(tr_raw)
    pickle.dump(st, open(os.path.join(cfg["results_dir"], "cache", "standardiser.pkl"), "wb"))
    tr, va = [tens(t, st) for t in tr_raw], [tens(t, st) for t in va_raw]
    d = tr_raw[0]["phi"].shape[1]
    sel, rows = {}, []
    mdir = os.path.join(cfg["results_dir"], "models"); os.makedirs(mdir, exist_ok=True)
    # Copy-with-substitution: choose tau on validation
    best = max(cfg["grid"]["copy_tau"], key=lambda tau: np.nanmean([average_precision(copy_scores(t, tau), t["y"]) for t in va_raw]))
    sel["CopySub"] = {"tau": best}
    for t in va_raw:
        rows.append({"method": "CopySub", "seed": -1, "cutoff": t["frac"], "AP": average_precision(copy_scores(t, best), t["y"])})
    # Gradient-boosted trees on phi
    Xtr = np.concatenate([t["phi"] for t in tr_raw]); ytr = np.concatenate([t["y"] for t in tr_raw])
    gb = None
    for dep in cfg["grid"]["gbt_depth"]:
        for lr in cfg["grid"]["gbt_lr"]:
            m = HistGradientBoostingClassifier(max_depth=dep, learning_rate=lr, max_iter=300, early_stopping=False,
                                               random_state=0).fit(Xtr, ytr)
            v = np.nanmean([average_precision(m.predict_proba(t["phi"])[:, 1], t["y"]) for t in va_raw])
            log(cfg, f"grid GBT depth={dep} lr={lr}: val AP={v:.4f}")
            if gb is None or v > gb[0]:
                gb = (v, dep, lr)
    sel["GBT"] = {"depth": gb[1], "lr": gb[2]}
    for seed in cfg["seeds"]:
        m = HistGradientBoostingClassifier(max_depth=gb[1], learning_rate=gb[2], max_iter=300, early_stopping=False,
                                           random_state=seed).fit(Xtr, ytr)
        pickle.dump(m, open(os.path.join(mdir, f"GBT_seed{seed}.pkl"), "wb"))
        for t in va_raw:
            rows.append({"method": "GBT", "seed": seed, "cutoff": t["frac"], "AP": average_precision(m.predict_proba(t["phi"])[:, 1], t["y"])})
    # Torch models
    for name in TORCH_MODELS:
        bestm = None
        for h in cfg["grid"]["hidden"]:
            for lr in cfg["grid"]["lr"]:
                t0 = time.time()
                model, v, ep = train_torch(cfg, name, h, lr, cfg["seeds"][0], tr, va, tr_raw, va_raw, d)
                log(cfg, f"grid {name} hidden={h} lr={lr}: val AP={v:.4f} epochs={ep} ({time.time() - t0:.0f}s)")
                if bestm is None or v > bestm[0]:
                    bestm = (v, h, lr, model)
        sel[name] = {"hidden": bestm[1], "lr": bestm[2]}
        for seed in cfg["seeds"]:
            model = bestm[3] if seed == cfg["seeds"][0] else train_torch(cfg, name, bestm[1], bestm[2], seed, tr, va, tr_raw, va_raw, d)[0]
            torch.save(model.state_dict(), os.path.join(mdir, f"{name}_seed{seed}.pt"))
            for t, r, a in zip(va, va_raw, ap_of(model, va, va_raw)):
                rows.append({"method": name, "seed": seed, "cutoff": r["frac"], "AP": a})
    json.dump(sel, open(os.path.join(cfg["results_dir"], "selected_hparams.json"), "w"), indent=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(cfg["results_dir"], "val_ap.csv"), index=False)
    log(cfg, "validation AP (model selection data; not the decision):\n" +
        df.groupby(["method", "cutoff"]).AP.mean().unstack().round(4).to_string())


def cmd_final(cfg):
    R = cfg["results_dir"]
    lock, done = os.path.join(R, "FINAL_LOCK"), os.path.exists(os.path.join(R, "test_ap.csv"))
    if os.path.exists(lock) and done:
        sys.exit("FINAL_LOCK exists and test results are saved. Run `report`. Do not delete the lock or re-run.")
    if os.path.exists(lock):
        log(cfg, "FINAL_LOCK exists but no test results were saved (previous run crashed); resuming.")
    sel = json.load(open(os.path.join(R, "selected_hparams.json")))
    open(lock, "a").write(time.strftime("opened %Y-%m-%d %H:%M:%S\n"))
    te_raw = tasks(cfg, "test")
    st = pickle.load(open(os.path.join(R, "cache", "standardiser.pkl"), "rb"))
    te = [tens(t, st) for t in te_raw]
    d = te_raw[0]["phi"].shape[1]
    rows = []
    for t in te_raw:
        rows.append({"method": "CopySub", "seed": -1, "cutoff": t["frac"], "AP": average_precision(copy_scores(t, sel["CopySub"]["tau"]), t["y"]),
                     "n_eligible": t["n_eligible"], "n_pos": t["n_pos"], "n_groups": t["n_groups"], "frac_covered": t["frac_covered"]})
    for seed in cfg["seeds"]:
        m = pickle.load(open(os.path.join(R, "models", f"GBT_seed{seed}.pkl"), "rb"))
        for t in te_raw:
            rows.append({"method": "GBT", "seed": seed, "cutoff": t["frac"], "AP": average_precision(m.predict_proba(t["phi"])[:, 1], t["y"])})
        for name in TORCH_MODELS:
            model = make(name, d, sel[name]["hidden"], cfg["update_rounds"])
            model.load_state_dict(torch.load(os.path.join(R, "models", f"{name}_seed{seed}.pt")))
            for r, a in zip(te_raw, ap_of(model, te, te_raw)):
                rows.append({"method": name, "seed": seed, "cutoff": r["frac"], "AP": a})
    pd.DataFrame(rows).to_csv(os.path.join(R, "test_ap.csv"), index=False)
    log(cfg, "final test evaluated once; now run `report`.")


def cmd_report(cfg):
    from .report import decide
    print(decide(cfg))


if __name__ == "__main__":
    cmd, cfg = sys.argv[1], yaml.safe_load(open(sys.argv[2]))
    os.makedirs(os.path.join(cfg["results_dir"], "cache"), exist_ok=True)
    with open(os.path.join(cfg["results_dir"], "environment.txt"), "w") as f:
        f.write(f"python {sys.version}\nnumpy {np.__version__}\npandas {pd.__version__}\ntorch {torch.__version__}\n"
                f"platform {platform.platform()}\ncpus {os.cpu_count()}\n")
    t0 = time.time()
    {"prepare": cmd_prepare, "train": cmd_train, "final": cmd_final, "report": cmd_report}[cmd](cfg)
    log(cfg, f"{cmd} finished in {time.time() - t0:.0f}s")
