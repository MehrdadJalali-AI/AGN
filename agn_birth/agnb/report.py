"""Pre-registered decision rule (see PREREGISTRATION.md). Reads only saved result files."""
import json
import os

import numpy as np
import pandas as pd

from .models import BASELINES, EXPERTS

PRIMARY = "R@20"


def _per_paper(df, method):
    d = df[df.method == method]
    return d.groupby("p")[[PRIMARY, "N@20", "R@50", "label"]].mean()   # mean over seeds per paper


def _boot(diff, B, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (B, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decide(cfg):
    R = cfg["results_dir"]
    path = os.path.join(R, "test_per_paper.csv")
    if not os.path.exists(path):
        return "No test results yet: run `final` first."
    df = pd.read_csv(path)
    info = json.load(open(os.path.join(R, "test_info.json")))
    methods = BASELINES + ["LR-Ranker", "AGN", "AGN-nogate", "AGN-nomem"]
    P = {m: _per_paper(df, m) for m in methods}
    summary = pd.DataFrame({m: P[m][[PRIMARY, "N@20", "R@50"]].mean() for m in methods}).T
    seed_sd = df[df.seed >= 0].groupby(["method", "seed"])[PRIMARY].mean().groupby("method").std()
    summary["R@20_sd_over_seeds"] = seed_sd
    baselines = BASELINES + ["LR-Ranker"]
    best = max(baselines, key=lambda m: summary.loc[m, PRIMARY])
    a, b = P["AGN"][PRIMARY], P[best][PRIMARY].loc[P["AGN"].index]
    diff = (a - b).values
    abs_gain = float(diff.mean())
    rel_gain = abs_gain / max(float(b.mean()), 1e-9)
    ci = _boot(diff, cfg["bootstrap"])
    abl = {}
    for m in ["AGN-nogate", "AGN-nomem"]:
        d = (a - P[m][PRIMARY].loc[a.index]).values
        abl[m] = {"mean_diff": float(d.mean()), "ci95": _boot(d, cfg["bootstrap"], 1)}
    lab = P["AGN"]["label"]
    big = lab.value_counts()
    big = big[big >= 50].index
    wins = [(a[lab == c].mean() >= b[lab == c].mean()) for c in big]
    frac_wins = float(np.mean(wins)) if wins else float("nan")
    c1 = abs_gain >= 0.02 and rel_gain >= 0.05
    c2 = ci[0] > 0
    c3 = abl["AGN-nogate"]["ci95"][0] > 0 and abl["AGN-nomem"]["ci95"][0] > 0
    c4 = frac_wins >= 0.70
    go = c1 and c2 and c3 and c4
    lines = ["# AGN kill test — pre-registered decision", "",
             f"Test: cutoff {info['cutoff']} -> arrivals {info['cutoff'] + 1}; papers evaluated: {info['n_used']} "
             f"(of {info['n_eligible']} eligible); candidate ceiling R@20 = {info['ceiling_R@20']:.3f}", "",
             "## Test results (per-paper mean; learned models averaged over seeds)", "",
             "```\n" + summary.round(4).to_string() + "\n```", "",
             f"Best baseline on test: **{best}** (R@20 = {summary.loc[best, PRIMARY]:.4f})",
             f"AGN R@20 = {summary.loc['AGN', PRIMARY]:.4f}; gain = {abs_gain:+.4f} absolute, {100 * rel_gain:+.1f}% relative; "
             f"95% bootstrap CI of gain = [{ci[0]:+.4f}, {ci[1]:+.4f}]",
             f"Ablations (AGN minus variant): " + "; ".join(f"{k}: {v['mean_diff']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in abl.items()),
             f"Subject areas with >=50 test papers: {len(big)}; AGN >= best baseline in {frac_wins:.0%}", "",
             "## Conditions",
             f"1. Gain >= 0.02 absolute AND >= 5% relative: {'PASS' if c1 else 'FAIL'}",
             f"2. 95% CI of gain excludes 0: {'PASS' if c2 else 'FAIL'}",
             f"3. Gate and memory both needed (CIs vs AGN-nogate and AGN-nomem exclude 0): {'PASS' if c3 else 'FAIL'}",
             f"4. Consistent across subject areas (>= 70%): {'PASS' if c4 else 'FAIL'}", "",
             f"## Verdict: **{'GO' if go else 'NO-GO — stop AGN (no paper)'}**", ""]
    gp = os.path.join(R, "test_gates.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp)
        lines += ["## Gate readout (descriptive only, not part of the decision)", "",
                  "Mean gate weight per mechanism: " + ", ".join(f"{e} {g[e].mean():.3f}" for e in EXPERTS if e in g), ""]
    out = "\n".join(lines)
    open(os.path.join(R, "DECISION.md"), "w").write(out)
    summary.to_csv(os.path.join(R, "test_summary.csv"))
    return out
