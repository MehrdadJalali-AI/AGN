"""Pre-registered decision rule (PREREGISTRATION.md). Reads saved results only."""
import os

import pandas as pd

BASELINES = ["CopySub", "PairAdditive", "GBT", "MLP"]


def decide(cfg):
    R = cfg["results_dir"]
    p = os.path.join(R, "test_ap.csv")
    if not os.path.exists(p):
        return "No test results yet: run `final` first."
    df = pd.read_csv(p)
    info = df[df.method == "CopySub"].set_index("cutoff")[["n_eligible", "n_pos", "n_groups", "frac_covered"]]
    mean = df.groupby(["method", "cutoff"]).AP.mean().unstack()
    per_seed = df[df.method == "AGN-Triad"].pivot(index="seed", columns="cutoff", values="AP")
    lines = ["# AGN-Triad kill test — pre-registered decision", "",
             "Test periods (cutoff = fraction of the event sequence; window = next 10% of events):", "",
             "```", info.to_string(), "```", "",
             "## Average precision on test (learned methods: mean over seeds)", "",
             "```", mean.round(4).to_string(), "```", "",
             "AGN-Triad per seed:", "", "```", per_seed.round(4).to_string(), "```", ""]
    ok = True
    for c in mean.columns:
        best = max(BASELINES, key=lambda m: mean.loc[m, c])
        a, b = mean.loc["AGN-Triad", c], mean.loc[best, c]
        rel, ab = a / b - 1, a - b
        passed = rel >= 0.15 and ab >= 0.01
        ok &= passed
        lines.append(f"- Cutoff {c}: best baseline {best} AP={b:.4f}; AGN-Triad AP={a:.4f}; "
                     f"gain {ab:+.4f} absolute, {100 * rel:+.1f}% relative -> {'PASS' if passed else 'FAIL'}")
    lines += ["", "Rule: AGN-Triad must be >= 15% relative AND >= 0.01 absolute above the best baseline in BOTH test deciles.",
              "HIT (Liu, Ma & Li) was not run in this pilot; a GO is provisional until it is compared.", "",
              f"## Verdict: **{'PROVISIONAL GO (pending HIT comparison)' if ok else 'NO-GO — stop AGN (no paper)'}**", ""]
    out = "\n".join(lines)
    open(os.path.join(R, "DECISION.md"), "w").write(out)
    return out
