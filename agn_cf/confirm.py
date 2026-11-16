"""Confirmation run frozen in PREREGISTRATION_CONFIRM.md. One run, after CONFIRM_LOCK."""
import csv
import json
import os
import time

import numpy as np
import torch

import agncf

agncf.LOG_PATH = os.path.join(agncf.RESULTS, "confirm_run.log")

LOCK = os.path.join(agncf.RESULTS, "CONFIRM_LOCK")
CSV_PATH = os.path.join(agncf.RESULTS, "confirm_per_target.csv")
DECISION = os.path.join(agncf.RESULTS, "CONFIRM_DECISION.md")
SUMMARY = os.path.join(agncf.RESULTS, "confirm_summary.json")
FINAL_LOCK_TEXT = open(agncf.LOCK).read()
CFG = {"lambda": 0.2, "steps": 16, "budget": 32}
BUDGET = 32
ATTACK_STEPS = 100
METHODS = ["B1", "B2", "Copy-best-of-10", "Copy-best-of-10-noedges", "constrained-attack", "AGN-CF"]


def log(msg):
    agncf.log(msg)


def edit_count(hard, prototype, binary):
    delta = (hard - prototype).abs()
    if binary:
        return int((delta > 0).sum().item())
    return int((delta >= agncf.SNAP).sum().item())


def cap_to_budget(hard, prototype, binary, budget):
    hard = hard.detach().clone()
    if not binary:
        delta = (hard - prototype).abs()
        hard = torch.where(delta >= agncf.SNAP, hard, prototype)
    delta = (hard - prototype).abs()
    changed = delta > 0 if binary else delta >= agncf.SNAP
    n_changed = int(changed.sum().item())
    if n_changed > budget:
        order = [i for i in range(delta.numel()) if bool(changed[i])]
        order.sort(key=lambda i: (-float(delta[i]), i))
        keep = set(order[:budget])
        for i in order[budget:]:
            hard[i] = prototype[i]
    return hard, edit_count(hard, prototype, binary)


def project_features(soft, prototype, x, stats, binary, fmax, budget):
    upper = 1.0 if binary else float(fmax)
    hard = soft.detach().clamp(0, upper)
    hard, edits = cap_to_budget(hard, prototype, binary, budget)
    hard, edits = agncf.discrete_from_soft(hard, prototype, binary)
    hard, edits = cap_to_budget(hard, prototype, binary, budget)
    dist_vec = torch.linalg.vector_norm(x - hard, dim=1)
    nearest = int(dist_vec.argmin())
    dist = float(dist_vec[nearest])
    if dist > stats["l2_p95"]:
        direction = hard - x[nearest]
        hard = (x[nearest] + direction * (stats["l2_p95"] / dist)).clamp(0, upper)
        hard, edits = agncf.discrete_from_soft(hard, prototype, binary)
        hard, edits = cap_to_budget(hard, prototype, binary, budget)
    active = int((hard > 0).sum().item())
    if active > stats["active_p95"]:
        pos = [i for i in range(hard.numel()) if float(hard[i]) > 0]
        pos.sort(key=lambda i: (0 if float((hard[i] - prototype[i]).abs()) > 0 else 1, float(hard[i]), i))
        for i in pos:
            if active <= stats["active_p95"]:
                break
            hard[i] = 0
            active -= 1
        hard, edits = cap_to_budget(hard, prototype, binary, budget)
    elif active < stats["active_p5"]:
        cands = [i for i in range(hard.numel()) if float(prototype[i]) > 0 and float(hard[i]) <= 0]
        cands.sort(key=lambda i: (-float(prototype[i]), i))
        for i in cands:
            if active >= stats["active_p5"]:
                break
            trial = hard.clone()
            trial[i] = prototype[i]
            trial, trial_edits = cap_to_budget(trial, prototype, binary, budget)
            if trial_edits > budget or float(trial[i]) <= 0:
                continue
            hard, edits = trial, trial_edits
            active = int((hard > 0).sum().item())
    ok, dist, _ = agncf.feature_state(hard, x, stats)
    degree_ok = stats["deg_min"] <= 1 <= stats["deg_max"]
    if edits > budget or not ok or not degree_ok:
        return None
    return hard, edits, dist


def record_from(feat, prototype, anchor, nbrs, flipped, margin, x, stats, binary):
    ok, dist, _ = agncf.feature_state(feat, x, stats)
    degree_ok = stats["deg_min"] <= len(nbrs) <= stats["deg_max"]
    edits = 0 if prototype is None else edit_count(feat, prototype, binary)
    return {"returned": 1, "flipped": int(flipped), "plausible": int(ok and degree_ok and flipped),
            "edits": edits, "edges": len(nbrs), "distance": dist, "anchor": int(anchor),
            "nbrs": nbrs, "feat": feat}


def copy_search(pack, target, desired, use_edges):
    x = pack["data"].x
    anchors = agncf.anchors_for(pack["train_ids"], pack["y"], pack["prob"], desired, 10)
    first = None
    for anchor in anchors:
        feat = x[anchor].detach()
        if use_edges:
            nbrs, flipped, margin = agncf.greedy(
                pack["model"], x, pack["data"].edge_index, feat, pack["neigh"][anchor],
                target, desired, pack["stats"], True)
        else:
            nbrs = [target]
            flipped, margin = agncf.evaluate(
                pack["model"], x, pack["data"].edge_index, feat, nbrs, target, desired)
        rec = record_from(feat, feat, anchor, nbrs, flipped, margin, x, pack["stats"], pack["binary"])
        if first is None:
            first = rec
        if flipped:
            return rec
    return first


def constrained_attack(pack, target, desired):
    x = pack["data"].x
    prototype_id = agncf.anchors_for(pack["train_ids"], pack["y"], pack["prob"], desired, 1)[0]
    prototype = x[prototype_id].detach()
    nbrs = [target]
    theta = torch.zeros(prototype.numel(), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=agncf.ADAM_LR)
    accepted = None
    ok, dist, _ = agncf.feature_state(prototype, x, pack["stats"])
    if ok and pack["stats"]["deg_min"] <= 1 <= pack["stats"]["deg_max"]:
        flipped, margin = agncf.evaluate(
            pack["model"], x, pack["data"].edge_index, prototype, nbrs, target, desired)
        accepted = (prototype, 0, dist, flipped, margin)
        if flipped:
            return attack_record(accepted, prototype_id, nbrs)
    for _ in range(ATTACK_STEPS):
        opt.zero_grad(set_to_none=True)
        soft = (prototype + theta).clamp(0, 1.0 if pack["binary"] else float(pack["fmax"]))
        loss = -agncf.margin_tensor(
            agncf.forward_logits(pack["model"], x, pack["data"].edge_index, soft, nbrs), target, desired)
        loss.backward()
        opt.step()
        with torch.no_grad():
            soft = (prototype + theta).clamp(0, 1.0 if pack["binary"] else float(pack["fmax"]))
            projected = project_features(soft, prototype, x, pack["stats"], pack["binary"], pack["fmax"], BUDGET)
        if projected is None:
            continue
        hard, edits, dist = projected
        flipped, margin = agncf.evaluate(
            pack["model"], x, pack["data"].edge_index, hard, nbrs, target, desired)
        accepted = (hard, edits, dist, flipped, margin)
        if flipped:
            break
    if accepted is None:
        return {"returned": 0, "flipped": 0, "plausible": 0, "edits": "", "edges": "",
                "distance": "", "anchor": prototype_id, "nbrs": []}
    return attack_record(accepted, prototype_id, nbrs)


def attack_record(accepted, anchor, nbrs):
    hard, edits, dist, flipped, _margin = accepted
    return {"returned": 1, "flipped": int(flipped), "plausible": int(flipped),
            "edits": edits, "edges": len(nbrs), "distance": dist, "anchor": int(anchor),
            "nbrs": nbrs, "feat": hard}


def final_test_ids():
    grouped = {}
    with open(agncf.TEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            grouped.setdefault((row["model"], row["dataset"]), set()).add(int(row["target"]))
    return {key: np.array(sorted(values), dtype=int) for key, values in grouped.items()}


def draw_targets(packs):
    saved = json.load(open(agncf.SELECTED))
    if [saved["lambda"], saved["steps"], saved["budget"]] != [0.2, 16, 32]:
        raise SystemExit("agncf_selected.json is not the frozen λ=0.2, steps=16, budget=32")
    tune = np.array(saved["tuning_ids"], dtype=int)
    final_ids = final_test_ids()
    for model_name in ("GCN", "GAT"):
        for name in agncf.DATASETS:
            if len(final_ids.get((model_name, name), [])) != 200:
                raise SystemExit(f"final-test ids for {model_name} {name} are not 200")
    targets = {}
    for model_name in ("GCN", "GAT"):
        for name in agncf.DATASETS:
            pack = packs[(model_name, name)]
            exclude = [pack["headroom"], final_ids[(model_name, name)]]
            if name == "Cora":
                exclude.append(tune)
            chosen = agncf.draw_ids(pack["correct"], np.concatenate(exclude), 200, 3)
            if len(np.intersect1d(chosen, pack["headroom"])) or len(np.intersect1d(chosen, final_ids[(model_name, name)])):
                raise SystemExit(f"{model_name} {name} confirmation targets intersect an excluded set")
            if name == "Cora" and len(np.intersect1d(chosen, tune)):
                raise SystemExit("Cora confirmation targets intersect the tuning set")
            targets[(model_name, name)] = chosen
            log(f"targets {model_name} {name}: {len(chosen)} from pool, "
                f"excluded headroom {len(pack['headroom'])} final {len(final_ids[(model_name, name)])}")
    return targets


def run_one(pack, targets, method):
    if method in ("B1", "B2", "AGN-CF"):
        return agncf.run_method(pack, targets, CFG, method)
    rows = []
    t0 = time.perf_counter()
    for target in targets:
        target = int(target)
        desired = int(pack["second"][target])
        if method == "Copy-best-of-10":
            rec = copy_search(pack, target, desired, True)
        elif method == "Copy-best-of-10-noedges":
            rec = copy_search(pack, target, desired, False)
        else:
            rec = constrained_attack(pack, target, desired)
        rec["target"] = target
        rec["desired"] = desired
        rows.append(rec)
    return rows, time.perf_counter() - t0


def write_decision(summary, seconds):
    lines = ["# AGN-CF confirmation", "",
             "Settings are the frozen λ = 0.2, steps = 16, budget = 32. No retuning.",
             "GANI: not run. The official repository https://github.com/alexfanjn/GANI trains a deeprobust GCN "
             "on the nettack split with seed 2021, injects 10 nodes, and searches a global homophily objective. "
             "It does not attack the frozen public-split models or insert one node to change one target's class.",
             "", "Plausible flip rate is the primary metric. Means are over targets for which a node was returned.", ""]
    for model in ("GCN", "GAT"):
        lines.append(f"## {model}")
        lines.append("")
        lines.append("```")
        lines.append(f"{'dataset':<10} {'method':<24} {'plausible':>12} {'raw':>12} {'edits':>14} {'edges':>12} {'distance':>14} {'seconds':>10}")
        for name in agncf.DATASETS:
            for method in METHODS:
                s = summary[model][name][method]
                edits = "NA" if s["mean_edits"] is None else repr(s["mean_edits"])
                edges = "NA" if s["mean_edges"] is None else repr(s["mean_edges"])
                dist = "NA" if s["mean_distance"] is None else repr(s["mean_distance"])
                lines.append(f"{name:<10} {method:<24} {s['plausible_flips']:>4}/{s['n']:<7} "
                             f"{s['flips']:>4}/{s['n']:<7} {edits:>14} {edges:>12} {dist:>14} {s['seconds']:>10.1f}")
        lines.append("```")
        lines.append("")
    copy_wins = attack_wins = gat_wins = 0
    lines.append("## Conditions")
    lines.append("")
    for name in agncf.DATASETS:
        agn = summary["GCN"][name]["AGN-CF"]["plausible_flips"]
        copy = summary["GCN"][name]["Copy-best-of-10"]["plausible_flips"]
        attack = summary["GCN"][name]["constrained-attack"]["plausible_flips"]
        ok_copy = agn >= copy + 20
        ok_attack = agn >= attack + 20
        copy_wins += int(ok_copy)
        attack_wins += int(ok_attack)
        lines.append(f"- GCN {name}: AGN-CF {agn}/200, Copy-best-of-10 {copy}/200, "
                     f"constrained-attack {attack}/200; +20 flips vs copy: {'PASS' if ok_copy else 'FAIL'}; "
                     f"+20 flips vs attack: {'PASS' if ok_attack else 'FAIL'}")
    for name in agncf.DATASETS:
        agn = summary["GAT"][name]["AGN-CF"]["plausible_flips"]
        copy = summary["GAT"][name]["Copy-best-of-10"]["plausible_flips"]
        ok = agn >= copy
        gat_wins += int(ok)
        lines.append(f"- GAT {name}: AGN-CF {agn}/200, Copy-best-of-10 {copy}/200; "
                     f"AGN-CF ≥ copy: {'PASS' if ok else 'FAIL'}")
    confirmed = copy_wins >= 2 and attack_wins >= 2 and gat_wins >= 2
    verdict = "CONFIRMED" if confirmed else "NOT CONFIRMED: the gain comes from candidate search, not from AGN-CF’s editing. Stop AGN-CF, with no paper."
    lines += ["", f"GCN datasets with AGN-CF ≥ Copy-best-of-10 + 20 flips: {copy_wins} of 3",
              f"GCN datasets with AGN-CF ≥ constrained attack + 20 flips: {attack_wins} of 3",
              f"GAT datasets with AGN-CF ≥ Copy-best-of-10: {gat_wins} of 3",
              "", f"## Verdict: **{verdict}**", "", f"Confirmation runtime seconds: {seconds!r}", ""]
    open(DECISION, "w").write("\n".join(lines))
    log("verdict " + ("CONFIRMED" if confirmed else "NOT CONFIRMED"))


def main():
    if os.path.exists(LOCK):
        raise SystemExit("CONFIRM_LOCK exists. The confirmation run must not be run again.")
    if open(agncf.LOCK).read() != FINAL_LOCK_TEXT:
        raise SystemExit("AGNCF_FINAL_LOCK changed before the confirmation run")
    os.makedirs(agncf.RESULTS, exist_ok=True)
    log("confirm preparing")
    packs = {}
    for name in agncf.DATASETS:
        gcn = agncf.prepare(name, "GCN")
        packs[("GCN", name)] = gcn
        packs[("GAT", name)] = agncf.prepare(name, "GAT", gcn)
    targets = draw_targets(packs)
    with open(LOCK, "w") as f:
        f.write(time.strftime("opened %Y-%m-%d %H:%M:%S\n"))
    log("confirm lock written")
    t0 = time.perf_counter()
    summary = {m: {} for m in ("GCN", "GAT")}
    fields = ["model", "dataset", "method", "target", "desired", "anchor", "flipped", "plausible",
              "edits", "edges", "distance", "returned"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for model_name in ("GCN", "GAT"):
            for name in agncf.DATASETS:
                summary[model_name][name] = {}
                pack = packs[(model_name, name)]
                ids = targets[(model_name, name)]
                for method in METHODS:
                    rows, seconds = run_one(pack, ids, method)
                    sm = agncf.summarise(rows)
                    sm["seconds"] = seconds
                    summary[model_name][name][method] = sm
                    log(f"confirm {model_name} {name} {method}: plausible {sm['plausible_flips']}/{sm['n']} "
                        f"raw {sm['flips']}/{sm['n']} edits={sm['mean_edits']} edges={sm['mean_edges']} "
                        f"dist={sm['mean_distance']} ({seconds:.1f}s)")
                    for rec in rows:
                        row = {k: rec.get(k, "") for k in fields}
                        row["model"] = model_name
                        row["dataset"] = name
                        row["method"] = method
                        writer.writerow(row)
                    f.flush()
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    write_decision(summary, time.perf_counter() - t0)
    if open(agncf.LOCK).read() != FINAL_LOCK_TEXT:
        raise SystemExit("AGNCF_FINAL_LOCK changed during the confirmation run")


if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    main()
