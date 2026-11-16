# Pre-registration — AGN-CF confirmation

Written before the confirmation run. The sections below are the protocol, copied unchanged. The implementation decisions that follow freeze choices the protocol leaves open. They are not to be changed after the run.

## Setup

The same GCN and GAT models and training as the final test (retrained identically, seed 0), on Cora, CiteSeer and PubMed.
Fresh targets: 200 per dataset (seed 3), correctly classified public-test nodes, disjoint from the headroom, tuning and final-test targets. Desired class = second-highest logit.
The same plausibility definition as PREREGISTRATION_AGNCF.md.

## Methods (same single-node budget, no retraining)

AGN-CF, frozen settings.
Copy-best-of-10: the same 10 anchors AGN-CF uses, with the same greedy extra-edge procedure but NO feature edits; return the first anchor (in AGN-CF’s anchor order) that flips the target, preferring the fewest edges.
Copy-best-of-10-noedges: the same 10 anchors, connected only to the target, no edits.
Constrained attack: the Tao et al. reimplementation started from the best B1 prototype, with feature changes limited to the same budget (32) and every step projected to satisfy the plausibility constraints; reject the result if it is not plausible.
GANI: run it only if official code is public and runs within reasonable effort; otherwise write “not run” with the reason.
B1 and B2, as before.

## Primary metric

Primary metric: plausible flip rate. Also report raw flip rate, mean edits, edges, distance and runtime.

## Decision rule

CONFIRMED only if, with the GCN, AGN-CF’s plausible flip rate is ≥ Copy-best-of-10 + 10 percentage points on at least 2 of 3 datasets, AND ≥ the constrained attack + 10 points on at least 2 of 3 datasets, AND, with the GAT, AGN-CF ≥ Copy-best-of-10 on at least 2 of 3 datasets.
Otherwise NOT CONFIRMED: the gain comes from candidate search, not from AGN-CF’s editing. Stop AGN-CF, with no paper.

## Implementation decisions frozen before any run

Settings are read from `results/agncf_selected.json` and must be λ = 0.2, steps = 16, budget = 32. They are not retuned. AGN-CF is the frozen search in `agncf.py`.

Models are retrained by the same `prepare` path as the final test. Headroom exclusion ids are that path’s GCN ids, shared by the GCN and the GAT. Tuning ids are the saved Cora list in `agncf_selected.json`, excluded on Cora only. Final-test ids are the target ids already stored in `results/agncf_test_per_target.csv` for that model and dataset. They are not redrawn, and `results/AGNCF_FINAL_LOCK` is not modified.

Confirmation targets are drawn with `Generator(3)`, 200 nodes, separately for each model and dataset, from that model’s correctly classified public-test ids minus the exclusions above, then sorted. If a pool has fewer than 200, stop. The lock `results/CONFIRM_LOCK` is written after those draws have been checked and before any method is evaluated. If the lock already exists, the run refuses to start.

Copy-best-of-10 uses the same ten anchors, in the same confidence order, with the smaller index winning a tie. Each anchor keeps its own features. Edges start at the target and then follow the frozen greedy rule, including its degree check, and stop when the target flips. The first anchor in that order whose greedy result flips is returned. If none flips, the first anchor’s greedy result is returned so the means are defined. Copy-best-of-10-noedges uses the same anchors and the same return rule, with the single edge to the target and no greedy additions. Neither method edits features.

The constrained attack is the Tao reimplementation with these restrictions. It starts from the B1 prototype, the single most confident training node of the desired class. It adds one edge, to the target, and no others. Additive features start at 0 and are trained with Adam, learning rate 0.1, for at most 100 steps, on `−margin` only. There is no λ term; the budget is hard. After every step the candidate is projected in this order: clamp to the feature domain; keep at most 32 coordinates, the largest absolute deviations, ties breaking to the smaller index, and reset the rest to the prototype; discretise with the AGN-CF rule (binary threshold 0.5, otherwise snap movements below 0.05 back to the prototype); if the distance to the nearest real node exceeds the 95th percentile, scale the candidate onto that ball around the nearest real node, clamp, and discretise and cap the budget again; if the active-feature count is above the 95th percentile, zero the smallest positive coordinates, edited coordinates first, ties breaking to the smaller index, until it is inside the band; if it is below the 5th percentile, restore the largest currently inactive prototype values, ties breaking to the smaller index, and stop before the edit budget would be exceeded. A step is accepted only if the projected node is inside the budget and passes all three plausibility checks. Rejection does not roll Adam back. The untouched prototype is kept when it is already plausible. The search stops at the first accepted flip. If none flips, the last accepted plausible node is returned. If no plausible node was accepted, nothing is returned, and that target counts as neither a raw nor a plausible flip.

GANI is not run. The official repository `https://github.com/alexfanjn/GANI` is public, and `main.py` was retrieved. It trains a deeprobust GCN on the nettack split with seed 2021, injects 10 nodes, and searches a global homophily objective with a genetic algorithm of population 40 for 100 iterations. It does not attack the frozen public-split GCN or GAT, and it does not insert one node to move one target to the second-highest class. Fitting it to this protocol would be a different method.

With 200 targets, “+10 percentage points” is a difference of 20 plausible flips: the test is `agn_flips >= other_flips + 20`. The GAT comparison is `agn_flips >= copy_flips`. A dataset counts toward “at least 2 of 3” only when that inequality holds. Means use the same returned-node rule as the final test. Flip rates use all 200 targets as the denominator.
