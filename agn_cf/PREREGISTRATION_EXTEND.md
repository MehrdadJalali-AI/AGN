# Pre-registration — AGN extension

Written before any extension run. The sections below are the protocol, copied unchanged. The implementation decisions that follow freeze choices the protocol leaves open. They are not to be changed after the tuning run or the final run.

## Protocol

Continuous editing. For non-binary features, AGN edits a bounded number of feature dimensions (the same budget of 32) with continuous, clipped steps. Values stay within each feature’s observed min–max, and the result is snapped to the dataset’s value range. Binary datasets keep the existing editing unchanged. Tune this one new component only on 100 tuning targets of Amazon-Photo (seed 1), with a small grid recorded in the file. Nothing else is retuned.
New datasets (none used before): Amazon-Computers, Amazon-Photo, Coauthor-CS and Coauthor-Physics (PyTorch Geometric). Use a random 60/20/20 split, seed 0. Report each dataset’s feature type.
Models: GCN, GAT and GraphSAGE (2 layers, standard settings, seed 0). Record test accuracy.
Targets: 200 correctly classified test nodes per dataset and model (seed 2); desired class = second-highest logit.
Plausibility: the same definition as before, plus an independent check. Train a one-class detector (e.g. isolation forest on real nodes’ features plus degree, fixed settings, seed 0) on real nodes only; report the fraction of inserted nodes it flags as outliers. It is reported, not used in the decision.
Methods: AGN, Copy-best-of-10, Copy-best-of-10-noedges, constrained attack, B1, B2. Add GANI only if official code runs.
Primary metric: plausible flip rate.
Decision rule:
GO (write the paper) if, averaged over the three models, AGN ≥ the best of Copy-best-of-10 and the constrained attack + 10 percentage points on at least 3 of the 4 new datasets, AND AGN’s outlier rate on the independent detector is not higher than Copy-best-of-10’s by more than 10 points on average.
Otherwise NO-GO: AGN is limited to binary features. Report this plainly, with no paper.

## Implementation decisions frozen before any run

`results/AGNCF_FINAL_LOCK` and `results/CONFIRM_LOCK` are not modified. AGN-CF keeps λ = 0.2, 16 steps and budget 32 from `results/agncf_selected.json`.

A dataset is binary when every stored feature value is 0 or 1, and continuous otherwise. Loaded before this run, Amazon-Computers, Amazon-Photo, Coauthor-CS and Coauthor-Physics are all binary. The continuous editor is still defined below. On a binary dataset it is not used.

The continuous editor, used only for non-binary features, clips every coordinate on every step to that coordinate’s observed minimum and maximum. Its Adam learning rate is the tuned component. The grid is {0.01, 0.05, 0.1}. After the steps, at most 32 coordinates may leave the anchor: the largest absolute deviations, ties breaking to the smaller index. Those coordinates snap to the nearest value observed in that feature; an equal distance keeps the value closer to the anchor, then the smaller value. Every other coordinate stays at the anchor. The loss, edge search, plausibility rejection, step count and budget are the frozen AGN-CF ones. The grid is scored by AGN on 100 Amazon-Photo GCN targets drawn with `Generator(1)` from correctly classified test nodes. Because that dataset is binary, the learning rate is not read. The search is executed once and that score is recorded for every row. The tie-break is the frozen one: higher plausible flip rate, higher raw flip rate, fewer edits, fewer edges, earlier row. The chosen rate is stored and would be used on a non-binary dataset.

The split is an unstratified permutation from `Generator(0)`. Validation and test each receive `n // 5` nodes and training receives the rest. GCN and GAT use the architectures and optimisation already frozen for AGN-CF, for 200 epochs, with the weights after the last epoch. GraphSAGE is two `SAGEConv` layers, hidden size 64, mean aggregation, root weight on, projection off, normalisation off, ReLU and dropout 0.5 after the first layer, Adam learning rate 0.01, weight decay 5e-4, 200 epochs, cross-entropy, seed 0, full batch, weights after the last epoch. The three-layer neighbour-sampled OGB example is not used.

Targets are 200 correctly classified test nodes, `Generator(2)`, drawn separately for each model and dataset, then sorted. Amazon-Photo also excludes the 100 tuning ids for every model. If a pool has fewer than 200, stop. The desired class is the second-highest logit. A flip is `argmax == desired` after one insertion.

Plausibility is the three-part rule in `PREREGISTRATION_AGNCF.md`, computed on each new dataset. The detector is `sklearn.ensemble.IsolationForest(n_estimators=100, max_samples="auto", contamination="auto", random_state=0, n_jobs=1)`, fit once per dataset on real nodes. Each real row is the feature vector with the node’s degree appended. An inserted row uses the same features and the degree of the inserted node. Outlier means `predict == -1`. The outlier rate is the fraction of returned nodes flagged. It is not a plausibility constraint.

Copy-best-of-10, Copy-best-of-10-noedges, the constrained attack, B1 and B2 are the confirmation methods, including the constrained attack’s projection, 100 steps and single edge. GANI is not run: `https://github.com/alexfanjn/GANI` trains a deeprobust GCN on the nettack split, injects 10 nodes and optimises a global homophily objective. It does not attack these frozen models or insert one node to change one target’s class.

With 200 targets per model, the three-model average is the total over 600 targets. “+10 percentage points” is 60 plausible flips: `agn_flips >= max(copy_flips, attack_flips) + 60`. The outlier comparison uses, on each dataset, the unweighted mean of the three model rates, then the unweighted mean of the four dataset gaps `AGN − Copy-best-of-10`. The condition holds when that mean gap is at most 0.10. Both conditions are required. Otherwise the verdict is NO-GO: AGN is limited to binary features.

The target logit may be computed from the same two-layer equation on the nodes whose states the insertion changes. Before the final run this local value is checked against the full model. A mismatch is a bug in that computation. `results/EXTEND_LOCK` is written after tuning has been saved and the final targets have been checked, and before any final method is evaluated. If the lock exists, the final run refuses to start.
