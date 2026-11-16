# Pre-registration — AGN-CF

Written before any AGN-CF run. The sections below are the protocol. The implementation decisions that follow freeze choices the protocol leaves open. They are not to be changed after any run.

**Models:** the same 2-layer GCN as the headroom check (retrained identically, seed 0) on Cora, CiteSeer and PubMed. Secondary model: a 2-layer GAT (standard PyTorch Geometric settings, seed 0).

**Targets:**
- correctly classified public-test nodes, excluding the 200 headroom targets;
- tuning set: 100 Cora targets (seed 1);
- final test set: 200 targets per dataset (seed 2), disjoint from the tuning set;
- desired class = the class with the second-highest logit.

**Plausibility (defined from real data):** an inserted node is plausible only if all three hold:
1. its L2 distance to the nearest real node is ≤ the 95th percentile of real nodes' leave-one-out nearest-neighbour distances;
2. its number of active features is within the 5th–95th percentile of real nodes;
3. its degree is within the real degree range.

**AGN-CF method** ("copy-then-edit", building on the finding that new nodes resemble existing ones):
- anchors: the top 10 training nodes of the desired class by confidence;
- for each anchor, start from its features and edit them with a gradient-based continuous relaxation, then binarise; loss = −(desired-class margin) + λ·(number of edited features);
- edges: always to the target, plus up to (median degree − 1) extra edges chosen greedily among the anchor's neighbours, only when they increase the margin;
- the plausibility constraints are enforced during the search (an edit that violates them is rejected);
- return the flipping candidate with the fewest edited features, then the fewest edges.

λ, the step count and the feature-edit budget are tuned on the tuning set only, with a small grid recorded in the file.

**Baselines (same targets, same single-node budget, no retraining):**
- B1 copy-prototype and B2 copy-prototype-plus, as in the headroom check;
- **Tao et al. single-node injection:** use the official code if it is public and runs; otherwise reimplement a gradient-based single-node, single-edge injection with unconstrained features, and label it "reimplementation";
- ablation **AGN-CF-noplaus:** the same search without the plausibility constraints;
- ablation **random-anchor edit:** a random training node of the desired class as the only anchor.

**Metrics:**
- **primary: plausible flip rate** = the fraction of targets flipped by a node that is plausible under the definition above;
- raw flip rate;
- mean edited features (minimality);
- edges used;
- nearest-real-node distance;
- runtime.

**Decision rule (final test only):**
- **GO** if, with the GCN on ALL three datasets, AGN-CF's plausible flip rate is ≥ the best baseline's plausible flip rate + 15 percentage points AND ≥ 1.25 × that rate, where the best baseline is the strongest of B1, B2 and Tao et al. AND, with the GAT, AGN-CF's gain over the best baseline is ≥ +5 points on at least 2 of the 3 datasets.
- Otherwise **NO-GO**: stop AGN-CF, with no paper.

## Implementation decisions frozen before any run

Headroom exclusion ids are recomputed with the GCN in `headroom.py` (seed 0, public split, 200 epochs, final parameters): correctly classified test nodes, `numpy.random.Generator(0).choice` of 200, then sorted. The same ids are excluded for the GCN and the GAT.

Correct classification and the desired class are computed for the model being explained, in eval mode, on the original graph. The desired class is the second `argmax` after masking the top logit. A flip is `argmax == desired class` on the target after one insertion. Insertions are not cumulative.

Target draws, each with its own generator, from the model's correctly classified test ids minus the excluded ids, then sorted:

- Cora tuning: GCN pool, `Generator(1)`, 100 nodes. These ids are saved and are not used as final-test targets for either model.
- Final test: `Generator(2)`, 200 nodes, separately for each model and dataset. Cora also excludes the tuning ids. If a pool has fewer than the required count, stop. The protocol cannot be completed by shrinking the set.

Confidence is the softmax probability of the desired class. Ties break to the smaller node index, as in the headroom check. Anchors are the top 10 such training nodes, or all of them if fewer than 10 exist. B1 and B2 use only the top one, and B2's extra neighbours follow the headroom rule (`floor(median degree)` neighbours, ordered by higher degree then smaller index), not the AGN-CF edge rule.

Degree is the number of distinct neighbours in the input graph, before `GCNConv` or `GATConv` self-loops. Median and percentiles use `numpy` on that degree vector and `numpy.percentile(..., method="linear")`. Active features are coordinates strictly greater than 0. Leave-one-out nearest-neighbour distances are Euclidean distances in float32. An inserted node is compared with `distance <= p95`, `p5 <= active count <= p95`, and `minimum degree <= degree <= maximum degree`.

GAT, matching the architecture and optimisation in PyTorch Geometric 2.6.1 `examples/gat.py`, on the same unnormalised Planetoid features as the GCN: two `GATConv` layers, 8 hidden units, 8 heads then 1 head, attention dropout 0.6, feature dropout 0.6, ELU, Adam learning rate 0.005, weight decay 5e-4, 200 epochs, cross-entropy, seed 0, parameters after the last epoch. The example comment about extra PubMed heads is not what that script runs, and it is not used here.

Copy-then-edit, for every anchor:

- The margin is `logit[desired] − max of the other logits` at the target.
- Continuous relaxation: `θ` starts at 0 and is trained with Adam, learning rate 0.1. Soft features are `clamp(anchor + θ, 0, 1)` on binary datasets (every stored value is 0 or 1) and `clamp(anchor + θ, 0, max feature)` otherwise. The training loss is `−margin + λ · L1(soft − anchor)`. Rejection does not roll Adam back.
- Then binarise. Binary datasets: threshold the soft vector at 0.5. Other datasets: coordinates that moved by less than 0.05 snap back to the anchor; the others stay at the soft value. Edited features are the coordinates that differ from the anchor after that step. A discrete vector whose edit count exceeds the budget is not a candidate.
- The untouched anchor is always eligible as a zero-edit vector.
- Edges are chosen on the discrete vector. Start with the edge to the target. Add up to `max(0, floor(median degree) − 1)` extra edges. Each addition tries the anchor's original neighbours except the target and keeps the one with the largest margin, ties breaking to the smaller index, and only if that margin is strictly larger. Stop when the target flips or no addition increases the margin.
- While plausibility is enforced, a discrete feature vector that fails the distance or active-feature test is rejected, and an extra edge that would leave the degree range is not available. AGN-CF-noplaus uses the same search without those rejections.
- The returned node is the flipping candidate with the fewest edited features, then the fewest edges, then the highest margin, then the smallest anchor id. If nothing flips, return the feasible candidate under that same order so the descriptive means are defined. If a method has no feasible candidate, it does not flip that target and the target is omitted from the means.
- Result-preserving evaluation order: the untouched anchor is scored before any gradient step. If it flips, later edited vectors of that anchor are not generated, because each of them has at least one edit. Once any anchor has produced a zero-edit flip, later anchors are not edited. If that flip uses one edge, a later anchor is checked only with the single edge to the target; otherwise only its untouched feature vector is passed to the edge search. Under the order above, an edited vector or a vector with more edges cannot replace a zero-edit, one-edge flip.

The grid, scored only by AGN-CF on the Cora GCN tuning set, is λ ∈ {0.2, 1.0, 5.0}, steps ∈ {16}, budget ∈ {8, 32}. The chosen triple maximises plausible flip rate, then raw flip rate, then minimises mean edited features, then mean edges, then the earlier grid row. It is used for every dataset, both models, AGN-CF, AGN-CF-noplaus, and the random-anchor ablation. Random anchors are drawn with `Generator(3)` in increasing target-id order, one training node of the desired class, restarted for each model and dataset.

Means of edited features, edges, and nearest-node distance are over targets for which a node was returned. Flip rates use all targets as the denominator.

Tao et al.: the public repository `TaoShuchang/G-NIA` was retrieved. It does not run for this protocol. It loads a dataset zip and surrogate checkpoints that are not in the repository, and it attacks its own hidden-size-64 surrogate rather than the frozen headroom GCN. The baseline is therefore a reimplementation: one new node, one edge to the target, features initialised at 0 and left unconstrained, Adam learning rate 0.1 for at most 100 steps on `−margin`, stopping at the first flip. Its edited-feature count is the number of coordinates with absolute value above 1e-6. It is labelled "reimplementation".
