# Pre-registration — AGN-Triad kill test (concept 3)
Written 2026-11-16, before any model was trained on real data. Frozen with `configs/killtest.yaml` and `agnt/`.
Do not edit after `final` has run.

## Question
Can a neural scorer whose only three-way capacity is a learned redistribution that preserves pair-level
evidence predict which three familiar tags will be combined for the first time better than copying,
pair-based models, gradient-boosted trees and an unconstrained neural network with the same features?
If not, AGN stops (author's standing rule: no negative-result paper).

## Data
tags-math-sx (Benson et al. 2018, simplicial closure; github.com/arbenson/ScHoLP-Data): 822,059 timestamped
math.stackexchange questions, each a set of 1–5 tags. Events are sorted by time; cutoffs are fractions of the
event sequence. Vocabulary: the 80 most frequent tags in the first 20% of events (fixed).
Prior exposure: before freezing, only history-based task sizes for train/val cutoffs were checked for
feasibility (eligible triples 34–46k, ~900–1,500 positives per cutoff, groups cover ~96–98% of triples).
No model was trained on real data and the test periods (80–100%) were never built.

## Task
At cutoff c, eligible triples = three tags whose three pairs have each co-occurred before c and which have never
co-occurred as a triple before c. Label = the triple occurs in an event in [c, c + 10% of events). All eligible
triples are scored, including those no update group covers.
Train cutoffs 20, 30, 40, 50%; validation 60, 70%; **final test 80% and 90% (opened once)**.

## Features (history before c only)
Per pair: log co-occurrence count, log recency-weighted count (scale 10% of events), recency of last
co-occurrence, log number of closed triangles of the pair. Per tag: log frequency. Per triple: log number of
common co-occurrence neighbours; log number of fourth tags whose three triangles with the pairs are all closed.
Triple feature vector phi = sorted pair features (12) + sorted tag frequencies (3) + 2 motif counts = 17.

## Methods
- CopySub: recency-weighted copying with one-tag substitution: for each pair of the triple, sum over
  historically closed triangles on that pair of exp(-age/tau) x cosine similarity between the substituted
  tags' co-occurrence profiles. tau chosen on validation from {0.5, 1, 2}.
- PairAdditive: sum of a learned pair function over the 3 pairs + learned tag function (no three-way term).
- GBT: HistGradientBoosting on phi; depth {3, 6} x learning rate {0.05, 0.1}, 300 iterations, chosen on validation.
- MLP: unconstrained 2-layer MLP on phi.
- **AGN-Triad**: PairAdditive base + residual from 2 rounds of pair-preserving updates over 20,000 sampled
  groups of 8 eligible triples {x_i, y_j, z_k} (i, j, k in {0,1}, 6 distinct tags), signs (-1)^(i+j+k).
  alpha = MLP([sum_s v_s phi_s, sum_s v_s logit_s]); z_s += alpha v_s. Each update leaves the residual sum of
  every pair unchanged (unit-tested).
- Not run in this pilot: HIT (Liu, Ma & Li). A GO is therefore provisional until HIT is compared.
Torch models: full-batch BCE with positive weighting, Adam, early stopping on mean validation AP
(patience 25, max 300 epochs); grid hidden {32, 64} x lr {0.001, 0.003} with seed 0; chosen setting retrained
with seeds 0, 1, 2. GBT seeds 0, 1, 2.

## Metric and decision rule
Primary metric: average precision over all eligible triples at each test cutoff; learned methods averaged
over seeds. **PROVISIONAL GO** only if, at BOTH test cutoffs (80% and 90%), AGN-Triad AP >= 1.15 x the best
baseline AND >= best baseline + 0.01 (best of CopySub, PairAdditive, GBT, MLP, chosen per cutoff on test,
i.e. the strictest comparison). Otherwise **NO-GO**: stop AGN entirely; no negative-result manuscript.
