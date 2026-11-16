# Pre-registration — AGN birth-model kill test
Written 2026-11-16, before any real-data run. Frozen together with `configs/killtest.yaml` and the code in `agnb/`.
Do not edit after `final` has run.

## Question
When a new paper is born, can a neural model that mixes classical growth mechanisms with a learned
"birth memory" predict its individual references better than every classical growth model and a learned
linear combination of them? If not, AGN stops (author's standing rule: no negative-result paper).

## Data
ogbn-arxiv: 169,343 CS papers, citation edges (citing -> cited), publication year, 128-d text features
(OGB's averaged word embeddings of title + abstract), 40 subject labels.
Leakage caveat: OGB's word embeddings were trained on the full corpus, including later papers.

## Temporal protocol
| Role | Past graph | New papers (arrivals) |
|---|---|---|
| Train | year <= 2017 | 2018 (random 8,000 eligible) |
| Validation (model selection, early stopping) | year <= 2018 | 2019 (random 6,000 eligible) |
| **Final test, opened once** | year <= 2019 | **2020** (all eligible) |
Eligible arrival: >= 2 references into the past graph. References = citations to papers of year <= cutoff
(same-year citations excluded, since their order is unknown).
Prior exposure: aggregate statistics of 2018 and 2019 arrivals were examined in an earlier, unrelated
graph-heuristic study (union-based metrics). 2020 arrivals have never been analysed. 2020 is a partial year in OGB.

## Task and candidates
For each new paper p (given its text features and the snapshot), rank candidate papers as its references.
Candidates (identical for all methods): top-200 by citations received, top-200 by citations received in the
last year, top-300 by text cosine to p, and all references of the 30 most text-similar papers born in the
previous year. The candidate ceiling (best achievable Recall@20) is reported.

## Methods
Baselines: PA (preferential attachment), Fitness (last-year citations), Recency, Similarity (text cosine),
Copy-NN (copying model: references of the most similar paper born last year), Copy-kNN (similarity-weighted
copying from 30 last-year papers), LR-Ranker (listwise-trained linear combination of the five mechanism
features: popularity, fitness, age, cosine, soft copy).
AGN: five mechanism experts (popularity, fitness, recency, learned text similarity, learned birth-memory
attention over the 30 most similar last-year papers), z-scored per paper and mixed by a gate network
conditioned on the paper's text. Ablations: AGN-nogate (one global mixture), AGN-nomem (no memory expert).
Training: listwise softmax loss, all positives + 300 sampled negatives per paper, Adam, early stopping on
validation Recall@20 (patience 3, max 15 epochs). Grid on validation with seed 0: hidden {32, 64},
lr {0.001, 0.003}; LR-Ranker lr {0.01, 0.03, 0.1}. Chosen settings retrained with seeds 0, 1, 2.

## Metrics
Primary: per-paper Recall@20 (hits in top 20 / all of the paper's references), averaged over test papers;
learned models averaged over the 3 seeds per paper. Secondary: NDCG@20, Recall@50. Gate weights are
reported descriptively only.

## Decision rule (all four must hold on the 2020 test; otherwise NO-GO)
1. AGN minus the best baseline on test (best of the 7 baselines, chosen on test, i.e. the strictest
   comparison) >= 0.02 absolute AND >= 5% relative in Recall@20.
2. 95% paired bootstrap CI (2,000 resamples over papers) of that gain excludes 0.
3. AGN beats both AGN-nogate and AGN-nomem with 95% CIs excluding 0 (gate and memory are both needed).
4. AGN >= the best baseline in >= 70% of subject areas with >= 50 test papers.
GO -> next phase (more temporal networks across domains, novelty search, mechanism analysis).
NO-GO -> stop AGN entirely. No negative-result manuscript.
