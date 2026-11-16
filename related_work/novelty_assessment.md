# Novelty assessment

## Bottom line

**Verdict: partly novel.** The search found no paper that matches the full proposed package: a content-conditioned, per-arriving-node reference-set ranker that (i) gates *named classical growth scores*, (ii) adds a learned attention-over-recent-similar-births/reference-sets expert, (iii) evaluates the whole outgoing set on a strictly temporal holdout, and (iv) presents gates as descriptive mechanism readouts. The ingredients are well represented individually. The closest prior work, especially discrete-choice formation, makes a claim of a new *mechanism-estimation framework* untenable without direct comparisons and synthetic recovery tests.

## Ten closest papers

| Paper | What it does | Overlap | Exact difference from AGN |
|---|---|---:|---|
| Overgoor, Benson & Ugander (2019), `overgoor2019choosing` | Fits conditional multinomial-logit formation choices from target-node features; unifies PA, closure, and fitness. | High | It uses interpretable statistical choice features/mixtures rather than learned birth-memory attention and a feature-conditioned neural gate; its coefficient interpretation is stronger than AGN's present gate interpretation. |
| Gupta & Porter (2022), `gupta2022mixed` | Models repeated network choices with mixed/repeated-choice logit and studies a patent citation case. | High | It is a discrete-choice estimator, not an attention-based ranker of an observed new node's full reference set with named neural experts. |
| Krapivsky & Redner (2005), `krapivsky2005copying` | Gives a stochastic copying growth process. | High | Its copying is a generative rule; AGN learns which recent similar births to attend to and scores their observed references conditional on paper text. |
| Bhagavatula et al. (2018), `bhagavatula2018citation` | Retrieves text-nearest papers and discriminatively reranks citation candidates. | High | It has no temporal growth-expert decomposition, candidate eligibility protocol, or copying-like memory over prior births. |
| TGN, `rossi2020tgn` | Maintains learnable node memories for timed-event link prediction, including inductive settings. | High | Its memory is node state updated by events; AGN's memory is an explicit attention distribution over recently born, text-similar source nodes and their outgoing references, mixed with fixed classical scores. |
| TGAT, `xu2020tgat` | Uses time encoding and temporal attention to learn inductive embeddings for link prediction. | High | It has no mechanism-specific expert scores or per-birth gate; it predicts generic future links rather than a citation birth's complete reference set. |
| DyGFormer, `yu2023dygformer` | Uses historical interaction sequences, co-occurrence encoding, and a transformer for dynamic link prediction. | Medium-high | It learns a monolithic temporal representation, not an explicit combination of PA/fitness/recency/similarity/copying experts. |
| TIGGER, `gupta2022tigger` | Autoregressively generates temporal interaction graphs with point processes, including an inductive variant. | Medium-high | It models graph/event generation, including timing, whereas AGN conditions on observed arrivals and ranks their citations; AGN is not a full graph generator. |
| Touwen et al. (2024), `touwen2024learning` | Uses interpretable dynamic features and supervised model selection among simulated growth mechanisms. | Medium | It selects network-level simulated model classes rather than predicting each new node's links or using text-conditioned gates/memory. |
| Golosovsky (2018), `golosovsky2018mechanisms` | Analyses the ambiguity between PA and fitness and proposes recursive search with memory. | Medium | It is analytical/aggregate and does not learn a node-specific mixture or evaluate reference recommendation. |

## Direct question: has the complete construction already been published?

**Not located in this verified search; do not phrase this as proof of absence.** The closest evidence is overlapping, not identical:

- Overgoor et al. state that each new edge is a **“choice”** and that their framework unifies preferential attachment, triadic closure, and fitness ([paper](https://arxiv.org/abs/1811.05008)). This establishes the classical-feature/formation-choice precedent, including mixture estimation, but not a neural gate over those experts or attention over predecessor reference sets.
- TGN describes a **“novel combination of memory modules and graph-based operators”** ([paper](https://arxiv.org/abs/2006.10637)). This establishes learned temporal memory for prediction, but not a copying-style source-birth/reference-set memory or classical-mechanism gate.
- TGAT says it can **“inductively infer embeddings for both new and observed nodes”** ([paper](https://arxiv.org/abs/2002.07962)). This establishes cold-start temporal representation learning, not the proposed reference-set task or interpretable named experts.
- TIGGER combines **“temporal point processes with auto-regressive modeling”** ([paper](https://arxiv.org/abs/2203.03564)). This establishes temporal graph generation, not a discriminative ranker with observed new-node content.
- Touwen et al. call their contribution a **“model-selection method for dynamic networks”** ([paper](https://doi.org/10.1038/s41598-024-61940-4)). It is a direct precedent for interpreting mechanisms, but operates at the network/model-class level.

Accordingly, the defensible claim is: *AGN combines established components in a previously unlocated configuration and asks a sharper per-birth prediction question.* The phrase “first” is not defensible from this search.

## What AGN can and cannot claim

If the preregistered experiment succeeds, AGN can claim that, **on the stated ogbn-arxiv temporal protocol and candidate set**, its specified ranker improves selected retrieval metrics over the included baselines; that its two ablations test the utility of its gate and memory; and that its gates are descriptive, model-specific allocations among the five implemented score channels. The candidate ceiling, subject-area stratification, and paired bootstrap are important constraints on that claim.

It cannot yet claim that any observed weight measures a true causal growth mechanism, that copying rather than popularity explains node births, that the learned memory is a new copying model, that the result generalizes beyond citation graphs, that it establishes a universal growth law, or that it predicts real-world references from raw contemporary manuscript text. The OGB feature construction and temporal-candidate restriction also limit claims about real deployment. A positive result does not identify mechanisms because feature correlations, candidate construction, and flexible encoders can redistribute signal among experts.

## Reviewer-demanded baselines missing from `agn_birth/`

1. **Conditional multinomial-logit discrete choice.** Fit PA/degree, age, text similarity, author/field where permitted, and closure/copy features on exactly AGN's temporal choice sets. Report likelihood, calibration, Recall@K, and coefficient uncertainty. Use [torch-choice](https://github.com/gsbDBI/torch-choice) or [mlogit](https://cran.r-project.org/package=mlogit); this is the most important missing baseline because it is the closest interpretation-aware comparator to `overgoor2019choosing` and `gupta2022mixed`.
2. **Mixed/repeated-choice logit.** Compare an MNL model, a mixed-logit/repeated-choice model, and an interaction-aware specification; this tests heterogeneity without silently calling neural gate weights causal. The published Gupta--Porter formulation is the target reference, while `mlogit` supports random-utility estimation.
3. **TGAT and TGN under the identical birth protocol.** Train inductively with strictly causal message/history access and the same candidate lists. Official/reference implementations: [TGAT](https://github.com/mludolph/TGAT) and [TGN](https://github.com/twitter-research/tgn). Report the no-text and text-enabled versions separately.
4. **DyGFormer/GraphMixer-class long-history predictors.** Use [DyGLib](https://github.com/yule-BUAA/DyGLib), whose paper releases a unified evaluation library. This tests whether AGN's memory benefit is simply a weaker version of sequence-history encoding.
5. **Strong content citation reranker.** Reproduce or adapt [Citeomatic](https://github.com/allenai/citeomatic) on the same temporally valid corpus. This determines whether “birth memory” adds value over a modern content retrieval-and-rerank stack.
6. **Temporal benchmark implementation and simple temporal heuristics.** Evaluate EdgeBank/recency/popularity and selected TGN-family models using [TGB](https://github.com/shenyangHuang/TGB); report a candidate-unrestricted supplemental setting if tractable. TGB's disciplined negative protocol helps expose candidate-set artifacts.
7. **Synthetic identifiability recovery.** Generate births from known PA, aging, fitness, copying, and mixtures; test whether gates recover the known mixture and whether correlated mechanisms are separable. This is necessary before using gates as anything beyond descriptive readouts.

## Method-reading record

The method sections/readmes examined in this session for the ten closest works covered: conditional-choice/MNL feature modelling and sampled evaluation (Overgoor); repeated-choice/mixed-logit setup (Gupta--Porter); copying mechanism (Krapivsky--Redner); candidate retrieval and discriminative reranking (Bhagavatula et al.); memory/message modules (TGN); temporal attention and inductive protocol (TGAT); historical-sequence transformer (DyGFormer); point-process autoregression (TIGGER); synthetic mechanism-classification features (Touwen et al.); and the PA--fitness identifiability argument/recursive-search memory (Golosovsky). This is a targeted reading record, not a systematic-review guarantee.

## Suggested items not included

These are deliberately absent from `references.bib`; complete, internally consistent metadata plus an abstract were **not** secured from the searched source in this session, so no citation has been reconstructed from memory.

- Leskovec, Kleinberg, and Faloutsos, *Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations* (including the forest-fire model): not found with a resolved record/abstract in this session.
- CAWN / “Causal Anonymous Walks” for temporal graphs: not found with a resolved record/abstract in this session.
- GraphMixer: not found with a resolved record/abstract in this session.
- TagGen: not found with a resolved record/abstract in this session.
- *Temporal Graph Neural Network-Powered Paper Recommendation on Dynamic Citation Networks*: a search result identified the arXiv record, but did not supply a complete author record for verification; left out.
- A graph-specific mixture-of-experts paper that establishes gate weights as scientifically valid growth-mechanism estimates: not found. Generic MoE theory is represented only by `chen2022moe`.
