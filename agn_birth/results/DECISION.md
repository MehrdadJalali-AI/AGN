# AGN kill test — pre-registered decision

Test: cutoff 2019 -> arrivals 2020; papers evaluated: 8065 (of 8065 eligible); candidate ceiling R@20 = 0.435

## Test results (per-paper mean; learned models averaged over seeds)

```
              R@20    N@20    R@50  R@20_sd_over_seeds
PA          0.0641  0.0802  0.0914                 NaN
Fitness     0.0661  0.0831  0.0953                 NaN
Recency     0.0166  0.0157  0.0259                 NaN
Similarity  0.0435  0.0470  0.0629                 NaN
Copy-NN     0.1016  0.1033  0.1287                 NaN
Copy-kNN    0.1478  0.1671  0.2076                 NaN
LR-Ranker   0.1396  0.1614  0.1989              0.0016
AGN         0.1440  0.1540  0.2147              0.0011
AGN-nogate  0.1417  0.1488  0.2151              0.0022
AGN-nomem   0.1280  0.1361  0.1941              0.0004
```

Best baseline on test: **Copy-kNN** (R@20 = 0.1478)
AGN R@20 = 0.1440; gain = -0.0038 absolute, -2.6% relative; 95% bootstrap CI of gain = [-0.0070, -0.0007]
Ablations (AGN minus variant): AGN-nogate: +0.0023 CI [+0.0008, +0.0037]; AGN-nomem: +0.0160 CI [+0.0142, +0.0177]
Subject areas with >=50 test papers: 22; AGN >= best baseline in 55%

## Conditions
1. Gain >= 0.02 absolute AND >= 5% relative: FAIL
2. 95% CI of gain excludes 0: FAIL
3. Gate and memory both needed (CIs vs AGN-nogate and AGN-nomem exclude 0): PASS
4. Consistent across subject areas (>= 70%): FAIL

## Verdict: **NO-GO — stop AGN (no paper)**

## Gate readout (descriptive only, not part of the decision)

Mean gate weight per mechanism: popularity 0.000, fitness 0.273, recency 0.000, similarity 0.522, memory 0.205
