# AGN-CF confirmation

Settings are the frozen λ = 0.2, steps = 16, budget = 32. No retuning.
GANI: not run. The official repository https://github.com/alexfanjn/GANI trains a deeprobust GCN on the nettack split with seed 2021, injects 10 nodes, and searches a global homophily objective. It does not attack the frozen public-split models or insert one node to change one target's class.

Plausible flip rate is the primary metric. Means are over targets for which a node was returned.

## GCN

```
dataset    method                      plausible          raw          edits        edges       distance    seconds
Cora       B1                         71/200       71/200                0.0          1.0            0.0        0.6
Cora       B2                         50/200       50/200                0.0          4.0            0.0        0.6
Cora       Copy-best-of-10            97/200       97/200                0.0        1.535            0.0       28.2
Cora       Copy-best-of-10-noedges    92/200       92/200                0.0          1.0            0.0        4.6
Cora       constrained-attack         94/200       94/200              3.205          1.0 1.2913136649131776      137.9
Cora       AGN-CF                    148/200      148/200                2.2        1.035 0.7103936475515366      162.1
CiteSeer   B1                         77/200      117/200                0.0          1.0            0.0        1.2
CiteSeer   B2                         79/200      114/200                0.0        2.515            0.0        1.1
CiteSeer   Copy-best-of-10            95/200      137/200                0.0          1.2            0.0       12.2
CiteSeer   Copy-best-of-10-noedges    95/200      137/200                0.0          1.0            0.0        4.6
CiteSeer   constrained-attack        123/200      123/200              2.985          1.0 1.261041830778122      178.1
CiteSeer   AGN-CF                    171/200      171/200              1.255          1.0 0.4591431057453155      147.2
PubMed     B1                         93/200       93/200                0.0          1.0            0.0        1.8
PubMed     B2                         64/200       64/200                0.0          3.0            0.0        1.8
PubMed     Copy-best-of-10           152/200      152/200                0.0         1.36            0.0       33.8
PubMed     Copy-best-of-10-noedges   147/200      150/200                0.0          1.0            0.0        6.3
PubMed     constrained-attack        164/200      164/200              16.64          1.0 0.22619131490588187       68.8
PubMed     AGN-CF                    152/200      152/200                0.0        1.015            0.0      164.5
```

## GAT

```
dataset    method                      plausible          raw          edits        edges       distance    seconds
Cora       B1                         59/200       59/200                0.0          1.0            0.0        0.9
Cora       B2                         80/200       80/200                0.0        3.685            0.0        0.9
Cora       Copy-best-of-10            99/200       99/200                0.0        2.125            0.0       35.9
Cora       Copy-best-of-10-noedges    86/200       86/200                0.0          1.0            0.0        5.9
Cora       constrained-attack         85/200       85/200              2.665          1.0 1.341817142367363      228.6
Cora       AGN-CF                    121/200      121/200               1.05        1.465 0.317184693813324      250.4
CiteSeer   B1                         71/200      102/200                0.0          1.0            0.0        1.4
CiteSeer   B2                         84/200      118/200                0.0         2.21            0.0        1.4
CiteSeer   Copy-best-of-10            90/200      128/200                0.0         1.47            0.0       17.2
CiteSeer   Copy-best-of-10-noedges    83/200      121/200                0.0          1.0            0.0        6.3
CiteSeer   constrained-attack        114/200      114/200              4.235          1.0 1.5280291640758514      235.6
CiteSeer   AGN-CF                    153/200      153/200               0.93          1.3 0.3278120857477188      203.0
PubMed     B1                        101/200      101/200                0.0          1.0            0.0        3.9
PubMed     B2                        109/200      109/200                0.0         2.23            0.0        5.0
PubMed     Copy-best-of-10           124/200      124/200                0.0          1.4            0.0       90.0
PubMed     Copy-best-of-10-noedges   111/200      111/200                0.0          1.0            0.0       18.0
PubMed     constrained-attack        164/200      164/200              15.52          1.0 0.211795591711998      153.9
PubMed     AGN-CF                    124/200      124/200                0.0          1.3            0.0      589.0
```

## Conditions

- GCN Cora: AGN-CF 148/200, Copy-best-of-10 97/200, constrained-attack 94/200; +20 flips vs copy: PASS; +20 flips vs attack: PASS
- GCN CiteSeer: AGN-CF 171/200, Copy-best-of-10 95/200, constrained-attack 123/200; +20 flips vs copy: PASS; +20 flips vs attack: PASS
- GCN PubMed: AGN-CF 152/200, Copy-best-of-10 152/200, constrained-attack 164/200; +20 flips vs copy: FAIL; +20 flips vs attack: FAIL
- GAT Cora: AGN-CF 121/200, Copy-best-of-10 99/200; AGN-CF ≥ copy: PASS
- GAT CiteSeer: AGN-CF 153/200, Copy-best-of-10 90/200; AGN-CF ≥ copy: PASS
- GAT PubMed: AGN-CF 124/200, Copy-best-of-10 124/200; AGN-CF ≥ copy: PASS

GCN datasets with AGN-CF ≥ Copy-best-of-10 + 20 flips: 2 of 3
GCN datasets with AGN-CF ≥ constrained attack + 20 flips: 2 of 3
GAT datasets with AGN-CF ≥ Copy-best-of-10: 3 of 3

## Verdict: **CONFIRMED**

Confirmation runtime seconds: 2802.73758375
