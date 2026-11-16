# AGN extension

Continuous learning rate selected on Amazon-Photo: 0.01. The tuning set is binary, so every grid row used the existing binary editor and the earliest row was selected. λ = 0.2, steps = 16, budget = 32 were not retuned.
GANI: not run. The official repository https://github.com/alexfanjn/GANI trains a deeprobust GCN on the nettack split, injects 10 nodes, and searches a global homophily objective.

Feature types and test accuracy:

- Amazon-Computers: binary
  - GCN: 1897/2750
  - GAT: 2443/2750
  - SAGE: 2475/2750
- Amazon-Photo: binary
  - GCN: 417/1530
  - GAT: 1446/1530
  - SAGE: 1427/1530
- Coauthor-CS: binary
  - GCN: 3417/3666
  - GAT: 3401/3666
  - SAGE: 3440/3666
- Coauthor-Physics: binary
  - GCN: 6639/6898
  - GAT: 6618/6898
  - SAGE: 6673/6898

Plausible flip rate is the primary metric. Means are over returned nodes. Outlier rate is the fraction of returned nodes flagged by the isolation forest.

## GCN

```
dataset            method                      plausible          raw        edits      edges       distance    outlier
Amazon-Computers   B1                         19/200       21/200              0.0        1.0            0.0      0.785
Amazon-Computers   B2                          4/200        5/200              0.0      22.95            0.0      0.785
Amazon-Computers   Copy-best-of-10            31/200       33/200              0.0       5.26            0.0       0.79
Amazon-Computers   Copy-best-of-10-noedges    29/200       31/200              0.0        1.0            0.0      0.795
Amazon-Computers   constrained-attack         40/200       40/200     28.097560975609756        1.0 4.96699389015756        1.0
Amazon-Computers   AGN                        34/200       34/200            0.005      1.255          0.005        1.0
Amazon-Photo       B1                          0/200        0/200              0.0        1.0            0.0        1.0
Amazon-Photo       B2                          0/200        0/200              0.0        3.0            0.0        1.0
Amazon-Photo       Copy-best-of-10             0/200        0/200              0.0        1.0            0.0        1.0
Amazon-Photo       Copy-best-of-10-noedges     0/200        0/200              0.0        1.0            0.0        1.0
Amazon-Photo       constrained-attack          0/200        0/200             25.3        1.0 4.487921686172485        1.0
Amazon-Photo       AGN                         0/200        0/200              0.0        1.0            0.0        1.0
Coauthor-CS        B1                         27/200       63/200              0.0        1.0            0.0        0.0
Coauthor-CS        B2                         15/200       28/200              0.0      6.975            0.0        0.0
Coauthor-CS        Copy-best-of-10            31/200       89/200              0.0       1.17            0.0      0.035
Coauthor-CS        Copy-best-of-10-noedges    31/200       89/200              0.0        1.0            0.0      0.035
Coauthor-CS        constrained-attack         37/200       37/200     16.86021505376344        1.0 3.3722156645149313        0.0
Coauthor-CS        AGN                        70/200       70/200     0.06748466257668712        1.0 0.04693424262883473        0.0
Coauthor-Physics   B1                          7/200       59/200              0.0        1.0            0.0        0.0
Coauthor-Physics   B2                          1/200        8/200              0.0       11.0            0.0        0.0
Coauthor-Physics   Copy-best-of-10             8/200       79/200              0.0       1.04            0.0        0.0
Coauthor-Physics   Copy-best-of-10-noedges     8/200       79/200              0.0        1.0            0.0        0.0
Coauthor-Physics   constrained-attack         12/200       12/200         19.53125        1.0        3.90625        0.0
Coauthor-Physics   AGN                        45/200       45/200     0.4370860927152318 1.0264900662251655 0.20945528406181083        0.0
```

## GAT

```
dataset            method                      plausible          raw        edits      edges       distance    outlier
Amazon-Computers   B1                          8/200        8/200              0.0        1.0            0.0      0.995
Amazon-Computers   B2                         11/200       11/200              0.0      6.315            0.0      0.995
Amazon-Computers   Copy-best-of-10            14/200       14/200              0.0       2.86            0.0      0.985
Amazon-Computers   Copy-best-of-10-noedges    11/200       11/200              0.0        1.0            0.0      0.985
Amazon-Computers   constrained-attack         13/200       13/200            30.68        1.0 5.426892671585083      0.995
Amazon-Computers   AGN                        14/200       14/200              0.0      1.785            0.0      0.735
Amazon-Photo       B1                          3/200        3/200              0.0        1.0            0.0       0.79
Amazon-Photo       B2                          5/200        5/200              0.0      5.875            0.0       0.79
Amazon-Photo       Copy-best-of-10            13/200       13/200              0.0        2.6            0.0        0.8
Amazon-Photo       Copy-best-of-10-noedges     8/200        8/200              0.0        1.0            0.0        0.8
Amazon-Photo       constrained-attack         11/200       11/200            31.52        1.0 5.572001340389252       0.98
Amazon-Photo       AGN                        13/200       13/200              0.0       1.71            0.0      0.535
Coauthor-CS        B1                         16/200       16/200              0.0        1.0            0.0        0.0
Coauthor-CS        B2                         29/200       29/200              0.0      4.395            0.0        0.0
Coauthor-CS        Copy-best-of-10            41/200       41/200              0.0      3.395            0.0        0.0
Coauthor-CS        Copy-best-of-10-noedges    25/200       25/200              0.0        1.0            0.0        0.0
Coauthor-CS        constrained-attack         33/200       33/200             29.2        1.0 5.176021549701691        0.0
Coauthor-CS        AGN                        64/200       64/200            2.305      1.895 0.5009233009815216        0.0
Coauthor-Physics   B1                          2/200        3/200              0.0        1.0            0.0        0.0
Coauthor-Physics   B2                          8/200        9/200              0.0      4.895            0.0        0.0
Coauthor-Physics   Copy-best-of-10            21/200       23/200              0.0       2.84            0.0        0.0
Coauthor-Physics   Copy-best-of-10-noedges    15/200       16/200              0.0        1.0            0.0        0.0
Coauthor-Physics   constrained-attack         13/200       13/200            29.54        1.0 5.398165535926819        0.0
Coauthor-Physics   AGN                        66/200       66/200             4.55       1.81 0.9839515733718872        0.0
```

## SAGE

```
dataset            method                      plausible          raw        edits      edges       distance    outlier
Amazon-Computers   B1                          7/200        7/200              0.0        1.0            0.0       0.34
Amazon-Computers   B2                         14/200       14/200              0.0      5.355            0.0       0.34
Amazon-Computers   Copy-best-of-10            24/200       24/200              0.0      3.585            0.0       0.37
Amazon-Computers   Copy-best-of-10-noedges    11/200       11/200              0.0        1.0            0.0      0.345
Amazon-Computers   constrained-attack         11/200       11/200           30.745        1.0 5.441760326623917       0.57
Amazon-Computers   AGN                        24/200       24/200              0.0      2.525            0.0       0.57
Amazon-Photo       B1                          9/200        9/200              0.0        1.0            0.0        1.0
Amazon-Photo       B2                         10/200       10/200              0.0      9.215            0.0        1.0
Amazon-Photo       Copy-best-of-10            19/200       19/200              0.0      4.235            0.0        1.0
Amazon-Photo       Copy-best-of-10-noedges    10/200       10/200              0.0        1.0            0.0        1.0
Amazon-Photo       constrained-attack         12/200       12/200            30.56        1.0 5.402295715808869        1.0
Amazon-Photo       AGN                        19/200       19/200              0.0       3.29            0.0      0.995
Coauthor-CS        B1                          8/200       13/200              0.0        1.0            0.0       0.07
Coauthor-CS        B2                          9/200       14/200              0.0       6.56            0.0       0.07
Coauthor-CS        Copy-best-of-10            14/200       22/200              0.0      4.625            0.0       0.07
Coauthor-CS        Copy-best-of-10-noedges     8/200       13/200              0.0        1.0            0.0       0.07
Coauthor-CS        constrained-attack          9/200        9/200     28.094594594594593        1.0 5.13338083189887        0.0
Coauthor-CS        AGN                        20/200       20/200             0.08      2.315 0.02805883765220642        0.0
Coauthor-Physics   B1                          3/200        7/200              0.0        1.0            0.0        0.0
Coauthor-Physics   B2                          4/200       10/200              0.0       11.0            0.0        0.0
Coauthor-Physics   Copy-best-of-10             4/200       14/200              0.0        5.6            0.0        0.0
Coauthor-Physics   Copy-best-of-10-noedges     4/200       13/200              0.0        1.0            0.0        0.0
Coauthor-Physics   constrained-attack          5/200        5/200     19.33613445378151        1.0 4.226569275896089        0.0
Coauthor-Physics   AGN                        21/200       21/200            0.445       3.73 0.13625076293945312        0.0
```

## Conditions

- Amazon-Computers: AGN 72/600, Copy-best-of-10 69/600, constrained-attack 64/600, best 69/600; +60 flips: FAIL
- Amazon-Photo: AGN 32/600, Copy-best-of-10 32/600, constrained-attack 23/600, best 32/600; +60 flips: FAIL
- Coauthor-CS: AGN 154/600, Copy-best-of-10 86/600, constrained-attack 79/600, best 86/600; +60 flips: PASS
- Coauthor-Physics: AGN 132/600, Copy-best-of-10 33/600, constrained-attack 30/600, best 33/600; +60 flips: PASS
- Amazon-Computers outlier gap AGN − Copy-best-of-10 = 0.05333333333333323
- Amazon-Photo outlier gap AGN − Copy-best-of-10 = -0.08999999999999986
- Coauthor-CS outlier gap AGN − Copy-best-of-10 = -0.035
- Coauthor-Physics outlier gap AGN − Copy-best-of-10 = 0.0
- mean outlier gap = -0.017916666666666657; ≤ 0.10: PASS

Datasets with the +60 flip margin: 2 of 4

## Verdict: **NO-GO: AGN is limited to binary features. No paper.**

Final-run runtime seconds: 14794.714232082999
