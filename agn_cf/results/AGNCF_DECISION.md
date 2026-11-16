# AGN-CF final test

Tao et al. is a reimplementation. Plausible flip rate is the primary metric. Means are over targets for which a node was returned.

## GCN

```
dataset    method              plausible          raw          edits      edges       distance
Cora       B1                 86/200       86/200                0.0        1.0            0.0
Cora       B2                 70/200       70/200                0.0        4.0            0.0
Cora       Tao-reimpl          0/200      200/200           1411.505        1.0 4.585518891811371
Cora       AGN-CF            151/200      151/200              1.855      1.035 0.6110368835926056
Cora       AGN-CF-noplaus    121/200      177/200               3.01       1.03 0.9844042319059372
Cora       random-anchor     104/200      104/200     1.5894736842105264 1.394736842105263 0.49588763023677623
CiteSeer   B1                 79/200      123/200                0.0        1.0            0.0
CiteSeer   B2                 78/200      118/200                0.0       2.56            0.0
CiteSeer   Tao-reimpl          0/200      200/200           3684.485        1.0 6.461151721477509
CiteSeer   AGN-CF            167/200      167/200              0.935        1.0 0.3338335746526718
CiteSeer   AGN-CF-noplaus    121/200      171/200               0.89        1.0 0.33367721140384676
CiteSeer   random-anchor     140/200      140/200     1.1546391752577319 1.1649484536082475 0.4716820286721298
PubMed     B1                 90/200       90/200                0.0        1.0            0.0
PubMed     B2                 61/200       61/200                0.0        3.0            0.0
PubMed     Tao-reimpl          0/200      200/200              500.0        1.0 2.2513827300071716
PubMed     AGN-CF            149/200      149/200                0.0        1.0            0.0
PubMed     AGN-CF-noplaus    145/200      149/200                0.0        1.0            0.0
PubMed     random-anchor     101/200      101/200                0.0 1.2164948453608246            0.0
```

## GAT

```
dataset    method              plausible          raw          edits      edges       distance
Cora       B1                 65/200       65/200                0.0        1.0            0.0
Cora       B2                 82/200       82/200                0.0      3.725            0.0
Cora       Tao-reimpl          0/200      200/200            1390.01        1.0 4.495758563280106
Cora       AGN-CF            115/200      115/200              0.995       1.46 0.26462899446487426
Cora       AGN-CF-noplaus    101/200      164/200               4.94      1.265 1.2115991669893265
Cora       random-anchor      80/200       80/200     0.6230366492146597 2.06282722513089 0.21414027788252107
CiteSeer   B1                 68/200       94/200                0.0        1.0            0.0
CiteSeer   B2                 85/200      112/200                0.0       2.16            0.0
CiteSeer   Tao-reimpl          0/200      200/200           3647.455        1.0 6.300980896949768
CiteSeer   AGN-CF            151/200      151/200                1.0       1.33 0.36190290868282315
CiteSeer   AGN-CF-noplaus     81/200      158/200              1.285      1.285 0.4374396526813507
CiteSeer   random-anchor     116/200      116/200     1.145945945945946 1.481081081081081 0.38594694330885604
PubMed     B1                 90/200       90/200                0.0        1.0            0.0
PubMed     B2                 99/200       99/200                0.0       2.17            0.0
PubMed     Tao-reimpl          0/200      200/200              497.5        1.0 2.193182536736131
PubMed     AGN-CF            116/200      116/200                0.0      1.305            0.0
PubMed     AGN-CF-noplaus    116/200      128/200              1.425      1.275 0.2383576250076294
PubMed     random-anchor      75/200       75/200                0.0 1.634517766497462            0.0
```

## GCN conditions

- Cora: best baseline B1 plausible flip rate = 0.43; AGN-CF = 0.755; +15 points and ≥ 1.25×: PASS
- CiteSeer: best baseline B1 plausible flip rate = 0.395; AGN-CF = 0.835; +15 points and ≥ 1.25×: PASS
- PubMed: best baseline B1 plausible flip rate = 0.45; AGN-CF = 0.745; +15 points and ≥ 1.25×: PASS

## GAT conditions

- Cora: best baseline B2 = 0.41; AGN-CF = 0.575; gain = 0.16499999999999998; ≥ +5 points: PASS
- CiteSeer: best baseline B2 = 0.425; AGN-CF = 0.755; gain = 0.33; ≥ +5 points: PASS
- PubMed: best baseline B2 = 0.495; AGN-CF = 0.58; gain = 0.08499999999999996; ≥ +5 points: PASS

GCN condition on all three datasets: PASS
GAT gains of at least +5 points: 3 of 3

## Verdict: **GO**

Final-test runtime seconds: 3673.306840708
