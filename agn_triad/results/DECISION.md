# AGN-Triad kill test — pre-registered decision

Test periods (cutoff = fraction of the event sequence; window = next 10% of events):

```
        n_eligible  n_pos  n_groups  frac_covered
cutoff                                           
0.8        46662.0  703.0   20000.0      0.955574
0.9        47545.0  738.0   20000.0      0.952109
```

## Average precision on test (learned methods: mean over seeds)

```
cutoff           0.8     0.9
method                      
AGN-Triad     0.0924  0.0976
CopySub       0.0652  0.0773
GBT           0.0786  0.0829
MLP           0.0914  0.1000
PairAdditive  0.0922  0.0982
```

AGN-Triad per seed:

```
cutoff     0.8     0.9
seed                  
0       0.0919  0.0974
1       0.0916  0.0975
2       0.0937  0.0980
```

- Cutoff 0.8: best baseline PairAdditive AP=0.0922; AGN-Triad AP=0.0924; gain +0.0002 absolute, +0.2% relative -> FAIL
- Cutoff 0.9: best baseline MLP AP=0.1000; AGN-Triad AP=0.0976; gain -0.0023 absolute, -2.3% relative -> FAIL

Rule: AGN-Triad must be >= 15% relative AND >= 0.01 absolute above the best baseline in BOTH test deciles.
HIT (Liu, Ma & Li) was not run in this pilot; a GO is provisional until it is compared.

## Verdict: **NO-GO — stop AGN (no paper)**
