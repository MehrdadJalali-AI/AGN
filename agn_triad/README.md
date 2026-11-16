# AGN-Triad — kill test (concept 3: first-time combinations with pair-preserving updates)

Status: code ready and smoke-tested; **not yet run on real data**. Read `PREREGISTRATION.md` first.
Data are included in `data/raw/` (tags-math-sx, from github.com/arbenson/ScHoLP-Data).

```
pip install -r requirements.txt
bash scripts/smoke.sh     # synthetic; checks code only
bash scripts/run_all.sh   # real run; ONE-TIME test
```
Results: `results/DECISION.md`, `results/test_ap.csv`, `results/val_ap.csv`, `results/selected_hparams.json`,
`results/run.log`, `results/environment.txt`. `results/FINAL_LOCK` prevents re-running the test. Never delete it.
