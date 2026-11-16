# AGN — birth-model kill test (ogbn-arxiv)

Status: code ready, **not yet run on real data**. Read `PREREGISTRATION.md` first.

```
pip install -r requirements.txt
bash scripts/smoke.sh        # 1 min, synthetic data: checks the code only, results meaningless
bash scripts/run_all.sh      # real run: download (~80 MB), candidates, training, ONE-TIME test, decision
```
Results: `results/DECISION.md` (verdict), `results/test_summary.csv`, `results/test_per_paper.csv`,
`results/val_per_paper.csv`, `results/grid_log.json`, `results/run.log`, `results/environment.txt`.
`results/FINAL_LOCK` prevents re-running the test year. Never delete it.

Layout: `agnb/data.py` (data, snapshots) · `agnb/candidates.py` · `agnb/models.py` (baselines, AGN) ·
`agnb/run.py` (prepare/train/final/report) · `agnb/report.py` (decision rule) · `configs/`.
