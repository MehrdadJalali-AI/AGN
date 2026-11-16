# AGN

AGN inserts one node into a trained graph neural network so that a chosen target node changes its predicted class. The classifier is not retrained. The inserted node is copied from a real training node and then edited, and it is kept only when its features and its degree stay inside the range of real nodes.

This repository is the latest experiment. It contains the frozen protocols, the code that implements them, and the results of the completed run.

## Layout

All of the experiment is in `agn_cf/`.

| Path | Role |
| --- | --- |
| `PREREGISTRATION.md` | Headroom check, written before that run |
| `PREREGISTRATION_AGNCF.md` | Tuning and primary test |
| `PREREGISTRATION_CONFIRM.md` | Fresh-target confirmation |
| `PREREGISTRATION_EXTEND.md` | Co-authorship and Amazon graphs |
| `headroom.py` | Headroom baselines |
| `agncf.py` | Tuning and primary test |
| `confirm.py` | Confirmation |
| `extend.py` | Four-graph extension |
| `extend_fast.py` | Local logit check used by the extension |
| `results/` | Outputs of the completed run, including the lock files |

`results/` is the reference run. Do not edit those files, and do not run the scripts in place on top of them. A rerun has to use a separate results directory, as described below.

## Environment that produced the published results

- Python 3.9.6
- numpy 2.0.2
- scipy 1.13.1
- torch 2.8.0
- torch_geometric 2.6.1
- scikit-learn 1.6.1
- CPU only, on macOS arm64

Training and search use the seeds fixed in the protocols. A rerun on another machine or another PyTorch build can change floating-point distances. Compare the integer flip counts with the published files. Those counts are the result.

## Setup

From `agn_cf/`:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==2.0.2 scipy==1.13.1 scikit-learn==1.6.1 torch==2.8.0
pip install torch_geometric==2.6.1
```

The scripts train on CPU. They do not select a GPU.

## Data

The datasets are not in this repository. The first run downloads them into `agn_cf/data/` through PyTorch Geometric:

- Planetoid, public split: Cora, CiteSeer, PubMed
- Amazon: Computers, Photo
- Coauthor: CS, Physics

Leave several gigabytes free. Coauthor-Physics is the large download. The download needs a network connection, and it happens inside the scripts. No separate data command is required.

## Run from scratch

The published `results/` directory already contains lock files. `agncf.py final`, `confirm.py`, and `extend.py` refuse to start when their own lock is present. `confirm.py` also reads `results/AGNCF_FINAL_LOCK` on import, and `extend.py` reads both earlier locks on import. The locks are there so a finished run cannot be overwritten by accident.

To repeat the run without touching the published outputs:

```bash
cd agn_cf
mv results results_published
mkdir results
python headroom.py
python agncf.py tune
python agncf.py final
python confirm.py
python extend.py
```

Run the five commands in that order, from `agn_cf/`, with the virtual environment active. Each command is a full stage. There are no extra arguments.

1. `python headroom.py` trains one GCN per citation graph and runs the four copy baselines. It writes `results/headroom.json`, `results/environment.txt`, and `results/run.log`.
2. `python agncf.py tune` scores the frozen λ and budget grid on 100 Cora GCN targets drawn with seed 1. It writes `results/agncf_selected.json`. The published selection is λ = 0.2, 16 steps, and a feature budget of 32.
3. `python agncf.py final` trains the GCN and the GAT, draws 200 targets per model and dataset with seed 2, writes `results/AGNCF_FINAL_LOCK`, and only then evaluates the methods. It stops if that lock already exists. It also stops if `agncf_selected.json` is missing. Outputs are `results/agncf_test_per_target.csv`, `results/agncf_test_summary.json`, `results/agncf_examples.json`, `results/agncf_run.log`, and `results/AGNCF_DECISION.md`.
4. `python confirm.py` draws 200 fresh targets with seed 3, disjoint from the headroom, tuning, and primary-test ids. It stops if `results/CONFIRM_LOCK` already exists, or if the selected settings are not λ = 0.2, 16 steps, and budget 32. Outputs are `results/confirm_per_target.csv`, `results/confirm_summary.json`, `results/confirm_run.log`, and `results/CONFIRM_DECISION.md`.
5. `python extend.py` tunes the continuous learning rate on Amazon-Photo and then evaluates GCN, GAT, and GraphSAGE on Amazon-Computers, Amazon-Photo, Coauthor-CS, and Coauthor-Physics. Those four graphs are binary, so the continuous editor is not applied. The learning-rate grid is still recorded, and the earliest tied row is kept. The script stops if `results/EXTEND_LOCK` already exists. Outputs are `results/extend_selected.json`, `results/extend_per_target.csv`, `results/extend_summary.json`, `results/extend_run.log`, and `results/EXTEND_DECISION.md`.

Model training uses seed 0 and 200 epochs, and it keeps the weights from the last epoch. Target draws use `numpy.random.Generator`: seed 0 for the headroom sample and for the 60/20/20 extension split, seed 1 for tuning, seed 2 for the primary test and the extension targets, and seed 3 for the confirmation targets.

## Check a rerun against the published files

After the five commands, compare these files with `results_published/`:

- `headroom.json`
- `agncf_selected.json`
- `agncf_test_summary.json` and `agncf_test_per_target.csv`
- `confirm_summary.json` and `confirm_per_target.csv`
- `extend_summary.json` and `extend_per_target.csv`
- `AGNCF_DECISION.md`, `CONFIRM_DECISION.md`, and `EXTEND_DECISION.md`

The published decision files record GO for the primary test, CONFIRMED for the fresh citation targets, and NO-GO for the four-graph extension rule. Match the flip counts in those files. Timestamps inside the lock files, and the recorded runtimes, will differ.

`CHANGES.md` records three code fixes made before the locked runs: SciPy is required to read Planetoid, a GAT lookup had to read the GCN prediction dictionary, and scikit-learn is required for the isolation forest. Those fixes do not change the protocols.
