# Code changes

## 1. Add SciPy so Planetoid can load

`Planetoid` stores features and labels as SciPy sparse pickles. The first run crashed in `torch_geometric/io/planetoid.py` with `ModuleNotFoundError: No module named 'scipy'` before any GCN was trained. SciPy is only the file format used by the specified dataset. No model, baseline, threshold, seed, or decision rule was changed.

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,3 +1,4 @@
 numpy>=1.24
+scipy>=1.11
 torch>=2.1
 torch_geometric>=2.5
```

## 2. AGN-CF crash fixes before the first run

Two faults in `agncf.py` would have stopped the final test before a verdict. Neither changes the frozen search, the grid, the metrics, or the decision rule.

The GAT preparation received the GCN pack and read it as `gcn_pack[2]`. The pack is a dictionary, so that lookup raises `KeyError` and the headroom exclusion ids would not be taken from the GCN predictions. The lookup is `gcn_pack["pred"]`.

The per-target CSV is required to name the model, dataset, and method. Those three fields were missing from the written row. They are now filled from the evaluation loop.

## 3. Add scikit-learn for the extension detector

The extension protocol reports an isolation forest trained on real nodes. The environment had no `sklearn` module, so the detector could not be constructed. scikit-learn 1.6.1 is only that detector. No model, search, threshold, seed, or decision rule was changed.

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -2,3 +2,4 @@
 scipy>=1.11
 torch>=2.1
 torch_geometric>=2.5
+scikit-learn>=1.3
```
