# Pre-registration — AGN-CF headroom check

Written before any headroom run. Baselines only. AGN-CF itself is not designed and is not run.

## Protocol

Data: Cora, CiteSeer, and PubMed from PyTorch Geometric `Planetoid`, public split. One 2-layer GCN per dataset, seed 0, standard PyTorch Geometric GCN settings:

- `GCNConv` layers, hidden size 16, ReLU between them, dropout 0.5 on the hidden representation during training only;
- `GCNConv` defaults, including self-loops and symmetric normalisation inside the layer;
- Adam, learning rate 0.01, weight decay 5e-4, full-batch cross-entropy on the training nodes, 200 epochs;
- the parameters after epoch 200 are used, with no reload of a best-validation checkpoint;
- test accuracy is computed in eval mode on the public test mask.

Targets: 200 test nodes that this GCN classifies correctly, drawn without replacement by `numpy.random.Generator(0)`, then handled in increasing node-index order. If a dataset has fewer than 200 correct test nodes, the run stops. The desired class of a target is the class with the second-highest logit on the original graph.

Each baseline inserts exactly one new node into a fresh copy of the original graph and runs the trained GCN once in eval mode, without retraining. A flip occurs only if the target's predicted class becomes the desired class. A change to any other class is not a flip.

- B1 Copy-prototype: copy the features of the training node of the desired class whose softmax probability on that class is highest. Ties break to the smaller node index. Connect the new node only to the target.
- B2 Copy-prototype-plus: same node as B1. Connect it to the target, and also to that prototype's neighbours, excluding the target, capped at `floor(median degree)` neighbours. Neighbours are ordered by higher original degree, then smaller node index. The median is the median of the original undirected degrees.
- B3 Class-mean: the mean feature vector of the training nodes of the desired class. If every feature value in the dataset is 0 or 1, binarise the mean at 0.5 (values at least 0.5 become 1). Otherwise keep the mean. Connect the new node only to the target.
- B4 Random real node: features of a training node drawn uniformly with replacement by a separate `numpy.random.Generator(0)`, one draw per target in increasing node-index order. Connect it only to the target.

Degree is the number of distinct neighbours in the input graph, before the self-loops that `GCNConv` adds. Edges used are that degree of the inserted node.

Plausibility, reported per baseline and dataset: the mean Euclidean distance from the inserted feature vector to the nearest original node, and the fraction of insertions whose degree lies between the minimum and maximum original degree, inclusive.

## Pre-registered rule

- **NO-GO** (no room for AGN-CF) if B1 or B2 reaches a flip rate ≥ 90% on all three datasets.
- **GO** (worth designing AGN-CF) if the best simple flip rate is < 60% on at least two datasets.
- Otherwise **UNCLEAR**: report and let the user decide.

This rule is not to be changed after the results are seen.
