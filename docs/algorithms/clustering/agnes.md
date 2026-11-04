# AGNES

## Algorithm

Agglomerative nesting starts with one cluster per observation and repeatedly merges
the closest pair. Single, complete, and average linkage define cluster distance
using the minimum, maximum, and mean pairwise observation distances respectively.
The resulting hierarchy can be visualized as a dendrogram.

## Inputs and limitations

- Pass a non-empty rectangular list of numeric observations and choose
  `1 <= n_cluster <= len(samples)`.
- Distances are Euclidean. Scale features before fitting when their units differ.
- Pairwise distances require quadratic memory, so use small teaching datasets.
- Plotting requires `pip install "toyml[plot]"`; clustering itself has no runtime dependencies.

The fitted linkage matrix is available as `linkage_matrix_` and is rebuilt on each fit.

## Minimal example

```python
from toyml.clustering import AGNES

samples = [[0.0], [0.1], [4.0], [4.1]]
model = AGNES(n_cluster=2, linkage="average")
print(model.fit_predict(samples))
```

## API

::: toyml.clustering.agnes
