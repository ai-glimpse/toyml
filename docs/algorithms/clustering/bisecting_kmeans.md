# Bisecting K-means

## Algorithm

Bisecting K-means starts with all observations in one cluster, then uses two-way
K-means splits to reduce total within-cluster squared error. The splits form a binary
cluster tree.

## Inputs and limitations

- Pass a non-empty rectangular list of numeric observations and `1 <= k <= len(samples)`.
- The implementation uses Euclidean distance and random K-means initialization.
- Splitting stops when no candidate reduces the error, so fewer than `k` clusters may result.
- `predict` accepts a list of samples.
- Tree plotting needs the optional `toyml[plot]` dependencies.

## Minimal example

```python
from toyml.clustering import BisectingKMeans

samples = [[0.0], [0.1], [4.0], [4.1]]
model = BisectingKMeans(k=2)
print(model.fit_predict(samples))
print(model.cluster_tree_.get_clusters())
```

## API

[](){#kmeans-bisect}
::: toyml.clustering.bisecting_kmeans
    options:
        members:
        - BisectingKMeans
