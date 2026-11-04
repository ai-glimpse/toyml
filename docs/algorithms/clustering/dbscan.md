# DBSCAN

## Algorithm

DBSCAN groups observations connected through dense neighborhoods. `eps` sets the
neighborhood radius and `min_samples` sets the density needed for a core point,
including the point itself. It can discover non-spherical clusters without choosing
the number of clusters in advance; isolated observations are noise.

## Inputs and limitations

- Pass non-empty, equally sized numeric feature lists, `eps > 0`, and `min_samples >= 1`.
- Distances are Euclidean; feature scale strongly affects the radius.
- This implementation stores all pairwise distances and is intended for small datasets.
- Noise samples receive label `-1`.
- `noises_` can contain non-core border points that belong to a cluster.

## Minimal example

```python
from toyml.clustering import DBSCAN

samples = [[0.0], [0.1], [4.0], [4.1]]
model = DBSCAN(eps=0.2, min_samples=2)
print(model.fit_predict(samples))
```

## API

::: toyml.clustering.dbscan
