# K-means

## Algorithm

K-means partitions observations into $k$ clusters, minimizing the sum of squared
distances from each observation $x$ to its cluster centroid $\mu_j$:

$$
\underset{C_1,\ldots,C_k}{\operatorname{minimize}}
\sum_{j=1}^{k}\sum_{x\in C_j}\lVert x-\mu_j\rVert_2^2
$$

Each iteration assigns observations to the nearest centroid and recomputes the
centroids from those assignments. Initialization affects which local solution is found.

## Inputs and limitations

- Pass a non-empty rectangular list of finite numeric observations and an integer `1 <= k <= len(samples)`.
- Distances are Euclidean; scale features first when their units differ.
- Choose `centroids_init_method="random"` or `"kmeans++"`.
- Duplicate locations can produce fewer non-empty clusters than `k`, with a warning log.
  All `k` centers retain valid coordinates.
- Set `random_seed` to an integer for reproducible fits.
- `predict` accepts one sample; `fit_predict` returns one label per training observation.

## Minimal example

```python
from toyml.clustering import KMeans

samples = [[0.0], [0.1], [4.0], [4.1]]
model = KMeans(k=2, random_seed=42)
print(model.fit_predict(samples))
print(model.predict([0.2]))
```

## API

[](){#kmeans-simple}
::: toyml.clustering.kmeans
