# Isolation Forest

## Algorithm

Isolation Forest builds trees by randomly selecting a feature and a split value.
Observations isolated by fewer splits receive higher anomaly scores. Averaging path
lengths across trees reduces the influence of a single random partition.

## Inputs and limitations

- Pass equally sized numeric feature lists with at least two observations and some variation.
- Choose `n_itree >= 1` and, when supplied, `2 <= max_samples <= len(samples)`.
- `score` and `predict` accept one sample. Higher scores are more anomalous;
  `predict` returns `-1` for outliers and `1` for normal observations.
- `score_threshold` is a score cutoff, not an expected fraction of outliers.
- Set `random_seed` to an integer for reproducible fits.

## Minimal example

```python
from toyml.ensemble import IsolationForest

samples = [[-0.2], [0.0], [0.1], [0.3], [8.0]]
model = IsolationForest(n_itree=20, random_seed=42).fit(samples)
print(model.score([8.0]))
print(model.predict([8.0]))
```

## API

::: toyml.ensemble.isolation_forest
    options:
        members:
        - IsolationForest
        - IsolationTree
