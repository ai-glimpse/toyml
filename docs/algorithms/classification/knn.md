# KNN

## Algorithm

K-nearest neighbors classifies a sample using nearby labeled training observations.
Feature scale affects Euclidean distance, so ToyML offers optional standardization.
Training stores the observations; prediction computes their distances to the query
and returns the most frequent label among the `k` closest observations.

## Inputs and limitations

- Pass a non-empty list of equally sized numeric feature lists and one label per observation.
- Use a positive `k` no greater than the training sample count. `predict` accepts one feature list.
- Labels should be immutable, hashable values.
- Select `std_transform=False` for already scaled data.

## Minimal example

```python
from toyml.classification import KNN

samples = [[0.0], [0.2], [4.0], [4.2]]
model = KNN(k=2, std_transform=False).fit(samples, ["near", "near", "far", "far"])
print(model.predict([0.1]))
```

## API

::: toyml.classification.knn
