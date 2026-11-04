# KNN

## Algorithm

K-nearest neighbors classifies a sample using nearby labeled training observations.
Feature scale affects Euclidean distance, so ToyML offers optional standardization.
Training stores the observations; prediction computes their distances to the query.

## Inputs and limitations

- Pass a non-empty list of equally sized numeric feature lists and one label per observation.
- Use a positive `k` no greater than the training sample count. `predict` accepts one feature list.
- Standardization modifies the supplied lists; pass copies
  when the original data must be preserved, or select `std_transform=False` for already scaled data.
- The current neighbor selection uses a stride through sorted distances, so results can differ
  from standard nearest-`k` voting. Treat this as a teaching implementation.

## Minimal example

```python
from toyml.classification import KNN

samples = [[0.0], [0.2], [4.0], [4.2]]
model = KNN(k=2, std_transform=False).fit(samples, ["near", "near", "far", "far"])
print(model.predict([0.1]))
```

## API

::: toyml.classification.knn
