# KNN

## Algorithm

K-nearest neighbors classifies a sample using nearby labeled training observations.
Feature scale affects Euclidean distance, so ToyML offers optional standardization.
Training stores the observations; prediction computes their distances to the query
and returns the most frequent label among the `k` closest observations.

## Inputs and limitations

- Each `fit` validates `k`, `std_transform`, and `distance_metric` together with the training data.
  After changing any of them, fit the same classifier again before calling `predict`.
  The keyword-only `distance_metric` currently supports `"euclidean"` (the default).
- Fit on at least `k` samples. Each row must have the same positive number of finite numeric
  features and one hashable label. A successful refit replaces the previous training data.
  A failed refit preserves the previously fitted attributes.
- With the default `std_transform=True`, training and prediction use the training set's
  feature means and sample standard deviations. Constant features use a scale of one.
  Select `False` to work in the original feature units.
- Pass one feature vector to `predict`. Equidistant samples retain training order.
  If labels tie for the most votes, the first tied label in nearest-neighbor order wins.

## Minimal example

```python
from toyml.classification import KNN

samples = [[0.0], [0.2], [4.0], [4.2]]
model = KNN(k=2, std_transform=False).fit(samples, ["near", "near", "far", "far"])
print(model.predict([0.1]))
```

## Numerical stability

The standardizer stores a mean and scale per feature. If subtracting the mean would
overflow, it divides both operands by the scale before subtracting. A standardized
value outside the floating-point range raises `ValueError`. The standardizer's
`fit_transform` saves the fitted statistics before transforming the training rows.

Euclidean distance uses `math.hypot` so that squaring very large or very small coordinate
differences does not distort the ranking. A distance beyond the floating-point range
ranks as infinity. Samples with such distances follow the same tie rule.

## API

::: toyml.classification.knn
