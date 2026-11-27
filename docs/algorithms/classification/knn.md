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
- Pass one feature vector to `predict`. Samples whose computed distances round to the same
  float retain training order. For example, with standardization disabled, `[1, 2**-27]`
  and `[1, 0]` both have a computed distance of `1.0` from the origin. With `k=1`,
  the first training row gets the vote.
  If labels tie for the most votes, the first tied label in nearest-neighbor order wins.

## Minimal example

```python
from toyml.classification import KNN

samples = [[0.0], [0.2], [4.0], [4.2]]
model = KNN(k=2, std_transform=False).fit(samples, ["near", "near", "far", "far"])
print(model.predict([0.1]))
```

## Numerical stability

If a varying feature's sample standard deviation rounds to zero or is subnormal,
the standardizer divides that column by the smallest positive normal float
(`2**-1022`) and computes the mean and standard deviation in that unit. It uses the
same unit for later queries. For the training column `[0, 0, 0, 0, 5e-324]`,
the four zeros have a standard score of about `-0.4472` and the last value scores
`1.7889`. A new value of `1e-323` scores about `4.0249`.

If subtracting the mean would overflow, the standardizer divides both operands
by the scale before subtracting. A standardized
value outside the floating-point range raises `ValueError`. The standardizer's
`fit_transform` saves the fitted statistics before transforming the training rows.
Large integers can lose precision during centering. For a training column containing
`2**53` and `2**53 + 1`, both samples receive a standardized value of `0.0`.

Euclidean distance uses `math.hypot` so that squaring very large or very small coordinate
differences does not distort the ranking. A distance beyond the floating-point range
ranks as infinity. Samples with such distances follow the same tie rule.

## API

::: toyml.classification.knn
