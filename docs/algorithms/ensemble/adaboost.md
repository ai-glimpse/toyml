# AdaBoost

## Algorithm

AdaBoost trains weak classifiers in sequence, increasing the weight of observations
that previous classifiers misclassified. Predictions combine the classifiers' votes,
weighted by their training errors. ToyML includes a one-dimensional threshold classifier
as a simple weak learner.

## Inputs and limitations

- Pass non-empty numeric feature lists and one binary label (`-1` or `1`) per observation.
- `OneDimensionClassifier` uses only the first feature; use one-dimensional samples for this learner.
- Choose a positive `n_weak_learner`. Custom learners must implement the `BaseWeakLearner` interface.
- A weak learner with zero weighted training error is retained with a finite vote weight
  of `1.0`, and training stops successfully.
- A weak learner with weighted training error at least `0.5` is discarded and training
  stops without a warning. If no weak learners have been retained, `fit` raises `ValueError`.
- `predict` accepts one sample.

## Minimal example

```python
from toyml.ensemble import AdaBoost
from toyml.ensemble.adaboost import OneDimensionClassifier

samples = [[float(i)] for i in range(10)]
labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
model = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(samples, labels)
print(model.predict([2.0]))
```

## API

::: toyml.ensemble.adaboost
