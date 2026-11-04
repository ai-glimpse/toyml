# Naive Bayes

## Algorithm

Naive Bayes combines a class prior with a likelihood for each feature, assuming
features are conditionally independent given the class. ToyML provides Gaussian
likelihoods for continuous features, multinomial likelihoods for count features,
and categorical likelihoods for discrete feature values.

## Inputs and limitations

- Pass a non-empty rectangular list of features and an equally long list of integer labels.
- Gaussian training needs at least two observations per class with the default unbiased
  variance estimate. The current implementation uses uniform class priors.
- Multinomial inputs should be non-negative counts. Categorical inputs should use the same
  category encoding at training and prediction; unseen values currently contribute zero
  log-likelihood, so keep prediction categories within the training vocabulary.
- `predict`, `predict_proba`, and `predict_log_proba` each accept one sample.

## Minimal example

```python
from toyml.classification.naive_bayes import GaussianNaiveBayes

samples = [[0.0], [0.2], [4.0], [4.2]]
model = GaussianNaiveBayes().fit(samples, [0, 0, 1, 1])
print(model.predict([0.1]))
print(model.predict_proba([0.1]))
```

## API

::: toyml.classification.naive_bayes
