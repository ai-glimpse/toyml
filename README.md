# ToyML: Machine Learning from Scratch

<p align="center">
  <img src="https://raw.githubusercontent.com/ai-glimpse/toyml/master/docs/assets/cover.png" alt="ToyML Cover" width="200" height="200">
</p>

<!-- --8<-- [start:overview] -->

[![Python](https://img.shields.io/pypi/pyversions/toyml.svg?color=%2334D058)](https://pypi.org/project/toyml/)
[![PyPI](https://img.shields.io/pypi/v/toyml?color=%2334D058&label=pypi%20package)](https://pypi.org/project/toyml/)
![Codecov](https://codecov.io/gh/ai-glimpse/toyml/branch/master/graph/badge.svg)
[![Documentation](https://github.com/ai-glimpse/toyml/actions/workflows/docs.yml/badge.svg)](https://github.com/ai-glimpse/toyml/actions/workflows/docs.yml)
[![Python CI](https://github.com/ai-glimpse/toyml/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-glimpse/toyml/actions/workflows/ci.yml)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/types-ty-blue)](https://docs.astral.sh/ty/)
[![prek](https://img.shields.io/badge/hooks-prek-blue)](https://prek.j178.dev/)
[![Zensical](https://img.shields.io/badge/docs-Zensical-blue)](https://zensical.org/)

There are machine learning algorithms implemented from scratch.
Let's learn machine learning with simple toy code.


## Models

- [x] Clustering: K-means
- [x] Clustering: DBSCAN
- [x] Clustering: Hierarchical (Bisecting K-means & AGNES)
- [x] Classification: KNN
- [x] Classification: Naive Bayes
- [x] Ensemble: AdaBoost
- [x] Ensemble: Isolation Forest


## Installation

Requires Python 3.11 or later. The core package has no runtime dependencies.

```bash
pip install toyml
```

For dendrograms and cluster-tree plots:

```bash
pip install "toyml[plot]"
```

## Development

Use uv for the locked environment and just for development commands:

```bash
just setup
just hooks
just check
```

See the [contributing guide](https://github.com/ai-glimpse/toyml/blob/master/CONTRIBUTING.md)
for individual lint, test, build, and Zensical preview commands.


## Release notes

See [GitHub Releases](https://github.com/ai-glimpse/toyml/releases) for version history and release notes.

<!-- --8<-- [end:overview] -->
