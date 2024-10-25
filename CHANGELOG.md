# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Add support for Python3.13


## [0.3.1] - 2024-09-19

### Changed
- Improve the one dimension classifier implementation in adaboost algorithm [#117](https://github.com/ai-glimpse/toyml/pull/117)

## [0.3.1.rc0] - 2024-09-19

### Fixed
- Fix the codecov report issue in ci [#112](https://github.com/ai-glimpse/toyml/pull/112)

### Changed
- Improve the variable names in kmeans algorithm [#114](https://github.com/ai-glimpse/toyml/pull/114)
- Improve the documentation and linters [#115](https://github.com/ai-glimpse/toyml/pull/115)

## [0.3.0] - 2024-09-12

### Docs
- Improve the documentation


## [0.3.0.rc0] - 2024-09-12

### Refactor
- Refactored knn and adaboost algorithm
- Make all the algorithms in single file (almost, except for kmeans++)

## [0.3.0.dev1] - 2024-09-06

### Refactor
- Refactored all the clustering algorithm

### Fixed
- Kmeans simple implementation fix centroid calculation
- Bisect K-means cluster with wrong dataset index

### Changed
- Package management migrates from Poetry to Uv


## [0.2.0] - 2022-12-03

### Added
- Clustering: DBSCAN, Hierarchical(Agnes&Diana), Kmeans
- Classification: KNN
- Ensemble: Boosting(AdaBoost)

### Changed
- Use Poetry to manage the package
- Use MkDocs to build documentation

## [0.1.0] - 2022-12-02
### Added
- Project clean & reorg
