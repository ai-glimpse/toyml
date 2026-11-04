"""Fit two clusters and classify a new observation using the public API.

Run from the repository root: uv run --locked python examples/clustering/kmeans.py
"""

from toyml.clustering import KMeans


def main() -> None:
    samples = [[1.0, 1.0], [1.0, 2.0], [5.0, 3.0], [6.0, 4.0]]
    model = KMeans(k=2, random_seed=42).fit(samples)
    print(f"Training labels: {model.labels_}")
    print(f"Centroids: {model.centroids_}")
    print(f"Prediction for [1.5, 1.5]: {model.predict([1.5, 1.5])}")


if __name__ == "__main__":
    main()
