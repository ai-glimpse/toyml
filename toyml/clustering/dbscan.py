import math
import random

from collections import deque
from dataclasses import dataclass, field

from toyml.utils.linear_algebra import distance_matrix, euclidean_distance
from toyml.utils.types import Clusters


@dataclass
class DbScan:
    """
    dbscan algorithm.

    Ref:
    1. Zhou
    2. Han
    3. Kassambara
    4. Wikipedia
    """

    eps: float = 0.5
    """The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    This is not a maximum bound on the distances of points within a cluster.
    This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    (same as sklearn)
    """
    min_samples: int = 5
    """The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    This includes the point itself. If min_samples is set to a higher value,
    DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.
    (same as sklearn)
    """
    clusters: Clusters = field(default_factory=list)
    core_objects: list[int] = field(default_factory=list)
    noises: list[int] = field(default_factory=list)
    distance_matrix_: list[list[float]] = field(default_factory=list)
    k_: int = 0
    n_: int = 0

    def _get_neighbors(self, i: int) -> list[int]:
        neighbors = []
        for j in range(self.n_):
            if i != j and self.distance_matrix_[i][j] <= self.eps:
                neighbors.append(j)
        return neighbors

    def _get_core_objects(self) -> list[int]:
        for i in range(self.n_):
            if self._is_core_object(i):
                self.core_objects.append(i)
            else:
                self.noises.append(i)
        return self.core_objects

    def fit(self, dataset: list[list[float]]) -> Clusters:
        self.n_ = len(dataset)
        self.distance_matrix_ = distance_matrix(dataset)

        # get core objects
        self._get_core_objects()
        # initialize the unvisited set
        unvisited = set(range(self.n_))
        # core objects used for training
        random.shuffle(self.core_objects)
        core_object_set = set(self.core_objects)
        while core_object_set:
            unvisited_old = unvisited.copy()
            o = core_object_set.pop()
            queue: deque = deque()
            queue.append(o)
            unvisited.remove(o)
            while len(queue) > 0:
                q = queue.popleft()
                neighbors = self._get_neighbors(q)
                if self._is_core_object(q, neighbors):
                    delta = set(neighbors) & unvisited
                    # remove from unvisited
                    for point in delta:
                        queue.append(point)
                        unvisited.remove(point)
            self.k_ += 1
            cluster = unvisited_old.difference(unvisited)
            self.clusters.append(list(cluster))
            core_object_set = core_object_set.difference(cluster)
        return self.clusters

    def _is_core_object(self, i: int, neighbors: None | list[int] = None) -> bool:
        if neighbors is None:
            neighbors = self._get_neighbors(i)
        # `+ 1` here to include the point itself
        if len(neighbors) + 1 >= self.min_samples:
            return True
        return False

    def predict(self, point: list[float], dataset: list[list[float]]) -> int:
        min_dist = math.inf
        best_label = -1
        for i in range(self.k_):
            cluster_vectors = [dataset[j] for j in self.clusters[i]]
            dist = sum(euclidean_distance(point, p) for p in cluster_vectors)
            if dist < min_dist:
                min_dist = dist
                best_label = i
        print(f"The label of {point} is {best_label}")
        return best_label

    def print_cluster(self, dataset: list[list[float]]) -> None:
        for i in range(self.k_):
            # the ith clusters
            cluster_i = self.clusters[i]
            print(f"Cluster[{i}]: {[dataset[j] for j in cluster_i]}")
        print(f"Noise Data: {[dataset[j] for j in self.noises]}")

    def print_label(self) -> None:
        # -1 for noise data
        labels = [-1] * self.n_
        for i in range(self.k_):
            for sample_index in self.clusters[i]:
                labels[sample_index] = i
        # we leave the label of noise data to None
        print("Sample labels: ", labels)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DbScan(eps=3, min_samples=2)
    print(dbscan.fit(dataset))
    dbscan.print_cluster(dataset)
    dbscan.print_label()
    dbscan.predict([0.0, 0.4], dataset)
