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

    eps: float
    min_pts: int
    """The minimum number of points in a cluster to be considered a core object.
    (which don't include the core object itself)"""
    k: int = 0
    clusters: Clusters = field(default_factory=list)
    core_objects: list[int] = field(default_factory=list)
    noises: list[int] = field(default_factory=list)
    _dist_matrix: list[list[float]] = field(default_factory=list)
    _n: int = 0

    def _get_neighbors(self, i: int) -> list[int]:
        neighbors = []
        for j in range(self._n):
            if i != j and self._dist_matrix[i][j] <= self.eps:
                neighbors.append(j)
        return neighbors

    def _get_core_objects(self) -> list[int]:
        for i in range(self._n):
            neighbors = self._get_neighbors(i)
            if len(neighbors) >= self.min_pts:
                self.core_objects.append(i)
            else:
                self.noises.append(i)
        return self.core_objects

    def fit(self, dataset: list[list[float]]) -> Clusters:
        self._n = len(dataset)
        self._dist_matrix = distance_matrix(dataset)

        # get core objects
        self._get_core_objects()
        # initialize the unvisited set
        unvisited = set(range(self._n))
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
                if len(neighbors) >= self.min_pts:
                    delta = set(neighbors) & unvisited
                    # remove from unvisited
                    for point in delta:
                        queue.append(point)
                        unvisited.remove(point)
            self.k += 1
            cluster = unvisited_old.difference(unvisited)
            self.clusters.append(list(cluster))
            core_object_set = core_object_set.difference(cluster)
        return self.clusters

    def predict(self, point: list[float], dataset: list[list[float]]) -> int:
        min_dist = math.inf
        best_label = -1
        for i in range(self.k):
            cluster_vectors = [dataset[j] for j in self.clusters[i]]
            dist = sum(euclidean_distance(point, p) for p in cluster_vectors)
            if dist < min_dist:
                min_dist = dist
                best_label = i
        print(f"The label of {point} is {best_label}")
        return best_label

    def print_cluster(self, dataset: list[list[float]]) -> None:
        for i in range(self.k):
            # the ith clusters
            cluster_i = self.clusters[i]
            print(f"Cluster[{i}]: {[dataset[j] for j in cluster_i]}")
        print(f"Noise Data: {[dataset[j] for j in self.noises]}")

    def print_label(self) -> None:
        # -1 for noise data
        labels = [-1] * self._n
        for i in range(self.k):
            for sample_index in self.clusters[i]:
                labels[sample_index] = i
        # we leave the label of noise data to None
        print("Sample labels: ", labels)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DbScan(eps=3, min_pts=1)
    print(dbscan.fit(dataset))
    dbscan.print_cluster(dataset)
    dbscan.print_label()
    dbscan.predict([0.0, 0.4], dataset)
