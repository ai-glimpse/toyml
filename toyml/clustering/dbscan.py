import math
import random

from collections import deque

from toyml.utils.linear_algebra import distance_matrix, euclidean_distance
from toyml.utils.types import Clusters


class DbScan:
    """
    dbscan algorithm.

    Ref:
    1. Zhou
    2. Han
    3. Kassambara
    4. Wikipedia
    """

    def __init__(self, dataset: list[list[float]], eps: float, min_pts: int = 3) -> None:
        self._dataset = dataset
        self._n = len(self._dataset)
        # distance matrix
        self._dist_matrix = distance_matrix(self._dataset)
        self._eps = eps
        # we do not include the point i itself as a neighbor
        # as the algorithm does, so we minus one here to convert
        self._min_pts = min_pts - 1
        self._core_objects: list[int] = []
        self._noises: list[int] = []
        # the number of clusters
        self._k = 0
        self._clusters: Clusters = []

    def _get_neighbors(self, i: int) -> list[int]:
        neighbors = []
        for j in range(self._n):
            if i != j and self._dist_matrix[i][j] <= self._eps:
                neighbors.append(j)
        return neighbors

    def _get_core_objects(self) -> list[int]:
        for i in range(self._n):
            neighbors = self._get_neighbors(i)
            if len(neighbors) >= self._min_pts:
                self._core_objects.append(i)
            else:
                self._noises.append(i)
        return self._core_objects

    def fit(self) -> Clusters:
        self._get_core_objects()
        # initialize the unvisited set
        unvisited = set(range(self._n))
        # core objects used for training
        random.shuffle(self._core_objects)
        core_object_set = set(self._core_objects)
        while core_object_set:
            unvisited_old = unvisited.copy()
            o = core_object_set.pop()
            queue: deque = deque()
            queue.append(o)
            unvisited.remove(o)
            while len(queue) > 0:
                q = queue.popleft()
                neighbors = self._get_neighbors(q)
                if len(neighbors) >= self._min_pts:
                    delta = set(neighbors) & unvisited
                    # remove from unvisited
                    for point in delta:
                        queue.append(point)
                        unvisited.remove(point)
            self._k += 1
            cluster = unvisited_old.difference(unvisited)
            self._clusters.append(list(cluster))
            core_object_set = core_object_set.difference(cluster)
        return self._clusters

    def predict(self, point: list[float]) -> int:
        min_dist = math.inf
        best_label = -1
        for i in range(self._k):
            cluster_vectors = [self._dataset[j] for j in self._clusters[i]]
            dist = sum(euclidean_distance(point, p) for p in cluster_vectors)
            if dist < min_dist:
                min_dist = dist
                best_label = i
        print(f"The label of {point} is {best_label}")
        return best_label

    def print_cluster(self) -> None:
        for i in range(self._k):
            # the ith clusters
            cluster_i = self._clusters[i]
            print(f"Cluster[{i}]: {[self._dataset[j] for j in cluster_i]}")
        print(f"Noise Data: {[self._dataset[j] for j in self._noises]}")

    def print_label(self) -> None:
        # -1 for noise data
        labels = [-1] * self._n
        for i in range(self._k):
            for sample_index in self._clusters[i]:
                labels[sample_index] = i
        # we leave the label of noise data to None
        print("Sample labels: ", labels)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DbScan(dataset, 3, 2)
    print(dbscan.fit())
    dbscan.print_cluster()
    dbscan.print_label()
    dbscan.predict([0.0, 0.4])
