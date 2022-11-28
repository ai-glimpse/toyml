import math
import random

from collections import deque

from toyml.utils.linear_algebra import distance_matrix, euclidean_distance
from toyml.utils.types import Clusters, DataSet, List, Vector

"""
TODO:
1. Plot
2. Test
"""


class DbScan:
    """
    dbscan algorithm.

    Ref:
    1. Zhou
    2. Han
    3. Kassambara
    4. Wikipedia
    """

    def __init__(self, dataset: DataSet, eps: float, MinPts: int = 3) -> None:
        self._dataset = dataset
        self._n = len(self._dataset)
        # distance matrix
        self._dist_matrix = distance_matrix(self._dataset)
        self._eps = eps
        # we does not include the point i itself as a neighbor
        # as the algorithm does, so we minus one here to convert
        self._MinPts = MinPts - 1
        self._coreObjects = []
        self._noises = []
        # the number of clusters
        self._k = 0
        self._clusters = []

    def _getNeighbors(self, i: int) -> List[int]:
        neighbors = []
        for j in range(self._n):
            if i != j and self._dist_matrix[i][j] <= self._eps:
                neighbors.append(j)
        return neighbors

    def _getCoreObjects(self) -> List[int]:
        for i in range(self._n):
            neighbors = self._getNeighbors(i)
            if len(neighbors) >= self._MinPts:
                self._coreObjects.append(i)
            else:
                self._noises.append(i)
        return self._coreObjects

    def fit(self) -> Clusters:
        # initialize the unvisit set
        F = set(range(self._n))
        # core objects used for training
        random.shuffle(self._coreObjects)
        Omg = set(self._coreObjects)
        while Omg:
            F_old = F.copy()
            o = Omg.pop()
            Q = deque()
            Q.append(o)
            F.remove(o)
            while len(Q) > 0:
                q = Q.popleft()
                neighbors = self._getNeighbors(q)
                if len(neighbors) >= self._MinPts:
                    delta = set(neighbors) & F
                    # remove from unvisit
                    for point in delta:
                        Q.append(point)
                        F.remove(point)
            self._k += 1
            C_k = F_old.difference(F)
            self._clusters.append(C_k)
            Omg = Omg.difference(C_k)
        return self._clusters

    def predict(self, point: Vector) -> int:
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
    dataset = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DbScan(dataset, 3, 2)
    print(dbscan._getCoreObjects())
    print(dbscan.fit())
    dbscan.print_cluster()
    dbscan.print_label()
    dbscan.predict([0.0, 0.4])
