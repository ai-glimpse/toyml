from typing import List, Union

"""General"""
Number = Union[float, int]

"""DataSet"""
# float vectors
Vector = List[float]
Vectors = List[Vector]

# dataset
DataSet = List[Vector]

# labels
Label = int
Labels = List[Label]

"""clustering"""
# cluster(we store sample index in cluster)
Cluster = List[int]
# clusters
Clusters = List[Cluster]

# distance matrix
DistMat = List[List[float]]

"""classification"""
# classifier with sample weight considered
Weights = List[Number]

"""Special"""

# Sometimes, we use Any to fight for the cases where pyright
# not works.
