# K-means

K-means partitions observations into $k$ clusters, minimizing the sum of squared
distances from each observation $x$ to its cluster centroid $\mu_j$:

$$
\underset{C_1,\ldots,C_k}{\operatorname{minimize}}
\sum_{j=1}^{k}\sum_{x\in C_j}\lVert x-\mu_j\rVert_2^2
$$

[](){#kmeans-simple}
::: toyml.clustering.kmeans
