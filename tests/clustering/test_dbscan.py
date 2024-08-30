import math
import unittest

from toyml.clustering.dbscan import DBSCAN, Dataset


class TestDbScan(unittest.TestCase):
    def setUp(self):
        self.dataset = [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]]

    def test_dbscan_clustering(self):
        dbscan = DBSCAN(eps=3, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters_

        # Check the number of clusters
        self.assertEqual(len(clusters), 2)

        # Check the content of clusters
        self.assertIn([0, 1, 2], clusters)
        self.assertIn([3, 4], clusters)

        # Check that the last point is not in any cluster (noise)
        self.assertTrue(all(5 not in cluster for cluster in clusters))

    def test_dbscan_edge_cases(self):
        # Test with all points as noise
        dbscan = DBSCAN(eps=0.1, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters_
        self.assertEqual(len(clusters), 0)

        # Test with all points in one cluster
        dbscan = DBSCAN(eps=100, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters_
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), len(self.dataset))

    def test_dataset(self):
        dataset = Dataset(self.dataset)

        # Test dataset initialization
        self.assertEqual(dataset.n, 6)
        self.assertEqual(len(dataset.distance_matrix_), 6)
        self.assertEqual(len(dataset.distance_matrix_[0]), 6)

        # Test distance matrix calculation
        expected_distance = math.sqrt(2)  # Distance between [1.0, 2.0] and [2.0, 3.0]
        self.assertAlmostEqual(dataset.distance_matrix_[0][2], expected_distance)

        # Test get_neighbors method
        neighbors = dataset.get_neighbors(0, 2.0)
        # The neighbors don't include the point itself
        self.assertEqual(set(neighbors), {1, 2})

        # Test get_core_objects method
        core_objects, noises = dataset.get_core_objects(3, 2)
        self.assertEqual(core_objects, {0, 1, 2, 3, 4})
        self.assertEqual(set(noises), {5})


if __name__ == "__main__":
    unittest.main()
