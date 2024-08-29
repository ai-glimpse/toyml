import unittest

from toyml.clustering.dbscan import DBSCAN


class TestDbScan(unittest.TestCase):
    def setUp(self):
        self.dataset = [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]]

    def test_dbscan_clustering(self):
        dbscan = DBSCAN(eps=3, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters

        # Check the number of clusters
        self.assertEqual(len(clusters), 2)

        # Check the content of clusters
        self.assertIn([0, 1, 2], clusters)
        self.assertIn([3, 4], clusters)

        # Check that the last point is not in any cluster (noise)
        self.assertTrue(all(5 not in cluster for cluster in clusters))

    def test_dbscan_predict(self):
        dbscan = DBSCAN(eps=3, min_samples=2).fit(self.dataset)

        # Test prediction for a point close to the first cluster
        label = dbscan.predict([1.5, 2.5], dataset=self.dataset)
        self.assertEqual(label, 0)

        # Test prediction for a point close to the second cluster
        label = dbscan.predict([7.5, 7.5], dataset=self.dataset)
        self.assertEqual(label, 1)

        # Test prediction for a point far from all clusters
        label = dbscan.predict([50.0, 50.0], dataset=self.dataset)
        self.assertNotEqual(label, -1)  # It should assign to the closest cluster, not noise

    def test_dbscan_edge_cases(self):
        # Test with all points as noise
        dbscan = DBSCAN(eps=0.1, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters
        self.assertEqual(len(clusters), 0)

        # Test with all points in one cluster
        dbscan = DBSCAN(eps=100, min_samples=2).fit(self.dataset)
        clusters = dbscan.clusters
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), len(self.dataset))


if __name__ == "__main__":
    unittest.main()
