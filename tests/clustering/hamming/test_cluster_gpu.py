#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest

import torch

try:
    from fast_transformers.clustering.hamming import cluster_cuda
except ImportError:
    pass


class TestClusterGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No CUDA capable device detected")

    def test_long_clusters(self):
        for bits in range(1, 63):
            hashes = torch.cat([
                torch.zeros(50).long(),
                torch.ones(50).long() * (2**bits - 1)
            ]).view(1, 1, 100)[:,:,torch.randperm(100)].cuda()
            lengths = torch.full((1,), 100, dtype=torch.int32).cuda()
            centroids = torch.empty(1, 1, 2, dtype=torch.int64).cuda()
            distances = torch.empty(1, 1, 100, dtype=torch.int32).cuda()
            bitcounts = torch.empty(1, 1, 2, bits, dtype=torch.int32).cuda()
            clusters = torch.empty(1, 1, 100, dtype=torch.int32).cuda()
            counts = torch.empty(1, 1, 2, dtype=torch.int32).cuda()

            cluster_cuda.cluster(
                hashes,
                lengths,
                centroids,
                distances,
                bitcounts,
                clusters,
                counts,
                10,
                bits
            )
            self.assertEqual(
                tuple(sorted(centroids.cpu().numpy().ravel().tolist())),
                (0, 2**bits - 1)
            )
            self.assertTrue(torch.all(counts==50))

    def test_two_clusters(self):
        hashes = torch.cat([
            torch.zeros(50).long(),
            torch.full((50,), 255, dtype=torch.int64)
        ]).view(1, 1, 100)[:, :, torch.randperm(100)].cuda()
        lengths = torch.full((1,), 100, dtype=torch.int32).cuda()
        centroids = torch.empty(1, 1, 2, dtype=torch.int64).cuda()
        distances = torch.empty(1, 1, 100, dtype=torch.int32).cuda()
        bitcounts = torch.empty(1, 1, 2, 8, dtype=torch.int32).cuda()
        clusters = torch.empty(1, 1, 100, dtype=torch.int32).cuda()
        counts = torch.empty(1, 1, 2, dtype=torch.int32).cuda()

        cluster_cuda.cluster(
            hashes,
            lengths,
            centroids,
            distances,
            bitcounts,
            clusters,
            counts,
            10,
            8
        )
        self.assertEqual(
            tuple(sorted(centroids.cpu().numpy().ravel().tolist())),
            (0, 255)
        )
        self.assertTrue(torch.all(counts==50))

    def test_power_of_2_clusters(self):
        hashes = torch.cat([
            torch.full((10,), 1<<i, dtype=torch.int64)
            for i in range(8)
        ]).view(1, 1, 80)[:, :, torch.randperm(80)].cuda()
        lengths = torch.full((1,), 80, dtype=torch.int32).cuda()
        centroids = torch.empty(1, 1, 8, dtype=torch.int64).cuda()
        distances = torch.empty(1, 1, 80, dtype=torch.int32).cuda()
        bitcounts = torch.empty(1, 1, 8, 8, dtype=torch.int32).cuda()
        clusters = torch.empty(1, 1, 80, dtype=torch.int32).cuda()
        counts = torch.empty(1, 1, 8, dtype=torch.int32).cuda()

        cluster_cuda.cluster(
            hashes,
            lengths,
            centroids,
            distances,
            bitcounts,
            clusters,
            counts,
            2000,
            8
        )
        self.assertEqual(
            tuple(sorted(centroids.cpu().numpy().ravel().tolist())),
            (1, 2, 4, 8, 16, 32, 64, 128)
        )
        self.assertTrue(torch.all(counts==10))

    def test_many_sequences(self):
        hashes = torch.cat([
            torch.zeros(50).long(),
            torch.full((50,), 255, dtype=torch.int64)
        ]).view(1, 1, 100)[:, :, torch.randperm(100)].repeat(5, 3, 1).cuda()
        lengths = torch.full((5,), 100, dtype=torch.int32).cuda()
        centroids = torch.empty(5, 3, 2, dtype=torch.int64).cuda()
        distances = torch.empty(5, 3, 100, dtype=torch.int32).cuda()
        bitcounts = torch.empty(5, 3, 2, 8, dtype=torch.int32).cuda()
        clusters = torch.empty(5, 3, 100, dtype=torch.int32).cuda()
        counts = torch.empty(5, 3, 2, dtype=torch.int32).cuda()

        cluster_cuda.cluster(
            hashes,
            lengths,
            centroids,
            distances,
            bitcounts,
            clusters,
            counts,
            10,
            8
        )
        self.assertTrue(torch.all(centroids.min(-1)[0] == 0))
        self.assertTrue(torch.all(centroids.max(-1)[0] == 255))
        self.assertTrue(torch.all(counts==50))


if __name__ == "__main__":
    unittest.main()

