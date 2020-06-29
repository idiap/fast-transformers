#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import unittest

import torch

from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask


class TestMasking(unittest.TestCase):
    def test_full_mask(self):
        m = FullMask(N=10)
        self.assertEqual(m.shape, (10, 10))
        self.assertTrue(torch.all(m.bool_matrix))
        self.assertTrue(torch.all(m.float_matrix == 1))
        self.assertTrue(torch.all(m.additive_matrix == 0))

        with self.assertRaises(ValueError):
            m = FullMask(torch.rand(10))

        m = FullMask(torch.rand(10, 5) > 0.5)
        self.assertEqual(m.shape, (10, 5))

    def test_lengths(self):
        m = LengthMask(torch.tensor([1, 2, 3]))
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(torch.all(
            m.float_matrix.sum(axis=1) == torch.tensor([1, 2, 3.])
        ))
        self.assertTrue(torch.all(
            m.lengths == torch.tensor([1, 2, 3])
        ))
        for i, n in enumerate(m.lengths):
            self.assertTrue(torch.all(torch.isinf(m.additive_matrix[i, n:])))

    def test_max_lengths(self):
        m = LengthMask(torch.tensor([1, 2, 3]), max_len=10)
        self.assertEqual(m.shape, (3, 10))
        self.assertTrue(torch.all(
            m.float_matrix.sum(axis=1) == torch.tensor([1, 2, 3.])
        ))
        self.assertTrue(torch.all(
            m.lengths == torch.tensor([1, 2, 3])
        ))
        for i, n in enumerate(m.lengths):
            self.assertTrue(torch.all(torch.isinf(m.additive_matrix[i, n:])))

    def test_casting_to_lengths(self):
        m = FullMask(torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]) > 0)
        self.assertEqual(m.shape, (3, 3))
        self.assertTrue(torch.all(m.lengths == torch.tensor([1, 2, 3])))

        m = FullMask(torch.tensor([
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]) > 0)
        with self.assertRaises(ValueError):
            m.lengths

    def test_full_mask_constructor_arguments(self):
        m = FullMask(torch.rand(10, 10) > 0.5)
        self.assertEqual(m.shape, (10, 10))
        self.assertFalse(m.all_ones)

        m = FullMask(10)
        self.assertEqual(m.shape, (10, 10))
        self.assertTrue(m.all_ones)

        m = FullMask(10, 5)
        self.assertEqual(m.shape, (10, 5))
        self.assertTrue(m.all_ones)

    def test_lower_triangular(self):
        m = TriangularCausalMask(3)
        self.assertTrue(m.lower_triangular)
        self.assertTrue(torch.all(m.bool_matrix == (torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]) > 0)))

        m = FullMask(torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ]) > 0)
        self.assertTrue(m.lower_triangular)

        m = FullMask(torch.tensor([
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]) > 0)
        self.assertFalse(m.lower_triangular)

        m = LengthMask(torch.tensor([1, 1, 3]))
        self.assertFalse(m.lower_triangular)
        m = LengthMask(torch.tensor([1, 2, 3]))
        self.assertTrue(m.lower_triangular)
        m = LengthMask(torch.tensor([1, 2, 3]), max_len=4)
        self.assertTrue(m.lower_triangular)


if __name__ == "__main__":
    unittest.main()
