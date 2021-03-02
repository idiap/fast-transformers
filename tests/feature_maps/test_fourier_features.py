#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import unittest

import torch

from fast_transformers.attention import AttentionLayer, LinearAttention
from fast_transformers.feature_maps.fourier_features import \
    RandomFourierFeatures, SmoothedRandomFourierFeatures, Favor, \
    GeneralizedRandomFeatures
from fast_transformers.masking import FullMask


class TestFourierFeatures(unittest.TestCase):
    def test_omega(self):
        f = RandomFourierFeatures(32, n_dims=64, orthogonal=True)
        f.new_feature_map("cpu")
        self.assertLess(
            torch.abs(
                f.omega.t().matmul(f.omega)[torch.eye(32) == 0]
            ).max().item(),
            1e-4
        )

    def test_rff(self):
        for ortho in [False, True]:
            f = RandomFourierFeatures(32, n_dims=32*1000, softmax_temp=1,
                                      orthogonal=ortho)
            f.new_feature_map("cpu")

            x = torch.randn(100, 32) * 0.15
            y = torch.randn(100, 32) * 0.15
            phi_x = f(x)
            phi_y = f(y)

            rbf_xy = torch.exp(-((x[:, None] - y[None, :])**2).sum(-1)/2)
            rbf_xy_hat = phi_x.matmul(phi_y.t())

            self.assertLess(
                ((rbf_xy - rbf_xy_hat)**2).mean().item(),
                1e-4
            )

            f = SmoothedRandomFourierFeatures(32, n_dims=32*1000,
                                              softmax_temp=1, orthogonal=ortho,
                                              smoothing=1.0)
            f.new_feature_map("cpu")
            phi_x = f(x)
            phi_y = f(y)
            rbf_xy = torch.exp(-((x[:, None] - y[None, :])**2).sum(-1)/2) + 1
            rbf_xy_hat = phi_x.matmul(phi_y.t())

            self.assertLess(
                ((rbf_xy - rbf_xy_hat)**2).mean().item(),
                1e-4
            )

    def test_prf(self):
        for ortho in [False, True]:
            f = Favor(32, n_dims=32*1000, softmax_temp=1, orthogonal=ortho)

            f.new_feature_map("cpu")

            x = torch.randn(100, 32) * 0.15
            y = torch.randn(100, 32) * 0.15
            phi_x = f(x)
            phi_y = f(y)

            sm_xy = torch.exp(x.mm(y.t()))
            sm_xy_hat = phi_x.mm(phi_y.t())

            self.assertLess(
                ((sm_xy - sm_xy_hat)**2).mean().item(),
                1e-3
            )

    def test_grf(self):
        f = GeneralizedRandomFeatures(32, n_dims=128)
        f.new_feature_map("cpu")
        x = torch.randn(100, 32)
        phi_x = f(x)
        self.assertEqual((100, 128), phi_x.shape)

    def test_feature_map_sharing(self):
        x = torch.rand(3, 100, 4*32)
        f = Favor.factory(n_dims=64)
        att = AttentionLayer(
            LinearAttention(32, f),
            4*32,
            4
        )

        attn_mask = FullMask(100)
        lengths = FullMask(3, 100)
        y = att(x, x, x, attn_mask, lengths, lengths)
        y = att(y, y, y, attn_mask, lengths, lengths)
        y.sum().backward()

    def test_feature_map_redraw(self):
        x = torch.rand(3, 100, 32)
        f = Favor(32, n_dims=64)

        f.new_feature_map("cpu")
        fx1 = f(x)
        f.new_feature_map("cpu")
        fx2 = f(x)
        self.assertFalse(torch.allclose(fx1, fx2))

        f = Favor(32, n_dims=64, redraw=2)
        f.new_feature_map("cpu")
        fx1 = f(x)
        f.new_feature_map("cpu")
        fx2 = f(x)
        f.new_feature_map("cpu")
        fx3 = f(x)
        self.assertTrue(torch.allclose(fx1, fx2))
        self.assertFalse(torch.allclose(fx2, fx3))

        f = Favor(32, n_dims=64, redraw=1, deterministic_eval=True)
        f.new_feature_map("cpu")
        fx1 = f(x)
        f.new_feature_map("cpu")
        fx2 = f(x)
        f.eval()
        f.new_feature_map("cpu")
        fx3 = f(x)
        self.assertFalse(torch.allclose(fx1, fx2))
        self.assertTrue(torch.allclose(fx2, fx3))


if __name__ == "__main__":
    unittest.main()
