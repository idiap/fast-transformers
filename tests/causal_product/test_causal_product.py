#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import unittest

import numpy as np
import torch

from fast_transformers.causal_product import causal_dot_product


class TestCausalProduct(unittest.TestCase):
    def _zero_grad(self, *tensors):
        for t in tensors:
            if t.grad is not None:
                t.grad[...] = 0

    def _test_api(self, device):
        for t in range(10):
            N = 2
            H = 4
            L = 100
            S = 100
            E = np.random.randint(10, 256)
            M = np.random.randint(10, 256)
            Q = torch.rand(N, H, L, E).to(device).requires_grad_(True)
            K = torch.rand(N, H, S, E).to(device).requires_grad_(True)
            V = torch.randn(N, H, S, M).to(device).requires_grad_(True)

            self._zero_grad(Q, K, V)
            QK = torch.einsum("nhle,nhse->nhls", Q, K)
            mask = torch.tril(torch.ones(L, S))[None, None].to(device)
            QK = QK * mask
            QK = QK / (QK.sum(-1, keepdim=True) + 1e-6)
            V_new = torch.einsum("nhls,nhsm->nhlm", QK, V)
            V_new.sum().backward()
            grad = [torch.clone(x.grad) for x in [Q, K, V]]

            self._zero_grad(Q, K, V)
            V_new_hat = causal_dot_product(Q, K, V)
            Z = torch.einsum(
                "nhle,nhle->nhl",
                Q,
                torch.cumsum(K, dim=-2) + 1e-6
            ).unsqueeze(-1)

            V_new_hat = V_new_hat / Z
            V_new_hat.sum().backward()
            grad_hat = [torch.clone(x.grad) for x in [Q, K, V]]

            self.assertLess(
                torch.abs(V_new - V_new_hat).max(),
                5e-4
            )
            for g1, g2 in zip(grad, grad_hat):
                self.assertLess(
                    torch.abs(g1-g2).max(),
                    5e-4
                )

    def test_api_cpu(self):
        self._test_api("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA capable device")
    def test_api_cuda(self):
        self._test_api("cuda")


if __name__ == "__main__":
    unittest.main()
