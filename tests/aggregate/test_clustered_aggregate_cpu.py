#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import unittest
import os
import numpy as np
import time

import torch

try:
    from fast_transformers.aggregate import clustered_aggregate, \
            clustered_broadcast
except ImportError:
    pass


class TestAggregateCPU(unittest.TestCase):

    def test_aggregate(self):
        N = 2
        H = 4
        L = 80
        E = 2
        C = 4

        for i in range(30):
            C = np.random.randint(5, 10)
            L = np.random.randint(1, 30) * C
            E = np.random.randint(10, 128)
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L E C: "
                       "{} {} {} {} {}").format(N, H, L, E, C))

            x = torch.rand((N, H, L, E)).cpu()
            g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cpu()
            f = torch.ones(N, H, C).cpu() * (C / L)
            counts = torch.ones_like(f, dtype=torch.int32) * (L // C)
            y = torch.zeros(N, H, C, E).cpu()
            lengths = torch.full((N,), L, dtype=torch.int32).to(x.device)

            sorted_g, sorted_gi = torch.sort(g.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

            q_offset = torch.arange(N*H, device=x.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)

            # sorted queries, keys, values
            s_x = x.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
            y = clustered_aggregate(
                s_x, sorted_g.view(N, H, -1), f, lengths, y
            )
            for i in range(C):
                self.assertLess(
                    torch.abs(
                        x[:, :, i::C, :].mean(2) - y[:, :, i, :]
                    ).max().item(),
                    1e-6
                )

    def test_aggregate_masked(self):
        N = 10
        H = 3
        L = 40
        E = 32
        C = 4

        for i in range(30):
            C = np.random.randint(5, 10)
            L = np.random.randint(2, 30) * C
            E = np.random.randint(10, 128)
            if os.getenv("VERBOSE_TESTS", ""):
                print(("Testing: N H L E C: "
                       "{} {} {} {} {}").format(N, H, L, E, C))

            x = torch.rand((N, H, L, E)).cpu()
            g = (torch.arange(L) % C).view(1, 1, L).repeat(N, H, 1).int().cpu()
            g[:, :, -C:] = C + 1
            c = (L // C) - 1

            lengths = torch.full((N,), L-C, dtype=torch.int32).to(x.device)
            f = torch.ones(N, H, C).cpu() / float(c)
            counts = torch.ones_like(f, dtype=torch.int32) * c
            y = torch.zeros(N, H, C, E).cpu()

            sorted_g, sorted_gi = torch.sort(g.view(N*H, -1), dim=-1)
            sorted_rev_gi = torch.argsort(sorted_gi, dim=-1)

            q_offset = torch.arange(N*H, device=x.device).unsqueeze(-1) * L
            q_flat = (sorted_gi + q_offset).reshape(-1)

            # sorted queries, keys, values
            s_x = x.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)
            y = clustered_aggregate(
                s_x, sorted_g.view(N, H, -1), f, lengths, y
            )

            for i in range(C):
                x_m = x[:, :, i::C, :][:, :, :-1, :].mean(2)
                self.assertLess(
                    torch.abs(
                        x_m - y[:, :, i, :]
                    ).max().item(),
                    1e-6
                )


if __name__ == "__main__":
    unittest.main()
