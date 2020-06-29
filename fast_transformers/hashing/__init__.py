#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#


import torch

from .hash_cpu import compute_hashes as compute_hashes_cpu
try:
    from .hash_cuda import compute_hashes as compute_hashes_cuda
except ImportError:
    pass


def compute_hashes(X, A, H=None):
    device = X.device
    if H is None:
        H = torch.zeros(len(X), dtype=torch.int64, device=device)
    else:
        H.zero_()
    if A.shape[1] != X.shape[1] + 1:
        raise ValueError("The hash requires a bias")

    if device.type == "cpu":
        compute_hashes_cpu(X, A, H)
    else:
        compute_hashes_cuda(X, A, H)

    return H
