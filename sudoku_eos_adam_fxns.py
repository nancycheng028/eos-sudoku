"""
Sudoku EoS Alternative Adam Functions

Functions adapted from https://github.com/locuslab/edge-of-stability/blob/github/src/adam.py
Which is the Github Repository corresponding to the paper "Adaptive Gradient Methods at the Edge of Stability" by 
Jeremy Cohen et al., available here: https://arxiv.org/abs/2207.14484
"""

from typing import List, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigs, eigsh
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                batch: Dataset, vector: Tensor, P: Tensor = None):
    """Compute a Hessian-vector product.
    
    If the optional preconditioner P is not set to None, return P^{-1/2} H P^{-1/2} v rather than H v.
    """
    p = len(parameters_to_vector(network.parameters()))
    n = len(batch)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    if P is not None:
        vector = vector / P.cuda().sqrt()

    x, y, m = (t.to(device='cuda') for t in batch)
    logits = network(x)
    loss = loss_fn(logits, y, m)

    grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads)
    if P is not None:
        hvp = hvp / P.cuda().sqrt()
    return hvp

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)

    try: 
        evals, evecs = eigsh(operator, k=neigs)
    except:
        evals, evecs = eigs(operator, k=neigs)

    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                            neigs=6, P=None):
    """ Compute the leading Hessian eigenvalues.
    
    If preconditioner P is not set to None, return top eigenvalue of P^{-1/2} H P^{-1/2} rather than H.
    """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, P=P).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals

def get_adam_nu(optimizer) -> torch.Tensor:
    vec = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            vec.append(state['exp_avg_sq'].view(-1))
    return torch.cat(vec)

def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])
