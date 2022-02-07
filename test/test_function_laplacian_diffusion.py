#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn
import numpy as np
from sklearn.preprocessing import normalize

from data import get_dataset
from function_laplacian_diffusion import LaplacianODEFunc
from GNN import GNN
from block_constant import ConstantODEblock
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from utils import get_rw_adj, get_sym_adj
from test_params import OPT


class DummyDataset():
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class FunctionLaplacianDiffusionTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)
    self.data = Data(x=self.x, edge_index=self.edge)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
                'K': 10,
                'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'augment': False, 'adjoint': False,
                'block': 'constant', 'function': 'laplacian',
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'rewiring': None, 'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None,
                'jacobian_norm2': None, 'total_deriv': None, 'directional_penalty': None, 'step_size': 1, 'data_norm': 'rw',
                'max_iters': 10, 'beltrami': False}
    self.opt = {**OPT, **opt}

    self.dataset = get_dataset(self.opt, '../data', False)

  def tearDown(self) -> None:
    pass

  def test_block_toy(self):
    # construct a pyg dataset
    num_nodes = 3
    dataset = DummyDataset(self.data, num_nodes)
    gnn = GNN(self.opt, dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, ConstantODEblock))
    func = odeblock.odefunc
    self.assertTrue(isinstance(func, LaplacianODEFunc))
    func.edge_index, func.edge_weight = get_rw_adj(self.edge, edge_weight=None, norm_dim=1,
                                                   fill_value=self.opt['self_loop_weight'], num_nodes=3,
                                                   dtype=None)
    sym_adj = get_sym_adj(self.data, self.opt)

    rw_adj = to_dense_adj(func.edge_index, edge_attr=func.edge_weight).numpy().squeeze()
    input_adj = to_dense_adj(dataset.data.edge_index).numpy().squeeze()
    augmented_input = input_adj + self.opt['self_loop_weight'] * np.identity(num_nodes)
    augmented_degree = augmented_input.sum(axis=1)  # symmetric so axis doesn't matter
    sqr_degree = np.sqrt(augmented_degree)
    test_sym_adj = np.divide(np.divide(augmented_input, sqr_degree[:, None]), sqr_degree[None, :])

    test_rw_adj = normalize(augmented_input, norm='l1', axis=0)
    sym_adj = sym_adj.to_dense()
    print('rw adjacency', rw_adj)
    print('test rw adjacency', test_rw_adj)
    self.assertTrue(np.allclose(test_rw_adj, rw_adj))
    self.assertTrue(np.allclose(test_sym_adj, sym_adj.numpy().squeeze()))
    print('sym adjacency', sym_adj)

  def test_block_cora(self):
    data = self.dataset.data
    self.opt['hidden_dim'] = self.dataset.num_features
    self.opt['heads'] = 1
    gnn = GNN(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, ConstantODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)


if __name__ == '__main__':
  tests = FunctionLaplacianDiffusionTests()
  tests.setUp()
  tests.test_block_cora()
