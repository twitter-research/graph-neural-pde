#!/usr/bin/env python
# -*- coding: utf-8 -*-
# needed for CI/CD
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn
from data import get_dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import numpy as np
from utils import get_rw_adj, gcn_norm_fill_val
from sklearn.preprocessing import normalize
from test_params import OPT


class DummyDataset():
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


def get_rw_numpy(arr, self_loops, norm_dim):
  new_arr = arr + np.identity(arr.shape[0]) * self_loops
  dim = 0 if norm_dim == 1 else 1
  return normalize(new_arr, norm='l1', axis=dim)


class UtilsTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)
    self.data = Data(x=self.x, edge_index=self.edge)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Cora', 'self_loop_weight': 0, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
                'K': 10,
                'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'linear_attention': True, 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'rewiring': None, 'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None,
                'jacobian_norm2': None, 'total_deriv': None, 'directional_penalty': None, 'beltrami': False}
    self.opt = {**OPT, **opt}
    self.dataset = get_dataset(self.opt, '../data', False)

  def tearDown(self) -> None:
    pass

  def test_gcn_norm_fill_val(self):
    edge_index, edge_weight = gcn_norm_fill_val(self.edge, None, self.opt['self_loop_weight'], self.x.shape[0],
                                                dtype=self.x.dtype)

  def self_loop_test(self, base_adj, self_loop=0, norm_dim=0):
    edge_index, edge_weight = get_rw_adj(self.edge, norm_dim=norm_dim, fill_value=self_loop,
                                         num_nodes=self.x.shape[0])
    dense_rw_adj = to_dense_adj(edge_index, edge_attr=edge_weight).numpy().squeeze()
    numpy_arr = get_rw_numpy(base_adj, self_loop, norm_dim)
    print('self loop', self_loop)
    print('numpy arr', numpy_arr)
    print('torch arr', dense_rw_adj)
    self.assertTrue(np.allclose(numpy_arr, dense_rw_adj))

  def test_get_rw_adj(self):
    base_adj = to_dense_adj(self.edge).numpy().squeeze()
    self_loops = [0, 0.3, 1, 3.2]
    for self_loop in self_loops:
      self.self_loop_test(base_adj, self_loop, norm_dim=0)
      self.self_loop_test(base_adj, self_loop, norm_dim=1)

if __name__ == '__main__':
  tests = UtilsTests()
  tests.setUp()
  tests.test_get_rw_adj()
