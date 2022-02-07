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
import numpy as np

from data import get_dataset
from function_laplacian_diffusion import LaplacianODEFunc
from GNN import GNN
from block_mixed import MixedODEblock
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from test_params import OPT


class DummyDataset():
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class MixedODEBlockTests(unittest.TestCase):
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
    opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'sc', 'heads': 2,
                'K': 10, 'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc',
                'hidden_dim': 6, 'block': 'mixed', 'function': 'laplacian', 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'rewiring': None, 'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None,
                'total_deriv': None, 'directional_penalty': None, 'jacobian_norm2': None, 'step_size':1, 'max_iter': 10, 'beltrami': False}
    self.opt = {**OPT, **opt}

    self.dataset = get_dataset(self.opt, '../data', False)

  def tearDown(self) -> None:
    pass

  def test_block_toy(self):
    # construct a pyg dataset
    dataset = DummyDataset(self.data, 3)
    self.opt['heads'] = 1
    self.opt['hidden_dim'] = 2  # same as the raw data so we don't have to first encode it
    gnn = GNN(self.opt, dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, MixedODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    self.assertTrue(odeblock.gamma.item() == 0.)

  def test_get_mixed_attention(self):
    dataset = DummyDataset(self.data, 3)
    self.opt['heads'] = 1
    self.opt['hidden_dim'] = 2  # same as the raw data so we don't have to first encode it
    gnn = GNN(self.opt, dataset, device=self.device)
    odeblock = gnn.odeblock
    attention = odeblock.get_attention_weights(self.x)
    mixed_att_weights = odeblock.get_mixed_attention(self.x)
    mixed_att = to_dense_adj(odeblock.odefunc.edge_index,
                             edge_attr=mixed_att_weights).detach().numpy().squeeze()
    rw_arr = to_dense_adj(odeblock.odefunc.edge_index,
                          edge_attr=odeblock.odefunc.edge_weight).detach().numpy().squeeze()
    att_arr = to_dense_adj(odeblock.odefunc.edge_index, edge_attr=attention).detach().numpy().squeeze()
    gamma = torch.sigmoid(odeblock.gamma).detach().numpy()
    mixed_att_test = (1 - gamma) * att_arr + gamma * rw_arr
    self.assertTrue(np.allclose(mixed_att, mixed_att_test))

  def test_block_cora(self):
    data = self.dataset.data
    self.opt['hidden_dim'] = self.dataset.num_features
    self.opt['heads'] = 1
    gnn = GNN(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, MixedODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    self.assertTrue(odeblock.gamma.item() == 0.)
    self.assertTrue(odeblock.odefunc.edge_weight.shape is not None)
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    self.opt['heads'] = 2
    try:
      gnn = GNN(self.opt, self.dataset, device=self.device)
      self.assertTrue(False)
    except AssertionError:
      pass


if __name__ == '__main__':
  tests = MixedODEBlockTests()
  tests.setUp()
  tests.test_block_toy()
