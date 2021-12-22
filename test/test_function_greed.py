#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GREED attention
Note: Greed only works WITHOUT self-loops
"""
import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
from function_greed import ODEFuncGreed
from torch_geometric.utils import softmax, to_dense_adj
from data import get_dataset
from test_params import OPT


class Data:
  def __init__(self, edge_index, x):
    self.x = x
    self.edge_index = edge_index
    self.edge_attr = None


class GreedTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Citeseer', 'self_loop_weight': 0, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
           'K': 10,
           'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
           'hidden_dim': 6, 'linear_attention': True, 'augment': False, 'adjoint': False,
           'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
           'mixed_block': True, 'max_nfe': 1000, 'mix_features': False, 'attention_dim': 32, 'rewiring': None,
           'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None,
           'total_deriv': None, 'directional_penalty': None, 'beltrami': True}
    self.opt = {**OPT, **opt}
    self.data = Data(self.edge, self.x)
    self.greed_func = ODEFuncGreed(2, 2, self.opt, self.data, torch.device('cpu'))

  def tearDown(self) -> None:
    pass

  def test_get_deg_inv_sqrt(self):
    pass

  def test_symmetrically_normalise(self):
    pass

  def test_get_tau(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    self.assertTrue(tau.shape == torch.Size((self.edge.shape[1], 1)))
    self.assertTrue(tau.shape == tau_transpose.shape)
    # check that tau(2,1) is  tau_transpose(1,2) and vice versa
    self.assertTrue(tau[2] == tau_transpose[3])
    self.assertTrue(tau[3] == tau_transpose[2])

  def test_get_gamma(self):
    tau, tau_transpose = self.greed_func.get_tau(self.x)
    gamma = self.greed_func.get_gamma(self.x, tau, tau_transpose, epsilon=1e-6)
    self.assertTrue(list(gamma.shape) == [self.edge.shape[1]])

  def test_get_dynamics(self):
    pass
