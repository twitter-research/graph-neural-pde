#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test early stop
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn

from data import get_dataset
from function_laplacian_diffusion import LaplacianODEFunc
from GNN_early import GNNEarly
from block_constant import ConstantODEblock
from utils import get_rw_adj
from test_params import OPT


class EarlyStopTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10,
                'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'block': 'constant', 'function': 'laplacian', 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'dopri5',
                'rewiring': None, 'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None,
                'jacobian_norm2': None, 'total_deriv': None, 'directional_penalty': None, 'step_size': 1, 'data_norm': 'rw',
                'earlystopxT': 3, 'max_iters': 10, 'beltrami': False}
    self.opt = {**OPT, **opt}
    self.dataset = get_dataset(self.opt, '../data', False)

  def tearDown(self) -> None:
    pass

  def test_block(self):
    data = self.dataset.data
    t = 1
    out_dim = 6
    func = LaplacianODEFunc(self.dataset.data.num_features, out_dim, self.opt, data, self.device)
    func.edge_index, func.edge_weight = get_rw_adj(data.edge_index, edge_weight=None, norm_dim=1,
                                                   fill_value=self.opt['self_loop_weight'],
                                                   num_nodes=data.num_nodes,
                                                   dtype=data.x.dtype)
    out = func(t, data.x)
    print(out.shape)
    self.assertTrue(out.shape == (self.dataset.data.num_nodes, self.dataset.num_features))
    gnn = GNNEarly(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, ConstantODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    self.assertTrue(odeblock.test_integrator.data.x.shape == data.x.shape)
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    gnn.set_solver_m2()
    gnn.set_solver_data(data)
    out = odeblock(data.x)
    print('ode block out', out)
    self.assertTrue(data.x.shape == out.shape)

  def test_rk4(self):
    data = self.dataset.data
    t = 1
    out_dim = 6
    self.opt['method'] = 'rk4'
    func = LaplacianODEFunc(self.dataset.data.num_features, out_dim, self.opt, data, self.device)
    func.edge_index, func.edge_weight = get_rw_adj(data.edge_index, edge_weight=None, norm_dim=1,
                                                   fill_value=self.opt['self_loop_weight'],
                                                   num_nodes=data.num_nodes,
                                                   dtype=data.x.dtype)
    out = func(t, data.x)
    print(out.shape)
    self.assertTrue(out.shape == (self.dataset.data.num_nodes, self.dataset.num_features))
    gnn = GNNEarly(self.opt, self.dataset, device=self.device)
    odeblock = gnn.odeblock
    self.assertTrue(isinstance(odeblock, ConstantODEblock))
    self.assertTrue(isinstance(odeblock.odefunc, LaplacianODEFunc))
    self.assertTrue(odeblock.test_integrator.data.x.shape == data.x.shape)
    gnn.train()
    out = odeblock(data.x)
    self.assertTrue(data.x.shape == out.shape)
    gnn.eval()
    gnn.set_solver_m2()
    gnn.set_solver_data(data)
    out = odeblock(data.x)
    print('ode block out', out)
    self.assertTrue(data.x.shape == out.shape)

  def test_gnn(self):
    gnn = GNNEarly(self.opt, self.dataset, device=self.device)
    gnn.train()
    out = gnn(self.dataset.data.x)
    print(out.shape)
    print(torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))
    self.assertTrue(out.shape == torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))
    gnn.eval()
    out = gnn(self.dataset.data.x)
    self.assertTrue(out.shape == torch.Size([self.dataset.data.num_nodes, self.dataset.num_classes]))
    solver = gnn.odeblock.test_integrator.solver
    self.assertTrue(solver.best_val >= 0)
    self.assertTrue(solver.best_test >= 0)


if __name__ == '__main__':
  est = EarlyStopTests()
  est.setUp()
  est.test_gnn()
