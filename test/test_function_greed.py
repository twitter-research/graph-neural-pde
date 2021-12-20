#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test GREED attention
"""
import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
from function_greed import SpGraphGreedAttentionLayer, ODEFuncGreed
from torch_geometric.utils import softmax, to_dense_adj
from data import get_dataset
from test_params import OPT


class GreedAttentionTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {'dataset': 'Citeseer', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
                'K': 10,
                'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'linear_attention': True, 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'mixed_block': True, 'max_nfe': 1000, 'mix_features': False, 'attention_dim': 32, 'rewiring': None,
                'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None, 'total_deriv': None, 'directional_penalty': None, 'beltrami': False}
    self.opt = {**OPT, **opt}
  def tearDown(self) -> None:
    pass

