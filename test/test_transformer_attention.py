#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import unittest
import torch
from torch import tensor
from torch import nn
from unittest.mock import patch
import logging
from function_transformer_attention import SpGraphTransAttentionLayer, ODEFuncTransformerAtt
from torch_geometric.utils import softmax, to_dense_adj
from data import get_dataset


class AttentionTests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2, 1], [1, 0, 1, 2]])
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float)
    self.edge1 = tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    self.x1 = torch.ones((3, 2), dtype=torch.float)

    self.W = tensor([[2, 1], [3, 2]], dtype=torch.float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=torch.float)
    self.leakyrelu = nn.LeakyReLU(0.2)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.opt = {'dataset': 'Citeseer', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2,
                'K': 10,
                'attention_norm_idx': 0, 'add_source': False, 'alpha': 1, 'alpha_dim': 'vc', 'beta_dim': 'vc',
                'hidden_dim': 6, 'linear_attention': True, 'augment': False, 'adjoint': False,
                'tol_scale': 1, 'time': 1, 'ode': 'ode', 'input_dropout': 0.5, 'dropout': 0.5, 'method': 'euler',
                'mixed_block': True, 'max_nfe': 1000, 'mix_features': False, 'attention_dim': 32, 'rewiring': None,
                'no_alpha_sigmoid': False, 'reweight_attention': False, 'kinetic_energy': None, 'jacobian_norm2': None, 'total_deriv': None, 'directional_penalty': None}

  def tearDown(self) -> None:
    pass

  def test(self):
    h = torch.mm(self.x, self.W)
    edge_h = torch.cat((h[self.edge[0, :], :], h[self.edge[1, :], :]), dim=1)
    self.assertTrue(edge_h.shape == torch.Size([self.edge.shape[1], 2 * 2]))
    ah = self.alpha.mm(edge_h.t()).t()
    self.assertTrue(ah.shape == torch.Size([self.edge.shape[1], 1]))
    edge_e = self.leakyrelu(ah)
    attention = softmax(edge_e, self.edge[1])
    print(attention)

  def test_function(self):
    in_features = self.x.shape[1]
    out_features = self.x.shape[1]
    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(self.x, self.edge)  # should be n_edges x n_heads
    self.assertTrue(attention.shape == (self.edge.shape[1], self.opt['heads']))
    dense_attention1 = to_dense_adj(self.edge, edge_attr=attention[:, 0]).squeeze()
    dense_attention2 = to_dense_adj(self.edge, edge_attr=attention[:, 1]).squeeze()

    print('da2', dense_attention2)

    def get_round_sum(tens, n_digits=3):
      val = torch.sum(tens, dim=int(not self.opt['attention_norm_idx']))
      return (val * 10 ** n_digits).round() / (10 ** n_digits)

    self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention1), 1.)))
    self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention2), 1.)))
    self.assertTrue(torch.all(attention > 0.))
    self.assertTrue(torch.all(attention <= 1.))

    dataset = get_dataset(self.opt, '../data', False)
    data = dataset.data
    in_features = data.x.shape[1]
    out_features = data.x.shape[1]

    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(data.x, data.edge_index)  # should be n_edges x n_heads
    self.assertTrue(attention.shape == (data.edge_index.shape[1], self.opt['heads']))
    dense_attention1 = to_dense_adj(data.edge_index, edge_attr=attention[:, 0]).squeeze()
    dense_attention2 = to_dense_adj(data.edge_index, edge_attr=attention[:, 1]).squeeze()
    self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention1), 1.)))
    self.assertTrue(torch.all(torch.eq(get_round_sum(dense_attention2), 1.)))
    self.assertTrue(torch.all(attention > 0.))
    self.assertTrue(torch.all(attention <= 1.))

  def test_symmetric_attention(self):
    in_features = self.x1.shape[1]
    out_features = self.x1.shape[1]
    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(self.x1, self.edge1)  # should be n_edges x n_heads

    self.assertTrue(not torch.all(torch.eq(att_layer.Q.weight, att_layer.K.weight)))
    self.assertTrue(torch.all(torch.eq(attention, 0.5 * torch.ones((self.edge1.shape[1], self.x1.shape[1])))))

  def test_module(self):
    dataset = get_dataset(self.opt, '../data', False)
    t = 1
    out_dim = 6
    func = ODEFuncTransformerAtt(dataset.data.num_features, out_dim, self.opt, dataset.data, self.device)
    out = func(t, dataset.data.x)
    print(out.shape)
    self.assertTrue(out.shape == (dataset.data.num_nodes, dataset.num_features))

  def test_two_way_edge(self):
    dataset = get_dataset(self.opt, '../data', False)
    edge = dataset.data.edge_index
    print(f"is_undirected {dataset.data.is_undirected()}")

    edge_dict = {}

    for idx, src in enumerate(edge[0, :]):
      src = int(src)
      if src in edge_dict:
        edge_dict[src].add(int(edge[1, idx]))
      else:
        edge_dict[src] = set([int(edge[1, idx])])

    # 2*5429 = 10858 != 10556
    print(f"edge shape {edge.shape}")
    src_test = edge[:, edge[0, :] == 1][1, :]
    dst_test = edge[:, edge[1, :] == 1][0, :]
    # src_test = edge[:, edge[0, :]==1862]
    # dst_test = edge[:, edge[1, :]==1862]
    print('dst where src = 1', src_test)
    print('src where dst = 1', dst_test)

    for idx, dst in enumerate(edge[1, :]):
      dst = int(dst)
      # if not (edge[:, 0][idx] in edge_dict[dst]):
      #   print(edge[:,0][idx])
      #   print(edge_dict[dst])
      self.assertTrue(int(edge[0, idx]) in edge_dict[dst])
