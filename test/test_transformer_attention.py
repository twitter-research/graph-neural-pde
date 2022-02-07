#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
from torch_geometric.utils import softmax, to_dense_adj

from function_transformer_attention import SpGraphTransAttentionLayer, ODEFuncTransformerAtt
from data import get_dataset
from test_params import OPT
from utils import ROOT_DIR

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

    def get_round_sum(tens, n_digits=3):
      val = torch.sum(tens, dim=int(not self.opt['attention_norm_idx']))
      round_sum = (val * 10 ** n_digits).round() / (10 ** n_digits)
      print('round sum', round_sum)
      return round_sum

    self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention1), torch.ones(size=dense_attention1.shape))))
    self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention2), torch.ones(size=dense_attention1.shape))))
    self.assertTrue(torch.all(attention > 0.))
    self.assertTrue(torch.all(attention <= 1.))

    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', True)
    data = dataset.data
    in_features = data.x.shape[1]
    out_features = data.x.shape[1]

    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(data.x, data.edge_index)  # should be n_edges x n_heads
    self.assertTrue(attention.shape == (data.edge_index.shape[1], self.opt['heads']))
    dense_attention1 = to_dense_adj(data.edge_index, edge_attr=attention[:, 0]).squeeze()
    dense_attention2 = to_dense_adj(data.edge_index, edge_attr=attention[:, 1]).squeeze()
    print('sums:', torch.sum(torch.isclose(dense_attention1, torch.ones(size=dense_attention1.shape))), dense_attention1.shape)
    print('da1', dense_attention1)
    print('da2', dense_attention2)
    self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention1), torch.ones(size=dense_attention1.shape))))
    self.assertTrue(torch.all(torch.isclose(get_round_sum(dense_attention2), torch.ones(size=dense_attention2.shape))))
    self.assertTrue(torch.all(attention > 0.))
    self.assertTrue(torch.all(attention <= 1.))

  def test_symmetric_attention(self):
    in_features = self.x1.shape[1]
    out_features = self.x1.shape[1]
    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(self.x1, self.edge1)  # should be n_edges x n_heads

    self.assertTrue(torch.all(torch.isclose(att_layer.Q.weight, att_layer.K.weight)))
    self.assertTrue(torch.all(torch.eq(attention, 0.5 * torch.ones((self.edge1.shape[1], self.x1.shape[1])))))

  def test_module(self):
    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', False)
    t = 1
    out_dim = 6
    func = ODEFuncTransformerAtt(dataset.data.num_features, out_dim, self.opt, dataset.data, self.device)
    out = func(t, dataset.data.x)
    print(out.shape)
    self.assertTrue(out.shape == (dataset.data.num_nodes, dataset.num_features))

  def test_head_aggregation(self):
    in_features = self.x.shape[1]
    out_features = self.x.shape[1]
    self.opt['head'] = 4
    att_layer = SpGraphTransAttentionLayer(in_features, out_features, self.opt, self.device, concat=True)
    attention, _ = att_layer(self.x, self.edge)
    ax1 = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge, attention[:, idx], self.x.shape[0], self.x.shape[0], self.x) for idx in
         range(self.opt['heads'])], dim=0), dim=0)
    mean_attention = attention.mean(dim=1)
    ax2 = torch_sparse.spmm(self.edge, mean_attention, self.x.shape[0], self.x.shape[0], self.x)
    self.assertTrue(torch.all(torch.isclose(ax1,ax2)))

  def test_two_way_edge(self):
    dataset = get_dataset(self.opt, f'{ROOT_DIR}/data', False)
    edge = dataset.data.edge_index
    print(f"is_undirected {dataset.data.is_undirected()}")

    edge_dict = {}

    for idx, src in enumerate(edge[0, :]):
      src = int(src)
      if src in edge_dict:
        edge_dict[src].add(int(edge[1, idx]))
      else:
        edge_dict[src] = set([int(edge[1, idx])])

    print(f"edge shape {edge.shape}")
    src_test = edge[:, edge[0, :] == 1][1, :]
    dst_test = edge[:, edge[1, :] == 1][0, :]
    print('dst where src = 1', src_test)
    print('src where dst = 1', dst_test)

    for idx, dst in enumerate(edge[1, :]):
      dst = int(dst)
      self.assertTrue(int(edge[0, idx]) in edge_dict[dst])


if __name__ == '__main__':
  AT = AttentionTests()
  AT.test_symmetric_attention()