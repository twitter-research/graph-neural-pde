#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import unittest
import torch
from torch import tensor
from torch import nn
from GNN_ICML20 import gcn_norm_fill_val, coo2tensor, train_ray
from data import get_dataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from ray.tune.utils import diagnose_serialization
from functools import partial
import os
from GDE import GDE
from torch_geometric.data import Data
import torch_geometric.transforms as T
import numpy as np
import torch.nn.functional as F


class DummyDataset():
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes

  def __getitem__(self, item):
    return self.data


class GDETests(unittest.TestCase):
  def setUp(self):
    self.edge = tensor([[0, 2, 2], [1, 0, 1]], dtype=torch.long)
    self.x = tensor([[1., 2.], [3., 2.], [4., 5.]], dtype=torch.float32)
    self.W = tensor([[2, 1], [3, 2]], dtype=float)
    self.alpha = tensor([[1, 2, 3, 4]], dtype=float)
    self.leakyrelu = nn.LeakyReLU(0.2)
    self.data = Data(x=self.x, edge_index=self.edge)

  def tearDown(self) -> None:
    pass

  def test_loss(self):
    opt = dict(dataset='Cora', method='rk4', time=3, tol_scale=10, tol_scale_adjoint=10, hidden_dim=4, adjoint=False,
               dropout=0.5,
               self_loop_weight=1, GDE=True, use_lcc=False, rewiring=False, max_nfe=100)
    dataset = DummyDataset(self.data, 3)
    dataset.data = T.TargetIndegree()(dataset.data)
    model = GDE(opt, dataset, 'cpu')
    out = model(self.x)
    cora_labels = torch.tensor([0, 1, 0])
    loss = F.nll_loss(out, cora_labels)
    self.assertTrue(loss > 0)

  def test_train(self):
    opt = dict(dataset='Cora', method='rk4', time=3, tol_scale=10, tol_scale_adjoint=10, hidden_dim=4, adjoint=False,
               dropout=0.5,
               self_loop_weight=1, GDE=True, use_lcc=False, rewiring=False, max_nfe=100)
    dataset = DummyDataset(self.data, 3)
    dataset.data = T.TargetIndegree()(dataset.data)
    model = GDE(opt, dataset, 'cpu')

  def test_Cora(self):
    opt = dict(dataset='Cora', method='rk4', time=3, tol_scale=10, tol_scale_adjoint=10, hidden_dim=4, adjoint=False,
               dropout=0.5,
               self_loop_weight=1, GDE=True, use_lcc=False, rewiring=False, max_nfe=100)
    dataset = get_dataset(opt, '../data', False)
    dataset.data = T.TargetIndegree()(dataset.data)

    d = dataset.data
    dd = DummyDataset(self.data, 3)
    dd.data = T.TargetIndegree()(dd.data)
    ddd = dd.data
    print(
      'x types {}:{} edge_attr {}:{} types edge_index types {}:{}'.format(ddd.x.dtype, d.x.dtype, ddd.edge_attr.dtype,
                                                                          d.edge_attr.dtype, ddd.edge_index.dtype,
                                                                          d.edge_index.dtype))
    self.assertTrue(True)

  def test_transform(self):
    dataset = DummyDataset(self.data, 3)
    # for each edge (src,dst) store the degree of degree_dst / max(degree_dst)
    dataset.data = T.TargetIndegree()(dataset.data)
    self.assertTrue(dataset.data.edge_attr is not None)
    self.assertTrue(np.array_equal(dataset.data.edge_attr.numpy(), np.array([[1.], [0.5], [1.]])))


def main():
  data_dir = os.path.abspath("../data")
  trainable = partial(train_ray, data_dir=data_dir)
  diagnose_serialization(trainable)
  opt = {'dataset': 'Cora', 'improved': False, 'self_loop_weight': 1.}
  dataset = get_dataset(opt, '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = dataset.data
  edge_index1, edge_weight1 = gcn_norm(data.edge_index, data.edge_attr, data.num_nodes,
                                       opt['improved'], opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, data.edge_attr, opt['self_loop_weight'], data.num_nodes,
                                              opt['self_loop_weight'] > 0)
  assert torch.all(edge_index.eq(edge_index1))
  assert torch.all(edge_weight.eq(edge_weight1))
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  coo = coo2tensor(coo, device)


if __name__ == '__main__':
  main()
