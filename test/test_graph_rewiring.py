#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the graph rewiring
"""
import unittest
import torch
from data import get_dataset
from test_params import OPT
from GNN import GNN
from GNN_KNN import GNN_KNN
from graph_rewiring import apply_edge_sampling, edge_sampling, apply_KNN, apply_beltrami, add_edges, add_outgoing_attention_edges


class RewiringTests(unittest.TestCase):

  def setUp(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {
      'dataset': 'Cora',
      'edge_sampling_add': 0.1,
      'edge_sampling_rmv': 0.1,
      'edge_sampling_T': 'TN',
      'edge_sampling_space': 'attention',
      'block': 'attention'
    }
    self.opt = {**OPT, **opt}
    self.dataset = get_dataset(self.opt, '../data', True)

  def tearDown(self) -> None:
    pass

  def test_add_outgoing_attention_edges(self):
    self.opt['beltrami'] = False
    gnn = GNN(self.opt, self.dataset, device=self.device)
    # set attention weights
    out = gnn(self.dataset.data.x)
    M = 10
    edge_count = gnn.odeblock.odefunc.edge_index.shape[1]
    print(f"input edge count {edge_count}")
    ei = add_outgoing_attention_edges(gnn, M)
    edge_count = ei.shape[1]
    print(f"input edge count {edge_count}")
    self.assertTrue(ei.shape[1] == 2 * M)

  def test_apply_edge_sampling(self):
    print(self.opt)
    self.opt['beltrami'] = True
    self.opt['block'] = 'attention'
    pos_encoding = apply_beltrami(self.dataset.data, self.opt)
    self.opt['pos_enc_dim'] = pos_encoding.shape[1]
    gnn = GNN_KNN(self.opt, self.dataset, device=self.device)
    out = gnn(self.dataset.data.x, pos_encoding)
    self.opt['edge_sampling_rmv'] = 0.1
    self.opt['edge_sampling_add'] = 0
    input_edge_count = gnn.odeblock.odefunc.edge_index.shape[1]
    print(f"input edge count {input_edge_count}")
    apply_edge_sampling(self.dataset.data.x, pos_encoding, gnn, self.opt)
    edge_count = gnn.odeblock.odefunc.edge_index.shape[1]
    print(f"output edge count {edge_count}")
    self.assertTrue(gnn.odeblock.odefunc.edge_index.shape[1] < input_edge_count)

  def test_add_edges(self):
    self.opt['beltrami'] = False
    self.opt['block'] = 'attention'
    gnn = GNN(self.opt, self.dataset, device=self.device)
    out = gnn(self.dataset.data.x)
    edge_count = len(gnn.odeblock.odefunc.edge_index)
    print(f"input edge count {edge_count}")
    self.opt['edge_sampling_add'] = 0.1  # add this percentage of edges
    self.opt['edge_sampling_add_type'] == 'importance'
    new_edge_index = add_edges(gnn, self.opt)
    new_edge_count = len(new_edge_index)
    print(f"input edge count {new_edge_count}")


if __name__ == '__main__':
  est = RewiringTests()
  est.setUp()
  est.test_add_outgoing_attention_edges()
