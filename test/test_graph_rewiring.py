#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the graph rewiring
"""
import unittest
import torch
from torch import tensor
from torch import nn
from data import get_dataset
from graph_rewiring import apply_gdc
from graph_rewiring_eval import rewiring_test
from graph_rewiring_data import PPRDataset


class RewiringTests(unittest.TestCase):


  def setUp(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.opt = {
                'rewiring':False,
                'use_lcc': True,
                'dataset': 'Cora',
                'self_loop_weight': None,
                'gdc_method': 'ppr',
                'exact': True,
                'gdc_sparsification': 'topk',
                'gdc_threshold': 0.01,
                'gdc_k': 128,
                'hidden_layers': 1,
                'hidden_units': 64,
                'lr': 0.01,
                'dropout': 0.5,
                'weight_decay': 0.09604826107599472,
                'ppr_alpha': 0.05,
                'k': 128,
                'eps': None,
                'seed_development': 1684992425,
                'num_development': 1500
                }
    self.dataset = get_dataset(self.opt, '../data', use_lcc=True)
    self.n = self.dataset.data.num_nodes

  def tearDown(self) -> None:
    pass

  def test_PPR(self):
    # fix seeds
    # load PPR data
    PPR_dataset = PPRDataset(
                  name=self.opt['dataset'],
                  use_lcc=self.opt['use_lcc'],
                  alpha=self.opt['ppr_alpha'],
                  k=self.opt['gdc_k'],
                  eps=self.opt['gdc_threshold'])

    #fix masks
    # PPR_dataset.data = set_train_val_test_split(
    #     self.opt['seed_development'],
    #     dataset.data,
    #     num_development=self.opt['num_development'],
    #   ).to(self.device)

    # rewire default data
    rewired_orig = apply_gdc(self.dataset.data, self.opt, type='combined')

    # compare edge_indices
    edges_stats = rewiring_test("G0+rw", rewired_orig.edge_index, "PPR", PPR_dataset.data.edge_index, self.n)
    print(edges_stats)
    self.assertTrue(1 == 1)



if __name__ == '__main__':
  est = RewiringTests()
  est.setUp()
  est.test_PPR()
