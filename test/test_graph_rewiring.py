#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the graph rewiring
"""
import unittest
import torch
from torch import tensor
from torch import nn
from torch.optim import Adam, Optimizer
from data import get_dataset
from DIGL_data import PPRDataset, set_train_val_test_split, get_dataset as DIGL_get_dataset, train as DIGL_train, \
  evaluate as DIGL_evaluate
from test_params import OPT
from GNN import GNN
from GNN_KNN import GNN_KNN
from graph_rewiring import apply_edge_sampling, edge_sampling, apply_KNN, apply_beltrami

from GNN_GCN import GCN
import copy


class RewiringTests(unittest.TestCase):

  def setUp(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = {
      'dataset': 'Cora',
      'edge_sampling_add': 0.1,
      'edge_sampling_rmv': 0.1,
      'edge_sampling_T': 'TN',
      'edge_sampling_space': 'pos_distance_QK'
      # 'gdc_method': 'ppr',
      # 'exact': True,
      # 'gdc_sparsification': 'topk',
      # 'gdc_threshold': 0.01,
      # 'gdc_k': 128,
      # 'hidden_layers': 1,
      # 'hidden_units': 64,
      # 'lr': 0.01,
      # 'dropout': 0.5,
      # 'weight_decay': 0.09604826107599472,
      # 'decay': 0.09604826107599472,
      # 'ppr_alpha': 0.05,
      # 'k': 128,
      # 'eps': None,
      # 'seed_development': 12345,  # 1684992425,
      # 'num_development': 1500,
      # 'optimizer': 'adam',
      # 'epoch': 100,
      # 'use_labels': False, 'beltrami': False
    }
    self.opt = {**OPT, **opt}
    self.dataset = get_dataset(self.opt, '../data', False)
    # self.n = self.dataset.data.num_nodes

  def tearDown(self) -> None:
    pass

  def test_data(self):
    opt = self.opt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(opt, '../data', use_lcc=True)

    DIGL_dataset = DIGL_get_dataset('Cora', use_lcc=True)

    print("Test data edge_index")
    self.assertTrue(torch.equal(dataset.data.edge_index, DIGL_dataset.data.edge_index))

    DIGL_dataset.data = set_train_val_test_split(opt['seed_development'], DIGL_dataset.data,
                                                 num_development=opt['num_development'], ).to(device)

    print("Test train_mask")
    self.assertTrue(torch.equal(dataset.data.train_mask, DIGL_dataset.data.train_mask))
    print("Test val_mask")
    self.assertTrue(torch.equal(dataset.data.val_mask, DIGL_dataset.data.val_mask))
    print("Test test_mask")
    self.assertTrue(torch.equal(dataset.data.test_mask, DIGL_dataset.data.test_mask))

    model = GCN(self.opt, dataset).to(device)
    DIGL_model = copy.deepcopy(model).to(device)
    # DIGL_model = GCN(self.opt, dataset).to(device)

    print("Test Model Params - pre train")
    parameters = [p for p in model.parameters() if p.requires_grad]
    DIGL_parameters = [p for p in DIGL_model.parameters() if p.requires_grad]
    self.assertEqual(len(parameters), len(DIGL_parameters))

    for i in range(len(parameters)):
      self.assertTrue(torch.equal(parameters[i], DIGL_parameters[i]))

    # our code
    # optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    # optimizer = rewiring_get_optimizer(opt['optimizer'], model, opt['lr'], opt['weight_decay'])
    # for epoch in range(1, opt['epoch']):
    #     print(f"epoch {epoch}")
    #     loss = rewiring_train(model, optimizer, dataset.data)
    #     train_acc, val_acc, tmp_test_acc = test(model, dataset.data, opt)
    #     print(val_acc, tmp_test_acc)

    # DIGL code
    DIGL_optimizer = Adam(
      [
        {'params': DIGL_model.non_reg_params, 'weight_decay': 0},
        {'params': DIGL_model.reg_params, 'weight_decay': opt['weight_decay']}
      ],
      lr=opt['lr'])

    for epoch in range(1, opt['epoch']):
      print(f"epoch {epoch}")
      DIGL_train(DIGL_model, DIGL_optimizer, DIGL_dataset.data)
      eval_dict = DIGL_evaluate(DIGL_model, DIGL_dataset.data, test=True)
      print(eval_dict['val_acc'], eval_dict['test_acc'])

    print("Test Model Params - post train")

    print("Test Homophilly Calc")

  def test_edge_sampling(self):
    gnn = GNN(self.opt, self.dataset, device=self.device)

  def test_apply_edge_sampling(self):
    print(self.opt)
    gnn = GNN_KNN(self.opt, self.dataset, device=self.device)
    pos_encoding = apply_beltrami(self.dataset.data, self.opt)
    self.opt['edge_sampling_rmv'] = 1
    apply_edge_sampling(self.dataset.data, pos_encoding, gnn, self.opt)
    self.assertTrue(len(gnn.odeblock.odefunc.edge_index) == 0)


if __name__ == '__main__':
  est = RewiringTests()
  est.setUp()
  est.test_apply_edge_sampling()
