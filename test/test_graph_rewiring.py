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
from graph_rewiring import apply_gdc
from graph_rewiring_eval import rewiring_train, rewiring_test, rewiring_get_optimizer
from run_GNN import print_model_params, get_optimizer, test_OGB, test#, train
from DIGL_data import PPRDataset, set_train_val_test_split, get_dataset as DIGL_get_dataset, train as DIGL_train, evaluate as DIGL_evaluate

from GNN_GCN import GCN
import copy

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
                'decay': 0.09604826107599472,
                'ppr_alpha': 0.05,
                'k': 128,
                'eps': None,
                'seed_development': 12345, #1684992425,
                'num_development': 1500,
                'optimizer': 'adam',
                'epoch': 100,
                'use_labels': False
                }
    # self.dataset = get_dataset(self.opt, '../data', use_lcc=True)
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

    DIGL_dataset.data = set_train_val_test_split(opt['seed_development'],DIGL_dataset.data,
        num_development=opt['num_development'],).to(device)

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

    #our code
    # optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    # optimizer = rewiring_get_optimizer(opt['optimizer'], model, opt['lr'], opt['weight_decay'])
    # for epoch in range(1, opt['epoch']):
    #     print(f"epoch {epoch}")
    #     loss = rewiring_train(model, optimizer, dataset.data)
    #     train_acc, val_acc, tmp_test_acc = test(model, dataset.data, opt)
    #     print(val_acc, tmp_test_acc)

    #DIGL code
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


  # def test_PPR(self):
  #   # fix seeds
  #   # load PPR data
  #   PPR_dataset = PPRDataset(
  #                 name=self.opt['dataset'],
  #                 use_lcc=self.opt['use_lcc'],
  #                 alpha=self.opt['ppr_alpha'],
  #                 k=self.opt['gdc_k'],
  #                 eps=self.opt['gdc_threshold'])
  #
  #   #fix masks
  #   # PPR_dataset.data = set_train_val_test_split(
  #   #     self.opt['seed_development'],
  #   #     dataset.data,
  #   #     num_development=self.opt['num_development'],
  #   #   ).to(self.device)
  #
  #   # rewire default data
  #   rewired_orig = apply_gdc(self.dataset.data, self.opt, type='combined')
  #
  #   # compare edge_indices
  #   edges_stats = rewiring_test("G0+rw", rewired_orig.edge_index, "PPR", PPR_dataset.data.edge_index, self.n)
  #   print(edges_stats)
  #   self.assertTrue(1 == 1)



# if __name__ == '__main__':
#   est = RewiringTests()
#   est.setUp()
  # est.test_data()
  # est.test_PPR()
