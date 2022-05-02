#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test attention
"""
import unittest
import torch
from torch import tensor
from torch import nn
import torch_sparse
from torch_geometric.utils import softmax, to_dense_adj
from data import get_dataset
from test_params import OPT
from greed_params import greed_test_params
from heterophilic import get_fixed_splits
from utils import *
import pandas as pd


class HeteroTests(unittest.TestCase):
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
    opt = greed_test_params(opt)  ###extra params for testing GREED
    self.opt = {**OPT, **opt}
  def tearDown(self) -> None:
    pass

  #check datasets.zip is same as pyg dataloaders
  def test_geom_zip_datasets(self):
    self.opt['not_lcc'] = False
    for ds in ["Cora", "Citeseer", "Pubmed"]:
      for lcc_tf in [True, False]:
        self.opt['dataset'] = ds
        self.opt['not_lcc'] = lcc_tf
        pyg_datset = get_dataset(self.opt, '../data', self.opt['not_lcc'])
        geom_dataset = get_dataset_geom(self.opt, '../datasets', self.opt['not_lcc'])
        self.assertTrue(torch.all(torch.isclose(pyg_datset.data.x, geom_dataset.data.x)))
    print("they all passed")


  #Check it’s true that Cora uses LCC and Citeseer does not by comparing to the provided datasets from geom-gcn
  #For citeseer check train/val/test for each split
  def test_get_fixed_splits(self):
    df_list = []
    for ds in ["Cora", "Citeseer", "Pubmed"]:
      for lcc_tf in [True, False]:
        self.opt['dataset'] = ds
        self.opt['not_lcc'] = lcc_tf
        dataset = get_dataset(self.opt, '../data', self.opt['not_lcc'])
        data = dataset.data.to(self.device)
        for rep in range(10):
          data = get_fixed_splits(data, self.opt['dataset'], rep)
          df_row = [ds, rep, lcc_tf,
                    data.train_mask.shape, torch.count_nonzero(data.train_mask).item(),
                    data.val_mask.shape, torch.count_nonzero(data.val_mask).item(),
                    data.test_mask.shape, torch.count_nonzero(data.test_mask).item(),
                    torch.count_nonzero(data.train_mask).item() + torch.count_nonzero(data.val_mask).item() + torch.count_nonzero(data.test_mask).item(),
                    data.x.shape]
          df_list.append(df_row)
    df_cols = ["ds", " rep", "not_lcc",
                "train_mask.shape", "nonzero(train_mask)",
                "val_mask.shape", "nonzero(val_mask)",
                "test_mask.shape", "nonzero(test_mask)",
                "sum_non_zero", "x.shape"]
    df = pd.DataFrame(df_list, columns=df_cols)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(df)


from data import *
def get_dataset_geom(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
  #edit these two lines to work in datasets.zip from geom_gcn
  ds = opt['dataset'].lower()
  path = data_dir #os.path.join(data_dir, ds.lower())
  # if ds in ['Cora', 'Citeseer', 'Pubmed']:
  if ds in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(path, ds)#, "geom-gcn" if opt["geom_gcn_splits"] else "public")
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds in ['chameleon', 'squirrel']:
    dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path,
                                     transform=T.ToSparseTensor())
    use_lcc = False  #  never need to calculate the lcc with ogb datasets
  elif ds == 'Karate':
    dataset = KarateClub()
    dataset.data.val_mask = ~dataset.data.train_mask
    dataset.data.test_mask = ~dataset.data.train_mask
    use_lcc = False

  else:
    raise Exception('Unknown dataset.')

  if use_lcc:
    lcc = get_largest_connected_component(dataset)

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    data = Data(
      x=x_new,
      edge_index=torch.LongTensor(edges),
      y=y_new,
      train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
      val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
    )
    dataset.data = data
  if opt['rewiring'] is not None:
    dataset.data = rewire(dataset.data, opt, data_dir)
  # if opt['data_homoph']: #todo does this need to go after any rewiring / make undirected steps
  #   dataset.data = hetro_edge_addition(dataset.data, opt)

  train_mask_exists = True
  try:
    dataset.data.train_mask
  except AttributeError:
    train_mask_exists = False

  if ds == 'ogbn-arxiv':
    split_idx = dataset.get_idx_split()
    ei = to_undirected(dataset.data.edge_index)
    data = Data(
    x=dataset.data.x,
    edge_index=ei,
    y=dataset.data.y,
    train_mask=split_idx['train'],
    test_mask=split_idx['test'],
    val_mask=split_idx['valid'])
    dataset.data = data
    train_mask_exists = True

  #todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
  if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
    dataset.data = set_train_val_test_split(
      12345,
      dataset.data,
      num_development=5000 if ds == "CoauthorCS" else 1500)

  return dataset

if __name__ == '__main__':
  AT = HeteroTests()
  AT.test_symmetric_attention()