"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os
from os import path
import numpy as np
import gdown
import scipy
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, KarateClub, LINKXDataset, Twitch#, WikipediaNetwork #todo for AWS download hetero using PyG
from graph_rewiring import get_two_hop, apply_gdc
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, is_undirected
from graph_rewiring import make_symmetric, apply_pos_dist_rewire
from heterophilic import WebKB, Actor, WikipediaNetwork, Genius #todo then copy from ds/ds/geom/raw to ds/ds/raw and process using this
from data_synth_hetero import get_pyg_syn_cora, get_SBM
from utils import DummyDataset

DATA_PATH = '../data'

def rewire(data, opt, data_dir):
  rw = opt['rewiring']
  if rw == 'two_hop':
    data = get_two_hop(data)
  elif rw == 'gdc':
    data = apply_gdc(data, opt)
  elif rw == 'pos_enc_knn':
    data = apply_pos_dist_rewire(data, opt, data_dir)
  return data


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
  ds = opt['dataset']
  path = os.path.join(data_dir, ds)
  if ds in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(path, ds)#, "geom-gcn" if opt["geom_gcn_splits"] else "public")
  elif ds in ['Computers', 'Photo']:
    dataset = Amazon(path, ds)
  elif ds == 'CoauthorCS':
    dataset = Coauthor(path, 'CS')
  elif ds in ['cornell', 'texas', 'wisconsin']:
    if opt['data_feat_norm']:
      dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    else:
      dataset = WebKB(root=path, name=ds)
  elif ds in ['chameleon', 'squirrel']:
    if opt['data_feat_norm']:
      dataset = WikipediaNetwork(root=path, name=ds, transform=T.NormalizeFeatures())
    else:
      dataset = WikipediaNetwork(root=path, name=ds)
  elif ds == 'film':
    dataset = Actor(root=path, transform=T.NormalizeFeatures())
  elif ds == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name=ds, root=path, transform=T.ToSparseTensor())
    use_lcc = False  #  never need to calculate the lcc with ogb datasets
  elif ds in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55"]:#, "genius"]: #need to verify settings to match LINKX paper
    if opt['data_feat_norm']:
      dataset = LINKXDataset(root=path, name=ds, transform=T.NormalizeFeatures())
    else:
      dataset = LINKXDataset(root=path, name=ds)
    use_lcc = False
  elif ds in ["genius"]:
    if opt['data_feat_norm']:
      dataset = Genius(root=path, name=ds, transform=T.NormalizeFeatures())
    else:
      dataset = Genius(root=path, name=ds)
    use_lcc = False
  elif ds == 'arxiv-year':
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path, transform=T.ToSparseTensor())
    nclass = 5
    label = even_quantile_labels(dataset.data.node_year.flatten().numpy(), nclass, verbose=False)
    dataset.data.y = torch.as_tensor(label).reshape(-1, 1)
    splits_lst = np.load(f'../linkx_splits/arxiv-year-splits.npy', allow_pickle=True)
    train_mask, val_mask, test_mask  = process_fixed_splits(splits_lst)
    dataset.data.train_mask = train_mask
    dataset.data.val_mask = val_mask
    dataset.data.test_mask = test_mask
    return dataset
  elif ds == 'snap-patents':
    dataset = load_snap_patents_mat(nclass=5)
    return dataset
  # elif ds in ["Twitch"]: #need to verify settings to match LINKX paper
  #   if opt['data_feat_norm']:
  #     dataset = Twitch(root=path, name=ds, transform=T.NormalizeFeatures())
  #   else:
  #     dataset = Twitch(root=path, name=ds)
  #   use_lcc = False
  elif ds == 'Karate':
    dataset = KarateClub()
    dataset.data.val_mask = ~dataset.data.train_mask
    dataset.data.test_mask = ~dataset.data.train_mask
    use_lcc = False
  elif ds == 'syn_cora':
    dataset = get_pyg_syn_cora(data_dir, opt, rep=1)
    use_lcc = False
  elif ds == "SBM":
    dataset = get_SBM(data_dir, opt)
    dataset.data = set_train_val_test_split(12345, dataset.data, num_development=int(opt['sbm_ng']*opt['sbm_n']*0.75), num_per_class=int(opt['sbm_ng']*0.5))
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
  # if ds in ['ogbn-arxiv','arxiv-year']:
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


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1):
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]

  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

  data.train_mask = get_mask(train_idx)
  data.val_mask = get_mask(val_idx)
  data.test_mask = get_mask(test_idx)

  return data


#this code is taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/dataset.py#L190
#used to convert ogbn-arxiv labels into large hetero version

def even_quantile_labels(vals, nclasses, verbose=True):
  """ partitions vals into nclasses by a quantile based split,
  where the first class is less than the 1/nclasses quantile,
  second class is less than the 2/nclasses quantile, and so on

  vals is np array
  returns an np array of int class labels
  """
  label = -1 * np.ones(vals.shape[0], dtype=np.int)
  interval_lst = []
  lower = -np.inf
  for k in range(nclasses - 1):
    upper = np.nanquantile(vals, (k + 1) / nclasses)
    interval_lst.append((lower, upper))
    inds = (vals >= lower) * (vals < upper)
    label[inds] = k
    lower = upper
  label[vals >= lower] = nclasses - 1
  interval_lst.append((lower, np.inf))
  if verbose:
    print('Class Label Intervals:')
    for class_idx, interval in enumerate(interval_lst):
      print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
  return label

#adapting
#https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L221
#load splits from here https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
def process_fixed_splits(splits_lst): #, sub_dataset):
  train_list, val_list, test_list = [], [], []
  for i in range(len(splits_lst)):
    train_list.append(torch.as_tensor(splits_lst[i]['train']).unsqueeze(-1))
    val_list.append(torch.as_tensor(splits_lst[i]['valid']).unsqueeze(-1))
    test_list.append(torch.as_tensor(splits_lst[i]['test']).unsqueeze(-1))
  train = torch.cat(train_list, dim=1)
  val = torch.cat(val_list, dim=1)
  test = torch.cat(test_list, dim=1)
  return train, val, test

#adapting - https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/dataset.py#L257
def load_snap_patents_mat(nclass=5):
  dataset_drive_url = {'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia'}
  splits_drive_url = {'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N'}

  if not path.exists(f'{DATA_PATH}/snap_patents.mat'):
    p = dataset_drive_url['snap-patents']
    print(f"Snap patents url: {p}")
    gdown.download(id=dataset_drive_url['snap-patents'], \
                   output=f'{DATA_PATH}/snap_patents.mat', quiet=False)

  fulldata = scipy.io.loadmat(f'{DATA_PATH}/snap_patents.mat')
  num_classes = 5
  #get data
  edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
  node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
  num_nodes = int(fulldata['num_nodes'])
  years = fulldata['years'].flatten()
  label = even_quantile_labels(years, nclass, verbose=False)
  label = torch.tensor(label, dtype=torch.long)
  #get splits
  name = 'snap-patents'
  if not os.path.exists(f'../linkx_splits/snap-patents-splits.npy'):
    assert name in splits_drive_url.keys()
    gdown.download(
      id=splits_drive_url[name], \
      output=f'../linkx_splits/{name}-splits.npy', quiet=False)
  splits_lst = np.load(f'../linkx_splits/{name}-splits.npy', allow_pickle=True)
  train, val, test = process_fixed_splits(splits_lst)
  data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes,
                train_mask=train, val_mask=val, test_mask=test)

  dataset = DummyDataset(data, num_classes)

  return dataset


if __name__ == '__main__':
  # example for heterophilic datasets
  # from heterophilic import get_fixed_splits
  # opt = {'dataset': 'Cora', 'device': 'cpu'}
  # dataset = get_dataset(opt)
  # for fold in range(10):
  #   data = dataset[0]
  #   data = get_fixed_splits(data, opt['dataset'], fold)
  #   data = data.to(opt['device'])

  #calc ogbn-arxiv homoph
  from torch_geometric.utils import homophily
  opt = {'dataset': 'Cora', 'device': 'cpu', 'not_lcc': True, 'rewiring': None, 'geom_gcn_splits': False}
  dataset = get_dataset(opt, '../data', opt['not_lcc'])
  data = dataset.data
  data.edge_index = to_undirected(data.edge_index)
  label_homophil = homophily(edge_index=data.edge_index, y=data.y)
  print(f"label_homophil {label_homophil}")

