"""
Code taken from https://github.com/jianhao2016/GPRGNN/blob/master/src/dataset_utils.py
"""

import torch
import numpy as np
import os.path as osp

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops

from utils import ROOT_DIR

class Actor(InMemoryDataset):
  r"""The actor-only induced subgraph of the film-director-actor-writer
  network used in the
  `"Geom-GCN: Geometric Graph Convolutional Networks"
  <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
  Each node corresponds to an actor, and the edge between two nodes denotes
  co-occurrence on the same Wikipedia page.
  Node features correspond to some keywords in the Wikipedia pages.
  The task is to classify the nodes into five categories in term of words of
  actor's Wikipedia.

  Args:
      root (string): Root directory where the dataset should be saved.
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)
  """

  url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

  def __init__(self, root: str, transform: Optional[Callable] = None,
               pre_transform: Optional[Callable] = None):
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self) -> List[str]:
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
            ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

  @property
  def processed_file_names(self) -> str:
    return 'data.pt'

  def download(self):
    for f in self.raw_file_names[:2]:
      download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
    for f in self.raw_file_names[2:]:
      download_url(f'{self.url}/splits/{f}', self.raw_dir)

  def process(self):

    with open(self.raw_paths[0], 'r') as f:
      data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

      rows, cols = [], []
      for n_id, col, _ in data:
        col = [int(x) for x in col.split(',')]
        rows += [int(n_id)] * len(col)
        cols += col
      x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
      x = x.to_dense()

      y = torch.empty(len(data), dtype=torch.long)
      for n_id, _, label in data:
        y[int(n_id)] = int(label)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
      edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
      # Remove self-loops
      edge_index, _ = remove_self_loops(edge_index)
      # Make the graph undirected
      edge_index = to_undirected(edge_index)
      edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    train_masks, val_masks, test_masks = [], [], []
    for f in self.raw_paths[2:]:
      tmp = np.load(f)
      train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
      val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
      test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                val_mask=val_mask, test_mask=test_mask)
    data = data if self.pre_transform is None else self.pre_transform(data)
    torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
  r"""The Wikipedia networks introduced in the
  `"Multi-scale Attributed Node Embedding"
  <https://arxiv.org/abs/1909.13021>`_ paper.
  Nodes represent web pages and edges represent hyperlinks between them.
  Node features represent several informative nouns in the Wikipedia pages.
  The task is to predict the average daily traffic of the web page.

  Args:
      root (string): Root directory where the dataset should be saved.
      name (string): The name of the dataset (:obj:`"chameleon"`,
          :obj:`"crocodile"`, :obj:`"squirrel"`).
      geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
          pre-processed data as introduced in the `"Geom-GCN: Geometric
          Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
          in which the average monthly traffic of the web page is converted
          into five categories to predict.
          If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
          available.
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)

  """

  def __init__(self, root: str, name: str,
               transform: Optional[Callable] = None,
               pre_transform: Optional[Callable] = None):
    self.name = name.lower()
    assert self.name in ['chameleon', 'squirrel']
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self) -> str:
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self) -> str:
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self) -> Union[str, List[str]]:
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

  @property
  def processed_file_names(self) -> str:
    return 'data.pt'

  def download(self):
    pass

  def process(self):
    with open(self.raw_paths[0], 'r') as f:
      data = f.read().split('\n')[1:-1]
    x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
    x = torch.tensor(x, dtype=torch.float)
    y = [int(r.split('\t')[2]) for r in data]
    y = torch.tensor(y, dtype=torch.long)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
    edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)
    # Make the graph undirected
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)

    if self.pre_transform is not None:
      data = self.pre_transform(data)

    torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
  r"""The WebKB datasets used in the
  `"Geom-GCN: Geometric Graph Convolutional Networks"
  <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
  Nodes represent web pages and edges represent hyperlinks between them.
  Node features are the bag-of-words representation of web pages.
  The task is to classify the nodes into one of the five categories, student,
  project, course, staff, and faculty.
  Args:
      root (string): Root directory where the dataset should be saved.
      name (string): The name of the dataset (:obj:`"Cornell"`,
          :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
      transform (callable, optional): A function/transform that takes in an
          :obj:`torch_geometric.data.Data` object and returns a transformed
          version. The data object will be transformed before every access.
          (default: :obj:`None`)
      pre_transform (callable, optional): A function/transform that takes in
          an :obj:`torch_geometric.data.Data` object and returns a
          transformed version. The data object will be transformed before
          being saved to disk. (default: :obj:`None`)
  """

  url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
         'master/new_data')

  def __init__(self, root, name, transform=None, pre_transform=None):
    self.name = name.lower()
    assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

    super(WebKB, self).__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_dir(self):
    return osp.join(self.root, self.name, 'raw')

  @property
  def processed_dir(self):
    return osp.join(self.root, self.name, 'processed')

  @property
  def raw_file_names(self):
    return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

  @property
  def processed_file_names(self):
    return 'data.pt'

  def download(self):
    for name in self.raw_file_names:
      download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

  def process(self):
    with open(self.raw_paths[0], 'r') as f:
      data = f.read().split('\n')[1:-1]
      x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
      x = torch.tensor(x, dtype=torch.float32)

      y = [int(r.split('\t')[2]) for r in data]
      y = torch.tensor(y, dtype=torch.long)

    with open(self.raw_paths[1], 'r') as f:
      data = f.read().split('\n')[1:-1]
      data = [[int(v) for v in r.split('\t')] for r in data]
      edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
      edge_index = to_undirected(edge_index)
      # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
      edge_index, _ = remove_self_loops(edge_index)
      edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data = data if self.pre_transform is None else self.pre_transform(data)
    torch.save(self.collate([data]), self.processed_paths[0])

  def __repr__(self):
    return '{}()'.format(self.name)


def index_to_mask(index, size):
  mask = torch.zeros(size, dtype=torch.bool, device=index.device)
  mask[index] = 1
  return mask


def generate_random_splits(data, num_classes, train_rate=0.6, val_rate=0.2):
  """Generates training, validation and testing masks for node classification tasks."""
  percls_trn = int(round(train_rate * len(data.y) / num_classes))
  val_lb = int(round(val_rate * len(data.y)))

  indices = []
  for i in range(num_classes):
    index = (data.y == i).nonzero().view(-1)
    index = index[torch.randperm(index.size(0))]
    indices.append(index)

  train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

  rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
  rest_index = rest_index[torch.randperm(rest_index.size(0))]

  data.train_mask = index_to_mask(train_index, size=data.num_nodes)
  data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
  data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)

  return data


def get_fixed_splits(data, dataset_name, seed):
  #todo just added this to test sheaf experiments. Remove when done
  if dataset_name == 'gg_cora':
    dataset_name = 'cora'
  with np.load(f'{ROOT_DIR}/src/splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  # Remove the nodes that the label vectors are all zeros, they aren't assigned to any class
  if dataset_name in {'cora', 'citeseer', 'pubmed'}:
    data.train_mask[data.non_valid_samples] = False
    data.test_mask[data.non_valid_samples] = False
    data.val_mask[data.non_valid_samples] = False
    print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
    print("Nodes", data.x.size(0))
    print("Non valid", len(data.non_valid_samples))
  else:
    assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)

  return data
