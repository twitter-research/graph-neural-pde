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
  # with np.load(f'{ROOT_DIR}/splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
  with np.load(f'{ROOT_DIR}/splits/{dataset_name.lower()}_split_0.6_0.2_{seed}.npz') as splits_file:
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  if dataset_name in {'Cora', 'Citeseer'}:
    process_geom_masks(data, dataset_name)

  # Remove the nodes that the label vectors are all zeros, they aren't assigned to any class
  if dataset_name in {'cora', 'citeseer', 'pubmed'}:
    data.train_mask[data.non_valid_samples] = False
    data.test_mask[data.non_valid_samples] = False
    data.val_mask[data.non_valid_samples] = False
    print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
    print("Nodes", data.x.size(0))
    print("Non valid", len(data.non_valid_samples))
  else:
    print(data.train_mask.shape)
    print(torch.count_nonzero(data.train_mask))
    print(data.val_mask.shape)
    print(torch.count_nonzero(data.val_mask))
    print(data.test_mask.shape)
    print(torch.count_nonzero(data.test_mask))
    print(data.x.shape)
    # assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)

  return data


def process_geom_masks(data, dataset_name):
  '''eg for Cora with LCC and geom-gcn splits the train/val/test masks are 2708 vectors that sum to 2485. the LCC is 2485
  this function assumes:
  for Cora - load the lcc from the data loader and just need to reduce the geom-gcn splits to lcc number
  for Citeseer - load the full ds, and both reduce the geom-gcn splits and x to the number of non zero mask in that particular split
  for Pubmed - excluded as: mask size == split size = x size = lcc size = dataset size
  '''
  tot_masks = data.train_mask.int() + data.val_mask.int() + data.test_mask.int()
  geom_mask = tot_masks > 0
  data.train_mask = data.train_mask[geom_mask]
  data.val_mask = data.val_mask[geom_mask]
  data.test_mask = data.test_mask[geom_mask]

  # fix for Citeseer because splits uses a compbination of LCC and not LCC
  if dataset_name == "Citeseer" and geom_mask.sum() < data.x.shape[0]:
      lcc = get_largest_connected_component(data)
      x_new = data.x[lcc]
      y_new = data.y[lcc]
      row, col = data.edge_index.numpy()
      edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
      edges = remap_edges(edges, get_node_mapper(lcc))
      data.x = x_new
      data.y = y_new
      data.edge_index = torch.LongTensor(edges)



###copy from data.py to stop circular refs
def get_component(data, start: int = 0) -> set:
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes

def get_largest_connected_component(data) -> np.ndarray:
  remaining_nodes = set(range(data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(data, start)
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


#code taken from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/linkx_dataset.html#LINKXDataset
# as environment version PyG 2.0.3 does not contain genius dataset
import torch.nn.functional as F
# class LINKXDataset(InMemoryDataset):
class Genius(InMemoryDataset):
    r"""A variety of non-homophilous graph datasets from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

    .. note::
        Some of the datasets provided in :class:`LINKXDataset` are from other
        sources, but have been updated with new features and/or labels.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"penn94"`,
            :obj:`"reed98"`, :obj:`"amherst41"`, :obj:`"cornell5"`,
            :obj:`"johnshopkins55"`, :obj:`"genius"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data')
    gdrive_url = 'https://drive.google.com/uc?confirm=t&'

    facebook_datasets = [
      'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
    ]

    datasets = {
      # 'penn94': {
      #   'data.mat': f'{github_url}/facebook100/Penn94.mat'
      # },
      # 'reed98': {
      #   'data.mat': f'{github_url}/facebook100/Reed98.mat'
      # },
      # 'amherst41': {
      #   'data.mat': f'{github_url}/facebook100/Amherst41.mat',
      # },
      # 'cornell5': {
      #   'data.mat': f'{github_url}/facebook100/Cornell5.mat'
      # },
      # 'johnshopkins55': {
      #   'data.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
      # },
      'genius': {
        # 'data.mat': f'{github_url}/genius.mat'    ###<- editted
        'genius.mat': f'{github_url}/genius.mat'
      },
      # 'wiki': {
      #   'wiki_views2M.pt':
      #     f'{gdrive_url}id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
      #   'wiki_edges2M.pt':
      #     f'{gdrive_url}id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
      #   'wiki_features2M.pt':
      #     f'{gdrive_url}id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
      # }
    }

    splits = {
      'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
      self.name = name.lower()
      assert self.name in self.datasets.keys()
      super().__init__(root, transform, pre_transform)
      self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
      return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
      return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
      names = list(self.datasets[self.name].keys())
      if self.name in self.splits:
        names += [self.splits[self.name].split('/')[-1]]
      return names

    @property
    def processed_file_names(self) -> str:
      return 'data.pt'

    def download(self):
      for filename, path in self.datasets[self.name].items():
        download_url(path, self.raw_dir)#, filename=filename)
      if self.name in self.splits:
        download_url(self.splits[self.name], self.raw_dir)

    def _process_wiki(self):

      paths = {x.split('/')[-1]: x for x in self.raw_paths}
      x = torch.load(paths['wiki_features2M.pt'])
      edge_index = torch.load(paths['wiki_edges2M.pt']).t().contiguous()
      y = torch.load(paths['wiki_views2M.pt'])

      return Data(x=x, edge_index=edge_index, y=y)

    def _process_facebook(self):
      from scipy.io import loadmat

      mat = loadmat(self.raw_paths[0])

      A = mat['A'].tocsr().tocoo()
      row = torch.from_numpy(A.row).to(torch.long)
      col = torch.from_numpy(A.col).to(torch.long)
      edge_index = torch.stack([row, col], dim=0)

      metadata = torch.from_numpy(mat['local_info'].astype('int64'))

      xs = []
      y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
      x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
      for i in range(x.size(1)):
        _, out = x[:, i].unique(return_inverse=True)
        xs.append(F.one_hot(out).to(torch.float))
      x = torch.cat(xs, dim=-1)

      data = Data(x=x, edge_index=edge_index, y=y)

      if self.name in self.splits:
        splits = np.load(self.raw_paths[1], allow_pickle=True)
        sizes = (data.num_nodes, len(splits))
        data.train_mask = torch.zeros(sizes, dtype=torch.bool)
        data.val_mask = torch.zeros(sizes, dtype=torch.bool)
        data.test_mask = torch.zeros(sizes, dtype=torch.bool)

        for i, split in enumerate(splits):
          data.train_mask[:, i][torch.tensor(split['train'])] = True
          data.val_mask[:, i][torch.tensor(split['valid'])] = True
          data.test_mask[:, i][torch.tensor(split['test'])] = True

      return data

    def _process_genius(self):
      from scipy.io import loadmat

      mat = loadmat(self.raw_paths[0])
      edge_index = torch.from_numpy(mat['edge_index']).to(torch.long)
      x = torch.from_numpy(mat['node_feat']).to(torch.float)
      y = torch.from_numpy(mat['label']).squeeze().to(torch.long)

      return Data(x=x, edge_index=edge_index, y=y)

    def process(self):
      if self.name in self.facebook_datasets:
        data = self._process_facebook()
      elif self.name == 'genius':
        data = self._process_genius()
      elif self.name == 'wiki':
        data = self._process_wiki()
      else:
        raise NotImplementedError(
          f"chosen dataset '{self.name}' is not implemented")

      if self.pre_transform is not None:
        data = self.pre_transform(data)

      torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
      return f'{self.name.capitalize()}({len(self)})'