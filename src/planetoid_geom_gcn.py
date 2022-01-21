import torch
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import os.path as osp

from typing import Optional, Callable, List
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import coalesce
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops


class GeomGCNPlanetoid(InMemoryDataset):

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        data = self.get(0)
        self.data, self.slices = self.collate([data])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data = full_load_citation(self.name, self.raw_dir)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def full_load_citation(dataset_str, raw_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        path = osp.join(raw_dir, "ind.{}.{}".format(dataset_str, names[i]))
        with open(path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(osp.join(raw_dir, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    if len(test_idx_range_full) != len(test_idx_range):
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position, mark them
        # Follow H2GCN code
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        non_valid_samples = set(test_idx_range_full) - set(test_idx_range)
    else:
        non_valid_samples = set()
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    non_valid_samples = list(non_valid_samples.union(set(list(np.where(labels.sum(1) == 0)[0]))))
    labels = np.argmax(labels, axis=-1)

    features = features.todense()

    # Prepare in PyTorch Geometric Format
    sparse_mx = sp.coo_matrix(adj).astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    shape = torch.Size(sparse_mx.shape)
    edge_index, _ = coalesce(indices, None, shape[0], shape[1])

    # Remove self-loops
    edge_index, _ = remove_self_loops(edge_index)
    # Make the graph undirected
    edge_index = to_undirected(edge_index)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    non_valid_samples = torch.LongTensor(non_valid_samples)

    return Data(x=features, edge_index=edge_index, y=labels, num_node_features=features.size(1),
                non_valid_samples=non_valid_samples)
