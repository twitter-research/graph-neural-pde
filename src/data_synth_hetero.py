import matplotlib.pyplot as plt
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os.path as osp
import numpy as np
import pandas as pd
from torch_geometric.utils import degree, homophily, from_networkx
import torch
from graph_rewiring import dirichlet_energy
from greed_params import default_params
import networkx as nx
import matplotlib.pyplot as plt
from utils import DummyDataset

#adopted from https://github.com/GemsLab/H2GCN/tree/master/npz-datasets
class CustomDataset(Dataset):
  def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
    '''
    Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
    '''
    self.name = name.lower()
    self.setting = setting.lower()

    self.seed = seed
    self.url = None
    self.root = osp.expanduser(osp.normpath(root))
    self.data_folder = osp.join(root, self.name)
    self.data_filename = self.data_folder + '.npz'
    # Make sure dataset file exists
    assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!"
    self.require_mask = require_mask

    self.require_lcc = True if setting == 'nettack' else False
    self.adj, self.features, self.labels = self.load_data()
    self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
    if self.require_mask:
      self.get_mask()

  def get_adj(self):
    adj, features, labels = self.load_npz(self.data_filename)
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    if self.require_lcc:
      lcc = self.largest_connected_components(adj)

      # adj = adj[lcc][:, lcc]
      adj_row = adj[lcc]
      adj_csc = adj_row.tocsc()
      adj_col = adj_csc[:, lcc]
      adj = adj_col.tolil()

      features = features[lcc]
      labels = labels[lcc]
      assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels

  def get_train_val_test(self):
    if self.setting == "exist":
      with np.load(self.data_filename) as loader:
        idx_train = loader["idx_train"]
        idx_val = loader["idx_val"]
        idx_test = loader["idx_test"]
      return idx_train, idx_val, idx_test
    else:
      return super().get_train_val_test()

def get_pyg_syn_cora(path, opt, rep):
  dataset = CustomDataset(root=f"{path}/syn-cora", name=f"h{str(opt['target_homoph'])}-r{str(rep)}", setting="gcn", seed=None)
  pyg_dataset = Dpr2Pyg(dataset)
  return pyg_dataset

def get_SBM(path, opt):
  # opt = {"sbm_n": 3, "sbm_ng": 10, "sbm_pq": 2}
  opt['sbm_n'] = 3
  opt['sbm_ng'] = 10
  opt['sbm_pq'] = 2

  pq = opt['sbm_pq']
  N = opt['sbm_n']
  ng =  opt['sbm_ng']
  p = 1 - 1/(pq+1)
  q = 1/(pq+1)
  # sizes = [15, 15, 30]
  # probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
  sizes = N * [ng]
  probs = []
  for g in range(N):
    g_probs = []
    for n in range(N):
      if g == n:
        g_probs.append(p)
      else:
        g_probs.append(q)
    probs.append(g_probs)

  G = nx.stochastic_block_model(sizes, probs, seed=0)
  labels = [d['block'] for n,d in G.nodes(data=True)]
  pos = nx.spring_layout(G)
  nx.draw(G, pos, node_color=labels)
  plt.show()
  pyg_data = from_networkx(G)
  pyg_data.y = torch.tensor(labels, dtype=torch.long)
  pyg_data.x = torch.from_numpy(np.stack([pos[i]+[labels[i]] for i in range(N*ng)], axis=0)).type(torch.float)
  pyg_data.train_mask = torch.zeros(N*ng, dtype=torch.bool)
  pyg_data.test_mask = torch.zeros(N*ng, dtype=torch.bool)
  pyg_data.val_mask = torch.zeros(N*ng, dtype=torch.bool)

  pyg_dataset = DummyDataset(pyg_data, opt['sbm_n'])
  return pyg_dataset

def get_edge_cat(edge_index, y, num_classes):
  edges_cats = []
  class_list = []
  class_sublist = []
  for c in range(num_classes):
    label_mask = y == c
    torch.where(y==c)
    src_nodes = edge_index[1][y[edge_index[0]]==c]
    src_labels = y[src_nodes]
    bin_count = torch.bincount(src_labels, minlength=num_classes) / src_nodes.shape[0]
    edges_cats.append(np.round(bin_count.cpu().detach().numpy(),2))
  return edges_cats

if __name__ == "__main__":
  opt = {"sbm_n":3, "sbm_ng":10, "sbm_pq":1.0} # 0.3, "sbm_q": 0.5}
  get_SBM(path="../data/SBM/", opt=opt)


  # dataset = CustomDataset(root="../data/syn-cora", name="h0.00-r1", setting="gcn", seed=15)
  # adj = dataset.adj  # Access adjacency matrix
  # features = dataset.features  # Access node features
  # pyg_dataset = Dpr2Pyg(dataset)
  # ei = pyg_dataset.data.edge_index

# import torch
# from torch_geometric.data import Data, InMemoryDataset
# from torch.utils.data import WeightedRandomSampler

# def hetro_edge_addition(data, opt):
#   '''
#   Algorithm 1: Heterophilous Edge Addition
#   input : G = {V, E}, K, {Dc}^|C|_−1, c=0 and {Vc}^|C|−1_c=0
#   output: G` = {V, E}^0
#   Initialize G` = {V, E}, k = 1 ;
#
#   while 1 ≤ k ≤ K do
#     Sample node i ∼ Uniform(V);
#     Obtain the label, yi of node i;
#     Sample a label c ∼ Dyi;
#     Sample node j ∼ Uniform(Vc);
#   Update edge set E = E ∪ {(i, j)};
#   k ← k + 1;
#   return G = {V, E}
#   '''
#   edge_index = data.edge_index
#   y = data.y
#   num_class = y.max() + 1
#   target_homoph = opt['target_homoph']
#
#   Nc = []
#   for c in range(num_class):
#     Nc.append(torch.sum(y==c)) #work in dst (index j) nodes
#
#   Dc = {}
#   for c in range(num_class):
#     class_mask = y[edge_index[1]] == c
#     class_c_src = edge_index[0][class_mask] #all the source nodes for dst nodes of class c
#     temp_Nc = []
#     for k in range(num_class):
#       temp_Nc.append(torch.sum(y[class_c_src] == k)/Nc[k])
#     Dc = {c: temp_Nc for c in range(num_class)}
#
#   torch.rand()
#
#   # Let there be 9 samples and 1 sample in class 0 and 1 respectively
#   class_counts = [9.0, 1.0]
#   num_samples = sum(class_counts)
#   labels = [0, 0, ..., 0, 1]  # corresponding labels of samples
#
#   class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
#   weights = [class_weights[labels[i]] for i in range(int(num_samples))]
#   sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples), replacement=True)
#
#   # puv for a newly added node u in
#   # class i to connect with an existing node v in class j is proportional to both the class compatibility Hij
#   # between class i and j, and the degree dv of the existing node v
#
#   return data