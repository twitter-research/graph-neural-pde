from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os.path as osp
import numpy as np
import pandas as pd
from torch_geometric.utils import degree, homophily
import torch
from graph_rewiring import dirichlet_energy

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

def syn_cora_analysis(path="../data"):
  ths = ['0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']
  df_list = []
  for targ_hom in ths:
    for rep in range(3):
      dataset = CustomDataset(root=f"{path}/syn-cora", name=f"h{str(targ_hom)}-r{str(rep+1)}", setting="gcn", seed=None)
      pyg_dataset = Dpr2Pyg(dataset)
      data = pyg_dataset.data
      num_nodes = data.num_nodes
      num_edges = data.edge_index.shape[1]
      num_classes = data.y.max() + 1
      num_features = data.num_features
      degrees = degree(data.edge_index[0], num_nodes)
      degree_range = [degrees.min().cpu().detach().numpy(), degrees.max().cpu().detach().numpy()]
      av_degree = num_edges / num_nodes
      density = num_edges / num_nodes**2
      graph_edge_homophily = homophily(edge_index=data.edge_index, y=data.y, method='edge')
      graph_node_homophily = homophily(edge_index=data.edge_index, y=data.y, method='node')
      #label dirichlet
      de = dirichlet_energy(data.edge_index, data.edge_weight, num_nodes, data.y.unsqueeze(-1))
      #spectral stuff...
      edge_categories = get_edge_cat(data.edge_index, data.y, num_classes)

      row = [num_nodes, num_edges, num_classes, num_features, degree_range, av_degree, density, graph_edge_homophily, graph_node_homophily, de, edge_categories]
      np_row = []
      for item in row:
        try:
          np_row.append(np.round(item.cpu().detach().numpy(), 4))
        except:
          np_row.append(np.round(item, 4))

      df_list.append(np_row)

  df_cols = ["num_nodes", "num_edges", "num_classes", "num_features", "degree_range", "av_degree",
             "density", "graph_edge_homophily", "graph_node_homophily", "label_dirichlet", "edge_categories"]

  # todo initialise spectrum distribution proportional to graph homophily

  df = pd.DataFrame(df_list, columns=df_cols)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_colwidth', -1)
  print(df)

if __name__ == "__main__":
  # dataset = CustomDataset(root="../data/syn-cora", name="h0.00-r1", setting="gcn", seed=15)
  # adj = dataset.adj  # Access adjacency matrix
  # features = dataset.features  # Access node features
  # pyg_dataset = Dpr2Pyg(dataset)
  # ei = pyg_dataset.data.edge_index

  syn_cora_analysis()

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