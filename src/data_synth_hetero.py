import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import WeightedRandomSampler


def hetro_edge_addition(data, opt):
  '''
  Algorithm 1: Heterophilous Edge Addition
  input : G = {V, E}, K, {Dc}^|C|_−1, c=0 and {Vc}^|C|−1_c=0
  output: G` = {V, E}^0
  Initialize G` = {V, E}, k = 1 ;

  while 1 ≤ k ≤ K do
    Sample node i ∼ Uniform(V);
    Obtain the label, yi of node i;
    Sample a label c ∼ Dyi;
    Sample node j ∼ Uniform(Vc);
  Update edge set E = E ∪ {(i, j)};
  k ← k + 1;
  return G = {V, E}
  '''
  edge_index = data.edge_index
  y = data.y
  num_class = y.max() + 1
  target_homoph = opt['target_homoph']

  Nc = []
  for c in range(num_class):
    Nc.append(torch.sum(y==c)) #work in dst (index j) nodes

  Dc = {}
  for c in range(num_class):
    class_mask = y[edge_index[1]] == c
    class_c_src = edge_index[0][class_mask] #all the source nodes for dst nodes of class c
    temp_Nc = []
    for k in range(num_class):
      temp_Nc.append(torch.sum(y[class_c_src] == k)/Nc[k])
    Dc = {c: temp_Nc for c in range(num_class)}

  torch.rand()

  # Let there be 9 samples and 1 sample in class 0 and 1 respectively
  class_counts = [9.0, 1.0]
  num_samples = sum(class_counts)
  labels = [0, 0, ..., 0, 1]  # corresponding labels of samples

  class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
  weights = [class_weights[labels[i]] for i in range(int(num_samples))]
  sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples), replacement=True)

  # puv for a newly added node u in
  # class i to connect with an existing node v in class j is proportional to both the class compatibility Hij
  # between class i and j, and the degree dv of the existing node v

  return data