import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import time
from pykeops.torch import LazyTensor
import numpy as np
from data import get_dataset


class ODEFuncDorseyAtt(MessagePassing):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncDorseyAtt, self).__init__()
    # print('using attention diffusion')
    self.opt = opt
    self.device = device
    self.adj = None

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr
    self.x0 = None
    self.nfe = 0
    self.alpha = nn.Parameter(torch.ones([data.num_nodes, 1]))
    self.att_layer = SpGraphAttentionLayer(in_features, out_features, opt,
                                           device).to(device)
    self.att_layers = nn.ModuleList(
      [SpGraphAttentionLayer(in_features, out_features, opt,
                             device) for dummy_i in range(opt['heads'])]).to(device)

  def forward(self, t, x):  # t is needed when called by the integrator
    self.nfe += 1
    attentions = [att(x, self.edge_index) for att in self.att_layers]
    ax = torch.mean(torch.stack(
      [torch_sparse.spmm(self.edge_index, attention, x.shape[0], x.shape[0], x) for attention in attentions], dim=0),
      dim=0)
    f = ax - x
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt

    self.WQ = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
    nn.init.xavier_normal_(self.WQ.data, gain=1.414)

    self.WV = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
    nn.init.xavier_normal_(self.WV.data, gain=1.414)

    self.WK = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
    nn.init.xavier_normal_(self.WK.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def assign_to_centroid(self, x):
    pass

  def KMeans(self, x, K=10, Niter=10, verbose=True):
    #  todo this is not differentiable so need some soft differentiable mapping, but this must also reduce the
    #  dimensionality
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

      c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
      D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
      cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

      # Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
      Ncl = torch.bincount(cl)  # Class weights
      for d in range(D):  # Compute the cluster centroids with torch.bincount:
        c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
      print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
      print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
        Niter, end - start, Niter, (end - start) / Niter))

    return cl, c

  def get_attention(self, centroids):
    q = torch.mm(centroids, self.WQ)
    v = torch.mm(centroids, self.WV)
    prods = torch.mm(q, v.t()) / np.sqrt(self.WK.shape[1])
    attention = torch.softmax(prods, dim=1)
    return attention

  def forward(self, x, edge):
    alloc, centroids = self.KMeans(x)
    centroid_attention = self.get_attention(centroids)
    src_centroid = alloc[edge[0, :]]
    dst_centroid = alloc[edge[1, :]]
    edge_attention = centroid_attention[src_centroid, dst_centroid]
    attention = softmax(edge_attention, edge[self.opt['attention_norm_idx']])
    return attention

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads':1, 'K':10, 'attention_norm_idx':0}
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncDorseyAtt(dataset.data.num_features, 2, opt, dataset.data, device)
  out = func(t, dataset.data.x)
