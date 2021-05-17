import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianRewireODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianRewireODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    elif self.opt['function'] in ['laplacian_rewire']:
      self.add_khop_edges()
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def add_khop_edges(self, k=2):
    n = self.num_nodes
    #do k_hop
    for i in range(k):
      new_edges, new_weights =\
        torch_sparse.spspmm(self.odefunc.edge_index, self.odefunc.edge_weight,
                          self.odefunc.edge_index, self.odefunc.edge_weight, n, n, n, coalesced=False)
    self.edge_weight = 0.5 * self.edge_weight + 0.5 * new_weights
    cat = torch.cat([self.data_edge_index, new_edges], dim=1)
    self.edge_index = torch.unique(cat, sorted=False, return_inverse=False,
                              return_counts=False, dim=0)
    #threshold
    #normalise
  # self.odefunc.edge_index, self.odefunc.edge_weight =
  # get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  # num_nodes = maybe_num_nodes(edge_index, num_nodes)

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
