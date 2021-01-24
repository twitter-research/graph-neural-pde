import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.utils import to_undirected


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class OGBFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(OGBFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.adj = None

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    # ax = matmul(self.adj, x)
    ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    alpha = torch.sigmoid(self.alpha_train)
    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
