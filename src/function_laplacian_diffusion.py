import torch
from torch import nn
import torch_sparse
from base_classes import ODEFunc

from base_classes import ODEFunc


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, edge_index, edge_attr, device):
    super(LaplacianODEFunc, self).__init__(opt, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    # self.alpha_sc = nn.Parameter(torch.ones(1))
    # self.beta_sc = nn.Parameter(torch.ones(1))
    # self.register_buffer('alpha_sc',torch.ones(1))
    # self.register_buffer('beta_sc',torch.zeros(1))

  def sparse_multiply(self, x):
    if self.opt['block'] == 'attention':  # adj is a multihead attention
      # mean_attention = self.attention_weights.mean(dim=1)
      mean_attention = self.attention_weights
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] == 'mixed':  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if self.opt['alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    if self.opt['simple']:
      f = alpha * (ax - x)
    else:
      f = alpha * (ax - x) + self.beta_train * self.x0
    return f
