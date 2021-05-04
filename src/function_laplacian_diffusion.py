import torch
from torch import nn
import torch_sparse
from torch.nn.functional import normalize

from base_classes import ODEFunc
from feature_attention import TransAttentionLayer

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.w_rs = nn.Parameter(torch.rand((opt['hidden_dim'], opt['hidden_dim'])))  # right stochastic W
    # self.w_rs = f.normalize(w, p=1, dim=-1)
    self.d = nn.Parameter(torch.ones(opt['hidden_dim']))
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.sm = torch.nn.Softmax(dim=1)
    self.feature_attention = TransAttentionLayer(data.num_nodes, opt['attention_dim'], opt, device, concat=True)

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      # ax = torch.mean(torch.stack(
      #   [torch_sparse.spmm(self.edge_index, self.attention_weights[:, idx], x.shape[0], x.shape[0], x) for idx in
      #    range(self.opt['heads'])], dim=0), dim=0)
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    self.nfe += 1
    if self.opt['mix_features']:
      # d = torch.clamp(self.d, min=0, max=1)  # enforce evalues in (0,1)
      # w = torch.mm(self.w * d, torch.t(self.w))
      # x = torch.mm(x, w)
      # w_rs = normalize(self.w_rs, p=1, dim=-1)
      # w_rs = self.sm(self.w_rs)
      # x = torch.mm(x, w_rs)
      x = self.feature_attention(x)
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train

    f = alpha * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
