"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException, squareplus
from base_classes import ODEFunc
from function_transformer_attention import SpGraphTransAttentionLayer


class ODEFuncGreed(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncGreed, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    # todo write the correct form on the greed attention
    self.multihead_att_layer = SpGraphGreedAttentionLayer(in_features, out_features, opt,
                                                          device, edge_weights=self.edge_weight).to(device)

    def multiply_attention(self, x, attention, v=None):
      # todo would be nice if this was more efficient
      if self.opt['mix_features']:
        vx = torch.mean(torch.stack(
          [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
           range(self.opt['heads'])], dim=0),
          dim=0)
        ax = self.multihead_att_layer.Wout(vx)
      else:
        mean_attention = attention.mean(dim=1)
        ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
      return ax

    def forward(self, t, x):  # t is needed when called by the integrator
      if self.nfe > self.opt["max_nfe"]:
        raise MaxNFEException

      self.nfe += 1
      attention, values = self.multihead_att_layer(x, self.edge_index)
      ax = self.multiply_attention(x, attention, values)

      if not self.opt['no_alpha_sigmoid']:
        alpha = torch.sigmoid(self.alpha_train)
      else:
        alpha = self.alpha_train
      f = alpha * (ax - x)
      if self.opt['add_source']:
        f = f + self.beta_train * self.x0
      return f

    def __repr__(self):
      return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphGreedAttentionLayer(nn.Module):
  """
  Probably generating the tau from GREED
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphGreedAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])
    self.edge_weights = edge_weights

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % self.h == 0, "Number of heads ({}) must be a factor of the dimension size ({})".format(
      self.h, self.attention_dim)
    self.d_k = self.attention_dim // self.h

    self.Q = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.Q)

    self.V = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.V)

    self.K = nn.Linear(in_features, self.attention_dim)
    self.init_weights(self.K)

    self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def forward(self, x, edge):

    q = self.Q(x)
    k = self.K(x)
    v = self.V(x)

    # perform linear operation and split into h heads

    k = k.view(-1, self.h, self.d_k)
    q = q.view(-1, self.h, self.d_k)
    v = v.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_nodes, attention_dim, n_heads]

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    src = q[edge[0, :], :, :]
    dst_k = k[edge[1, :], :, :]

    # todo check this with Francesco
    tau = torch.tanh(src + dst_k)
    return tau

  def get_gamma(self, epsilon=10e-6):
    """
    get gamma_ij = |Z_ij|^2
    get eta(i) = \prod_{j \in N(i)} gamma_{ij}}^{1/d_i}
    """

  def get_tau(f, K, Q):
    """

    """
    return torch.tanh()

  def get_Z(self, x, edge):
    """
    W (tau(f_i,f_j)f_i/\sqrt{d_i} - tau(f_j,f_i)f_j/\sqrt(d_j))
    """
    src = x[edge[0, :]]
    dst = x[edge[1, :]]

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



