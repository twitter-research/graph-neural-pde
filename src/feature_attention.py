import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException
from base_classes import ODEFunc


class TransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(TransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = int(opt['heads'])

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
    self.sm = nn.Softmax(dim=-1)

    self.Wout = nn.Linear(self.d_k, in_features)
    self.init_weights(self.Wout)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      # nn.init.xavier_uniform_(m.weight, gain=1.414)
      # m.bias.data.fill_(0.01)
      nn.init.constant_(m.weight, 1e-5)

  def multiply_attention(self, x, attention, v=None):
    # todo would be nice if this was more efficient
    if False:
      pass
      # vx = torch.mean(torch.stack(
      #   [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
      #    range(self.opt['heads'])], dim=0),
      #   dim=0)
      # ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=0)
      ax = torch.mm(x, mean_attention)
    return ax

  def forward(self, x):
    # create matrices of dims (n_features, attention_dim)
    q = self.Q(x.T)
    k = self.K(x.T)
    v = self.V(x.T)

    # perform linear operation and split into h heads
    # to makes tensors of dim (n_features, n_heads, attention_dim / n_heads = d_k)
    q = q.view(-1, self.h, self.d_k)
    k = k.view(-1, self.h, self.d_k)
    v = v.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_heads, n_features, attention_dim]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    prods = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

    attention = self.sm(prods)
    x = self.multiply_attention(x, attention, v)
    return x

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
         'attention_norm_idx': 0, 'add_source': False,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }
  # dataset = get_dataset(opt, '../data', False)
  opt['heads'] = 2
  # todo the input dim here is the number of nodes, so need access to this
  func = TransAttentionLayer(5, 8, opt, device)
  x = torch.ones((5, 6))
  new_x = func(x)
  opt['heads'] = 1
  x = torch.rand((5, 6))
  func = TransAttentionLayer(5, 8, opt, device)
  new_x = func(x)
