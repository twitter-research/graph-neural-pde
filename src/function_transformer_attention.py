import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
import numpy as np
from data import get_dataset
from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncTransformerAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt, edge_index, edge_attr, device):
    super(ODEFuncTransformerAtt, self).__init__(opt, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(edge_index, edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = edge_index, edge_attr
    # self.alpha = nn.Parameter(torch.ones([data.num_nodes, 1]))
    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                          device).to(device)

  def multiply_attention(self, x, attention, v=None):
    # todo would be nice if this was more efficient
    if self.opt['mix_features']:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = self.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    attention, values = self.multihead_att_layer(x, self.edge_index)
    self.attention_weights = attention.mean(dim=1)
    ax = self.multiply_attention(x, attention, values)

    alpha = torch.sigmoid(self.alpha_train)
    # else:
    # alpha = self.alpha
    if self.opt['simple']:
      f = alpha * (ax - x)
    else:
      f = alpha * (ax - x) + self.beta_train * self.x0
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphTransAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True, edge_weights=None):
    super(SpGraphTransAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
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

    self.src_pow = nn.Parameter(torch.tensor(1.0))
    self.dst_pow =  nn.Parameter(torch.tensor(1.0))


  def init_weights(self, m):
    if type(m) == nn.Linear:
      nn.init.xavier_uniform_(m.weight)#, gain=1.414)
      #m.bias.data.fill_(0.01)
      # nn.init.constant_(m.weight, 1e-3)#e-5)


  def forward(self, x, edge):
    q = self.Q(x)
    k = self.K(x)
    v = self.V(x)

    # perform linear operation and split into h heads

    k = k.view(-1, self.h, self.d_k)
    q = q.view(-1, self.h, self.d_k)
    v = v.view(-1, self.h, self.d_k)

    # transpose to get dimensions [n_nodes, [attention_dim/n_heads], n_heads]

    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)

    src = q[edge[0, :], :, :]
    dst_k = k[edge[1, :], :, :]

    # S = torch.nn.Sigmoid()
    # src = S(src)
    # dst_k = S(dst_k)
    if self.opt['attention_type'] == "scaled_dot":
      prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
    elif self.opt['attention_type'] == "cosine_sim":
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)
    elif self.opt['attention_type'] == "cosine_power":
      prods = torch.sum(src * dst_k, dim=1) / (torch.pow(torch.linalg.norm(src,ord=2,dim=1)+1e-5, self.src_pow)
                                             *torch.pow(torch.linalg.norm(dst_k,ord=2,dim=1)+1e-5, self.dst_pow))
    elif self.opt['attention_type'] == "pearson":
      src_mu = torch.mean(src, dim=1, keepdim=True)
      dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
      src = src - src_mu
      dst_k = dst_k - dst_mu
      cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
      prods = cos(src, dst_k)

    elif self.opt['attention_type'] == "spearman":
      pass
    # https: // arxiv.org / pdf / 2002.08871.pdf
    # https: // github.com / google - research / fast - soft - sort /


    if self.opt['reweight_attention'] and self.edge_weights is not None:
      prods = prods * self.edge_weights.unsqueeze(dim=1)
    attention = softmax(prods, edge[self.opt['attention_norm_idx']])

    return attention, v


  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  dataset = get_dataset('Cora', '../data', False)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10, 'attention_norm_idx': 0, 'simple': True,
         'alpha_dim': 'sc', 'beta_dim': 'sc', 'max_nfe': 1000, 'mix_features': False
         }

  t = 1
  func = ODEFuncTransformerAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
