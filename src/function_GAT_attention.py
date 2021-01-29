import torch
from torch import nn
from torch_geometric.utils import softmax
import torch_sparse
from torch_geometric.utils.loop import add_remaining_self_loops
from data import get_dataset
from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncAtt(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncAtt, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features, opt,
                                                     device).to(device)
    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

  def multiply_attention(self, x, attention, wx):
    if self.opt['mix_features']:
      wx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], wx.shape[0], wx.shape[0], wx) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = torch.mm(wx, self.multihead_att_layer.Wout)
    else:
      ax = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], x.shape[0], x.shape[0], x) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
    return ax

  def forward(self, t, x):  # t is needed when called by the integrator

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1

    attention, wx = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, wx)
    # todo would be nice if this was more efficient

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
    self.h = opt['heads']

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414)

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge):
    wx = torch.mm(x, self.W)  # h: N x out
    h = wx.view(-1, self.h, self.d_k)
    h = h.transpose(1, 2)

    # Self-attention on the nodes - Shared attention mechanism
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
      self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention, wx

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = {'dataset': 'Cora', 'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'beta_dim': 'vc', 'heads': 2, 'K': 10, 'attention_norm_idx': 0,
         'add_source':False, 'alpha_dim': 'sc', 'beta_dim': 'vc', 'max_nfe':1000, 'mix_features': False}
  dataset = get_dataset(opt, '../data', False)
  t = 1
  func = ODEFuncAtt(dataset.data.num_features, 6, opt, dataset.data, device)
  out = func(t, dataset.data.x)
