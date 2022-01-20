"""
utility functions
"""
import os

import scipy
from scipy.stats import sem
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from torch_geometric.nn.conv.gcn_conv import gcn_norm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class MaxNFEException(Exception): pass


def rms_norm(tensor):
  return tensor.pow(2).mean().sqrt()


def make_norm(state):
  if isinstance(state, tuple):
    state = state[0]
  state_size = state.numel()

  def norm(aug_state):
    y = aug_state[1:1 + state_size]
    adj_y = aug_state[1 + state_size:1 + 2 * state_size]
    return max(rms_norm(y), rms_norm(adj_y))

  return norm


def print_model_params(model):
  total_num_params = 0
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      total_num_params += param.numel()
  print("Model has a total of {} params".format(total_num_params))


def adjust_learning_rate(optimizer, lr, epoch, burnin=50):
  if epoch <= burnin:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr * epoch / burnin


def gcn_norm_fill_val(edge_index, edge_weight=None, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not int(fill_value) == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-0.5)
  deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
  return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def coo2tensor(coo, device=None):
  indices = np.vstack((coo.row, coo.col))
  i = torch.LongTensor(indices)
  values = coo.data
  v = torch.FloatTensor(values)
  shape = coo.shape
  print('adjacency matrix generated with shape {}'.format(shape))
  # test
  return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)


def get_sym_adj(data, opt, improved=False):
  edge_index, edge_weight = gcn_norm(  # yapf: disable
    data.edge_index, data.edge_attr, data.num_nodes,
    improved, opt['self_loop_weight'] > 0, dtype=data.x.dtype)
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  return coo2tensor(coo)


def get_rw_adj_old(data, opt):
  if opt['self_loop_weight'] > 0:
    edge_index, edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                       fill_value=opt['self_loop_weight'])
  else:
    edge_index, edge_weight = data.edge_index, data.edge_attr
  coo = to_scipy_sparse_matrix(edge_index, edge_weight)
  normed_csc = normalize(coo, norm='l1', axis=0)
  return coo2tensor(normed_csc.tocoo())


def get_rw_adj(edge_index, edge_weight=None, norm_dim=1, fill_value=0., num_nodes=None, dtype=None):
  num_nodes = maybe_num_nodes(edge_index, num_nodes)

  if edge_weight is None:
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                             device=edge_index.device)

  if not fill_value == 0:
    edge_index, tmp_edge_weight = add_remaining_self_loops(
      edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

  row, col = edge_index[0], edge_index[1]
  indices = row if norm_dim == 0 else col
  deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
  deg_inv_sqrt = deg.pow_(-1)
  edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
  return edge_index, edge_weight


def mean_confidence_interval(data, confidence=0.95):
  """
  As number of samples will be < 10 use t-test for the mean confidence intervals
  :param data: NDarray of metric means
  :param confidence: The desired confidence interval
  :return: Float confidence interval
  """
  if len(data) < 2:
    return 0
  a = 1.0 * np.array(data)
  n = len(a)
  _, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return h


def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  return torch.sparse.FloatTensor(i, v * d, s.size())


def get_sem(vec):
  """
  wrapper around the scipy standard error metric
  :param vec: List of metric means
  :return:
  """
  if len(vec) > 1:
    retval = sem(vec)
  else:
    retval = 0.
  return retval


def get_full_adjacency(num_nodes):
  # what is the format of the edge index?
  edge_index = torch.zeros((2, num_nodes ** 2),dtype=torch.long)
  for idx in range(num_nodes):
    edge_index[0][idx * num_nodes: (idx + 1) * num_nodes] = idx
    edge_index[1][idx * num_nodes: (idx + 1) * num_nodes] = torch.arange(0, num_nodes,dtype=torch.long)
  return edge_index



from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
               num_nodes: Optional[int] = None) -> Tensor:
  r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
  out = src - src.max()
  # out = out.exp()
  out = (out + torch.sqrt(out ** 2 + 4)) / 2

  if ptr is not None:
    out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
  elif index is not None:
    N = maybe_num_nodes(index, num_nodes)
    out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
  else:
    raise NotImplementedError

  return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.cnt = 0

  def update(self, val):
    self.val = val
    self.sum += val
    self.cnt += 1

  def get_average(self):
    if self.cnt == 0:
      return 0
    return self.sum / self.cnt

  def get_value(self):
    return self.val


class DummyDataset(object):
  def __init__(self, data, num_classes):
    self.data = data
    self.num_classes = num_classes


class DummyData(object):
  def __init__(self, edge_index=None, edge_Attr=None, num_nodes=None):
    self.edge_index = edge_index
    self.edge_attr = edge_Attr
    self.num_nodes = num_nodes
