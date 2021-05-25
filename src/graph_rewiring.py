"""
functions to generate a graph from the input graph and features
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter
from torch_geometric.transforms.two_hop import TwoHop
from utils import get_rw_adj
# from torch_geometric.transforms import GDC
from pykeops.torch import LazyTensor
import os
import pickle

### for custom GDC
import torch
import numba
import numpy as np
from scipy.linalg import expm
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj, \
  remove_self_loops, dense_to_sparse, to_undirected
from torch_sparse import coalesce
from torch_scatter import scatter_add

POS_ENC_PATH = os.path.join("../data", "pos_encodings")


def jit(**kwargs):
  def decorator(func):
    try:
      return numba.jit(cache=True, **kwargs)(func)
    except RuntimeError:
      return numba.jit(cache=False, **kwargs)(func)

  return decorator


###

def get_two_hop(data):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  th = TwoHop()
  data = th(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def apply_gdc(data, opt, type="combined"):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  print('performing gdc transformation with method {}, sparsification {}'.format(opt['gdc_method'],
                                                                                 opt['gdc_sparsification']))
  if opt['gdc_method'] == 'ppr':
    diff_args = dict(method='ppr', alpha=opt['ppr_alpha'])
  else:
    diff_args = dict(method='heat', t=opt['heat_time'])

  if opt['gdc_sparsification'] == 'topk':
    sparse_args = dict(method='topk', k=opt['gdc_k'], dim=0)
    diff_args['eps'] = opt['gdc_threshold']
  else:
    sparse_args = dict(method='threshold', eps=opt['gdc_threshold'])
    diff_args['eps'] = opt['gdc_threshold']
  print('gdc sparse args: {}'.format(sparse_args))
  if opt['self_loop_weight'] != 0:
    gdc = GDC(float(opt['self_loop_weight']),
              normalization_in='sym',
              normalization_out='col',
              diffusion_kwargs=diff_args,
              sparsification_kwargs=sparse_args, exact=opt['exact'])
  else:
    gdc = GDC(self_loop_weight=None,
              normalization_in='sym',
              normalization_out='col',
              diffusion_kwargs=diff_args,
              sparsification_kwargs=sparse_args, exact=opt['exact'])
  if isinstance(data.num_nodes, list):
    data.num_nodes = data.num_nodes[0]

  if type == 'combined':
    data = gdc(data)
  elif type == 'densify':
    data = gdc.densify(data)
  elif type == 'sparsify':
    data = gdc.sparsify(data)

  elif type == 'pos_encoding':
    if opt['pos_enc_orientation'] == "row":  # encode row of S_hat
      return gdc.position_encoding(data)
    elif opt['pos_enc_orientation'] == "col":  # encode col of S_hat
      return gdc.position_encoding(data).T

  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def make_symmetric(data):
  n = data.num_nodes
  if data.edge_attr is not None:
    # todo need to write test compare to to_undirected source code
    ApAT_index = torch.cat([data.edge_index, data.edge_index[[1, 0], :]], dim=1)
    ApAT_value = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    ei, ew = torch_sparse.coalesce(ApAT_index, ApAT_value, n, n, op="add")

    # A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr, (n, n)).coalesce()
    # AT_ei, AT_ew = torch_sparse.transpose(data.edge_index, data.edge_attr, n, n)
    # AT = torch.sparse_coo_tensor(AT_ei, AT_ew, (n, n)).coalesce()
    # A_sym = (0.5 * A + 0.5 * AT).coalesce()
    # ei = A_sym.indices()
    # ew = A_sym.values()
  else:
    ei = to_undirected(data.edge_index)
    ew = None

  ei, ew = get_rw_adj(ei, edge_weight=ew, norm_dim=1, fill_value=0., num_nodes=n)

  return ei, ew


def dirichlet_energy(edge_index, edge_weight, n, X):
  if edge_weight is None:
    edge_weight = torch.ones(edge_index.size(1),
                             device=edge_index.device)
  de = torch_sparse.spmm(edge_index, edge_weight, n, n, X)
  return X.T @ de


def KNN(x, opt):
  # https://github.com/getkeops/keops/tree/3efd428b55c724b12f23982c06de00bc4d02d903
  k = opt['rewire_KNN_k']
  print(f"Rewiring with KNN: t={opt['rewire_KNN_T']}, k={opt['rewire_KNN_k']}")
  X_i = LazyTensor(x[:, None, :])  # (N, 1, hd)
  X_j = LazyTensor(x[None, :, :])  # (1, N, hd)

  # distance between all the grid points and all the random data points
  D_ij = ((X_i - X_j) ** 2).sum(-1)  # (N**2, hd) symbolic matrix of squared distances
  # H_ij = D_ij / (X_i[:,:,0] * X_j[:,:,0])
  # indKNN = H_ij.argKmin(k, dim=1)
  # D_ij = (-((X_i - X_j) ** 2).sum(-1)).exp()
  # todo split into feature and pos with scale params
  # self.output_var_p ** 2 * torch.exp(-torch.sum((src_p - dst_p) ** 2, dim=1) / (2 * self.lengthscale_p ** 2))

  # take the indices of the K closest neighbours measured in euclidean distance
  indKNN = D_ij.argKmin(k, dim=1)
  # LS = torch.linspace(0, len(indKNN.view(-1)), len(indKNN.view(-1)) + 1)[:-1].unsqueeze(0) // k
  LS = torch.linspace(0, len(indKNN.view(-1)), len(indKNN.view(-1)) + 1, dtype=torch.int64, device=indKNN.device)[
       :-1].unsqueeze(0) // k
  # LS     = torch.linspace(0, len(indKNN.view(-1)), len(indKNN.view(-1)) + 1, device=indKNN.device)[:-1].unsqueeze(0) // k

  ei = torch.cat([LS, indKNN.view(1, -1)], dim=0)

  if opt['rewire_KNN_sym']:
    ei = to_undirected(ei)

  return ei


@torch.no_grad()
def apply_KNN(data, pos_encoding, model, opt):
#todo KNN in a similar way to edge sampling where the QK project is taken from a forward
# pass in the model with custom 'attention type'
  if opt['rewire_KNN_T'] == "raw":
    ei = KNN(data.x, opt)  # rewiring on raw features here

  elif opt['rewire_KNN_T'] == "T0":
    z0 = model.forward_encoder(data.x, pos_encoding)
    if model.opt['use_labels']:
      y = z0[:, -model.num_classes:]
      z0 = z0[:, :-model.num_classes]
    p0 = z0[:, model.opt['feat_hidden_dim']:].contiguous()
    ei = KNN(p0, opt)

  elif opt['rewire_KNN_T'] == 'TN':
    zT = model.forward_ODE(data.x, pos_encoding)
    if model.opt['use_labels']:
      y = zT[:, -model.num_classes:]
      zT = zT[:, :-model.num_classes]
    pT = zT[:, model.opt['feat_hidden_dim']:].contiguous()
    ei = KNN(pT, opt)
  else:
    raise Exception("Need to set rewire_KNN_T")

  return ei


def edge_sampling(model, z, opt):

  if opt['edge_sampling_space'] == 'attention':
    # attention_weights = model.odeblock.get_attention_weights(z)
    attention_weights = model.odeblock.get_raw_attention_weights(z)
    mean_att = attention_weights.mean(dim=1, keepdim=False)
    threshold = torch.quantile(mean_att, 1 - opt['att_samp_pct'])
    mask = mean_att > threshold
  elif opt['edge_sampling_space'] in ['pos_distance','z_distance','pos_distance_QK','z_distance_QK']:
    #calc distance metric if edge_sampling_space is in:
      # ['pos_distance','z_distance']) if ['attention_type'] == exp_kernel_z or exp_kernel_pos as have xremoved queries / keys
      # ['pos_distance_QK','z_distance_QK']) for exp_kernel
      # ['z_distance_QK']) for any other attention type as don't learn the pos QKp(p) just QK(z), plus requires symmetric_attention
    temp_att_type = model.opt['attention_type']
    model.opt['attention_type'] = model.opt['edge_sampling_space'] #this changes the opt at all levels as opt is assigment link
    pos_enc_distances = model.odeblock.get_attention_weights(z) #forward pass of multihead_att_layer
    model.odeblock.multihead_att_layer.opt['attention_type'] = temp_att_type

    #threshold on distances so we throw away the biggest, opposite to attentions
    threshold = torch.quantile(pos_enc_distances, 1 - opt['edge_sampling_rmv'])
    mask = pos_enc_distances < threshold

  #renormalise attention - this should get done anyway in each block forward pass
  # model.odeblock.odefunc.edge_index = model.odeblock.odefunc.edge_index[:, mask.T]
  # sampled_attention_weights = renormalise_attention(mean_att[mask])
  # model.odeblock.odefunc.attention_weights = sampled_attention_weights

  # print('retaining {} of {} edges'.format(self.odefunc.edge_index.shape[1], self.data_edge_index.shape[1]))

  model.odeblock.odefunc.edge_index = model.odeblock.odefunc.edge_index[:, mask.T]

  if opt['edge_sampling_sym']:
    model.odeblock.odefunc.edge_index = to_undirected(model.odeblock.odefunc.edge_index)

  return model.odeblock.odefunc.edge_index

# def renormalise_attention(model, attention):
#   index = model.odeblock.odefunc.edge_index[model.opt['attention_norm_idx']]
#   att_sums = scatter(attention, index, dim=0, dim_size=model.num_nodes, reduce='sum')[index]
#   return attention / (att_sums + 1e-16)

def add_edges(model, opt):
  num_nodes = model.num_nodes
  # M = int(num_nodes * opt['edge_sampling_add'])
  num_edges = model.odeblock.odefunc.edge_index.shape[1]
  M = int(num_edges * opt['edge_sampling_add'])
  # generate new edges and add to edge_index
  if opt['edge_sampling_add_type'] == 'random':
    new_edges = np.random.choice(num_edges, size=(2, M), replace=True, p=None)
    new_edges = torch.tensor(new_edges, device=model.device)
    new_edges2 = new_edges[[1, 0], :]
    # cat = torch.cat([model.odeblock.odefunc.edge_index, new_edges], dim=1)
    cat = torch.cat([model.odeblock.odefunc.edge_index, new_edges, new_edges2], dim=1)
  elif opt['edge_sampling_add_type'] == 'anchored':
    pass
  elif opt['edge_sampling_add_type'] == 'importance':
    atts = model.odeblock.odefunc.attention_weights.mean(dim=1)
    # src = model.odeblock.odefunc.edge_index[0, :]
    dst = model.odeblock.odefunc.edge_index[1, :]

    importance = scatter(atts, dst, dim=0, dim_size=num_nodes, reduce='sum') #column sum to represent outgoing importance
    # anchors = torch.topk(importance, M, dim=0)[1]
    importance_probs = np.abs(importance.detach().numpy()) / np.abs(importance.detach().numpy()).sum()
    anchors = torch.tensor(np.random.choice(num_nodes, size=M, replace=True, p=importance_probs), device=model.device)
    anchors2 = torch.tensor(np.random.choice(num_nodes, size=M, replace=True, p=importance_probs), device=model.device)
    new_edges = torch.stack([anchors, anchors2], dim=0)
    new_edges2 = torch.stack([anchors2, anchors], dim=0)
    #todo this only adds 1 new edge to each important anchor
    cat = torch.cat([model.odeblock.odefunc.edge_index, new_edges, new_edges2], dim=1)
  elif opt['edge_sampling_add_type'] == 'degree': #proportional to degree
    pass

  new_ei = torch.unique(cat, sorted=False, return_inverse=False, return_counts=False, dim=1)
  return new_ei

@torch.no_grad()
def apply_edge_sampling(x, pos_encoding, model, opt):
  print(f"Rewiring with edge sampling")

  #add to model edge index
  model.odeblock.odefunc.edge_index = add_edges(model, opt)

  # get Z_T0 or Z_TN
  if opt['edge_sampling_T'] == "T0":
    z = model.forward_encoder(x, pos_encoding)
  elif opt['edge_sampling_T'] == 'TN':
    z = model.forward_ODE(x, pos_encoding)

  #sample the edges and update edge index in model
  model.odeblock.odefunc.edge_index = edge_sampling(model, z, opt)


def apply_beltrami(data, opt, data_dir='../data'):
  pos_enc_dir = os.path.join(f"{data_dir}", "pos_encodings")
  # generate new positional encodings
  # do encodings already exist on disk?
  fname = os.path.join(pos_enc_dir, f"{opt['dataset']}_{opt['pos_enc_type']}.pkl")
  print(f"[i] Looking for positional encodings in {fname}...")

  # - if so, just load them
  if os.path.exists(fname):
    print("    Found them! Loading cached version")
    with open(fname, "rb") as f:
      pos_encoding = pickle.load(f)

  # - otherwise, calculate...
  else:
    print("    Encodings not found! Calculating and caching them")
    # choose different functions for different positional encodings
    if opt['pos_enc_type'] == "GDC":
      pos_encoding = apply_gdc(data, opt, type="pos_encoding")
    else:
      print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
      quit()
    # - ... and store them on disk
    if not os.path.exists(pos_enc_dir):
      os.makedirs(pos_enc_dir)
    with open(fname, "wb") as f:
      pickle.dump(pos_encoding, f)

  return pos_encoding
  # data.x = torch.cat([data.x, pos_encoding], dim=1)
  # return data


# Editted PyGeo source code
class GDC(object):
  r"""Processes the graph via Graph Diffusion Convolution (GDC) from the
  `"Diffusion Improves Graph Learning" <https://www.kdd.in.tum.de/gdc>`_
  paper.

  .. note::

      The paper offers additional advice on how to choose the
      hyperparameters.
      For an example of using GCN with GDC, see `examples/gcn.py
      <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
      gcn.py>`_.

  Args:
      self_loop_weight (float, optional): Weight of the added self-loop.
          Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
      normalization_in (str, optional): Normalization of the transition
          matrix on the original (input) graph. Possible values:
          :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
          See :func:`GDC.transition_matrix` for details.
          (default: :obj:`"sym"`)
      normalization_out (str, optional): Normalization of the transition
          matrix on the transformed GDC (output) graph. Possible values:
          :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
          See :func:`GDC.transition_matrix` for details.
          (default: :obj:`"col"`)
      diffusion_kwargs (dict, optional): Dictionary containing the parameters
          for diffusion.
          `method` specifies the diffusion method (:obj:`"ppr"`,
          :obj:`"heat"` or :obj:`"coeff"`).
          Each diffusion method requires different additional parameters.
          See :func:`GDC.diffusion_matrix_exact` or
          :func:`GDC.diffusion_matrix_approx` for details.
          (default: :obj:`dict(method='ppr', alpha=0.15)`)
      sparsification_kwargs (dict, optional): Dictionary containing the
          parameters for sparsification.
          `method` specifies the sparsification method (:obj:`"threshold"` or
          :obj:`"topk"`).
          Each sparsification method requires different additional
          parameters.
          See :func:`GDC.sparsify_dense` for details.
          (default: :obj:`dict(method='threshold', avg_degree=64)`)
      exact (bool, optional): Whether to exactly calculate the diffusion
          matrix.
          Note that the exact variants are not scalable.
          They densify the adjacency matrix and calculate either its inverse
          or its matrix exponential.
          However, the approximate variants do not support edge weights and
          currently only personalized PageRank and sparsification by
          threshold are implemented as fast, approximate versions.
          (default: :obj:`True`)

  :rtype: :class:`torch_geometric.data.Data`
  """

  def __init__(self, self_loop_weight=1, normalization_in='sym',
               normalization_out='col',
               diffusion_kwargs=dict(method='ppr', alpha=0.15),
               sparsification_kwargs=dict(method='threshold',
                                          avg_degree=64), exact=True):
    self.self_loop_weight = self_loop_weight
    self.normalization_in = normalization_in
    self.normalization_out = normalization_out
    self.diffusion_kwargs = diffusion_kwargs
    self.sparsification_kwargs = sparsification_kwargs
    self.exact = exact

    if self_loop_weight:
      assert exact or self_loop_weight == 1

  @torch.no_grad()
  def __call__(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                             **self.diffusion_kwargs)
      edge_index, edge_weight = self.sparsify_dense(
        diff_mat, **self.sparsification_kwargs)
    else:
      edge_index, edge_weight = self.diffusion_matrix_approx(
        edge_index, edge_weight, N, self.normalization_in,
        **self.diffusion_kwargs)
      edge_index, edge_weight = self.sparsify_sparse(
        edge_index, edge_weight, N, **self.sparsification_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    data.edge_index = edge_index
    data.edge_attr = edge_weight

    return data

  def densify(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                             **self.diffusion_kwargs)
      edge_index, edge_weight = dense_to_sparse(diff_mat)
      # edge_index, edge_weight = self.sparsify_dense(
      #   diff_mat, **self.sparsification_kwargs)
    else:
      edge_index, edge_weight = self.diffusion_matrix_approx(
        edge_index, edge_weight, N, self.normalization_in,
        **self.diffusion_kwargs)
      # edge_index, edge_weight = self.sparsify_sparse(
      #   edge_index, edge_weight, N, **self.sparsification_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    data.edge_index = edge_index
    data.edge_attr = edge_weight

    return data

  def sparsify(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      # diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
      #                                        **self.diffusion_kwargs)
      diff_mat = to_dense_adj(edge_index,
                              edge_attr=edge_weight).squeeze()
      edge_index, edge_weight = self.sparsify_dense(
        diff_mat, **self.sparsification_kwargs)
    else:
      # edge_index, edge_weight = self.diffusion_matrix_approx(
      #   edge_index, edge_weight, N, self.normalization_in,
      #   **self.diffusion_kwargs)
      edge_index, edge_weight = self.sparsify_sparse(
        edge_index, edge_weight, N, **self.sparsification_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    data.edge_index = edge_index
    data.edge_attr = edge_weight

    return data

  def position_encoding(self, data):
    N = data.num_nodes
    edge_index = data.edge_index
    if data.edge_attr is None:
      edge_weight = torch.ones(edge_index.size(1),
                               device=edge_index.device)
    else:
      edge_weight = data.edge_attr
      assert self.exact
      assert edge_weight.dim() == 1

    if self.self_loop_weight:
      edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=self.self_loop_weight,
        num_nodes=N)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)

    if self.exact:
      edge_index, edge_weight = self.transition_matrix(
        edge_index, edge_weight, N, self.normalization_in)
      diff_mat = self.diffusion_matrix_exact(edge_index, edge_weight, N,
                                             **self.diffusion_kwargs)
      edge_index, edge_weight = dense_to_sparse(diff_mat)
      # edge_index, edge_weight = self.sparsify_dense(
      #   diff_mat, **self.sparsification_kwargs)
    else:
      edge_index, edge_weight = self.diffusion_matrix_approx(
        edge_index, edge_weight, N, self.normalization_in,
        **self.diffusion_kwargs)
      # edge_index, edge_weight = self.sparsify_sparse(
      #   edge_index, edge_weight, N, **self.sparsification_kwargs)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = self.transition_matrix(
      edge_index, edge_weight, N, self.normalization_out)

    return to_dense_adj(edge_index,
                        edge_attr=edge_weight).squeeze()

  def transition_matrix(self, edge_index, edge_weight, num_nodes,
                        normalization):
    r"""Calculate the approximate, sparse diffusion on a given sparse
    matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor): One-dimensional edge weights.
        num_nodes (int): Number of nodes.
        normalization (str): Normalization scheme:

            1. :obj:`"sym"`: Symmetric normalization
               :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
               \mathbf{D}^{-1/2}`.
            2. :obj:`"col"`: Column-wise normalization
               :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`.
            3. :obj:`"row"`: Row-wise normalization
               :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`.
            4. :obj:`None`: No normalization.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if normalization == 'sym':
      row, col = edge_index
      deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
      deg_inv_sqrt = deg.pow(-0.5)
      deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'col':
      _, col = edge_index
      deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
      deg_inv = 1. / deg
      deg_inv[deg_inv == float('inf')] = 0
      edge_weight = edge_weight * deg_inv[col]
    elif normalization == 'row':
      row, _ = edge_index
      deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
      deg_inv = 1. / deg
      deg_inv[deg_inv == float('inf')] = 0
      edge_weight = edge_weight * deg_inv[row]
    elif normalization is None:
      pass
    else:
      raise ValueError(
        'Transition matrix normalization {} unknown.'.format(
          normalization))

    return edge_index, edge_weight

  def diffusion_matrix_exact(self, edge_index, edge_weight, num_nodes,
                             method, **kwargs):
    r"""Calculate the (dense) diffusion on a given sparse graph.
    Note that these exact variants are not scalable. They densify the
    adjacency matrix and calculate either its inverse or its matrix
    exponential.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor): One-dimensional edge weights.
        num_nodes (int): Number of nodes.
        method (str): Diffusion method:

            1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
               Additionally expects the parameter:

               - **alpha** (*float*) - Return probability in PPR.
                 Commonly lies in :obj:`[0.05, 0.2]`.

            2. :obj:`"heat"`: Use heat kernel diffusion.
               Additionally expects the parameter:

               - **t** (*float*) - Time of diffusion. Commonly lies in
                 :obj:`[2, 10]`.

            3. :obj:`"coeff"`: Freely choose diffusion coefficients.
               Additionally expects the parameter:

               - **coeffs** (*List[float]*) - List of coefficients
                 :obj:`theta_k` for each power of the transition matrix
                 (starting at :obj:`0`).

    :rtype: (:class:`Tensor`)
    """
    if method == 'ppr':
      # α (I_n + (α - 1) A)^-1
      edge_weight = (kwargs['alpha'] - 1) * edge_weight
      edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                               fill_value=1,
                                               num_nodes=num_nodes)
      mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
      diff_matrix = kwargs['alpha'] * torch.inverse(mat)

    elif method == 'heat':
      # exp(t (A - I_n))
      edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                               fill_value=-1,
                                               num_nodes=num_nodes)
      edge_weight = kwargs['t'] * edge_weight
      mat = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
      undirected = is_undirected(edge_index, edge_weight, num_nodes)
      diff_matrix = self.__expm__(mat, undirected)

    elif method == 'coeff':
      adj_matrix = to_dense_adj(edge_index,
                                edge_attr=edge_weight).squeeze()
      mat = torch.eye(num_nodes, device=edge_index.device)

      diff_matrix = kwargs['coeffs'][0] * mat
      for coeff in kwargs['coeffs'][1:]:
        mat = mat @ adj_matrix
        diff_matrix += coeff * mat
    else:
      raise ValueError('Exact GDC diffusion {} unknown.'.format(method))

    return diff_matrix

  def diffusion_matrix_approx(self, edge_index, edge_weight, num_nodes,
                              normalization, method, **kwargs):
    r"""Calculate the approximate, sparse diffusion on a given sparse
    graph.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor): One-dimensional edge weights.
        num_nodes (int): Number of nodes.
        normalization (str): Transition matrix normalization scheme
            (:obj:`"sym"`, :obj:`"row"`, or :obj:`"col"`).
            See :func:`GDC.transition_matrix` for details.
        method (str): Diffusion method:

            1. :obj:`"ppr"`: Use personalized PageRank as diffusion.
               Additionally expects the parameters:

               - **alpha** (*float*) - Return probability in PPR.
                 Commonly lies in :obj:`[0.05, 0.2]`.

               - **eps** (*float*) - Threshold for PPR calculation stopping
                 criterion (:obj:`edge_weight >= eps * out_degree`).
                 Recommended default: :obj:`1e-4`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if method == 'ppr':
      if normalization == 'sym':
        # Calculate original degrees.
        _, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

      edge_index_np = edge_index.cpu().numpy()
      # Assumes coalesced edge_index.
      _, indptr, out_degree = np.unique(edge_index_np[0],
                                        return_index=True,
                                        return_counts=True)
      indptr = np.append(indptr, len(edge_index_np[0]))

      neighbors, neighbor_weights = GDC.__calc_ppr__(
        indptr, edge_index_np[1], out_degree, kwargs['alpha'],
        kwargs['eps'])
      ppr_normalization = 'col' if normalization == 'col' else 'row'
      edge_index, edge_weight = self.__neighbors_to_graph__(
        neighbors, neighbor_weights, ppr_normalization,
        device=edge_index.device)
      edge_index = edge_index.to(torch.long)

      if normalization == 'sym':
        # We can change the normalization from row-normalized to
        # symmetric by multiplying the resulting matrix with D^{1/2}
        # from the left and D^{-1/2} from the right.
        # Since we use the original degrees for this it will be like
        # we had used symmetric normalization from the beginning
        # (except for errors due to approximation).
        row, col = edge_index
        deg_inv = deg.sqrt()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv[row] * edge_weight * deg_inv_sqrt[col]
      elif normalization in ['col', 'row']:
        pass
      else:
        raise ValueError(
          ('Transition matrix normalization {} not implemented for '
           'non-exact GDC computation.').format(normalization))

    elif method == 'heat':
      raise NotImplementedError(
        ('Currently no fast heat kernel is implemented. You are '
         'welcome to create one yourself, e.g., based on '
         '"Kloster and Gleich: Heat kernel based community detection '
         '(KDD 2014)."'))
    else:
      raise ValueError(
        'Approximate GDC diffusion {} unknown.'.format(method))

    return edge_index, edge_weight

  def sparsify_dense(self, matrix, method, **kwargs):
    r"""Sparsifies the given dense matrix.

    Args:
        matrix (Tensor): Matrix to sparsify.
        num_nodes (int): Number of nodes.
        method (str): Method of sparsification. Options:

            1. :obj:`"threshold"`: Remove all edges with weights smaller
               than :obj:`eps`.
               Additionally expects one of these parameters:

               - **eps** (*float*) - Threshold to bound edges at.

               - **avg_degree** (*int*) - If :obj:`eps` is not given,
                 it can optionally be calculated by calculating the
                 :obj:`eps` required to achieve a given :obj:`avg_degree`.

            2. :obj:`"topk"`: Keep edges with top :obj:`k` edge weights per
               node (column).
               Additionally expects the following parameters:

               - **k** (*int*) - Specifies the number of edges to keep.

               - **dim** (*int*) - The axis along which to take the top
                 :obj:`k`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert matrix.shape[0] == matrix.shape[1]
    N = matrix.shape[1]

    if method == 'threshold':
      if 'eps' not in kwargs.keys():
        kwargs['eps'] = self.__calculate_eps__(matrix, N,
                                               kwargs['avg_degree'])

      edge_index = (matrix >= kwargs['eps']).nonzero(as_tuple=False).t()
      edge_index_flat = edge_index[0] * N + edge_index[1]
      edge_weight = matrix.flatten()[edge_index_flat]

    elif method == 'topk':
      assert kwargs['dim'] in [0, 1]
      sort_idx = torch.argsort(matrix, dim=kwargs['dim'],
                               descending=True)
      if kwargs['dim'] == 0:
        top_idx = sort_idx[:kwargs['k']]
        edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                   index=top_idx).flatten()

        row_idx = torch.arange(0, N, device=matrix.device).repeat(
          kwargs['k'])
        edge_index = torch.stack([top_idx.flatten(), row_idx], dim=0)
      else:
        top_idx = sort_idx[:, :kwargs['k']]
        edge_weight = torch.gather(matrix, dim=kwargs['dim'],
                                   index=top_idx).flatten()

        col_idx = torch.arange(
          0, N, device=matrix.device).repeat_interleave(kwargs['k'])
        edge_index = torch.stack([col_idx, top_idx.flatten()], dim=0)
    else:
      raise ValueError('GDC sparsification {} unknown.'.format(method))

    return edge_index, edge_weight

  def sparsify_sparse(self, edge_index, edge_weight, num_nodes, method,
                      **kwargs):
    r"""Sparsifies a given sparse graph further.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor): One-dimensional edge weights.
        num_nodes (int): Number of nodes.
        method (str): Method of sparsification:

            1. :obj:`"threshold"`: Remove all edges with weights smaller
               than :obj:`eps`.
               Additionally expects one of these parameters:

               - **eps** (*float*) - Threshold to bound edges at.

               - **avg_degree** (*int*) - If :obj:`eps` is not given,
                 it can optionally be calculated by calculating the
                 :obj:`eps` required to achieve a given :obj:`avg_degree`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if method == 'threshold':
      if 'eps' not in kwargs.keys():
        kwargs['eps'] = self.__calculate_eps__(edge_weight, num_nodes,
                                               kwargs['avg_degree'])

      remaining_edge_idx = (edge_weight >= kwargs['eps']).nonzero(
        as_tuple=False).flatten()
      edge_index = edge_index[:, remaining_edge_idx]
      edge_weight = edge_weight[remaining_edge_idx]
    elif method == 'topk':
      raise NotImplementedError(
        'Sparse topk sparsification not implemented.')
    else:
      raise ValueError('GDC sparsification {} unknown.'.format(method))

    return edge_index, edge_weight

  def __expm__(self, matrix, symmetric):
    r"""Calculates matrix exponential.

    Args:
        matrix (Tensor): Matrix to take exponential of.
        symmetric (bool): Specifies whether the matrix is symmetric.

    :rtype: (:class:`Tensor`)
    """
    if symmetric:
      e, V = torch.symeig(matrix, eigenvectors=True)
      diff_mat = V @ torch.diag(e.exp()) @ V.t()
    else:
      diff_mat_np = expm(matrix.cpu().numpy())
      diff_mat = torch.Tensor(diff_mat_np).to(matrix.device)
    return diff_mat

  def __calculate_eps__(self, matrix, num_nodes, avg_degree):
    r"""Calculates threshold necessary to achieve a given average degree.

    Args:
        matrix (Tensor): Adjacency matrix or edge weights.
        num_nodes (int): Number of nodes.
        avg_degree (int): Target average degree.

    :rtype: (:class:`float`)
    """
    sorted_edges = torch.sort(matrix.flatten(), descending=True).values
    if avg_degree * num_nodes > len(sorted_edges):
      return -np.inf

    left = sorted_edges[avg_degree * num_nodes - 1]
    right = sorted_edges[avg_degree * num_nodes]
    return (left + right) / 2.0

  def __neighbors_to_graph__(self, neighbors, neighbor_weights,
                             normalization='row', device='cpu'):
    r"""Combine a list of neighbors and neighbor weights to create a sparse
    graph.

    Args:
        neighbors (List[List[int]]): List of neighbors for each node.
        neighbor_weights (List[List[float]]): List of weights for the
            neighbors of each node.
        normalization (str): Normalization of resulting matrix
            (options: :obj:`"row"`, :obj:`"col"`). (default: :obj:`"row"`)
        device (torch.device): Device to create output tensors on.
            (default: :obj:`"cpu"`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    edge_weight = torch.Tensor(np.concatenate(neighbor_weights)).to(device)
    i = np.repeat(np.arange(len(neighbors)),
                  np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    if normalization == 'col':
      edge_index = torch.Tensor(np.vstack([j, i])).to(device)
      N = len(neighbors)
      edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    elif normalization == 'row':
      edge_index = torch.Tensor(np.vstack([i, j])).to(device)
    else:
      raise ValueError(
        f"PPR matrix normalization {normalization} unknown.")
    return edge_index, edge_weight

  @staticmethod
  @jit(nopython=True, parallel=True)
  def __calc_ppr__(indptr, indices, out_degree, alpha, eps):
    r"""Calculate the personalized PageRank vector for all nodes
    using a variant of the Andersen algorithm
    (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

    Args:
        indptr (np.ndarray): Index pointer for the sparse matrix
            (CSR-format).
        indices (np.ndarray): Indices of the sparse matrix entries
            (CSR-format).
        out_degree (np.ndarray): Out-degree of each node.
        alpha (float): Alpha of the PageRank to calculate.
        eps (float): Threshold for PPR calculation stopping criterion
            (:obj:`edge_weight >= eps * out_degree`).

    :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
    """
    alpha_eps = alpha * eps
    js = [[0]] * len(out_degree)
    vals = [[0.]] * len(out_degree)
    for inode_uint in numba.prange(len(out_degree)):
      inode = numba.int64(inode_uint)
      p = {inode: 0.0}
      r = {}
      r[inode] = alpha
      q = [inode]
      while len(q) > 0:
        unode = q.pop()

        res = r[unode] if unode in r else 0
        if unode in p:
          p[unode] += res
        else:
          p[unode] = res
        r[unode] = 0
        for vnode in indices[indptr[unode]:indptr[unode + 1]]:
          _val = (1 - alpha) * res / out_degree[unode]
          if vnode in r:
            r[vnode] += _val
          else:
            r[vnode] = _val

          res_vnode = r[vnode] if vnode in r else 0
          if res_vnode >= alpha_eps * out_degree[vnode]:
            if vnode not in q:
              q.append(vnode)
      js[inode] = list(p.keys())
      vals[inode] = list(p.values())
    return js, vals

  def __repr__(self):
    return '{}()'.format(self.__class__.__name__)
