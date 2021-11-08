"""
functions to generate a graph from the input graph and features
"""
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter
from torch_geometric.transforms.two_hop import TwoHop
from utils import get_rw_adj, get_full_adjacency
from pykeops.torch import LazyTensor
import os
import pickle
from distances_kNN import apply_dist_KNN, apply_dist_threshold, get_distances, apply_feat_KNN
from hyperbolic_distances import hyperbolize


### for custom GDC
import torch
import numba
import numpy as np
from scipy.linalg import expm
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj, \
   dense_to_sparse, to_undirected
from torch_sparse import coalesce
from torch_scatter import scatter_add


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
