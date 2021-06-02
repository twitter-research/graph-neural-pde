"""
functions to generate a graph from the input graph and features
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.transforms.two_hop import TwoHop
from torch_geometric.utils import add_self_loops, to_undirected, to_dense_adj, dense_to_sparse
from torch_geometric.transforms import GDC
from utils import get_rw_adj, get_full_adjacency
import os
import pickle
from hyperbolic_distances import hyperbolize


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
    ApAT_index = torch.cat([data.edge_index, data.edge_index[[1, 0], :]], dim=1)
    ApAT_value = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    ei, ew = torch_sparse.coalesce(ApAT_index, ApAT_value, n, n, op="add")
  else:
    ei = to_undirected(data.edge_index)
    ew = None

  ei, ew = get_rw_adj(ei, edge_weight=ew, norm_dim=1, fill_value=0., num_nodes=n)

  return ei, ew


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
      # pos_encoding = pickle.load(f)
      pos_encoding = pickle.load(f)
    if opt['pos_enc_type'].startswith("DW"):
      pos_encoding = pos_encoding['data']

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
    POS_ENC_PATH = os.path.join(data_dir, "pos_encodings")
    if not os.path.exists(POS_ENC_PATH):
      os.makedirs(POS_ENC_PATH)

    if opt['pos_enc_csv']:
      sp = pos_encoding.to_sparse()
      table_mat = np.concatenate([sp.indices(), np.atleast_2d(sp.values())], axis=0).T
      np.savetxt(f"{fname[:-4]}.csv", table_mat, delimiter=",")

    with open(fname, "wb") as f:
      pickle.dump(pos_encoding, f)

  return pos_encoding


class GDC(GDC):
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