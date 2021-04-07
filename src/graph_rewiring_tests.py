"""
functions to generate a graph from the input graph and features
"""
import argparse
import numpy as np
import torch
from torch_geometric.transforms.two_hop import TwoHop
from torch_geometric.transforms import GDC
from utils import get_rw_adj
from data import get_dataset
from graph_rewiring import get_two_hop, apply_gdc

def test_rewiring_churn(edge_index0, edge_index1):

  np_idx0 = edge_index0.numpy().T
  np_idx1 = edge_index1.numpy().T
  rows0 = np.ascontiguousarray(np_idx0).view(np.dtype((np.void, np_idx0.dtype.itemsize * np_idx0.shape[1])))
  rows1 = np.ascontiguousarray(np_idx1).view(np.dtype((np.void, np_idx1.dtype.itemsize * np_idx1.shape[1])))
  # todo use jax.numpy.in1d to do on GPU
  added_mask = np.in1d(rows1, rows0, assume_unique=True, invert=True)
  removed_mask = np.in1d(rows0, rows1, assume_unique=True, invert=True)

  print(f"Count Original Edges {np_idx0.shape[0]}")
  print(f"Count Final Edges {np_idx1.shape[0]}")
  print(f"Percent Final Edges {np_idx1.shape[0]/np_idx0.shape[0]:.4f}")
  print(f"Count/Percent Removed Edges {removed_mask.sum()}, {removed_mask.sum()/np_idx0.shape[0]:.4f}")
  print(f"Count/Percent Added Edges {added_mask.sum()}, {added_mask.sum()/np_idx0.shape[0]:.4f}")
  total = np_idx0.shape[0] + added_mask.sum() - removed_mask.sum()
  print(f"Check: orig {np_idx0.shape[0]:,} + added {added_mask.sum():,} "
        f"- removed {removed_mask.sum():,} = {total:,} == model {np_idx1.shape[0]:,}")


def main(opt):

  dataset = get_dataset(opt, '../data', False)
  gdc_data = apply_gdc(dataset.data, opt)
  test_rewiring_churn(dataset.data.edge_index, gdc_data.edge_index)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--use_cora_defaults', action='store_true',
                      help='Whether to run with best params for cora. Overrides the choice of dataset')
  # data args
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--data_norm', type=str, default='rw',
                      help='rw for random walk, gcn for symmetric gcn norm')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
  parser.add_argument('--label_rate', type=float, default=0.5, help='% of training labels to use when --use_labels is set.')
  # rewiring args
  parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
  parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
  parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
  parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
  parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                      help="obove this edge weight, keep edges when using threshold")
  parser.add_argument('--gdc_avg_degree', type=int, default=64,
                      help="if gdc_threshold is not given can be calculated by specifying avg degree")
  parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
  parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
  parser.add_argument('--att_samp_pct', type=float, default=1,
                      help="float in [0,1). The percentage of edges to retain based on attention scores")
  parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                      help='incorporate the feature grad in attention based edge dropout')
  parser.add_argument("--exact", action="store_true",
                      help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")

  args = parser.parse_args()
  opt = vars(args)
  main(opt)