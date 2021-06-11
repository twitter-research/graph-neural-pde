from data import get_dataset
from graph_rewiring_eval import rewiring_test
import argparse

'''
Shows how edge stats vary for a given dataset when a particular rewiring type is applied.

E.g.
python visualise_rewired_edge_stats.py --dataset Citeseer --gdc_k 4 --pos_enc_type HYPS02
python visualise_rewired_edge_stats.py --dataset Citeseer --gdc_sparsification threshold --pos_dist_quantile 0.01  --pos_enc_type HYPS02

'''


def main(opt):

  opt['beltrami'] = True

  opt['rewiring'] = None
  dataset_0 = get_dataset(opt, "../data", True)
  print(dataset_0.data)
 
  opt['rewiring'] = 'pos_enc_knn'
  dataset_1 = get_dataset(opt, "../data", True)
  print(dataset_1.data)
  
  n = dataset_0.data.num_nodes
  edge_index_0 = dataset_0.data.edge_index.detach().clone()
  edge_index_1 = dataset_1.data.edge_index.detach().clone()
  
  edges_stats = rewiring_test("Original", edge_index_0, "KNN", edge_index_1, n)
  print(edges_stats)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
  )
  parser.add_argument("--gdc_k", type=int, default=4, help="k in GDC's KNN")
  parser.add_argument("--pos_enc_type", type=str, default="HYP16", help="Positional encoder type")
  parser.add_argument("--gdc_sparsification", type=str, default="topk", help="Sparsification method (topk, threshold)")
  parser.add_argument("--pos_dist_quantile", type=float, default=0.01, help="Quantile")

 
  args = parser.parse_args()
  opt = vars(args)
  main(opt)
