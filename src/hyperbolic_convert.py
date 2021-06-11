import os
import os
import glob
import pickle
import argparse
import numpy as np
import torch

def main(opt):

  all_datasets = ["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CoauthorCS"]#, "ogbn-arxiv"]
  data_path = "../data/pos_encodings/"

  if opt['dataset'] == "ALL":
    datasets = all_datasets
  else:
    datasets = [opt['dataset']]


  for dataset in datasets:
    dirnames = f"{dataset}_HYPS*"
    mask_fname = os.path.join(data_path, f"{dataset}_lcc.pkl")
    with open(mask_fname, "rb") as f:
      mask = pickle.load(f)
      print(f"Mask {mask_fname} has shape {mask.shape}")
    
    directories = glob.glob(os.path.join(data_path, dirnames))


    for d in directories:
      print(f"[i] opening {d}...")
      data = np.load(os.path.join(d, "embeddings.npy"))

      print(f"    Embeddings have size {data.shape}")
      new_data = data[mask,:]
      print(f"    New data has shape {new_data.shape}")

      fname = f"{d}.pkl"
      print(f"    Dumping to {fname}")
      with open(fname, "wb") as f:
        pickle.dump(torch.Tensor(new_data), f)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='ALL',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')

  args = parser.parse_args()
  opt = vars(args)
  main(opt)
                        
