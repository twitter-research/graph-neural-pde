import os
import glob
import pickle
import argparse

def main(opt):

  all_datasets = ["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CoauthorCS", "ogbn-arxiv"]
  all_embedding_dims = [64, 128, 256]
  data_path = "../data/pos_encodings/"

  if opt['dataset'] == "ALL":
    datasets = all_datasets
  else:
    datasets = [opt['dataset']]

  if opt['embedding_dim'] == 0:
    embedding_dims = all_embedding_dims
  else:
    embedding_dims = [opt['embedding_dim']]

  for dataset in datasets:
    for embedding_dim in embedding_dims:
      fname = f"DW_{dataset}_emb_{embedding_dim:03d}*"
      
      pickles = glob.glob(os.path.join(data_path, fname))

      max_acc = 0
      best_emb = None

      for p in pickles:
        with open(p, "rb") as f:
          data = pickle.load(f)
          acc = data['acc']
          print(f"Model {p} has accuracy {acc}")
          if acc > max_acc:
            max_acc = acc
            best_emb = p

      print(f"=> The best model is {best_emb} with accuracy {max_acc}")

      print("Removing previous symlink...")
      os.system(f"rm {data_path}{dataset}_DW{embedding_dim}.pkl")

      command = f"ln -s {p[len(data_path):]} {data_path}{dataset}_DW{embedding_dim}.pkl"
      print(f"Running: {command}")
      os.system(command)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='ALL',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--embedding_dim', type=int, default=0,
                        help='Embedding dimension')


  args = parser.parse_args()
  opt = vars(args)
  main(opt)
                        
