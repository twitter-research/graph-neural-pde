import os
import glob
import pickle
import argparse


def main(opt):
  all_datasets = ["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CoauthorCS"]
  data_path = "../data/pos_encodings/"

  if opt['dataset'] == "ALL":
    datasets = all_datasets
  else:
    datasets = [opt['dataset']]

  for dataset in datasets:
    fname = f"{dataset}_DW*"
    mask_fname = os.path.join(data_path, f"{dataset}_lcc.pkl")

    pickles = glob.glob(os.path.join(data_path, fname))

    for p in pickles:
      with open(p, "rb") as f:
        content = pickle.load(f)
        data = content['data']
        print(f"Data {p} has shape {data.shape}")

      with open(mask_fname, "rb") as f:
        mask = pickle.load(f)
        print(f"Mask {mask_fname} has shape {mask.shape}")

      if data.shape[0] != mask.shape[0]:
        new_data = data[mask, :]
        print(f"New data has shape {new_data.shape}")

        print("Removing symlink...")
        os.system(f"rm {p}")

        print("Storing new file...")
        with open(p, "wb") as f:
          pickle.dump({"acc": content['acc'], "data": new_data}, f)

    # command = f"ln -s {p[len(data_path):]} {data_path}{dataset}_DW{opt['embedding_dim']}.pkl"
    # print(f"Running: {command}")
    # os.system(command)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='ALL',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')

  args = parser.parse_args()
  opt = vars(args)
  main(opt)
