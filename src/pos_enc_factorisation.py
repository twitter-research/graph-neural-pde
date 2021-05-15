"""
matrix factorisation of the positional encoding required for arxiv-ogbn
"""

import numpy as np
import argparse
import os
import pickle
from sklearn.decomposition import NMF
from graph_rewiring import apply_gdc
from data import get_dataset
import time

POS_ENC_PATH = os.path.join("../data", "pos_encodings")


def find_or_make_encodings(opt):
  # generate new positional encodings
  # do encodings already exist on disk?
  fname = os.path.join(POS_ENC_PATH, f"{opt['dataset']}_{opt['pos_enc_type']}.pkl")
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
    dataset = get_dataset(opt, '../data', False)
    data = dataset.data
    if opt['pos_enc_type'] == "GDC":
      pos_encoding = apply_gdc(data, opt, type="pos_encoding")
    else:
      print(f"[x] The positional encoding type you specified ({opt['pos_enc_type']}) does not exist")
      quit()
    # - ... and store them on disk
    if not os.path.exists(POS_ENC_PATH):
      os.makedirs(POS_ENC_PATH)
    with open(fname, "wb") as f:
      pickle.dump(pos_encoding, f)

  return pos_encoding


def main(opt):
  start_time = time.time()
  dim = opt['embedding_dim']
  type = opt['opt_pos_enc_type']
  model = NMF(n_components=dim, init='random', random_state=0, max_iter=opt['max_iter'])
  fname = os.path.join(POS_ENC_PATH, f"{opt['dataset']}_{opt['pos_enc_type']}.pkl")
  print(f"[i] Looking for positional encodings in {fname}...")

  pos_encodings = find_or_make_encodings(opt)

  # - if so, just load them

  W = model.fit_transform(pos_encodings)
  # H = model.components_
  end_time = time.time()
  print(f"compression to {dim} dim complete in {(end_time-start_time)} seconds")

  out_path = f"{opt['out_dir']}/compressed_pos_encodings_{dim}_{type}.pkl"
  with opt(out_path, 'wb') as f:
    pickle.dump(W, f)

  if not os.path.exists(opt['out_path']):
    os.makedirs(opt['out_path'])
  with open(fname, "wb") as f:
    pickle.dump(W, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--use_cora_defaults",
    action="store_true",
    help="Whether to run with best params for cora. Overrides the choice of dataset",
  )
  parser.add_argument(
    "--data_path", type=str, default=".", help="path to the positional encoding"
  )
  parser.add_argument(
    "--out_dir", type=str, default=".", help="path to save compressed encoding"
  )
  parser.add_argument(
    "--pos_enc_type", type=str, default="GDC", help="type of encoding to make only GDC currently implemented"
  )
  parser.add_argument(
    "--dataset", type=str, default="ogbn-arxiv", help="type of encoding to make only GDC currently implemented"
  )
  parser.add_argument(
    "--embedding_dim", type=int, default=1000, help="dimension of compressed encoding"
  )
  parser.add_argument(
    "--max_iter", type=int, default=1000, help="number of training iterations"
  )
  args = parser.parse_args()
  opt = vars(args)
  main(opt)
