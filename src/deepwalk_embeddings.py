import os.path as osp
import argparse

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import time
import pickle
from data import get_dataset


def main(opt):
    dataset_name = opt['dataset']

    print(f"[i] Generating embeddings for dataset: {dataset_name}")
    dataset = get_dataset(opt, '../data', opt['not_lcc'])
    data = dataset.data

    device = torch.device(f"cuda:{opt['gpu']}" if torch.cuda.is_available() else 'cpu')

    model = Node2Vec(data.edge_index, embedding_dim=opt['embedding_dim'], walk_length=opt['walk_length'],
                     context_size=opt['context_size'], walks_per_node=opt['walks_per_node'],
                     num_negative_samples=opt['neg_pos_ratio'], p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc, z


    ### here be main code
    t = time.time()
    for epoch in range(1, opt['epochs']+1):
        loss = train()
        train_t = time.time() - t
        t = time.time()
        acc, _ = test()
        test_t = time.time() - t
        print(f'Epoch: {epoch:02d}, Train: {train_t:.2f}, Test: {test_t:.2f},  Loss: {loss:.4f}, Acc: {acc:.4f}')


    acc, z = test()
    print(f"[i] Final accuracy is {acc}")
    print(f"[i] Embedding shape is {z.data.shape}")

    fname = "DW_%s_emb_%03d_wl_%03d_cs_%02d_wn_%02d_epochs_%03d.pickle" % (
      opt['dataset'], opt['embedding_dim'], opt['walk_length'], opt['context_size'], opt['walks_per_node'], opt['epochs']
    )

    print(f"[i] Storing embeddings in {fname}")
    
    with open(osp.join("../data/pos_encodings", fname), 'wb') as f:
      # make sure the pickle is not bound to any gpu, and store test acc with data
      pickle.dump({"data": z.data.to(torch.device("cpu")), "acc": acc}, f)



if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='Cora',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv')
  parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
  parser.add_argument('--walk_length', type=int, default=20, # note this can grow much bigger (paper: 40~100)
                        help='Walk length')
  parser.add_argument('--context_size', type=int, default=16,# paper shows increased perf until 16
                        help='Context size')
  parser.add_argument('--walks_per_node', type=int, default=16, # best paper results with 18
                        help='Walks per node')
  parser.add_argument('--neg_pos_ratio', type=int, default=1, 
                        help='Number of negatives for each positive')
  parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
  parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU id (default 0)')
  parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")


  args = parser.parse_args()
  opt = vars(args)
  opt['rewiring'] = None
  main(opt)
