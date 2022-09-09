from time import time
import multiprocessing as mp
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC


def get_zinc_data(split):
  path = '../data/ZINC'
  dataset = ZINC(path, subset=True, split=split)
  # dataset.num_classes = 1    #can't override existing value which is unique float values
  dataset.num_nodes = dataset.data.x.shape[0]
  dataset.data.edge_attr = None #not using edge features
  dataset.data.x = dataset.data.x.float()
  return dataset


def test_num_workers():
    train_dataset = get_zinc_data('train')
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=128, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
