import os
import torch
from torch_geometric.transforms import two_hop
from torch_geometric.datasets import Planetoid
from torch_sparse import spspmm, coalesce


data_dir = '../../data'
ds = 'Cora'
path = os.path.join(data_dir, ds)
dataset = Planetoid(path, ds)
data = dataset[0]

edge_index, edge_attr = data.edge_index, data.edge_attr
N = data.num_nodes
value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

print(f"edge_index {edge_index.shape}")
index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
print(f"edge_index {index.shape}")
index = torch.unique(index, dim=1)
print(f"edge_index {index.shape}")

# th = two_hop.TwoHop()
# print(f"edge_index {index.shape}")
# data = th(data)
# print(f"edge_index {data.edge_index.shape}")
