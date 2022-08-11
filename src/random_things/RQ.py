import torch
from torch import tensor
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import add_self_loops, to_undirected
from data import get_dataset
from torch_sparse import spspmm, coalesce

from utils import dirichlet_energy, rayleigh_quotient

class Data:
  def __init__(self, edge_index, x, y=None, train_mask=None):
    x = x
    edge_index = edge_index
    edge_attr = None
    y = y
    train_mask = train_mask
    num_features = x.shape[1]
    num_nodes = x.shape[0]

class DummyDataset():
  def __init__(self, data, num_classes):
    data = data
    num_classes = num_classes
    num_features = data.num_features
    num_nodes = data.num_nodes


opt = {'dataset': 'Cora', 'device': 'cpu', 'not_lcc': True, 'rewiring': None, 'geom_gcn_splits': False}
dataset = get_dataset(opt, '../../data', opt['not_lcc'])
edge_index = dataset.data.edge_index
edge_index = add_self_loops(edge_index)[0]
edge_index = to_undirected(edge_index)
edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
num_nodes = dataset.data.x.shape[0]
row, col = edge_index[0], edge_index[1]
deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
deg_inv_sqrt = deg.pow_(-0.5)
deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
X= dataset.data.x

# edge_index = tensor([[0, 2, 2, 1, 1, 3, 2, 3], [2, 0, 1, 2, 3, 1, 3, 2]])
# X = tensor([[1., 2.], [3., 2.], [4., 5.], [6, 7]], dtype=torch.float)
# y = tensor([1, 1, 0, 0])
n = X.shape[0]
# data = Data(edge_index, x, y)
# dataset = DummyDataset(data, 2)

de = dirichlet_energy(edge_index, n, X, edge_weight=None, norm_type=None)
rq = rayleigh_quotient(edge_index, n, X, edge_weight=None)

print(f"DE {de}")
print(f"RQ {rq}")
