import torch
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import add_self_loops, to_undirected
from data import get_dataset
from torch_sparse import spspmm, coalesce
import time
opt = {'dataset': 'Cora', 'device': 'cpu', 'not_lcc': True, 'rewiring': None, 'geom_gcn_splits': False}
dataset = get_dataset(opt, '../data', opt['not_lcc'])

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


#method1 - doing full spmm
t = time.time()
index_1, value_1 = spspmm(edge_index, edge_weight, edge_index, edge_weight, num_nodes, num_nodes, num_nodes)
mask = index_1[0] == index_1[1]
diags_1 = value_1[mask]
t1 = time.time() - t


#method2 - only works if A=A.T ie for A**2 step
t = time.time()
diags_2 = scatter(edge_weight**2, edge_index[1], dim=0, dim_size=num_nodes, reduce="add")
t2 = time.time() - t

#method 3 - using Ben's method with sparse hadamard on sparse_coo tensors
# useful for A^k = B @ A where B can be A^k-1
t = time.time()
index_trans = torch.stack([col, row], dim=0)
A = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
B = torch.sparse_coo_tensor(index_trans, edge_weight, (num_nodes, num_nodes))
C = B * A
diags_3 = scatter(C.values(), C.indices()[1], dim=0, dim_size=num_nodes, reduce="add")
t3 = time.time() - t

print(f"methods equal {torch.equal(diags_1, diags_2), torch.equal(diags_1, diags_3), torch.equal(diags_2, diags_3)}")
print(f"time 1 {t1}, time 2 {t2}, time 3 {t3}, t1/t2 = {t1/t2}, t1/t3 = {t1/t3}")

