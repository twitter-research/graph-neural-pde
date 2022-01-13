

import torch
from torch import tensor, ones
from torch_geometric.utils import to_dense_adj

n = 4
d = 3

A = torch.ones([n, d]) #do arange
B = torch.ones([n, d])
S_edge_index = tensor([[0, 2, 2, 1, 1, 3, 2, 3], [2, 0, 1, 2, 3, 1, 3, 2]])
S_edge_values = torch.ones(S_edge_index.shape[1])

def sparse_hadamard(A, B, S_edge_index, S_values=None):
    """
    Takes a sparse matrix S and, 2 dense matrices A, B and performs a sparse hadamard product
    Keeping only the elements of A @ B where S is non-zero
    Only keeps the rows_i in A and the cols_j in B where i,j in S
    @param S: a sparse Matrix
    @param A: a dense matrix
    @param B: a dense matrix
    @return: hp_values, hp_edge_index
    """
    if S_values is None:
      S_values = torch.ones(S_edge_index.shape[1])
    rows, cols = S_edge_index[0], S_edge_index[1]
    hp_values = torch.sum(A[rows] * B[cols], dim=1)
    hp_edge_index = S_edge_index

    return hp_edge_index, hp_values

C = sparse_hadamard(A, B, S_edge_index, S_edge_values)
print(C)
print(to_dense_adj(edge_index=C[0], edge_attr=C[1], max_num_nodes=n))

Sd = to_dense_adj(edge_index=S_edge_index, edge_attr=S_edge_values, max_num_nodes=n)
print(f"Sd {Sd}")
print(f"A @ B {A @ B.T}")
print(f"Sd * A @ B.T {Sd * (A @ B.T)}")