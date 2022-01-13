

import torch

n = 5
d = 3
A = torch.ones([]) #do arange
B = torch.ones([])
S_edge_index = torch.tensor([])
S_values = torch.tensor([])


def sparse_hadamard(self, S_edge_index, S_values, A, B):
    """
    Takes a sparse matrix S and, 2 dense matrices A, B and performs a sparse hadamard product
    Keeping only the elements of B @ C where S is non-zero
    Only keeps the rows_i in A and the cols_j in B where i,j in S
    @param A: a sparse Matrix
    @param B: a dense matrix
    @return: hp_values, hp_edge_index
    """

    return hp_values, hp_edge_index