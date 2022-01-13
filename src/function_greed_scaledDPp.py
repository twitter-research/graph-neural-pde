"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.utils import degree
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
import wandb

from utils import MaxNFEException
from base_classes import ODEFunc
from function_greed import ODEFuncGreed

class ODEFuncGreed_SDB(ODEFuncGreed):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreed_SDB, self).__init__(in_features, out_features, opt, data, device, bias=False)


  def sparse_hadamard_bilin(self, A, B, S_edge_index, S_values=None):
    """
    Takes a sparse matrix S and, 2 dense matrices A, B and performs a sparse hadamard product
    Keeping only the elements of A @ B.T where S is non-zero
    Only keeps the rows_i in A and the cols_j in B where i,j in S
    @param S: a sparse Matrix
    @param A: a dense matrix dim[n , d]
    @param B: a dense matrix dim[n, d]
    @return: hp_edge_index, hp_values
    """
    if S_values is None:
      S_values = torch.ones(S_edge_index.shape[1])
    rows, cols = S_edge_index[0], S_edge_index[1]
    hp_values = torch.sum(A[rows] * B[cols], dim=1)
    hp_edge_index = S_edge_index
    return hp_edge_index, hp_values

  def D_diagM(self, A, B, C):
    """
    Takes 3 dense matrices A, B, C and performs Dinv @ diag(A @ B.T)
    Keeping only the elements of A @ B where D is non-zero
    @return: values
    """
    values = self.deg_inv * torch.sum(A * B, dim=1)
    return values

  def get_dynamics(self, f, gamma, tau, tau_transpose, Ws):
    """
    These dynamics follow directly from differentiating the energy
    The structure of T0,T1 etc. is a direct result of differentiating the hyperbolic tangent nonlinearity.
    Note: changing the nonlinearity will change the equations
    Returns: L - a generalisation of the Laplacian, but using attention
             R1, R2 are a result of tau being a function of the features and both are zero if tau is not dependent on
             features
    """
    tau = tau.flatten()
    tau_transpose = tau_transpose.flatten()
    tau2 = tau * tau
    tau3 = tau * tau2

    if self.opt['test_tau_remove_tanh']:
      T0 = gamma * tau2
      T1 = gamma * tau * tau_transpose
      T2 = gamma * tau
      T3 = gamma * tau_transpose
      # sparse transpose changes the edge index instead of the data and this must be carried through
      transposed_edge_index, T3_transpose = torch_sparse.transpose(self.edge_index, T3, self.n_nodes, self.n_nodes)
      T5 = self.symmetrically_normalise(T3_transpose, transposed_edge_index)
    else:
      T0 = gamma * tau2
      T1 = gamma * tau * tau_transpose
      T2 = gamma * (tau - tau3)
      T3 = gamma * (tau_transpose - tau_transpose * tau2)
      # sparse transpose changes the edge index instead of the data and this must be carried through
      transposed_edge_index, T3_transpose = torch_sparse.transpose(self.edge_index, T3, self.n_nodes, self.n_nodes)
      T5 = self.symmetrically_normalise(T3_transpose, transposed_edge_index)

    L = self.get_laplacian_form(T1, T0)
    fWs = torch.matmul(f, Ws)
    R1 = self.get_R1(f, T2, T3, fWs)
    R2 = self.get_R2(T2, T5, transposed_edge_index, f, fWs)
    return L, R1, R2

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    Ws = self.W @ self.W.t()  # output a [d,d] tensor
    tau, tau_transpose = self.get_tau(x)
    metric = self.get_metric(x, tau, tau_transpose)

    gamma, eta = self.get_gamma(metric)
    if self.opt['test_omit_metric']:
      gamma = -torch.ones(gamma.shape, device=x.device) #setting metric equal to adjacency

    L, R1, R2 = self.get_dynamics(x, gamma, tau, tau_transpose, Ws)
    # L, R1, R2 = self.clipper([L, R1, R2])
    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    f = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)
    f = torch.matmul(f, Ws)

    if self.opt['test_tau_symmetric']:
      f + R1.unsqueeze(dim=-1) @ self.K.t() + R2.unsqueeze(dim=-1) @ self.K.t()
    else:
      f = f + R1.unsqueeze(dim=-1) @ self.K.t() + R2.unsqueeze(dim=-1) @ self.Q.t()

    f = f - 0.5 * self.mu * (x - self.x0)

    if self.opt['test_omit_metric'] and self.opt['test_mu_0']: #energy to use when Gamma is -adjacency and not the pullback and mu == 0
      energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2)
    elif self.opt['test_omit_metric']: #energy to use when Gamma is -adjacency and not the pullback and mu != 0
      energy = torch.sum(self.get_energy_gradient(x, tau, tau_transpose) ** 2) + self.mu * torch.sum((x - self.x0) ** 2)
    else:
      energy = self.get_energy(x, eta)

    if self.opt['wandb_track_grad_flow'] and self.epoch in self.opt['wandb_epoch_list'] and self.training:
      wandb.log({f"gf_e{self.epoch}_energy_change": energy - self.energy, f"gf_e{self.epoch}_energy": energy,
                 f"gf_e{self.epoch}_f": f ** 2, f"gf_e{self.epoch}_L": torch.sum(L ** 2),
                 f"gf_e{self.epoch}_R1": torch.sum(R1 ** 2), f"gf_e{self.epoch}_R2": torch.sum(R2 ** 2), f"gf_e{self.epoch}_mu": self.mu,
                 "grad_flow_step": self.wandb_step})

      self.wandb_step += 1

    self.energy = energy

    if self.opt['greed_momentum'] and self.prev_grad:
      f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      self.prev_grad = f
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
