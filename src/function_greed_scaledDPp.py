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

    self.K = Parameter(torch.Tensor(out_features, opt['dim_p_omega']))
    self.Q = Parameter(torch.Tensor(out_features, opt['dim_p_omega']))
    self.Omega = self.Q @ self.K.t()  # output a [d,d] tensor
    self.W = Parameter(torch.Tensor(out_features, opt['dim_p_w']))
    self.Ws = self.W @ self.W.t()  # output a [d,d] tensor


  def sparse_hadamard_bilin(self, A, B, edge_index, values=None):
    """
    Takes a sparse matrix S and, 2 dense matrices A, B and performs a sparse hadamard product
    Keeping only the elements of A @ B.T where S is non-zero
    Only keeps the rows_i in A and the cols_j in B where i,j in S
    @param S: a sparse Matrix
    @param A: a dense matrix dim[n , d]
    @param B: a dense matrix dim[n, d]
    @return: hp_edge_index, hp_values
    """
    if values is None:
      S_values = torch.ones(edge_index.shape[1])
    rows, cols = edge_index[0], edge_index[1]
    hp_values = torch.sum(A[rows] * B[cols], dim=1)
    return hp_values

  def D_diagM(self, A, B, D):
    """
    Takes 3 dense matrices A, B, C and performs D @ diag(A @ B.T)
    Keeping only the elements of A @ B where D is non-zero
    @return: values
    """
    values = D * torch.sum(A * B, dim=1)
    return values

  def sparse_sym_normalise(self, D, edge_index, values):
    src_x, dst_x = self.get_src_dst(D)
    src_x * values * dst_x

  def get_tau(self, x):
    """
    Tau plays a role similar to the diffusivity / attention function in BLEND. Here the choice of nonlinearity is not
    plug and play as everything is defined on the level of the energy and so the derivatives are hardcoded.
    #todo why do we hardcode the derivatives? Couldn't we just use autograd?
    @param x:
    @param edge:
    @return: a |E| dim vector for Tij and Tji
    """
    src_x, dst_x = self.get_src_dst(x)

    tau_arg_values = self.sparse_hadamard_bilin(src_x @ self.Omega, dst_x, self.edge_index)
    tau_trans_arg_values = self.sparse_hadamard_bilin(dst_x @ self.Omega, src_x, self.edge_index)

    if self.opt['test_tau_remove_tanh'] and not self.opt['test_tau_symmetric']:
      tau = (tau_arg_values) / self.opt['tau_reg']
      tau_transpose = (tau_trans_arg_values) / self.opt['tau_reg']
    elif self.opt['test_tau_remove_tanh'] and self.opt['test_tau_symmetric']:
      tau = (tau_arg_values) / self.opt['tau_reg']
      tau_transpose = (tau_trans_arg_values) / self.opt['tau_reg']
    elif not self.opt['test_tau_remove_tanh'] and self.opt['test_tau_symmetric']:
      tau = torch.tanh(tau_arg_values / self.opt['tau_reg'])
      tau_transpose = torch.tanh(tau_trans_arg_values / self.opt['tau_reg'])
    elif  self.opt['test_tau_outside']:
      tau = torch.tanh(tau_arg_values) / self.opt['tau_reg']
      tau_transpose = torch.tanh(tau_trans_arg_values) / self.opt['tau_reg']
    else:
      tau = torch.tanh(tau_arg_values / self.opt['tau_reg'])
      tau_transpose = torch.tanh(tau_trans_arg_values / self.opt['tau_reg'])

    return tau, tau_transpose


  def get_R1_term1(self, T2, f, fWs):
    g = torch.sum(f * fWs, dim=1)
    return (self.deg_inv * g)[self.edge_index[0, :]] * T2

  def get_R1_term2(self, T3, f, fWs):
    temp_sparse = self.sparse_sym_normalise(self, self.deg_inv_sqrt, self.edge_index, T3)
    return self.sparse_hadamard_bilin(f, fWs, self.edge_index, temp_sparse)

  def get_R2_term1(self, T5, f, fWs):
    return self.sparse_hadamard_bilin(f, fWs, self.edge_index, T5)

  def get_R2_term2(self, T2, f, fWs):
    return T2 * self.D_diagM(self, f, fWs, self.deg_inv)


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

    R1_term1 = self.get_R1_term1(T2, f, fWs)
    R1_term2 = self.get_R1_term2(T3, f, fWs)
    R2_term1 = self.get_R2_term1(T5, f, fWs)
    R2_term2 = self.get_R2_term2(T2, f, fWs)

    return L, R1_term1, R1_term2, R2_term1, R2_term2

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

    L, R1_term1, R1_term2, R2_term1, R2_term2 = self.get_dynamics(x, gamma, tau, tau_transpose, Ws)

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    f = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)
    fomegaT = torch.matmul(f, self.omega)
    fomega = torch.matmul(f, Ws)

    f = f + \
        + torch_sparse.spmm(edges, R1_term1, x.shape[0], x.shape[0], fomegaT) \
        - torch_sparse.spmm(edges, R1_term2, x.shape[0], x.shape[0], fomegaT) \
        - torch_sparse.spmm(edges, R2_term1, x.shape[0], x.shape[0], fomega) \
        + torch_sparse.spmm(edges, R2_term2, x.shape[0], x.shape[0], fomega) \
        - 0.5 * self.mu * (x - self.x0)

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
