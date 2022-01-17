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
    #todo this isn't great as it inits the KQW from super first.
    self.K = Parameter(torch.Tensor(opt['dim_p_omega'], out_features))
    self.Q = Parameter(torch.Tensor(opt['dim_p_omega'], out_features))
    self.W = Parameter(torch.Tensor(out_features, opt['dim_p_w']))
    # https://discuss.pytorch.org/t/how-to-initialize-weight-with-arbitrary-tensor/3432
    #https://discuss.pytorch.org/t/initialize-weights-using-the-matrix-multiplication-result-from-two-nn-parameter/120557/8

    # self.K = nn.Linear(out_features, opt['dim_p_omega'])
    # self.init_weights(self.K)
    # self.Q = nn.Linear(out_features, opt['dim_p_omega'])
    # self.init_weights(self.Q)
    # self.W = nn.Linear(out_features, opt['dim_p_w'])
    # self.init_weights(self.W)

    self.reset_parameters()

  # def init_weights(self, m):
  #   if type(m) == nn.Linear:
  #     # nn.init.xavier_uniform_(m.weight, gain=1.414)
  #     # m.bias.data.fill_(0.01)
  #     nn.init.constant_(m.weight, 1e-5)


  def sparse_hadamard_bilin(self, A, B, edge_index, values=None):
    """
    Takes a sparse matrix S and, 2 dense matrices A, B and performs a sparse hadamard product
    Keeping only the elements of A @ B.T where S is non-zero
    Only keeps the rows_i in A and the cols_j in B.T where i,j in S
    @param S: a sparse Matrix
    @param A: a dense matrix dim[n , d]
    @param B: a dense matrix dim[n, d]
    @return: hp_edge_index, hp_values
    """
    if values is None:
      values = torch.ones(edge_index.shape[1]).to(self.device)
    rows, cols = edge_index[0], edge_index[1]
    product = values * torch.sum(A[rows] * B[cols], dim=1)
    return product

  def D_diagM(self, A, B, D):
    """
    Takes 2 dense matrices A, B, and a diagonal D and performs D @ diag(A @ B.T)
    Keeping only the elements of A @ B where D is non-zero ie i==j
    @return: values
    """
    values = D * torch.sum(A * B, dim=1)
    return values

  def sparse_sym_normalise(self, D, edge_index, values):
    src_x, dst_x = self.get_src_dst(D)
    product = src_x * values * dst_x
    return product

  def get_tau(self, x, Omega):
    """
    Tau plays a role similar to the diffusivity / attention function in BLEND. Here the choice of nonlinearity is not
    plug and play as everything is defined on the level of the energy and so the derivatives are hardcoded.
    #todo why do we hardcode the derivatives? Couldn't we just use autograd?
    @param x:
    @param edge:
    @return: a |E| dim vector for Tij and Tji
    """
    src_x, dst_x = self.get_src_dst(x)

    tau_arg_values = self.sparse_hadamard_bilin(src_x @ Omega, dst_x, self.edge_index).unsqueeze(dim=-1) #unsqueeze needed for shape [E x 1]
    tau_trans_arg_values = self.sparse_hadamard_bilin(dst_x @ Omega, src_x, self.edge_index).unsqueeze(dim=-1)

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
    return (self.deg_inv * g)[self.edge_index[0, :]] * T2 #todo check edge index
    #indexing needs to match indexing of T2 which comes from Tau.
    #because Tau comes from sparse haddamard product which takes the rows i of f@Omega and the cols j  of f.T
    #so I'm happy on the understanding rows == i == src

  def get_R1_term2(self, T3, f, fWs):
    temp_sparse = self.sparse_sym_normalise(self.deg_inv_sqrt, self.edge_index, T3)
    return self.sparse_hadamard_bilin(f, fWs, self.edge_index, temp_sparse)

  def get_R2_term1(self, T5, f, fWs):
    return self.sparse_hadamard_bilin(f, fWs, self.edge_index, T5)

  def get_R2_term2(self, T2, f, fWs):
    product = T2 * self.D_diagM(f, fWs, self.deg_inv)[self.edge_index[1]] #todo this is a massive guess whether it's [0] or [1], went [1] because RHS multiplication
    return product #ie RHS multiplication with a diagonal matrix broadcasts over columns not rows
    #T2 is again rows == i == src. there are E non zero elements.
    #the second term is a diagonal matrix, so we scale the columns, ie index cols=j==dst

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
    # Omega = self.Q @ self.K.t()  # output a [d,d] tensor
    Omega = self.K.t() @ self.Q  # output a [d,d] tensor

    tau, tau_transpose = self.get_tau(x, Omega)
    metric = self.get_metric(x, tau, tau_transpose)

    if self.opt['test_omit_metric']:
      eta = torch.ones(metric.shape, device=x.device)
      gamma = -torch.ones(metric.shape, device=x.device) #setting metric equal to adjacency
    else:
      gamma, eta = self.get_gamma(metric, self.opt['gamma_epsilon'])

    L, R1_term1, R1_term2, R2_term1, R2_term2 = self.get_dynamics(x, gamma, tau, tau_transpose, Ws)

    edges = torch.cat([self.edge_index, self.self_loops], dim=1)
    Lf = torch_sparse.spmm(edges, L, x.shape[0], x.shape[0], x)

    fomegaT = torch.matmul(x, Omega.T)
    fomega = torch.matmul(x, Omega)

    f = torch.matmul(Lf, Ws)
    f = f + torch_sparse.spmm(self.edge_index, R1_term1, x.shape[0], x.shape[0], fomegaT)
    f = f - torch_sparse.spmm(self.edge_index, R1_term2, x.shape[0], x.shape[0], fomegaT)
    f = f - torch_sparse.spmm(self.edge_index, R2_term1, x.shape[0], x.shape[0], fomega)
    f = f + torch_sparse.spmm(self.edge_index, R2_term2, x.shape[0], x.shape[0], fomega)
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
                 f"gf_e{self.epoch}_R1_term1": torch.sum(R1_term1 ** 2), f"gf_e{self.epoch}_R1_term2": torch.sum(R1_term2 ** 2),
                 f"gf_e{self.epoch}_R2_term1": torch.sum(R2_term1 ** 2), f"gf_e{self.epoch}_R2_term2": torch.sum(R2_term2 ** 2),
                 f"gf_e{self.epoch}_mu": self.mu, "grad_flow_step": self.wandb_step})

      self.wandb_step += 1

    self.energy = energy

    if self.opt['greed_momentum'] and self.prev_grad:
      f = self.opt['momentum_alpha'] * f + (1 - self.opt['momentum_alpha']) * self.prev_grad
      self.prev_grad = f
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
