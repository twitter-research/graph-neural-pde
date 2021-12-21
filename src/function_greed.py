"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import torch_sparse
import torch_scatter
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter

from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncGreed(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreed, self).__init__(opt, data, device)

    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.n_nodes = data.x.shape[0]

    self.K = Parameter(torch.Tensor(out_features, 1))
    self.Q = Parameter(torch.Tensor(out_features, 1))

    self.deg_inv_sqrt = self.get_inv_root_degree(data)
    self.deg_inv = self.deg_inv_sqrt * self.deg_inv_sqrt

    self.weight = Parameter(torch.Tensor(in_features, out_features))

    if bias:
      self.bias = Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    self.mu = nn.Parameter(torch.ones(1))

    self.reset_parameters()

  def get_deg_inv_sqrt(self, data):
    deg = torch_sparse.sum(data.edge_index, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    return deg_inv_sqrt

  def reset_parameters(self):
    glorot(self.weight)
    zeros(self.bias)

  def symmetrically_normalise(self, M):
    """
    D^{-1/2}MD^{-1/2}
    where D is the degree matrix
    """
    M = torch_sparse.mul(M, self.deg_inv_sqrt.view(-1, 1))
    M = torch_sparse.mul(M, self.deg_inv_sqrt.view(1, -1))
    return M

  def get_tau(self, x, edge):
    src = x[edge[0, :]]
    dst = x[edge[1, :]]
    tau = torch.tanh(src.t @ self.K + dst.t @ self.Q)
    # todo check this is the correct transpose
    tau_transpose = torch.tanh(dst.t @ self.K + src.t @ self.Q)
    return tau, tau_transpose

  def get_gamma(self, x, edge, tau, tau_transpose, epsilon=10e-6):
    """
    get gamma_ij = |Z_ij|^2
    get eta(i) = \prod_{j \in N(i)} gamma_{ij}}^{1/d_i}
    """
    # todo break up this function
    src = x[edge[0, :]]
    dst = x[edge[1, :]]
    src_degree = self.deg_inv_sqrt[edge[0, :]]
    dst_degree = self.deg_inv_sqrt[edge[1, :]]

    src_term = torch.div(tau * src, src_degree)
    src_term.masked_fill_(src_term == float('inf'), 0.)
    dst_term = torch.div(tau_transpose * dst, dst_degree)
    dst_term.masked_fill_(dst_term == float('inf'), 0.)
    unormalised_gamma = self.W @ (src_term - dst_term)
    sml_gamma = torch.sum(unormalised_gamma * unormalised_gamma, dim=1)
    # todo check this
    eta = torch.pow(torch_scatter.scatter_mul(sml_gamma, edge[1, :]), self.deg_inv)
    src_eta = eta[edge[0, :]]
    dst_eta = eta[edge[1, :]]
    big_gamma = -torch.div(src_eta + dst_eta, 2 * sml_gamma)
    with torch.no_grad:
      mask = sml_gamma < epsilon
    big_gamma[mask] = 0
    return big_gamma

  def get_dynamics(self, f, gamma, tau, Ws):
    tau_square = tau * tau
    tau2 = tau_square @ f
    tau3 = tau * tau_square @ f
    T0 = gamma * tau2
    # todo tau is probably sparse so this won't work
    T1 = gamma * tau * tau.T
    T2 = gamma * (tau - tau3)
    T3 = gamma * (tau.T - tau.T * tau2)
    T4 = self.symmetrically_normalise(torch.diag(torch.sum(T2, dim=1)) - T3)
    T5 = self.symmetrically_normalise(T3.T)
    L = self.symmetrically_normalise(torch.diag(torch.sum(T0, dim=1)) - T1)
    fWs = torch.matmul(f, Ws)
    R1 = torch.sum(f * (T4 @ fWs), dim=1)
    # todo check that this is the same as what Francesco wrote
    R2 = T2.T @ torch.sum(self.symmetrically_normalise(f @ fWs), dim=1) - torch.sum(f * T5 @ fWs)
    return L, R1, R2

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    gamma, tau = self.multihead_att_layer(x, self.edge_index)
    Ws = self.weight.T @ self.weight
    L, R1, R2 = self.get_dynamics(x, gamma, tau, Ws)
    f = L @ x
    f = torch.matmul(f, self.Ws)
    f = f + R1 @ self.K.T + R2 @ self.Q.T
    f = f - self.mu(f - self.x0)
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
