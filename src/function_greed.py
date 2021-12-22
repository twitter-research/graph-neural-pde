"""
Implementation of the functions proposed in Graph embedding energies and diffusion
"""

import torch
from torch import nn
import torch_sparse
from torch_scatter import scatter_add, scatter_mul
from torch_geometric.utils.loop import add_remaining_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter

from utils import MaxNFEException
from base_classes import ODEFunc


class ODEFuncGreed(ODEFunc):

  def __init__(self, in_features, out_features, opt, data, device, bias=False):
    super(ODEFuncGreed, self).__init__(opt, data, device)
    assert opt['self_loop_weight'] == 0, 'greed does not work with self-loops as eta becomes zero everywhere'
    if opt['self_loop_weight'] > 0:
      self.edge_index, self.edge_weight = add_remaining_self_loops(data.edge_index, data.edge_attr,
                                                                   fill_value=opt['self_loop_weight'])
    else:
      self.edge_index, self.edge_weight = data.edge_index, data.edge_attr

    self.n_nodes = data.x.shape[0]

    self.K = Parameter(torch.Tensor(out_features, 1))
    self.Q = Parameter(torch.Tensor(out_features, 1))

    self.deg_inv_sqrt = self.get_deg_inv_sqrt(data)
    self.deg_inv = self.deg_inv_sqrt * self.deg_inv_sqrt

    self.W = Parameter(torch.Tensor(in_features, out_features))

    if bias:
      self.bias = Parameter(torch.Tensor(out_features))
    else:
      self.register_parameter('bias', None)

    self.mu = nn.Parameter(torch.ones(1))

    self.reset_parameters()

  def get_deg_inv_sqrt(self, data):
    edge_index = data.edge_index
    edge_weight = torch.ones((edge_index.size(1),), dtype=data.x.dtype,
                             device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=self.n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt = deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    return deg_inv_sqrt

  def reset_parameters(self):
    glorot(self.W)
    glorot(self.K)
    glorot(self.Q)
    zeros(self.bias)

  def symmetrically_normalise(self, M):
    """
    D^{-1/2}MD^{-1/2}
    where D is the degree matrix
    """
    M = torch_sparse.mul(M, self.deg_inv_sqrt.view(-1, 1))
    M = torch_sparse.mul(M, self.deg_inv_sqrt.view(1, -1))
    return M

  def get_tau(self, x):
    """
    Tau plays a role similar to the diffusivity / attention function in BLEND. Here the choice of nonlinearity is not
    plug and play as everything is defined on the level of the energy and so the derivatives are hardcoded.
    #todo why do we hardcode the derivatives? Couldn't we just use autograd?
    @param x:
    @param edge:
    @return:
    """
    src_x, dst_x = self.get_src_dst(x)
    tau = torch.tanh(src_x @ self.K + dst_x @ self.Q)
    tau_transpose = torch.tanh(dst_x @ self.K + src_x @ self.Q)
    return tau, tau_transpose

  def get_src_dst(self, x):
    """
    Get the values of a dense n-by-d matrix
    @param x:
    @return:
    """
    src = x[self.edge_index[0, :]]
    dst = x[self.edge_index[1, :]]
    return src, dst

  def get_gamma(self, x, tau, tau_transpose, epsilon=1e-6):
    """

    @param x:
    @param edge:
    @param tau:
    @param tau_transpose:
    @param epsilon:
    @return:
    """
    # todo break up this function
    src_x, dst_x = self.get_src_dst(x)
    src_degree, dst_degree = self.get_src_dst(self.deg_inv_sqrt)
    src_term = torch.div(tau * src_x, src_degree.unsqueeze(dim=1))
    src_term.masked_fill_(src_term == float('inf'), 0.)
    dst_term = torch.div(tau_transpose * dst_x, dst_degree.unsqueeze(dim=1))
    dst_term.masked_fill_(dst_term == float('inf'), 0.)
    temp = (src_term - dst_term) @ self.W
    edge_weight = torch.sum(temp * temp, dim=1)
    # The degree is the thing that provides a notion of dimension on a graph
    # eta is the determinant of the ego graph. This should slow down diffusion where the grad is large
    eta = torch.pow(scatter_mul(edge_weight, self.edge_index[1, :]), self.deg_inv)
    src_eta, dst_eta = self.get_src_dst(eta)
    big_gamma = -torch.div(src_eta + dst_eta, 2 * edge_weight)
    with torch.no_grad():
      mask = edge_weight < epsilon
      big_gamma[mask] = 0
    return big_gamma

  def get_dynamics(self, f, gamma, tau, Ws):
    """
    These dynamics follow directly from differentiating the energy
    The structure of T0,T1 etc. is a direct result of differentiating the hyperbolic tangent nonlinearity.
    Note: changing the nonlinearity will change the equations
    Returns: L - a generalisation of the Laplacian, but using attention
             R1, R2 are a result of tau being a function of the features and both are zero if tau is not dependent on
             features
    """
    # todo tau is probably sparse so this won't work
    tau2 = tau * tau
    tau3 = tau * tau2
    T0 = gamma * tau2
    T1 = gamma * tau * tau.T
    T2 = gamma * (tau - tau3)
    T3 = gamma * (tau.T - tau.T * tau2)
    # the next line has the spirit of a Laplacian e.g. D-A
    T4 = self.symmetrically_normalise(torch.diag(torch.sum(T2, dim=1)) - T3)
    T5 = self.symmetrically_normalise(T3.T)
    L = self.symmetrically_normalise(torch.diag(torch.sum(T0, dim=1)) - T1)
    fWs = torch.matmul(f, Ws)
    R1 = torch.sum(f * (T4 @ fWs), dim=1)
    # todo check that this is the same as what Francesco wrote
    R2 = T2.T @ self.deg_inv * torch.sum(f * fWs, dim=1) - torch.sum(f * T5 @ fWs, dim=1)
    return L, R1, R2

  def forward(self, t, x):  # t is needed when called by the integrator
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException

    self.nfe += 1
    Ws = self.W.T @ self.W
    tau, tau_transpose = self.get_tau(x)
    gamma = self.get_gamma(x, tau, tau_transpose, epsilon=1e-6)
    L, R1, R2 = self.get_dynamics(x, gamma, tau, Ws)
    f = L @ x
    f = torch.matmul(f, self.Ws)
    f = f + R1 @ self.K.T + R2 @ self.Q.T
    # todo check x0 is the encoded value of features + pos_enc
    f = f - self.mu(f - self.x0)
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
